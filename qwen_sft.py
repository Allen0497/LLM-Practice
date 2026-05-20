"""
qwen_sft.py — 监督微调（Supervised Fine-Tuning, SFT）脚本

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
背景：大模型训练的第二阶段

  预训练 → 【SFT】→ 偏好对齐（DPO/GRPO）→ 部署

预训练后的模型只会"续写文本"，不会"对话"。
SFT 用"问题-回答"格式的数据，教模型学会按指令回答问题。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心技术：
  1. ChatML 对话格式  —— 统一的多轮对话模板
  2. DataCollatorForCompletionOnlyLM  —— 只对 assistant 回答部分计算 loss
  3. LoRA（Low-Rank Adaptation）  —— 高效参数微调，只训练少量参数
  4. SFTTrainer（来自 trl 库）  —— 专为 SFT 设计的训练器
"""

import os
import torch
import wandb
from peft import LoraConfig
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.utils import find_files, formatting_prompts_func, print_trainable_parameters

# ── 环境配置 ───────────────────────────────────────────────────────────────────

# 防止 CUDA 显存碎片化导致 OOM，将单次分配上限设为 64MB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# ── 超参数配置 ─────────────────────────────────────────────────────────────────

WANDB_LOG = True
TMP_PATH = "/archive/share/cql/aaa/tmp"   # HuggingFace datasets 缓存目录

# 调试用：只取数据集的一小部分，正式训练时设为 -1 或 0 表示使用全量数据
TRAIN_SUBSET = 1000   # 训练集取前 1000 条
EVAL_SUBSET = 100     # 验证集取前 100 条

# ── 路径配置 ───────────────────────────────────────────────────────────────────

# SFT 结果保存目录（注意：这里是 sft-1，说明是第二轮 SFT）
output_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/sft-1"

# 从上一轮 SFT 的 checkpoint 继续训练（也可以直接从预训练 checkpoint 开始）
model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/sft/checkpoint-6000"

# ── 加载模型 & Tokenizer ───────────────────────────────────────────────────────

# torch_dtype=bfloat16：BF16 混合精度，节省显存
# attn_implementation="flash_attention_2"：FlashAttention2 加速注意力计算
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 打印可训练参数量，用于确认 LoRA 是否生效（全量微调时应为 100%）
print_trainable_parameters(model)

# ── 数据集加载与预处理 ─────────────────────────────────────────────────────────

# SFT 数据格式示例（conversations 字段）：
# [
#   {"from": "human", "value": "请帮我写一首关于春天的诗"},
#   {"from": "gpt",   "value": "春风又绿江南岸，明月何时照我还..."}
# ]
directories = ["Gen", "7M"]
data_files = find_files(directories, "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/sft")

# 只加载 conversations 列，节省内存
dataset = load_dataset(
    "parquet",
    data_files=data_files,
    split="train",
    columns=["conversations"],
    cache_dir=TMP_PATH
)

# 打乱数据，避免模型学到数据排列规律
dataset = dataset.shuffle(seed=42)

# 按 8:2 比例划分训练集和验证集
# .values() 返回 (train_dataset, test_dataset) 的字典值
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.2).values()

# 调试模式：只取子集，加快实验速度
if TRAIN_SUBSET > 0:
    train_dataset = train_dataset.select(range(TRAIN_SUBSET))
if EVAL_SUBSET > 0:
    valid_dataset = valid_dataset.select(range(EVAL_SUBSET))

# ── Data Collator：只对 assistant 回答计算 loss ────────────────────────────────
#
# SFT 的关键设计：模型不应该为"用户问题"部分的 token 计算 loss，
# 因为我们不需要模型学会"提问"，只需要学会"回答"。
#
# ChatML 格式的对话模板：
#   <|im_start|>user
#   你好，请介绍一下自己<|im_end|>
#   <|im_start|>assistant        ← response_template（分隔符）
#   我是一个AI助手...<|im_end|>
#
# DataCollatorForCompletionOnlyLM 会找到 response_template 的位置，
# 将其之前的所有 token 的 label 设为 -100（PyTorch 约定：-100 表示忽略该位置的 loss）

response_template = "<|im_start|>assistant\n"
# encode 时 add_special_tokens=False，避免在模板前后插入额外的特殊 token
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(
    response_template_ids,
    tokenizer=tokenizer,
    mlm=False  # 因果语言模型，不做掩码语言模型
)

# ── LoRA 配置（当前已注释，使用全量微调）─────────────────────────────────────
#
# LoRA 原理：不直接修改原始权重矩阵 W，而是在旁边添加两个小矩阵 A 和 B：
#   W' = W + α/r × (B × A)
#   其中 r（rank）远小于 W 的维度，大幅减少可训练参数量
#
# 参数说明：
#   r=8：低秩矩阵的秩，越大表达能力越强但参数越多
#   lora_alpha=16：缩放系数，实际学习率 = lr × alpha/r = lr × 2
#   target_modules：只对注意力机制的 q（query）和 v（value）投影矩阵加 LoRA
#                   （k_proj 和 o_proj 通常也可以加，看显存预算）
#   lora_dropout=0.01：LoRA 层的 dropout，防止过拟合
#   bias="none"：不训练 bias 参数
#   task_type="CAUSAL_LM"：因果语言模型任务
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── 训练参数配置 ───────────────────────────────────────────────────────────────

# SFTConfig 继承自 TrainingArguments，额外支持 SFT 专用参数
training_args = SFTConfig(
    output_dir=output_path,
    overwrite_output_dir=True,

    eval_steps=2000,          # 每 2000 步在验证集上评估一次

    # SFT 学习率通常比预训练大（1e-5），比继续预训练大（1e-8）
    # 因为 SFT 是在已有知识上学新技能，不需要极小学习率
    learning_rate=1e-5,

    warmup_ratio=0.1,              # 前 10% steps 线性预热
    lr_scheduler_type="cosine",    # 余弦退火调度

    num_train_epochs=3,
    per_device_train_batch_size=1,   # SFT 序列较长，batch_size 通常设为 1
    gradient_accumulation_steps=16,  # 等效 batch_size = 1 × 16 × GPU数

    save_steps=1000,       # 每 1000 步保存一次 checkpoint
    save_total_limit=3,    # 最多保留 3 个 checkpoint

    bf16=True,             # BF16 混合精度
    logging_steps=10,      # 每 10 步记录一次 loss
    report_to="wandb",     # 上报到 Weights & Biases
)

# ── WandB 实验追踪 ─────────────────────────────────────────────────────────────

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-sft",
        name="qwen-0.5B-sft"
    )

# ── 初始化 SFTTrainer ──────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,

    # peft_config=lora_config,  # 取消注释即启用 LoRA 微调
    # 当前注释掉 = 全量微调（Full Fine-Tuning），所有参数都参与训练

    args=training_args,

    # formatting_func：将原始 conversations 数据转换为 ChatML 格式字符串
    # 输入：{"conversations": [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
    # 输出：["<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"]
    formatting_func=formatting_prompts_func,

    data_collator=collator,   # 只对 assistant 部分计算 loss

    max_seq_length=100,       # 截断超过 100 token 的序列（调试用，正式训练应设更大值如 2048）
    packing=False,            # 不将多个短样本拼接成一个长序列（packing=True 可提高 GPU 利用率）
    dataset_num_proc=16,      # 数据预处理并行进程数
    dataset_batch_size=5000,  # 数据预处理的批大小
)

# ── 开始训练 & 保存 ────────────────────────────────────────────────────────────

print("Training...")
trainer.train()

# 保存最终模型权重
trainer.save_model()

# 单独保存 tokenizer，推理时需要与模型配套使用
tokenizer.save_pretrained(output_path)
