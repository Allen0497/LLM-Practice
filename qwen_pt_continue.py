"""
qwen_pt_continue.py — 继续预训练（Continue Pre-Training）脚本

用途：在已有预训练 checkpoint 的基础上，使用新数据集继续训练语言模型。
与 qwen_pt.py 的核心区别：
  1. MODEL_PATH 指向已有 checkpoint，而非随机初始化的模型
  2. learning_rate 极小（1e-8），避免灾难性遗忘（catastrophic forgetting）
  3. 数据集可以是与初次预训练不同的领域数据
"""

import os
import torch
from datasets import load_dataset, Dataset
import wandb
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AdamW,
)
from utils.utils import find_files, tokenize_dataset

# ── 超参数 & 路径配置 ──────────────────────────────────────────────────────────

TRUNK_SIZE = 512          # 每个训练样本的 token 长度（截断/拼接单位）
TMP_PATH = "/archive/share/cql/aaa/tmp"   # HuggingFace datasets 缓存目录
DATA_PATH = "data/pt"                     # 原始 parquet 数据根目录
OUTPUT_PATH = "results/pt"               # 模型 checkpoint 输出目录

# 关键：指向已有 checkpoint，而非从头初始化
# 继续预训练的起点是上一阶段训练到 step-10000 的权重
MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/pt/checkpoint-10000"

WANDB_LOG = True

# 防止 CUDA 显存碎片化导致 OOM，将单次分配上限设为 64MB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# ── 加载模型 & Tokenizer ───────────────────────────────────────────────────────

output_path = OUTPUT_PATH

# 从 checkpoint 恢复模型权重
# torch_dtype=bfloat16：使用 BF16 混合精度，节省显存且数值稳定性优于 FP16
# attn_implementation="flash_attention_2"：使用 FlashAttention2 加速注意力计算
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# tokenizer 同样从 checkpoint 目录加载，保证词表与模型一致
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ── 数据集准备 ─────────────────────────────────────────────────────────────────

tokenized_datapath = os.path.join(DATA_PATH, "tokenized_dataset")

if not os.path.isdir(tokenized_datapath):
    # 继续预训练使用的新领域数据（与初次预训练数据可以不同）
    directories = [
        "film_entertainment",   # 影视娱乐
        "literature_emotion",   # 文学情感
        "news_media",           # 新闻媒体
    ]

    # 收集所有 parquet 文件路径
    data_files = find_files(directories)

    # 加载原始文本数据集，只保留 "text" 列以节省内存
    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        columns=["text"],
        cache_dir=TMP_PATH
    )

    # 打乱数据顺序，避免模型学到数据排列规律
    dataset = dataset.shuffle(seed=42)

    def map_callback(examples):
        # tokenize_dataset：将文本 token 化并按 TRUNK_SIZE 切分/拼接
        # 返回 (result_dict, _)，result_dict 包含 input_ids 等字段
        result, _ = tokenize_dataset(examples, tokenizer, TRUNK_SIZE)
        return result

    # 批量 tokenize，num_proc=32 多进程加速
    train_dataset = dataset.map(
        map_callback,
        batched=True,
        batch_size=5000,
        remove_columns=dataset.column_names,  # 删除原始文本列，只保留 token 列
        num_proc=32,
    )

    # 持久化到磁盘，下次直接加载，跳过耗时的 tokenize 步骤
    train_dataset.save_to_disk(tokenized_datapath)

# 从磁盘加载已 tokenize 的数据集
train_dataset = Dataset.load_from_disk(tokenized_datapath)

# ── Data Collator ──────────────────────────────────────────────────────────────

# mlm=False：因果语言模型（CLM）不做掩码语言模型（MLM）
# collator 负责将样本 padding 到相同长度，并自动生成 labels（与 input_ids 相同）
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ── 训练参数 ───────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,

    # 继续预训练的核心：极小学习率，防止灾难性遗忘
    # 初次预训练通常用 1e-4 ~ 1e-3，这里用 1e-8 做微调式继续训练
    learning_rate=1e-8,

    warmup_ratio=0.1,              # 前 10% steps 做学习率线性预热
    lr_scheduler_type="cosine",    # 余弦退火调度，训练后期平滑降低学习率

    num_train_epochs=3,
    per_device_train_batch_size=24,
    gradient_accumulation_steps=16,  # 等效全局 batch_size = 24 * 16 * GPU数

    save_steps=1050,        # 每 1050 步保存一次 checkpoint
    save_total_limit=3,     # 最多保留 3 个 checkpoint，自动删除最旧的

    # 梯度检查点：用重计算换显存，训练速度略降但可支持更大 batch
    gradient_checkpointing=True,

    bf16=True,              # 启用 BF16 混合精度训练
    logging_steps=10,       # 每 10 步记录一次 loss 等指标
    report_to="wandb",      # 训练曲线上报到 Weights & Biases
)

# ── WandB 日志初始化 ───────────────────────────────────────────────────────────

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-pt",
        name="qwen-0.5B-pt-continue"  # run 名称与初次预训练区分
    )

# ── 自定义优化器（已注释，使用 Trainer 默认 AdamW）────────────────────────────
# 若需要自定义优化器参数（如 weight_decay），可取消注释并传入 optimizers 参数
# optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# ── 初始化 Trainer ─────────────────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    # optimizers=(optimizer, None)  # None 表示使用默认 lr_scheduler
)

# 清理显存碎片，为训练腾出空间
torch.cuda.empty_cache()

# ── 开始训练 & 保存 ────────────────────────────────────────────────────────────

# trainer.train() 会自动处理：多 GPU 分布式、梯度累积、混合精度、checkpoint 保存
trainer.train()

# 保存最终模型权重（safetensors 格式）
trainer.save_model()

# 单独保存 tokenizer，确保推理时词表与模型匹配
tokenizer.save_pretrained(output_path)
