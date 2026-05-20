"""
DPO 直接偏好优化训练脚本
========================

本脚本用于对 SFT 微调后的模型进行 DPO（Direct Preference Optimization）训练，
让模型学会生成更符合人类偏好的回答。

整体流程：
  1. 加载 SFT 阶段训练好的模型（作为起点）
  2. 加载偏好数据集（包含 prompt、chosen、rejected 三元组）
  3. 将数据预处理为 ChatML 格式
  4. 使用 DPOTrainer 进行训练
  5. 保存训练后的模型

DPO 的核心思想：
  传统 RLHF 需要先训练一个奖励模型（RM），再用 PPO 算法优化策略模型，流程复杂。
  DPO 巧妙地跳过了奖励模型，直接用偏好数据对（chosen vs rejected）来优化模型，
  数学上等价于 RLHF，但实现简单得多。

  简单来说：
  - 看到 chosen（好回答）→ 增大模型生成它的概率
  - 看到 rejected（差回答）→ 减小模型生成它的概率
  - 同时用参考模型（ref_model）防止模型偏离太远

依赖库：
  - trl: HuggingFace 的强化学习训练库，提供 DPOTrainer
  - transformers: 模型和分词器
  - wandb: 实验追踪和可视化
"""

# ============================================================
# 第一部分：导入依赖
# ============================================================
import os
import torch
import wandb                          # Weights & Biases 实验追踪平台
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# DPOConfig: DPO 训练的配置类（继承自 TrainingArguments，增加了 DPO 特有参数）
# DPOTrainer: DPO 训练器，封装了 DPO 损失函数计算和训练循环
from trl import DPOConfig, DPOTrainer
from utils.utils import find_files

# 解决 GPU 显存碎片化问题：限制 CUDA 内存分配器的最大分块为 64MB
# 当显存紧张时，这个设置可以减少显存碎片，避免 OOM（Out of Memory）错误
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# ============================================================
# 第二部分：路径和参数配置
# ============================================================

# 是否启用 wandb 日志记录（设为 False 可以离线训练）
WANDB_LOG = True

TMP_PATH = "/archive/share/cql/aaa/tmp"
REPO_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen"

# 训练输出路径（模型 checkpoint 会保存在这里）
output_path = "results/dpo"

# 偏好数据路径
data_path = "data/dpo"

# 基座模型路径：使用 SFT 阶段训练好的 checkpoint
# 注意：DPO 是在 SFT 模型的基础上继续训练，而不是从预训练模型开始
# 这是因为 DPO 需要模型已经具备基本的对话能力，才能在此基础上优化偏好
model_path = "results/sft-1/checkpoint-14000"
model_path = os.path.join(REPO_PATH, model_path)
output_path = os.path.join(REPO_PATH, output_path)
data_path = os.path.join(REPO_PATH, data_path)

# ============================================================
# 第三部分：加载模型和分词器
# ============================================================

# 加载 SFT 微调后的模型
# torch_dtype=torch.bfloat16: 使用 BF16 半精度，节省一半显存
# attn_implementation="flash_attention_2": 使用 FlashAttention-2 加速注意力计算
#   FlashAttention 通过优化 GPU 内存访问模式，大幅提升注意力计算速度并降低显存占用
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ============================================================
# 第四部分：加载偏好数据集
# ============================================================

# 加载 DPO 偏好数据（parquet 格式）
# 数据集中每条数据包含：
#   - prompt: 用户的问题
#   - chosen: 好的回答（人类/AI 偏好的）
#   - rejected: 差的回答（人类/AI 不偏好的）
data_files = [
    "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/dpo/test-00000-of-00001.parquet",
    "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/dpo/train-00000-of-00001.parquet"
]
dataset = load_dataset("parquet", data_files=data_files, split="train")

# 打乱数据集顺序（seed=42 保证可复现）
# 打乱是为了避免训练时数据的顺序偏差（比如同一类问题连续出现）
dataset = dataset.shuffle(seed=42)

# ============================================================
# 第五部分：数据预处理 —— 转换为 ChatML 格式
# ============================================================
# DPOTrainer 要求数据包含三个字段：prompt、chosen、rejected
# 这个函数将原始数据转换为 ChatML 格式的文本
#
# 转换前（原始数据）：
#   prompt: "如何学习编程？"
#   chosen: [{"role": "user", ...}, {"role": "assistant", "content": "好回答..."}]
#   rejected: [{"role": "user", ...}, {"role": "assistant", "content": "差回答..."}]
#
# 转换后（ChatML 格式）：
#   prompt:   "<|im_start|>user\n如何学习编程？<|im_end|>\n<|im_start|>assistant\n"
#   chosen:   "好回答...<|im_end|>"
#   rejected: "差回答...<|im_end|>"
#
# 注意：prompt 部分以 "assistant\n" 结尾但不包含回答内容，
#       chosen/rejected 只包含 assistant 的回答内容 + 结束符
#       DPOTrainer 会在内部将 prompt + chosen/rejected 拼接起来

def preprocess_dataset(examples):
    prompt, chosen, rejected = [], [], []
    for i in range(len(examples["prompt"])):
        # 构造 prompt：用户消息 + assistant 开头（等待模型续写）
        text = f"<|im_start|>user\n{examples['prompt'][i]}<|im_end|>\n<|im_start|>assistant\n"
        prompt.append(text)

        # 提取 chosen 回答（好回答）
        # assert 确保第二条消息确实是 assistant 的回答
        assert examples["chosen"][i][1]["role"] == "assistant"
        text = f"{examples['chosen'][i][1]['content']}<|im_end|>"
        chosen.append(text)

        # 提取 rejected 回答（差回答）
        assert examples["rejected"][i][1]["role"] == "assistant"
        text = f"{examples['rejected'][i][1]['content']}<|im_end|>"
        rejected.append(text)

    result = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    return result

# 对整个数据集应用预处理
# batched=True: 批量处理，速度更快
# batch_size=5000: 每批处理 5000 条
# remove_columns: 移除原始列，只保留处理后的 prompt/chosen/rejected
# num_proc=16: 使用 16 个进程并行处理
train_dataset = dataset.map(
    preprocess_dataset,
    batched=True,
    batch_size=5000,
    remove_columns=dataset.column_names,
    num_proc=16,
)

# ============================================================
# 第六部分：DPO 训练参数配置
# ============================================================
# DPOConfig 继承自 TrainingArguments，包含所有标准训练参数
# 加上 DPO 特有的参数（如 beta，控制 KL 散度惩罚强度）

training_args = DPOConfig(
    output_dir=output_path,            # 模型输出目录
    overwrite_output_dir=True,         # 覆盖已有输出

    # ---- 学习率相关 ----
    learning_rate=5e-7,                # 学习率：DPO 通常用很小的学习率（5e-7 ~ 5e-6）
                                       # 因为模型已经经过 SFT，只需要微调偏好方向
    warmup_ratio=0.1,                  # 预热比例：前 10% 的步数逐渐增大学习率
    lr_scheduler_type="cosine",        # 余弦退火：学习率先升后降，平滑收敛

    # ---- 训练规模 ----
    num_train_epochs=5,                # 训练 5 个 epoch
    per_device_train_batch_size=2,     # 每张 GPU 的 batch size
    gradient_accumulation_steps=16,    # 梯度累积 16 步
                                       # 等效 batch size = 2 × 16 = 32

    # ---- 保存和日志 ----
    save_steps=500,                    # 每 500 步保存一次 checkpoint
    save_total_limit=10,               # 最多保留 10 个 checkpoint（节省磁盘）
    bf16=True,                         # 使用 BF16 混合精度训练
    logging_steps=10,                  # 每 10 步记录一次日志
    report_to="wandb",                 # 日志发送到 wandb
)

# ============================================================
# 第七部分：初始化 wandb 实验追踪
# ============================================================

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-dpo",      # wandb 项目名
        name="qwen-0.5B-dpo"          # 本次实验名称
    )

# ============================================================
# 第八部分：创建 DPO 训练器并开始训练
# ============================================================

# DPOTrainer 的核心工作：
#   1. 自动创建参考模型（ref_model）：��制一份当前模型的权重，训练过程中冻结不更新
#   2. 对每个 (prompt, chosen, rejected) 三元组：
#      a. 用当前模型计算 chosen 和 rejected 的对数概率
#      b. 用参考模型计算 chosen 和 rejected 的对数概率
#      c. 计算 DPO 损失：鼓励当前模型相对于参考模型更偏好 chosen
#   3. 反向传播更新模型参数
#
# DPO 损失函数（简化版）：
#   loss = -log(σ(β * (log(π(chosen)/π_ref(chosen)) - log(π(rejected)/π_ref(rejected)))))
#   其中：
#     π = 当前模型，π_ref = 参考模型
#     β = KL 散度惩罚系数（默认 0.1），控制模型偏离参考模型的程度
#     σ = sigmoid 函数

trainer = DPOTrainer(
    model=model,                       # 要训练的模型（SFT checkpoint）
    train_dataset=train_dataset,       # 预处理后的偏好数据集
    args=training_args,                # 训练参数
    tokenizer=tokenizer,               # 分词器
    dataset_num_proc=16,               # 数据处理并行数
    max_length=1024,                   # prompt + response 的最大总长度
    max_prompt_length=512,             # prompt 部分的最大长度
                                       # 超过此长度的 prompt 会被截断
)

# 开始训练
trainer.train()

# ============================================================
# 第九部分：保存训练结果
# ============================================================

# 保存最终模型权重和分词器
# 保存后的模型可以直接用 AutoModelForCausalLM.from_pretrained() 加载
trainer.save_model()
tokenizer.save_pretrained(output_path)
