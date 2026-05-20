"""
知识蒸馏训练脚本（模型蒸馏）
======================================================

这是知识蒸馏的第二步：用 Teacher 模型的 logits 分布来指导 Student 模型训练。

与数据蒸馏（qwen_distill_data.py）的区别：
  数据蒸馏：Student 只学习 Teacher 的最终输出文本（模仿答案）
  模型蒸馏：Student 学习 Teacher 的 logits 概率分布（模仿思维方式）

模型蒸馏的核心思想（Hinton 2015）：
  大模型的 softmax 输出不只包含"正确答案"，
  还包含了模型对各个 token 的"置信度分布"——这是"暗知识"（Dark Knowledge）。
  例如：对于"1+1=?"，Teacher 可能给出：
    "2" → 0.95, "二" → 0.03, "两" → 0.01, ...
  这个分布比单纯的标签 "2" 包含更多信息。
  Student 通过对齐这个分布，能学到 Teacher 的推理模式。

本脚本的损失函数：
  总损失 = α × SFT损失 + (1-α) × KL散度损失
  SFT损失：让 Student 生成正确的文本（监督学习）
  KL散度损失：让 Student 的 logits 分布接近 Teacher（蒸馏学习）
  α=1 时退化为纯 SFT；α=0 时退化为纯蒸馏

数据来源：
  使用 numina-deepseek-DeepSeek-R1-Distill-Qwen-7B 数据集
  该数据集包含 DeepSeek-R1-Distill-Qwen-7B 生成的带推理链的数学题解答
  字段：problem（题目）+ generation（带推理链的回答）
"""

import os
import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.distill_utils import DistillConfig, DistillTrainer
from utils.utils import find_files, print_trainable_parameters, format_to_chatml

# 优化 CUDA 内存分配，减少显存碎片
# 蒸馏训练需要同时加载 Student 和 Teacher 两个模型，显存压力较大
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

WANDB_LOG = True
TMP_PATH = "/archive/share/cql/aaa/tmp"

# ============================================================
# 第一部分：路径配置
# ============================================================
output_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/distill"
# Student 模型：要被训练的小模型（Qwen2.5-0.5B）
model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"
# Teacher 模型：提供知识的大模型
# 注意：这里 teacher 和 student 用的是同一个模型路径（仅作演示）
# 实际蒸馏中，teacher 应该是更大、更强的模型（如 Qwen2.5-7B 或 DeepSeek-R1）
teacher_model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"

# ============================================================
# 第二部分：加载模型
# ============================================================
# Student 模型：需要训练，会被 DistillTrainer 更新参数
# flash_attention_2：使用 Flash Attention 2 加速注意力计算，节省显存
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
# Teacher 模型：只用于前向传播计算 logits，参数全程冻结（eval 模式）
# DistillTrainer 内部会调用 teacher_model.eval() 并禁用梯度计算
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print_trainable_parameters(model)

# ============================================================
# 第三部分：加载和预处理数据集
# ============================================================
# 数据集包含 Teacher 模型（DeepSeek-R1-Distill-Qwen-7B）生成的推理数据
# 字段：problem（数学题）+ generation（带推理链的完整回答）
directories = ["data"]
data_files = find_files(
    directories,
    "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/distill/numina-deepseek-DeepSeek-R1-Distill-Qwen-7B"
)
dataset = load_dataset(
    "parquet", data_files=data_files, split="train",
    columns=["problem", "generation"], cache_dir=TMP_PATH
)
dataset = dataset.shuffle(seed=42)
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.08).values()

# 将数据转换为 ChatML 格式
# format_to_chatml 将 (problem, generation) 转换为：
#   {"messages": [{"role": "user", "content": problem},
#                 {"role": "assistant", "content": generation}]}
# DistillTrainer 使用 DataCollatorForChatML 处理这种格式
train_data = format_to_chatml(train_dataset)
valid_data = format_to_chatml(valid_dataset)
train_dataset = Dataset.from_dict(train_data)
valid_dataset = Dataset.from_dict(valid_data)

# ============================================================
# 第四部分：训练参数配置
# ============================================================
training_args = DistillConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    eval_steps=1000,
    learning_rate=1e-5,
    warmup_ratio=0.1,

    # ---- 蒸馏特有参数 ----
    # temperature：软化 logits 分布的温度系数
    # 温度越高，概率分布越平滑，小概率 token 的信息被放大
    # Hinton 原论文建议 temperature=4~10，这里用 0.9（接近原始分布）
    temperature=0.9,

    # alpha：控制 SFT 损失和蒸馏损失的权重
    # 总损失 = alpha × SFT损失 + (1-alpha) × KL散度损失
    # alpha=1.0：纯 SFT，完全忽略 Teacher 的 logits（退化为普通 SFT）
    # alpha=0.0：纯蒸馏，只对齐 Teacher 的 logits 分布
    # alpha=0.5：两者各占一半（Hinton 原论文的推荐）
    # 这里 alpha=1 说明当前配置偏向 SFT，蒸馏损失权重为 0
    alpha=1,

    # max_new_tokens：Teacher 模型生成时的最大 token 数
    # 数学推理链可能很长，设为 1024 保证完整性
    max_new_tokens=1024,

    # ---- 常规训练参数 ----
    lr_scheduler_type="cosine",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,   # 等效 batch_size = 4 × 16 = 64
    save_steps=1000,
    save_total_limit=3,
    bf16=True,
    logging_steps=10,
    report_to="wandb",
)

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-deepseek-distill", name="q2.5-0.5B-r1-distill-qwen-7B"
    )

# ============================================================
# 第五部分：初始化 DistillTrainer 并训练
# ============================================================
# DistillTrainer 继承自 SFTTrainer，重写了 compute_loss 方法
# 核心改动：在标准 SFT 损失之外，增加了 KL 散度蒸馏损失
trainer = DistillTrainer(
    model=model,                    # Student 模型（要训练的）
    teacher_model=teacher_model,    # Teacher 模型（提供知识的）
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

print("Training...")
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(output_path)
