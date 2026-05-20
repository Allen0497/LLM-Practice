"""
GRPO（Group Relative Policy Optimization）训练脚本
======================================================

GRPO 是 DeepSeek-R1 提出的强化学习算法，用于训练大模型的推理能力。
与 PPO 最大的区别：**不需要单独训练奖励模型（RM）**，而是使用基于规则的奖励函数。

核心思想：
  1. 对同一个问题，让模型生成一组（Group）回答
  2. 用规则函数给每个回答打分
  3. 在组内做相对排名，好的回答加分，差的回答减分
  4. 用这个相对优势来更新策略

对比 PPO：
  PPO：需要 4 个模型（策略模型 + 参考模型 + 奖励模型 + 价值模型）
  GRPO：只需要 2 个模型（策略模型 + 参考模型），奖励由规则函数提供

本脚本实现了 DeepSeek-R1-Zero 风格的训练：
  - 从基座模型（Qwen2.5-0.5B）直接开始 GRPO 训练，不经过 SFT
  - 使用 <think>...</think><answer>...</answer> 格式引导模型学会"先思考再回答"
  - 奖励函数包括：格式奖励（是否遵循思考-回答格式）+ 准确性奖励（答案是否正确）
"""

import os
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from utils.grpo_utils import format_reward,accuracy_reward
from utils.utils import find_files,print_trainable_parameters,format_to_r1

# ============================================================
# 第一部分：环境配置
# ============================================================
# 优化 CUDA 内存分配策略
# max_split_size_mb=64：限制内存块���最大分割大小为 64MB
# 当 GPU 显存碎片化严重时，这个设置可以减少内存分配失败的概率
# GRPO 训练中需要同时为多个生成结果分配显存，碎片化问题比普通训练更严重
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# 是否使用 wandb 记录训练日志
WANDB_LOG = True
# 数据集缓存路径（加载 HuggingFace 数据集时使用）
TMP_PATH = "/archive/share/cql/aaa/tmp"

# ============================================================
# 第二部分：路径配置
# ============================================================
# 训练输出目录（保存 checkpoint 和最终模型）
# "grpo-zero" 表示这是 R1-Zero 风格：从基座模型直接 GRPO，不经��� SFT
output_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/grpo-zero"
# 基座模型路径（直接使用预训练模型，不使用 SFT 后的模型）
model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"
# 推理数据集路径（数学推理题，包含 problem 和 solution 两个字段）
data_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reasoning"

# ============================================================
# 第三部分：加载基座模型
# ============================================================
# 以 bfloat16 精度加载模型（节省显存，同时保持数值稳定性）
# 注意：GRPO 不需要像 PPO 那样使用 AutoModelForCausalLMWithValueHead
# 因为 GRPO 不需要价值模型（Value Head），这是它比 PPO 简单的一个原因
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 打印初始模型的参数量信息（应该是全部参数都可训练）
print_trainable_parameters(model)

# ============================================================
# 第四部分：加载和预处理数据集
# ============================================================
# 查找推理数据集目录下所有的 parquet 文件
directories = ["data"]
data_files = find_files(directories,"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reasoning")
# 加载数据集，只保留 problem（题目）和 solution（标准答案）两列
# problem 用于构造 prompt，solution 用于计算准确性奖励
dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["problem", "solution"], cache_dir=TMP_PATH)
dataset = dataset.shuffle(seed=42)
# 划分训练集和验证集（8% 作为验证集）
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.08).values()

# 将数据转换为 R1 格式的 prompt
# format_to_r1 会将每条数据转换为：
#   [{"role": "system", "content": "...先思考再回答..."},
#    {"role": "user", "content": problem}]
# 这个 system prompt 告诉模型要使用 <think>...</think><answer>...</answer> 格式
# 这是 GRPO 训练的关键设计：通过 prompt 引导模型学会"思维链"推理
train_dataset = train_dataset.map(format_to_r1)
train_dataset.remove_columns(["problem"])
valid_dataset = valid_dataset.map(format_to_r1)

# ============================================================
# 第五部分：LoRA 配置
# ============================================================
# 使用 LoRA 进行参数高效训练
# GRPO 训练中使用 LoRA 有额外的好处：
#   1. 节省显存（GRPO 需要同时生成多个回答，显存压力大）
#   2. 参考模型可以直接复用基座模型（去掉 LoRA 就是参考模型）
#      → 不需要额外的显存来存储参考模型
lora_config = LoraConfig(
    r=8,                              # LoRA 秩，控制低秩矩阵的维度
    lora_alpha=16,                    # LoRA 缩放系数（alpha/r = 2，表示 LoRA 更新的权重）
    lora_dropout=0.01,                # LoRA 层的 dropout 概率
    bias="none",                      # 不训练 bias 参数
    target_modules=["q_proj", "v_proj"],  # 只对 attention 的 Q 和 V 投影矩阵应用 LoRA
    task_type="CAUSAL_LM",            # 因果语言模型任务
)
# 将 LoRA 适配器应用到模型上
model = get_peft_model(model, lora_config)
# 打印 LoRA 后的可训练参数量（应该远小于全部参数量）
model.print_trainable_parameters()

# ============================================================
# 第六部分：GRPO 训练配置
# ============================================================
# GRPOConfig 继承自 TrainingArguments，但增加了 GRPO 特有的参数
training_args = GRPOConfig(
    output_dir=output_path,
    learning_rate=1e-5,               # 学习率，GRPO 通常使用较小的学习率保证训练稳定

    # ---- GRPO 特有参数 ----
    # remove_unused_columns=False 非常关键！
    # 默认情况下 HuggingFace Trainer 会移除模型不需要的列
    # 但我们的 accuracy_reward 函数需要访问 solution 列来判断答案是否正确
    # 如果被移除，奖励函数就无法获取标准答案了
    remove_unused_columns=False,

    # max_completion_length：模型每次生成的最大 token 数
    # 这里设为 256，因为数学题的推理过程不需要太长
    # 设太大会：1. 增加显存消耗  2. 减慢训练速度  3. 可能生成冗余内容
    max_completion_length=256,

    # num_generations：每个问题生成的回答数量（即"Group"的大小）
    # 这是 GRPO 的核心参数！
    # 对每个问题生成 8 个不同的回答，然后在这 8 个回答中做相对排名
    # 设太小（如2）：排名信号太弱，学习效率低
    # 设太大（如32）：显存不够，生成太慢
    # 8 是一个经验上的平衡点
    num_generations=8,

    # ---- 常规训练参数 ----
    gradient_accumulation_steps=16,   # 梯度累积步数（等效 batch_size = 24 × 16 = 384）
    num_train_epochs=3,               # 训练轮数
    bf16=True,                        # 使用 bfloat16 混合精度训练
    per_device_train_batch_size = 24,  # 每个 GPU 的 batch size
    max_prompt_length=512,            # prompt 的最大 token 数
    report_to="wandb",                # 训练日志报告到 wandb
    logging_steps=10,                 # 每 10 步记录一次日志
    save_strategy="steps",            # 按步数保存 checkpoint
    save_steps=10,                    # 每 10 步保存一次（设置较小便于观察训练过程）
)

# ============================================================
# 第七部分：初始化 wandb 实验追踪
# ============================================================
if WANDB_LOG:
    wandb.login()
    wandb.init(
        # 项目名称中包含 "grpo-zero" 表明这是 R1-Zero 风格的训练
        project="qwen-0.5B-deepseek-grpo-zero",name="q2.5-0.5B-r1-grpo-zero"
    )

# ============================================================
# 第八部分：初始化 GRPOTrainer 并开始训练
# ============================================================
# GRPOTrainer 是 TRL 库提供的 GRPO 训练器
# 与 PPOTrainer 相比，最大的区别是：
#   1. 不需要单独的奖励模型和价值模型
#   2. 通过 reward_funcs 参数传入规则函数列表
#   3. 内部自动处理分组生成和相对排名
#
# reward_funcs 参数接收一个奖励函数列表：
#   - format_reward：检查格式（是否包含 <think>...</think><answer>...</answer>）
#   - accuracy_reward：检查答案正确性（与标准答案比对）
# 最终奖励 = 所有奖励函数的加权和
trainer = GRPOTrainer(
    model=model, reward_funcs=[format_reward, accuracy_reward], args=training_args, train_dataset=train_dataset
)

# 开始训练
# GRPOTrainer 的训练循环（内部自动完成）：
#   1. 从训练集中采样一批问题
#   2. 对每个问题，用当前策略模型生成 num_generations=8 个回答
#   3. 对每个回答，调用所有 reward_funcs 计算奖励
#   4. 在每组 8 个回答中，计算组内相对优势（好的回答 vs 差的回答）
#   5. 用 PPO-Clip 风格的目标函数更新策略模型
#   6. 通过 KL 散度约束确保策略不会偏离参考模型太远
print("Training...")
trainer.train()
# 保存最终模型和 tokenizer
trainer.save_model()
tokenizer.save_pretrained(output_path)
