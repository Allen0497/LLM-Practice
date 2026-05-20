"""
PPO 近端策略优化训练脚本
========================

本脚本用于对模型进行 PPO（Proximal Policy Optimization，近端策略优化）训练，
这是经典 RLHF 流程的最后一步。

整体流程：
  1. 加载策略模型（带 Value Head 的语言模型）
  2. 加载训练好的奖励模型（上一步训练的 RM）
  3. 对每个 batch：
     a. 策略模型根据问题生成回答
     b. 奖励模型给回答打分
     c. PPO 算法根据奖励分数更新策略模型
  4. 重复直到收敛

PPO 的核心思想（强化学习视角）：
  - 策略模型 = "演员"（Actor），负责生成回答
  - 奖励模型 = "环境"（Environment），给回答打分
  - PPO 算法 = "教练"，根据分数指导演员改进
  - 目标：让演员生成的回答获得更高的分数

与 DPO 的区别：
  DPO：离线学习，直接从固定的偏好数据中学习
  PPO：在线学习，策略模型不断生成新回答 → 奖励模型实时打分 → 策略更新
       这个"生成→打分→更新"的循环是 PPO 的核心

依赖库：
  - trl: PPOTrainer, AutoModelForCausalLMWithValueHead
  - peft: LoRA 高效微调
  - accelerate: 分布式训练支持
"""

# ============================================================
# 第一部分：导入依赖
# ============================================================
import torch
import wandb
from accelerate import Accelerator       # HuggingFace 分布式训练框架
from datasets import load_dataset
from peft import LoraConfig               # LoRA 高效微调配置
from tqdm import tqdm
from transformers import (
    Adafactor,          # 内存高效的优化器（比 AdamW 省显存）
    AutoTokenizer,
    pipeline,           # HuggingFace 推理管道（用于加载奖励模型）
    set_seed,
)

# AutoModelForCausalLMWithValueHead：在语言模型基础上增加了一个 Value Head
#   语言模型头（LM Head）：预测下一个 token → 用于生成回答
#   价值头（Value Head）：预测当前状态的价值 → 用于 PPO 的优势估计
# PPOConfig：PPO 训练的配置类
# PPOTrainer：PPO 训练器，封装了 PPO 算法的完整训练循环
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# LengthSampler：随机采样生成长度（让模型生成不同长度的回答，增加多样性）
from trl.core import LengthSampler
from utils.utils import find_files, preprocess_ppo_dataset, collator_ppo
tqdm.pandas()

# ============================================================
# 第二部分：超参数和路径配置
# ============================================================

WANDB_LOG = True
NUM_EPOCH = 10                     # 外层循环的 epoch 数

# 策略模型：从预训练基座模型开始（PPO 会用 LoRA 微调）
MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"

# 奖励模型：上一步训练好的 RM
REWARD_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/rm/final_model"

# PPO 训练结果输出路径
OUTPUT_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/ppo"

TMP_PATH = "/archive/share/cql/aaa/tmp"
SEED = 42
MAX_LENGTH = 512
GRADIENT_CHECKPOINTING = True
LR = 1.41e-5                      # PPO 学习率
STEPS = 20000                      # PPO 总训练步数
BS = 8                             # mini batch size
TRAIN_SUBSET = 0                   # 数据子集大小（0 = 全部）
NUM_PROC = 16

set_seed(SEED)

# ============================================================
# 第三部分：初始化 wandb
# ============================================================

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-ppo", name="qwen-0.5B-ppo"
    )

# ============================================================
# 第四部分：加载和预处理数据集
# ============================================================

# PPO 的数据集只需要问题（不需要 chosen/rejected）
# 因为 PPO 是在线学习：策略模型自己生成回答，奖励模型实时打分
data_files = find_files(["rl"], "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reward/data")
train_dataset = load_dataset("parquet", data_files=data_files, split="train", cache_dir=TMP_PATH)
original_columns = train_dataset.column_names
if TRAIN_SUBSET > 0:
    train_dataset = train_dataset.select(range(TRAIN_SUBSET))

# 加载策略模型的分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载奖励模型的分词器（奖励模型可能用不同的分词器）
rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_PATH)
if rm_tokenizer.pad_token is None:
    rm_tokenizer.pad_token = rm_tokenizer.eos_token

# 预处理数据集
# preprocess_ppo_dataset 的作用（定义在 utils/utils.py）：
#   将问题转换为 "Question: {question}\n\nAnswer: " 的格式
#   然后 tokenize 为 input_ids
#   输出：{"query": "问题文本", "input_ids": [token_ids]}
dataset = train_dataset.map(
    lambda examples: preprocess_ppo_dataset(examples, tokenizer),
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=original_columns,
)
# 过滤超长样本
dataset = dataset.filter(lambda x: len(x["input_ids"]) < 512, batched=False, num_proc=NUM_PROC)
# 设置为 PyTorch tensor 格式（PPOTrainer 需要）
dataset.set_format(type="torch")

# ============================================================
# 第五部分：PPO 配置
# ============================================================

# 获取当前 GPU 设备编号（用于多卡训练时的设备分配）
current_device = Accelerator().local_process_index

# PPO 训练配置
config = PPOConfig(
    steps=STEPS,                       # 总训练步数
    model_name=MODEL_PATH,             # 模型名称（用于日志）
    learning_rate=LR,                  # 学习率
    log_with='wandb',                  # 日志工具
    batch_size=BS * 4,                 # 每次 PPO 更新使用的总样本数 = 32
    mini_batch_size=BS,                # 每个 mini batch 的大小 = 8
                                       # 一次 PPO 更新会把 batch 分成 32/8=4 个 mini batch
    gradient_accumulation_steps=4,     # 梯度累积步数
    optimize_cuda_cache=True,          # 优化 CUDA 缓存（减少显存碎片）

    # ---- PPO 特有参数 ----
    target_kl=0.1,                     # 目标 KL 散度
                                       # 当策略更新导致的 KL 散度超过此值时，提前停止当前 epoch
                                       # 这是 PPO 的核心约束——防止策略更新太大
    ppo_epochs=3,                      # 每次 PPO 更新的内部 epoch 数
                                       # 对同一批数据重复优化 3 次（提高数据利用率）
    seed=SEED,
    init_kl_coef=0.2,                  # KL 惩罚的初始系数
                                       # 控制策略偏离参考模型的惩罚强度
    adap_kl_ctrl=True,                 # 自适应 KL 控制
                                       # 自动调整 KL 系数，使实际 KL 散度接近 target_kl
)

# LoRA 配置（用于策略模型的高效微调）
lora_config = LoraConfig(
    r=16,                              # LoRA 秩（比 RM 的 r=8 更大，因为生成任务更复杂）
    lora_alpha=32,                     # 缩放系数
    lora_dropout=0.05,                 # dropout（比 RM 小，因为 PPO 本身有 KL 正则化）
    bias="none",                       # 不训练 bias 参数
    task_type="CAUSAL_LM",            # 任务类型：因果语言模型（生成）
)

# ============================================================
# 第六部分：加载策略模型（带 Value Head）
# ============================================================

# AutoModelForCausalLMWithValueHead：在普通语言模型基础上增加了 Value Head
#
# 模型结构：
#   ┌─────────────────────────────────────────────┐
#   │  Qwen2.5-0.5B (Transformer 基座)            │
#   │         │                                    │
#   │    ┌────┴────┐                               │
#   │    │         │                               │
#   │    ▼         ▼                               │
#   │  LM Head   Value Head                        │
#   │  (生成)    (价值估计)                         │
#   │    │         │                               │
#   │    ▼         ▼                               │
#   │  下一个     当前状态                          │
#   │  token     的价值                             │
#   │  概率分布   (标量)                            │
#   └─────────────────────────────────────────────┘
#
# Value Head 的作用：
#   PPO 算法需要估计"当前状态有多好"（即价值函数 V(s)）
#   Value Head 就是用来做这个估计的
#   它帮助计算"优势函数" A(s,a) = R - V(s)
#   优势函数告诉 PPO："这个回答比平均水平好多少？"

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map={"": current_device},   # 放到当前 GPU
    peft_config=lora_config,           # 使用 LoRA 微调
)

# 使用 Adafactor 优化器（比 AdamW 更省显存）
# AdamW 需要为每个参数维护两个状态（一阶矩和二阶矩）→ 显存 ×3
# Adafactor 通过矩阵分解近似二阶矩 → 显存更少
optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),  # 只优化可训练参数（LoRA）
    scale_parameter=False,     # 不使用参数缩放
    relative_step=False,       # 不使用相对步长（使用固定学习率）
    warmup_init=False,         # 不使用 warmup 初始化
    lr=config.learning_rate,   # 使用配置中的学习率
)

# 创建 PPO 训练器
# ref_model=None：PPOTrainer 会自动创建参考模型（复制策略模型的初始权重）
# collator_ppo：简单的数据整理函数，将 batch 中的数据按 key 组织
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,                    # 自动创建参考模型
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator_ppo,
    optimizer=optimizer,
)

# ============================================================
# 第七部分：加载奖励模型
# ============================================================

# 使用 HuggingFace pipeline 加载奖励模型
# 虽然叫 "sentiment-analysis"，但实际上是加载一个序列分类模型
# 奖励模型本质上就是一个序列分类模型（输出一个分数）
sent_kwargs = {
    "return_all_scores": True,         # 返回所有分数（不只是最高分的类别）
    "function_to_apply": "none",       # 不对输出做 softmax/sigmoid 变换
                                       # 直接返回原始 logits 作为奖励分数
    "batch_size": 16,                  # 推理 batch size
    "truncation": True,                # 超长文本自动截断
}

sentiment_pipe = pipeline(
    "sentiment-analysis",              # 任务类型（借用情感分析的 pipeline）
    model=REWARD_PATH,                 # 奖励模型路径
    device_map={"": current_device},   # 放到当前 GPU
    tokenizer=rm_tokenizer,            # 奖励模型的分词器
    return_token_type_ids=False,       # Qwen 不需要 token_type_ids
)
# 设置 pad_token_id（避免推理时的警告）
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id

# ============================================================
# 第八部分：生成配置
# ============================================================

# 策略模型生成回答时的参数
generation_kwargs = {
    "top_k": 0.0,                      # 不使用 top-k 采样
    "top_p": 1.0,                      # 不使用 nucleus 采样（top-p=1 等于不过滤）
    "do_sample": True,                 # 使用采样（而非贪心解码）
                                       # PPO 需要探索性，所以必须采样
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,           # 设置一个很大的 eos_token_id
                                       # 目的是让模型不会过早停止生成
                                       # 生成长度由 length_sampler 控制
}

# 随机采样生成长度：每次生成的回答长度在 32~128 token 之间随机
# 这样做的好处：让模型学会生成不同长度的回答，增加训练多样性
output_min_length = 32
output_max_length = 128
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# ============================================================
# 第九部分：PPO 训练主循环
# ============================================================
# PPO 训练的核心循环，每个 batch 执行三步：
#
#   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
#   │  步骤 1      │     │  步骤 2      │     │  步骤 3      │
#   │  策略模型    │ ──→ │  奖励模型    │ ──→ │  PPO 更新    │
#   │  生成回答    │     │  给回答打分  │     │  更新策略    │
#   └──────────────┘     └──────────────┘     └──────────────┘
#
# 这就是强化学习的经典循环：行动 → 获得奖励 → 更新策略

for epoch in tqdm(range(NUM_EPOCH), "epoch: "):
    # 检查是否超过 PPO 配置的总 epoch 数
    if epoch >= config.total_ppo_epochs:
        break

    for batch in tqdm(ppo_trainer.dataloader):
        # ---- 步骤 1：策略模型生成回答 ----
        question_tensors = batch["input_ids"]

        # ppo_trainer.generate() 使用策略模型为每个问题生成回答
        # return_prompt=False：只返回生成的回答部分（不包含问题）
        # length_sampler：随机决定每个回答的生成长度
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        # 将 token ids 解码为文本（用于送给奖励模型）
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # ---- 步骤 2：奖励模型打分 ----
        # 将问题和回答拼接，送给奖励模型打分
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        # 提取奖励分数（每个回答一个标量分数）
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

        # ---- 步骤 3：PPO 更新 ----
        # ppo_trainer.step() 是 PPO 算法的核心：
        #   1. 计算旧策略的对数概率和价值估计
        #   2. 计算优势函数 A = R - V（回答比平均水平好多少）
        #   3. 用 PPO-Clip 目标函数更新策略模型
        #   4. 同时更新 Value Head（让价值估计更准确）
        #   5. 计算 KL 散度，如果超过 target_kl 则提前停止
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        # 记录训练统计信息到 wandb
        ppo_trainer.log_stats(stats, batch, rewards)

    # 每个 epoch 保存一次 checkpoint
    if epoch and epoch % 1 == 0:
        ppo_trainer.save_pretrained(OUTPUT_PATH + f"step_{epoch}")

# ============================================================
# 第十部分：保存最终模型
# ============================================================
ppo_trainer.save_pretrained(OUTPUT_PATH)
