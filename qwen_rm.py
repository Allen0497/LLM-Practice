"""
奖励模型（Reward Model）训练脚本
================================

本脚本用于训练一个奖励模型（RM），它能给模型的回答打分，
分数越高表示回答越好。训练好的奖励模型将在 PPO 训练中作为"裁判"使用。

整体流程：
  1. 加载预训练模型，改造为序列分类模型（输出一个标量分数）
  2. 用 LoRA 高效微调（只训练少量参数）
  3. 加载偏好数据集（好回答 vs 差回答）
  4. 用 pairwise ranking loss 训练：让好回答的分数 > 差回答的分数
  5. 保存训练好的奖励模型

奖励模型的作用：
  在 RLHF (PPO) 流程中，奖励模型充当"裁判"的角色：
  - 策略模型生成一个回答
  - 奖励模型给这个回答打分
  - PPO 算法根据分数来更新策略模型
  分数高 → 鼓励策略模型生成类似的回答
  分数低 → 抑制策略模型生成类似的回答

与 DPO 的区别：
  DPO 直接从偏好数据学习，不需要奖励模型
  PPO 需要一个独立的奖励模型来提供实时的奖励信号
  本脚本训练的奖励模型就是为 PPO 准备的

依赖库：
  - peft: 参数高效微调库（LoRA）
  - evaluate: HuggingFace 评估指标库
  - 自定义 RewardTrainer 和 RewardDataCollatorWithPadding（见 utils/rm_utils.py）
"""

# ============================================================
# 第一部分：导入依赖
# ============================================================
import evaluate                    # HuggingFace 评估指标库
import numpy as np
import wandb
import torch
from datasets import load_dataset

# LoRA 相关：参数高效微调，只训练模型中很小一部分参数
# LoraConfig: LoRA 配置（秩、缩放系数、dropout 等）
# TaskType: 任务类型枚举（SEQ_CLS = 序列分类）
# get_peft_model: 将普通模型转换为 LoRA 模型
from peft import LoraConfig, TaskType, get_peft_model

from transformers import (
    AutoModelForSequenceClassification,  # 序列分类模型（输出一个分数而非生成文本）
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# 自定义的奖励模型工具类（详见 utils/rm_utils.py）
# RewardDataCollatorWithPadding: 处理偏好对数据的 padding
# RewardTrainer: 实现 pairwise ranking loss 的自定义训练器
from utils.rm_utils import RewardDataCollatorWithPadding, RewardTrainer
from utils.utils import find_files, preprocess_rm_dataset

# ============================================================
# 第二部分：超参数和路径配置
# ============================================================

WANDB_LOG = True

# 注意：奖励模型从预训练基座模型开始训练（不是 SFT 模型）
# 因为奖励模型的任务是"打分"而非"生成"，不需要对话能力
MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"
OUTPUT_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/rm"
TMP_PATH = "/archive/share/cql/aaa/tmp"

SEED = 42                          # 随机种子（保证可复现）
MAX_LENGTH = 512                   # 输入序列最大长度
GRADIENT_CHECKPOINTING = True      # 梯度检查点：用时间换显存
                                   # 训练时不保存所有中间激活值，需要时重新计算
                                   # 可以大幅降低显存占用，但训练速度会慢约 20%
LR = 2e-5                         # 学习率
BS = 32                            # batch size
TRAIN_SUBSET = 0                   # 训练集子集大小（0 = 使用全部数据）
EVAL_SUBSET = 0                    # 验证集子集大小（0 = 使用全部数据）
NUM_PROC = 16                      # 数据处理并行进程数

# 设置全局随机种子，确保实验可复现
set_seed(SEED)

# ============================================================
# 第三部分：加载偏好数据集
# ============================================================

# 加载训练集（包含偏好对：question + response_j + response_k）
# response_j 是好回答，response_k 是差回答
data_files = find_files(["reward"], "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reward/data")
train_dataset = load_dataset("parquet", data_files=data_files[:1], split="train", cache_dir=TMP_PATH)
train_dataset = train_dataset.shuffle(seed=42)

# 加载验证集（用于训练过程中评估模型的准确率）
data_files = find_files(["evaluation"], "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reward/data")
eval_dataset = load_dataset("parquet", data_files=data_files[:1], split="train", cache_dir=TMP_PATH)
eval_dataset = eval_dataset.shuffle(seed=42)

# 可选：截取子集用于快速调试
if TRAIN_SUBSET > 0:
    train_dataset = train_dataset.select(range(TRAIN_SUBSET))
if EVAL_SUBSET > 0:
    eval_dataset = eval_dataset.select(range(EVAL_SUBSET))

# ============================================================
# 第四部分：加载模型（序列分类 + LoRA）
# ============================================================

# LoRA 配置
# 奖励模型使用 LoRA 微调，而不是全量微调，原因：
#   1. 节省显存和训练时间
#   2. 奖励模型的任务相对简单（打分），不需要更新所有参数
#   3. LoRA 有正则化效果，防止过拟合
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,    # 任务类型：序列分类（Sequence Classification）
    inference_mode=False,           # 训练模式（非推理模式）
    r=8,                            # LoRA 秩：低秩矩阵的维度
                                    # r 越大，可训练参数越多，表达能力越强
                                    # r=8 是一个常用的平衡值
    lora_alpha=32,                  # LoRA 缩放系数：实际缩放 = alpha/r = 32/8 = 4
                                    # 控制 LoRA 更新的幅度
    lora_dropout=0.1,               # LoRA dropout：防止过拟合
)

# 加载序列分类模型
# AutoModelForSequenceClassification vs AutoModelForCausalLM：
#   CausalLM：输入文本 → 输出下一个 token 的概率分布（用于生成）
#   SequenceClassification：输入文本 → 输出一个标量分数（用于分类/打分）
#
# num_labels=1：输出一个分数（而不是多个类别的概率）
# 模型结构：Qwen 基座 + 一个线性分类头（hidden_size → 1）
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=1, torch_dtype=torch.bfloat16
)

# 将普通模型转换为 LoRA 模型
# 只有 LoRA 适配器的参数会被训练，原始模型参数冻结
model = get_peft_model(model, peft_config)

# 打印可训练参数数量（通常只有原始模型的 0.1% ~ 1%）
model.print_trainable_parameters()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_auth_token=True)

# 设置 padding token
# Qwen 模型默认没有 pad_token，这里用 eos_token 代替
# 这是一个常见的做法，因为 padding 部分会被 attention_mask 屏蔽
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 关闭 KV cache（与梯度检查点不兼容）
# KV cache 是推理加速技术，训练时不需要
model.config.use_cache = not GRADIENT_CHECKPOINTING

# ============================================================
# 第五部分：数据预处理
# ============================================================

original_columns = train_dataset.column_names

# preprocess_rm_dataset 的作用（定义在 utils/utils.py）：
# 将原始数据转换为模型输入格式：
#   输入：question, response_j, response_k
#   输出：input_ids_j, attention_mask_j, input_ids_k, attention_mask_k
#
# 转换格式：
#   "Question: {question}\n\nAnswer: {response_j}" → tokenize → input_ids_j
#   "Question: {question}\n\nAnswer: {response_k}" → tokenize → input_ids_k

train_dataset = train_dataset.map(
    lambda examples: preprocess_rm_dataset(examples, tokenizer),
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=original_columns,
)
# 过滤掉超过最大长度的样本（避免截断导致信息丢失）
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= MAX_LENGTH and len(x["input_ids_k"]) <= MAX_LENGTH,
    num_proc=NUM_PROC,
)

# 对验证集做同样的处理
eval_dataset = eval_dataset.map(
    lambda examples: preprocess_rm_dataset(examples, tokenizer),
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=original_columns,
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_j"]) <= MAX_LENGTH and len(x["input_ids_k"]) <= MAX_LENGTH,
    num_proc=NUM_PROC,
)

# ============================================================
# 第六部分：定义评估指标
# ============================================================

# 加载准确率指标
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """
    计算奖励模型的准确率。

    准确率的定义：在所有偏好对中，模型给好回答打的分 > 差回答打的分 的比例。

    eval_pred 包含：
      predictions: shape (2, batch_size)，第 0 行是 rewards_j，第 1 行是 rewards_k
      labels: 空（奖励模型没有传统意义上的 label）

    np.argmax(predictions, axis=0)：
      对每个样本，看 rewards_j 和 rewards_k 哪个更大
      如果 rewards_j > rewards_k → argmax = 0（正确，因为 j 是好回答）
      如果 rewards_k > rewards_j → argmax = 1（错误）

    labels = np.zeros(...)：
      正确答案全是 0（因为 j 总是好回答）

    最终：predictions == labels 的比例就是准确率
    """
    predictions, _ = eval_pred
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

# ============================================================
# 第七部分：训练参数配置
# ============================================================

training_args = TrainingArguments(
    output_dir=f"/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/rm/{MODEL_PATH.split('/')[-1]}_peft_stack-exchange-paired__{TRAIN_SUBSET}_{LR}",
    learning_rate=LR,                          # 学习率 2e-5（比 DPO 大，因为 LoRA 只更新少量参数）
    per_device_train_batch_size=BS,            # 每 GPU batch size = 32
    per_device_eval_batch_size=BS,             # 验证时的 batch size
    num_train_epochs=3,                        # 训练 3 个 epoch
    weight_decay=0.001,                        # 权重衰减（L2 正则化，防止过拟合）
    eval_strategy="steps",                     # 按步数进行验证
    eval_steps=500,                            # 每 500 步验证一次
    save_strategy="steps",                     # 按步数保存 checkpoint
    save_steps=500,                            # 每 500 步保存一次
    gradient_accumulation_steps=16,            # 梯度累积 16 步
                                               # 等效 batch size = 32 × 16 = 512
    gradient_checkpointing=GRADIENT_CHECKPOINTING,  # 梯度检查点（省显存）
    remove_unused_columns=False,               # 不自动移除"多余"列
                                               # 必须设为 False！因为我们的数据有自定义列名
                                               # （input_ids_j/k），Trainer 默认会把它们删掉
    label_names=[],                            # 没有传统 label（奖励模型用 pairwise loss）
    bf16=True,                                 # BF16 混合精度
    logging_strategy="steps",
    logging_steps=10,
    lr_scheduler_type="linear",                # 线性学习率衰减
    seed=SEED,
    report_to="wandb",
)

# ============================================================
# 第八部分：初始化 wandb 并开始训练
# ============================================================

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-rm", name="qwen-0.5B-rm"
    )

# 创建 RewardTrainer（自定义训练器，实现了 pairwise ranking loss）
# RewardDataCollatorWithPadding 负责将偏好对数据 padding 到相同长度
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)

# 开始训练
trainer.train()

# ============================================================
# 第九部分：保存奖励模型
# ============================================================
# 保存 LoRA 适配器权重（不是完整模型，只保存 LoRA 部分）
# 加载时需要先加载基座模型，再加载 LoRA 权重
model.save_pretrained(OUTPUT_PATH)
