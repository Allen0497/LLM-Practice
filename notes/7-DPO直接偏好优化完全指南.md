# 🎯 DPO 直接偏好优化完全指南 —— 从零理解到动手实践

> 本文档配合 `qwen_dpo.py` 代码，手把手带你理解 DPO 训练的每一个环节。
> 假设你已经完成了预训练、SFT、评估和偏好数据合成的学习，我们在此基础上继续深入。

---

## 📋 目录

- [第零章：DPO 在大模型训练中的位置](#第零章dpo-在大模型训练中的位置)
- [第一章：从 RLHF 到 DPO —— 为什么需要偏好对齐？](#第一章从-rlhf-到-dpo--为什么需要偏好对齐)
- [第二章：DPO 的数学原理（直觉版）](#第二章dpo-的数学原理直觉版)
- [第三章：DPO 的数据格式 —— 偏好三元组](#第三章dpo-的数据格式--偏好三元组)
- [第四章：参考模型（Reference Model）的作用](#第四章参考模型reference-model的作用)
- [第五章：DPOTrainer 与训练参数详解](#第五章dpotrainer-与训练参数详解)
- [第六章：代码逐行对照解读](#第六章代码逐行对照解读)
- [第七章：DPO vs RLHF(PPO) 的对比](#第七章dpo-vs-rlhfppo-的对比)
- [附录：常见问题](#附录常见问题)

---

## 第零章：DPO 在大模型训练中的位置

```
┌─────────────────────────────────────────────────────────────────────┐
│                     大模型训练完整流程                                │
│                                                                     │
│   ① 预训练（Pre-Training）                                          │
│   │  让模型阅读海量文本，学会语言的基本规律                           │
│   │                                                                 │
│   ② 指令微调（SFT）                                                 │
│   │  用"问题-回答"数据教模型如何按指令回答                           │
│   │                                                                 │
│   ③ 偏好对齐                                                        │
│   │                                                                 │
│   │  ③-a 偏好数据合成 ✅ 已完成                                     │
│   │  │                                                               │
│   │  ③-b DPO 直接偏好优化    ← 📍 我们现在在这里！                  │
│   │  │   用偏好数据直接训练模型，无需奖励模型                        │
│   │  │                                                               │
│   │  ③-c 训练奖励模型（RM）                                         │
│   │  ③-d PPO 近端策略优化                                           │
│   │                                                                 │
│   ④ 评估与部署                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**打个比方：**

| 阶段 | 类比 | 做什么 |
|------|------|--------|
| 预训练 | 上学读书12年 | 博览群书，积累知识 |
| SFT | 岗前培训 | 学会按要求完成任务 |
| 偏好数据合成 | 收集考试卷子 | 准备"好答案 vs 差答案"的对比材料 |
| **DPO** | **看对比案例学习** | **直接从"好 vs 差"的对比中学会什么更好** |

---

## 第一章：从 RLHF 到 DPO —— 为什么需要偏好对齐？

### 1.1 SFT 之后还缺什么？

SFT 教会了模型"怎么回答问题"，但模型并不知道"什么样的回答更好"：

```
用户：  "推荐一部电影"

SFT 模型可能的回答 A（好）：
  "推荐《肖申克的救赎》，这是一部关于希望与自由的经典电影，
   豆瓣评分 9.7，讲述了银行家安迪在监狱中不放弃希望的故事。"

SFT 模型可能的回答 B（差）：
  "电影有很多种类型，比如动作片、喜剧片、科幻片、恐怖片、
   爱情片、纪录片、动画片..."
   ↑ 没有真正推荐，只是在罗列类型！
```

SFT 模型对这两种回答的生成概率可能差不多，因为它只学了"怎么回答"，没学"哪种回答更好"。

### 1.2 传统 RLHF 的做法

在 DPO 出现之前，业界用 RLHF（Reinforcement Learning from Human Feedback）来解决这个问题：

```
传统 RLHF 流程（三步走）：

┌────────────────────────��─────────────────────────────────────┐
│                                                                │
│  第一步：收集偏好数据                                          │
│  人类标注员对比两个回答，选出更好的那个                         │
│                                                                │
│  第二步：训练奖励模型（Reward Model）                          │
│  用偏好数据训练一个"裁判"，它能给任何回答打分                  │
│                                                                │
│  第三步：PPO 强化学习训练                                      │
│  用奖励模型的分数作为奖励信号，用 PPO 算法优化策略模型          │
│                                                                │
│  问题：流程复杂、训练不稳定、需要同时维护 4 个模型              │
│  （策略模型、参考模型、奖励模型、价值模型）                     │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 DPO 的革命性简化

2023 年，斯坦福大学提出了 DPO（Direct Preference Optimization），核心发现是：

> **不需要训练奖励模型！可以直接从偏好数据中学习。**

```
DPO 流程（一步到位）：

┌──────────────────────────────────────────────────────────────┐
│                                                                │
│  输入：SFT 模型 + 偏好数据（chosen vs rejected）              │
│                                                                │
│  训练：直接优化模型，让它更偏好 chosen、更远离 rejected         │
│                                                                │
│  输出：对齐后的模型                                            │
│                                                                │
│  优势：简单、稳定、只需要 2 个模型（策略模型 + 参考模型）      │
└──────────────────────────────────────────────────────────────┘
```

| 对比 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 需要奖励模型 | ✅ 需要单独训练 | ❌ 不需要 |
| 需要的模型数 | 4 个 | 2 个 |
| 训练稳定性 | 较差（RL 训练不稳定） | 较好（类似 SFT 的训练方式） |
| 实现复杂度 | 高 | 低 |
| 效果 | 好 | 相当甚至更好 |

---

## 第二章：DPO 的数学原理（直觉版）

### 2.1 核心直觉

DPO 的核心思想可以用一句话概括：

> **让模型生成 chosen 的概率变高，生成 rejected 的概率变低，但不要偏离原始模型太远。**

```
训练前：
  P(chosen)  = 0.3    P(rejected) = 0.25
  ↑ 模型对好回答和差回答的生成概率差不多

训练后：
  P(chosen)  = 0.6    P(rejected) = 0.05
  ↑ 模型明显更倾向于生成好回答
```

### 2.2 DPO 损失函数（直觉解释）

```
DPO Loss = -log(σ(β × (Δchosen - Δrejected)))

其中：
  Δchosen   = log P_model(chosen)  - log P_ref(chosen)
  Δrejected = log P_model(rejected) - log P_ref(rejected)

  P_model = 当前正在训练的模型
  P_ref   = 参考模型（训练开始时的模型副本，冻结不更新）
  β       = 温度参数（默认 0.1），控制偏离程度
  σ       = sigmoid 函数
```

**用大白话解释：**

```
Δchosen 的含义：
  "相比训练前，当前模型对 chosen 的偏好增加了多少？"
  如果 Δchosen > 0，说明模型更喜欢 chosen 了 ✅

Δrejected 的含义：
  "相比训练前，当前模型对 rejected 的偏好增加了多少？"
  如果 Δrejected < 0，说明模型更不喜欢 rejected 了 ✅

Δchosen - Δrejected 的含义：
  "模型对 chosen 的偏好增量 vs 对 rejected 的偏好增量"
  这个差值越大，说明模型越能区分好坏 ✅

β 的作用：
  控制模型可以偏离参考模型多远
  β 越大 → 允许偏离越多 → 训练更激进
  β 越小 → 限制偏离 → 训练更保守
```

### 2.3 为什么需要参考模型？

如果没有参考模型的约束，模型可能会"走极端"：

```
没有参考模型约束：
  模型可能把 chosen 的概率推到 99.99%
  但同时把其他所有正常回答的概率也压到接近 0
  → 模型变成只会说一种话的"复读机"

有参考模型约束：
  模型只能在参考模型的基础上做"微调"
  chosen 概率提高一些，rejected 概率降低一些
  但整体分布不会偏离太远
  → 模型保持多样性，同时学会了偏好
```

这就像给模型加了一根"橡皮筋"——可以往偏好方向拉，但拉太远就会被弹回来。

---

## 第三章：DPO 的数据格式 —— 偏好三元组

### 3.1 DPO 需要什么数据？

DPO 训练需要的数据非常简单，就是**偏好三元组**：

```
(prompt, chosen, rejected)

prompt:   用户的问题
chosen:   好的回答（人类偏好的）
rejected: 差的回答（人类不偏好的）
```

### 3.2 ChatML 格式转换

本脚本使用 Qwen 的 ChatML 格式。原始数据需要转换为模型能理解的格式：

```
转换前（原始数据）：
{
  "prompt": "如何学习编程？",
  "chosen": [
    {"role": "user", "content": "如何学习编程？"},
    {"role": "assistant", "content": "建议从 Python 入手..."}   ← 好回答
  ],
  "rejected": [
    {"role": "user", "content": "如何学习编程？"},
    {"role": "assistant", "content": "编程是一门技术..."}       ← 差回答
  ]
}

转换后（ChatML 格式）：
{
  "prompt":   "<|im_start|>user\n如何学习编程？<|im_end|>\n<|im_start|>assistant\n",
  "chosen":   "建议从 Python 入手...<|im_end|>",
  "rejected": "编程是一门技术...<|im_end|>"
}
```

**关键设计：**
- `prompt` 以 `assistant\n` 结尾，表示"等待模型续写"
- `chosen` 和 `rejected` 只包含 assistant 的回答内容
- DPOTrainer 会在内部自动拼接 `prompt + chosen` 和 `prompt + rejected`

### 3.3 数据预处理代码

```python
def preprocess_dataset(examples):
    prompt, chosen, rejected = [], [], []
    for i in range(len(examples["prompt"])):
        # 构造 prompt（用户消息 + assistant 开头）
        text = f"<|im_start|>user\n{examples['prompt'][i]}<|im_end|>\n<|im_start|>assistant\n"
        prompt.append(text)

        # 提取 chosen（好回答）
        assert examples["chosen"][i][1]["role"] == "assistant"
        text = f"{examples['chosen'][i][1]['content']}<|im_end|>"
        chosen.append(text)

        # 提取 rejected（差回答）
        assert examples["rejected"][i][1]["role"] == "assistant"
        text = f"{examples['rejected'][i][1]['content']}<|im_end|>"
        rejected.append(text)

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
```

---

## 第四章：参考模型（Reference Model）的作用

### 4.1 什么是参考模型？

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  训练开始时：                                                 │
│                                                               │
│  SFT 模型权重 ──→ 复制一份 ──→ 参考模型（ref_model）         │
│       │                              │                        │
│       ▼                              ▼                        │
│  策略模型（policy）            冻结，不更新                    │
│  会被 DPO 损失更新             始终保持 SFT 的状态             │
│                                                               │
│  训练过程中：                                                 │
│  策略模型不断更新 ←── DPO Loss ──→ 参考模型保持不变           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 参考模型的三个作用

**作用一：防止模型"忘记"**

```
没有参考模型：
  模型为了讨好偏好数据，可能忘记 SFT 学到的基本对话能力
  → "灾难性遗忘"

有参考模型：
  KL 散度惩罚确保模型不会偏离 SFT 状态太远
  → 保持基本能力的同时学习偏好
```

**作用二：提供"基线"**

```
DPO 不是简单地"提高 chosen 概率"
而是"相对于参考模型，提高 chosen 的概率优势"

这意味着：如果参考模型已经很喜欢某个 chosen
  → 当前模型不需要再大幅提高（已经够好了）
如果参考模型不太喜欢某个 chosen
  → 当前模型需要更多地提高（还有改进空间）
```

**作用三：稳定训练**

参考模型就像一个"锚点"，防止训练过程中模型参数剧烈波动。

### 4.3 DPOTrainer 如何处理参考模型？

```python
# 在 DPOTrainer 内部（你不需要手动做这些）：
# 1. 自动复制当前模型作为参考模型
ref_model = copy.deepcopy(model)
ref_model.eval()  # 设为评估模式，不计算梯度

# 2. 训练时同时用两个模型计算概率
policy_chosen_logps = model(prompt + chosen)      # 策略模型对 chosen 的概率
policy_rejected_logps = model(prompt + rejected)  # 策略模型对 rejected 的概率
ref_chosen_logps = ref_model(prompt + chosen)     # 参考模型对 chosen 的概率
ref_rejected_logps = ref_model(prompt + rejected) # 参考模型对 rejected 的概率

# 3. 计算 DPO 损失
loss = dpo_loss(policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps, beta=0.1)
```

---

## 第五章：DPOTrainer 与训练参数详解

### 5.1 DPOConfig 关键参数

```python
training_args = DPOConfig(
    # ---- 基础参数 ----
    output_dir=output_path,            # checkpoint 保存路径
    overwrite_output_dir=True,         # 覆盖已有输出

    # ---- 学习率 ----
    learning_rate=5e-7,                # DPO 的学习率通常很小
    warmup_ratio=0.1,                  # 前 10% 步数预热
    lr_scheduler_type="cosine",        # 余弦退火调度器

    # ---- 训练规模 ----
    num_train_epochs=5,                # 训练轮数
    per_device_train_batch_size=2,     # 每 GPU batch size
    gradient_accumulation_steps=16,    # 梯度累积步数

    # ---- 精度和日志 ----
    bf16=True,                         # BF16 混合精度
    logging_steps=10,                  # 日志频率
    save_steps=500,                    # 保存频率
)
```

**为什么 DPO 的学习率这么小（5e-7）？**

```
SFT 的学习率：  通常 1e-5 ~ 5e-5
DPO 的学习率：  通常 5e-7 ~ 5e-6
                ↑ 比 SFT 小 10-100 倍！

原因：
  SFT 是在教模型"新技能"（从不会对话到会对话），需要较大的更新
  DPO 是在"微调偏好"（模型已经会对话，只是调整方向），只需要轻微的更新
  学习率太大 → 模型偏离参考模型太远 → 灾难性遗忘
```

### 5.2 DPOTrainer 关键参数

```python
trainer = DPOTrainer(
    model=model,                # 要训练的策略模型
    train_dataset=train_dataset,# 偏好数据集
    args=training_args,         # 训练配置
    tokenizer=tokenizer,        # 分词器
    dataset_num_proc=16,        # 数据处理并行数
    max_length=1024,            # prompt + response 最大总长度
    max_prompt_length=512,      # prompt 最大长度
)
```

**max_length 和 max_prompt_length 的关系：**

```
|←─────�� max_length = 1024 ──────────────────→|
|←── max_prompt_length = 512 ──→|←── 512 ──→|
|         prompt 部分             | response 部分|

如果 prompt 超过 512 token → 从左侧截断
如果 prompt + response 超过 1024 token → response 从右侧截断
```

### 5.3 DPO 特有的隐含参数

DPOTrainer 还有一些重要的默认参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `beta` | 0.1 | KL 散度惩罚系数，控制偏离参考模型的程度 |
| `loss_type` | "sigmoid" | 损失函数类型（原始 DPO 论文的公式） |
| `label_smoothing` | 0.0 | 标签平滑，防止过拟合 |

**beta 参数的影响：**

```
beta = 0.01  →  几乎不允许偏离参考模型（太保守，学不到东西）
beta = 0.1   →  适度偏离（推荐值，平衡学习和稳定性）
beta = 0.5   →  允许较大偏离（激进，可能不稳定）
beta = 1.0   →  大幅偏离（很激进，容易崩溃）
```

---

## 第六章：代码逐行对照解读

### 6.1 整体代码结构

```python
# ┌─────────────────────────────────────────────────┐
# │ 第一部分：导入依赖                                │
# │ 第二部分：路径和参数配置                          │
# │ 第三部分：加载模型和分词器                        │
# │ 第四部分：加载偏好数据集                          │
# │ 第五部分：数据预处理（ChatML 格式转换）           │
# │ 第六部分：DPO 训练参数配置                        │
# │ 第七部分：初始化 wandb                            │
# │ 第八部分：创建 DPOTrainer 并训练                  │
# │ 第九部分：保存模型                                │
# └─────────────────────────────────────────────────┘
```

### 6.2 模型加载

```python
# 加载 SFT checkpoint（不是预训练模型！）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,              # 半精度节省显存
    attn_implementation="flash_attention_2"   # FlashAttention 加速
)
```

**为什么从 SFT checkpoint 开始？**

```
预训练模型 → 只会续写，不会对话 → 无法理解偏好数据中的对话
SFT 模型   → 已经会对话 → 可以在此基础上学习"什么回答更好"

所以 DPO 的起点必须是 SFT 模型，而不是预训练模型。
```

### 6.3 训练过程可视化

如果启用了 wandb，你可以在训练过程中观察以下指标：

```
wandb 仪表板上的关键指标：

📈 train/loss
   DPO 损失值，应该逐渐下降
   如果不下降 → 学习率可能太小
   如果剧烈波动 → 学习率可能太大

📈 train/rewards/chosen
   模型对 chosen 回答的隐式奖励，应该逐渐上升

📈 train/rewards/rejected
   模型对 rejected 回答的隐式奖励，应该逐渐下降

📈 train/rewards/margins
   chosen 和 rejected 的奖励差距，应该逐渐增大
   这是最重要的指标——差距越大，说明模型越能区分好坏
```

---

## 第七章：DPO vs RLHF(PPO) 的对比

### 7.1 流程对比

```
RLHF (PPO) 流程：
  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
  │ 偏好  │ →  │ 训练  │ →  │ PPO  │ →  │ 对齐  │
  │ 数据  │    │  RM   │    │ 训练 │    │ 模型  │
  └──────┘    └──────┘    └──────┘    └──────┘
                 ↑ 需要额外训练一个奖励模型

DPO 流程：
  ┌──────┐    ┌──────┐    ┌──────┐
  │ 偏好  │ →  │ DPO  │ →  │ 对齐  │
  │ 数据  │    │ 训练 │    │ 模型  │
  └──────┘    └──────┘    └──────┘
                 ↑ 直接从偏好数据学习，跳过 RM
```

### 7.2 详细对比

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 训练步骤 | 3 步（数据→RM→PPO） | 1 步（数据→DPO） |
| 需要的模型 | 4 个（策略、参考、奖励、价值） | 2 个（策略、参考） |
| 显存需求 | 很高（4 个模型同时在 GPU） | 较低（2 个模型） |
| 训练稳定性 | 较差（RL 训练天然不稳定） | 较好（本质是监督学习） |
| 超参数敏感度 | 高（PPO 有很多超参数要调） | 低（主要就一个 β） |
| 代码复杂度 | 高 | 低（几十行代码） |
| 理论等价性 | 原始方法 | 数学上等价于 RLHF |
| 在线学习 | ✅ 可以在线生成新数据 | ❌ 只能用离线数据 |
| 适用场景 | 需要持续优化的场景 | 有现成偏好数据的场景 |

### 7.3 什么时候选 DPO？什么时候选 PPO？

```
选 DPO 的情况：
  ✅ 已经有偏好数据（或能合成）
  ✅ 显存有限
  ✅ 想要简单稳定的训练
  ✅ 第一次做偏好对齐（推荐从 DPO 开始）

选 PPO 的情况：
  ✅ 需要在线生成和评估（边训练边生成新数据）
  ✅ 有足够的计算资源
  ✅ 需要更精细的奖励信号（不只是二元偏好）
  ✅ 已经有训练好的奖励模型
```

---

## 附录：常见问题

### Q1：DPO 训练需要多少偏好数据？

| 数据量 | 效果 | 建议 |
|--------|------|------|
| < 1000 条 | 效果有限 | 仅用于验证流程 |
| 1000-5000 条 | 有明显效果 | 小规模实验 |
| 5000-50000 条 | 效果好 | 正式训练推荐 |
| > 50000 条 | 收益递减 | 注意数据质量比数量重要 |

### Q2：训练多少个 epoch 合适？

DPO 通常训练 1-5 个 epoch。本脚本设置了 5 个 epoch。

```
epoch 太少 → 模型没学到足够的偏好
epoch 太多 → 过拟合偏好数据，泛化能力下降

建议：观察 wandb 上的 rewards/margins 指标
  如果还在上升 → 可以继续训练
  如果开始下降或震荡 → 该停了
```

### Q3：DPO 训练后模型变差了怎么办？

常见原因和解决方案：

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 回答变短/变空 | 学习率太大 | 降低 learning_rate |
| 回答变得重复 | 过拟合 | 减少 epoch 或增加数据 |
| 回答质量下降 | 偏好数据质量差 | 检查数据，确保 chosen 确实比 rejected 好 |
| 模型"忘记"了 SFT 能力 | β 太大 | 减小 beta 值 |

### Q4：DPO 训练需要多少显存？

```
DPO 需要同时加载 2 个模型（策略模型 + 参考模型）

Qwen2.5-0.5B (BF16):
  策略模型：~1 GB
  参考模型：~1 GB
  梯度和优化器：~2 GB
  总计：~4-6 GB → 单张消费级 GPU 即可

Qwen2.5-7B (BF16):
  总计：~40-50 GB → 需要 A100 或多卡
```

### Q5：`gradient_accumulation_steps=16` 是什么意思？

```
per_device_train_batch_size = 2   → 每步处理 2 条数据
gradient_accumulation_steps = 16  → 累积 16 步的梯度再更新

等效 batch size = 2 × 16 = 32

为什么要梯度累积？
  显存不够放大 batch → 用小 batch 多累积几步 → 效果等价于大 batch
  这是"用时间换空间"的经典技巧
```

### Q6：DPO 和上一步的偏好数据合成是什么关系？

```
偏好数据合成（qwen_dpo_data.py）：
  输入：问题集
  输出：(question, response_j, response_k, rating_j, rating_k)
  作用：制造偏好数据

DPO 训练（qwen_dpo.py）：
  输入：偏好数据 (prompt, chosen, rejected)
  输出：对齐后的模型
  作用：用偏好数据训练模型

两者的关系：数据合成是"做菜准备食材"，DPO 训练是"用食材做菜"
```

---

> 📌 **下一步学习：** DPO 训练完成后，请继续阅读《训练奖励模型 RM 完全指南》，学习 RLHF 的另一条路线——先训练奖励模型，再用 PPO 优化。
