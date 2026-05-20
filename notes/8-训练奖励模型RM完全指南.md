# ⚖️ 训练奖励模型（Reward Model）完全指南 —— 从零理解到动手实践

> 本文档配合 `qwen_rm.py` 和 `utils/rm_utils.py` 代码，手把手带你理解奖励模型训练的每一个环节。
> 假设你已经完成了偏好数据合成和 DPO 的学习，我们在此基础上学习 RLHF 的另一条路线。

---

## 📋 目录

- [第零章：奖励模型在大模型训练中的位置](#第零章奖励模型在大模型训练中的位置)
- [第一章：什么是奖励模型？为什么需要它？](#第一章什么是奖励模型为什么需要它)
- [第二章：奖励模型的架构 —— 从生成到打分](#第二章奖励模型的架构--从生成到打分)
- [第三章：Pairwise Ranking Loss —— 配对排序损失](#第三章pairwise-ranking-loss--配对排序损失)
- [第四章：LoRA 高效微调奖励模型](#第四章lora-高效微调奖励模型)
- [第五章：数据处理与 DataCollator](#第五章数据处理与-datacollator)
- [第六章：代码逐行对照解读](#第六章代码逐行对照解读)
- [第七章：奖励模型的评估与调优](#第七章奖励模型的评估与调优)
- [附录：常见问题](#附录常见问题)

---

## 第零章：奖励模型在大模型训练中的位置

```
┌─────────────────────────────────────────────────────────────────────┐
│                     大模型训练完整流程                                │
│                                                                     │
│   ① 预训练（Pre-Training）                                          │
│   ② 指令微调（SFT）                                                 │
│   ③ 偏好对齐                                                        │
│   │                                                                 │
│   │  路线 A（简单）：DPO 直接偏好优化 ✅ 已完成                     │
│   │  │  偏好数据 → 直接训练模型                                     │
│   │  │                                                               │
│   │  路线 B（经典 RLHF）：                                          │
│   │  │  ③-c 训练奖励模型（RM）  ← 📍 我们现在在这里！              │
│   │  │  │   训练一个"裁判"来给回答打分                              │
│   │  │  │                                                           │
│   │  │  ③-d PPO 近端策略优化                                       │
│   │  │      用奖励模型指导策略模型优化                               │
│   │                                                                 │
│   ④ 评估与部署                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**打个比方：**

| 阶段 | 类比 | 做什么 |
|------|------|--------|
| 预训练 | 上学读书 | 积累知识 |
| SFT | 岗前培训 | 学会按要求完成任务 |
| DPO | 看对比案例学习 | 直接从好坏对比中学习 |
| **训练 RM** | **培训一个考官** | **教会一个"裁判"如何给回答打分** |
| PPO | 在考官指导下练习 | 根据考官的打分不断改进 |

---

## 第一章：什么是奖励模型？为什么需要它？

### 1.1 两条偏好对齐路线

在上一篇中我们学了 DPO——直接从偏好数据训练模型。但还有一条更经典的路线：

```
路线 A：DPO（一步到位）
  偏好数据 ──→ 直接训练策略模型
  优点：简单
  缺点：只能用离线数据

路线 B：RLHF = RM + PPO（两步走）
  偏好数据 ──→ 训练奖励模型（RM）──→ RM 指导 PPO 训练策略模型
  优点：RM 可以给任何新回答打分（在线学习）
  缺点：流程复杂
```

### 1.2 奖励模型是什么？

奖励模型（Reward Model, RM）本质上是一个**打分器**：

```
输入：一个问题 + 一个回答
输出：一个分数（标量），表示这个回答有多好

例如：
  输入："如何学编程？" + "建议从 Python 入手，先学基础语法..."
  输出：4.2 分

  输入："如何学编程？" + "编程是一个复杂的话题，涉及很多方面..."
  输出：1.8 分
```

### 1.3 奖励模型 vs 裁判模型

你可能会问：上一步偏好数据合成中不是已经有"裁判模型"（DeepSeek）了吗？

| 对比 | 偏好数据合成中的裁判 | 奖励模型 |
|------|---------------------|---------|
| 是什么 | 通用大模型（如 DeepSeek） | 专门训练的打分模型 |
| 大小 | 很大（几十B 参数） | 可以很小（0.5B） |
| 速度 | 慢（需要生成文本） | 快（只输出一个数字） |
| 成本 | 高（API 调用费用） | 低（本地推理） |
| 用途 | 离线打分（造数据时用） | 在线打分（PPO 训练时实时用） |

奖励模型就像是把大裁判的"评分能力"蒸馏到一个小模型中，让它能快速、低成本地给回答打分。

---

## 第二章：奖励模型的架构 —— 从生成到打分

### 2.1 普通语言模型 vs 奖励模型

```
普通语言模型（AutoModelForCausalLM）：
  输入：[token1, token2, ..., tokenN]
  输出：[vocab_size] 的概率分布（预测下一个 token）
  用途：生成文本

奖励模型（AutoModelForSequenceClassification）：
  输入：[token1, token2, ..., tokenN]
  输出：一个标量分数
  用途：给文本打分
```

### 2.2 架构改造

奖励模型的改造非常简单——只是把最后的"语言模型头"换成了"分类头"：

```
┌─────────────────────────────────────────────────────────┐
│                                                           │
│  普通语言模型：                                           │
│  输入 → [Transformer 层 × N] → LM Head → vocab_size      │
│                                  (hidden → 151936)        │
│                                                           │
│  奖励模型：                                               │
│  输入 → [Transformer 层 × N] → Score Head → 1             │
│                                  (hidden → 1)             │
│                                  ↑ 只改了这里！            │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

```python
# 代码中的实现：
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=1,              # 输出 1 个分数（不是多分类）
    torch_dtype=torch.bfloat16
)
```

`num_labels=1` 是关键——它告诉模型"我只需要输出一个数字"。

### 2.3 分数是怎么算出来的？

```
输入文本："Question: 如何学编程？\n\nAnswer: 建议从 Python 入手..."
    │
    ▼
Tokenizer 分词
    │
    ▼
[token_1, token_2, ..., token_N]
    │
    ▼
Transformer 编码（24 层 self-attention）
    │
    ▼
[hidden_1, hidden_2, ..., hidden_N]  ← 每个 token 的隐藏状态
    │
    ▼
取最后一个 token 的隐藏状态 hidden_N  ← 包含了整个序列的信息
    │
    ▼
线性层：hidden_N (896维) → score (1维)
    │
    ▼
输出：3.7（这个回答的奖励分数）
```

### 2.4 最后一层（score 头）属于哪部分参数？

**结论先行：score 头是新增的、随机初始化的参数，不是预训练模型自带的。**

很多初学者会困惑：奖励模型是从预训练（或 SFT）模型继承来的，那把"映射到词表的 lm_head"换成"映射到一个分数的 score 头"，这一层到底是从哪里来的？

#### 参数来源拆解

| 模块 | 来源 | 训练时是否更新 |
|------|------|----------------|
| Embedding 层 | 预训练权重（继承） | ✅ 继续微调 |
| 全部 Transformer Block（24 层） | 预训练权重（继承） | ✅ 继续微调 |
| `lm_head`（H → V，词表头） | **被丢弃**（不再使用） | — |
| `score`（H → 1，打分头） | **新增，随机初始化** | ✅ 从零学起 |

其中 H = hidden_size（Qwen2.5-0.5B 是 896），V = vocab_size（约 151936）。

#### 结构对比图

```
预训练 / SFT 模型（CausalLM）：
    hidden_states (B, L, H=896)
            │
            ▼
       lm_head: Linear(896 → 151936)   ← 预训练参数
            │
            ▼
        logits (B, L, V)                ← 用于预测下一个 token


奖励模型（SequenceClassification, num_labels=1）：
    hidden_states (B, L, H=896)
            │
            ▼
       score:   Linear(896 → 1)         ← 🆕 新增、随机初始化
            │
            ▼
        reward (B, L, 1) → 取最后一个非 pad token → 标量分数
```

#### 加载时实际发生了什么？

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B", num_labels=1
)
```

执行这行代码时，HuggingFace 会：

1. **加载 backbone**：embedding + 所有 Transformer Block 的预训练权重 ✅
2. **丢弃 lm_head**：因为分类任务用不到词表映射
3. **新建 score 头**：因为 checkpoint 里没有 `score.weight`，会随机初始化
4. **打印 warning**：
   ```
   Some weights of Qwen2ForSequenceClassification were not initialized
   from the model checkpoint at ... and are newly initialized: ['score.weight']
   You should probably TRAIN this model on a downstream task...
   ```
   看到这个 warning **不要慌**，这正是预期行为——它在告诉你"score 头是新的，需要训练"。

#### 为什么 H→1 就够了？不会信息不足吗？

你可能会想：词表有 15 万维，现在只输出 1 个数，是不是太"稀薄"了？  
其实正相反——**hidden_states 本身已经编码了完整的语义信息**，`lm_head` 的作用只是把它"解码"成下一个 token 的概率分布。RM 不需要预测 token，只需要从 hidden_states 中"读出"一个偏好分数，所以 `Linear(H, 1)` 就够了，**关键的语言理解能力都在 backbone 里**。

#### 训练时两部分如何协同？

- **Backbone**：在已有的语言理解能力基础上**继续微调**，学会"什么样的回答是好回答"。
- **Score 头**：从零学起，学习"如何把 hidden state 压缩成一个偏好分数"。
- 两者**一起反向传播、一起更新**，没有冻结。
- 实践中 score 头收敛很快（参数量小），主要训练时间花在调整 backbone 上。

#### 一句话记忆

> **RM = 预训练 backbone（继承参数，继续微调） + 新的 score 头（H→1，随机初始化，从零学起）。**

---

## 第三章：Pairwise Ranking Loss —— 配对排序损失

### 3.1 为什么不用普通的回归损失？

你可能会想：既然奖励模型输出一个分数，为什么不直接用 MSE 损失训练？

```
方案 A：回归损失（MSE）
  训练数据：(回答, 绝对分数)  如 ("好回答", 4.5)
  问题：绝对分数很难标注！
    "这个回答到底是 3.5 分还是 4.0 分？" → 标注员也说不清

方案 B：配对排序损失（Pairwise Ranking Loss）✅ 本脚本使用
  训练数据：(好回答, 差回答)  只需要知道谁更好
  优点：相对比较比绝对打分容易得多！
    "这两个回答哪个更好？" → 标注员很容易判断
```

### 3.2 损失函数详解

```python
loss = -log(sigmoid(reward_j - reward_k))
```

这个公式来自 InstructGPT 论文，我们来拆解它：

```
reward_j：好回答的分数
reward_k：差回答的分数

reward_j - reward_k：分数差

sigmoid(分数差)：
  分数差 > 0 → sigmoid 接近 1（模型判断正确 ✅）
  分数差 < 0 → sigmoid 接近 0（模型判断错误 ❌）
  分数差 = 0 → sigmoid = 0.5（模型无法区分）

-log(sigmoid(分数差))：
  sigmoid 接近 1 → -log(1) ≈ 0    → loss 小 ✅
  sigmoid 接近 0 → -log(0) → +∞   → loss 大 ❌
```

**用图来理解：**

```
loss
  ↑
  │  ╲
  │   ╲
  │    ╲
  │     ╲
  │      ╲___________
  ���
  └──────────────────→ reward_j - reward_k
       ←差回答分更高    好回答分更高→
       （loss 大）      （loss 小）
```

### 3.3 代码实现

```python
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 分别计算好回答和差回答的分数
        rewards_j = model(input_ids=inputs["input_ids_j"],
                         attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"],
                         attention_mask=inputs["attention_mask_k"])[0]

        # 配对排序损失：让 rewards_j > rewards_k
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        return loss
```

---

## 第四章：LoRA 高效微调奖励模型

### 4.1 为什么用 LoRA？

奖励模型的任务（打分）比生成文本简单得多，不需要更新所有参数：

```
全量微调：
  更新 Qwen2.5-0.5B 的全部 ~500M 参数
  显存需求大，训练慢
  容易过拟合（参数太多，数据相对少）

LoRA 微调：
  冻结原始 ~500M 参数
  只训练新增的 ~1M 参数（约 0.2%）
  显存需求小，训练快
  天然的正则化效果
```

### 4.2 LoRA 参数解读

```python
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   # 序列分类任务
    inference_mode=False,          # 训练模式
    r=8,                           # LoRA 秩
    lora_alpha=32,                 # 缩放系数
    lora_dropout=0.1,              # dropout
)
```

**LoRA 秩（r）的直觉理解：**

```
原始权重矩阵 W：896 × 896 = 802,816 个参数

LoRA 分解：W + ΔW = W + A × B
  A：896 × 8 = 7,168 个参数
  B：8 × 896 = 7,168 个参数
  总计：14,336 个参数（只有原来的 1.8%）

r 越大 → 可训练参数越多 → 表达能力越强 → 但也越容易过拟合
r=8 是一个经验上的好选择
```

**缩放系数（lora_alpha）的作用：**

```
实际更新 = (lora_alpha / r) × ΔW = (32 / 8) × ΔW = 4 × ΔW

lora_alpha/r 越大 → LoRA 更新的影响越大
通常 lora_alpha = 2~4 倍的 r
```

### 4.3 LoRA 是怎么"插"进 RM 的？—— 结构层面的整合

前面我们知道：**RM = 预训练 backbone + 新增 score 头**。  
现在再加上 LoRA，整个模型可以拆成三部分来理解：

```
┌──────────────────────────────────────────────────────┐
│  Embedding                              ❄️ 冻结      │ ← 继承预训练
├──────────────────────────────────────────────────────┤
│  Transformer Block × 24                              │
│  ┌────────────────────────────────────────────────┐ │
│  │  q_proj (H × H)         ❄️ 冻结                │ │
│  │    └─ 旁路: A(r×H) → B(H×r)   ✅ 训练（LoRA）  │ │
│  │  v_proj (H × H)         ❄️ 冻结                │ │
│  │    └─ 旁路: A(r×H) → B(H×r)   ✅ 训练（LoRA）  │ │
│  │  k_proj, o_proj, MLP    ❄️ 冻结（默认不加）   │ │
│  └────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────┤
│  score (H → 1)                          ✅ 全量训练 │ ← 必须 modules_to_save
└─────────────────────��────────────────────────────────┘
```

**前向计算的变化**：每个被选中的 Linear 层 `W (H×H)` 从

$$y = Wx \quad\longrightarrow\quad y = Wx + \tfrac{\alpha}{r} \cdot BAx$$

原始 W 不动，只训练 A、B 两个低秩矩阵。

### 4.4 三个关键问题

#### Q1：LoRA 应用在哪些层？

由 `LoraConfig` 中的 `target_modules` 决定。常见做法：

```python
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,           # ⭐ 必须是序列分类
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 只在 Q、V 上加 LoRA
    modules_to_save=["score"],            # ⭐ 关键：score 头全量训练
)
```

#### Q2：score 头怎么办？—— RM + LoRA 最易踩的坑

**score 头是新增的、随机初始化的，不能只用 LoRA 适配它。**

LoRA 的本质是"在已有权重旁边并联微调"，但 score 头**根本没有"已有权重"**——它是从零开始的随机矩阵。如果不做特殊处理，PEFT 默认会把它一起冻结，结果就是 score 头永远停留在随机初始值，训练完全失败。

解决方案就是 `modules_to_save=["score"]`：告诉 PEFT "这个模块跳过 LoRA 包装、直接全量训练并保存"。

|  | 普通层（q_proj 等） | score 头 |
|---|---|---|
| **原始权重** | ❄️ 冻结 | 🆕 随机初始化 |
| **训练时更新** | LoRA 适配器 (A, B) | 全部参数 |
| **保存到 checkpoint** | A, B 矩阵（几 MB） | 完整 `score.weight` |

#### Q3：为什么 RM 特别适合用 LoRA？

| 角度 | 说明 |
|------|------|
| **数据量小** | 偏好数据通常几千~几万对，全参微调容易过拟合，LoRA 的低秩约束反而是天然正则 |
| **任务窄** | RM 只学"打分"这一件事，不需要改动 backbone 的语言能力 |
| **显存紧** | RM 训练每步要前向两条样本（chosen + rejected），显存翻倍，LoRA 能省一大块 |
| **学习率匹配** | 这就解释了 Q2：RM 用 2e-5（LoRA 需要更大学习率），DPO 全参微调用 5e-7 |

### 4.5 训练产物与推理加载

```
训练完成后的 checkpoint 目录：
  ├── adapter_model.safetensors   ← LoRA 的 A, B 矩阵（几 MB，超小）
  ├── adapter_config.json         ← LoRA 配置（r、alpha、target_modules 等）
  └── score.safetensors           ← score 头全量权重（modules_to_save 的功劳）
```

推理时的加载流程：

```python
from transformers import AutoModelForSequenceClassification
from peft import PeftModel

# 1. 加载原始 base 模型（带随机 score 头）
base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen2.5-0.5B", num_labels=1
)

# 2. 套上 LoRA 适配器 + 训练好的 score 头
rm_model = PeftModel.from_pretrained(base_model, "你的LoRA输出路径")

# 此时三者就位：
#   ✅ base 权重（原始预训练）
#   ✅ LoRA 适配器（训练得到的 A、B）
#   ✅ score 头（训练得到的全量权重）
```

### 4.6 一句话总结

> **RM + LoRA = 冻结 backbone 大部分参数 → 在注意力的 q_proj/v_proj 旁加 LoRA 适配器（学打分语义） → score 头全量训练（学输出标量）。**
>
> 三者协同：**backbone** 提供语言理解，**LoRA** 微调出"偏好敏感性"，**score 头** ��这种敏感性压成一个分数。

---

## 第五章：数据处理与 DataCollator

### 5.1 数据预处理流程

```
原始数据：
  question: "如何学编程？"
  response_j: "建议从 Python 入手..."    ← 好回答
  response_k: "编程是一个复杂的话题..."   ← 差回答

        │ preprocess_rm_dataset()
        ▼

拼接为完整文本：
  text_j: "Question: 如何学编程？\n\nAnswer: 建议从 Python 入手..."
  text_k: "Question: 如何学编程？\n\nAnswer: 编程是一个复杂的话题..."

        │ tokenizer()
        ▼

Token 化：
  input_ids_j:      [2182, 345, ...]
  attention_mask_j:  [1, 1, ...]
  input_ids_k:      [2182, 345, ...]
  attention_mask_k:  [1, 1, ...]

        │ filter(len <= MAX_LENGTH)
        ▼

过滤超长样本（保证不超过 512 token）
```

### 5.2 RewardDataCollatorWithPadding

普通的 DataCollator 只需要处理一组 input_ids，但奖励模型有两组（j 和 k）：

```
普通 DataCollator：
  样本1: [101, 202, 303]
  样本2: [101, 202, 303, 404, 505]
  padding 后：
  样本1: [101, 202, 303, 0, 0]      ← 补 0 到最长
  样本2: [101, 202, 303, 404, 505]

RewardDataCollatorWithPadding：
  需要分别对 j 和 k 做 padding，然后合并：
  batch = {
      "input_ids_j":      [...],   ← j 组 padding 后的结果
      "attention_mask_j":  [...],
      "input_ids_k":      [...],   ← k 组 padding 后的结果
      "attention_mask_k":  [...],
      "return_loss": True
  }
```

---

## 第六章：代码逐行对照解读

### 6.1 整体代码结构

```python
# ┌─────────────────────────────────────────────────┐
# │ qwen_rm.py                                       │
# │                                                   │
# │ 第一部分：导入依赖                                │
# │ 第二部分：超参数和路径配置                        │
# │ 第三部分：加载偏好数据集                          │
# │ 第四部分：加载模型（序列分类 + LoRA）             │
# │ 第五部分：数据预处理                              │
# │ 第六部分：定义评估指标                            │
# │ 第七部分：训练参数配置                            │
# │ 第八部分：训练                                    │
# │ 第九部分：保存模型                                │
# ├─────────────────────────────────────────────────┤
# │ utils/rm_utils.py                                 │
# │                                                   │
# │ RewardDataCollatorWithPadding：偏好对 padding     │
# │ RewardTrainer：pairwise ranking loss 训练器       │
# └─────────────────────────────────────────────────┘
```

### 6.2 关键代码解读

**模型加载——为什么用预训练模型而不是 SFT 模型？**

```python
# DPO 用的是 SFT 模型（需要对话能力）
# RM 用的是预训练基座模型（只需要理解能力，不需要生成能力）
MODEL_PATH = ".../Qwen2.5-0.5B"  # 预训练模型，不是 SFT checkpoint

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=1  # 改造为打分模型
)
```

**评估指标——准确率的含义：**

```python
def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # predictions shape: (2, batch_size)
    # predictions[0] = rewards_j（好回答的分数）
    # predictions[1] = rewards_k（差回答的分数）

    predictions = np.argmax(predictions, axis=0)
    # 如果 rewards_j > rewards_k → argmax = 0 → 正确
    # 如果 rewards_k > rewards_j → argmax = 1 → 错误

    labels = np.zeros(predictions.shape)  # 正确答案全是 0
    return accuracy.compute(predictions=predictions, references=labels)
```

**训练参数中的关键设置：**

```python
training_args = TrainingArguments(
    remove_unused_columns=False,  # 必须！否则自定义列会被删除
    label_names=[],               # 必须！奖励模型没有传统 label
    gradient_checkpointing=True,  # 省显存（用时间换空间）
)
```

---

## 第七章：奖励模型的评估与调优

### 7.1 关键指标

训练过程中需要关注的指标：

```
📈 eval/accuracy
   奖励模型在验证集上的准确率
   含义：模型正确判断"好回答 > 差回答"的比例
   目标：> 70%（随机猜测是 50%）

📈 train/loss
   训练损失，应该逐渐下降
   如果不下降 → 学习率可能不合适
   如果下降后又上升 → 过拟合了
```

### 7.2 好的奖励模型是什么样的？

| 指标 | 差 | 一般 | 好 | 很好 |
|------|-----|------|-----|------|
| 准确率 | < 60% | 60-70% | 70-80% | > 80% |

### 7.3 常见调优方向

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 准确率低 | 数据质量差 | 检查偏好数据，确保 j 确实比 k 好 |
| 准确率低 | 模型太小 | 换更大的基座模型 |
| 过拟合 | 数据太少 | 增加数据量或增大 LoRA dropout |
| 训练不稳定 | 学习率太大 | 降低学习率 |

---

## 附录：常见问题

### Q1：奖励模型和 DPO 中的"隐式奖励"有什么关系？

```
DPO 的一个重要发现：
  DPO 训练后的模型隐含了一个奖励函数！
  reward(x, y) = β × log(π_model(y|x) / π_ref(y|x))

  也就是说，DPO 模型本身就可以当奖励模型用
  但它的打分需要同时运行策略模型和参考模型，效率较低

独立的奖励模型：
  一个专门用来打分的模型，直接输出分数
  效率更高，可以在 PPO 训练中实时使用
```

> **澄清："小模型"是相对说法，不是绝对小**
>
> 前文提到"把大裁判的能力蒸馏到一个小模型"以及这里说的"专门的小模型"，容易让人误以为 RM 是一个参数量很小的独立网络。实际上：
>
> - **结构上**：RM = 完整的预训练 backbone（embedding + 全部 Transformer Block）+ 一个 `Linear(H, 1)` 的 score 头。**它和原始 LLM 的参数量几乎一样**（只少了 lm_head，多了一个非常小的 score 头）。
> - **"小"是相对谁而言**：
>   - 相对于 **作为大裁判的通用大模型**（如 DeepSeek-V3、GPT-4，几百 B 参数），我们用 0.5B / 7B 做 RM 确实小很多 → 这是"小模型"的本意。
>   - 相对于 **被它评分的策略模型**，RM 通常**同量级甚至更小**（实践中常用同尺寸或更小的 backbone 来做 RM，以节省 PPO 阶段的显存）。
> - **"专门"才是关键词**：RM 的特点不是参数少，而是**任务专一**——只输出一个标量分数，不做生成，因此推理一次只需一次前向、不需要自回归解码，速度比让大模型生成评语快得多。
>
> 一句话：**RM 的"小"和"快"主要来自"只前向一次输出一个数"，而不是参数量本身有多小。**

### Q2：为什么 RM 的学习率（2e-5）比 DPO（5e-7）大这么多？

```
DPO：在 SFT 模型基础上微调，模型已经很好了，只需要轻微调整
     → 学习率很小（5e-7）

RM：从预训练模型开始，要学一个全新的任务（打分）
    而且用了 LoRA（只更新少量参数，需要更大的学习率来补偿）
    → 学习率较大（2e-5）
```

### Q3：`remove_unused_columns=False` 为什么必须设置？

```
HuggingFace Trainer 默认会自动删除模型 forward() 不需要的列
但我们的数据有自定义列名（input_ids_j, input_ids_k 等）
Trainer 不认识这些列，会把它们删掉
→ RewardTrainer.compute_loss() 就拿不到数据了

设置 remove_unused_columns=False 告诉 Trainer："别删任何列！"
```

### Q4：`label_names=[]` 为什么要设为空列表？

```
普通分类任务：数据中有 "labels" 列，Trainer 会自动找到它
奖励模型：没有传统的 labels（我们用 pairwise loss，不需要标签）

如果不设置 label_names=[]，Trainer 会报错说找不到 labels
设为空列表告诉 Trainer："这个任务没有标签，别找了"
```

### Q5：训练好的奖励模型怎么用？

```python
# 加载奖励模型
from peft import PeftModel

base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen2.5-0.5B", num_labels=1
)
rm_model = PeftModel.from_pretrained(base_model, "results/rm")

# 给一个回答打分
text = "Question: 如何学编程？\n\nAnswer: 建议从 Python 入手..."
inputs = tokenizer(text, return_tensors="pt")
score = rm_model(**inputs).logits.item()
print(f"奖励分数: {score}")  # 如：3.7
```

下一步在 PPO 训练中，这个奖励模型会被用来实时给策略模型的回答打分。

### Q6：gradient_checkpointing 是什么？为什么能省显存？

```
正常训练：
  前向传播时保存所有中间激活值（用于反向传播计算梯度）
  显存占用 = 模型参数 + 所有中间激活值
  ↑ 中间激活值可能比模型参数还大！

梯度检查点：
  前向传播时只保存少数"检查点"的激活值
  反向传播时，需要某层的激活值时重新计算
  显存占用 = 模型参数 + 少量检查点激活值
  ↑ 显存大幅减少，但训练速度慢约 20%（因为要重新计算）

这是一个经典的"时间换空间"策略
```

---

> 📌 **下一步学习：** 奖励模型训练完成后，请继续阅读《PPO 近端策略优化完全指南》，学习如何用奖励模型指导策略模型的优化。
