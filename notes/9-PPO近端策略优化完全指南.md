# 🎮 PPO 近端策略优化完全指南 —— 从零理解到动手实践

> 本文档配合 `qwen_ppo.py` 代码，手把手带你理解 PPO 训练的每一个环节。
> 假设你已经完成了偏好数据合成、DPO、奖励模型的学习，这是强化学习部分的最后一篇。

---

## 📋 目录

- [第零章：PPO 在大模型训练中的位置](#第零章ppo-在大模型训练中的位置)
- [第一章：什么是 PPO？强化学习基础概念](#第一章什么是-ppo强化学习基础概念)
- [第二章：RLHF 中的 PPO —— 四个模型的协作](#第二章rlhf-中的-ppo--四个模型的协作)
- [第三章：PPO 算法核心原理](#第三章ppo-算法核心原理)
- [第四章：Value Head —— 价值估计](#第四章value-head--价值估计)
- [第五章：PPO 训练参数详解](#第五章ppo-训练参数详解)
- [第六章：代码逐行对照解读](#第六章代码逐行对照解读)
- [第七章：四种偏好对齐方法总结对比](#第七章四种偏好对齐方法总结对比)
- [附录：常见问题](#附录常见问题)

---

## 第零章：PPO 在大模型训练中的位置

```
┌─────────────────────────────────────────────────────────────────────┐
│                     大模型训练完整流程                                │
│                                                                     │
│   ① 预训练（Pre-Training）                                          │
│   ② 指令微调（SFT）                                                 │
│   ③ 偏好对齐                                                        │
│   │                                                                 │
│   │  路线 A：DPO 直接偏好优化 ✅ 已完成                             │
│   │                                                                 │
│   │  路线 B：经典 RLHF                                              │
│   │  │  ③-c 训练奖励模型（RM）✅ 已完成                             │
│   │  │  ③-d PPO 近端策略优化   ← 📍 我们现在在这里！               │
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
| 训练 RM | 培训考官 | 教会裁判如何打分 |
| **PPO** | **在考官指导下反复练习** | **不断生成回答、接受打分、改进策略** |

PPO 是整个 RLHF 流程的"最后一公里"——让模型在奖励模型的指导下不断自我改进。

---

## 第一章：什么是 PPO？强化学习基础概念

### 1.1 用强化学习的视角看文本生成

PPO 来自强化学习（Reinforcement Learning, RL）领域。要理解 PPO，先要建立 RL 的基本概念：

```
强化学习的基本框架：

  ┌─────────┐   动作(action)   ┌─────────┐
  │  智能体  │ ──────────────→ │   环境   │
  │ (Agent)  │ ←────────────── │(Environ) │
  └─────────┘   奖励(reward)   └─────────┘

在大模型 RLHF 中：
  智能体 = 策略模型（生成回答）
  动作   = 生成的每一个 token
  环境   = 奖励模型（给回答打分）
  奖励   = 奖励模型输出的分数
  状态   = 已经生成的 token 序列
```

### 1.2 把文本生成看作"游戏"

```
想象策略模型在"玩一个游戏"：

游戏规则：
  1. 收到一个问题（游戏开始）
  2. 每一步选择一个 token（做出动作）
  3. 重复直到生成完整回答（游戏结束）
  4. 奖励模型给回答打分（获得奖励）

目标：学会一种"策略"，让获得的奖励分数尽可能高

例如：
  问题："推荐一部电影"

  策略模型第1步：选择 "推" → 状态变为 "推"
  策略模型第2步：选择 "荐" → 状态变为 "推荐"
  策略模型第3步：选择 "《" → 状态变为 "推荐《"
  ...
  策略模型第N步：选择 "。" → 回答完成

  奖励模型打分：4.2 分 → 这个策略不错！
```

### 1.3 为什么叫"近端策略优化"？

PPO 的全称是 Proximal Policy Optimization（近端策略优化）：

```
"策略"（Policy）：模型生成 token 的概率分布
"优化"（Optimization）：让策略变得更好（获得更高奖励）
"近端"（Proximal）：每次只做小幅度的更新，不要偏离太远

为什么要"近端"？
  如果一次更新太大，策略可能突然变得很差（"策略崩溃"）
  PPO 通过限制每次更新的幅度来保证训练稳定性
  这就像学走路——每次只迈一小步，而不是试图一步跨十米
```

---

## 第二章：RLHF 中的 PPO —— 四个模型的协作

### 2.1 PPO 训练需要的四个模型

这是 PPO 最复杂的地方——需要同时维护四个模型：

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO 训练中的四个模型                       │
│                                                               │
│  ① 策略模型（Policy Model / Actor）                          │
│     作用：生成回答                                            │
│     状态：持续更新（这是我们要训练的目标）                     │
│     本脚本：AutoModelForCausalLMWithValueHead + LoRA          │
│                                                               │
│  ② 参考模型（Reference Model）                               │
│     作用：提供 KL 散度约束的基准                              │
│     状态：冻结不更新（策略模型的初始副本）                     │
│     本脚本：PPOTrainer 自动创建（ref_model=None）             │
│                                                               │
│  ③ 奖励模型（Reward Model）                                  │
│     作用：给回答打分                                          │
│     状态：冻结不更新（上一步训练好的 RM）                     │
│     本脚本：通过 pipeline("sentiment-analysis") 加载          │
│                                                               │
│  ④ 价值模型（Value Model / Critic）                          │
│     作用：估计当前状态的价值（用于计算优势函数）              │
│     状态：持续更新                                            │
│     本脚本：策略模型的 Value Head（共享 Transformer 基座）     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 四个模型如何协作？

```
每个训练步骤：

  问题 ──→ ① 策略模型 ──→ 生成回答
                              │
                              ▼
            ③ 奖励模型 ──→ 奖励分数 (R)
                              │
            ④ 价值模型 ──→ 价值估计 (V)
                              │
                              ▼
                    优势 A = R - V
                    "这个回答比平均水平好多少？"
                              │
            ② 参考模型 ──→ KL 惩罚
                    "策略偏离初始模型多远？"
                              │
                              ▼
                    PPO 损失函数
                              │
                              ▼
                更新 ① 策略模型 和 ④ 价值模型
```

### 2.3 与 DPO 的显存对比

```
DPO：2 个模型
  策略模型 + 参考模型
  Qwen2.5-0.5B: ~4-6 GB

PPO：4 个模型（但有优化技巧）
  策略模型（含 Value Head）+ 参考模型 + 奖励模型
  实际上：
    - 策略模型和价值模型共享 Transformer 基座（只多一个 Value Head）
    - 参考模型可以用 LoRA 的方式共享基座（PPOTrainer 的优化）
    - 奖励模型是独立的
  Qwen2.5-0.5B: ~8-12 GB
```

---

## 第三章：PPO 算法核心原理

### 3.1 PPO 的训练循环

```
PPO 训练的每一步：

┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  1. 收集经验（Rollout）                                      │
│     策略模型对一批问题生成回答                                │
│     奖励模型给每个回答打分                                    │
│                                                               │
│  2. 计算优势（Advantage Estimation）                         │
│     优势 = 奖励 - 价值估计                                   │
│     "这个回答比我预期的好多少？"                              │
│                                                               │
│  3. 策略更新（Policy Update）                                │
│     用 PPO-Clip 目标函数更新策略模型                          │
│     限制更新幅度，防止策略崩溃                                │
│                                                               │
│  4. 价值更新（Value Update）                                 │
│     更新 Value Head，让价值估计更准确                         │
│                                                               │
│  5. KL 散度检查                                              │
│     如果策略偏离参考模型太远，调整 KL 惩罚系数                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 PPO-Clip 目标函数（直觉版）

PPO 最核心的创新是 Clip（裁剪）机制：

```
普通策略梯度：
  更新 = 优势 × 概率比率
  问题：如果优势很大，更新也很大 → 策略可能崩溃

PPO-Clip：
  概率比率 r = π_new(a|s) / π_old(a|s)
  裁剪后的比率 = clip(r, 1-ε, 1+ε)    （ε 通常 = 0.2）

  目标函数 = min(r × A, clip(r, 1-ε, 1+ε) × A)

  效果：
    如果优势 A > 0（好动作）：
      r 最多增大到 1+ε = 1.2 → 概率最多增加 20%
    如果优势 A < 0（差动作）：
      r 最多减小到 1-ε = 0.8 → 概率最多减少 20%
```

**用图来理解 Clip 机制：**

```
目标函数值
    ↑
    │        ╱ 不裁剪（可能更新太大）
    │       ╱
    │      ╱
    │─────╱──────── 裁剪后（更新被限制）
    │    ╱
    │   ╱
    │──╱─────────→ 概率比率 r
    │ ╱
    │╱
    0   0.8  1.0  1.2
        1-ε       1+ε
```

### 3.3 KL 散度惩罚

除了 Clip 机制，PPO 还用 KL 散度惩罚来约束策略：

```
总奖励 = 奖励模型分数 - β × KL(策略模型 || 参考模型)

KL 散度衡量两个概率分布的差异：
  KL = 0  → 策略模型和参考模型完全一样
  KL 很大 → 策略模型偏离参考模型很远

β 是 KL 惩罚系数：
  β 大 → 惩罚重 → 策略不敢偏离 → 保守
  β 小 → 惩罚轻 → 策略可以大胆探索 → 激进

本脚本使用自适应 KL 控制（adap_kl_ctrl=True）：
  如果实际 KL > target_kl → 增大 β（加强约束）
  如果实际 KL < target_kl → 减小 β（放松约束）
  自动维持 KL 在目标值附近
```

---

## 第四章：Value Head —— 价值估计

### 4.1 为什么需要 Value Head？

```
假设策略模型生成了一个回答，奖励模型给了 3.5 分。

问题：3.5 分算好还是差？
  如果平均分是 2.0 → 3.5 分很好！应该鼓励这种回答
  如果平均分是 4.0 → 3.5 分偏差！应该抑制这种回答

Value Head 的作用就是估计"平均分"（即价值函数 V(s)）
有了它，我们就能计算优势函数：
  A = R - V = 3.5 - V
  A > 0 → 比预期好 → 鼓励
  A < 0 → 比预期差 → 抑制
```

### 4.2 模型架构

```python
# AutoModelForCausalLMWithValueHead 的结构：
#
# 输入 tokens
#     │
#     ▼
# ┌─────────────────────┐
# │  Transformer 基座    │  ← 共享的，用 LoRA 微调
# │  (24 层 attention)   │
# └──────────┬──────────┘
#            │
#     ┌──────┴──────┐
#     │             │
#     ▼             ▼
# ┌────────┐  ┌──────────┐
# │ LM Head│  │Value Head│
# │(生成)  │  │(价值估计)│
# │896→151K│  │ 896→1    │
# └────────┘  └──────────┘
#     │             │
#     ▼             ▼
# 下一个token    价值分数
# 的概率分布     (标量)
```

Value Head 非常小（只有 896 个参数），但作用很大。

---

## 第五章：PPO 训练参数详解

### 5.1 PPOConfig 关键参数

```python
config = PPOConfig(
    # ---- 基础参数 ----
    steps=20000,                   # 总训练步数
    learning_rate=1.41e-5,         # 学习率

    # ---- Batch 相关 ----
    batch_size=32,                 # 每次 PPO 更新的总样本数
    mini_batch_size=8,             # mini batch 大小
    gradient_accumulation_steps=4, # 梯度累积

    # ---- PPO 特有参数 ----
    ppo_epochs=3,                  # 每批数据重复优化的次数
    target_kl=0.1,                 # 目标 KL 散度
    init_kl_coef=0.2,             # KL 惩罚初始系数
    adap_kl_ctrl=True,            # 自适应 KL 控制
)
```

**参数详解：**

| 参数 | 值 | 含义 |
|------|-----|------|
| `batch_size` | 32 | 每次收集 32 个样本的经验 |
| `mini_batch_size` | 8 | 将 32 个样本分成 4 个 mini batch 更新 |
| `ppo_epochs` | 3 | 对这 32 个样本重复优化 3 轮 |
| `target_kl` | 0.1 | 如果 KL 散度超过 0.1，提前停止当前 epoch |
| `init_kl_coef` | 0.2 | KL 惩罚的初始权重 |
| `adap_kl_ctrl` | True | 自动调整 KL 系数 |

### 5.2 生成参数

```python
generation_kwargs = {
    "top_k": 0.0,          # 不限制候选 token 数量
    "top_p": 1.0,          # 不做 nucleus 采样
    "do_sample": True,     # 必须采样！PPO 需要探索性
}

output_length_sampler = LengthSampler(32, 128)
# 每次随机采样一个生成长度（32~128 token）
```

**为什么 PPO 必须用采样而不是贪心解码？**

```
贪心解码（do_sample=False）：
  每次选概率最高的 token → 每次生成相同的回答
  → 没有探索性 → 模型无法发现更好的回答

采样（do_sample=True）：
  按概率分布随机选 token → 每次生成不同的回答
  → 有探索性 → 模型可能偶然生成更好的回答 → 学习到新策略

这就像：
  贪心 = 只走你知道的最短路 → 可能错过更好的路
  采样 = 偶尔尝试新路线 → 可能发现捷径
```

### 5.3 Adafactor 优化器

```python
optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=config.learning_rate,
)
```

**为什么用 Adafactor 而不是 AdamW？**

```
AdamW：
  为每个参数维护 2 个状态（一阶矩 m 和二阶矩 v）
  显存 = 模型参数 × 3（参数 + m + v）

Adafactor：
  用矩阵分解近似二阶矩，大幅减少状态存储
  显存 ≈ 模型参数 × 1.5

PPO 已经需要维护多个模型，显存很紧张
Adafactor 的省显存特性在这里非常有价值
```

---

## 第六章：代码逐行对照解读

### 6.1 整体代码结构

```python
# ┌─────────────────────────────────────────────────┐
# │ 第一部分：导入依赖                                │
# │ 第二部分：超参数和路径配置                        │
# │ 第三部分：初始化 wandb                            │
# │ 第四部分：加载和预处理数据集                      │
# │ 第五部分：PPO 配置                                │
# │ 第六部分：加载策略模型（带 Value Head）            │
# │ 第七部分：加载奖励模型                            │
# │ 第八部分：生成配置                                │
# │ 第九部分：PPO 训练主循环                          │
# │ 第十部分：保存最终模型                            │
# └─────────────────────────────────────────────────┘
```

### 6.2 核心训练循环

```python
for epoch in range(NUM_EPOCH):
    for batch in ppo_trainer.dataloader:
        # 步骤 1：策略模型生成回答
        question_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # 步骤 2：奖励模型打分
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

        # 步骤 3：PPO 更新
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
```

**ppo_trainer.step() 内部做了什么？**

```
step() 的内部流程：

1. 计算旧策略的对数概率
   old_logprobs = log π_old(response | question)

2. 计算价值估计
   values = value_head(question + response)

3. 计算优势函数（GAE - Generalized Advantage Estimation）
   advantages = rewards - values  （简化版）

4. PPO 更新（重复 ppo_epochs=3 次）：
   for mini_batch in split(batch, mini_batch_size):
     new_logprobs = log π_new(response | question)
     ratio = exp(new_logprobs - old_logprobs)
     clipped_ratio = clip(ratio, 1-ε, 1+ε)
     policy_loss = -min(ratio × advantages, clipped_ratio × advantages)
     value_loss = (values - rewards)²
     loss = policy_loss + 0.5 × value_loss

     if KL > target_kl: break  # 提前停止

5. 更新 KL 系数（自适应控制）
```

### 6.3 奖励模型的加载方式

```python
# 用 pipeline 加载奖励模型（巧妙的做法）
sentiment_pipe = pipeline(
    "sentiment-analysis",      # 借用情感分析的 pipeline
    model=REWARD_PATH,         # 实际加载的是奖励模型
    tokenizer=rm_tokenizer,
)

# 使用时：
pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
```

为什么用 `sentiment-analysis` pipeline？因为奖励模型本质上就是一个序列分类模型（输出一个分数），和情感分析模型的接口完全一样。这是一个复用现有基础设施的巧妙做法。

---

## 第七章：四种偏好对齐方法总结对比

学完了所有四个部分，让我们做一个全面的对比：

### 7.1 流程对比

```
方法 1：DPO（最简单）
  偏好数据 ──→ DPO 训练 ──→ 对齐模型
  需要：2 个模型

方法 2：RLHF = RM + PPO（最经典）
  偏好数据 ──→ 训练 RM ──→ PPO 训练 ──→ 对齐模型
  需要：4 个模型
```

### 7.2 详细对比表

| 维度 | DPO 数据合成 | DPO 训练 | RM 训练 | PPO 训练 |
|------|-------------|---------|---------|---------|
| 脚本 | `qwen_dpo_data.py` | `qwen_dpo.py` | `qwen_rm.py` | `qwen_ppo.py` |
| 输入 | 问题集 | 偏好三元组 | 偏好对 | 问题集 + RM |
| 输出 | 偏好数据 | 对齐模型 | 奖励模型 | 对齐模型 |
| 模型数 | 2+1(API) | 2 | 1 | 4 |
| 训练方式 | 不训练 | 监督学习 | 监督学习 | 强化学习 |
| 复杂度 | 低 | 低 | 中 | 高 |
| 显存需求 | 高(2个8B) | 中 | 低(LoRA) | 高(4个模型) |

### 7.3 选择建议

```
如果你是初学者，想快速体验偏好对齐：
  → 选 DPO（简单、稳定、效果好）

如果你需要在线学习（边训练边生成新数据）：
  → 选 PPO（可以持续改进）

如果你的场景需要精细的奖励信号：
  → 选 RM + PPO（奖励模型可以提供连续的分数）

如果你计算资源有限：
  → 选 DPO（只需要 2 个模型）

实际工业界的趋势：
  2023 年：PPO 为主（OpenAI 的 InstructGPT/ChatGPT）
  2024 年：DPO 和 PPO 并存，DPO 越来越流行
  2025 年：GRPO 等新方法兴起（本项目也有 qwen_grpo.py）
```

---

## 附录：常见问题

### Q1：PPO 训练不稳定怎么办？

PPO 训练比 DPO 更容易出问题，常见解决方案：

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 奖励分数不上升 | 学习率太小 | 增大 learning_rate |
| 奖励突然崩溃 | 学习率太大 | 减小 learning_rate |
| KL 散度爆炸 | 更新太激进 | 减小 target_kl |
| 回答变得重复 | 探索不足 | 增大 temperature |
| 回答变得无意义 | 奖励模型被"欺骗" | 检查 RM 质量 |

### Q2：什么是"奖励黑客"（Reward Hacking）？

```
奖励黑客是 PPO 训练中最常见的问题：

策略模型发现了一种"作弊"方式：
  生成某种特定模式的文本，能骗过奖励模型获得高分
  但这种文本对人类来说毫无意义

例如：
  奖励模型可能对"长回答"给高分
  → 策略模型学会生成又长又啰嗦的回答
  → 奖励分数很高，但回答质量很差

解决方案：
  1. KL 散度惩罚（本脚本已使用）
  2. 提高奖励模型的质量
  3. 定期人工检查生成的回答
```

### Q3：`ref_model=None` 是什么意思？

```python
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,  # 不手动传入参考模型
    ...
)
```

当 `ref_model=None` 时，PPOTrainer 会自动处理：
- 如果使用了 LoRA，参考模型就是去掉 LoRA 适配器的基座模型（不需要额外显存）
- 如果没有使用 LoRA，会复制一份模型作为参考模型

使用 LoRA 时的这个优化非常巧妙——不需要额外的显存来存储参考模型。

### Q4：`eos_token_id=100_000` 为什么设这么大？

```python
generation_kwargs = {
    "eos_token_id": 100_000,  # 一个不存在的 token id
}
```

这是一个技巧：设置一个不存在的 eos_token_id，让模型不会因为生成了 eos token 而提前停止。生成长度完全由 `length_sampler` 控制（32~128 token）。

这样做的原因：PPO 训练需要模型生成固定长度范围的回答，如果模型过早停止，会影响训练效果。

### Q5：PPO 训练需要多少显存？

```
Qwen2.5-0.5B + LoRA + PPO：
  策略模型（含 Value Head）：~1.5 GB
  参考模型（LoRA 共享基座）：~0 GB（额外）
  奖励模型：~1 GB
  优化器状态：~1 GB
  激活值和梯度：~2-4 GB
  总计：~6-8 GB

如果用更大的模型（如 7B）：
  总计：~50-80 GB → 需要多卡或 DeepSpeed
```

### Q6：PPO 和 DPO 的效果谁更好？

```
这是一个没有定论的问题，取决于具体场景：

PPO 的优势场景：
  - 有高质量的奖励模型
  - 需要在线学习和持续改进
  - 计算资源充足

DPO 的优势场景：
  - 有高质量的偏好数据
  - 计算资源有限
  - 需要简单稳定的训练流程

学术界的共识：
  在大多数 benchmark 上，DPO 和 PPO 的效果相当
  但 PPO 在某些需要精细控制的场景中略有优势
```

---

> 📌 **恭喜！** 你已经完成了大模型强化学习部分的全部四个模块的学习：
> 1. ✅ DPO 偏好数据合成
> 2. ✅ DPO 直接偏好优化
> 3. ✅ 训练奖励模型 RM
> 4. ✅ PPO 近端策略优化
>
> 你现在已经掌握了大模型从预训练到偏好对齐的完整流程！
