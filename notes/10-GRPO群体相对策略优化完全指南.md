# 🧠 GRPO 群体相对策略优化完全指南 —— 从 PPO 到 DeepSeek-R1 的进化

> 本文档配合 `qwen_grpo.py` 和 `utils/grpo_utils.py` 代码，手把手带你理解 GRPO 训练的每一个环节。
> 假设你已经完成了 DPO、RM、PPO 的学习，这是强化学习部分的进阶篇。

---

## 📋 目录

- [第零章：GRPO 在大模型训练中的位置](#第零章grpo-在大模型训练中的位置)
- [第一章：从 PPO 到 GRPO —— 为什么需要新方法？](#第一章从-ppo-到-grpo--为什么需要新方法)
- [第二章：GRPO 核心原理详解](#第二章grpo-核心原理详解)
- [第三章：DeepSeek-R1 的技术路线](#第三章deepseek-r1-的技术路线)
- [第四章：奖励函数设计 —— GRPO 的灵魂](#第四章奖励函数设计--grpo-的灵魂)
- [第五章：GRPO 训练参数详解](#第五章grpo-训练参数详解)
- [第六章：代码逐行对照解读](#第六章代码逐行对照解读)
- [第七章：PPO vs GRPO 全面对比](#第七章ppo-vs-grpo-全面对比)
- [第八章：R1-Zero 的"顿悟"现象](#第八章r1-zero-的顿悟现象)
- [附录：常见问题](#附录常见问题)

---

## 第零章：GRPO 在大模型训练中的位置

```
┌───────────────────────────────────────────────────────────────────────┐
│                       大模型训练完整流程                                │
│                                                                       │
│   ① 预训练（Pre-Training）        ✅ 已完成                          │
│   ② 指令微���（SFT）               ✅ 已完成                          │
│   ③ 偏好对齐                                                         │
│   │                                                                   │
│   │  路线 A：DPO 直接偏好优化      ✅ 已完成                         │
│   │                                                                   │
│   │  路线 B：经典 RLHF（RM + PPO） ✅ 已完成                         │
│   │                                                                   │
│   │  路线 C：GRPO（DeepSeek-R1 路线） ← 📍 我们现在在这里！         │
│   │      不需要 RM，使用规则奖励                                      │
│   ��      重点训练推理能力                                              │
│   │                                                                   │
│   ④ 知识蒸馏（下一步）                                                │
│   ⑤ 评估与部署                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

**打个比方：**

| 方法 | 类比 | 特点 |
|------|------|------|
| DPO | 拿着标准答案对照学习 | 简单直接，但依赖数据质量 |
| PPO | 有老师（RM）在旁边指导练习 | 效果好但需要先培训老师 |
| **GRPO** | **自己做多道题，互相对比找规律** | **不需要老师，自我进化** |

---

## 第一章：从 PPO 到 GRPO —— 为什么需要新方法？

### 1.1 PPO 的三大痛点

你在上一篇学习了 PPO，应该体会到了它的复杂性：

```
痛点 1：需要训练和维护 4 个模型
┌──────────────────────────────────────────────────┐
│  PPO 需要同时加载：                                │
│    ① 策略模型（Policy）   → 要训练                │
│    ② 参考模型（Reference）→ 冻结，用于 KL 约束    │
│    ③ 奖励模型（Reward）   → 冻结，需要提前训练     │
│    ④ 价值模型（Value）    → 要训练                 │
│                                                    │
│  显存需求：4 个模型 ≈ 模型参数量 × 4               │
│  对于 7B 模型，至少需要 50-80 GB 显存              │
└──────────────────────────────────────────────────┘

痛点 2：奖励模型（RM）本身就很难训练
  - 需要大量高质量的人类偏好数据
  - RM 的质量直接决定了 PPO 的上限
  - 容易出现"奖励黑客"（Reward Hacking）问题

痛点 3：训练过程不稳定
  - 4 个模型协作，任何一个出问题都会影响整体
  - 超参数敏感，调参困难
  - 训练开销大，迭代慢
```

### 1.2 GRPO 如何解决这些问题？

GRPO（Group Relative Policy Optimization）由 DeepSeek 在 2024 年提出，核心思想是：

```
GRPO 的三大简化：

1. 去掉奖励模型 → 使用规则函数代替
   PPO：奖励 = RM(回答)            → 需要训练 RM
   GRPO：奖励 = rule_func(回答)    → 直接定义规则

2. 去掉���值模型 → 使用组内平均值代替
   PPO：优势 = 奖励 - V(状态)      → 需要 Value Head
   GRPO：优势 = 奖励 - 组内平均奖励  → 不需要额外模型

3. 模型数量：4 → 2
   PPO：策略 + 参考 + 奖励 + 价值 = 4 个模型
   GRPO：策略 + 参考 = 2 个模型
```

---

## 第二章：GRPO 核心原理详解

### 2.1 "Group"的含义 —— 分组采样

GRPO 最核心的创新就是"Group（分组）"：

```
PPO 的做法：
  问题 Q → 生成 1 个回答 → 奖励模型打分 → 更新策略
  问题：只有 1 个回答，无法比较好坏

GRPO 的做法：
  问题 Q → 生成 G 个回答 → 规则打分 → 组内相对排名 → 更新策略

具体例子（num_generations=8）：
  问题："1+1等于几？"

  回答 1: "<think>直接算</think><answer>2</answer>"      → 准确✓ 格式✓ → 2.0分
  回答 2: "<think>1加1等于2</think><answer>2</answer>"    → 准确✓ 格式✓ → 2.0分
  回答 3: "答案是2"                                       → 准确✓ 格式✗ → 1.0分
  回答 4: "<think>算一下</think><answer>3</answer>"       → 准确✗ 格式✓ → 1.0分
  回答 5: "我不知道"                                      → 准确✗ 格式✗ → 0.0分
  回答 6: "<think>嗯...</think><answer>2</answer>"        → 准确✓ 格式✓ → 2.0分
  回答 7: "2"                                             → 准确✓ 格式✗ → 1.0分
  回答 8: "<think>思考</think><answer>1</answer>"         → 准确✗ 格式✓ → 1.0分

  组内平均分 = (2+2+1+1+0+2+1+1)/8 = 1.25

  相对优势 A = 每个回答的分数 - 组内平均分：
  回答 1: A = 2.0 - 1.25 = +0.75  → 鼓励！（格式好+答案对）
  回答 2: A = 2.0 - 1.25 = +0.75  → 鼓励！
  回答 5: A = 0.0 - 1.25 = -1.25  → 抑制！（什么都不对）
  回答 4: A = 1.0 - 1.25 = -0.25  → 轻微抑制（格式对但答案错）
```

### 2.2 GRPO 的数学公式

```
GRPO 的目标函数：

                    1    G
  J_GRPO(θ) = ─── × Σ  [ min(r_i × Â_i, clip(r_i, 1-ε, 1+ε) × Â_i) - β × KL_i ]
                G   i=1

其中：
  G = 组的大小（num_generations，本脚本中 G=8）
  r_i = π_θ(a_i|q) / π_old(a_i|q)    → 新旧策略的概率比率
  Â_i = (R_i - mean(R)) / std(R)      → 标准化后的组内相对优势（关键！）
  ε = clip 范围（通常 0.2���
  β = KL 惩罚系数
  KL_i = 策略模型和参考模型之间的 KL 散度

关键区别（与 PPO 对比）：
  PPO 的优势：  A = R - V(s)         → 需要价值模型 V
  GRPO 的优势：Â = (R - mean(R)) / std(R) → 只需要组内统计量！
```

### 2.3 为什么组内相对排名有效？

```
直觉理解：

假设你参加一场考试：
  绝对评分：你考了 70 分 → 好还是差？不知道！
  相对评分：你在班里排第 3 → 很好！

PPO 使用"绝对评分"（Value Head 估计的基准）
GRPO 使用"相对评分"（同组其他回答的平均分作为基准）

GRPO 的优势：
  1. 基准更准确：同一个问题的多个回答，直接比较更公平
  2. 自带归一化：不需要额外训练价值网络来估计基准
  3. 对奖励尺度不敏感：只关心相对好坏，不关心绝对分数

GRPO 的代价：
  需要对每个问题生成多个回答 → 推理计算量增大
  但省去了训练和推理奖励模型的计算量 → 总体可能更高效
```

### 2.4 GRPO 的完整训练流程

```
┌──────────────────────────────────────────────────────────────────┐
│                    GRPO 训练的一个完整步骤                         │
│                                                                    │
│  步骤 1：采样一批问题 {q_1, q_2, ..., q_B}                       │
│         B = per_device_train_batch_size = 24                      │
│                                                                    │
│  步骤 2：对每个问题，生成 G 个回答                                 │
│         q_1 → {a_1^1, a_1^2, ..., a_1^8}                         │
│         q_2 → {a_2^1, a_2^2, ..., a_2^8}                         │
│         ...                                                        │
│         总共生成 B × G = 24 × 8 = 192 个回答                      │
│                                                                    │
│  步骤 3：用奖励函数给每个回答打分                                  │
│         R_i = format_reward(a_i) + accuracy_reward(a_i)           │
│                                                                    │
│  步骤 4：在每个组内计算相对优势                                    │
│         对于问题 q_j 的第 i 个回答：                               │
│         Â_i = (R_i - mean(R_j^1...R_j^8)) / std(R_j^1...R_j^8)  │
│                                                                    │
│  步骤 5：用 PPO-Clip 目标函数更新策略                              │
│         L = -min(r × Â, clip(r, 0.8, 1.2) × Â) + β × KL         │
│                                                                    │
│  步骤 6：记录日志（奖励、KL 散度、损失等）                        │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 第三章：DeepSeek-R1 的技术路线

### 3.1 DeepSeek-R1 系列模型

GRPO 来自 DeepSeek 的 R1 系列论文，这里梳理一下背景：

```
DeepSeek-R1 系列的发展脉络：

2024.01  DeepSeek-V2      → 基础模型（MoE 架构）
2024.05  DeepSeek-V2.5    → 改进版基础模型
2024.12  DeepSeek-V3      → 更强的基础模型（671B MoE）
2025.01  DeepSeek-R1-Zero → 纯 RL 训练出推理能力的突破性实验
         DeepSeek-R1      → 完整的推理模型（RL + 蒸馏）
         DeepSeek-R1-Distill → 蒸馏版本（1.5B~70B）
```

### 3.2 R1 的两条技术路线

DeepSeek-R1 论文提出了两条路线，我们的代码实现的是 R1-Zero：

```
路线 1：R1-Zero（本脚本实现的）
═══════════════════════════════
  基座模型 ──→ 直接 GRPO 训练 ──→ 推理模型
  
  特点：
  - 不需要 SFT，从基座模型直接开始
  - 只用规则奖励（格式 + 准确性）
  - 模型会自发涌现"思维链"推理能力
  - 训练数据：只需要数学题（有标准答案的）
  
  优点：简单、优雅、发现了"涌现"现象
  缺点：回答格式不够规范、可能混合语言、推理链可读性差


路线 2：R1（完整版）
═══════════════════════════════
  基座模型 ──→ SFT（冷启动）──→ GRPO ──→ SFT（拒绝采样）──→ GRPO ──→ 最终模型
                                                                    │
                                                                    ▼
                                                              蒸馏到小模型
  特点：
  - 先用少量高质量 CoT 数据做 SFT 冷启动
  - 然后 GRPO 训练
  - 再用 GRPO 模型生成数据做拒绝采样 SFT
  - 最终再次 GRPO 训练
  - 最后蒸馏到各种大小的模型
  
  优点：效果更好、回答更规范
  缺点：流程更复杂
```

### 3.3 为什么叫 "R1-Zero"？

```
"Zero" 的含义：零人工标注数据

  传统 RLHF（如 ChatGPT）：
    需要人类标注偏好数据 → 训练 RM → PPO
    人工成本高、标注一致性难保证

  R1-Zero：
    只需要数学题的标准答案（可以自动验证）
    奖励函数是规则定义的，不需要人类参与
    → "Zero" 人工偏好标注

  这是 GRPO 的一个重要优势：
    对于有明确标准答案的领域（数学、编程、逻辑推理），
    可以完全自动化强化学习训练
```

---

## 第四章：奖励函数设计 —— GRPO 的灵魂

### 4.1 奖励函数的整体架构

```
GRPO 的奖励函数设计理念：

  总奖励 = 格式奖励 + 准确性奖励

  ┌─────────────────────────────────────────────────────┐
  │                                                       │
  │  格式奖励 (format_reward)                            │
  │  ┌─────────────────────────────────────┐             │
  │  │ 检查：是否包含 <think>...</think>     │             │
  │  │       和 <answer>...</answer>          │             │
  │  │ 得分：正确 → 1.0，错��� → 0.0          │             │
  │  │ 作用：教模型"怎么说"                   │             │
  │  └─────────────────────────────────────┘             │
  │           +                                           │
  │  准确性奖励 (accuracy_reward)                        │
  │  ┌─────────────────────────────────────┐             │
  │  │ 检查：<answer> 中的答案是否正确       │             │
  │  │ 得分：正确 → 1.0，错误 → 0.0          │             │
  │  │ 作用：教模型"说什么"                   │             │
  │  └─────────────────────────────────────┘             │
  │           =                                           │
  │  总分范围：0.0 ~ 2.0                                 │
  │                                                       │
  └─────────────────────────────────────────────────────┘
```

### 4.2 格式奖励详解

```python
def format_reward(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```

```
正则表达式分解：

  ^<think>     ← 必须以 <think> 开头
  .*?          ← 任意思考内容（非贪婪匹配）
  </think>     ← 思考结束标签
  \s*          ← 允许中间有空白字符（换行、空格等）
  <answer>     ← 答案开始标签
  .*?          ← 任意答案内容
  </answer>$   ← 必须以 </answer> 结尾

合格的回答示例：
  ✅ "<think>让我算一下，1+1=2</think><answer>2</answer>"
  ✅ "<think>这道题需要用勾股定理...</think>\n<answer>5</answer>"

不合格的回答示例：
  ❌ "答案是2"                         → 没有 <think> 标签
  ❌ "<think>思考</think>答案是2"      → 没有 <answer> 标签
  ❌ "我觉得<think>嗯</think><answer>2</answer>"  → 不是以 <think> 开头
```

### 4.3 准确性奖励详解

```python
def accuracy_reward(completions, **kwargs):
    solutions = kwargs["solution"]  # 标准答案
    # ... 
    gold_parsed = parse(solution, ...)     # 解析标准答案
    answer_parsed = parse(content, ...)    # 解析模型答案
    rewards.append(float(verify(answer_parsed, gold_parsed)))  # 数学等价验证
```

```
math_verify 库的工作原理：

  步骤 1：提取 LaTeX 表达式
    输入："<think>...</think><answer>\boxed{42}</answer>"
    提取："\boxed{42}" → 42

  步骤 2：数学等价验证
    verify(模型答案, 标准答案)
    
    支持的等价判断：
      "2"   vs "2.0"      → True（数值相等）
      "1/2" vs "\frac{1}{2}" → True（分数等价）
      "x^2" vs "x²"       → True（表示等价）

为什么这种方式比训练 RM 更好？（在数学领域）
  1. 100% 准确：规则验证不会出错，RM 可能误判
  2. 无需标注数据：标准答案本身就在数据集里
  3. 没有"奖励黑客"：模型无法欺骗数学验证
```

### 4.4 可以自定义的其他奖励函数

```
DeepSeek-R1 论文中还探讨了其他可能的奖励：

1. 长度奖励：鼓励模型生成适当长度的推理链
   def length_reward(completions):
       lengths = [len(c) for c in completions]
       return [1.0 if 50 < l < 500 else 0.5 for l in lengths]

2. 一致性奖励：鼓励模型的推理过程和最终答案一致
   （检查 <think> 中的推导是否得出 <answer> 中的结论）

3. 语言一致性奖励：鼓励模型使用一致的语言
   （R1-Zero 中发现模型会混合中英文推理）

4. 代码执行奖励：对于编程题，直接运行代码验证
   def code_reward(completions):
       return [1.0 if run_code(c).success else 0.0 for c in completions]

关键原则：
  GRPO 的奖励函数应该是"可验证的"——要么对要么错，
  不需要人类主观判断。这是它适合数学/编程等领域的根本原因。
```

---

## 第五章：GRPO 训练参数详解

### 5.1 GRPOConfig 核心参数

```python
training_args = GRPOConfig(
    # ---- GRPO 特有参数 ----
    num_generations=8,           # 每个问题生成的回答数量（组大小 G）
    max_completion_length=256,   # 每个回答的最大 token 数
    max_prompt_length=512,       # 输入 prompt 的最大 token 数
    remove_unused_columns=False, # 保留 solution 列给奖励函数使用

    # ---- 常规训练参数 ----
    learning_rate=1e-5,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    bf16=True,
    per_device_train_batch_size=24,
)
```

**参数详解：**

| 参数 | 值 | 含义 | 调参建议 |
|------|-----|------|---------|
| `num_generations` | 8 | 每个问题生成多少个回答 | 增大→排名更稳定，但显存↑ |
| `max_completion_length` | 256 | 每个回答最长多少 token | 太短→推理不完整，太长→浪费算力 |
| `max_prompt_length` | 512 | prompt 最长多少 token | 取决于题目长度 |
| `remove_unused_columns` | False | 是否移除数据集多余列 | 必须 False！奖励函数需要 solution 列 |
| `learning_rate` | 1e-5 | 学习率 | GRPO 通常用较小学习率 |
| `gradient_accumulation_steps` | 16 | 梯度累积步数 | 等效 batch = 24×16 = 384 |
| `per_device_train_batch_size` | 24 | 每 GPU batch 大小 | 受显存限制 |

### 5.2 关键参数的显存计算

```
GRPO 显存占用分析（Qwen2.5-0.5B + LoRA）：

模型本身：
  基座模型（bfloat16）：~1 GB
  LoRA 适配器：~0.01 GB
  参考模型：共享基座，额外 ~0 GB（LoRA 的优势！）

生成阶段（最占显存的阶段）：
  每个回答：max_completion_length × hidden_dim = 256 × 896
  一批回答：batch_size × num_generations = 24 × 8 = 192 个回答
  KV Cache：192 × (512+256) × 896 × 24层 × 2(K,V) ≈ 非常大！
  
  实际中 GRPOTrainer 会自动分批生成，不会一次性生成所有回答

优化器状态（AdamW）：
  只需要为 LoRA 参数维护状态：~0.05 GB

总计估算：~6-10 GB（单卡 Qwen2.5-0.5B + LoRA）
```

### 5.3 num_generations 的影响

```
num_generations 是 GRPO 最重要的超参数：

G=2（最小值）：
  ┌────────────────────────────────────┐
  │ 回答 A: 1.0分                      │
  │ 回答 B: 0.0分                      │
  │ 平均: 0.5                          │
  │ A 的优势: +0.5, B 的优势: -0.5     │
  │                                    │
  │ 问题：只有 2 个样本，排名信号太弱   │
  │ 如果两个都对或都错，优势为 0，学不到 │
  └────────────────────────────────────┘

G=8（本脚本的设置）：
  ┌────────────────────────────────────┐
  │ 8 个回答覆盖更多的好坏情况          │
  │ 排名信号更稳定                      │
  │ 方差估计更准确                      │
  │ 经验上是性价比最高的选择            │
  └────────────────────────────────────┘

G=64（DeepSeek-R1 论文的设置）：
  ┌────────────────────────────────────┐
  │ 排名信号最稳定                      │
  │ 但生成 64 个回答需要大量算力         │
  │ 适合大规模训练集群                  │
  └────────────────────────────────────┘

经验公式：
  G 越大 → 优势估计越准 → 训练越稳定
  G 越大 → 生成成本越高 → 训练越慢
  推荐：显存够就用 8~16，资源充足用 32~64
```

---

## 第六章：代码逐行对照解读

### 6.1 整体代码结构

```python
# ┌─────────────────────────────────────────────────┐
# │ 第一部分：环境配置（CUDA 内存优化）              │
# │ 第二部分：路径配置                                │
# │ 第三部分：加载基座模型                            │
# │ 第四部分：加载和预处理数据集                      │
# │ 第五部分：LoRA 配置                               │
# │ 第六部分：GRPO 训练配置                           │
# │ 第七部分：初始化 wandb                            │
# │ 第八部分：创建 GRPOTrainer 并训练                 │
# └─────────────────────────────────────────────────┘
```

### 6.2 数据预处理：format_to_r1

```python
# 在 utils/utils.py 中定义
def format_to_r1(example):
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within "
        "<think> </think> and <answer> </answer> tags..."
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }
```

```
这个函数的关键设计：

1. System Prompt 的作用：
   告诉模型要使用 <think>...</think><answer>...</answer> 格式
   这与 format_reward 的正则表达式相呼应
   → System Prompt 定义期望格式，format_reward 检查是否遵循

2. 输出格式：
   {"prompt": [系统消息, 用户消息]}
   GRPOTrainer 会自动将 prompt 传给模型生成回答

3. 为什么只有 prompt 没有 response？
   因为 GRPO 是在线学习（Online RL）：
   回答由模型在训练过程中实时生成，不是预先准备好的
   这与 SFT/DPO 最大的区别！
```

### 6.3 GRPOTrainer 的核心

```python
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset
)
```

```
GRPOTrainer 初始化时做了什么：

1. model: 策略模型（带 LoRA 的 Qwen2.5-0.5B）

2. reward_funcs: 奖励函数列表
   - 可以传入多个函数，GRPOTrainer 会自动将它们的分数相加
   - 每个函数签名：func(completions, **kwargs) → list[float]
   - kwargs 会包含数据集中的所有列（因为 remove_unused_columns=False）

3. 参考模型：
   GRPOTrainer 会自动创建参考模型
   使用 LoRA 时，参考模型 = 去掉 LoRA 的基座模型（不需要额外显存）

4. 不需要传入：
   - 奖励模型（用 reward_funcs 代替）
   - 价值模型（用组内平均代替）
   - 参考模型（自动创建）

对比 PPOTrainer：
  PPOTrainer(config, model, ref_model, tokenizer, ...)
  需要手动传入参考模型、手动调用奖励模型
```

### 6.4 训练循环（GRPOTrainer.train() 内部）

```
GRPOTrainer 的训练循环（简化伪代码）：

for epoch in range(num_train_epochs):
    for batch in dataloader:
        prompts = batch["prompt"]           # 一批问题

        # ── 步骤 1：生成多个回答 ──
        all_completions = []
        for prompt in prompts:
            # 对每个问题生成 num_generations=8 个回答
            completions = model.generate(
                prompt, 
                num_return_sequences=8,
                max_new_tokens=256,
                do_sample=True,          # 必须采样！需要多样性
                temperature=0.7,
            )
            all_completions.append(completions)

        # ── 步骤 2：计算奖励 ──
        rewards = []
        for func in reward_funcs:        # [format_reward, accuracy_reward]
            func_rewards = func(all_completions, **batch)
            rewards += func_rewards      # 多个奖励函数的分数相加

        # ── 步骤 3：计算组内相对优势 ──
        for group in groups_of_8(rewards):
            mean_r = mean(group)
            std_r = std(group)
            advantages = [(r - mean_r) / (std_r + 1e-8) for r in group]

        # ── 步骤 4：PPO-Clip 策略更新 ──
        old_logprobs = compute_logprobs(model_old, completions)
        new_logprobs = compute_logprobs(model, completions)
        ratio = exp(new_logprobs - old_logprobs)
        clipped_ratio = clip(ratio, 1-0.2, 1+0.2)
        loss = -min(ratio * advantages, clipped_ratio * advantages)

        # ── 步骤 5：KL 散度约束 ──
        kl = compute_kl(model, ref_model, completions)
        loss += beta * kl

        # ── 步骤 6：反向传播更新 ──
        loss.backward()
        optimizer.step()
```

---

## 第七章：PPO vs GRPO 全面对比

### 7.1 架构对比

```
PPO 架构：
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ 策略模型  │   │ 参考模型  │   │ 奖励模型  │   │ 价值模型  │
  │ (训练)   │   │ (冻结)   │   │ (冻结)   │   │ (训练)   │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘
       ↓               ↓              ↓              ↓
    生成回答       KL 约束基准      给回答打分     估计状态价值
                                      ↓              ↓
                                 优势 = 奖励 - 价值


GRPO 架构：
  ┌──────────┐   ┌──────────┐   ┌──────────────────────────┐
  │ 策略模型  │   │ 参考模型  │   │   规则奖励函数            │
  │ (训练)   │   │ (冻结)   │   │ format_reward             │
  └──────────┘   └──────────┘   │ accuracy_reward           │
       ↓               ↓        └──────────────────────────┘
  生成 G 个回答    KL 约束基准              ↓
       ↓                              给每个回答打分
       ↓                                    ↓
  优势 = (奖励 - 组内平均) / 组内标准差    ← ←
```

### 7.2 详细对比表

| 维度 | PPO | GRPO |
|------|-----|------|
| **模型数量** | 4（策略+参考+奖励+价值） | 2（策略+参考） |
| **奖励来源** | 训练好的奖励模型（神经网络） | 规则函数（代码定义） |
| **优势计算** | A = R - V(s)（需要价值网络） | Â = (R - μ) / σ（组内统计） |
| **生成数量** | 每个问题 1 个回答 | 每个问题 G 个回答 |
| **显存需求** | 高（4 个模型） | ��（2 个模型+多次生成） |
| **训练前提** | 需要先训练 RM | 需要设计奖励函数 |
| **适用领域** | 通用（依赖 RM 质量） | 有明确标准的领域（数学、编程） |
| **训练稳定性** | 较差（4 个模型协调） | 较好（结构简单） |
| **代码复杂度** | 高 | 低 |
| **论文代表** | InstructGPT (OpenAI) | DeepSeek-R1 (DeepSeek) |

### 7.3 五种偏好对齐方法总结

学到这里，你已经掌握了所有主流的偏好对齐方法：

```
┌─────────────────────────────────────────────────────────────────┐
│                     偏好对齐方法全景图                            │
│                                                                   │
│  离线方法（使用固定数据集）：                                     │
│  ┌──────────────────────────────────────────────┐               │
│  │  DPO：直接从偏好对中学习                      │               │
│  │  优点：简单、稳定                             │               │
│  │  缺点：依赖数据质量、不能在线改进             │               │
│  └──────────────────────────────────────────────┘               │
│                                                                   │
│  在线方法（训练中实时生成数据）：                                 │
│  ┌──────────────────────────────────────────────┐               │
│  │  PPO：经典 RLHF，奖励模型 + 价值网络         │               │
│  │  优点：通用性强                               │               │
│  │  缺点：复杂、不稳定、需要训练 RM             │               │
│  ├──────────────────────────────────────────────┤               │
│  │  GRPO：分组相对优化，规则奖励                 │               │
│  │  优点：简单、不需要 RM、适合推理任务          │               │
│  │  缺点：需要可验证的奖励函数                   │               │
│  └──────────────────────────────────────────────┘               │
│                                                                   │
│  数据构造方法：                                                   │
│  ┌──────────────────────────────────────────────┐               │
│  │  DPO 数据合成：用模型生成偏好对               │               │
│  │  RM 训练：训练奖励模型                        │               │
│  └──────────────────────────────────────────────┘               │
│                                                                   │
│  选择指南：                                                       │
│  有偏好数据 + 资源少 → DPO                                       │
│  通用场景 + 有 RM → PPO                                          │
│  数学/编程 + 有标准答案 → GRPO ✨                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第八章：R1-Zero 的"顿悟"现象

### 8.1 什么是"顿悟"（Aha Moment）？

DeepSeek-R1 论文中最引人注目的发现：

```
在 GRPO 训练过程中，模型展现了"顿悟"（Aha Moment）现象：

训练初期（前几百步）：
  模型：<think></think><answer>42</answer>
  → 学会了格式，但思考内容为空，直接猜答案
  → format_reward = 1.0, accuracy_reward ≈ 0

训练中期（几千步后）：
  模型：<think>let me think... the answer might be...</think><answer>42</answer>
  → 开始尝试填充思考内容，但推理不靠谱
  → format_reward = 1.0, accuracy_reward ≈ 0.2

训练后期（"顿悟"发生后）：
  模型：<think>First, I need to... Then by applying... Therefore...</think><answer>7</answer>
  → 突然学会了有逻辑的推理链！
  → format_reward = 1.0, accuracy_reward ≈ 0.6+

关键观察：
  推理能力不是逐渐提升的，而是在某个时刻突然涌现
  就像人类学习中的"恍然大悟"
```

### 8.2 为什么会出现顿悟？

```
可能的解释：

1. 奖励驱动的自然进化：
   - 初期：模型随机探索，偶尔格式正确的回答得到 format_reward
   - 中期：模型学会了格式，开始探索填充什么内容能提高 accuracy_reward
   - 后期：某次探索中，模型发现"先分步骤推理"能显著提高准确率
         → 这种策略被强化 → 推理能力涌现

2. 组内竞争机制：
   - 在同一组的 G 个回答中，有推理过程的回答更可能答对
   - 答对的回答获得正优势，被鼓励
   - 没有推理的回答获得负优势，被抑制
   - 经过大量训练步骤，"先推理再回答"的策略自然胜出

3. 与人类学习的类比：
   学生做数学题的过程：
   初期：瞎猜答案
   中期：知道要写过程，但经常写错
   后期：突然掌握了解题思路，正确率飙升
```

### 8.3 训练曲线的典型模式

```
奖励分数
    ↑
2.0 │                                    ╭───────── 趋于稳定
    │                                ╭──╯
    │                            ╭──╯
1.5 │                       ╭──╯
    │                   ╭──╯
    │               ╭──╯    ← "顿悟"发生！准确率快速上升
1.0 │──────────╭──╯
    │          │   ← 格式学会了，准确率开始缓慢提升
    │      ╭──╯
0.5 │  ╭──╯
    │╭╯    ← 逐渐学会正确格式
    │╯
0.0 └──────────────────────────────────────→ 训练步数
    0     500    1000   1500   2000   2500
```

---

## 附录：常见问题

### Q1：GRPO 和 PPO 哪个更好？

```
没有绝对的好坏，取决于应用场景：

GRPO 更适合的场景：
  ✅ 有明确标准答案的任务（数学、编程、逻辑推理）
  ✅ 资源有限（不需要训练和加载 RM）
  ✅ 想要训练推理能力
  ✅ 不想标注偏好数据

PPO 更适合的场景：
  ✅ 开放式任务（对话、写作、翻译）
  ✅ 没有明确标准答案
  ✅ 已经有训练好的 RM
  ✅ 需要根据人类偏好优化
```

### Q2：为什么 GRPO 不需要价值模型？

```
PPO 需要价值模型 V(s) 来估计"基准水平"：
  优势 = 奖励 - V(s)
  V(s) 是一个需要训练的神经网络，本身可能不准确

GRPO 用一个更简单的方法估计"基准水平"：
  优势 = (奖励 - 组内平均奖励) / 组内标准差
  组内平均奖励是一个统计量，不需要训练

为什么这样做可行？
  因为同一个问题的多个回答，它们的"基准水平"应该相同
  （都是同一个问题，难度一样）
  所以用这些回答的平均分作为基准是合理的

对比：
  PPO 的 V(s)：是一个训练出来的"猜测"→ 可能不准
  GRPO 的 mean(R)：是同组回答的真实平均分 → 更客观
```

### Q3：`remove_unused_columns=False` 为什么是必需的？

```python
# 数据集中有这些列：
# prompt: 问题（GRPOTrainer 使用）
# solution: 标准答案（accuracy_reward 使用）
# problem: 原始题目文本

# 如果 remove_unused_columns=True（默认值）：
# HuggingFace Trainer 会移除模型不需要的列
# solution 和 problem 会被删除
# → accuracy_reward 函数中 kwargs["solution"] 会报 KeyError！

# 所以必须设为 False，保留所有列
training_args = GRPOConfig(
    remove_unused_columns=False,  # 关键！
)
```

### Q4：GRPO 训练需要多少数据？

```
数据需求取决于任务复杂度：

数学推理任务：
  - DeepSeek-R1-Zero：使用了约 10 万道数学题
  - 本脚本：根据 data/reasoning 目录下的数据量
  - 建议最少：5000 道题（每道题生成 8 个回答 = 4 万训练样本）

数据质量要求：
  - 必须有标准答案（solution 列）
  - 题目难度适中（太简单→全部答对→没有排名信号）
  - 难度多样（让模型从简单到困难逐步进步）

与其他方法的对比：
  SFT：需要高质量的 (问题, 回答) 对
  DPO：需要 (问题, 好回答, 差回答) 三元组
  PPO：需要问��� + 训练好的 RM
  GRPO：只需要 (问题, 标准答案) 对 ← 最简单！
```

### Q5：为什么选择 LoRA 而不是全参数训练？

```
GRPO + LoRA 的三个好处：

1. 节省显存（最重要）：
   全参数训练：模型参数 × 2（参数 + 优化器状态）
   LoRA 训练：模型参数 × 1 + LoRA 参数 × 2（LoRA 很小）
   → GRPO 需要多次生成，显存压力本身就大

2. 参考模型免费获得：
   LoRA 训练时，参考模型 = 基座模型（冻结的部分）
   不需要额外加载一个完整的参考模型
   → 进一步节省显存

3. 防止灾难性遗忘：
   只更新少量参数，保留了基座模型的大部分能力
   → 不会因为 RL 训练而丧失基础的语言能力
```

### Q6：如何观察训练是否在进步？

```
在 wandb 中关注这些指标：

1. reward/mean（平均奖励）
   应该逐渐上升
   如果长期不升 → 学习率可能太小，或奖励函数有问题

2. reward/format（格式奖励）
   通常先快速上升到接近 1.0（模型很快学会格式）

3. reward/accuracy（准确性奖励）
   上升较慢，可能出现"顿悟"式的跳跃

4. kl（KL 散度）
   应该保持在合理范围（通常 < 10）
   如果太大 → 策略偏离太远，可能需要增大 KL 惩罚

5. loss（训练损失）
   应该逐渐下降
   如果震荡剧烈 → 学习率可能太大
```

### Q7：GRPO 能用于非数学任务吗？

```
可以！关键是要设计合适的奖励函数。

代码生成任务：
  def code_reward(completions, **kwargs):
      test_cases = kwargs["test_cases"]
      rewards = []
      for code, tests in zip(completions, test_cases):
          passed = run_tests(code, tests)
          rewards.append(passed / len(tests))  # 通过率
      return rewards

逻辑推理任务：
  def logic_reward(completions, **kwargs):
      answers = kwargs["answer"]
      for completion, answer in zip(completions, answers):
          # 提取最终答案并比较
          ...

难以使用 GRPO 的任务：
  ❌ 开放式对话（没有标准答案）
  ❌ 创意写作（好坏难以用规则判断）
  ❌ 主观问答（需要人类偏好判断）
  → 这些任务更适合 PPO（使用训练好的 RM）或 DPO
```

---

> 📌 **恭喜！** 你已经完成了 GRPO 的学习。回顾一下你的学习路径：
>
> **训练阶段：**
> 1. ✅ 预训练（Pre-Training）
> 2. ✅ Qwen 网络结构
> 3. ✅ 续训练（Continue Pre-Training）
> 4. ✅ 监督微调（SFT）
> 5. ✅ 大模型评估（Evaluation）
>
> **偏好对齐：**
> 6. ✅ DPO 偏好数据合成
> 7. ✅ DPO 直接偏好优化
> 8. ✅ 奖励模型 RM
> 9. ✅ PPO 近端策略优化
> 10. ✅ **GRPO 群体相对策略优化**（本篇）
>
> **下一步：** 知识蒸馏（Distillation）—— 学习如何将 GRPO 训练出的强推理能力迁移到更小的模型上。
> 这正是 DeepSeek-R1 技术路线的下半场！
