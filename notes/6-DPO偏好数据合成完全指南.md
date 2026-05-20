# 🔄 DPO 偏好数据合成完全指南 —— 从零理解到动手实践

> 本文档配合 `qwen_dpo_data.py` 代码，手把手带你理解 DPO 偏好数据合成的每一个环节。
> 假设你已经完成了预训练、SFT、评估的学习，我们在此基础上进入强化学习的第一步。

---

## 📋 目录

- [第零章：偏好数据合成在大模型训练中的位置](#第零章偏好数据合成在大模型训练中的位置)
- [第一章：什么是偏好数据？为什么需要它？](#第一章什么是偏好数据为什么需要它)
- [第二章：偏好数据的构造方法](#第二章偏好数据的构造方法)
- [第三章：distilabel 数据合成框架](#第三章distilabel-数据合成框架)
- [第四章：双模型生成策略](#第四章双模型生成策略)
- [第五章：EvolQuality —— 回答质量进化](#第五章evolquality--回答质量进化)
- [第六章：UltraFeedback —— 多维度评分](#第六章ultrafeedback--多维度评分)
- [第七章：代码逐行对照解读](#第七章代码逐行对照解读)
- [第八章：数据质量与常见问题](#第八章数据质量与常见问题)
- [附录：常见问题](#附录常见问题)

---

## 第零章：偏好数据合成在大模型训练中的位置

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
│   ③ 偏好对齐（RLHF / DPO）                                         │
│   │                                                                 │
│   │  ③-a 偏好数据合成          ← 📍 我们现在在这里！                │
│   │  │   构造"好回答 vs 差回答"的数据对                              │
│   │  │                                                               │
│   │  ③-b DPO 直接偏好优化                                           │
│   │  │   用偏好数据直接训练模型                                      │
│   │  │                                                               │
│   │  ③-c 训练奖励模型（RM）                                         │
│   │  │   训练一个"裁判"来给回答打分                                  │
│   │  │                                                               │
│   │  ③-d PPO 近端策略优化                                           │
│   │      用奖励模型指导策略模型优化                                   │
│   │                                                                 │
│   ④ 评估与部署                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**打个比方：**

| 阶段 | 类比 | 做什么 |
|------|------|--------|
| 预训练 | 上学读书12年 | 博览群书，积累知识 |
| SFT | 岗前培训 | 学会按要求完成任务 |
| **偏好数据合成** | **收集考试卷子** | **准备"好答案 vs 差答案"的对比材料** |
| DPO/PPO | 职业素养训练 | 学会什么该说什么不该说 |

偏好数据合成是偏好对齐的"地基"——没有高质量的偏好数据，后续的 DPO 和 PPO 训练都无从谈起。

---

## 第一章：什么是偏好数据？为什么需要它？

### 1.1 SFT 模型的局限

经过 SFT 训练后，模型已经学会了"按指令回答问题"。但 SFT 模型有一个根本问题：

```
用户：  "如何提高学习效率？"

SFT 模型回答 A（还不错）：
  "可以尝试番茄工作法，每25分钟专注学习，休息5分钟。
   同时建议做笔记、定期复习，保持充足睡眠。"

SFT 模型回答 B（有问题）：
  "学习效率的提高需要从多个维度考虑。首先，学习效率是一个
   复杂的概念，涉及到认知科学、教育心理学等多个领域..."
   ↑ 啰嗦、空洞、没有实际建议！
```

SFT 只教会了模型"怎么回答"，但没教它"什么样的回答更好"。模型可能生成正确但啰嗦的回答、正确但不友好的回答、甚至有害的回答。

### 1.2 偏好数据的核心思想

偏好数据的本质就是告诉模型：**面对同一个问题，哪种回答更好，哪种更差。**

```
偏好数据的基本结构：

┌─────────────────────────────────────────────────┐
│  问题 (prompt):  "如何提高学习效率？"             │
│                                                   │
│  ✅ chosen（好回答）:                             │
│     "可以尝试番茄工作法，每25分钟专注学习..."      │
│                                                   │
│  ❌ rejected（差回答）:                           │
│     "学习效率的提高需要从多个维度考虑..."          │
└─────────────────────────────────────────────────┘
```

有了这样的数据，DPO/PPO 就能训练模型：
- **增大** 生成 chosen 类回答的概率
- **减小** 生成 rejected 类回答的概率

### 1.3 偏好数据从哪来？

| 来源 | 做法 | 优缺点 |
|------|------|--------|
| **人工标注** | 雇人对比两个回答，选出更好的 | 质量最高，但成本极高、速度慢 |
| **AI 自动合成**（本脚本） | 用不同模型生成 + AI 裁判打分 | 成本低、速度快，质量取决于裁判模型 |
| **已有数据集** | 使用 UltraFeedback、HH-RLHF 等开源数据 | 最方便，但可能不适合你的场景 |

本脚本采用的是 **AI 自动合成** 方案，这也是目前业界最主流的做法。

---

## 第二章：偏好数据的构造方法

### 2.1 本脚本的三步流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                    偏好数据合成流水线                              │
│                                                                   │
│  输入：一批问题（questions）                                      │
│                                                                   │
│  步骤一：双模型独立生成                                           │
│  ┌──────────┐    ┌──────────┐                                    │
│  │  Qwen3   │    │  Llama3  │                                    │
│  │  8B      │    │  8B      │                                    │
│  └────┬─────┘    └────┬─────┘                                    │
│       │               │                                           │
│       ▼               ▼                                           │
│    回答 A           回答 B                                        │
│       │               │                                           │
│  步骤二：EvolQuality 质量进化                                     │
│       │               │                                           │
│       ▼               ▼                                           │
│  ┌─────────────────────────┐                                     │
│  │    DeepSeek（裁判模型）   │                                    │
│  │    改写润色两个回答       │                                    │
│  └─────────────────────────┘                                     │
│       │               │                                           │
│       ▼               ▼                                           │
│   进化回答 A'      进化回答 B'                                    │
│       │               │                                           │
│  步骤三：UltraFeedback 打分                                      │
│       └───────┬───────┘                                           │
│               ▼                                                   │
│  ┌─────────────────────────┐                                     │
│  │    DeepSeek（裁判模型）   │                                    │
│  │    多维度对比打分         │                                    │
│  └─────────────────────────┘                                     │
│               │                                                   │
│               ▼                                                   │
│  输出：(问题, 回答A', 回答B', 分数A, 分数B, 理由)                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 为什么要三步而不是一步？

你可能会问：为什么不直接让两个模型生成，然后人工/AI 判断哪个好？

| 步骤 | 作用 | 如果跳过会怎样 |
|------|------|----------------|
| 双模型生成 | 产生质量有差异的原始回答 | 只用一个模型，回答风格太相似，难以区分好坏 |
| EvolQuality | 放大质量差异，让好的更好 | 两个回答差距太小，裁判模型难以准确打分 |
| UltraFeedback | 多维度量化评分 | 没有分数就无法确定 chosen/rejected |

### 2.3 最终数据格式

每条数据包含以下字段：

```python
{
    "question":              "原始问题",
    "original_response_j":   "模型A的原始回答",
    "original_response_k":   "模型B的原始回答",
    "response_j":            "模型A回答经过进化后的版本",
    "response_k":            "模型B回答经过进化后的版本",
    "rating_j":              4.5,    # 进化后回答A的评分（1-5）
    "rating_k":              2.0,    # 进化后回答B的评分（1-5）
    "rationales":            ["评分理由A", "评分理由B"]
}
```

后续 DPO 训练时，会根据 `rating_j` 和 `rating_k` 的大小关系来确定：
- 分数更高的 → `chosen`（模型应该学习的好回答）
- 分数更低的 → `rejected`（模型应该避免的差回答）

---

## 第三章：distilabel 数据合成框架

### 3.1 什么是 distilabel？

[distilabel](https://github.com/argilla-io/distilabel) 是 Argilla 团队开发的 **AI 数据合成框架**。它把"调用 LLM 生成数据"这件事标准化了，提供了一系列开箱即用的数据处理步骤。

```
传统做法（手动写代码）：
  自己写 prompt → 自己调 API → 自己解析返回 → 自己处理异常
  ↑ 每个项目都要重复造轮子

distilabel 做法（标准化框架）：
  选择预定义的 Task → 传入 LLM → 自动处理一切
  ↑ 像搭积木一样组合
```

### 3.2 本脚本用到的三个核心组件

| 组件 | 类名 | 作用 | 输入 | 输出 |
|------|------|------|------|------|
| LLM 封装 | `TransformersLLM` | 封装本地 HF 模型 | 对话消息列表 | 生成的文本 |
| LLM 封装 | `OpenAILLM` | 封装 OpenAI 兼容 API | 对话消息列表 | 生成的文本 |
| 质量进化 | `EvolQuality` | 让 LLM 改写润色回答 | 问题+回答 | 进化后的回答 |
| 多维评分 | `UltraFeedback` | 让 LLM 多维度打分 | 问题+多个回答 | 分数+理由 |

### 3.3 TransformersLLM vs OpenAILLM

```python
# TransformersLLM：加载本地模型，在你的 GPU 上运行
# 优点：免费、数据不出服务器
# 缺点：需要 GPU 显存，速度取决于硬件
llm_a = TransformersLLM(model="path/to/local/model")
llm_a.load()  # 加载模型权重到 GPU

# OpenAILLM：调用远程 API（支持任何兼容 OpenAI 格式的 API）
# 优点：不占本地 GPU，可以用更强的模型
# 缺点：需要付费，数据会发送到远程服务器
judge_llm = OpenAILLM(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key="your-key"
)
```

本脚本的设计思路：
- **生成模型**用本地模型（TransformersLLM）→ 免费，且能控制模型差异
- **裁判模型**用 API（OpenAILLM）→ 可以用更强的模型来打分，质量更高

---

## 第四章：双模型生成策略

### 4.1 为什么用两个不同的模型？

这是本脚本最巧妙的设计之一。用两个**能力有差异**的模型来生成回答：

```
模型 A：Qwen3-8B-Instruct
  - 阿里的 Qwen 系列，中文能力强
  - 指令遵循能力好
  - 通常生成质量较高的回答

模型 B：Meta-Llama-3-8B-Instruct-Chinese
  - Meta 的 Llama 系列，经过中文适配
  - 中文能力相对弱一些
  - 回答质量可能不如 Qwen
```

这样做的好处：

| 方案 | 做法 | 问题 |
|------|------|------|
| 同一模型 + 不同温度 | 一个模型用 temp=0.3 和 temp=1.0 | 风格差异大，但质量差异不明显 |
| 同一模型 + 不同 prompt | 一个模型用好 prompt 和差 prompt | 人为制造差异，不够自然 |
| **两个不同模型** | 能力不同的模型自然产生差异 | ✅ 最自然、最真实的质量差异 |

### 4.2 生成过程详解

```python
# 将问题包装成对话格式
def chat(msg):
    return [{"role": "user", "content": msg}]

# 生成参数
GEN_KWARGS = dict(
    max_new_tokens=256,   # 限制回答长度，避免过长
    temperature=0.7       # 适度的随机性
)

# 两个模型分别生成
gen_a = llm_a.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]
gen_b = llm_b.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]
```

**关于 temperature（温度参数）：**

```
temperature = 0.0  →  每次生成完全相同（贪心解码）
temperature = 0.7  →  有一定随机性，但大体稳定（本脚本使用）
temperature = 1.0  →  标准随机采样
temperature > 1.0  →  非常随机，可能产生不连贯的文本
```

选择 0.7 是因为：太低会让两个模型的回答过于模板化，太高会产生低质量的随机文本。

---

## 第五章：EvolQuality —— 回答质量进化

### 5.1 什么是 EvolQuality？

EvolQuality 来自论文 [WizardLM](https://arxiv.org/abs/2304.12244) 的思想，核心idea是：

> **让一个强大的 LLM 对现有回答进行改写和润色，使其质量更高。**

```
原始回答（可能有瑕疵）：
  "北京是中国首都，有很多景点。"

经过 EvolQuality 进化后：
  "北京是中华人民共和国的首都，拥有3000多年的建城史。
   作为六朝古都，这里汇聚了故宫、天坛、颐和园等世界
   文化遗产，同时也是现代中国的政治、文化中心。"
```

### 5.2 为什么需要这一步？

```
没有 EvolQuality 的情况：
  模型A回答：3.5 分
  模型B回答：3.0 分
  差距：0.5 分 → 裁判模型很难准确区分

有 EvolQuality 的情况：
  模型A回答进化后：4.5 分（好的更好了）
  模型B回答进化后：3.5 分（差的也改善了，但幅度小）
  差距：1.0 分 → 裁判模型更容易准确打分
```

EvolQuality 的作用就像"放大镜"——它放大了两个回答之间的质量差异，让后续的打分更加准确。

### 5.3 代码解读

```python
# 初始化：指定裁判模型和进化轮数
evol_quality = EvolQuality(llm=judge_llm, num_evolutions=1)
evol_quality.load()

# 使用：传入问题和回答，得到进化后的回答
evo_a = evol_quality.process([{"instruction": q, "response": gen_a}])
evo_a = next(evo_a)[0]["evolved_response"]
```

**参数说明：**
- `num_evolutions=1`：进化 1 轮。每多一轮，裁判模型会在上一轮的基础上继续改写
- 轮数越多，回答质量越高，但 API 调用次数也成倍增加（成本和时间）

---

## 第六章：UltraFeedback —— 多维度评分

### 6.1 什么是 UltraFeedback？

UltraFeedback 来自论文 [UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377)，是一种让 AI 从**多个维度**对回答进行评分的方法。

与简单的"哪个更好"二选一不同，UltraFeedback 会从以下维度分别打分：

```
┌─────────────────────────────────────────────────────────────┐
│                UltraFeedback 评分维度                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📚 Helpfulness（有帮助性）                                   │
│     回答是否真正解决了用户的问题？                              │
│     评分：1（完全没用）→ 5（非常有帮助）                       │
│                                                               │
│  🎯 Honesty（诚实性）                                        │
│     回答是否承认不确定性？是否有编造内容？                      │
│     评分：1（严重编造）→ 5（完全诚实）                         │
│                                                               │
│  📋 Instruction-following（指令遵循）                         │
│     回答是否按照用户的要求来？格式、长度是否符合？              │
��     评分：1（完全偏题）→ 5（完美遵循）                         │
│                                                               │
│  ✅ Truthfulness（真实性）                                    │
│     回答中的事实是否正确？                                     │
│     评分：1（严重错误）→ 5（完全准确）                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 评分过程

```python
# 将两个进化后的回答一起送给裁判模型
feedback = ultrafeedback.process([{
    "instruction": q,              # 原始问题
    "generations": [evo_a, evo_b]  # 两个待评分的回答
}])
feedback = next(feedback)

# 获取评分结果
ratings    = feedback[0]["ratings"]      # 如 [4.5, 2.0]
rationales = feedback[0]["rationales"]   # 评分理由
```

**裁判模型内部的工作流程：**

```
输入给裁判模型的 prompt（简化版）：

"请对以下两个回答进行评分（1-5分），从有帮助性、诚实性、
 指令遵循、真实性四个维度评价，并给出理由。

 问题：如何提高学习效率？

 回答1：可以尝试番茄工作法...
 回答2：学习效率的提高需要从多个维度考虑...

 请按以下格式输出：
 回答1评分：X
 回答2评分：X
 理由：..."
```

### 6.3 为什么用 AI 裁判而不是人工？

| 方面 | 人工标注 | AI 裁判（本脚本） |
|------|----------|-------------------|
| 成本 | 每条数据 $0.5-2 | 每条数据 < $0.01 |
| 速度 | 每人每小时 50-100 条 | 每分钟数百条 |
| 一致性 | 不同标注员标准不同 | 同一模型标准一致 |
| 质量 | 最高（人类判断） | 较高（取决于裁判模型能力） |
| 可扩展性 | 难以大规模扩展 | 轻松扩展到百万级 |

实际上，研究表明强大的 AI 裁判（如 GPT-4、DeepSeek）的评分与人类标注的一致性已经非常高。

---

## 第七章：代码逐行对照解读

### 7.1 整体代码结构

```python
# ┌─────────────────────────────────────────────────┐
# │ 第一部分：导入依赖                                │
# │ 第二部分：路径和参数配置                          │
# │ 第三部分：加载问题数据集                          │
# │ 第四部分：加载生成模型（两个）                    │
# │ 第五部分：加载裁判模型（DeepSeek API）            │
# │ 第六部分：初始化数据处理步骤                      │
# │ 第七部分：辅助函数和生成参数                      │
# │ 第八部分：核心循环（生成→进化→打分）              │
# │ 第九部分：保存偏好数据                            │
# └─────────────────────────────────────────────────┘
```

### 7.2 导入部分

```python
from distilabel.llms import OpenAILLM        # API 模型封装
from distilabel.llms import TransformersLLM   # 本地模型封装
from distilabel.steps.tasks import (
    TextGeneration,   # 基础文本生成（未直接使用）
    EvolQuality,      # 回答质量进化
    UltraFeedback     # 多维度评分
)
```

### 7.3 模型加载

```python
# 本地生成模型（需要 GPU 显存）
llm_a = TransformersLLM(model=MODEL_A_PATH)
llm_a.load()  # 加载到 GPU

# API 裁判模型（不占本地资源）
judge_llm = OpenAILLM(
    model="deepseek-chat",
    base_url=r"https://api.deepseek.com/v1",
    api_key=DS_KEY
)
judge_llm.load()  # 初始化 API 客户端
```

**注意：** 两个本地模型（Qwen 8B + Llama 8B）同时加载需要约 32GB 显存。如果显存不够，可以：
1. 使用量化模型（4bit/8bit）
2. 先加载一个模型生成完再加载另一个
3. 减小 `TRAIN_SUBSET` 先用少量数据测试

### 7.4 核心循环

```python
for q in tqdm(questions, desc="Generating/Rewrite/Scoring"):
    # 步骤1：双模型生成
    gen_a = llm_a.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]
    gen_b = llm_b.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]

    # 步骤2：质量进化
    evo_a = evol_quality.process([{"instruction": q, "response": gen_a}])
    evo_a = next(evo_a)[0]["evolved_response"]
    # ... evo_b 同理

    # 步骤3：打分
    feedback = ultrafeedback.process([{
        "instruction": q,
        "generations": [evo_a, evo_b]
    }])
    # 提取分数和理由
    ratings = feedback[0]["ratings"]
    rationales = feedback[0]["rationales"]
```

**数据流图：**

```
问题 q
  │
  ├──→ llm_a.generate() ──→ gen_a ──→ evol_quality ──→ evo_a ──┐
  │                                                               │
  └──→ llm_b.generate() ──→ gen_b ──→ evol_quality ──→ evo_b ──┤
                                                                  │
                                                                  ▼
                                                          ultrafeedback
                                                                  │
                                                                  ▼
                                                    (ratings, rationales)
```

### 7.5 数据保存

```python
df = pd.DataFrame(records)
df.to_parquet(os.path.join(DATA_PATH, "dpo_data.parquet"), index=False)
```

保存为 parquet 格式的好处：
- 列式存储，读取特定列时速度极快
- 自带压缩，文件体积小
- HuggingFace `datasets` 库原生支持加载

---

## 第八章：数据质量与常见问题

### 8.1 如何判断合成数据的质量？

生成完数据后，建议做以下检查：

```python
import pandas as pd

df = pd.read_parquet("dpo_data.parquet")

# 1. 检查评分分布
print(df["rating_j"].describe())
print(df["rating_k"].describe())

# 2. 检查评分差异
df["rating_diff"] = df["rating_j"] - df["rating_k"]
print(df["rating_diff"].describe())

# 3. 抽样查看具体内容
sample = df.sample(5)
for _, row in sample.iterrows():
    print(f"问题: {row['question'][:50]}...")
    print(f"评分: {row['rating_j']} vs {row['rating_k']}")
    print("---")
```

**健康的数据应该是：**
- 评分不全是一样的（说明裁判模型确实在区分）
- 评分差异有一定分布（不是所有都是 5 vs 1）
- 抽样查看时，高分回答确实比低分回答好

### 8.2 数据量建议

| 用途 | 建议数据量 | 说明 |
|------|-----------|------|
| 快速验证流程 | 20-50 条 | 本脚本默认 `TRAIN_SUBSET=20` |
| 小规模实验 | 500-2000 条 | 能看到 DPO 训练效果 |
| 正式训练 | 5000-50000 条 | 业界常见规模 |

### 8.3 成本估算

假设使用 DeepSeek API（价格约 ¥1/百万 token）：

```
每条数据的 API 调用：
  - EvolQuality × 2 次（两个回答各进化一次）
  - UltraFeedback × 1 次

每次调用约 500-1000 token

20 条数据：约 ¥0.1（几乎免费）
2000 条数据：约 ¥10
20000 条数据：约 ¥100
```

---

## 附录：常见问题

### Q1：为什么选 DeepSeek 作为裁判模型？

DeepSeek 是目前性价比最高的中文 AI API 之一：
- 中文理解能力强
- API 价格低廉
- 兼容 OpenAI API 格式，代码改动最小

你也可以替换为其他 API（如 GPT-4、Claude），只需修改 `OpenAILLM` 的参数。

### Q2：两个生成模型可以换成其他的吗？

当然可以。关键是两个模型的**能力要有差异**，这样才能自然产生质量不同的回答。比如：
- 7B vs 1.5B（大小差异）
- Instruct vs Base（微调差异）
- 中文强模型 vs 中文弱模型（语言能力差异）

### Q3：EvolQuality 的进化轮数设多少合适？

- `num_evolutions=1`：推荐，性价比最高
- `num_evolutions=2-3`：质量更高，但 API 成本翻倍
- `num_evolutions>3`：收益递减，不推荐

### Q4：生成的数据如何用于 DPO 训练？

下一篇指南《DPO 直接偏好优化完全指南》会详细讲解。简单来说：

```python
# 在 DPO 训练脚本中，会这样使用偏好数据：
# 1. 加载 parquet 文件
# 2. 根据 rating 确定 chosen 和 rejected
# 3. 构造 DPO 训练所需的三元组：(prompt, chosen, rejected)
# 4. 用 DPOTrainer 进行训练
```

### Q5：`TRAIN_SUBSET = 20` 是什么意思？

这是为了**快速调试**而设置的。只取前 20 条数据运行，几分钟就能跑完整个流程。确认代码没问题后，再把这个值改大（或设为 0 使用全部数据）进行正式的数据合成。

---

> 📌 **下一步学习：** 数据合成完成后，请继续阅读《DPO 直接偏好优化完全指南》，学习如何用这些偏好数据训练模型。
