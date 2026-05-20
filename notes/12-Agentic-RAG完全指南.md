# 🤖 Agentic RAG 完全指南 —— 让大模型学会"自己查资料"

> 本文档配合 `qwen_agentic_rag.py` 与 `utils/utils.py::SemanticRetriever` 代码，
> 系统讲解从训练完成的大模型到"能用、好用"的智能体应用之间的关键一跃。
> 假设你已经完成了从预训练到知识蒸馏的全部 11 篇学习，这是大模型"落地"的第一站。

---

## 📋 目录

- [第零章：从训练到应用的最后一公里](#第零章从训练到应用的最后一公里)
- [第一章：为什么大模型需要 RAG？](#第一章为什么大模型需要-rag)
- [第二章：传统 RAG 的工作流程](#第二章传统-rag-的工作流程)
- [第三章：从 RAG 到 Agentic RAG](#第三章从-rag-到-agentic-rag)
- [第四章：核心组件 1 —— Embedding 模型](#第四章核心组件-1--embedding-模型)
- [第五章：核心组件 2 —— 向量数据库 FAISS](#第五章核心组件-2--向量数据库-faiss)
- [第六章：核心组件 3 —— 文档分块](#第六章核心组件-3--文档分块)
- [第七章：核心组件 4 —— smolagents 智能体](#第七章核心组件-4--smolagents-智能体)
- [第八章：代码逐段拆解](#第八章代码逐段拆解)
- [第九章：调参与扩展](#第九章调参与扩展)
- [附录：常见问题](#附录常见问题)

---

## 第零章：从训练到应用的最后一公里

```
┌───────────────────────────────────────────────────────────────────────┐
│                    大模型训练 + 应用完整流程                            │
│                                                                       │
│   ① 预训练（Pre-Training）          ✅ 第 1 篇                       │
│   ② 网络结构（Qwen 架构）           ✅ 第 2 篇                       │
│   ③ 续训练（Continue PT）           ✅ 第 3 篇                       │
│   ④ 监督微调（SFT）                 ✅ 第 4 篇                       │
│   ⑤ 大模型评估                       ✅ 第 5 篇                       │
│   ⑥ DPO 数据合成                    ✅ 第 6 篇                       │
│   ⑦ DPO 偏好对齐                    ✅ 第 7 篇                       │
│   ⑧ 奖励模型 RM                      ✅ 第 8 篇                       │
│   ⑨ PPO 强化学习                    ✅ 第 9 篇                       │
│   ⑩ GRPO                             ✅ 第 10 篇                      │
│   ⑪ 知识蒸馏                         ✅ 第 11 篇                      │
│  ─────── 训练完成，模型上线 ───────                                  │
│   ⑫ Agentic RAG（智能体 + 检索增强）  ← 📍 我们在这里                │
│   ⑬ 部署 / 评测胜率（待补）                                           │
└───────────────────────────────────────────────────────────────────────┘
```

**类比：**

| 训练阶段 | Agentic RAG |
|---|---|
| 培养一个"博学的学者" | 给学者配上"图书馆 + 网络"和"调研助理" |
| 模型参数里压缩了知识 | 让模型学会"什么时候去查、查什么、怎么用" |
| 知识冻结在训练那一刻 | 知识可以实时更新 |

---

## 第一章：为什么大模型需要 RAG？

### 1.1 训练完的模型有 4 个硬伤

```
┌─────────────────────────────────────────────────────────���───┐
│  硬伤一：知识截止（Knowledge Cutoff）                        │
│    模型只知道训练那一刻之前的世界                             │
│    例：训练截止 2024-01，问"2025 年奥运会冠军"它不知道       │
│                                                             │
│  硬伤二：幻觉（Hallucination）                              │
│    模型会"编造"看起来很对但其实是错的内容                    │
│    例：编一个不存在的论文标题、错引用一个公司财报数据         │
│                                                             │
│  硬伤三：领域局限                                            │
│    通用预训练数据中没怎么覆盖的领域（公司内部文档、法律条款） │
│                                                             │
│  硬伤四：可追溯性差                                          │
│    模型给出答案，但用户没法验证"这是哪里说的"                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 RAG 是怎么解决的？

**RAG = Retrieval-Augmented Generation（检索增强生成）**

核心思想就一句话：**让模型在回答问题前，先从外部资料库里"翻书"**。

```
不用 RAG：
  Q: 2025 年公司的主营业务是？
  → LLM 直接回答（基于参数里的旧知识，可能错）

用 RAG：
  Q: 2025 年公司的主营业务是？
  → ① 从公司年报库检索相关段落
  → ② 把段落 + 问题一起塞给 LLM
  → ③ LLM 基于资料生成答案（可靠、可引用）
```

### 1.3 RAG 解决了哪些痛点？

| 痛点 | RAG 怎么解决 |
|---|---|
| 知识截止 | 资料库可以随时更新，模型参数不动 |
| 幻觉 | 答案锚定在检索到的真实文本上 |
| 领域局限 | 把领域文档塞进资料库即可 |
| 可追溯 | 检索结果里带着 metadata（url、文件名、页码） |

---

## 第二章：传统 RAG 的工作流程

### 2.1 离线阶段（建索引，做一次）

```
┌────────────────────────────────────────────────────────────┐
│   原始文档（PDF / 网页 / Markdown / 数据库）                │
│              │                                             │
│              ▼                                             │
│   清洗 + 分块（Chunking）                                  │
│              │  每块 200~500 token                         │
│              ▼                                             │
│   Embedding 模型（如 gte-small-zh）                        │
│              │  每块 → 一个稠密向量（512 维）              │
│              ▼                                             │
│   向量数据库（FAISS / Milvus / Pinecone）                  │
│              │  存：向量 + 原文 + metadata                 │
│              ▼                                             │
│   持久化到磁盘                                             │
└────────────────────────────────────────────────────────────┘
```

### 2.2 在线阶段（每次提问都做）

```
用户提问
    │
    ▼
Embedding 模型 ─────► query 向量
    │
    ▼
向量数据库（top-k 最近邻搜索）
    │
    ▼
检索到 k 个最相关的文档块
    │
    ▼
拼接成 prompt：
   "已知资料：{doc1}\n{doc2}\n...\n请回答：{query}"
    │
    ▼
LLM 生成答案
    │
    ▼
返回给用户（可附带来源）
```

### 2.3 传统 RAG 的瓶颈

```
痛点 1：检索是"一锤子买卖"
   query 写得不好 → 检索到无关文档 → LLM 也没辙

痛点 2：top-k 是固定数量
   有的问题 1 个文档就够，有的需要 10 个，固定 k=5 容易过/欠

痛点 3：单一信息源
   只查本地知识库，不会主动去网上搜

痛点 4：没法多步推理
   "X 公司的 CEO 上一份工作是什么？"
   → 需要先查"X 公司 CEO 是谁" → 再查"那个人的简历"
   → 传统 RAG 只能查一次
```

---

## 第三章：从 RAG 到 Agentic RAG

### 3.1 一句话区别

> **传统 RAG**：检索是流程的一部分，写死的。
> **Agentic RAG**：检索是 Agent 的一个工具，由 LLM 自己决定要不要调、怎么调、调几次。

### 3.2 流程对比图

```
┌─ 传统 RAG ────────────────────────────────────────────┐
│                                                       │
│  Query ──► 检索 ──► 拼 prompt ──► LLM ──► 回答        │
│            (固定)   (固定)        (固定)              │
│                                                       │
└───────────────────────────────────────────────────────┘

┌─ Agentic RAG ─────────────────────────────────────────┐
│                                                       │
│  Query ──► LLM (Agent)                                │
│              │                                        │
│              ├── [思考] 需要查资料吗？查什么？         │
│              │                                        │
│              ├──► 工具 1：本地检索                     │
│              │     ↑↓                                 │
│              ├──► 工具 2：网络搜索                     │
│              │     ↑↓                                 │
│              ├──► 工具 3：调用 API / 计算器            │
│              │     ↑↓                                 │
│              ├── [思考] 信息够了吗？                  │
│              │                                        │
│              └──► 最终回答                            │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### 3.3 Agentic RAG 的"自主性"体现在哪里？

```
能力 1：自主决策
  "今天天气怎么样？" → 走 WebSearch（时效性）
  "牛顿第二定律是什么？" → 走本地知识库（百科）
  "1+1=?" → 直接回答（无需工具）

能力 2：多轮调用
  Q: "查 A 公司在 B 国的销售额"
  Step 1: 检索 → 找到 A 公司 2024 财报
  Step 2: 发现财报没分国家，重新检索 → "A 公司 B 国 销售"
  Step 3: 综合多次结果给出答案

能力 3：错误自纠
  Step 1 检索结果不相关 → Agent 看到后改写 query 重检索
  这是普通 RAG 做不到的

能力 4：多工具组合
  "比较 A 和 B 公司去年净利润"
  → 检索 A 公司 → 检索 B 公司 → 用计算器算差值
```

### 3.4 ReAct 推理模式（Agent 的"思考-行动"循环）

```
ReAct = Reason + Act
就是 Agent 一直在循环：
  Thought（想想该干啥） → Action（调工具）
                       → Observation（看工具返回啥）
                       → Thought ...
                       → Final Answer

样例 trace：
┌────────────────────────────────────────────────────┐
│ User: "Qwen3 模型的参数量是多少？"                  │
│                                                    │
│ Thought: 这是较新的模型信息，本地百科可能没有        │
│ Action:  webserach_tool(query="Qwen3 parameters")  │
│ Observation: 搜索结果显示 Qwen3 有 multiple sizes  │
│                                                    │
│ Thought: 还需要更详细的尺寸列表                     │
│ Action:  semantic_retriever(query="Qwen3 模型规格") │
│ Observation: 检索到具体的参数规格表                 │
│                                                    │
│ Thought: 信息足够了                                 │
│ Final Answer: Qwen3 系列包括 0.6B/1.7B/4B/8B...    │
└────────────────────────────────────────────────────┘
```

---

## 第四章：核心组件 1 —— Embedding 模型

### 4.1 什么是 Embedding？

> **Embedding = 把文本变成向量（一串数字）**

```
"我喜欢猫" ──► [0.21, -0.13, 0.88, ..., 0.05]   (512 维)
"我爱小猫" ──► [0.19, -0.10, 0.85, ..., 0.07]   (512 维)
"今天股价跌了" ──► [-0.42, 0.61, -0.23, ..., 0.91] (512 维)
```

关键性质：**语义相近的文本，向量也相近**（余弦相似度高）。

| 文本对 | 余弦相似度 |
|---|---|
| "我喜欢猫" vs "我爱小猫" | 0.92（很高） |
| "我喜欢猫" vs "今天股价跌了" | 0.05（很低） |

### 4.2 本项目用的 Embedding 模型：gte-small-zh

```
模型：阿里通义实验室开源的 General Text Embedding (GTE) 中文小型版
路径：/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/gte-small-zh

特点：
  - 中文优化（双语种 zh-en）
  - 体积小（~30MB），CPU 也能跑
  - 输出维度 512
  - 最大输入 512 token
```

### 4.3 Embedding 是怎么训出来的？

简单说：**对比学习**（Contrastive Learning）。

```
训练目标：
  让正样本对（语义相近的句子）的向量距离 → 小
  让负样本对（语义无关的句子）的向量距离 → 大

训练数据：
  - 同一段文档的两段（正对）
  - 平行翻译（"the cat" 和 "猫"，跨语言正对）
  - 问答对（query 和 answer 是正对）
  - 同 batch 内其他句子（负对）

损失函数：InfoNCE Loss（对比学习经典损失）
```

### 4.4 在 RAG 里的位置

```
离线建库：
  for chunk in 所有文档块:
      vec = embedding_model.encode(chunk)
      vector_db.add(vec, chunk)

在线检索：
  query_vec = embedding_model.encode(user_query)
  top_k_chunks = vector_db.search(query_vec, k=7)
```

---

## 第五章：核心组件 2 —— 向量数据库 FAISS

### 5.1 为什么需要专门的向量数据库？

```
朴素方法：
  for vec in 所有向量:
      sim = cosine(query_vec, vec)
  返回最大的 k 个

  问题：
    - 100 万文档 × 512 维 = 5亿次乘加
    - 每次查询都全扫一遍 → 太慢
```

### 5.2 FAISS 简介

**FAISS = Facebook AI Similarity Search**

Meta 开源的向量相似度搜索库，C++ 实现 + Python 绑定，超级快。

提供多种索引类型：

| 索引类型 | 复杂度 | 精度 | 用法 |
|---|---|---|---|
| `IndexFlatL2` | O(N·d) | 100% | 数据量小（< 100 万） |
| `IndexIVFFlat` | O(√N·d) | ~99% | 中等规模（百万级） |
| `IndexHNSWFlat` | O(log N) | ~98% | 大规模（千万级） |
| `IndexPQ` | 极快 | ~90% | 超大规模 + 内存敏感 |

**本项目用的是默认的 Flat（暴力搜索），因为数据量不大。**

### 5.3 余弦相似度 vs L2 距离

```
余弦相似度（COSINE）：
  cos(θ) = (a·b) / (|a||b|)
  - 范围 [-1, 1]，1 表示同向，-1 表示反向
  - 只看方向，不看长度
  - 适合文本：未归一化的 embedding 模长不同也能正确比较

L2 距离（欧氏距离）：
  d = sqrt(sum((a_i - b_i)^2))
  - 范围 [0, +∞)
  - 既看方向也看长度
  - 如果 embedding 已归一化，L2 距离和 COSINE 等价

本项目选 COSINE，更通用。
```

### 5.4 FAISS 在 LangChain 里的封装

```python
# 建库
vectordb = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)

# 持久化
vectordb.save_local(INDEX_DIR)
# 生成两个文件：
#   index.faiss  ← 向量的二进制索引
#   index.pkl    ← Document 对象（原文 + metadata）

# 加载
vectordb = FAISS.load_local(
    INDEX_DIR,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,  # pkl 文件需要显式允许
)

# 搜索
docs = vectordb.similarity_search(query, k=7)
# 返回 List[Document]，每个 Document 有 .page_content 和 .metadata
```

---

## 第六章：核心组件 3 —— 文档分块

### 6.1 为什么必须分块？

```
原因 1：Embedding 模型有最大输入长度
  gte-small-zh 最大 512 token
  一篇 1 万字的文章硬塞 → 截断 → 后半部分丢失

原因 2：检索粒度
  整篇文档 → 一个向量 → 表达一个"主题"
  小段文本 → 一个向量 → 精确表达一个"知识点"
  做 QA 时显然要后者：用户问的是某个知识点

原因 3：上下文窗口
  LLM 上下文有限，不可能把整本书塞进 prompt
  检索小块就刚好填进 prompt
```

### 6.2 分块的关键参数

```
chunk_size：每块的最大 token 数
  - 太小（< 100）：上下文不足，单块不能独立理解
  - 太大（> 1000）：检索精度下降，多个知识点混在一起
  - 经验值：200~500

chunk_overlap：相邻块的重叠 token 数
  - 防止信息在边界处被切断
  - 经验值：chunk_size 的 10~20%
  - 例：chunk_size=200, overlap=20

separators：分隔符优先级（递归分割器）
  默认顺序：["\n\n", "\n", ". ", " ", ""]
  解读：
    先按 "\n\n"（段落）切，每段如果还太大
    再按 "\n"（行）切，行还太大
    再按 ". "（句子）切，句子还太大
    再按 " "（词）切，最后才按字符切
  好处：尽量保持语义完整
```

### 6.3 RecursiveCharacterTextSplitter 工作原理

```
输入：一篇 1500 token 的文章
chunk_size=500, overlap=50

Step 1：尝试按 "\n\n" 切，得到 3 个段落 [800, 400, 300]
Step 2：第一段 800 还太大 → 在该段内按 "\n" 切 → [400, 400]
Step 3：所有段都 ≤ 500 ✓
Step 4：合并相邻小段（前提是合并后 ≤ chunk_size）
        最终 [400, 400, 400, 300]
Step 5：相邻块之间生成 50 token 的 overlap
        [块A: 0-400]
        [块B: 350-750]  ← 与块A重叠 50
        [块C: 700-1100]
        ...
```

### 6.4 分块的去重

代码里这一段：

```python
unique_texts = {}
for doc in source_docs:
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)
```

**为什么要去重？**

知识库里经常有重复内容（比如百科页面里的"参见"、引用同一段定义的多个词条）。
如果不去重：
- 索引大小膨胀
- 检索时同一个内容占用多个 top-k 名额，挤掉真正多样化的结果

---

## 第七章：核心组件 4 —— smolagents 智能体

### 7.1 smolagents 是什么？

> HuggingFace 出的 **轻量级 Agent 框架**，对标 LangChain Agent / LlamaIndex Agent。
> 设计哲学：少抽象、好理解、易扩展。

### 7.2 三个核心概念

#### 概念 1：Model（大脑）

```python
model = OpenAIServerModel(
    model_id="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=DS_KEY,
    flatten_messages_as_text=True,
)
```

可以是任何兼容 OpenAI API 的模型：
- DeepSeek（本项目用的）
- OpenAI GPT-4
- 本地 vLLM 部署的 Qwen
- Anthropic Claude（通过适配器）

#### 概念 2：Tool（工具）

工具就是 Agent 可以"调用"的函数。每个工具必须有：

| 字段 | 作用 |
|---|---|
| `name` | 工具名（snake_case） |
| `description` | 工具说明（**Agent 据此判断要不要调用**，最关键） |
| `inputs` | 参数定义（JSON Schema 格式） |
| `output_type` | 返回值类型 |
| `forward()` | 实际执行函数 |

#### 概念 3：Agent（决策者）

```python
agent = ToolCallingAgent(tools=[retriever_tool, webserach_tool], model=model)
agent.run("你的问题")
```

`ToolCallingAgent` 在内部自动维护 ReAct 循环。

### 7.3 自定义工具的标准模板

来看本项目的 `SemanticRetriever`：

```python
class SemanticRetriever(Tool):
    name: str = "semantic_retriever"
    description: str = (
        "Return the top-k documents whose embeddings are most similar to the "
        "input query. The query should be phrased affirmatively, not as a question."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The search phrase, expressed in affirmative form and semantically "
                "aligned with the target documents."
            ),
        }
    }
    output_type = "string"

    def __init__(self, vectordb, *, top_k=7, **kwargs):
        super().__init__(**kwargs)
        self._db = vectordb
        self._top_k = top_k

    def forward(self, query: str) -> str:
        docs = self._db.similarity_search(query, k=self._top_k)
        formatted = [
            f"===== Document {idx} =====\n{doc.page_content}"
            for idx, doc in enumerate(docs)
        ]
        return "\nRetrieved documents:\n" + "\n".join(formatted)
```

### 7.4 ⚠️ Tool description 是关键

Agent 完全靠 description 判断"要不要用、什么时候用"。写得好不好直接决定 Agent 表现。

```
❌ 坏例子：
   description = "搜索"

❌ 也不够好：
   description = "Returns relevant documents"

✅ 好例子：
   description = (
     "Return the top-k documents whose embeddings are most similar to the "
     "input query. The query should be phrased affirmatively, not as a question."
   )
   ↑ 说清楚：
     - 干啥的（返回相似文档）
     - 输入约束（要陈述句而非疑问句，因为 Embedding 模型对疑问句敏感）
```

### 7.5 ToolCallingAgent 的内部循环

```
def run(query):
    messages = [system_prompt, user_query]
    for step in range(max_steps):
        # 1. 让 LLM 看着工具列表 + 历史对话，决定下一步
        response = model.chat(messages, tools=self.tools)

        # 2. 解析返回
        if response 包含 tool_call:
            # 3. 执行工具
            tool_name, tool_args = parse(response)
            result = self.tools[tool_name].forward(**tool_args)
            # 4. 把工具结果塞回 messages，进入下一轮
            messages.append({"role": "tool", "content": result})
        else:
            # LLM 输出了最终答案
            return response.content
```

---

## 第八章：代码逐段拆解

### 8.1 项目结构

```
qwen_agentic_rag.py        ← 主脚本
utils/utils.py             ← 包含 SemanticRetriever 类（自定义工具）
data/rag/                  ← 知识库 parquet 文件
data/faiss_rag/            ← FAISS 索引（首次运行后生成）
data/gte-small-zh/         ← Embedding 模型
```

### 8.2 第一部分：配置参数

```python
DS_KEY = "sk-xxxxxxxx"
DATA_PATH = "data/rag"
TMP_PATH = "/archive/share/cql/aaa/tmp"
EMBEDDING_MODEL_PATH = ".../gte-small-zh"
INDEX_DIR = "data/faiss_rag"
SUBSET = -1
```

**注意点：**
- `DS_KEY` 写死在代码里 ❌ 真实生产环境用环境变量
- `INDEX_DIR` 复用机制可大幅省时（首次构建慢，之后秒加载）
- `SUBSET` 调试时设为正数（例如 100）快速跑通

### 8.3 第二、三部分：建/加载 FAISS 索引

```python
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

if Path(INDEX_DIR).exists():
    # 复用：直接加载，秒级
    vectordb = FAISS.load_local(INDEX_DIR, embeddings=embedding_model,
                                allow_dangerous_deserialization=True)
else:
    # 首次构建：完整流程
    knowledge_base = datasets.load_dataset("parquet", ...)
    source_docs = [Document(page_content=..., metadata=...) for ...]
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(...)
    docs_processed = [...]  # 分块 + 去重
    vectordb = FAISS.from_documents(docs_processed, embedding_model,
                                    distance_strategy=DistanceStrategy.COSINE)
    vectordb.save_local(INDEX_DIR)
```

**性能要点：**

| 步骤 | 时间复杂度 | 备注 |
|---|---|---|
| 加载 parquet | O(N) | I/O 瓶颈 |
| 分块 | O(N · L)，L 为文档长度 | CPU 瓶颈 |
| Embedding | O(M · d)，M 为块数 | **GPU 瓶颈，最慢** |
| 建 FAISS Flat 索引 | O(M · d) | 内存瓶颈 |
| 检索 top-k | O(M · d) per query | 数��大时换 IVF/HNSW |

### 8.4 第四部分：配置 LLM

```python
model = OpenAIServerModel(
    model_id="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=DS_KEY,
    flatten_messages_as_text=True,
)
```

**为什么选 DeepSeek？**
- 中文好
- 函数调用（tool calling）能力强
- API 便宜（远低于 GPT-4）
- 兼容 OpenAI 协议，无需改代码

**`flatten_messages_as_text=True` 的作用：**
有些模型对结构化的 tool_call 消息支持不完善，把多轮对话"扁平化"为纯文本传入更稳定。

### 8.5 第五部分：注册工具 + 创建 Agent

```python
retriever_tool = SemanticRetriever(vectordb)  # 本地知识库
webserach_tool = WebSearchTool()              # smolagents 内置网络搜索
agent = ToolCallingAgent(tools=[retriever_tool, webserach_tool], model=model)
```

**WebSearchTool** 默认调用 DuckDuckGo（无需 API Key），如果想用 Google 需要额外配置。

### 8.6 第六部分：交互循环

```python
while True:
    query = input("Enter your query: ")
    agent_output = agent.run(query)
    print("Response:", agent_output)
```

**`agent.run()` 内部完整流程：**

```
1. system prompt 自动注入：
     "You are an agent. You have these tools: [tool_descriptions]..."

2. 用户问题入队

3. 循环 max_steps 次（默认 6 次）：
   a. 调用 DeepSeek，传入 messages + tools 列表
   b. 解析 DeepSeek 返回：
      - 如果是 tool_call → 执行对应工具的 forward()
      - 如果是 final_answer → 退出循环

4. 将工具返回结果塞回 messages，继续下一轮

5. 返回最终答案
```

---

## 第九章：调参与扩展

### 9.1 RAG 性能调优清单

```
检索质量低？
  □ chunk_size 太大 → 减小到 200~300
  □ chunk_overlap 不够 → 增加到 chunk_size 的 15~20%
  □ Embedding 模型太小 → 换 gte-large 或 bge-m3
  □ top_k 太小 → 增加到 10~20，让 LLM 自己筛选
  □ query 改写：让 LLM 把用户原始 query 改成"陈述句"再检索

回答幻觉多？
  □ system_prompt 加约束："只能基于提供的文档回答，没找到就说不知道"
  □ 检索结果带 citation：让 LLM 标注引用来源
  □ 用更大的 LLM（GPT-4 / Claude / DeepSeek-V3）

速度慢？
  □ 索引换 IVF/HNSW
  □ Embedding 模型上 GPU
  □ 缓存常见 query 的检索结果
  □ 减少 max_steps 限制 Agent 多步推理
```

### 9.2 加更多工具

```python
from smolagents import Tool

class CalculatorTool(Tool):
    name = "calculator"
    description = "Perform arithmetic calculations like 2+3, sqrt(16)."
    inputs = {"expression": {"type": "string", "description": "Math expression"}}
    output_type = "string"

    def forward(self, expression: str) -> str:
        return str(eval(expression))   # ⚠️ 生产环境别用 eval，要用安全计算器

agent = ToolCallingAgent(
    tools=[retriever_tool, webserach_tool, CalculatorTool()],
    model=model,
)
```

### 9.3 让 Agent 用本地的 Qwen 而不是 DeepSeek

```python
# 起一个 vLLM 服务器
# vllm serve /path/to/Qwen2.5-7B-Instruct --port 8000

from smolagents import OpenAIServerModel
model = OpenAIServerModel(
    model_id="Qwen2.5-7B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",  # vLLM 不校验
)
```

这样就把"训练完的小模型"接到了 Agentic RAG 系统里 —— 训练 + 应用闭环！

### 9.4 多模态扩展

未来可以加：
- 图片检索工具（用 CLIP embedding）
- 表格 SQL 工具（用 LLM 写 SQL 查数据库）
- 代码执行工具（让 Agent 写 Python 算结果）

---

## 附录：常见问题

### Q1：分块时 chunk_size 写多大合适？

> 经验值：英文 256-512 token，中文 200-400 token。
>
> 原则：
> - 不要超过 Embedding 模型的最大输入（gte-small-zh 是 512）
> - 用户问的是"知识点"还是"概念"？知识点小、概念大
> - 跑一批问题对比 top-1 准确率，找到最优值

### Q2：top_k 怎么选？

> 经验值：5~10。
>
> 太小：相关文档可能没召回
> 太大：噪声多，LLM 容易被无关信息干扰
> 折中：召回 20，再用 reranker 精排到 5

### Q3：什么时候选传统 RAG，什么时候用 Agentic RAG？

| 场景 | 选哪个 |
|---|---|
| 简单单跳问答（"X 是什么"） | 传统 RAG（够用且便宜） |
| 多跳推理（"X 公司 CEO 的母校是哪所"） | Agentic RAG |
| 需要时效性 + 历史知识混合 | Agentic RAG |
| 对延迟敏感（< 1s） | 传统 RAG |
| 信息源单一 | 传统 RAG |
| 信息源多（本地 + 网络 + API） | Agentic RAG |

### Q4：FAISS 和其他向量库（Milvus / Pinecone / Chroma）怎么选？

| 方案 | 适用场景 |
|---|---|
| FAISS | 单机、文档量 < 千万、不需要持久化服务 |
| Chroma | 轻量级、嵌入式、SQLite 风格 |
| Milvus | 大规模分布式、需要分片+复制 |
| Pinecone | 云服务、零运维、按量付费 |
| Weaviate | 开源、图查询能力强 |

本项目选 FAISS：项目内嵌、无需额外服务、性能足够。

### Q5：Embedding 模型怎么选？

| 模型 | ��度 | 大小 | 评价 |
|---|---|---|---|
| gte-small-zh | 512 | 30MB | 本项目用，CPU 友好 |
| bge-small-zh | 512 | 95MB | 中文同梯队 |
| bge-large-zh | 1024 | 1.3GB | 中文最强之一 |
| bge-m3 | 1024 | 2.3GB | 多语言 + 长文本（8K） |
| text-embedding-3-large（OpenAI） | 3072 | API | 最强但贵 |

**建议路线：**
- 原型：gte-small-zh
- 上线：bge-large-zh 或 bge-m3
- 预算充足 + 多语言：OpenAI text-embedding-3-large

### Q6：`allow_dangerous_deserialization=True` 安全吗？

`FAISS.load_local()` 加载的 `index.pkl` 是 Python pickle 文件。
pickle 反序列化可以执行任意代码 —— 如果 pkl 文件来源不可信（比如从网上下的别人的索引），可能被植入恶意代码。

✅ 自己生成、自己加载：安全
❌ 加载陌生人提供的 pkl：危险

### Q7：Agent 一直循环不停止怎么办？

```python
agent = ToolCallingAgent(
    tools=[...],
    model=model,
    max_steps=6,        # 限制最多调几次工具
    verbosity_level=2,  # 打印详细日志方便排查
)
```

常见原因：
- 工具 description 写得不清晰，LLM 不知道何时停止
- 检索结果质量差，LLM 一直觉得"信息不够"
- LLM 本身能力不足，无法判断"已经够了"

### Q8：怎么给 Agentic RAG 加"权威性"约束？

```python
# 自定义 Agent system prompt
agent = ToolCallingAgent(
    tools=[...],
    model=model,
)
# smolagents 内部 prompt 可以通过 prompt_templates 自定义
# 加约束："对于事实性问题，必须先调用 semantic_retriever，否则不能回答"
```

或者干脆：**让你的工具 forward 函数返回时附带强约束**：

```python
def forward(self, query):
    docs = ...
    return (
        f"[INSTRUCTION TO MODEL: 必须基于以下检索结果回答，不得编造]\n\n"
        + "\n".join([d.page_content for d in docs])
    )
```

---

## 🎉 学习清单：Agentic RAG 核心知识

- [x] RAG 解决了大模型的 4 个硬伤：知识截止、幻觉、领域局限、可追溯
- [x] 传统 RAG 流程：分块 → Embedding → 向量库 → 检索 → 拼 prompt → LLM
- [x] Agentic RAG 与传统 RAG 的区别：检索是工具而非固定流程
- [x] Embedding 把语义编码为向量，相近文本向量也相近
- [x] FAISS 是高效向量搜索库，提供 Flat/IVF/HNSW 等索引
- [x] 文档分块的关键参数：chunk_size / chunk_overlap / separators
- [x] smolagents 三要素：Model、Tool、Agent
- [x] 自定义工具的标准结构：name / description / inputs / output_type / forward
- [x] ReAct 循环：Thought → Action → Observation → Thought → ... → Final Answer
- [x] 工具 description 是 Agent 决策的关键
- [x] FAISS 索引可以持久化，加速二次启动

---

## 🚀 下一步学习方向

完成本篇后，你已经把"训练好的模型 → 落地的应用"打通了。继续可以学：

1. **第 13 篇：LLM-as-a-Judge + 模型部署**
   - `qwen_judge.py`：用大模型当裁判评测胜率（Win Rate）
   - `qwen_chat.py` / `qwen_r1_chat.py`：本地推理与思维链对话
   - `utils/merge_peft_adapter.py`：把 LoRA 适配器合并回基座
   
2. **第 14 篇：显存优化实战**（可选）
   - `qwen_mem.py`：CUDA Memory Summary 解读
   - 梯度累积、混合精度、DeepSpeed ZeRO 阶段差异

3. **进阶方向**：
   - 把训练好的 Qwen 接入 Agentic RAG（vLLM 部署 + 替换 DeepSeek）
   - 加更多工具：SQL、Python REPL、API 调用
   - 多 Agent 协作（一个 Agent 检索、一个 Agent 校对、一个 Agent 写作）
