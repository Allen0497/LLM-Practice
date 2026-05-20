"""
Agentic RAG（智能体检索增强生成）脚本
======================================================

RAG（Retrieval-Augmented Generation）是让大模型"查资料再回答"的技术。
Agentic RAG 在此基础上更进一步：让模型成为一个"智能体"（Agent），
能够自主决定何时查资料、查什么、是否需要网络搜索。

传统 RAG vs Agentic RAG：
  传统 RAG：用户提问 → 固定检索 → 拼接上下文 → 模型回答
  Agentic RAG：用户提问 → Agent 思考 → 自主选择工具（检索/搜索）→ 回答
               Agent 可以多次调用工具、组合信息、自我纠错

本脚本的架构：
  ┌─────────────────────────────────────────────────────────┐
  │                    Agentic RAG 系统                       │
  │                                                           │
  │  用户提问 ──→ ToolCallingAgent（DeepSeek 驱动）          │
  │                    │                                      │
  │                    ├── 工具 1：SemanticRetriever          │
  │                    │   （本地知识库语义检索）              │
  │                    │                                      │
  │                    ├── 工具 2：WebSearchTool              │
  │                    │   （互联网搜索）                      │
  │                    │                                      │
  │                    └──→ 综合信息生成回答                  │
  └─────────────────────────────────────────────────────────┘

技术栈：
  - smolagents：HuggingFace 的轻量级 Agent 框架
  - LangChain：文档处理和向量数据库
  - FAISS：Facebook 的高效向量相似度搜索库
  - HuggingFace Embeddings：文本向量化模型（gte-small-zh）
  - DeepSeek API：驱动 Agent 的大语言模��
"""

import os
import datasets
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from smolagents import OpenAIServerModel, WebSearchTool, ToolCallingAgent
from utils.utils import find_files, SemanticRetriever

# ============================================================
# 第一部分：配置参数
# ============================================================
# DeepSeek API Key（用于驱动 Agent 的推理能力）
# ⚠️ 严禁硬编码 key 到代码里，使用环境变量读取：
#   export DEEPSEEK_API_KEY=sk-xxxxxx
# 然后通过 os.environ.get 取出
DS_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DS_KEY:
    raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")
# RAG 知识库数据路径（parquet 格式，包含 content 和 url 字段）
# 本项目使用 baidu_baike 百科数据，content 为词条正文，url 为词条页面地址
DATA_PATH = "data/rag"
# 数据集缓存路径（HuggingFace datasets 解析 parquet 时会在此处缓存中间结果）
TMP_PATH = "/archive/share/cql/aaa/tmp"
# Embedding 模型路径（gte-small-zh：阿里通义实验室开源的中文向量模型）
# 输入：任意中文文本（最长 512 token） / 输出：512 维稠密向量
# 训练目标：让"语义相近的句子向量距离也近"，因此可用于语义检索
EMBEDDING_MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/gte-small-zh"
# FAISS 索引保存目录
# 构建索引耗时（百万级文档需要几十分钟），构建一次落盘后下次直接 load_local 即可
INDEX_DIR = "data/faiss_rag"
# 数据子集大小（-1 表示使用全部数据，正数表示只取前 N 条用于快速调试）
SUBSET = -1

# ============================================================
# 第二部分：加载 Embedding 模型
# ============================================================
# 查找知识库数据文件
directories = ["data"]
data_files = find_files(directories, DATA_PATH)

# 加载 Embedding 模型
# HuggingFaceEmbeddings 封装了 sentence-transformers 的模型
# gte-small-zh 是一个轻量级中文向量模型（~30MB），适合本地部署
# 它将任意长度的中文文本映射为固定维度的向量（如 512 维）
# 语义相近的文本，其向量的余弦相似度也高
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

# ============================================================
# 第三部分：构建或加载 FAISS 向量索引
# ============================================================
# 如果索引已存在，直接加载（避免重复构建，节省时间）
if Path(INDEX_DIR).exists():
    # allow_dangerous_deserialization=True：允许反序列化 pickle 文件
    # FAISS 索引保存时使用了 pickle，加载时需要显式允许
    vectordb = FAISS.load_local(
        INDEX_DIR,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    # ── 步骤 1：加载原始知识库 ──
    # 数据集包含 content（文档内容）和 url（来源链接）两个字段
    knowledge_base = datasets.load_dataset(
        "parquet", data_files=data_files, split="train", cache_dir=TMP_PATH
    )
    if SUBSET > 0:
        knowledge_base = knowledge_base.select(range(SUBSET))

    # 将数据集转换为 LangChain 的 Document 对象
    # Document 包含 page_content（文本内容）和 metadata（元数据）
    # metadata 在检索时会一并返回，可用于：
    #   - 给用户展示来源链接（"答案来自 https://..."）
    #   - 二次过滤（按 url 域名筛选可信源）
    #   - RAG 引用追溯（防止模型幻觉，让用户可验证）
    source_docs = [
        Document(page_content=doc["content"], metadata={"url": doc["url"]})
        for doc in knowledge_base
    ]

    # ── 步骤 2：文档分块（Chunking）──
    # 为什么要分块？
    #   1. Embedding 模型有最大长度限制（通常 512 token）
    #   2. 检索时需要精确定位相关段落，而非整篇文档
    #   3. 小块文本的语义更聚焦，检索精度更高
    #
    # RecursiveCharacterTextSplitter：递归字符分割器
    # "递归"的含义：按优先级依次尝试不同的分隔符
    #   先尝试 "\n\n"（段落分隔）→ 再 "\n"（行分隔）→ 再 "."（句子）→ 再 " "（词）→ 最后 ""（字符）
    #   这样可以尽量保持语义完整性
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH),
        chunk_size=200,        # 每个块最多 200 个 token
        chunk_overlap=20,      # 相邻块之间重叠 20 个 token（避免信息在边界处丢失）
        add_start_index=True,  # 记录每个块在原文中的起始位置
        strip_whitespace=True, # 去除首尾空白
        separators=["\n\n", "\n", ".", " ", ""],  # 分隔符优先级
    )

    # ── 步骤 3：分块并去重 ──
    docs_processed = []
    unique_texts = {}  # 用于去重的字典
    for doc in tqdm(source_docs):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            # ��重：相同内容的块只保留一份
            # 知识库中可能有重复内容，去重可以减少索引大小和检索噪声
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)

    # ── 步骤 4：构建 FAISS 向量索引 ──
    # FAISS（Facebook AI Similarity Search）：高效的向量相似度搜索库
    # 工作原理：
    #   1. 对每个文档块调用 embedding_model 生成向量（这一步最慢，需要前向推理）
    #   2. 将所有向量存入 FAISS 索引（默认是 IndexFlatL2 暴力搜索，N 大时可换 IVF/HNSW）
    #   3. 查询时，将 query 也转为向量，在索引中找最近邻
    #
    # 复杂度对比：
    #   - 暴力搜索（Flat）：O(N·d)，精确但慢
    #   - HNSW 等近似搜索：O(log N)，牺牲少量精度换速度
    #   - 本项目数据量不大，默认 Flat 即可
    #
    # COSINE 距离策略：使用余弦相似度衡量向量间的相似性
    # 余弦相似度只关注方向，不关注长度，适合文本语义匹配
    # （L2 距离会受向量模长影响，对未归一化的 embedding 不友好）
    vectordb = FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    # 保存索引到本地，下次可以直接加载（生成 index.faiss + index.pkl 两个文件）
    # index.faiss：纯向量的二进制索引；index.pkl：Document 对象（包含原文+metadata）
    vectordb.save_local(INDEX_DIR)

# ============================================================
# 第四部分：配置 Agent 的 LLM 后端
# ============================================================
# 使用 DeepSeek API 作为 Agent 的"大脑"
# Agent 需要一个强大的 LLM 来：
#   1. 理解用户问题
#   2. 决定调用哪个工具
#   3. 构造工具调用参数
#   4. 综合工具返回的信息生成最终回答
#
# OpenAIServerModel：smolagents 提供的兼容 OpenAI API 格式的模型接口
# DeepSeek API 兼容 OpenAI 格式，所以可以直接使用
model = OpenAIServerModel(
    model_id="deepseek-chat",                  # 使用 DeepSeek-Chat 模型
    api_base="https://api.deepseek.com/v1",    # DeepSeek API 地址
    api_key=DS_KEY,
    # flatten_messages_as_text=True：将多轮对话展平为纯文本
    # 某些模型对 tool_call 格式支持不完善时需要开启
    flatten_messages_as_text=True,
)

# ============================================================
# 第五部分：创建 Agent 并注册工具
# ============================================================
# SemanticRetriever：自定义的语义检索工具（定义在 utils/utils.py）
# 它封装了 FAISS 向量数据库的检索功能，作为 smolagents 的 Tool 使用
# Agent 可以调用它来从本地知识库中检索相关文档
retriever_tool = SemanticRetriever(vectordb)

# WebSearchTool：smolagents 内置的网络搜索工具
# 当本地知识库中没有相关信息时，Agent 可以选择搜索互联网
webserach_tool = WebSearchTool()

# ToolCallingAgent：smolagents 的工具调用型智能体
# 工作流程：
#   1. 接收用户问题
#   2. LLM 分析问题，决定是否需要调用工具
#   3. 如果需要，生成工具调用指令（函数名 + 参数）
#   4. 执行工具，获取结果
#   5. LLM 根据工具结果生成最终回答
#   6. 如果信息不够，可以再次调用工具（多轮推理）
#
# 与简单 RAG 的区别：
#   简单 RAG：每次固定检索 top-k 文档
#   Agentic RAG：Agent 自主决定是否检索、检索什么、是否需要补充搜索
agent = ToolCallingAgent(tools=[retriever_tool, webserach_tool], model=model)

# ============================================================
# 第六部分：交互式对话循环
# ============================================================
# 用户可以持续提问，Agent 会自主选择工具来回答
#
# Agent 的多步推理示例（ReAct 式循环）：
#   ┌───────────────────────────────────────────────────────┐
#   │ User:  量子计算最新进展是什么？                        │
#   │ Agent: [思考] 这是时效性问题，本地百科可能过时         │
#   │        [Action] webserach_tool(query="量子计算 2026")  │
#   │        [Observation] 搜索结果...                       │
#   │        [思考] 信息不够具体，再查一下技术细节           │
#   │        [Action] semantic_retriever(query="量子比特")   │
#   │        [Observation] 检索到的文档...                   │
#   │        [Final Answer] 综合两类信息生成回答             │
#   └───────────────────────────────────────────────────────┘
while True:
    query = input("Enter your query: ")
    # agent.run() 内部流程：
    #   1. 将 query + 工具描述 + system prompt 发送给 LLM
    #   2. LLM 返回工具调用指令（如 semantic_retriever(query="..."))
    #   3. smolagents 解析指令并执行对应工具的 forward()
    #   4. 把工具返回的字符串拼回对话上下文，再次发给 LLM
    #   5. LLM 综合信息生成最终回答（或继续调用其他工具）
    #   6. 达到 max_steps 或 LLM 输出 final_answer 时返回
    agent_output = agent.run(query)

    print("Response:", agent_output)
