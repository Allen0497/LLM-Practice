from __future__ import annotations
import os
from itertools import chain
import torch
import time
import gc
from langchain_core.vectorstores import VectorStore
from smolagents import Tool
from typing import List


# ============================================================
# SemanticRetriever：语义检索工具（供 Agentic RAG 使用）
# ============================================================
# 这个类将 FAISS 向量数据库封装为 smolagents 的 Tool
# 使得 Agent 可以通过"函数调用"的方式使用语义检索
#
# smolagents Tool 的工作原理：
#   1. Agent（LLM）看到工具的 name、description、inputs 描述
#   2. 根据用户问题，决定是否调用这个工具
#   3. 如果调用，生成参数（如 query="大模型训练方法"）
#   4. smolagents 框架自动调用 forward() 方法执行检索
#   5. 将返回的文档文本反馈给 Agent
#
# 设计要点：
#   - description 要清晰描述工具的能力，帮助 Agent 判断何时使用
#   - inputs 定义参数格式，帮助 Agent 构造正确的调用
#   - output_type 告诉 Agent 返回值的类型
class SemanticRetriever(Tool):
    """
    语义检索工具：从向量数据库中检索与查询最相似的 k 个文档。

    工作流程：
      query（文本）→ embedding_model（向量化）→ FAISS（最近邻搜索）→ top-k 文档

    Parameters
    ----------
    vectordb : VectorStore
        已初始化的向量数据库实例（如 FAISS），需实现 similarity_search API
    top_k : int, optional
        返回的文档数量，默认 7
    """

    # ── Tool 元数据（Agent 通过这些信息决定是否调用）──
    name: str = "semantic_retriever"
    # description 是 Agent 判断何时使用此工具的关键
    # 写得好 → Agent 在合适的时机调用；写得差 → Agent 可能误用或不用
    description: str = (
        "Return the top-k documents whose embeddings are most similar to the "
        "input query. The query should be phrased affirmatively, not as a question."
    )
    # inputs 定义工具接受的参数格式（JSON Schema 风格）
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

    def __init__(self, vectordb: VectorStore, *, top_k: int = 7, **kwargs) -> None:
        # 调用父类 Tool 的 __init__，触发 smolagents 内部对 name/description/inputs
        # 等元数据的校验和注册（缺失或类型错误会在这里抛异常）
        super().__init__(**kwargs)
        # 用下划线前缀表示这是私有属性，不参与序列化为 Tool 描述
        self._db: VectorStore = vectordb
        self._top_k: int = top_k

    def forward(self, query: str) -> str:
        """
        执行语义检索，返回格式化的文档列表。

        Agent 调用此工具时，smolagents 框架会自动调用这个方法。
        返回的字符串会被反馈给 Agent，Agent 据此生成最终回答。

        约定：
          - forward 的参数签名必须与 self.inputs 中声明的字段一致（这里是 query: str）
          - 返回值类型必须与 self.output_type 一致（这里是 string）
          - 抛出的异常会被 Agent 看到，Agent 可能据此调整后续行为
        """
        if not isinstance(query, str):
            raise TypeError("`query` must be a string.")

        docs = self._similar_docs(query)

        # 将检索到的文档格式化为可读的文本
        # Agent 会阅读这些文本，从中提取信息来回答用户问题
        # 用 ===== Document N ===== 分隔，便于 LLM 区分多个来源
        formatted = [
            f"===== Document {idx} =====\n{doc.page_content}" for idx, doc in enumerate(docs)
        ]
        return "\nRetrieved documents:\n" + "\n".join(formatted)

    def _similar_docs(self, query: str) -> List:
        """
        调用向量数据库的相似度搜索，返回 top-k 最相似的文档。

        similarity_search 内部流程：
          1. 用 embedding_model 把 query 编码为向量
          2. 在 FAISS 索引中按 distance_strategy（这里是 COSINE）查找最近邻
          3. 取索引位置反查存储的 Document（包含原文 + metadata）
          4. 返回 List[Document]，长度为 k
        """
        return self._db.similarity_search(query, k=self._top_k)

def format_to_r1(example):
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def collator_ppo(data):
    # for PPO use
    return {key: [d[key] for d in data] for key in data[0]}

def preprocess_ppo_dataset(examples,tokenizer):
    # for PPO use
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

def preprocess_rm_dataset(examples,tokenizer):
    # for RM use
    # Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples

def format_to_chatml(data):
    formatted_data = []
    for sample in data:
        problem = sample["problem"]
        generation = sample["generation"]
        
        formatted_data.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": generation}
            ]
        )
    return {"messages": formatted_data}

def formatting_prompts_func_distill(example):
    # for distill use
    output_texts = []
    for i in range(len(example["problem"])):
        human_text = example["problem"][i]
        gpt_text = example["generation"][i]
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func(example):
    # for sft use
    output_texts = []
    for i in range(len(example["conversations"])):
        for item in example["conversations"][i]:
            if item["from"] == "human":
                human_text = item["value"]
            elif item["from"] == "gpt":
                gpt_text = item["value"]
            else:
                raise ValueError(f"Unknown sender: {item['from']}")
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts

def find_files(dirs,path="data/pt"):
    """
    遍历目录，查找所有文件
    """
    files = []
    for dir in dirs:
        base_path = os.path.join(path, dir)
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files

def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def tokenize_dataset(examples,tokenizer,block_size=512):
    """
    预处理预训练数据集，将文本分词并分块
    """
    eos_token = "<|im_end|>"
    text_examples = [text + eos_token for text in examples["text"]]  # 添加结束符
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)

    concatenated_examples = {
        k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
    }
    
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    total_length = (total_length // block_size) * block_size  # 对齐块大小

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result,total_length

def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量。
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
