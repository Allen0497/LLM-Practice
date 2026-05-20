"""
DPO 偏好数据合成脚本
====================

本脚本用于自动合成 DPO（Direct Preference Optimization，直接偏好优化）训练所需的偏好对数据。

整体流程：
  1. 加载原始问题数据集
  2. 用两个不同的模型（Model A 和 Model B）分别对每个问题生成回答
  3. 用 EvolQuality 对回答进行质量进化（让裁判模型改写润色回答）
  4. 用 UltraFeedback 对进化后的回答进行打分和评价
  5. 将所有结果保存为 parquet 格式，供后续 DPO 训练使用

为什么需要偏好数据？
  DPO 训练需要"同一个问题的两个回答 + 哪个更好"这样的数据对。
  通过让两个能力不同的模型生成回答，再用裁判模型打分，
  就能自动构造出 (chosen, rejected) 偏好对，无需人工标注。

依赖库：
  - distilabel: Argilla 团队开发的数据合成框架，提供 LLM 调用和数据处理的标准化接口
  - datasets: HuggingFace 数据集加载库
  - pandas: 数据处理和保存
"""

# ============================================================
# 第一部分：导入依赖
# ============================================================
import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset

# distilabel 是一个专门用于 LLM 数据合成的框架
# OpenAILLM: 兼容 OpenAI API 格式的 LLM 封装（这里用来调用 DeepSeek 的 API 作为裁判模型）
# TransformersLLM: 本地 HuggingFace Transformers 模型的封装
from distilabel.llms import OpenAILLM
from distilabel.llms import TransformersLLM

# TextGeneration: 基础文本生成任务（本脚本未直接使用，但作为其他任务的基础）
# EvolQuality: 回答质量进化——让裁判模型对回答进行改写和润色，提升回答质量
# UltraFeedback: 多维度回答评分——让裁判模型从多个维度对回答打分并给出理由
from distilabel.steps.tasks import TextGeneration, EvolQuality, UltraFeedback

from tqdm import tqdm
from utils.utils import find_files

# ============================================================
# 第二部分：路径和参数配置
# ============================================================

# 临时文件缓存路径（用于 HuggingFace datasets 的缓存）
TMP_PATH        = "/archive/share/cql/aaa/tmp"

# 最终生成的偏好数据保存路径
DATA_PATH       = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data"

# 原始问题数据文件（从 reward 目录下查找包含 "rl" 关键字的 parquet 文件）
DATA_FILES      = find_files(["rl"],"/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/reward/data")

# 模型 A：Qwen3-8B-Instruct（较强的中文模型，通常生成质量更高的回答）
MODEL_A_PATH    = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen3-8B-Instruct"

# 模型 B：Meta-Llama-3-8B-Instruct-Chinese（中文适配的 Llama 模型，回答质量可能略低）
# 使用两个能力有差异的模型，是为了自然地产生质量不同的回答对
MODEL_B_PATH    = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Meta-Llama-3-8B-Instruct-Chinese"

# 只取前 20 条数据做演示（设为 0 或负数则使用全部数据）
TRAIN_SUBSET    = 20

# DeepSeek API Key（用于调用 DeepSeek 作为裁判模型）
DS_KEY         = "xxx"

# ============================================================
# 第三部分：加载问题数据集
# ============================================================

# 从 parquet 文件加载数据集
# split="train" 表示加载训练集部分
ds = load_dataset("parquet", data_files=DATA_FILES,
                        split="train", cache_dir=TMP_PATH)

# 如果设置了子集大小，只取前 N 条（方便调试和演示）
if TRAIN_SUBSET > 0:
    ds = ds.select(range(TRAIN_SUBSET))

# 提取所有问题文本
questions = ds["question"]

# ============================================================
# 第四部分：加载生成模型（两个能力不同的模型）
# ============================================================

# 加载模型 A（Qwen3-8B-Instruct）
# TransformersLLM 会自动处理 tokenizer 加载、设备分配等
llm_a = TransformersLLM(model=MODEL_A_PATH)
llm_a.load()  # 实际加载模型权重到 GPU

# 加载模型 B（Llama-3-8B-Instruct-Chinese）
llm_b = TransformersLLM(model=MODEL_B_PATH)
llm_b.load()

# ============================================================
# 第五部分：加载裁判模型（DeepSeek API）
# ============================================================

# 裁判模型使用 DeepSeek 的 API（兼容 OpenAI 格式）
# 裁判模型的职责：
#   1. 在 EvolQuality 中改写润色回答
#   2. 在 UltraFeedback 中对回答进行多维度打分
judge_llm = OpenAILLM(
    model="deepseek-chat",                      # DeepSeek 的对话模型
    base_url=r"https://api.deepseek.com/v1",    # API 地址
    api_key=DS_KEY                               # API 密钥
)
judge_llm.load()

# ============================================================
# 第六部分：初始化数据处理步骤
# ============================================================

# EvolQuality（回答质量进化）
# 原理：让裁判模型对原始回答进行改写，使其更加完善、准确、详细
# num_evolutions=1 表示只进化一轮（进化次数越多，改写越多，但也越慢）
# 这一步的目的是让回答质量有更明显的差异，便于后续打分区分
evol_quality   = EvolQuality(llm=judge_llm, num_evolutions=1)
evol_quality.load()

# UltraFeedback（多维度回答评分）
# 原理：让裁判模型从 helpfulness（有帮助性）、honesty（诚实性）、
#       instruction-following（指令遵循）、truthfulness（真实性）等维度
#       对两个回答分别打分（1-5分），并给出评分理由
# 这一步产生的分数就是 DPO 训练中判断 chosen/rejected 的依据
ultrafeedback  = UltraFeedback(llm=judge_llm)
ultrafeedback.load()

# 用于收集所有生成结果的列表
records = []

# ============================================================
# 第七部分：辅助函数和生成参数
# ============================================================

def chat(msg):
    """
    将用户消息包装成对话格式。
    大多数指令模型要求输入是 [{"role": "user", "content": "..."}] 的格式，
    这个函数就是做这个转换。
    """
    return [{"role": "user", "content": msg}]

# 生成参数配置
GEN_KWARGS = dict(
    max_new_tokens=256,   # 最多生成 256 个 token（控制回答长度）
    temperature=0.7       # 温度参数：0.7 是一个平衡创造性和稳定性的值
                          # 越高越随机多样，越低越确定保守
)

# ============================================================
# 第八部分：核心循环 —— 生成、进化、打分
# ============================================================
# 对每个问题执行三步流水线：
#
#   ┌──────────┐     ┌──────────────┐     ┌───────────────┐
#   │ 步骤 (i) │     │  步骤 (ii)   │     │  步骤 (iii)   │
#   │ 双模型   │ ──→ │ EvolQuality  │ ──→ │ UltraFeedback │
#   │ 独立生成 │     │ 质量进化改写 │     │ 多维度打分    │
#   └──────────┘     └──────────────┘     └───────────────┘
#
# 最终每条数据包含：原始回答、进化后回答、评分和评分理由

for q in tqdm(questions, desc="Generating/Rewrite/Scoring"):

    # ---- 步骤 (i)：两个模型独立生成回答 ----
    # 同一个问题分别交给 Model A（Qwen）和 Model B（Llama）回答
    # 由于两个模型能力不同，生成的回答质量自然会有差异
    # inputs 参数需要是列表的列表（批量接口），所以用 [chat(q)] 包一层
    # 返回结构：[{"generations": ["回答文本"]}]，取 [0]["generations"][0] 得到文本
    gen_a = llm_a.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]
    gen_b = llm_b.generate(inputs=[chat(q)], **GEN_KWARGS)[0]["generations"][0]

    # ---- 步骤 (ii)：EvolQuality 质量进化 ----
    # 将原始回答交给裁判模型（DeepSeek）进行改写润色
    # 裁判模型会在保持原意的基础上，让回答更加完善、准确、有条理
    # process() 返回一个生成器，用 next() 取出第一批结果
    # 结果结构：[{"evolved_response": "改写后的回答"}]
    evo_a = evol_quality.process([{"instruction": q, "response": gen_a}])
    evo_b = evol_quality.process([{"instruction": q, "response": gen_b}])
    evo_a = next(evo_a)[0]["evolved_response"]
    evo_b = next(evo_b)[0]["evolved_response"]

    # ---- 步骤 (iii)：UltraFeedback 多维度打分 ----
    # 将进化后的两个回答一起交给裁判模型打分
    # 裁判模型会从多个维度（有帮助性、诚实性、指令遵循、真实性）评分
    # 每个回答得到一个 1-5 的分数，以及详细的评分理由
    # 分数高的回答将作为 DPO 训练中的 chosen（偏好回答）
    # 分数低的回答将作为 DPO 训练中的 rejected（拒绝回答）
    feedback = ultrafeedback.process([{
        "instruction": q,
        "generations": [evo_a, evo_b]
    }])
    feedback = next(feedback)
    ratings     = feedback[0]["ratings"]      # 两个回答的分数列表，如 [4.5, 2.0]
    rationales  = feedback[0]["rationales"]   # 两个回答的评分理由列表

    # 将本条数据的所有信息收集到 records 中
    records.append({
        "question":q,                       # 原始问题
        "original_response_j":gen_a,        # 模型 A 的原始回答
        "original_response_k":gen_b,        # 模型 B 的原始回答
        "response_j":evo_a,                 # 模型 A 回答经过进化后的版本
        "response_k":evo_b,                 # 模型 B 回答经过进化后的版本
        "rating_j":ratings[0],              # 进化后回答 A 的评分
        "rating_k":ratings[1],              # 进化后回答 B 的评分
        "rationales":rationales             # 评分理由（用于分析和调试）
    })

# ============================================================
# 第九部分：保存偏好数据
# ============================================================
# 将所有记录转为 DataFrame 并保存为 parquet 格式
# parquet 是一种列式存储格式，读写速度快、压缩率高，是大数据场景的标准格式
# 保存后的文件可以直接被 HuggingFace datasets 加载，供 DPO 训练使用
#
# 后续使用时，会根据 rating_j 和 rating_k 的大小关系来确定：
#   - rating 更高的回答 → chosen（模型应该学习的好回答）
#   - rating 更低的回答 → rejected（模型应该避免的差回答）
df = pd.DataFrame(records)
df.to_parquet(os.path.join(DATA_PATH, "dpo_data.parquet"), index=False)
