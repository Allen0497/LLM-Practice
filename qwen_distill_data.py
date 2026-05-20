"""
蒸馏数据生成脚本
======================================================

知识蒸馏的第一步：用大模型（Teacher）生成高质量的推理数据，
供后续训练小模型（Student）使用。

两种蒸馏路线：
  路线 A：数据蒸馏（本脚本）
    大模型推理 → 生成带推理链的回答 → 用这些数据 SFT 小模型
    优点：简单，小模型只需要做 SFT
    缺点：小模型只是"模仿"大模型的输出，不涉及概率分布层面的学习

  路线 B：模型蒸馏（qwen_distill.py）
    大模型和小模型同时前向传播 → 对齐 logits 分布（KL 散度）
    优点：小模型学到大模型的"思维方式"（概率分布），而非只是答案
    缺点：需要同时加载两个模型，显存要求高

本脚本使用 distilabel 框架，通过 vLLM 高速推理大模型，
批量生成带推理链的数学题解答，作为蒸馏训练数据。

DeepSeek-R1 的数据蒸馏路线：
  DeepSeek-R1（大模型，有推理能力）
      ↓ 生成带 <think>...</think> 推理链的回答
  蒸馏数据集（problem + generation）
      ↓ SFT 或模型蒸馏
  DeepSeek-R1-Distill-Qwen-1.5B/7B/14B/32B（小模型）
"""

from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

# ============================================================
# 第一部分：Prompt 模板设计
# ============================================================
# 这个模板告诉大模型（Teacher）如何回答数学题
# \boxed{} 是 LaTeX 数学格式，用于框住最终答案
# 这与 accuracy_reward 中的 LatexExtractionConfig 相呼应：
#   生成数据时用 \boxed{} 包裹答案 → 验证时从 \boxed{} 中提取答案
# {{ instruction }} 是 distilabel 的模板变量，会被替换为实际题目
prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}:
{{ instruction }}"""

# ============================================================
# 第二部分：加载原始数学题数据集
# ============================================================
# numina-deepseek-DeepSeek-R1-Distill-Qwen-7B 是一个��经包含
# DeepSeek-R1-Distill-Qwen-7B 生成结果的数据集
# 这里 select(range(10)) 只取前 10 条用于演示，实际训练需要全量数据
dataset = load_dataset(
    "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/distill/numina-deepseek-DeepSeek-R1-Distill-Qwen-7B",
    split="train"
).select(range(10))

# ============================================================
# 第三部分：配置 Teacher 模型
# ============================================================
# 使用 DeepSeek-R1-Distill-Qwen-7B 作为 Teacher 模型
# 这是一个已经具备推理能力的模型（由 DeepSeek-R1 蒸馏而来）
# 注释中说"最好用 r1"，意思是用原始的 DeepSeek-R1 效果更好，
# 但 R1 是 671B 的 MoE 模型，资源要求极高，所以用 7B 蒸馏版代替
model_id = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/DeepSeek-R1-Distill-Qwen-7B"

# ============================================================
# 第四部分：构建 distilabel Pipeline
# ============================================================
# distilabel 是 Argilla 开发的数据生成框架，专门用于构建 LLM 训练数据
# Pipeline 是一个有向无环图（DAG），每个节点是一个数据处理步骤
with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    # vLLM：高性能 LLM 推理引擎
    # 相比直接用 transformers 推理，vLLM 通过 PagedAttention 技术
    # 大幅提升吞吐量（通常快 10-20 倍），非常适合批量数据生成
    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            # tensor_parallel_size：张量并行度
            # 设为 1 表示单卡推理；如果有多卡可以设为 2/4/8
            "tensor_parallel_size": 1,
            # max_model_len：模型支持的最大序列长度
            # 7B 模型推理数学题，8192 token 通常足够
            "max_model_len": 8192,
        },
        generation_kwargs={
            # temperature=0.6：适中的随机性
            # 太低（如 0.1）→ 回答太确定，多样性差
            # 太高（如 1.0）→ 回答太随机，质量下降
            # 0.6 是 DeepSeek 官方推荐的推理温度
            "temperature": 0.6,
            # max_new_tokens=8192：允许生成很长的推理链
            # 数学推理可能需要很多步骤，不能截断
            "max_new_tokens": 8192,
        },
    )

    # TextGeneration：distilabel 内置的文本生成步骤
    # 它会将数据集中的每条数据填入 prompt_template，然后调用 llm 生成回答
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm,
        template=prompt_template,
        # num_generations=4：对每道题生成 4 个不同的回答
        # 多个回答可以用于：
        #   1. 拒绝采样（Rejection Sampling）：只保留答案正确的回答
        #   2. 数据增强：增加训练数据多样性
        num_generations=4,
        # input_mappings：将数据集的列名映射到模板变量
        # "instruction" 是模板中的变量名，"problem" 是数据集的列名
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )

# ============================================================
# 第五部分：运行 Pipeline
# ============================================================
if __name__ == "__main__":
    # pipeline.run() 会：
    # 1. 启动 vLLM 服务
    # 2. 批量处理数据集中的每条数据
    # 3. 将生成结果保存到 distiset 对象中
    # distiset 包含原始数据 + 生成的回答，可以直接用于后续训练
    distiset = pipeline.run(dataset=dataset)
