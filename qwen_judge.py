"""
LLM-as-a-Judge 评测脚本（胜率评估）
======================================================

这是大模型训练完成后的"高级评测"方法，与第 5 篇笔记里的传统 benchmark 评测互补。

传统评测 vs LLM-as-a-Judge：
  传统 benchmark（MMLU/CMMLU 等）：
    - 答案是固定的（A/B/C/D 选项或精确字符串）
    - 适合考察知识、推理等"有标准答案"的任务
    - 但写作、对话、总结这类开放任务没法这么评

  LLM-as-a-Judge：
    - 用一个强大的模型（如 GPT-4、Llama-3-70B）当"裁判"
    - 给两个回答，让裁判判定"哪个更好"
    - 累计统计 → 胜率（Win Rate）
    - 适合开放式任务

胜率（Win Rate）：
    被评估模型 vs 参考答案（reference），由裁判判定哪个更好
    Win Rate = 被评估模型获胜次数 / 总样本数 × 100%

本脚本流程：
  ┌──────────────────────────────────────────────────────────┐
  │  ① 加载 TLDR 数据集（验证集）                              │
  │     每条样本包含 prompt（要总结的文本）和 completion（人写的总结）│
  │                                                          │
  │  ② 用待评估模型为每个 prompt 生成总结                     │
  │     → 用 vLLM 加速批量推理                                │
  │                                                          │
  │  ③ 调用 PairwiseJudge 进行成对比较：                      │
  │     裁判模型看 [reference, model_completion]，选出更好的一个│
  │                                                          │
  │  ④ 统计胜率��模型胜过 reference 的比例                    │
  └──────────────────────────────────────────────────────────┘

技术栈：
  - vLLM：超快的批量推理引擎（PagedAttention）
  - TRL：HuggingFace 的 RLHF 库，提供 PairwiseJudge
  - HfPairwiseJudge：用本地 HF 模型当裁判
  - OpenAIPairwiseJudge：用 OpenAI API（如 gpt-4o）当裁判
  - TLDR 数据集：Reddit 帖子 + 人工总结，是 RLHF 经典评测集

关于 trl 提供的两种 Judge：
  HfPairwiseJudge：内部加载本地 HF 模型（默认 Llama-3-70B-Instruct）
                  优点：完全本地、无 API 费用
                  ��点：需要大显存（70B 模型）
  OpenAIPairwiseJudge：调用 OpenAI API
                  优点：模型强、无需本地资源
                  缺点：要钱、要联网、数据可能被记录
"""

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams

from trl import HfPairwiseJudge, OpenAIPairwiseJudge


"""
使用示例：

# 用本地 Llama-3-70B 当裁判（默认）
python qwen_judge.py --model_name_or_path vwxyzjn/rloo_tldr --num_examples 1000
# 输出：Model win rate: 31.40%
# 解读：被评估模型 31.4% 的样本胜过人工总结，68.6% 不如人工总结

# 用 GPT-4o-mini 当裁判
python qwen_judge.py --model_name_or_path vwxyzjn/ppo_tldr --judge_model gpt-4o-mini --num_examples 1000
# 输出：Model win rate: 63.00%
"""


# ============================================================
# 第一部分：脚本参数定义
# ============================================================
# 使用 dataclass + HfArgumentParser 是 HuggingFace 推荐的 CLI 参数方式
# 比 argparse 更结构化，自动从 type hint 推断参数类型
@dataclass
class ScriptArguments:
    r"""
    脚本参数。

    Args:
        model_name_or_path (`str`):
            要评估的模型路径或 HuggingFace 模型 ID
        judge_model (`str`, *optional*, defaults to `"meta-llama/Meta-Llama-3-70B-Instruct"`):
            裁判模型路径或 ID
            可选：'gpt-3.5-turbo-0125' / 'gpt-4o-mini' / 'meta-llama/Meta-Llama-3-70B-Instruct'
        num_examples (`int` or `None`, *optional*, defaults to `None`):
            评估的样本数。None 表示用整个验证集（约 6000 条），跑起来很慢
            建议先用 100~1000 跑通流程
    """

    model_name_or_path: str = field(metadata={"help": "Model name or path to the model to evaluate."})
    judge_model: str = field(
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        metadata={
            "help": "Model name or path to the model to use as a judge. E.g., 'gpt-3.5-turbo-0125' or "
            "'meta-llama/Meta-Llama-3-70B-Instruct'."
        },
    )
    num_examples: Optional[int] = field(default=None, metadata={"help": "Number of examples to evaluate."})


# 解析命令行参数（解析失败会自动打印帮助信息并退出）
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# ============================================================
# 第二部分：加载评测数据集
# ============================================================
# trl-lib/tldr：经典的总结任务数据集
#   - 来源：Reddit TL;DR 数据集（OpenAI WebGPT 等论文都在用）
#   - 每条样本：
#     * prompt：一段较长的文本（Reddit 帖子）
#     * completion：人工写的简短总结（gold reference）
#   - validation split：约 6000 条
dataset = load_dataset("trl-lib/tldr", split="validation")

# 调试时可以只取前 N 条
if script_args.num_examples is not None:
    dataset = dataset.select(range(script_args.num_examples))

# 提取两个字段
prompts = dataset["prompt"]                    # 输入：原文
reference_completions = dataset["completion"]  # 参考答案：人工总结

# ============================================================
# 第三部分：用 vLLM 批量生成模型回答
# ============================================================
# vLLM 是 UC Berkeley 的高速推理引擎，核心创新：
#   1. PagedAttention：把 KV cache 像虚拟内存一样分页管理，显存利用率高
#   2. Continuous batching：动态合并不同长度请求到同一 batch
#   3. 比 HuggingFace transformers 原生 generate 快 5~24 倍
#
# 评测要跑成百上千个样本，用 vLLM 几分钟搞定，HF 原生可能要几十分钟

# SamplingParams 控制生成行为
sampling_params = SamplingParams(
    temperature=0.0,    # 0 = 完全贪心解码（同输入永远同输出，评测要稳定）
    top_p=0.95,         # nucleus sampling（temperature=0 时其实不起作用）
    max_tokens=200,     # 最长生成 200 token，TLDR 总结够用了
)

# 加载模型到 vLLM
# tensor_parallel_size=1：单卡推理；多卡可设为 GPU 数量做张量并行
llm = LLM(model=script_args.model_name_or_path, tensor_parallel_size=1)

# 批量生成：一次传入所有 prompts，vLLM 内部自动调度
# outputs 是 List[RequestOutput]，每个有 .outputs[0].text 字段
outputs = llm.generate(prompts, sampling_params)

# 提取生成的文本（strip 去除首尾空白）
model_completions = [output.outputs[0].text.strip() for output in outputs]

# ============================================================
# 第四部分：用 LLM 裁判进行成对评测
# ============================================================
# 根据 judge_model 名称选择裁判类型
#   - 含 "gpt" → 走 OpenAI API
#   - 否则     → 走本地 HF 模型
if "gpt" in script_args.judge_model:
    # OpenAIPairwiseJudge 内部需要 OPENAI_API_KEY 环境变量
    judge = OpenAIPairwiseJudge(script_args.judge_model)
else:
    # HfPairwiseJudge 会用 vLLM 加载本地模型（70B 需要大显存）
    judge = HfPairwiseJudge(script_args.judge_model)

# 准备成对数据：每对包含 [reference, model_completion]
# 重要：顺序固定为 [c0=reference, c1=model_completion]
# judge.judge() 返回每条样本的"更好选项的索引"（0 或 1）
completions = [[c0, c1] for c0, c1 in zip(reference_completions, model_completions)]

# 调用裁判（内部 prompt 大致是：
#   "Given the prompt {prompt}, which response is better?
#    Response A: {c0}
#    Response B: {c1}
#    Answer with A or B.")
# 注意：trl 内部会做 position bias 处理（A/B 顺序对结果有影响，需要消偏）
best_idxs = judge.judge(prompts, completions)

# 统计：被评估模型（索引 1）赢了多少次
# best_idxs.count(1) → 模型赢的样本数
# best_idxs.count(0) → 参考答案赢的样本数
model_win_rate = best_idxs.count(1) / len(best_idxs)
print(f"Model win rate: {model_win_rate*100:.2f}%")

# 解读胜率：
#   < 50%：被评估模型不如参考答案
#   = 50%：和参考答案相当
#   > 50%：超过参考答案
#
# 论文里的典型数字：
#   - SFT 模型 vs 人工总结：~30%（人工还是好不少）
#   - PPO/DPO 后：~60-70%（已经超过人工）
#   - GPT-4 当裁判：通常比 Llama-3-70B 当裁判给出的胜率更高
