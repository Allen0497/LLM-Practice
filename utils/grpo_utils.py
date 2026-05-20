"""
GRPO 奖励函数模块
======================================================

GRPO 与 PPO 最本质的区别就在这里：
  PPO 使用一个训练好的神经网络（奖励模型）来打分
  GRPO 使用基于规则的函数来打分

这种设计有几个优点：
  1. 不需要额外训练奖励模型 → 简化流程
  2. 奖励信号明确、可解释 → 不会出现"奖励黑客"问题
  3. 可以灵活组合多个奖励函数 → 多维度引导模型行为

本模块定义了两个奖励函数：
  - format_reward：格式奖励，检查模型是否学会了"先思考后回答"的格式
  - accuracy_reward：准确性奖励，检查模型的最终答案是否正确

在 DeepSeek-R1 的原始论文中，这两个奖励的设计哲学是：
  格式奖励 → 教会模型"怎么说"（推理格式）
  准确性奖励 → 教会模型"说什么"（正确内容）

两者缺一不可：
  只有准确性奖励 → 模型可能直接猜答案，不展示推理过程
  只有格式奖励 → 模型学会了格式，但推理内容可能是胡说八道
"""

import re
from math_verify import LatexExtractionConfig, parse, verify

def format_reward(completions, **kwargs):
    """
    格式奖励函数：检查模型生成的回答是否符合 <think>...</think><answer>...</answer> 格式

    这是 GRPO 训练中"教模型学会推理格式"的关键奖励函数。

    DeepSeek-R1 的核心创新之一就是：
      通过 RL 训练（而非 SFT）让模型自发学会使用思维链（Chain of Thought）。
      format_reward 就是引导模型学会这种推理格式的信号。

    参数:
        completions: 模型生成的回答列表
            结构: [[{"role": "assistant", "content": "..."}], ...]
            每个元素是一条完整的对话回答

        **kwargs: 额外参数（如 solution），由 GRPOTrainer 自动传入
            这些参数来自数据集中的列（因为设置了 remove_unused_columns=False）

    返回:
        list[float]: 每个回答的格式奖励分数
            1.0 = 格式正确（包含完整的 <think>...</think><answer>...</answer>）
            0.0 = 格式不正确

    格式要求（正则表达式解析）:
        ^<think>    → 必须以 <think> 开头
        .*?         → 中间是思考过程（非贪婪匹配）
        </think>    → 思考结束标签
        \\s*         → 允许空白字符
        <answer>    → 答案开始标签
        .*?         → 中间是最终答案
        </answer>$  → 必须以 </answer> 结尾

    设计细节:
        - 使用 0/1 的离散奖励，而非连续分数
        - 这是一种"硬约束"：要么完全正确，要么完全错误
        - 在训练初期，大部分回答格式不对，format_reward 多为 0
        - 随着训练推进，模型逐渐学会正确格式，reward 上升
        - 这个"顿悟"过程是 R1-Zero 论文中观察到的有趣现象
    """
    # 匹配完整的 <think>思考过程</think><answer>最终答案</answer> 格式
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    # 从 completions 中提取每条回答的文本内容
    # completions 的格式是 [[{"role": "assistant", "content": "..."}], ...]
    # 所以 completion[0]["content"] 取出助手回答的文本
    completion_contents = [completion[0]["content"] for completion in completions]
    # 对每条回答进行格式匹配
    matches = [re.match(pattern, content) for content in completion_contents]
    # 匹配成功得 1.0 分，失败得 0.0 分
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, **kwargs):
    """
    准确性奖励函数：检查模型的最终答案是否与标准答案一致

    这是 GRPO 训练中"教模型学会正确推理"的关键奖励函数。

    使用 math_verify 库来解析和验证数学答案：
      1. 从模型回答中提取 LaTeX 格式的答案
      2. 从标准答案中提取 LaTeX 格式的答案
      3. 进行数学等价性验证（而非字符串匹配）

    为什么不能用简单的字符串匹配？
      因为数学答案有多种等价表示：
        "1/2" == "0.5" == "\\frac{1}{2}"
        "x^2 + 2x + 1" == "(x+1)^2"
      math_verify 可以处理这些等价情况

    参数:
        completions: 模型生成的回答列表（同 format_reward）
        **kwargs: 必须包含 "solution" 键
            solution 来自数据集中的 solution 列（标准答案）
            这就是为什么 GRPOConfig 中要设置 remove_unused_columns=False

    返回:
        list[float]: 每个回答的准确性奖励分数
            1.0 = 答案正确
            0.0 = 答���错误或无法解析

    特殊情况处理:
        - 如果标准答案无法解析（gold_parsed 为空）→ 给 1.0 分（宽容处理）
        - 如果模型答案无法解析 → 给 0.0 分
        - 如果验证过程出现异常 → 给 0.0 分（安全兜底）
    """
    # 从 kwargs 中获取标准答案列表
    # 这些答案来自数据集的 solution 列
    solutions = kwargs["solution"]
    # 提取每条模型回答的文本内容
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        # 解析标准答案中的 LaTeX 数学表达式
        # extraction_mode="first_match"：只取第一个匹配到的数学表达式
        # LatexExtractionConfig()：使用默认的 LaTeX 提取配置
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        # 解析模型回答中的 LaTeX 数学表达式
        # 模型的答案通常在 <answer>...</answer> 标签内
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                # verify() 进行数学等价性验证
                # 返回 True/False，转为 float 得到 1.0/0.0
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                # 验证过程中出现任何异常（如格式不兼容），给 0 分
                rewards.append(0.0)
        else:
            # 标准答案无法解析时，宽容处理，给满分
            # 这种情况通常是数据集中某些答案格式特殊
            rewards.append(1.0)
    return rewards
