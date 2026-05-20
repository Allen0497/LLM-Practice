"""
Qwen-R1 风格思维链对话脚本（用 GRPO 训练后的模型）
======================================================

本脚本用来跟一个"会思考"的模型对话——它在回答前会先输出推理过程。
对应的 checkpoint 是用 GRPO（第 10 篇笔记）训练出来的，模型已经学会了
DeepSeek-R1 的输出格式：

  <think>
    思考过程：第一步 ... 第二步 ... 所以答案是 ...
  </think>
  <answer>
    最终答案
  </answer>

与 qwen_chat.py（基础对话）的区别：
  ┌────────────────────────────────────┬────────────────────────────────────┐
  │  qwen_chat.py（基础对话）           │  qwen_r1_chat.py（思维链对话）       │
  ├────────────────────────────────────┼────────────────────────────────────┤
  │  无 system prompt                   │  有专门的 system prompt 教模型格式   │
  │  max_new_tokens=128                 │  max_new_tokens=4096（思考占字数多） │
  │  直接给答案                         │  先 <think> 再 <answer>             │
  │  适合 SFT 后的模型                  │  适合 GRPO 后的模型                  │
  │  用 tokenizer.decode（单条）        │  用 batch_decode（批量更通用）       │
  └────────────────────────────────────┴────────────────────────────────────┘

什么是思维链（Chain-of-Thought, CoT）？
  让模型在最终答案前显式输出推理步骤，能显著提升复杂任务（数学、逻辑）的准确率。
  原理：
    - 简单问题：一步出答案没问题
    - 复杂问题：直接出答案的"概率密度"分散，模型容易猜错
    - 给模型"思考空间"，让它一步步分解，每步都更确定

OpenAI o1 / DeepSeek-R1 的训练秘诀：
  通过 RL（如 GRPO）让模型学会自主使用更长的思维链来解决复杂问题，
  而不是直接给答案。本脚本演示的就是这种模型的推理形态。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 第一部分：模型加载
# ============================================================
# 这里加载的是 GRPO 训练后的 checkpoint，不是基础模型
# checkpoint-1041 表示训练了 1041 步后的快照
model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/grpo/checkpoint-1041"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,   # bf16 推理省显存
    device_map="auto",            # 自动分配设备
    # 这里没用 flash_attention_2，所以兼容性更好（旧 GPU 也能跑）
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ============================================================
# 第二部分：System Prompt（教模型按 R1 格式输出）
# ============================================================
# 这个 system prompt 在 GRPO 训练时也用了同样的版本
# 一致性很重要：训练时啥格式，推理时也得是啥格式，不然模型可能"懵"
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
# 翻译：
#   "用户和助手的对话。用户提问，助手解答。
#    助手先在脑子里思考推理过程，然后给用户答案。
#    推理过程和答案分别包裹在 <think>...</think> 和 <answer>...</answer> 标签里。"

# ============================================================
# 第三部分：交互式对话循环
# ============================================================
while True:
    user_input = input("user：")

    # ─── 构造完整的 ChatML 格式 prompt ───
    # 三段式：system → user → assistant（待续写）
    # 与 qwen_chat.py 的区别：多了 system 段
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # ─── 编码 ───
    # 注意这里传入 [text] 是个 list（即使只有一条），返回的 input_ids 是 [1, seq_len]
    # 这种写法天然支持以后扩展为批量推理（多条同时跑）
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # ─── 生成 ───
    # max_new_tokens=4096：思维链可能很长，要给足空间
    # 数学推理常常需要 1000+ token，预留 4096 比较稳
    generated_ids = model.generate(**model_inputs, max_new_tokens=4096)

    # ─── 切掉 prompt 部分 ───
    # 这是个列表推导，处理批量情况：
    #   对于每一对 (input_ids, output_ids)，从 input 长度处往后切
    #   单条样本时等价于 outputs[0][inputs.input_ids.shape[-1]:]
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # ─── 批量解码 ───
    # batch_decode 接受 List[Tensor]，返回 List[str]
    # 比 decode 更通用：单条/批量都能处理
    # [0] 取第一条（也是唯一一条）
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 输出会自带 <think>...</think><answer>...</answer> 结构
    print("assistant：", response)
