"""
Qwen 模型基础对话脚本（最简推理版）
======================================================

这是模型训练完成后"用起来"的最简形态：单轮对话推理。
适合用来快速验证模型是否能正常生成、SFT 后效果如何。

整体流程：
  ┌────────────────────────────────────────────────────────┐
  │  ① 加载模型与分词器（bf16 + Flash Attention 加速）       │
  │  ② 进入交互循环：                                        │
  │     a. 接收用户输入                                      │
  │     b. 包装成 ChatML 格式                                │
  │     c. tokenizer 编码 → tensor                           │
  │     d. model.generate 生成                               │
  │     e. 切掉 prompt 部分，只 decode 新生成的 token        │
  │     f. 打印答案                                          │
  └────────────────────────────────────────────────────────┘

ChatML 格式（Qwen 系列使用的对话模板）：
  <|im_start|>system
  系统提示<|im_end|>
  <|im_start|>user
  用户问题<|im_end|>
  <|im_start|>assistant
  助手回答<|im_end|>

  - <|im_start|> / <|im_end|> 是特殊 token（在词表里有专门的 ID）
  - 模型在 SFT 阶段学会了：看到 "<|im_start|>assistant\n" 后开始生成回答
  - 看到 <|im_end|> 时停止
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 第一部分：模型加载
# ============================================================
# 模型目录：Qwen2.5-0.5B 基础模型（也可以换成你 SFT/DPO 后的 checkpoint）
model_dir = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"

# from_pretrained 加载模型权重
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    # bfloat16：推理时用半精度即可（节省一半显存，速度更快，精度足够）
    torch_dtype=torch.bfloat16,
    # Flash Attention 2：高效注意力实现
    #   - 把 attention 计算融合成一个 CUDA kernel
    #   - 显存从 O(N²) 降到 O(N)，速度快 2~3 倍
    #   - 需要 GPU 计算能力 ≥ 8.0（Ampere 架构及以上）
    attn_implementation="flash_attention_2",
    # device_map="auto"：自动把模型权重分配到所有可见 GPU 上
    # 单卡时全部放在 cuda:0，多卡时按显存大小切分
    device_map="auto",
)

# Tokenizer 把字符串和 token id 互相转换
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# ============================================================
# 第二部分：交互式对话循环
# ============================================================
while True:
    # 接收用户输入（input 会阻塞等待回车）
    user_input = input("question：")

    # ─── 构造 ChatML 格式的 prompt ───
    # 注意末尾是 "<|im_start|>assistant\n"
    # 这是个"半句话"——告诉模型"现在该你说话了，从这里接着写"
    # 模型会从 \n 后开始 token-by-token 生成助手回答
    formatted_input = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    # ─── 编码：字符串 → token id tensor ───
    # return_tensors="pt"：返回 PyTorch tensor（"pt" = PyTorch）
    # .to(model.device)：把 tensor 搬到模型所在的设备（GPU）
    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

    # ─── 生成 ───
    # **inputs 解包后传入：input_ids 和 attention_mask
    # max_new_tokens=128：最多新生成 128 个 token（不算 prompt 部分）
    #
    # 默认采样策略：
    #   - 没指定 do_sample → 走贪心解码（每步取概率最大的 token）
    #   - 想多样化输出可以加：do_sample=True, temperature=0.7, top_p=0.9
    #
    # 停止条件：
    #   - 生成到 EOS token（<|im_end|>）自动停
    #   - 达到 max_new_tokens 强制停
    outputs = model.generate(**inputs, max_new_tokens=128)

    # ─── 切掉 prompt 部分 ───
    # outputs[0] 形状是 [prompt_len + new_tokens]
    # 我们只想看新生成的部分，所以从 prompt 长度处切片
    # inputs.input_ids.shape[-1] 就是 prompt 的 token 长度
    response_tokens = outputs[0][inputs.input_ids.shape[-1]:]

    # ─── 解码：token id → 字符串 ───
    # skip_special_tokens=True：跳过 <|im_end|> 等特殊 token
    # 不加这个会看到一堆 <|im_end|><|endoftext|> 后缀
    answer = tokenizer.decode(response_tokens, skip_special_tokens=True)

    print("response：", answer)
