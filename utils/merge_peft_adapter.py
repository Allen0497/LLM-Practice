"""
PEFT/LoRA 适配器合并脚本
======================================================

LoRA 训练完成后，会得到一个"小补丁"（adapter，几十 MB），需要和原始基座模型
（几百 MB ~ 几十 GB）合并，得到一个完整的、可独立推理的模型。

为什么需要合并？
  ┌──────────────────────────────────────────────────────────────────┐
  │  不合并的情况：                                                   │
  │    推理时要做：base_model + adapter → 加载两次，前向多一次计算    │
  │    每个 LoRA 层都有 W + ΔW（=BA）的拼接，速度慢 5~10%             │
  │                                                                  │
  │  合并后：                                                        │
  │    把 ΔW 加到 W 里：W' = W + α*BA                                │
  │    得到一个普通的稠密模型，加载和推理跟原模型完全一样             │
  │    部署到 vLLM、TensorRT-LLM 等推理引擎更省心                    │
  └──────────────────────────────────────────────────────────────────┘

LoRA 数学回顾（更详细的看第 7 篇 DPO 笔记）：
  原始线性层：y = Wx           （W 形状 [d_out, d_in]）
  LoRA 改造：  y = Wx + (α/r) * B(Ax)
              其中 A: [r, d_in], B: [d_out, r]，r 远小于 d_out, d_in
              训练时只更新 A 和 B（参数量 ≈ r * (d_in + d_out)）

  合并：W_merged = W + (α/r) * BA
        合并后只剩一个 W_merged，推理快、部��简单

合并后的"代价"：
  - 失去切换不同任务的能力（多个 LoRA 不能动态切换）
  - 失去重新合并的可能（除非保留原始 adapter）
  → 建议：保留 adapter 文件夹（小），同时保留合并后的模型用于部署

本脚本流程：
  ┌──────────────────────────────────────────┐
  │  ① 加载基座模型（bf16 节省显存）          │
  │  ② 在基座模型上挂载 LoRA adapter          │
  │  ③ 调用 merge_and_unload() 合并           │
  │  ④ 保存合并后的模型 + tokenizer           │
  └──────────────────────────────────────────┘
"""

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 第一部分：路径配置
# ============================================================
# AdapterModelPath：LoRA 训练保存的 checkpoint 目录
# 里面有 adapter_config.json + adapter_model.safetensors
# 注意：这个目录只有 LoRA 参数（A 和 B 矩阵），没有完整模型权重
AdapterModelPath = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/rm/Qwen2.5-0.5B_peft_stack-exchange-paired__0_2e-05/checkpoint-1908"

# BaseModelPath：原始基座模型
# 必须和 LoRA 训练时用的基座一致（adapter_config.json 里有记录）
BaseModelPath = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/Qwen2.5-0.5B"

# OutputPath：合并后的完整模型保存路径
OutputPath = "results/rm/final_model"

# ============================================================
# 第二部分：加载基座模型
# ============================================================
# 注意：基座模型加载时不要用 device_map="auto"
# 因为后面要 merge_and_unload，多卡分片会让合并出问题
# 这里默认加载到 CPU 或单卡上
model = AutoModelForCausalLM.from_pretrained(
    BaseModelPath,
    return_dict=True,           # forward 返回 dict 而不是 tuple（更易用）
    torch_dtype=torch.bfloat16, # 用 bf16 节省内存
)

# 加载 tokenizer（合并模型也要带 tokenizer 才能完整使用）
tokenizer = AutoTokenizer.from_pretrained(BaseModelPath)

# ============================================================
# 第三部分：在基座模型上挂载 LoRA adapter
# ============================================================
# PeftModel.from_pretrained 做的事：
#   1. 读取 AdapterModelPath/adapter_config.json，知道哪些层有 LoRA
#   2. 把 LoRA 的 A、B 矩阵注入到对应层
#   3. 包装成一个 PeftModel 对象，forward 时自动加上 LoRA 修正
model = PeftModel.from_pretrained(model, AdapterModelPath)

# 切换到 eval 模式：关闭 dropout、关闭 LoRA 的训练用 scaling 等
# 合并前必须 eval，否则可能合错
model.eval()

# ============================================================
# 第四部分：合并 LoRA 到基座 ★★★
# ============================================================
# merge_and_unload() 是 PEFT 库的核心 API
#   merge：对每个 LoRA 层，计算 W_merged = W + (α/r) * BA
#   unload：移除 PeftModel 的包装，返回纯净的 transformers 模型
#
# 等价的伪代码：
#   for layer in model.layers:
#       if has_lora(layer):
#           layer.weight.data += (alpha/r) * (layer.lora_B @ layer.lora_A)
#           del layer.lora_A, layer.lora_B
#   return base_model
model = model.merge_and_unload()

# ============================================================
# 第五部分：保存合并后的模型
# ============================================================
# save_pretrained 会保存：
#   - config.json：模型配置
#   - model.safetensors（或 pytorch_model.bin）：权重
#   - generation_config.json：生成默认参数
model.save_pretrained(f"{OutputPath}")
tokenizer.save_pretrained(f"{OutputPath}")

# 现在 OutputPath 里就是一个标准的 HF 模型，可以：
#   - 用 transformers AutoModelForCausalLM.from_pretrained 直接加载
#   - 用 vLLM 高速部署
#   - 用 GGUF 工具转换给 llama.cpp 用
#   - 上传到 HuggingFace Hub

# 上传 HF（需要先 huggingface-cli login，且 OutputPath 是合法的仓库名）
# model.push_to_hub(f"{OutputPath}", use_temp_dir=False)
