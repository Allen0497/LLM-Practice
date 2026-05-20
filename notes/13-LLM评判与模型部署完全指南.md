# 🚀 LLM 评判与模型部署完全指南

> 本文档配合 `qwen_judge.py`、`qwen_chat.py`、`qwen_r1_chat.py`、`utils/merge_peft_adapter.py` 四份代码，
> 把"训练完成 → 评测胜率 → LoRA 合并 → 推理部署"的最后一公里串起来。
> 假设你已经完成了从预训练到 Agentic RAG 的全部 12 篇学习。

---

## 📋 目录

- [第零章：本篇在大模型流程中的位置](#第零章本篇在大模型流程中的位置)
- [第一章：为什么需要 LLM-as-a-Judge？](#第一章为什么需要-llm-as-a-judge)
- [第二章：胜率评测原理（qwen_judge.py）](#第二章胜率评测原理qwen_judgepy)
- [第三章：vLLM 高速推理引擎](#第三章vllm-高速推理引擎)
- [第四章：ChatML 模板与基础对话（qwen_chat.py）](#第四章chatml-模板与基础对话qwen_chatpy)
- [第五章：思维链对话与 R1 格式（qwen_r1_chat.py）](#第五章思维链对话与-r1-格式qwen_r1_chatpy)
- [第六章：LoRA 适配器合并（merge_peft_adapter.py）](#第六章lora-适配器合并merge_peft_adapterpy)
- [第七章：完整部署链路](#第七章完整部署链路)
- [附录：常见问题](#附录常见问题)

---

## 第零章：本篇在大模型流程中的位置

```
┌────────────────────────────────────────────────────────────────────────┐
│                     大模型训练 + 应用完整流程                            │
│                                                                        │
│   ① 预训练（PT）                    ✅ 第 1 篇                         │
│   ② Qwen 网络结构                    ✅ 第 2 篇                         │
│   ③ 续训练                           ✅ 第 3 篇                         │
│   ④ SFT                              ✅ 第 4 篇                         │
│   ⑤ 大模型评估（Benchmark 类）       ✅ 第 5 篇                         │
│   ⑥ DPO 数据合成                     ✅ 第 6 篇                         │
│   ⑦ DPO                              ✅ 第 7 篇                         │
│   ⑧ RM                               ✅ 第 8 篇                         │
│   ⑨ PPO                              ✅ 第 9 篇                         │
│   ⑩ GRPO                             ✅ 第 10 篇                        │
│   ⑪ 知识蒸馏                         ✅ 第 11 篇                        │
│   ⑫ Agentic RAG                      ✅ 第 12 篇                        │
│  ─── 训练完成的模型，怎么真正用起来？───                              │
│   ⑬ LLM 评判 + 部署                  ← 📍 我们在这里                   │
│       ├ LLM-as-a-Judge 胜率评测       (qwen_judge.py)                  │
│       ├ ChatML 推理                   (qwen_chat.py)                   │
│       ├ 思维链推理                    (qwen_r1_chat.py)                │
│       └ LoRA 合并到基座               (merge_peft_adapter.py)          │
└────────────────────────────────────────────────────────────────────────┘
```

**类比：** 第 5 篇评估是"考试卷面分"（标准化考试），本篇评估是"现场对决看谁更强"（人工/AI 裁判看综合表现）；剩下三个脚本则是模型从"代码里"走到"线上"的最后步骤。

---

## 第一章：为什么需要 LLM-as-a-Judge？

### 1.1 第 5 篇 Benchmark 评测的局限

```
传统 benchmark（MMLU / CMMLU / C-Eval / GSM8K）：
  ✓ 客观（A/B/C/D 选项或精确字符串匹配）
  ✓ 可复现（输入固定，分数固定）
  ✗ 只能考"知识题、推理题"
  ✗ 没法评"开放任务"：写文章、对话、总结、代码风格、安全性、风格一致性
```

例如同一个总结任务：

| 模型 A 输出 | 模型 B 输出 |
|---|---|
| "这篇文章讲了 AI 的现状和未来" | "AI 进入 2025 年后能力跃升明显，本文从训练范式、推理框架两条线展开论述。" |

直觉上 B 显然更好，但 benchmark 没办法给"好坏分"。

### 1.2 LLM-as-a-Judge 的思路

> **找一个比被评估模型更强的大模型，让它当裁判。**

```
裁判模型（如 GPT-4o / Llama-3-70B）
        │
        ├─ 看到 [问题, 回答A, 回答B]
        ├─ 给出判断："A 更好" 或 "B 更好"
        ├─ 反复评估很多样本
        └─ 累计胜率（Win Rate）
```

### 1.3 胜率（Win Rate）的含义

```
被评估模型 vs 参考答案（人工写的或某个基线模型）

Win Rate = 被评估模型胜过参考答案的样本数 / 总样本数 × 100%

解读：
  < 50%  →  被评估模型不如参考答案
  = 50%  →  打平
  > 50%  →  超过参考答案
```

论文里的典型数字（TLDR 总结任务）：

| 训练阶段 | Win Rate（vs 人工总结） |
|---|---|
| 仅 SFT | ~25-35% |
| SFT + DPO | ~50-60% |
| SFT + RM + PPO | ~55-70% |
| GPT-4 | ~80%+ |

### 1.4 LLM 裁判的局限性

```
⚠ 偏见 1：位置偏差（Position Bias）
   裁判倾向选第一个出现的回答（或最后一个，模型不同偏好不同）
   解决：每个样本评估两次，A/B 顺序交换求平均

⚠ 偏见 2：长度偏差（Length Bias）
   裁判倾向选更长的回答（即使内容质量一样）
   解决：在 prompt 里加约束"长短不影响判断"

⚠ 偏见 3：自我偏好（Self Preference）
   GPT-4 当裁判时倾向于给 GPT 风格的回答更高分
   解决：用多个不同家族的模型当裁判取平均

⚠ 偏见 4：表面优势
   写得花哨但事实错误的回答可能被打高分
   解决：在 prompt 里强调"准确性 > 风格"
```

trl 库的 `PairwiseJudge` 内部已经做了部分消偏处理。

---

## 第二章：胜率评测原理（qwen_judge.py）

### 2.1 完整流程

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│   TLDR 数据集（Reddit 帖子 + 人工总结）                    │
│              │                                            │
│              ├──► prompt（要总结的文本）                   │
│              └──► reference completion（人工总结）         │
│                                                           │
│   prompt ──► 被评估模型 ──► model completion              │
│                  ▲                                        │
│                  └──── vLLM 加速批量推理                  │
│                                                           │
│   [reference, model_completion] ──► PairwiseJudge          │
│                                            │              │
│                                            ▼              │
│                                       裁判返回 0 或 1     │
│                                       (0=ref 赢, 1=model 赢)│
│                                                           │
│   统计 → Win Rate                                          │
└───────────────────────────────────────────────────────────┘
```

### 2.2 代码核心拆解

```python
# 1. 加载数据集（取前 N 条快速测试）
dataset = load_dataset("trl-lib/tldr", split="validation")
prompts = dataset["prompt"]
reference_completions = dataset["completion"]

# 2. 用 vLLM 批量生成模型回答
sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
llm = LLM(model=model_path, tensor_parallel_size=1)
outputs = llm.generate(prompts, sampling_params)
model_completions = [o.outputs[0].text.strip() for o in outputs]

# 3. 配对：[reference, model_completion]
completions = [[c0, c1] for c0, c1 in zip(reference_completions, model_completions)]

# 4. 调用裁判
judge = HfPairwiseJudge(judge_model)  # 或 OpenAIPairwiseJudge
best_idxs = judge.judge(prompts, completions)

# 5. 统计胜率
model_win_rate = best_idxs.count(1) / len(best_idxs)
```

### 2.3 关键参数选择

```
temperature=0.0
  → 贪心解码，保证可复现
  → 评测时绝对不要用采样（每次结果不同没法比）

max_tokens=200
  → TLDR 总结一般几十词
  → 太短会截断，影响评测公平性

裁判模型
  → 强烈推荐 GPT-4 / Claude / DeepSeek-V3 这类强模型
  → 弱裁判 = 噪声大 = 评测结果不可信

样本数
  → 100 条：调试看流程
  → 500 条：大致趋势
  → 1000+ 条：发表论文级别
```

### 2.4 两种 PairwiseJudge 对比

```
┌──────────────────────┬──────────────────────────┬──────────────────────────┐
│                      │  HfPairwiseJudge         │  OpenAIPairwiseJudge     │
├──────────────────────┼──────────────────────────┼──────────────────────────┤
│  内部用什么           │  本地 HF 模型 + vLLM      │  OpenAI API              │
│  默认裁判             │  Llama-3-70B-Instruct    │  gpt-4o-mini             │
│  显存需求             │  70B 模型≥160GB          │  零（远端推理）          │
│  费用                 │  电费                     │  按 token 收费            │
│  数据隐私             │  完全本地                 │  传给 OpenAI              │
│  适用场景             │  公司内部数据             │  公开数据、快速实验      │
└──────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## 第三章：vLLM 高速推理引擎

### 3.1 为什么需要 vLLM？

评测一个模型常常要跑 **��千个 prompt**，用 HuggingFace transformers 原生 `generate` 慢得感人。

```
对比测试（Qwen2.5-7B，1000 prompts，A100）：
  HuggingFace generate（默认）  → 约 25 分钟
  HuggingFace generate（batch=8）→ 约 8 分钟
  vLLM                          → 约 1.5 分钟

vLLM 比原生快 ~16 倍
```

### 3.2 vLLM 的两个核心创新

#### 创新 1：PagedAttention

```
传统 KV Cache 管理：
  - 每个请求预分配最大长度的连续显存
  - 实际只用一部分 → 大量碎片浪费

PagedAttention（借鉴操作系统的虚拟内存）：
  - 把 KV Cache 切成固定大小的"页"（page）
  - 按需分配页面，不连续也行
  - 显存利用率从 ~20% 提升到 ~95%
  → 同样显存能跑更大 batch
```

#### 创新 2：Continuous Batching

```
传统 Static Batching：
  Batch 启动后大家一起跑，一起结束
  长样本拖慢短样本，GPU 大量时间在等

Continuous Batching：
  请求级别的动态调度
  谁结束了立刻塞新的请求进 batch
  GPU 一直保持高利用率
```

### 3.3 vLLM 基本用法

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="path/to/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,        # 单卡=1，多卡=GPU 数量
    gpu_memory_utilization=0.9,    # 显存利用率上限
    max_model_len=4096,            # 最大上下文长度
)

# 配置生成参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop=["<|im_end|>"],           # 遇到这个 token 停止
)

# 批量生成
prompts = ["问题1", "问题2", "问题3", ...]
outputs = llm.generate(prompts, sampling_params)

# 提取结果
for output in outputs:
    print(output.outputs[0].text)
```

### 3.4 部署成 OpenAI 兼容 API

```bash
vllm serve /path/to/Qwen2.5-7B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096
```

启动后可用 OpenAI Python SDK 调用：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

---

## 第四章：ChatML 模板与基础对话（qwen_chat.py）

### 4.1 ChatML 是什么？

> Qwen / OpenAI 等系列使用的对话模板，统一用特殊 token 标记角色和边界。

```
完整格式（多轮对话示例）：
  <|im_start|>system
  你是一个有帮助的助手<|im_end|>
  <|im_start|>user
  你好<|im_end|>
  <|im_start|>assistant
  你好！有什么可以帮你的？<|im_end|>
  <|im_start|>user
  介绍一下 Python<|im_end|>
  <|im_start|>assistant
  Python 是一种...<|im_end|>
```

### 4.2 关键特殊 token

| Token | 含义 | 在词表里的 ID（Qwen2.5）|
|---|---|---|
| `<|im_start|>` | 角色开始 | 151644 |
| `<|im_end|>` | 角色结束 | 151645 |
| `<|endoftext|>` | 文档结束 | 151643 |

这些 token **不会被拆成多个子词**，整体作为一个 token 处理。SFT 训练时模型学会了"看到 `<|im_start|>assistant\n` 就开始扮演助手"。

### 4.3 推理时的 prompt 构造

```python
# 单轮对话
text = (
    f"<|im_start|>user\n{user_input}<|im_end|>\n"
    f"<|im_start|>assistant\n"   # ← 末尾不加 <|im_end|>，让模型续写
)
```

⚠ **关键：assistant 标签只开不关**

模型会从 `<|im_start|>assistant\n` 后面接着写，直到生成 `<|im_end|>` 自动停止。
如果你把 `<|im_end|>` 也写上了，模型会以为"对话已经结束"，啥也不生成。

### 4.4 为什么要切掉 prompt 部分？

```python
outputs = model.generate(**inputs, max_new_tokens=128)
# outputs[0] 形状 [prompt_len + new_tokens]
# 包含原始 prompt + 新生成内容

response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
# 切掉前面的 prompt，只保留新生成的部分
```

不切掉的话，decode 出来会看到"用户问题 + 模型回答"的完整 prompt，浪费输出。

### 4.5 推理加速选项

```python
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,         # ✅ 半精度，显存减半
    attn_implementation="flash_attention_2",  # ✅ Flash Attention 2，提速 2-3x
    device_map="auto",                  # ✅ 自动放卡
)
```

| 优化项 | 显存影响 | 速度影响 | 适用条件 |
|---|---|---|---|
| `bfloat16` | -50% | +30% | 几乎所有现代 GPU |
| `flash_attention_2` | -30% | +200% | A100/H100/B300 (≥sm80) |
| `device_map="auto"` | 多卡分担 | 中性 | 多 GPU |
| 量化（int8/int4） | -75% | +20% | 显存极度紧张时 |

---

## 第五章：思维链对话与 R1 格式（qwen_r1_chat.py）

### 5.1 思维链（CoT）原理

```
普通回答：
  Q: 一个篮子有 3 个苹果，吃掉 1 个，又买了 5 个，现在多少？
  A: 7 个

思维链回答：
  Q: 一个篮子有 3 个苹果，吃掉 1 个，又买了 5 个，现在多少？
  A: <think>
       开始有 3 个
       吃掉 1 个 → 还剩 3 - 1 = 2 个
       又买 5 个 → 现在 2 + 5 = 7 个
     </think>
     <answer>7 个</answer>
```

**为什么 CoT 有效？**

数学上等价于：让模型用更多 token 来"思考"，每个 token 的"决策范围"更小，错误率更低。

实证：在 GSM8K 上，简单 CoT prompt 能让模型准确率翻倍。

### 5.2 与基础对话的区别

```python
# qwen_chat.py（基础版）
text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
max_new_tokens = 128

# qwen_r1_chat.py（R1 版）
text = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"   # ← 额外 system
    f"<|im_start|>user\n{q}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
max_new_tokens = 4096   # ← 思考要占更多 token
```

### 5.3 R1 格式的 system prompt

```
A conversation between User and Assistant. The user asks a question, and the
Assistant solves it. The assistant first thinks about the reasoning process in
the mind and then provides the user with the answer. The reasoning process and
answer are enclosed within <think> </think> and <answer> </answer> tags,
respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
```

**关键约束：**

```
1. 必须先 <think> 后 <answer>（不能反过来）
2. 两个标签都必须闭合
3. 推理时的 system prompt 要和训练时（GRPO）一致
```

### 5.4 batch_decode vs decode

```python
# 单条样本：两种都行
ans1 = tokenizer.decode(output_ids[0], skip_special_tokens=True)
ans2 = tokenizer.batch_decode([output_ids[0]], skip_special_tokens=True)[0]

# 批量样本：必须用 batch_decode
ans_list = tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)
```

`batch_decode` 等价于 `[decode(x) for x in batch]`，但内部更高效。

### 5.5 思维链推理的常见坑

```
坑 1：max_new_tokens 太小
  现象：思考被截断，没有 <answer>
  解决：≥ 2048，复杂数学题用 4096+

坑 2：推理时不带 system prompt
  现象：模型直接给答案，不思考
  解决：必须加上和训练时一样的 system prompt

坑 3：训练用 R1 格式，推理用普通格式
  现象：模型生成乱七八糟
  解决：训练 / 推理 prompt 必须严格一致

坑 4：GRPO 训练步数不够
  现象：思维链格式有时对有时错
  解决：训练到 ≥ 1000 步，看 wandb 上 format reward 收敛
```

---

## 第六章：LoRA 适配器合并（merge_peft_adapter.py）

### 6.1 为什么 LoRA 要合并？

回顾 LoRA：

```
原始线性层：y = Wx
LoRA 改造：  y = Wx + (α/r) * B(Ax)
            其中 A: [r, d_in], B: [d_out, r]，r ≪ d_out, d_in
```

**不合并时：** 推理每一步都要算 `Wx + (α/r)·BAx`，多了一次矩阵乘法

**合并：** 提前算好 `W' = W + (α/r)·BA`，推理时只用 W'，速度恢复

### 6.2 合并的好处

```
✓ 推理速度恢复（不用多算 LoRA 那部分）
✓ 显存占用一致（不用多放 A、B 矩阵）
✓ 部署简单（vLLM、TensorRT-LLM、GGUF 都直接支持）
✓ 模型可以独立分发（不依赖原 base 模型）
```

### 6.3 合并的代价

```
✗ 失去多 adapter 切换能力
   不合并时：一个 base + N 个小 adapter，可以热切换不同任务
   合并后：每个任务一份完整权重，磁盘占用爆炸

✗ 失去继续训练的灵活性
   合并后只能基于合并模型继续训练
   不合并能继续训练 LoRA 参数
```

### 6.4 代码核心

```python
# 1. 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    BaseModelPath,
    return_dict=True,
    torch_dtype=torch.bfloat16,
)

# 2. 套上 LoRA adapter
model = PeftModel.from_pretrained(model, AdapterModelPath)
model.eval()   # 必须 eval，关掉 dropout 等

# 3. 一行合并 ★
model = model.merge_and_unload()

# 4. 保存
model.save_pretrained(OutputPath)
tokenizer.save_pretrained(OutputPath)
```

### 6.5 `merge_and_unload()` 的内部逻辑

```python
# 伪代码
def merge_and_unload(self):
    for name, module in self.named_modules():
        if isinstance(module, LoraLayer):
            # ΔW = (α / r) * B @ A
            delta_w = (module.scaling) * (module.lora_B.weight @ module.lora_A.weight)
            # W_new = W + ΔW
            module.base_layer.weight.data += delta_w.to(module.base_layer.weight.dtype)
            # 移除 LoRA 模块
            del module.lora_A, module.lora_B
    # 解包：从 PeftModel 拿回纯 transformers 模型
    return self.base_model.model
```

### 6.6 何时合并 / 何时保留 adapter？

```
✓ 合并：
  - 部署到生产环境（追求速度）
  - 上传到 HF Hub（用户可直接 from_pretrained）
  - 转换 GGUF / ONNX / TensorRT 格式
  - 给推理引擎（vLLM / SGLang / TensorRT-LLM）

✓ 保留 adapter：
  - 多任务切换（vLLM 支持 dynamic adapter loading）
  - 继续 LoRA 训练
  - 节省磁盘（一个 base + 多个 adapter）
```

### 6.7 合并后的部署链路

```
┌──────────────────────────────────────────────────────────────┐
│  base_model + adapter (LoRA checkpoint)                      │
│         │                                                    │
│         │ merge_peft_adapter.py                              │
│         ▼                                                    │
│  full_model (HuggingFace 格式)                                │
│         │                                                    │
│         ├──► vLLM serve 直接部署 (本项目推荐)                │
│         ├──► transformers AutoModelForCausalLM 加载           │
│         ├──► llama.cpp + GGUF 转换 (CPU/Mac 部署)            │
│         ├──► TensorRT-LLM (NVIDIA 极速部署)                  │
│         └──► SGLang (KV cache 复用最强)                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 第七章：完整部署链路

把本篇 4 个脚本串起来，就是大模型从训练到上线的完整最后一公里：

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   [LoRA 训练 SFT/RM/DPO]                                        │
│         │ → results/.../checkpoint-XXXX/  (adapter)            │
│         │                                                      │
│         ▼                                                      │
│   ① merge_peft_adapter.py                                      │
│     合并 LoRA → 完整模型                                       │
│         │                                                      │
│         ▼                                                      │
│   ② qwen_chat.py / qwen_r1_chat.py                             │
│     单机推理验证（看模型能不能正常说话）                        │
│         │                                                      │
│         ▼                                                      │
│   ③ qwen_judge.py                                              │
│     对比基线（vs 人工答案 / 旧版本模型）                       │
│     看胜率是否上涨                                             │
│         │                                                      │
│         ▼                                                      │
│   ④ vLLM serve                                                  │
│     上线生产环境 OpenAI 兼容 API                              │
│         │                                                      │
│         ▼                                                      │
│   ⑤ 接入 Agentic RAG / 业务系统（第 12 篇）                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 一个完整的发布流程

```bash
# Step 1：合并 LoRA
python utils/merge_peft_adapter.py
# 输出：results/rm/final_model/

# Step 2：本地推理验证
python qwen_chat.py
# 手工敲几个问题看回答是否合理

# Step 3：胜率评测（如果是聊天/写作模型）
python qwen_judge.py \
    --model_name_or_path results/rm/final_model \
    --judge_model gpt-4o-mini \
    --num_examples 500
# 看到 Win rate > 50% 才算合格

# Step 4：vLLM 启动 OpenAI API
vllm serve results/rm/final_model --port 8000

# Step 5：跑业务（如 Agentic RAG）
# 把第 12 篇的 OpenAIServerModel 的 api_base 改成 http://localhost:8000/v1
```

---

## 附录：常见问题

### Q1：胜率多少算合格？

> 完全没有"合格线"，只有"对比基线"。
> 要看你想超过谁：
> - **超过 SFT 基线**：DPO/PPO 至少应该让胜率从 30% → 50%+
> - **超过人工**：能做到 60%+ 已经很强
> - **超过 GPT-3.5**：算商业可用
> - **超过 GPT-4**：开源 SOTA 水平

### Q2：HfPairwiseJudge 显存不够怎么办？

> 三个选项：
> 1. 换更小的裁判（如 Qwen2.5-32B-Instruct）—— 但裁判越弱噪声越大
> 2. 切换到 OpenAIPairwiseJudge —— 用 API（要钱）
> 3. 量化（int8/int4 加载 70B）—— 速度还能保持，效果略降

### Q3：评测的样本数选多少？

| 样本数 | 用途 | 95% 置信区间宽度（约） |
|---|---|---|
| 100 | 调试流程 | ±10% |
| 500 | 内部评估 | ±4.5% |
| 1000 | 论文初稿 | ±3% |
| 5000+ | 论文正式 | ±1.5% |

> 经验：先 100 跑通，再 1000 出正式数字。

### Q4：vLLM 报 OOM（显存爆了）怎么办？

```python
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.85,   # 默认 0.9，调小一点
    max_model_len=2048,            # 默认会读 config，主动限制更短
    dtype="bfloat16",              # 显式指定
    swap_space=4,                  # 4GB CPU swap，缓 OOM
)
```

### Q5：基础对话和思维链对话用同一个模型可以吗？

> 看模型怎么训的：
> - 仅 SFT 的模型：用基础对话，没思维链能力
> - GRPO/R1 风格训练的模型：必须用思维链 prompt，否则模型可能也会输出 `<think>` 但格式混乱
> - 通用模型（GPT-4 / Claude）：两种都能用，加 system prompt 就能切换

### Q6：merge_peft_adapter 报 device mismatch？

> 经常是因为基座 / adapter 在不同设备上：
> ```python
> # 都加载到 CPU 然后再合并
> model = AutoModelForCausalLM.from_pretrained(BaseModelPath, torch_dtype=torch.bfloat16, device_map="cpu")
> model = PeftModel.from_pretrained(model, AdapterModelPath)
> model = model.merge_and_unload()
> # 合并完再搬到 GPU 推理
> model = model.cuda()
> ```

### Q7：合并后的模型大小怎么变化？

| 阶段 | 大小（Qwen2.5-0.5B 例） |
|---|---|
| 基座 | 1.0 GB（bf16） |
| LoRA adapter（r=8） | ~5 MB |
| 合并后 | 1.0 GB（和基座一样大） |

合并 **不增加参数量**，只是把 ΔW 加到 W 里。

### Q8：怎么把合并后的模型上传 HuggingFace Hub？

```bash
# 1. 登录
huggingface-cli login

# 2. 在脚本里改成 push
model.push_to_hub("your-username/your-model-name", use_temp_dir=False)
tokenizer.push_to_hub("your-username/your-model-name", use_temp_dir=False)
```

记得在仓库里加 README.md，说明：基座是什么、训练数据、训练配置、评测胜率等。

### Q9：vLLM 之外还有哪些推理引擎？

| 引擎 | 优势 | 劣势 |
|---|---|---|
| **vLLM** | 易用、社区大、PagedAttention | 编译时间长、依赖较多 |
| **SGLang** | KV cache 复用最强、speculative decoding | 生态略小 |
| **TensorRT-LLM** | NVIDIA 卡上极致速度 | 编译复杂、模型支持滞后 |
| **TGI** (Text Generation Inference) | HF 出品，与 transformers 衔接好 | 速度比 vLLM 略慢 |
| **llama.cpp** | CPU/Mac 也能跑、GGUF 量化 | 不适合服务大并发 |
| **ollama** | 一键启动、本地玩 | 性能不如 vLLM |

---

## 🎉 学习清单：本篇核心知识

- [x] 为什么需要 LLM-as-a-Judge：开放任务无标准答案
- [x] 胜率（Win Rate）的含义和参考数值
- [x] LLM 裁判的 4 个偏见以及消偏方法
- [x] vLLM 的两个核心创新：PagedAttention + Continuous Batching
- [x] vLLM 比原生 transformers 快 5-24 倍
- [x] ChatML 格式的特殊 token：`<|im_start|>` / `<|im_end|>`
- [x] assistant 标签在 prompt 里只开不关，让模型续写
- [x] 思维链（CoT）的原理：让模型用更多 token 思考
- [x] R1 格式的 system prompt 必须和训练时一致
- [x] LoRA 合并：W' = W + (α/r)·BA
- [x] `merge_and_unload()` 的内部逻辑
- [x] 完整部署链路：训练 → 合并 → 验证 → 评测 → 上线

---

## 🚀 下一步学习方向

到这里你已经把 mini_qwen 项目的全部代码学完了！整个大模型训练 + 应用的闭环：

```
预训练 → SFT → DPO/PPO/GRPO → 蒸馏 → 评估 → 合并 → 部署 → Agentic 应用
```

进阶方向：

1. **显存优化实战**（可选笔记 14 篇）
   - `qwen_mem.py`：CUDA Memory Summary 解读
   - 梯度累积、混合精度、DeepSpeed ZeRO 阶段差异
   - 选择合适的 batch_size / sequence_length

2. **更大规模训练**
   - DeepSpeed ZeRO Stage 3 / FSDP（数百亿参数）
   - 流水线并行 / 张量并行（分布式训练）
   - 训练数据规模化（清洗、去重、配比）

3. **更强的 RL 算法**
   - DAPO、SimPO 等新偏好优化方法
   - Process Reward Model（步骤级奖励）
   - Tool Use RL（让模型学会调用工具）

4. **生产级部署**
   - 多模型路由（小模型先看 → 难题给大模型）
   - 推理缓存��Redis 存常见问答）
   - 监控（Prometheus + Grafana 看延迟、QPS、错误率）

5. **多模态扩展**
   - Qwen2-VL：视觉 + 语言
   - Whisper：语音输入
   - 配套训练数据流水线

恭喜你走完了大模型训练全流程！🎓
