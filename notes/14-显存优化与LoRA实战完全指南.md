# 💾 显存优化与 LoRA 实战完全指南

> 本文档配合 `qwen_mem.py` 与 `utils/utils.py` 中的 `clear_memory` / `print_trainable_parameters`，
> 把"训练时显存到底花在哪、怎么省、OOM 怎么排查"一次讲透。
> 这是 mini_qwen 项目的收官篇，也是大模型工程师的必修课。

---

## 📋 目录

- [第零章：本篇在大模型流程中的位置](#第零章本篇在大模型流程中的位置)
- [第一章：GPU 显存到底花在哪？](#第一章gpu-显存到底花在哪)
- [第二章：估算公式（看 model 知显存）](#第二章估算公式看-model-知显存)
- [第三章：torch.cuda.memory_summary 怎么读](#第三章torchcudamemory_summary-怎么读)
- [第四章：四把"省显存"利器](#第四章四把省显存利器)
- [第五章：LoRA 为什么这么省？](#第五章lora-为什么这么省)
- [第六章：DeepSpeed ZeRO 三阶段](#第六章deepspeed-zero-三阶段)
- [第七章：qwen_mem.py 代码逐段拆解](#第七章qwen_mempy-代码逐段拆解)
- [第八章：OOM 排查方法论](#第八章oom-排查方法论)
- [附录：常见问题](#附录常见问题)

---

## 第零章：本篇在大模型流程中的位��

```
┌────────────────────────────────────────────────────────────────────────┐
│                     大模型训练 + 应用完整流程                            │
│                                                                        │
│   ① 预训练（PT）                    ✅ 第 1 篇                         │
│   ② Qwen 网络结构                    ✅ 第 2 篇                         │
│   ③ 续训练                           ✅ 第 3 篇                         │
│   ④ SFT                              ✅ 第 4 篇                         │
│   ⑤ 大模型评估                       ✅ 第 5 篇                         │
│   ⑥-⑪ DPO / RM / PPO / GRPO / 蒸馏  ✅ 第 6-11 篇                     │
│   ⑫ Agentic RAG                      ✅ 第 12 篇                       │
│   ⑬ LLM 评判 + 部署                  ✅ 第 13 篇                       │
│  ─── 工程师必修课 ───                                                 │
│   ⑭ 显存优化与 LoRA 实战             ← 📍 我们在这里                   │
└────────────────────────────────────────────────────────────────────────┘
```

**学完前 13 篇你已经会"训"模型，本篇让你会"训得动"模型。**

显存不够是大模型训练第一大拦路虎。模型越大、序列越长、batch 越大 → 显存爆炸。
本篇会让你彻底搞明白：
- 显存为什么会爆？
- 每一招（LoRA / 混合精度 / 梯度累积 / DeepSpeed）怎么省？
- 报 OOM 时第一反应该做什么？

---

## 第一章：GPU 显存到底花在哪？

### 1.1 训练时显存的 5 大去向

```
┌────────────────────────────────────────────────────────────┐
│  GPU 总显存                                                 │
│                                                            │
│  ┌────────────────┐  ① 模型参数（Parameters）              │
│  │ Parameters     │     例：7B 模型 fp32 = 28 GB           │
│  │                │                                        │
│  ├────────────────┤  ② 梯度（Gradients）                   │
│  │ Gradients      │     大小 = 参数大小                    │
│  │                │     例：7B fp32 = 28 GB                 │
│  ├────────────────┤                                        │
│  │ Optimizer      │  ③ 优化器状态（Optimizer States）      │
│  │ States         │     Adam: 2x params (m + v)             │
│  │ (m, v)         │     7B fp32 = 56 GB                    │
│  ├────────────────┤                                        │
│  │ Activations    │  ④ 激活值（Activations）               │
│  │                │     反向传播要用，正向时缓存            │
│  │                │     ∝ batch × seq_len² × layers        │
│  ├────────────────┤                                        │
│  │ KV Cache /     │  ⑤ 临时缓冲（Kernel buffers）          │
│  │ Misc           │     CUDA kernel 临时空间、碎片等        │
│  └────────────────┘                                        │
└────────────────────────────────────────────────────────────┘
```

### 1.2 训练 vs 推理的差异

```
推理：
  - 只需要模型参数 + 当前推理的 KV Cache + 少量中间结果
  - 7B 模型 fp16 推理 ≈ 14 GB

训练：
  - 模型参数 + 梯度 + 优化器状态 + 全部激活
  - 7B 模型 fp32 训练 ≈ 28 + 28 + 56 + 激活 = 100+ GB
  → 显存翻 5-10 倍！
```

### 1.3 显存怎么理解：一个生活类比

```
你装修房子：
  ① Parameters    = 装修后的房子（永久占地）
  ② Gradients     = 修改方案的图纸（每次改前都要画）
  ③ Optimizer     = 工具箱（Adam 是大工具箱：动量 + 方差）
  ④ Activations   = 中间产生的施工废料（修完才清理）
  ⑤ Misc          = 工人摆放工具的临时空地

OOM = 总占地超过房子面积
```

---

## 第二章：估算公式（看 model 知显存）

### 2.1 速记公式

> **训练显存 ≈ 参数量 × 字节数 × (1 + 1 + 2) + 激活值**
>
> = 参数量 × 字节数 × 4 + 激活

| 数据类型 | 字节数 | 参数 1B 时显存（仅参数）|
|---|---|---|
| fp32 | 4 | 4 GB |
| fp16 / bf16 | 2 | 2 GB |
| int8 | 1 | 1 GB |
| int4 | 0.5 | 500 MB |

### 2.2 Adam 优化器为什么是 2x 参数？

Adam 给每个参数维护两个状态：
- 一阶动量 `m`（梯度的滑动平均，1x 参数）
- 二阶动量 `v`（梯度平方的滑动平均，1x 参数）

**两者通常以 fp32 存储**（即使用混合精度训练，状态也是 fp32 保证精度）。

```
7B 模型 + Adam + fp32：
  参数：7B × 4 = 28 GB
  梯度：7B × 4 = 28 GB
  Adam：7B × 4 × 2 = 56 GB
  ───────────────────────
  小计：112 GB（还没算激活）

激活：~20-40 GB（取决于 seq_len 和 batch）

总计：130-150 GB → 单张 A100 80GB 完全装不下
```

### 2.3 激活值的估算

激活值的精确估算很复杂，简化公式：

```
单个 transformer 层的激活 ≈ 12 × batch × seq_len × hidden_dim × layers × bytes

例：Qwen2.5-7B（hidden=4096, layers=32）
  batch=1, seq_len=2048, fp16：
  ≈ 12 × 1 × 2048 × 4096 × 32 × 2 ≈ 6.4 GB

  batch=4, seq_len=4096, fp16：
  ≈ 12 × 4 × 4096 × 4096 × 32 × 2 ≈ 51 GB
```

**关键观察：seq_len 翻倍，激活翻 2 倍；batch 翻倍，激活也翻 2 倍。**

### 2.4 速查表（fp16 + Adam，含激活粗估）

| 模型大小 | 推理 | 全参数训练 | LoRA 训练 |
|---|---|---|---|
| 0.5B | 1 GB | 8 GB | 2 GB |
| 1.5B | 3 GB | 24 GB | 5 GB |
| 7B | 14 GB | 110 GB | 22 GB |
| 13B | 26 GB | 200 GB | 38 GB |
| 70B | 140 GB | 1.1 TB | 180 GB |

> 7B 全参数训练在 80GB A100 上根本不够，必须 ZeRO/FSDP 多卡分摊或 LoRA。

---

## 第三章：torch.cuda.memory_summary 怎么读

### 3.1 一份典型输出

```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |   2148 MiB |   2580 MiB |   8120 MiB |   5972 MiB |
|       from large pool |   2120 MiB |   2540 MiB |   8000 MiB |   5880 MiB |
|       from small pool |     28 MiB |     40 MiB |    120 MiB |     92 MiB |
|---------------------------------------------------------------------------|
| Active memory         |   2148 MiB |   2580 MiB |   8120 MiB |   5972 MiB |
|---------------------------------------------------------------------------|
| Requested memory      |   2080 MiB |   2510 MiB |   8050 MiB |   5970 MiB |
|---------------------------------------------------------------------------|
| GPU reserved memory   |   3072 MiB |   3072 MiB |   3072 MiB |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |    924 MiB |   1100 MiB |   3940 MiB |   3016 MiB |
|---------------------------------------------------------------------------|
| Allocations           |        842 |       1023 |       3245 |       2403 |
|---------------------------------------------------------------------------|
```

### 3.2 关键字段含义

| 字段 | 含义 | 怎么看 |
|---|---|---|
| **Allocated memory** | 当前分配给 tensor 的显存 | **最重要**，看这个判断是否有泄漏 |
| **Active memory** | 当前在用的显存 | 通常等于 Allocated |
| **GPU reserved memory** | PyTorch 从 GPU 申请的总池子 | 包含 Allocated + 缓存空闲块 |
| **Cur Usage** | 当前用量 | 现场状态 |
| **Peak Usage** | 历史峰值 | 判断 OOM 风险，看这个 |
| **Tot Alloc / Tot Freed** | 累计分配/释放 | 看是否在频繁 malloc/free |
| **Allocations** | 当前活跃 tensor 数 | 数量异常多 → 可能有泄漏 |

### 3.3 典型场景判断

#### 场景 A：训练正常运行

```
Allocated: 50 GB  Peak: 55 GB  Reserved: 60 GB
→ 健康。Reserved 略大于 Peak 是正常缓存
```

#### 场景 B：显存逐步增长（泄漏）

```
Step 100:  Allocated 30 GB
Step 200:  Allocated 35 GB
Step 300:  Allocated 40 GB
→ 有泄漏！每个 step 没完全释放
→ 排查：检查是否有 tensor 没 .detach() 就保存到 list 里
```

#### 场景 C：显存碎片化

```
Allocated: 20 GB
Reserved:  60 GB
→ 碎片严重，PyTorch 拿了 60GB 但只能用 20GB
→ 解决：torch.cuda.empty_cache() 或重启训练
```

#### 场景 D：即将 OOM

```
Allocated: 75 GB / 80 GB（A100）
Peak:      78 GB
→ 危险，下一个 step 可能炸
→ 解决：减 batch、降 seq_len、开梯度累积
```

---

## 第四章：四把"省显存"利器

### 4.1 利器 1：混合精度（Mixed Precision）

**原理：用 bf16/fp16 存参数和激活，但 Adam 优化器状态保持 fp32。**

```
全 fp32：参数 28GB + 梯度 28GB + Adam 56GB = 112 GB
bf16 训练（默认推荐）：
  参数 14GB（bf16） + 梯度 14GB（bf16） + Adam 56GB（fp32）= 84 GB
  → 节省 ~25%
```

```python
training_args = SFTConfig(
    ...,
    bf16=True,              # ✅ B300/H100/A100 上首选
    # fp16=True,            # 老 GPU（V100）用这个
)
```

**bf16 vs fp16：**

| | bf16 | fp16 |
|---|---|---|
| 范围 | 同 fp32（指数 8 位）| 小（指数 5 位） |
| 精度 | 较低（尾数 7 位）| 较高（尾数 10 位）|
| 训练稳定性 | ✅ 几乎不需要 GradScaler | 需要 GradScaler 防上溢/下溢 |
| 硬件支持 | A100 / H100 / B300 / TPU | 几乎所有现代 GPU |
| **推荐** | **大模型训练默认** | 老硬件 fallback |

### 4.2 利器 2：梯度累积（Gradient Accumulation）

**原理：连续 N 个 micro-batch 累加梯度，再更新一次参数。**

```
等效 batch_size = per_device_batch_size × gradient_accumulation_steps × num_gpus

例：想要 batch=64 但显存只够 batch=4
  per_device_train_batch_size = 4
  gradient_accumulation_steps = 16
  num_gpus = 1
  → 等效 batch = 4 × 16 × 1 = 64
  → 显存占用：等效 batch=4（不变！）
  → 训练效果：等效 batch=64（变大！）
```

```python
training_args = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    ...,
)
```

**代价**：训练速度变慢（更新频率降低 N 倍），但效果上等价于大 batch。

### 4.3 利器 3：梯度检查点（Gradient Checkpointing）

**原理：反向传播时不保存所有激活，需要时重新计算。**

```
不开启：
  正向：保存所有 32 层的激活    → 激活显存 30 GB
  反向：直接读激活算梯度

开启：
  正向：每 4 层只保存一次       → 激活显存 8 GB（-75%）
  反向：缺失的激活实时重算      → 计算量 +30%
```

```python
training_args = SFTConfig(
    ...,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

**代价**：训练速度变慢 20-30%，但激活显存减少 70%+。

### 4.4 利器 4：LoRA（低秩适配）

**原理：冻结基座，只训练注入的低秩矩阵。**

```
全参数训练 7B：
  参数 14GB + 梯度 14GB + Adam 56GB = 84 GB（bf16+fp32 Adam）

LoRA 训练 7B（r=8）：
  基座参数 14GB（冻结，无梯度无优化器）
  LoRA 参数 ~30MB
  LoRA 梯度 ~30MB
  LoRA Adam ~120MB
  小计：14 GB
  → 节省 83%
```

详见 [第五章](#第五章lora-为什么这么省) 的展开。

### 4.5 利器组合：实战配方

```python
# 配方 A：单卡 80GB 训练 7B 模型（中等显存）
training_args = SFTConfig(
    bf16=True,                             # 混合精度
    gradient_checkpointing=True,           # 梯度检查点
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,         # 等效 batch=16
)
peft_config = LoraConfig(r=16, ...)        # LoRA

# 配方 B：单卡 24GB 训练 7B 模型（小显存）
training_args = SFTConfig(
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
)
peft_config = LoraConfig(r=8, ...)
# 加 4-bit QLoRA：load_in_4bit=True

# 配方 C：4 卡训练 70B 模型（大模型必选 ZeRO）
# 参考第六章 DeepSpeed
```

---

## 第五章：LoRA 为什么这么省？

### 5.1 LoRA 数学回顾

```
原始线性层：y = Wx           （W: [d_out, d_in]）

LoRA 改造：
  W_effective = W + (α/r) * B @ A
  其中：A: [r, d_in], B: [d_out, r]，r ≪ min(d_out, d_in)

训练时：
  - W 冻结（不算梯度，不存优化器状态）
  - 只更新 A 和 B
```

### 5.2 LoRA 的显存账本

以 Qwen2.5-7B 为例（target_modules=["q_proj", "v_proj"], r=8, α=16）：

```
全参数（基座）：7B × 2 字节（bf16） = 14 GB ──── 冻结
LoRA 参数计算：
  - 32 层 × 2 modules（q,v）× (4096×8 + 8×4096) = 32 × 2 × 65536 = 4.2M 参数
  - 4.2M × 4 字节（fp32）≈ 17 MB

显存对比：
┌─────────────────────────────────────────────────────────────────┐
│                  全参数训练       LoRA 训练                       │
├─────────────────────────────────────────────────────────────────┤
│  参数            14 GB           14 GB（基座 bf16）                │
│  梯度            14 GB           17 MB（只对 LoRA 部分）           │
│  Adam 优化器     56 GB           68 MB（17MB×4 即 m+v 各 fp32）   │
│  激活            ~20 GB          ~20 GB（前向激活一样多）          │
│  ───────────────────────                                        │
│  总计            ~104 GB         ~14 GB + 激活                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 LoRA 的精度代价

```
全参数训练：能改变模型每一个权重
LoRA：     ΔW 必须是低秩（rank=r）的形式

→ LoRA 表达能力受限
→ 但实证：r=8/16 已经能达到全参数训练 95%+ 效果
→ r 越大效果越好，显存代价也越大
```

**经验值：**

| 任务难度 | r | lora_alpha | 适用场景 |
|---|---|---|---|
| 简单（短指令、风格调整） | 4-8 | 8-16 | 客户微调、风格适配 |
| 中等（多任务 SFT） | 16-32 | 32-64 | 通用 SFT |
| 困难（大幅改变模型行为） | 64-128 | 128-256 | 特定领域大调整 |

### 5.4 target_modules 选择

```python
# 最少：只动 attention 的 Q、V
target_modules=["q_proj", "v_proj"]
# 显存最省，效果一般

# 标准：attention 全部
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
# 推荐起点

# 完整：attention + MLP
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
# 显存翻倍但效果最好（接近全参数训练）

# 自动发现：
import peft
target_modules = "all-linear"   # 所有 nn.Linear
```

### 5.5 QLoRA：4-bit 基座 + LoRA

```
LoRA 基础上再加一层：把基座量化到 4-bit
  - 基座参数：14GB → 3.5GB（int4）
  - LoRA 仍用 bf16/fp32 训练
  → 7B 模型可在 12GB 显存上训练

代价：基座精度有一点损失，但训练效果很接近 LoRA
```

```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",   # 优于 fp4
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)
# 后续 LoRA 流程一样
```

---

## 第六章：DeepSpeed ZeRO 三阶段

### 6.1 ZeRO 解决什么问题？

> **多卡训练大模型时，每张卡都存一份完整模型 → 显存浪费。**

```
传统数据并行（DDP）：
  GPU 0: [完整模���] + [数据 batch 1]
  GPU 1: [完整模型] + [数据 batch 2]
  ...
  → 每张卡都存完整模型、梯度、优化器
  → 4 卡训练 7B：每张卡都吃 100+ GB

ZeRO 思路：把"完整模型"拆开存
```

### 6.2 三个阶段对比

```
┌──────────────────────────────────────────────────────────────────┐
│  阶段              拆什么          显存（每张卡）  通信开销         │
├──────────────────────────────────────────────────────────────────┤
│  ZeRO-1           Adam 状态        基础 - 25%      低             │
│  ZeRO-2           Adam + 梯度      基础 - 50%      中             │
│  ZeRO-3           参数全拆         基础 - 75%      高             │
│  FSDP（PyTorch）   ≈ ZeRO-3        ≈ ZeRO-3        ≈ ZeRO-3      │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 配置示例（accelerate_config.yaml）

本项目用的是 ZeRO Stage 1：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 1                # 选 1/2/3
  zero3_init_flag: false
  offload_optimizer_device: none      # 也可以 cpu，进一步省卡显存
  offload_param_device: none
  gradient_accumulation_steps: 1
fp16: false
bf16: true
machine_rank: 0
num_machines: 1
num_processes: 3                # 几张卡
```

### 6.4 ZeRO 阶段选择策略

```
能用就用低阶段：
  ZeRO-1 通信开销最小，训练速度最快
  ZeRO-3 通信最贵，但能装下最大的模型

选择流程：
  显存够用？ → 不开 ZeRO（DDP）
  优化器状态太大？ → ZeRO-1
  + 梯度也大？ → ZeRO-2
  + 模型本身就装不下？ → ZeRO-3 或 FSDP

进一步压榨显存（速度更慢）：
  + offload_optimizer_device: cpu  → 优化器状态卸载到 CPU
  + offload_param_device: cpu      → 参数卸载到 CPU（极慢但能跑超大模型）
```

### 6.5 实战示例

```bash
# 单卡（无 ZeRO）
python qwen_pt.py

# 多卡 + ZeRO-1（本项目预训练在用）
accelerate launch --config_file accelerate_config.yaml qwen_pt.py

# 多卡 + ZeRO-3 + CPU offload（极致省显存）
# 改 accelerate_config.yaml：
#   zero_stage: 3
#   offload_optimizer_device: cpu
#   offload_param_device: cpu
```

---

## 第七章：qwen_mem.py 代码逐段拆解

### 7.1 三个监控关键点

```python
# 关键点 1：模型加载完，开始训练前
print("===============origin================")
print_trainable_parameters(model)        # 看 trainable% （LoRA 后应 < 1%）
print(torch.cuda.memory_summary())       # 看基线显存

# 关键点 2：IPython.embed 暂停
import IPython; IPython.embed()           # 进交互 shell 手动探索

# 关键点 3：每个 step 打印
class CUDAMemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print(f"[Step {state.global_step}] CUDA Memory Summary:")
        print(torch.cuda.memory_summary())
trainer.add_callback(CUDAMemoryCallback())
```

### 7.2 三个利器：clear_memory 的内部

```python
def clear_memory():
    # 1. 删除已知占显存的大对象
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'trainer' in globals(): del globals()['trainer']
    ...

    # 2. Python GC（让循环引用对象被回收）
    gc.collect()

    # 3. 清空 PyTorch 的缓存池
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

**何时调用 clear_memory？**

```
✓ 数据预处理完，训练前：释放数据 loading 残留
✓ 训练完，要���模型时：释放旧模型再加载新的
✓ 推理多轮后显存爬升：定期清理碎片
✗ 训练 step 之间：没必要，每个 step 自动清理
```

### 7.3 print_trainable_parameters 的输出解读

```python
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || "
          f"all params: {all_param} || "
          f"trainable%: {100 * trainable_params / all_param}")
```

**典型输出：**

```
基座模型（无 LoRA）：
  trainable params: 494032768 || all params: 494032768 || trainable%: 100.0

加上 LoRA(r=8, q_proj, v_proj)：
  trainable params: 540672 || all params: 494573440 || trainable%: 0.109
  ↑ 只有 0.1% 参数可训练，显存账单立刻降一个数量级
```

### 7.4 max_seq_length 的影响

```python
trainer = SFTTrainer(
    ...,
    max_seq_length=10,    # 教学用极端值！
)
```

**显存对比（实际跑 Qwen2.5-0.5B）：**

| max_seq_length | 单 batch 激活显存 | 推荐用途 |
|---|---|---|
| 10 | ~50 MB | 仅调试，看不出训练效果 |
| 256 | ~500 MB | 短指令任务 |
| 512 | ~1.5 GB | 通用 SFT 起点 |
| 1024 | ~5 GB | 中等长度任务 |
| 2048 | ~18 GB | 长对话 / 文档总结 |
| 4096 | ~70 GB | 超长上下文（必须开梯度检查点）|

> 显存随 seq_len 平方增长是 attention 的固有性质：O(n²)。

### 7.5 gradient_accumulation_steps 实战

代码里这一行被注释掉了：

```python
# gradient_accumulation_steps=16,
```

**取消注释 vs 不取消的区别：**

```
不开梯度累积（per_device_batch_size=1）：
  等效 batch = 1
  更新频率 = 1 step/update
  训练慢、loss 噪声大、收敛差

开梯度累积（per_device_batch_size=1, accum=16）：
  等效 batch = 16
  更新频率 = 16 steps/update
  显存几乎不变（!）、loss 平滑、收敛好

→ 显存吃紧时强烈建议开
```

---

## 第八章：OOM 排查方法论

### 8.1 OOM 的 5 步排查流程

```
报错：CUDA out of memory. Tried to allocate X GB...
         ↓
┌─────────────────────────────────────────────────────────┐
│  Step 1：看报错时机                                       │
│    模型加载就 OOM   → 模型本身放不下，必须 ZeRO 或量化     │
│    第一个 step OOM  → 激活+梯度爆了，减 batch/seq_len     │
│    训练几十步后 OOM → 显存泄漏，找无 detach 的 tensor      │
│    eval 时 OOM     → eval batch 太大，单独设小            │
│    某个 batch OOM  → 数据有超长样本，加 max_seq_length 截断│
│         ↓                                                │
│  Step 2：定位显存大户                                    │
│    torch.cuda.memory_summary() → 看 Peak Usage           │
│    nvidia-smi → 看哪些进程在占                           │
│         ↓                                                │
│  Step 3：按"代价递增"顺序应用招数                       │
│    a. bf16=True                              0 代价      │
│    b. gradient_checkpointing=True            速度 -25%   │
│    c. per_device_train_batch_size 减半        无副作用    │
│    d. gradient_accumulation_steps 翻倍        无副作用    │
│    e. max_seq_length 减半                    可能影响效果 │
│    f. 上 LoRA                                效果略降    │
│    g. 上 QLoRA / 8bit                        效果略降    │
│    h. ZeRO Stage 1/2/3                       多卡通信开销 │
│    i. CPU offload                            速度 -50%+  │
│         ↓                                                │
│  Step 4：每改一项就跑 1 个 step 验证                     │
│         ↓                                                │
│  Step 5：还不行就缩模型规模 / 升级硬件                   │
└─────────────────────────────────────────────────────────┘
```

### 8.2 常见 OOM 场景与对策

#### 场景 1：刚加载模型就 OOM

```
原因：模型 fp32 加载，自身就装不下
对策：
  model = AutoModelForCausalLM.from_pretrained(
      model_path,
      torch_dtype=torch.bfloat16,    # ← 加这行立即省一半
      device_map="auto",
  )
```

#### 场景 2：每步显存稳步上升（泄漏）

```python
# ❌ 错误：tensor 还带着计算图被存起来
losses = []
for step in range(100):
    loss = compute_loss(...)
    losses.append(loss)              # 整个计算图都被引用了！
    loss.backward()

# ✅ 正确：只存数值，不存计算图
losses = []
for step in range(100):
    loss = compute_loss(...)
    losses.append(loss.item())       # .item() 只取标量值
    loss.backward()
```

#### 场景 3：偶发 OOM（某些 batch 炸）

```
原因：数据集中混入了超长样本
排查：
  for i, sample in enumerate(dataset):
      length = len(tokenizer(sample["text"]))
      if length > 2000:
          print(f"Sample {i}: length {length}")

对策：
  - 数据预处理时丢弃超长样本
  - 或者用 truncation=True, max_length=2048 强制截断
```

#### 场景 4：eval 时 OOM（训练正常）

```
原因：eval 默认用更大的 batch_size，且开了 generate（KV cache 大）
对策：
  training_args = SFTConfig(
      per_device_eval_batch_size=1,             # eval batch 单独控制
      eval_accumulation_steps=4,                # eval 时也累积
  )
```

#### 场景 5：DeepSpeed ZeRO-3 里 OOM

```
通常是 zero3_init_flag 没配对：
  zero3_init_flag: true  ← ZeRO-3 必须开，否则 init 时占满显存
```

### 8.3 调试小工具

```python
# 工具 1：定位某段代码的显存增量
mem_before = torch.cuda.memory_allocated() / 1024**3
# ... 你想测的代码 ...
mem_after = torch.cuda.memory_allocated() / 1024**3
print(f"This block used: {mem_after - mem_before:.2f} GB")

# 工具 2：清空缓存看真实占用
torch.cuda.empty_cache()
print(f"Real usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 工具 3：找占显存最多的 tensor
import gc
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.is_cuda:
            print(obj.size(), obj.dtype, obj.device)
    except:
        pass
```

---

## 附录：常见问题

### Q1：bf16 和 fp16 哪个更好？

> **A100 / H100 / B300 用 bf16，老 GPU（V100/T4）用 fp16。**
>
> bf16 的范围和 fp32 一样大，几乎不用 GradScaler，训练稳定。
> fp16 精度高但范围小，loss 容易溢出，需要 GradScaler 动态缩放。

### Q2：开了 gradient_checkpointing 后训练慢了 30%，能省吗？

> 长序列训练（seq_len ≥ 2048）几乎必须开。
> 短序列时收益小代价大，可以不开。
> 简单判断：激活显存超过总显存 30% 就该开。

### Q3：LoRA r 和 α 怎么调？

> 经验法则：α = 2r（即 scaling = α/r = 2）
>
> r 选择：
> - 显存吃紧：r=4 或 8
> - 一般场景：r=16
> - 大幅度调整：r=32 或 64
>
> 调整时只改 r，α 保持 = 2r。

### Q4：训练时显示 Reserved 远大于 Allocated 怎么办？

> 这是显存碎片化，PyTorch 拿了大池子但只用了一部分。
>
> 解决：
> 1. 训练循环里偶尔调 `torch.cuda.empty_cache()`（损失少量速度）
> 2. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（PyTorch 2.0+）

### Q5：DeepSpeed 和 FSDP 选哪个？

| 方面 | DeepSpeed | FSDP（PyTorch 原生）|
|---|---|---|
| 配置复杂度 | 中（YAML/JSON）| 低（Python API） |
| 生态 | 大（Microsoft 在维护） | 中（PyTorch 官方）|
| HuggingFace 支持 | ✅ 早期支持 | ✅ 也很好 |
| 性能 | ZeRO-3 略快 | 接近 |
| 推荐 | 老项目、要 offload | 新项目、纯 PyTorch 栈 |

### Q6：训练已经 ZeRO-3 了还 OOM，怎么办？

> 终极手段：
> 1. CPU offload（offload_optimizer_device: cpu）
> 2. NVMe offload（offload_param_device: nvme）
> 3. 上 LoRA / QLoRA（量级降低）
> 4. 减 seq_length 或 batch
> 5. 加更多 GPU
> 6. 缩小模型（7B → 1.5B → 0.5B）
>
> 通常前 3 招就能解决 95% 的问题。

### Q7：QLoRA 会不会比 LoRA 差很多？

> 实测差距很小（< 2%）。
>
> 论文 [QLoRA] 在多个任务上证明：4-bit QLoRA ≈ 16-bit LoRA。
> 显存却能再省 60%+，对于个人开发者非常友好。

### Q8：什么情况下应该用全参数训练，不用 LoRA？

```
✓ 全参数：
  - 预训练（PT）
  - 续训练（Continue PT）
  - 大幅度修改模型行为（如 R1 风格 RL）
  - 数据量很大（> 100k 样本）

✓ LoRA：
  - SFT（小数据）
  - DPO / PPO 偏好对齐（防灾难性遗忘）
  - 多任务适配（一个基座 + 多 adapter）
  - 显存有限
```

### Q9：怎么估算"我能训得动多大的模型"？

```
速算公式：
  能训练的最大参数量（B）≈ GPU 显存（GB）/ 16    （全参数 + bf16 + Adam）
                          ≈ GPU 显存（GB）/ 4     （LoRA + bf16）

例：
  24GB 显存（RTX 4090）：
    全参数：1.5B
    LoRA：6B（实测能跑 7B QLoRA）

  80GB 显存（A100/H100）：
    全参数：5B
    LoRA：20B（实测能跑 70B QLoRA）

  4 卡 80GB + ZeRO-3：
    全参数：~25B（理论），实际 ~13B 跑得舒服
```

### Q10：本项目的 mini_qwen 选 0.5B 是因为这是最优大小吗？

> **不是。** 选 0.5B 是因为：
> - 作为教学项目，跑得快、显存占用小
> - 单卡能完整跑通 PT/SFT/DPO/PPO/GRPO 全流程
> - 让初学者把时间花在"理解原理"而不是"等训练"
>
> 实际生产中，最低应该用 7B（千问、Llama），更好的是 32B/72B。
> 学完整套流程后，把脚本改成更大模型 + DeepSpeed ZeRO，就是工业级训练。

---

## 🎉 学习清单：本篇核心知识

- [x] 训练显存的 5 大去向：参数 + 梯度 + 优化器 + 激活 + 临时
- [x] Adam 是 2x 参数量的优化器（m + v）
- [x] 速算公式：参数 × 字节数 × 4 + 激活
- [x] memory_summary 关键字段：Allocated / Peak / Reserved
- [x] 四把利器：bf16、梯度累积、梯度检查点、LoRA
- [x] LoRA 节省显存的本质：基座冻结 → 没有梯度和优化器
- [x] ZeRO 三阶段：1（优化器）/ 2（+ 梯度）/ 3（+ 参数）
- [x] OOM 排查 5 步流程
- [x] 常见 OOM 场景对策（加载就炸 / 泄漏 / 偶发 / eval / ZeRO-3）

---

## 🚀 学完之后

恭喜你完整通关 mini_qwen 项目！整个学习路径回顾：

```
预训练 → 网络结构 → 续训练 → SFT → 评估 → DPO数据 → DPO →
RM → PPO → GRPO → 蒸馏 → Agentic RAG → 评判+部署 → 显存优化
```

**你现在掌握的能力（按训练流程梳理）：**

| 能力 | 对应技术栈 |
|---|---|
| 数据准备 | parquet / datasets / 数据合成 / 蒸馏数据生成 |
| 预训练 | accelerate / DeepSpeed ZeRO / 多卡训练 |
| 微调 | SFT / LoRA / QLoRA / DataCollatorForCompletionOnlyLM |
| 偏好对齐 | DPO / RM / PPO / GRPO（DeepSeek-R1 路线）|
| 评测 | Benchmark（MMLU 等）+ LLM-as-a-Judge 胜率 |
| 部署 | vLLM / LoRA 合并 / OpenAI 兼容 API |
| 应用 | Agentic RAG / 工具调用 / FAISS 检索 |
| 工程 | 显存优化 / OOM 排查 / 混合精度 / 梯度累积 |

**真正能让你成为 LLM 工程师的下一步：**

1. **改造 mini_qwen 为生产级**
   - 模型 0.5B → 7B/32B
   - 单机 → 多机分布式
   - 全流程脚本化 + CI/CD
   - 加监控告警

2. **跟进前沿**
   - 新算法：DAPO、SimPO、KTO、SPIN
   - 新架构：MoE、Mamba、Linear Attention
   - 长上下文：YaRN、RoPE 扩展、Ring Attention

3. **深入特定领域**
   - 多模态：Qwen-VL / LLaVA / Whisper
   - 智能体：函数调用、ReAct、多 Agent 协作
   - 推理优化：speculative decoding、PagedAttention 内核

4. **真实业务落地**
   - 客户场景的领域适配
   - RAG 系统工程化
   - Prompt Engineering 与 LLM Ops

学习从来不是终点。**祝你在大模型的世界里走得更远。** 🎓
