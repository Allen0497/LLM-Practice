# 🎯 大模型监督微调（SFT）完全指南 —— 从零理解到动手实践

> 本文档配合 `qwen_sft.py` 代码，手把手带你理解大模型监督微调的每一个环节。
> 假设你已经看过预训练完全指南，我们在此基础上继续深入。

---

## 📋 目录

- [第零章：SFT 在大模型训练中的位置](#第零章sft-在大模型训练中的位置)
- [第一章：什么是 SFT？为什么需要它？](#第一章什么是-sft为什么需要它)
- [第二章：SFT 数据格式 —— ChatML 对话模板](#第二章sft-数据格式--chatml-对话模板)
- [第三章：只对回答计算 Loss —— DataCollatorForCompletionOnlyLM](#第三章只对回答计算-loss--datacollatorforcompletiononlylm)
- [第四章：LoRA —— 高效微调的秘密武器](#第四章lora--高效微调的秘密武器)
- [第五章：SFTTrainer 与训练参数详解](#第五章sfttrainer-与训练参数详解)
- [第六章：代码逐行对照解读](#第六章代码逐行对照解读)
- [第七章：全量微调 vs LoRA 微调的选择](#第七章全量微调-vs-lora-微调的选择)
- [附录：常见问题](#附录常见问题)

---

## 第零章：SFT 在大模型训练中的位置

```
┌─────────────────────────────────────────────────────────────────────┐
│                     大模型训练完整流程                                │
│                                                                     │
│   ① 预训练（Pre-Training）                                          │
│   │  让模型阅读海量文本，学会语言的基本规律                           │
│   │  输出：会"续写"但不会"对话"的基座模型                            │
│   │                                                                 │
│   ② 指令微调（SFT）              ← 📍 我们现在在这里！              │
│   │  用"问题-回答"数据教模型如何按指令回答                           │
│   │  输出：会对话、会听指令的模型                                    │
│   │                                                                 │
│   ③ 偏好对齐（RLHF / DPO / GRPO）                                  │
│   │  让模型回答更符合人类偏好（更有帮助、更安全）                     │
│   │                                                                 │
│   ④ 评估与部署                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**打个比方：**

| 阶段 | 类比 | 做什么 |
|------|------|--------|
| 预训练 | 上学读书12年 | 博览群书，积累知识 |
| **SFT** | **岗前培训** | **学会按要求完成具体任务** |
| 偏好对齐 | 职业素养训练 | 学会什么该说什么不该说 |

---

## 第一章：什么是 SFT？为什么需要它？

### 1.1 预训练模型的局限

预训练完成后，模型学会了"续写文本"，但它并不知道如何"对话"：

```
# 预训练模型的行为（只会续写）
输入：  "请介绍一下北京"
输出：  "请介绍一下北京的历史文化，请介绍一下北京的美食，请介绍一下..."
        ↑ 它以为这是一道题目，继续往下写题目！
```

```
# SFT 之后的模型行为（学会了回答）
输入：  "请介绍一下北京"
输出：  "北京是中国的首都，有着3000多年的建城史。著名景点包括故宫、长城..."
        ↑ 它知道这是一个问题，给出了合适的回答！
```

### 1.2 SFT 的核心思想

SFT 用**有标注的"问题-回答"对**来训练模型，让模型学会：
1. 识别用户意图
2. 按照指令格式回答
3. 给出有帮助的内容

训练数据格式：
```json
{
  "conversations": [
    {"from": "human", "value": "请帮我写一首关于春天的诗"},
    {"from": "gpt",   "value": "春风轻抚柳丝长，百花争艳满园香..."}
  ]
}
```

### 1.3 SFT 与预训练的 Loss 计算区别

```
预训练：对所有 token 计算 loss（模型要学会预测每一个字）

  今 天 天 气 很 好
  ↑ ↑ ↑ ↑ ↑ ↑
  全部都计算 loss

SFT：只对 assistant 的回答部分计算 loss（不需要学会"提问"）

  <user>请介绍北京</user> <assistant>北京是首都...</assistant>
  ←────────────────────→ ←──────────────────────────────→
       label = -100（忽略）        计算 loss ✓
```

这是 SFT 最关键的设计之一，后面第三章会详细讲解。

---

## 第二章：SFT 数据格式 —— ChatML 对话模板

### 2.1 为什么需要统一格式？

模型需要知道"谁在说话"。如果直接把问题和回答拼在一起，模型无法区分哪部分是用户说的，哪部分是自己应该说的。

### 2.2 ChatML 格式详解

本项目使用 **ChatML（Chat Markup Language）** 格式：

```
<|im_start|>user
你好，请介绍一下你自己<|im_end|>
<|im_start|>assistant
我是一个AI语言助手，我可以帮你回答问题、写作、分析数据...<|im_end|>
```

特殊 token 说明：

| Token | 含义 | 作用 |
|-------|------|------|
| `<\|im_start\|>` | 对话轮次开始 | 标记一段话的开始 |
| `<\|im_end\|>` | 对话轮次结束 | 标记一段话的结束 |
| `user` | 角色标识 | 表示这是用户说的话 |
| `assistant` | 角色标识 | 表示这是模型说的话 |

### 2.3 formatting_prompts_func 函数解析

```python
# utils/utils.py 中的函数
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["conversations"])):
        for item in example["conversations"][i]:
            if item["from"] == "human":
                human_text = item["value"]
            elif item["from"] == "gpt":
                gpt_text = item["value"]
        # 拼装成 ChatML 格式
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts
```

转换示例：

```
原始数据：
  {"from": "human", "value": "1+1等于几？"}
  {"from": "gpt",   "value": "1+1等于2。"}

转换后：
  "<|im_start|>user\n1+1等于几？<|im_end|>\n<|im_start|>assistant\n1+1等于2。<|im_end|>"
```

---

## 第三章：只对回答计算 Loss —— DataCollatorForCompletionOnlyLM

### 3.1 为什么要屏蔽用户问题的 Loss？

想象一下，如果对用户问题也计算 loss：
- 模型会同时学习"如何提问"和"如何回答"
- 这会让模型产生混乱，有时候它会把回答写成问题的形式
- 浪费计算资源（我们不需要模型学会提问）

### 3.2 实现原理：label = -100

PyTorch 的交叉熵损失函数有一个约定：**label 为 -100 的位置会被忽略，不计入 loss**。

```python
# DataCollatorForCompletionOnlyLM 的工作原理（伪代码）
labels = input_ids.copy()

# 找到 response_template 的位置
template_pos = find_position("<|im_start|>assistant\n")

# 将 template 之前的所有 label 设为 -100
labels[:template_pos] = -100

# 结果：
# input_ids: [user_tokens..., template_tokens, assistant_tokens...]
# labels:    [-100, -100, ..., -100, assistant_tokens...]
#             ↑ 忽略                  ↑ 只计算这部分的 loss
```

### 3.3 代码实现

```python
response_template = "<|im_start|>assistant\n"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(
    response_template_ids,
    tokenizer=tokenizer,
    mlm=False
)
```

`add_special_tokens=False` 很重要：如果加了特殊 token，编码出来的 ID 序列会多出额外 token，导致模板匹配失败。

---

## 第四章：LoRA —— 高效微调的秘密武器

### 4.1 全量微调的问题

全量微调（Full Fine-Tuning）需要更新模型的**所有参数**：
- Qwen2.5-0.5B 有约 5 亿参数
- 每个参数用 BF16 存储需要 2 字节
- 仅模型权重就需要 ~1GB 显存
- 加上梯度、优化器状态，实际需要 **3-4 倍**显存 = ~4GB

对于更大的模型（7B、70B），全量微调根本无法在单卡上完成。

### 4.2 LoRA 的核心思想

LoRA（Low-Rank Adaptation）的关键洞察：

> **微调时，权重的变化量 ΔW 是低秩的（low-rank）**
> 也就是说，ΔW 可以用两个小矩阵的乘积来近似

```
原始权重矩阵 W（d×d，不动）
         +
低秩矩阵 B×A（d×r + r×d，只训练这部分）
         ↓
W' = W + (α/r) × B × A
```

其中 r（rank）远小于 d，例如 d=4096，r=8，参数量对比：

```
原始 W：4096 × 4096 = 16,777,216 个参数
LoRA：  4096 × 8 + 8 × 4096 = 65,536 个参数
节省：  99.6% 的参数！
```

### 4.3 LoRA 参数详解

```python
lora_config = LoraConfig(
    r=8,                              # 秩（rank）：越大表达能力越强，参数越多
    lora_alpha=16,                    # 缩放系数：实际缩放比例 = alpha/r = 2
    target_modules=["q_proj", "v_proj"],  # 只对注意力的 Q、V 矩阵加 LoRA
    lora_dropout=0.01,                # Dropout 防止过拟合
    bias="none",                      # 不训练 bias
    task_type="CAUSAL_LM",            # 因果语言模型
)
```

**为什么选 q_proj 和 v_proj？**

Transformer 注意力机制中有四个投影矩阵：Q、K、V、O。
研究表明，对 Q 和 V 加 LoRA 效果最好，是性价比最高的选择。

```
注意力计算：
  Q = X × W_q   ← LoRA ✓
  K = X × W_k
  V = X × W_v   ← LoRA ✓
  O = Attention(Q,K,V) × W_o
```

### 4.4 LoRA 的显存对比

| 方式 | 可训练参数 | 显存占用 |
|------|-----------|---------|
| 全量微调 | 100% | ~4× 模型大小 |
| LoRA (r=8) | ~0.4% | ~1.1× 模型大小 |

### 4.5 本脚本中 LoRA 被注释掉了

```python
# peft_config=lora_config,  # 取消注释即启用 LoRA
```

当前使用**全量微调**。这说明作者在这个阶段有足够的显存，或者认为全量微调效果更好。

---

## 第五章：SFTTrainer 与训练参数详解

### 5.1 SFTTrainer vs 普通 Trainer

`SFTTrainer` 来自 `trl`（Transformer Reinforcement Learning）库，是专为 SFT 设计的训练器，在普通 `Trainer` 基础上增加了：

- 自动处理 `formatting_func`（数据格式转换）
- 支持 `packing`（序列打包）
- 原生支持 `peft_config`（LoRA 等）
- 内置 `max_seq_length` 截断

### 5.2 关键参数解析

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,       # SFT 有验证集，可以监控过拟合

    formatting_func=formatting_prompts_func,  # 数据格式转换函数
    data_collator=collator,           # 只对 assistant 计算 loss

    max_seq_length=100,   # ⚠️ 这里设为 100 是调试值，正式训练应设 1024~4096
    packing=False,        # 不打包序列
    dataset_num_proc=16,  # 数据处理并行数
    dataset_batch_size=5000,
)
```

**packing 是什么？**

```
packing=False（默认）：每个样本单独一条，短样本会有大量 padding
  [样本1: 50 tokens] [padding: 50 tokens]
  [样本2: 30 tokens] [padding: 70 tokens]

packing=True：将多个短样本拼接成一个长序列，提高 GPU 利用率
  [样本1: 50 tokens][样本2: 30 tokens][样本3: 20 tokens]
```

### 5.3 SFTConfig 训练参数

```python
training_args = SFTConfig(
    learning_rate=1e-5,              # SFT 典型学习率（比预训练大，比继续预训练大得多）
    per_device_train_batch_size=1,   # SFT 序列长，batch_size 通常为 1
    gradient_accumulation_steps=16,  # 等效 batch_size = 16
    eval_steps=2000,                 # 每 2000 步评估一次验证集 loss
    ...
)
```

**三个阶段的学习率对比：**

```
预训练：         1e-4  （从零学习，需要大步伐）
继续预训练：     1e-8  （防止遗忘，极小步伐）
SFT：           1e-5  （在已有知识上学新技能，中等步伐）
```

---

## 第六章：代码逐行对照解读

### 完整数据流

```
原始数据（parquet 文件）
    ↓ load_dataset
conversations 字段：[{"from":"human",...}, {"from":"gpt",...}]
    ↓ formatting_prompts_func（在 SFTTrainer 内部调用）
ChatML 格式字符串："<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"
    ↓ tokenizer
token IDs：[151644, 872, 198, ..., 151645, 198, 151644, 77091, 198, ...]
    ↓ DataCollatorForCompletionOnlyLM
input_ids：[151644, 872, 198, ..., 151645, 198, 151644, 77091, 198, ...]
labels：   [-100,  -100, -100, ..., -100,  -100, -100,  -100,  -100, assistant_token_ids...]
    ↓ 模型前向传播 + 交叉熵 loss（只计算 assistant 部分）
loss → 反向传播 → 更新参数
```

### 训练集划分

```python
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.2).values()
```

```
全量数据集（100%）
├── 训练集（80%）：用于更新模型参数
└── 验证集（20%）：用于监控过拟合，不参与训练
```

---

## 第七章：全量微调 vs LoRA 微调的选择

### 如何选择？

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 显存充足（>40GB）| 全量微调 | 效果更好，收敛更稳定 |
| 显存有限（<24GB）| LoRA | 大幅减少显存占用 |
| 快速实验验证 | LoRA | 训练速度更快 |
| 生产环境最终模型 | 全量微调 | 效果通常更优 |
| 多任务适配 | LoRA | 每个任务一套 LoRA 权重，共享基座 |

### 启用 LoRA 只需一行

```python
# 取消注释这一行即可从全量微调切换到 LoRA 微调
trainer = SFTTrainer(
    ...
    peft_config=lora_config,  # ← 取消注释
    ...
)
```

启用后，`print_trainable_parameters(model)` 会显示：
```
trainable params: 2,097,152 || all params: 494,032,896 || trainable%: 0.42
# 只有 0.42% 的参数参与训练！
```

---

## 附录：常见问题

**Q：max_seq_length=100 是不是太小了？**

A：是的，这是调试用的值。正式训练应根据数据集的平均长度设置，通常为 512~4096。设太小会截断很多样本，导致模型学不到完整的对话。

**Q：为什么 per_device_train_batch_size=1？**

A：SFT 的序列通常比预训练长（包含完整对话），显存占用更大。batch_size=1 配合 gradient_accumulation_steps=16 等效于 batch_size=16，在不超显存的前提下保证训练稳定性。

**Q：SFT 之后模型会忘记预训练学到的知识吗？**

A：会有一定程度的遗忘，但 SFT 数据量远小于预训练，影响有限。如果担心遗忘，可以在 SFT 数据中混入少量预训练数据。

**Q：验证集 loss 下降但训练集 loss 不下降，正常吗？**

A：不正常，通常说明学习率太小或数据有问题。反过来，训练 loss 下降但验证 loss 上升，说明过拟合，需要减少 epoch 或增加 dropout。

**Q：LoRA 训练完后，保存的是 0.5B 还是 0.51B？怎么合并权重？**

这是一个非常好的问题。答案是：**取决于你处于哪个阶段**，LoRA 模型有三种存在形态。

### 形态一：训练中 —— 0.5B（冻结）+ 2M（LoRA）= 同时存在但分开

训练时，GPU 显存里同时有两部分：

```
┌─────────────────────────────────────────────────────────┐
│  GPU 显存中的模型                                         │
│                                                          │
│  原始模型参数 W（0.5B = 494M 参数）                       │
│  ├── q_proj: W_q [896×896]  ← 冻结，不更新 ❄️            │
│  ├── k_proj: W_k [896×128]  ← 冻结 ❄️                   │
│  ├── v_proj: W_v [896×128]  ← 冻结 ❄️                   │
│  ├── FFN 各层               ← 冻结 ❄️                   │
│  └── ...                                                 │
│                                                          │
│  LoRA 额外参数（约 2M 参数）                               │
│  ├── q_proj 的 LoRA:  A [896×8] + B [8×896]  ← 训练 🔥  │
│  └── v_proj 的 LoRA:  A [896×8] + B [8×896]  ← 训练 🔥  │
│                                                          │
│  前向传播时：output = x × W_q + x × B_q × A_q × (α/r)   │
│             原始部分不变 ↑       ↑ LoRA 部分叠加上去       │
└─────────────────────────────────────────────────────────┘
```

此时模型总参数 = 494M + 2M = 496M ≈ 0.496B，但只有 2M 在更新。

### 形态二：保存时 —— 只保存 2M 的 LoRA 权重（超级小！）

`trainer.save_model()` 在 LoRA 模式下，**只保存 LoRA 的 A、B 矩阵**，不保存原始的 0.5B 参数：

```
保存目录结构：
results/sft/
├── adapter_config.json      ← LoRA 配置（r=8, alpha=16, target_modules 等）
├── adapter_model.safetensors ← 只有 LoRA 权重，约 8MB！
└── tokenizer 相关文件

对比全量微调的保存：
results/sft/
├── model.safetensors         ← 完整模型权重，约 1GB
├── config.json
└── tokenizer 相关文件
```

**为什么只存 LoRA 部分？**

因为原始的 0.5B 参数一个都没动过（冻结的），存了也是浪费。用的时候再把原始模型加载回来，把 LoRA 叠上去就行。

这也是 LoRA 的一大优势：**一个基座模型 + 多个小小的 LoRA 文件 = 多个不同任务的模型**。

```
基座模型（0.5B，1GB）  ← 只存一份
  ├── + LoRA_对话（8MB）  → 对话模型
  ├── + LoRA_翻译（8MB）  → 翻译模型
  ├── + LoRA_代码（8MB）  → 代码模型
  └── + LoRA_医疗（8MB）  → 医疗模型

总共只需要 1GB + 32MB，而不是 4 × 1GB = 4GB
```

### 形态三：合并后 —— 回到 0.5B（但参数值变了）

部署上线时，我们通常把 LoRA 权重**合并回**原始模型，得到一个普通的 0.5B 模型：

```
合并公式：
  W_new = W_original + (α/r) × B × A

  W_original: [896×896] 原始权重（0.5B 的一部分）
  B × A:      [896×8] × [8×896] = [896×896] 跟原始权重同样大小
  W_new:      [896×896] 合并后的新权重（大小不变！）
```

**关键理解：合并是矩阵加法，不是拼接。结果矩阵的大小跟原始一模一样。**

```
合并前：                              合并后：
┌──────────────┐  ┌──────┐           ┌──────────────┐
│ W_q [896×896]│ +│ B×A  │    →      │W_q' [896×896]│
│ 原始参数      │  │[896×896]         │ 新参数        │
│ （没变过）    │  │（学到的）│         │（融合了新知识）│
└──────────────┘  └──────┘           └──────────────┘
  大小不变！        ���开后同样大小        大小还是不变！
```

所以合并后的模型**还是 0.5B（494M 参数）**，跟原始模型大小完全一样，但参数的值变了（融入了 LoRA 学到的知识）。

### 合并代码（`utils/merge_peft_adapter.py`）

```python
# 第1步：加载原始基座模型（0.5B 参数）
model = AutoModelForCausalLM.from_pretrained(BaseModelPath)

# 第2步：把 LoRA 权重加载到模型上（此时模型 = 0.5B + 2M）
model = PeftModel.from_pretrained(model, AdapterModelPath)

# 第3步：合并！W_new = W + (α/r) × B × A
# merge_and_unload() 做两件事：
#   merge: 把 LoRA 的 B×A 加到原始 W 上
#   unload: 删除 LoRA 的 A、B 矩阵，模型恢复为普通结构
model = model.merge_and_unload()

# 第4步：保存合并后的模型（跟普通 0.5B 模型一模一样的格式）
model.save_pretrained(OutputPath)
```

### 三种形态总结

```
                    参数量          文件大小      用途
                    ──────          ────────      ────
训练中              494M + 2M       -             训练
                    (冻结) (更新)

保存 LoRA           只存 2M         ~8 MB         存储、多任务切换
                                                  需要配合基座模型使用

合并后              494M            ~1 GB         部署推理
                    (值已改变)                     独立使用，不需要基座
```

```
时间线：

  基座模型 0.5B ──→ 加载 ──→ 冻结 0.5B + 训练 LoRA 2M
                              │
                    ┌─────────┴──────────┐
                    ↓                    ↓
              保存 LoRA（8MB）      合并后保存（1GB）
              需要基座才能用         独立可用
              适合多任务切换         适合部署上线
```

**一句话总结：LoRA 训练时多了 2M 参数，保存时只存这 2M（8MB），合并后回到 0.5B（1GB）但参数值已经变了。模型大小始终是 0.5B，不会变成 0.51B。**
