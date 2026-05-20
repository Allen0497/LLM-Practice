"""
显存监控与 LoRA SFT 实战脚本
======================================================

本脚本是一份"显存调优教学版"——把 SFT 训练流程跑起来，但每一步都把
显存用量打印出来，让你看清楚显存到底花在哪里、LoRA 怎么省、batch_size
和 seq_length 如何影响显存。

与第 4 篇 SFT 完全指南的区别：
  qwen_sft.py（第 4 篇）：完整 SFT 训练，全参数微调
  qwen_mem.py（本篇）：    SFT 训练 + 显存监控 + LoRA + IPython 交互
                          专门用来"摸清显存底细"

学习目标：
  1. 看明白 torch.cuda.memory_summary() 的输出
  2. 理解全参数 vs LoRA 的显存差异
  3. 知道每个超参（batch_size / seq_length / grad_accum）怎么影响显存
  4. 学会用 IPython.embed 在训练前 / 中暂停查看状态
  5. 掌握 OOM（Out Of Memory）的排查思路

整体流程：
  ┌────────────────────────────────────────────────────────┐
  │  ① 加载 SFT 后的 checkpoint 作为新基座                    │
  │  ② 打印基座模型的可训练��数量 + 显存快照（基线）          │
  │  ③ IPython.embed 暂停（让你交互查看状态）                 │
  │  ④ 加载并预处理数据（小子集）                             │
  │  ⑤ 配置 LoRA + Trainer                                   │
  │  ⑥ 训练前 clear_memory（释放无用对象）                   │
  │  ⑦ 注册 CUDAMemoryCallback：每个 step 打印显存           │
  │  ⑧ 开始训练，观察显存变化                                 │
  └────────────────────────────────────────────────────────┘

关键模块说明：
  print_trainable_parameters：打印 trainable / total 参数量比例
                               LoRA 后通常只有 0.1%~1% 参数可训练
  torch.cuda.memory_summary()：PyTorch 的标准显存报告
                                包含 allocated / reserved / 历史峰值等
  CUDAMemoryCallback：自定义 Trainer 回调，每 step 打印一次显存
  clear_memory：手动 GC + torch.cuda.empty_cache，释放可回收显存
"""

import os
import torch
import wandb
from peft import LoraConfig
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.utils import find_files, clear_memory, print_trainable_parameters, formatting_prompts_func

# ============================================================
# 第一部分：实验配置（参数都集中在顶部，方便对比实验）
# ============================================================
WANDB_LOG = True
TMP_PATH = "/archive/share/cql/aaa/tmp"
TRAIN_SUBSET = 1000   # 训练子集大小，调试时小，正式训练设 -1（全量）
EVAL_SUBSET = 100     # 验证子集大小

# 输出目录：保存 LoRA checkpoint 的位置
output_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/sft-1"
# 输入模型：用前一轮 SFT 训练好的 checkpoint 当基座（继续训练 / 二次微调）
model_path = "/archive/share/cql/LLM-FoR-ALL/mini_qwen/results/sft/checkpoint-6000"

# ============================================================
# 第二部分：加载基座模型（基线显存测量）
# ============================================================
# 注意没指定 torch_dtype，默认是 fp32（每个参数 4 字节）
# 这是为了演示"全精度加载有多大"
# 真实训练时应该指定 torch_dtype=torch.bfloat16 节省一半显存
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("===============origin================")
# 打印参数量：此时还没套 LoRA，所有参数都可训练
# 应该看到：trainable% 接近 100%
print_trainable_parameters(model)
# 打印 GPU 显存快照：完整模型加载后的状态
# 重点看："Allocated memory" 这一行，对 0.5B 模型 fp32 加载约 2GB
print(torch.cuda.memory_summary())

# ★ IPython.embed 在这里暂停脚本，进入交互式 shell
# 你可以手动执行：
#   torch.cuda.memory_allocated() / 1024**3  → 看当前 GPU 用了多少 GB
#   model                                     → 看模型结构
#   exit()                                    → 退出 shell，继续脚本
# 这是显存调试的杀手锏：先停下来摸清状态，再继续
import IPython; IPython.embed()

# ============================================================
# 第三部分：加载 SFT 数据集
# ============================================================
# 使用 Infinity-Instruct 中的 Gen 和 7M 子集（与 qwen_sft.py 一致）
directories = ["Gen", "7M"]
data_files = find_files(directories, "/archive/share/cql/LLM-FoR-ALL/mini_qwen/data/sft")
dataset = load_dataset("parquet", data_files=data_files, split="train",
                        columns=["conversations"], cache_dir=TMP_PATH)
dataset = dataset.shuffle(seed=42)
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.2).values()

# 截取小子集（教学/调试用）
if TRAIN_SUBSET > 0:
    train_dataset = train_dataset.select(range(TRAIN_SUBSET))
if EVAL_SUBSET > 0:
    valid_dataset = valid_dataset.select(range(EVAL_SUBSET))

# ============================================================
# 第四部分：数据整理器（只对 assistant 部分计算 loss）
# ============================================================
# response_template 是 ChatML 中的 assistant 起始标签
# DataCollatorForCompletionOnlyLM 会把这个模板之前的 token 标记为 -100
# loss 只在 assistant 回答部分计算（不学习 user 提问）
response_template = "<|im_start|>assistant\n"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)

# ============================================================
# 第五部分：LoRA 配置（关键：显存大幅降低）
# ============================================================
# LoRA 数学回顾：y = Wx + (α/r) * BAx
#   r=8：低秩矩阵的秩
#   lora_alpha=16：缩放系数（实际缩放 = α/r = 2）
#   target_modules=["q_proj", "v_proj"]：只在 attention 的 Q、V 上加 LoRA
#                                        想效果更好可以加 k_proj, o_proj, gate_proj 等
#   lora_dropout=0.01：LoRA 内部 dropout（防过拟合，可选）
#
# 显存对比（Qwen2.5-0.5B 为例）：
#   全参数训练：模型 1GB + 梯度 1GB + 优化器状态 4GB（Adam 2x params） = 6GB
#   LoRA 训练：  模型 1GB + 梯度 ~10MB + 优化器 ~20MB              = ~1GB
#   → 显存减少 80%+
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.01,
    bias="none",          # 不训练 bias（节省显存）
    task_type="CAUSAL_LM",
)

# ============================================================
# 第六部分：训练参数（节选了影响显存的几个关键参数）
# ============================================================
training_args = SFTConfig(
    output_dir=output_path,
    overwrite_output_dir=True,
    eval_steps=2000,
    learning_rate=1e-5,        # LoRA 通常用比全参数训练大 10-100 倍的 lr
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3,

    # ★ 影响显存的关键参数：
    per_device_train_batch_size=1,    # 每张卡每个 step 的样本数
                                        # 显存吃紧时设 1，宽裕时��到 4/8/16
    # gradient_accumulation_steps=16, # （注释掉了）梯度累积：累积 N 个 micro-batch
                                        # 等效 batch = per_device * accum * num_gpus
                                        # 显存不变但等效 batch 变大
    save_steps=1000,
    save_total_limit=3,
    # bf16=True,                       # （注释掉了）混合精度
                                        # 开启后显存减半、速度更快、精度几乎无损
    logging_steps=10,
    report_to="wandb",
)

if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-sft",
        name="qwen-0.5B-sft",
    )

# ============================================================
# 第七部分：初始化 SFTTrainer
# ============================================================
# ★ max_seq_length=10 是教学用极端值！
# 真实训练时通常用 512 / 1024 / 2048
# 这里设成 10 是为了：
#   1. 让 attention 计算极快（CUDA Memory Summary 立刻能看到效果）
#   2. 显著降低 activation 显存（O(seq_len²) 复杂度的部分变小）
#   3. 调试用，看不出真实效果
#
# 显存占用与 seq_length 关系：
#   每层 attention activation：O(batch * heads * seq_len²)
#   seq_len 翻倍 → activation 显存 4 倍
#   这就是为什么长序列训练特别吃显存
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=lora_config,                 # ★ 传入 LoRA 配置，Trainer 自动包装
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=10,                       # ⚠ 教学用极端值，真实训练改成 512+
    packing=False,                           # 不打包多个样本（packing=True 提速但需要 attention mask 支持）
    dataset_num_proc=16,                     # 数据预处理并行进程数
    dataset_batch_size=1,
)

# 训练开始前清理一次：释放数据加载和预处理时残留的中间变量
clear_memory()

# ============================================================
# 第八部分：自定义 Callback——每个 step 打印显存
# ============================================================
# TrainerCallback 是 transformers 提供的 hook 机制
# on_step_end：每个训练 step 结束后被调用
# 这里每个 step 都打印（频率高、日志爆炸），实战时建议改为 % 100 == 0
from transformers import TrainerCallback

class CUDAMemoryCallback(TrainerCallback):
    """
    显存监控回调：每个 step 结束后打印 CUDA Memory Summary
    教学用，能直观看到训练过程中显存变化（前向 / 反向 / 优化器更新各阶段）
    """
    def on_step_end(self, args, state, control, **kwargs):
        # state.global_step：当前训练步数（跨 epoch 累计）
        if state.global_step % 1 == 0:   # 每 1 步打印一次（建议改成 100）
            print(f"[Step {state.global_step}] CUDA Memory Summary:")
            print(torch.cuda.memory_summary())

trainer.add_callback(CUDAMemoryCallback())

# ============================================================
# 第九部分：开始训练
# ============================================================
print("Training...")
trainer.train()                          # 主循环：前向 → 反向 → 优化器更新 → 重复
trainer.save_model()                     # 保存 LoRA adapter（不是完整模型！）
tokenizer.save_pretrained(output_path)

# 训练完成后，下一步通常是：
#   1. 用 utils/merge_peft_adapter.py 把 LoRA 合并回基座（见第 13 篇）
#   2. 用 qwen_chat.py 验证模型推理（见第 13 篇）
#   3. 用 qwen_judge.py 评估胜率（见第 13 篇）
