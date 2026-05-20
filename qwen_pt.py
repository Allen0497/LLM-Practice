"""
=====================================================================================
Qwen2.5-0.5B 预训练脚本 (Pre-Training Script)
=====================================================================================

【什么是预训练？】
    预训练是大模型训练的第一步，也是最基础的一步。
    简单来说，就是让模型阅读海量的文本数据，学会"语言"本身的规律。
    就像一个婴儿通过大量听和看来学习语言一样，模型通过预训练来学习：
    - 词语之间的关系（比如"北京"后面大概率跟"是中国的首都"）
    - 语法结构（主谓宾等）
    - 世界知识（各行业领域的常识）

    预训练使用的是"自回归语言建模"（Causal Language Modeling, CLM）的方式：
    给定前面的文字，预测下一个字/词（Next Token Prediction）。
    例如输入"今天天气"，模型要学会预测下一个token可能是"很"或"不"。

【本脚本做了什么？】
    1. 从 Qwen2.5-0.5B 的配置文件（config.json）随机初始化一个全新的模型
       （注意：不是加载预训练好的权重，而是从零开始训练！）
    2. 加载三个行业领域的中英文语料数据（影视娱乐、文学情感、新闻媒体）
    3. 对文本做分词（tokenize）和分块（chunking）处理
    4. 用 HuggingFace Trainer 进行训练
    5. 保存训练好的模型

【数据来源】
    数据集：BAAI/IndustryCorpus2（北京智源研究院的行业语料库）
    下载地址：https://modelscope.cn/datasets/BAAI/IndustryCorpus2/summary
    本项目只下载其中3个行业子集（不要下载全部！数据量巨大）：
    - film_entertainment（影视娱乐）
    - literature_emotion（文学情感）
    - news_media（新闻媒体）
    数据格式为 parquet 文件，每条数据包含一个 "text" 字段，即一段纯文本。

【启动方式】
    单卡训练：CUDA_VISIBLE_DEVICES=0 python qwen_pt.py
    多卡训练：accelerate launch --config_file accelerate_config.yaml qwen_pt.py
=====================================================================================
"""

import os
import torch
from datasets import load_dataset, Dataset
import wandb
from transformers import (
    AutoConfig,            # 自动加载模型配置（config.json）
    AutoModelForCausalLM,  # 自动加载因果语言模型（Causal LM，即 GPT 类自回归模型）
    AutoTokenizer,         # 自动加载分词器（tokenizer），负责���文本切分成 token
    DataCollatorForLanguageModeling,  # 数据整理器，负责把一个 batch 的样本拼成张量，并自动生成 labels
    Trainer,               # HuggingFace 的训练器，封装了训练循环、梯度更新、保存检查点等逻辑
    TrainingArguments,     # 训练超参数的配置类
    AdamW,                 # AdamW 优化器（这里虽然 import 了但默认用 Trainer 自带的优化器）
)
from utils.utils import find_files, tokenize_dataset

# =====================================================================================
# 第一部分：超参数与路径配置
# =====================================================================================

# TRUNK_SIZE（分块大小）：每个训练样本的 token 长度
# 预训练数据会被拼接后按照此长度切块。512 表示每个样本包含 512 个 token。
# 这个值越大，模型能看到的上下文越长，但显存占用也越大。
TRUNK_SIZE = 512

# TMP_PATH：数据集加载时的临时缓存目录（load_dataset 的 cache_dir）
TMP_PATH = ".cache/hf_datasets"

# DATA_PATH：预训练数据存放目录，下载的3个行业数据子集放在这里
DATA_PATH = "data/pt"

# OUTPUT_PATH：训练结果输出目录，包括模型检查点（checkpoint）和最终模型
OUTPUT_PATH = "results/pt"

# CONFIG_PATH：Qwen2.5-0.5B 模型的配置文件目录
# 注意：这个目录里只有 config.json 和 tokenizer 文件，没有模型权重（.safetensors）！
# 因为我们是从零开始预训练，不需要预训练好的权重。
CONFIG_PATH = "models/Qwen2.5-0.5B"

# WANDB_LOG：是否使用 Weights & Biases 记录训练日志（可视化loss曲线等）
# 如果你没有 wandb 账号或不需要，可以设为 False
WANDB_LOG = True

# 设置 PyTorch CUDA 内存分配策略
# max_split_size_mb:64 表示限制 CUDA 内存块的最大分割大小为 64MB
# 这有助于减少显存碎片化（fragmentation），防止 OOM（显存不足）错误
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

# =====================================================================================
# 第二部分：加载模型和分词器
# =====================================================================================

output_path = OUTPUT_PATH
model_path = CONFIG_PATH

# 第2.1步：加载模型配置
# AutoConfig.from_pretrained() 会读取 model_path 下的 config.json 文件
# config.json 定义了模型的架构超参数，例如：
#   - hidden_size: 896        (隐藏层维度)
#   - num_hidden_layers: 24   (Transformer 层数)
#   - num_attention_heads: 14 (注意力头数)
#   - vocab_size: 151936      (词表大小)
#   - intermediate_size: 4864 (FFN 中间层维度)
config = AutoConfig.from_pretrained(model_path)

# 第2.2步：根据配置创建一个全新的模型（随机初始化权重）
# ⚠️ 关键区别：
#   - AutoModelForCausalLM.from_pretrained() → 加载已训练好的权重（用于微调或推理）
#   - AutoModelForCausalLM.from_config()     → 随机初始化权重（用于从零预训练）
# 参数说明：
#   - torch_dtype=torch.bfloat16: 使用 BFloat16 半精度，显存占用减半，训练速度更快
#   - attn_implementation="sdpa": 使用 PyTorch 原生的 Scaled Dot-Product Attention（SDPA）
#     SDPA 是 PyTorch 2.0+ 内置的高效注意力实现，无需额外安装包
#     如果你安装了 flash-attn 包，也可以改为 "flash_attention_2" 以获得更好的性能
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

# 第2.3步：加载分词器（Tokenizer）
# 分词器的作用是将自然语言文本转换成模型能理解的数字序列（token ids）
# 例如："你好世界" → [23187, 11045, ...]
# Qwen 使用的是 BPE (Byte Pair Encoding) 分词算法，词表大小约 15 万
tokenizer = AutoTokenizer.from_pretrained(model_path)

# =====================================================================================
# 第三部分：数据预处理
# =====================================================================================

# tokenized_datapath：处理好的数据集的保存路径
# 第一次运行时会对原始数据进行分词和分块，然后保存到磁盘
# 后续再运行时直接从磁盘加载，避免重复处理（节省大量时间）
tokenized_datapath = os.path.join(DATA_PATH, "tokenized_dataset")

if not os.path.isdir(tokenized_datapath):
    # ======= 如果还没有处理好的数据，就从原始 parquet 文件开始处理 =======

    # 定义要使用的三个行业数据子目录
    # 这三个目录对应 BAAI/IndustryCorpus2 数据集中的三个行业领域：
    #   - film_entertainment: 影视娱乐领域语料
    #   - literature_emotion: 文学情感领域语料
    #   - news_media: 新闻媒体领域语料
    directories = [
        "film_entertainment",
        "literature_emotion",
        "news_media",
    ]

    # 第3.1步：查找所有 parquet 数据文件
    # find_files() 函数会遍历上述目录，找到所有 .parquet 文件的路径
    data_files = find_files(directories)

    # 第3.2步：使用 HuggingFace datasets 库加载数据
    # 参数说明：
    #   - "parquet": 指定数据格式为 parquet（一种高效的列式存储格式）
    #   - data_files: 要加载的文件路径列表
    #   - split="train": 指定为训练集
    #   - columns=["text"]: 只加载 "text" 列（原始文本内容），忽略其他列以节省内存
    #   - cache_dir: 缓存目录
    dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["text"], cache_dir=TMP_PATH)

    # 第3.3步：打乱数据集顺序
    # seed=42 保证每次打乱的结果一致（可复现）
    # 打乱数据很重要，因为来自同一行业的数据在文件中是连续存放的，
    # 如果不打乱，模型会先学完一个领域再学另一个，容易"遗忘"之前学的内容
    dataset = dataset.shuffle(seed=42)

    # 第3.4步：对数据集进行分词和分块处理
    # 定义 map 回调函数，对每个 batch 的数据进行处理
    def map_callback(examples):
        # tokenize_dataset() 函数做两件事：
        #   1. 分词（Tokenize）：将文本转换成 token id 序列，并在末尾添加 <|im_end|> 结束符
        #   2. 分块（Chunking）：将所有 token 拼接在一起，然后按 TRUNK_SIZE(512) 切成等长的块
        #      例如：3篇文章共 2000 个 token → 拼接 → 切成 3 块（512*3=1536），丢弃末尾 464 个 token
        #      这样做的好处是充分利用每个训练样本的长度，不浪费算力在 padding 上
        result, _ = tokenize_dataset(examples, tokenizer, TRUNK_SIZE)
        return result

    # dataset.map() 是 HuggingFace datasets 的核心方法，对数据集进行批量处理
    # 参数说明：
    #   - batched=True: 批量处理模式，一次处理多条数据（更高效）
    #   - batch_size=5000: 每批处理 5000 条原始数据
    #   - remove_columns: 移除原始列（"text"），只保留处理后的列（"input_ids", "attention_mask"）
    #   - num_proc=32: 使用 32 个进程并行处理（加速数据预处理）
    train_dataset = dataset.map(
        map_callback,
        batched=True,
        batch_size=5000,
        remove_columns=dataset.column_names,
        num_proc=32,
    )

    # 第3.5步：将处理好的数据集保存到磁盘，下次直接加载
    train_dataset.save_to_disk(tokenized_datapath)

# 第3.6步：从磁盘加载处理好的数据集
# 无论是刚处理完还是之前已经处理过，都从磁盘加载（保证一致性）
train_dataset = Dataset.load_from_disk(tokenized_datapath)

# =====================================================================================
# 第四部分：数据整理器（Data Collator）
# =====================================================================================

# DataCollatorForLanguageModeling 的作用：
#   1. 将一个 batch 中的多个样本拼成一个张量（tensor）
#   2. 自动生成 labels（标签）：对于因果语言模型（CLM），labels 就是 input_ids 向右移一位
#      例如 input_ids =  [A, B, C, D]
#           labels    =  [B, C, D, E]  （预测下一个 token）
#      实际实现中 labels 和 input_ids 相同，模型内部会自动做移位
#   3. mlm=False 表示不使用掩码语言建模（MLM，即 BERT 那种方式），
#      而是使用因果语言建模（CLM，即 GPT 那种从左到右预测的方式）
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# =====================================================================================
# 第五部分：训练超参数配置
# =====================================================================================

training_args = TrainingArguments(
    # --- 输出配置 ---
    output_dir=output_path,          # 模型检查点和训练结果的保存路径
    overwrite_output_dir=True,       # 如果输出目录已存在，覆盖它

    # --- 学习率配置 ---
    learning_rate=1e-5,              # 初始学习率（1e-5 = 0.00001）
                                     # 预训练通常用 1e-4 到 1e-5，太大模型不收敛，太小训练太慢
    warmup_ratio=0.1,               # 学习率预热比例：前 10% 的训练步数中，学习率从 0 线性增加到初始值
                                     # 预热可以防止训练初期梯度过大导致模型崩溃
    lr_scheduler_type="cosine",     # 学习率调度策略：余弦退火（cosine annealing）
                                     # 学习率会像余弦曲线一样，从最高值平滑下降到接近 0
                                     # 这比固定学习率效果好，让模型后期能更精细地调整参数

    # --- 训练轮次与批大小 ---
    num_train_epochs=3,              # 训练轮数：整个数据集过 3 遍
    per_device_train_batch_size=24,  # 每张 GPU 上的 batch 大小（每次送入 24 个样本）
    gradient_accumulation_steps=16,  # 梯度累积步数：每 16 个 mini-batch 才真正更新一次参数
                                     # 等效 batch size = 24 × 16 = 384（单卡情况下）
                                     # 梯度累积可以在显存不足时模拟大 batch 训练

    # --- 检查点保存 ---
    save_steps=10_000,               # 每训练 10000 步保存一次检查点（checkpoint）
    save_total_limit=3,              # 最多保留 3 个检查点，旧的会被自动删除（节省磁盘）

    # --- 显存优化 ---
    gradient_checkpointing=True,     # 梯度检查点：用时间换空间的策略
                                     # 正常训练时需要保存所有中间激活值（activation��用于反向传播
                                     # 开启后只保存部分激活值，需要时重新计算，显存占用大幅减少
                                     # 代价是训练速度约慢 20-30%

    # --- 混合精度训练 ---
    bf16=True,                       # 使用 BFloat16 混合精度训练
                                     # BF16 比 FP32 占用一半显存，且比 FP16 数值范围更大，更稳定
                                     # 适合现代 GPU（A100, H100, B300 等都支持 BF16）

    # --- 日志配置 ---
    logging_steps=10,                # 每 10 步记录一次日志（打印 loss 等）
    report_to="wandb",               # 日志报告方式：发送到 Weights & Biases 平台
                                     # 如果不想用 wandb，可以改成 "none" 或 "tensorboard"
)

# =====================================================================================
# 第六部分：Weights & Biases 日志初始化（可选）
# =====================================================================================

# wandb 是一个实验跟踪平台，可以实时查看 loss 曲线、学习率变化等
# 如果不需要，把 WANDB_LOG 设为 False 即可跳过
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if WANDB_LOG and local_rank == 0:
    # 通过环境变量读取，避免密钥泄漏：先 `export WANDB_API_KEY=xxx`
    # 或直接在终端执行 `wandb login`（key 会写入 ~/.netrc）
    wandb.login()
    wandb.init(
        project="qwen-0.5B-pt",  # wandb 项目名称
        name="qwen-0.5B-pt"      # 本次实验的运行名称
    )

# =====================================================================================
# 第七部分：创建 Trainer 并开始训练
# =====================================================================================

# 自定义优化器（可选，当前被注释掉了）
# 如果不指定优化器，Trainer 会默认使用 AdamW 优化器
# AdamW 是 Adam 的改进版，增加了正确的权重衰减（weight decay）
# optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# 初始化 Trainer
# Trainer 是 HuggingFace 对训练循环的高级封装，它帮你处理了：
#   - 数据加载和 batch 构造
#   - 前向传播、计算 loss、反向传播、梯度更新
#   - 混合精度训练
#   - 梯度累积
#   - 学习率调度
#   - 检查点保存和恢复
#   - 多卡分布式训练（配合 accelerate 使用）
#   - 日志记录
trainer = Trainer(
    model=model,                  # 要训练的模型
    args=training_args,           # 训练超参数
    data_collator=collator,       # 数据整理器（负责 batch 构造和 labels 生成）
    train_dataset=train_dataset,  # 训练数据集
    # optimizers=(optimizer, None)  # 可选：自定义优化器，None 表示使用默认的学习率调度器
)

# 清理 GPU 显存缓存，释放之前数据预处理等操作占用的显存
torch.cuda.empty_cache()

# 🚀 开始训练！
# trainer.train() 会执行完整的训练循环：
#   for epoch in range(num_train_epochs):      # 遍历每个 epoch
#       for batch in dataloader:                # 遍历每个 batch
#           loss = model(batch)                 # 前向传播，计算 loss
#           loss.backward()                     # 反向传播，计算梯度
#           if step % gradient_accumulation_steps == 0:
#               optimizer.step()                # 更新参数
#               scheduler.step()                # 更新学习率
#               optimizer.zero_grad()           # 清零梯度
trainer.train()

# 训练完成后保存最终模型
trainer.save_model()                    # 保存模型权重到 output_path 目录
tokenizer.save_pretrained(output_path)  # 保存分词器到同一目录（推理时需要）
