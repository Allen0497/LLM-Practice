# 🤖 mini_qwen

> **大模型训练 + 应用全流程实战项目**  
> 用 Qwen2.5-0.5B 这个小巧的基座，把 PT → SFT → DPO/PPO/GRPO → 蒸馏 → Agentic RAG → 部署的完整链路跑一遍。  
> 配套 **14 篇完全指南**（共 12,000+ 行，500+ KB），每个脚本都有详细中文注释。

---

## 🎯 项目定位

```
不是教你"调用 API"，而是教你：
  ✓ 从随机权重开始，怎么把"语言能力"压进神经网络（预训练）
  ✓ 怎么让模型学会听人话（SFT）
  ✓ 怎么让模型分清好坏（DPO/RM/PPO/GRPO 三条路线）
  ✓ 怎么让大模型把能力"传授"给小模型（蒸馏）
  ✓ 怎么让模型自己查资料、调工具、做决策（Agentic RAG）
  ✓ 怎么把训好的模型真正"用起来"（合并 / 部署 / 胜率评测）
  ✓ 怎么解决训练时显存爆炸的工程问题
```

**适合人群：** 大模型初学者、想从应用层进阶到训练层的工程师、需要从零搞清 RLHF / R1 路线的同学。

---

## ✨ 项目特色

| 特色 | 说明 |
|---|---|
| 🎓 **配套 14 篇完全指南** | 每个训练阶段一篇，含 ASCII 图、表格、Q&A、公式推导 |
| 🐣 **小模型跑通全流程** | 用 0.5B 让单机也能完整跑完所有阶段（不需要多机集群）|
| 🔄 **三种偏好对齐路线** | DPO / RM+PPO / GRPO 全覆盖（包含 DeepSeek-R1 路线）|
| 📝 **每行代码都有注释** | 中文注释解释 WHY 而非 WHAT，对照笔记可深入理解 |
| 🚀 **从训练到落地闭环** | LoRA 合并 + vLLM 部署 + Agentic RAG 应用 |
| 🛠 **工程实战** | 显存监控、OOM 排查、DeepSpeed ZeRO 配置、混合精度 |

---

## 📂 项目架构

### 目录结构

```
mini_qwen/
├── README.md                        # ★ 本文件
├── README_old.md                    # 旧版 README 备份
│
├── accelerate_config.yaml           # accelerate + DeepSpeed ZeRO 1 多卡配置
├── download_data.sh                 # 一键下载所有数据集
├── requirements.txt                 # 主依赖（除 GRPO）
├── requirements_grpo.txt            # GRPO 专用依赖（含 vLLM）
│
├── data/                            # 数据集（gitignore，运行 download_data.sh 后生成）
├── results/                         # 训练 checkpoint（gitignore）
├── logs/                            # 训练日志
├── wandb/                           # wandb 记录
│
├── ── 训练脚本（11 个）──────────────────────────
├── qwen_pt.py                       # 预训练
├── qwen_pt_continue.py              # 续训练
├── qwen_sft.py                      # 监督微调
├── qwen_eval.py                     # 评估（MMLU 等）
├── qwen_dpo_data.py                 # DPO 偏好数据合成
├── qwen_dpo.py                      # DPO 训练
├── qwen_rm.py                       # 奖励模型训练
├── qwen_ppo.py                      # PPO 训练
├── qwen_grpo.py                     # GRPO 训练（R1 风格）
├── qwen_distill_data.py             # 蒸馏数据生成
├── qwen_distill.py                  # 蒸馏训练
│
├── ── 应用脚本（4 个）──────────────────────────
├── qwen_agentic_rag.py              # Agentic RAG 智能体（FAISS+smolagents）
├── qwen_judge.py                    # LLM-as-a-Judge 胜率评估
├── qwen_chat.py                     # 基础对话推理
├── qwen_r1_chat.py                  # 思维链对话推理（R1 风格）
│
├── ── 工程脚本 ────────────────────────────────
├── qwen_mem.py                      # 显存监控 + LoRA SFT 教学版
│
├── utils/                           # 工具模块
│   ├── utils.py                     # 通用工具（find_files、format 函数、SemanticRetriever）
│   ├── datasets.py                  # 数据集预处理
│   ├── distill_utils.py             # 蒸馏专用工具
│   ├── grpo_utils.py                # GRPO 专用工具（reward 函数等）
│   ├── rm_utils.py                  # RM 专用工具
│   └── merge_peft_adapter.py        # LoRA 适配器合并到基座
│
└── notes/                           # ★ 14 篇笔记 + 1 篇总结
    ├── 00-大模型学习总结.md
    ├── 1-预训练完全指南.md
    ├── 2-Qwen网络结构详解.md
    ├── 3-续训练完全指南.md
    ├── 4-监督微调SFT完全指南.md
    ├── 5-大模型评估完全指南.md
    ├── 6-DPO偏好数据合成完全指南.md
    ├── 7-DPO直接偏好优化完全指南.md
    ├── 8-训练奖励模型RM完全指南.md
    ├── 9-PPO近端策略优化完全指南.md
    ├── 10-GRPO群体相对策略优化完全指南.md
    ├── 11-知识蒸馏完全指南.md
    ├── 12-Agentic-RAG完全指南.md
    ├── 13-LLM评判与模型部署完全指南.md
    └── 14-显存优化与LoRA实战完全指南.md
```

### 全流程架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          🚀 mini_qwen 全流程架构                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  数据                  训练                   评估             应用            │
│  ──                    ──                     ──               ──             │
│                                                                              │
│  IndustryCorpus2 ──► PT (qwen_pt) ──────────►                                │
│   (5.3B tokens)        ↓                                                     │
│                     续训练 (qwen_pt_continue)                                │
│                        ↓                                                     │
│  Infinity-Instruct ►  SFT (qwen_sft) ──────►  qwen_eval (MMLU)               │
│                        ↓                                                     │
│           ┌────────────┼────────────────┐                                    │
│           ▼            ▼                ▼                                    │
│       DPO 路线      RM+PPO 路线      GRPO 路线                              │
│  Infinity-Pref      stack-exchange    NuminaMath                            │
│   ↓                   ↓                ↓                                    │
│  qwen_dpo_data       qwen_rm          qwen_grpo                             │
│  qwen_dpo            qwen_ppo          ↓                                    │
│                                       qwen_r1_chat                          │
│           └────────────┬────────────────┘                                    │
│                        ▼                                                     │
│           numina-r1-7b ► 蒸馏 (qwen_distill_data → qwen_distill)            │
│                        ↓                                                     │
│  ─────────────── 应用层 ───────────────                                     │
│                        │                                                     │
│  baidu_baike + gte-small-zh ► Agentic RAG (qwen_agentic_rag)                │
│                        │                                                     │
│  ─────────────── 部署 ───────────────                                       │
│                        │                                                     │
│  LoRA → merge_peft_adapter → vLLM → OpenAI 兼容 API                         │
│                        │                                                     │
│  TLDR ► qwen_judge（胜率评估）                                               │
│                                                                              │
│  ─────────────── 工程必修 ───────────────                                   │
│                                                                              │
│  qwen_mem.py：显存监控 / LoRA / DeepSpeed ZeRO / OOM 排查                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 环境准备

### 1. Python 版本

推荐 **Python 3.10**（项目在该版本上充分测试）。

### 2. 安装依赖（主流程）

```bash
pip install -r requirements.txt
```

主要依赖版本：

```
torch          == 2.5.1+cu118
transformers   == 4.45.0
trl            == 0.11.4
peft           == 0.14.0
accelerate     == 1.1.1
deepspeed      == 0.15.4
distilabel     == 1.5.3
modelscope     == 1.23.1
```

### 3. 安装依赖（GRPO 专用）

GRPO 训练需要 **vLLM** 作为高速采样后端，且依赖更新版本的 transformers/trl：

```bash
# 建议另开一个 conda 环境
conda create -n mini_qwen_grpo python=3.10
conda activate mini_qwen_grpo
pip install -r requirements_grpo.txt
```

主要差异：

```
transformers   == 4.49.0  # 比主环境新
trl            == 0.15.2  # 比主环境新
vllm           == 0.7.3   # GRPO 必需
math_verify    == 0.5.2   # 数学题答案校验
openai         == 1.66.2  # 调用 DeepSeek API
```

### 4. 多卡训练配置

`accelerate_config.yaml` 默认 3 卡训练 + DeepSpeed ZeRO 1 + bf16：

```yaml
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 3        # ← 改成你的 GPU 数量
```

如需改 ZeRO 阶段（2/3）或 CPU offload，请编辑此文件。详见 [第 14 篇](notes/14-显存优化与LoRA实战完全指南.md)。

---

## 📦 数据下载

```bash
bash download_data.sh
```

涉及数据集（共 ~10 个，存在 `data/` 目录）：

| 数据集 | 用途 | 体积 |
|---|---|---|
| BAAI/IndustryCorpus2 | 预训练（影视/新闻/文学）| ~5.3B tokens |
| Hendrycks/MMLU | 评估 | ~50 MB |
| BAAI/Infinity-Instruct | SFT 指令微调 | 数 GB |
| BAAI/Infinity-Preference | DPO 偏好对齐 | ~100 MB |
| HuggingFaceH4/numina-deepseek-r1-qwen-7b | 蒸馏（老师生成的推理数据） | ~500 MB |
| swift/stack-exchange-paired | RM / PPO 偏好数据 | ~1 GB |
| AI-MO/NuminaMath-TIR | GRPO 推理数据 | ~200 MB |
| gxlzgdmds/baidu_baike | Agentic RAG 知识库 | 数 GB |
| AI-ModelScope/gte-small-zh | RAG 中文 Embedding 模型 | ~30 MB |

> **TIPS**：如果只想跑某个阶段，请打开 `download_data.sh` 注释掉无关行，分次下载。

---

## 🚀 完整运行流程

### 阶段 ①：预训练 PT （[第 1 篇](notes/1-预训练完全指南.md)）

让随机初始化的模型学会"语言"。

```bash
# 单卡（慢，仅调试用）
CUDA_VISIBLE_DEVICES=0 python qwen_pt.py

# 多卡（推荐，3 卡 ~20 小时跑完 0.5B 模型）
accelerate launch --config_file accelerate_config.yaml qwen_pt.py

# 后台运行
nohup accelerate launch --config_file accelerate_config.yaml \
     qwen_pt.py > logs/output_pt.log 2>&1 &
```

### 阶段 ②：续训练（可选，[第 3 篇](notes/3-续训练完全指南.md)）

```bash
python qwen_pt_continue.py
```

### 阶段 ③：监督微调 SFT（[第 4 篇](notes/4-监督微调SFT完全指南.md)）

让模型学会"听懂人话"，采用 ChatML 格式 + 只对 assistant 部分计算 loss。

```bash
# 多卡（推荐）
accelerate launch --config_file accelerate_config.yaml qwen_sft.py

# 后台
nohup accelerate launch --config_file accelerate_config.yaml \
     qwen_sft.py > logs/output_sft.log 2>&1 &
```

### 阶段 ④：评估（[第 5 篇](notes/5-大模型评估完全指南.md)）

```bash
python qwen_eval.py \
    --checkpoint-path results/sft/checkpoint-XXX \
    --eval_data_path data/mmlu
```

### 阶段 ⑤：偏好对齐（三选一或组合）

#### 路线 A：DPO（[第 6-7 篇](notes/7-DPO直接偏好优化完全指南.md)）

```bash
# Step 1：合成偏好数据
python qwen_dpo_data.py

# Step 2：DPO 训练
accelerate launch --config_file accelerate_config.yaml qwen_dpo.py
```

#### 路线 B：RM + PPO（[第 8-9 篇](notes/9-PPO近端策略优化完全指南.md)）

```bash
# Step 1：训奖励模型
python qwen_rm.py

# Step 2：PPO 训练（用 RM 当奖励信号）
python qwen_ppo.py
```

#### 路线 C：GRPO（[第 10 篇](notes/10-GRPO群体相对策略优化完全指南.md)，DeepSeek-R1 路线）

```bash
# 切换到 GRPO 专用环境
conda activate mini_qwen_grpo

CUDA_VISIBLE_DEVICES=0 python qwen_grpo.py
```

### 阶段 ⑥：知识蒸馏（[第 11 篇](notes/11-知识蒸馏完全指南.md)）

```bash
# Step 1：用 DeepSeek-R1 等大模型生成推理数据（也可直接用现成的蒸馏数据集）
python qwen_distill_data.py

# Step 2：用蒸馏数据 SFT 小模型
accelerate launch --config_file accelerate_config.yaml qwen_distill.py
```

### 阶段 ⑦：LoRA 合并（[第 13 篇](notes/13-LLM评判与模型部署完全指南.md)）

LoRA 训练得到的是 adapter，部署前要合并回基座：

```bash
# 修改 utils/merge_peft_adapter.py 里的路径
python utils/merge_peft_adapter.py
```

### 阶段 ⑧：模型推理验证

```bash
# 普通对话（SFT 后的模型）
python qwen_chat.py

# 思维链对话（GRPO 后的 R1 风格模型）
python qwen_r1_chat.py
```

### 阶段 ⑨：胜率评估（[第 13 篇](notes/13-LLM评判与模型部署完全指南.md)）

```bash
# 用本地 70B 模型当裁判
python qwen_judge.py \
    --model_name_or_path results/grpo/final_model \
    --num_examples 500

# 用 GPT-4o-mini 当裁判（需要 OPENAI_API_KEY）
python qwen_judge.py \
    --model_name_or_path results/grpo/final_model \
    --judge_model gpt-4o-mini \
    --num_examples 500
```

### 阶段 ⑩：Agentic RAG 智能体应用（[第 12 篇](notes/12-Agentic-RAG完全指南.md)）

```bash
# 修改 qwen_agentic_rag.py 里的 DS_KEY（DeepSeek API Key）
python qwen_agentic_rag.py
```

启动后是交互式对话框，模型会自主决定调用本地知识库检索 / 网络搜索来回答问题。

### 阶段 ⑪：vLLM 部署（OpenAI 兼容 API）

```bash
vllm serve results/grpo/final_model \
    --port 8000 \
    --tensor-parallel-size 1
```

启动后可用任意 OpenAI SDK 调用。

---

## 🛠 工程实践

### 显存优化（[第 14 篇](notes/14-显存优化与LoRA实战完全指南.md)）

显存吃紧时按"性价比"递增上招数：

| 招数 | 显存收益 | 速度代价 | 配置方式 |
|---|---|---|---|
| `bf16=True` | 25%+ | +30%（更快）| training_args |
| `flash_attention_2` | 30% | +200%（更快）| `attn_implementation="flash_attention_2"` |
| `gradient_accumulation` | 0（等效大 batch） | 慢更新 | `gradient_accumulation_steps=N` |
| `gradient_checkpointing` | 70%+（激活）| -25% | training_args |
| LoRA | 80%+ | 略降 | `peft_config=LoraConfig(...)` |
| QLoRA | 95%+ | 略降 | `BitsAndBytesConfig(load_in_4bit=True)` |
| ZeRO-1/2/3 | 多卡分摊 | +通信 | `accelerate_config.yaml` |

### 显存监控

```bash
# 教学版：每个 step 打印 CUDA Memory Summary
python qwen_mem.py
```

### OOM 排查 5 步

1. **看时机**：加载就 OOM → 模型本身放不下；训练几十步后 OOM → 显存泄漏
2. **看大户**：`torch.cuda.memory_summary()` 看 Peak Usage
3. **按代价递增**：`bf16` → `gradient_checkpointing` → 减 batch → 减 seq_len → LoRA → ZeRO
4. **每改一项跑 1 step 验证**
5. **还不行就缩模型规模**

---

## 📚 笔记导航

每个训练阶段都有一篇配套的"完全指南"，包含原理、代码逐行解读、调参经验、常见 Q&A：

| # | 标题 | 一句话速记 |
|---|---|---|
| [00](notes/00-大模型学习总结.md) | **大模型学习总结** | **★ 把所有内容串起来的总图，先看这篇** |
| [1](notes/1-预训练完全指南.md) | 预训练完全指南 | next-token，把"语言能力"压进权重 |
| [2](notes/2-Qwen网络结构详解.md) | Qwen 网络结构详解 | Decoder-Only / RoPE / GQA / SwiGLU / RMSNorm |
| [3](notes/3-续训练完全指南.md) | 续训练完全指南 | 通用基座上继续吃领域数据 |
| [4](notes/4-监督微调SFT完全指南.md) | 监督微调 SFT | 用指令-回答对让模型"听懂人话" |
| [5](notes/5-大模型评估完全指南.md) | 大模型评估 | MMLU / CMMLU / C-Eval 标准化考试 |
| [6](notes/6-DPO偏好数据合成完全指南.md) | DPO 偏好数据合成 | 用 LLM 造 chosen/rejected 数据 |
| [7](notes/7-DPO直接偏好优化完全指南.md) | DPO 直接偏好优化 | 不用 RM，直接优化偏好对 |
| [8](notes/8-训练奖励模型RM完全指南.md) | 训练奖励模型 RM | 学一个"打分器"判断回答好坏 |
| [9](notes/9-PPO近端策略优化完全指南.md) | PPO 近端策略优化 | 用 RM 当奖励信号做 RL 训练 |
| [10](notes/10-GRPO群体相对策略优化完全指南.md) | GRPO | DeepSeek-R1 路线，无 RM、组内归一化 |
| [11](notes/11-知识蒸馏完全指南.md) | 知识蒸馏 | 大模型当老师，能力转移给小模型 |
| [12](notes/12-Agentic-RAG完全指南.md) | Agentic RAG | FAISS + smolagents + 工具调用 |
| [13](notes/13-LLM评判与模型部署完全指南.md) | LLM 评判与部署 | 胜率评测 + ChatML + LoRA 合并 |
| [14](notes/14-显存优化与LoRA实战完全指南.md) | 显存优化与 LoRA 实战 | OOM 排查 + 四把省显存利器 |

> **学习建议**：第一遍按顺序跟着代码跑通；第二遍带着问题反复读笔记；第三遍把所有脚本改造成你自己想训的模型。

---

## ❓ 常见问题

### Q1：我只有一张卡能跑吗？

> 可以。0.5B 模型单卡（≥ 24GB）即可跑完所有阶段。
> 把命令里的 `accelerate launch` 改为 `CUDA_VISIBLE_DEVICES=0 python` 即可。
> 全流程估算时间（单卡 A100）：PT ~65h、SFT ~6h、DPO ~3h、PPO ~5h、GRPO ~10h。

### Q2：可以把 0.5B 换成 7B / 32B 吗？

> 可以，但需要：
> - 修改脚本中的 `model_path` 指向新模型
> - 启用更激进的 ZeRO 阶段（编辑 `accelerate_config.yaml` 设 `zero_stage: 3`）
> - 视情况启用 LoRA / QLoRA 节省显存
> - 详见 [第 14 篇](notes/14-显存优化与LoRA实战完全指南.md)

### Q3：DPO / PPO / GRPO 三条路线该选哪个？

| 选哪个 | 理由 |
|---|---|
| 数据有限、追求简单 | DPO |
| 要精细控制、奖励信号复杂 | RM + PPO |
| 推理任务（数学/代码）、有规则可判 | GRPO |

### Q4：训练中报 OOM 怎么办？

> 见 [第 14 篇](notes/14-显存优化与LoRA实战完全指南.md) 的 5 步排查流程。
> 速效药：先开 `bf16=True`、`gradient_checkpointing=True`、`per_device_train_batch_size=1`、`gradient_accumulation_steps=16`。

### Q5：训练完的模型怎么部署？

> 三步：
> 1. `python utils/merge_peft_adapter.py`（合并 LoRA）
> 2. `python qwen_chat.py`（本地推理验证）
> 3. `vllm serve <merged_model_path>`（生产部署）

### Q6：可以把 mini_qwen 接入 ChatGPT 那种网页吗？

> 可以。`vllm serve` 启动的是 OpenAI 兼容 API，可直接对接：
> - LobeChat / NextChat 等开源前端
> - LangChain / LlamaIndex 等框架
> - 自己写的 Web 应用（直接用 `openai` Python SDK，base_url 指向 vLLM）

### Q7：项目里的 wandb 怎么配置？

> 训练脚本里有 `wandb.login(key=...)` 行，填你自己的 API key 即可。
> 也可以在终端执行 `wandb login`，把 key 写入 `~/.netrc`。
> 不想用 wandb 可以把 `report_to="wandb"` 改成 `report_to="none"`。

### Q8：能完整跑通需要哪些前置知识？

> 推荐前置（不一定全部需要）：
> - PyTorch 基础（tensor、autograd、nn.Module）
> - Transformer 架构（[第 2 篇](notes/2-Qwen网络结构详解.md) 会讲）
> - HuggingFace 生态（transformers / datasets / accelerate）
> - Linux 命令行 + GPU 基本概念

---

## 🙏 致谢

- 原作者：**古希腊掌管代码的神**（项目最初版本）
- 核心依赖：HuggingFace transformers / TRL / PEFT / DeepSpeed / vLLM / smolagents
- 训练数据：BAAI、AI-MO、HuggingFaceH4、modelscope 等开源社区
- DeepSeek 团队：开源 R1 让小模型也能学会推理

---

## 📜 License

代码部分遵循原项目 License。  
笔记内容采用 **CC BY-SA 4.0** 协议，欢迎转载、改写、二次创作，请保留原作者信息。

---

## 🚀 一句话寄语

> **学完 mini_qwen，你不止会"调用大模型"，更会"造大模型"。**  
> 训练从来不是终点，理解每一步在解决什么问题，才是真本事。  
> Happy training! 🤖

---

<details>
<summary>📋 快速命令参考（点击展开）</summary>

```bash
# ── 环境 ──
pip install -r requirements.txt           # 主环境
pip install -r requirements_grpo.txt      # GRPO 环境
bash download_data.sh                     # 下数据

# ── 训练 ──
accelerate launch --config_file accelerate_config.yaml qwen_pt.py        # 预训练
accelerate launch --config_file accelerate_config.yaml qwen_sft.py       # SFT
python qwen_dpo_data.py && python qwen_dpo.py                            # DPO
python qwen_rm.py && python qwen_ppo.py                                  # PPO
python qwen_grpo.py                                                       # GRPO
python qwen_distill_data.py && python qwen_distill.py                    # 蒸馏

# ── 评估 ──
python qwen_eval.py --checkpoint-path <ckpt> --eval_data_path data/mmlu  # MMLU
python qwen_judge.py --model_name_or_path <ckpt> --num_examples 500      # 胜率

# ── 部署 ──
python utils/merge_peft_adapter.py                                       # 合并 LoRA
python qwen_chat.py                                                      # 普通对话
python qwen_r1_chat.py                                                   # 思维链对话
vllm serve <merged_model_path>                                           # OpenAI API

# ── 应用 ──
python qwen_agentic_rag.py                                               # Agentic RAG

# ── 调试 ──
python qwen_mem.py                                                       # 显存监控
```

</details>
