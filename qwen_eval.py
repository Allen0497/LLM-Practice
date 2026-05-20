"""
qwen_eval.py — 大模型评估脚本（MMLU 基准测试）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
背景：大模型训练完后，怎么知道它"够不够聪明"？

  预训练 → SFT → 偏好对齐 → 【评估】 → 部署
                              ↑ 我们在这里

  评估的目的：用标准化的考试题给模型打分，量化它的能力
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本脚本评估的基准：MMLU（Massive Multitask Language Understanding）
  - 由加州大学伯克利分校提出的大模型综合评估基准
  - 包含 57 个学科的多选题（4 个选项 ABCD）
  - 涵盖 STEM、人文、社科、其他四大类
  - 是大模型最常用的"通用知识"评测基准之一

核心评估方法：
  1. 给模型一道选择题（含 ABCD 四个选项）
  2. 看模型对 " A"、" B"、" C"、" D" 这4个 token 的预测概率
  3. 概率最高的就是模型的答案
  4. 与标准答案对比，计算准确率
"""

import os
from typing import List
import pandas as pd
import numpy as np
import argparse
import torch
from tqdm import tqdm
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


# ── 模型加载 ───────────────────────────────────────────────────────────────────

def load_models_tokenizer(args):
    """
    加载待评估��模型和分词器
    评估阶段模型设为 eval() 模式（关闭 dropout、不计算梯度）
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        pad_token='<|extra_0|>',       # 指定 padding token（用特殊保留 token）
        eos_token='<|endoftext|>',     # 指定句末结束 token
        # padding_side='left' 是评估关键：
        # 因果语言模型预测的是"最后一个 token 之后的下一个 token"
        # 如果右侧 padding，pad token 会出现在末尾，模型会基于 pad 做预测，结果错误
        # 左侧 padding 时，真实内容的最后一个 token 仍在序列末尾
        padding_side='left',
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="auto",            # 自动把模型分配到可用 GPU
        trust_remote_code=True
    ).eval()                          # ⚠️ 评估模式：关闭 dropout，固定 BatchNorm

    # 加载生成配置（虽然本脚本只取 logits，不做生成，但保持配置一致）
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True
    )
    return model, tokenizer


# ── Prompt 构造 ────────────────────────────────────────────────────────────────

def format_example(line, include_answer=True):
    """
    把一行数据格式化为题目 prompt

    输入示例（DataFrame 的一行）：
      question: "What is 2+2?"
      A: "3"   B: "4"   C: "5"   D: "6"
      answer: "B"

    输出（include_answer=True，用于 few-shot 示例）：
      Question: What is 2+2?
      A. 3
      B. 4
      C. 5
      D. 6
      Answer: B

    输出（include_answer=False，用于实际测试题）：
      Question: What is 2+2?
      A. 3
      B. 4
      C. 5
      D. 6
      Answer:                  ← 留白，让模型预测
    """
    example = "Question: " + line["question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\nAnswer: " + line["answer"] + "\n\n"
    else:
        example += "\nAnswer:"
    return example


def generate_few_shot_prompt(k, subject, dev_df):
    """
    生成 few-shot prompt：在测试题前面放 k 个带答案的示例

    Few-shot Learning（少样本学习）：
      在 prompt 中给模型几个"做题示范"，让它学会题目格式后再答题
      MMLU 标准评测使用 5-shot（k=5）

    生成的 prompt 示例（k=2）：
      The following are multiple choice questions (with answers) about machine learning.

      Question: 第1道示例题...
      Answer: A

      Question: 第2道示例题...
      Answer: C

      ↑ 这是 few-shot 部分，后面会拼上真正要答的题目
    """
    def format_subject(subject):
        # "high_school_math" → "high school math"
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()

    # 主题说明：告诉模型这是关于什么领域的题
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )

    if k == -1:
        k = dev_df.shape[0]
    # 从 dev 集（开发集）取 k 个带答案的示例
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True,
        )
    return prompt


# ── 核心评估逻辑 ───────────────────────────────────────────────────────────────

def get_logits(tokenizer, model, inputs: List[str]):
    """
    把 prompt 送入模型，获取最后一个位置的 logits（每个 token 的预测分数）

    这是评估的核心：我们不让模型自由生成文本，而是直接看它对下一个 token 的预测概率
    """
    # 1. 分词，padding='longest' 表示按 batch 内最长序列做 padding
    input_ids = tokenizer(inputs, padding='longest')["input_ids"]
    input_ids = torch.tensor(input_ids, device=model.device)

    # 2. 长序列截断：如果超过最大长度，从前面截掉（保留末尾的题目部分）
    if input_ids.shape[1] > args.max_seq_len:
        input_ids = input_ids[:, input_ids.shape[1] - args.max_seq_len + 1:]
    tokens = {"input_ids": input_ids}

    # 3. attention_mask：告诉模型哪些位置是真实内容（非 pad）
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # 4. 前向传播，获取 logits
    # outputs 形状: [batch_size, seq_len, vocab_size]
    outputs = model(input_ids, attention_mask=attention_mask)["logits"]

    # 5. 只取最后一个位置的 logits（即"Answer:"之后要预测的下一个 token）
    # 形状: [batch_size, vocab_size]
    logits = outputs[:, -1, :]

    # 6. softmax 转成概率分布
    log_probs = torch.nn.functional.softmax(logits, dim=-1)
    return log_probs, {"tokens": tokens}


@torch.no_grad()  # 评估时不需要梯度，节省显存
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    k=5,
    dev_df=None,
    few_shot=False,
    save_result_dir=None,
    batch_size=1,
    **kwargs,
):
    """
    评估单个学科的所有测试题

    流程：
      1. 构造 few-shot prompt（用 dev 集的 5 道题做示范）
      2. 对每道测试题，把 few-shot + 测试题拼成完整 prompt
      3. 送入模型，获取最后位置的 logits
      4. 只看 ABCD 四个 token 的概率，取最大的那个作为预测答案
      5. 与标准答案对比，记录正确与否
    """
    result = []   # 存放每道题的预测答案
    score = []    # 存放每道题的对错（1/0）

    # 生成 few-shot 前缀（如果开启 few_shot）
    few_shot_prompt = (
        generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else []
    )
    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
    if args.debug:
        print(f"few_shot_prompt: {few_shot_prompt}")

    # 关键：把" A"、" B"、" C"、" D"四个 token 编码成 ID
    # 注意前面的空格！因为 prompt 末尾是 "Answer:"，模型预测的下一个 token 通常带空格前缀
    # tokenizer(" A")["input_ids"] 返回的是 [token_id]，4 个拼起来是 [id_A, id_B, id_C, id_D]
    choices_ids = torch.tensor(
        tokenizer(" A")["input_ids"] + tokenizer(" B")["input_ids"] +
        tokenizer(" C")["input_ids"] + tokenizer(" D")["input_ids"]
    ).unsqueeze(0).to(model.device)

    # 按 batch 遍历测试题
    idx_list = list(range(0, len(test_df), batch_size))
    for i in tqdm(idx_list):
        full_prompt_list = []
        answer_list = []
        # 处理当前 batch 内的每道题
        for row in test_df.iloc[i:i+batch_size].to_dict(orient='records'):
            # 测试题不含答案（让模型预测）
            question = format_example(row, include_answer=False)
            # 拼接 few-shot 示例 + 测试题
            full_prompt = few_shot_prompt + question
            full_prompt_list.append(full_prompt)
            if 'answer' in row:
                answer_list.append(row['answer'])

        # 模型前向传播，获取最后位置的概率分布
        # logits 形状: [batch_size, vocab_size]，包含全部 ~15 万 token 的概率
        logits, input_info = get_logits(tokenizer, model, full_prompt_list)

        # 关键操作：只取 ABCD 四个 token 的概率，并重新 softmax
        # gather: 根据 choices_ids 的索引，从 vocab_size 维度中"挑"出对应位置的概率
        # 结果形状: [batch_size, 4]，每行是 [P(A), P(B), P(C), P(D)]
        # 再 softmax 一次，让 4 个概率相加为 1（归一化）
        softval = logits.gather(1, choices_ids.expand(logits.size(0), -1)).softmax(1)

        # BF16/FP16 转 FP32，避免数值精度问题
        if softval.dtype in {torch.bfloat16, torch.float16}:
            softval = softval.to(dtype=torch.float32)
        probs = softval.detach().cpu().numpy()

        # 处理每道题的结果
        for i in range(len(probs)):
            # 记录每个选项的概率
            for j, choice in enumerate(choices):
                all_probs[f"prob_{choice}"].append(probs[i][j])
            # argmax 取概率最高的选项作为预测答案
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs[i])]

            # 与标准答案对比
            if answer_list != []:
                correct = 1 if pred == answer_list[i] else 0
                score.append(correct)
                if args.debug:
                    print(f'{question} pred: {pred} ref: {answer_list[i]}')
            result.append(pred)

    # 保存结果到 csv（包含每道题的预测、各选项概率、对错）
    if save_result_dir:
        test_df["model_output"] = result
        for i, choice in enumerate(choices):
            test_df[f"prob_{choice}"] = all_probs[f"prob_{choice}"]
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    return score


# ── MMLU 总分计算 ──────────────────────────────────────────────────────────────

def cal_mmlu(res):
    """
    计算 MMLU 整体准确率，按四大类别（STEM/人文/社科/其他）分别统计

    最终输出示例：
      stem ACC: 35.42
      Humanities ACC: 38.71
      other ACC: 41.23
      social ACC: 42.85
      AVERAGE ACC: 39.55
    """
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0
    hard_cnt = 0
    hard_acc_sum = 0.0

    # 遍历四大类别
    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.0
        acc_norm_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0

        # 遍历该类别下的所有学科
        for tt in TASK_NAME_MAPPING[class_]:
            acc_sum += sum(res[tt])      # 该学科答对的题数
            cnt += len(res[tt])          # 该学科总题数

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    print("\n\n\n", "total cnt:", cnt, "\n")
    # 打印每个类别的准确率
    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict:
            print("%s ACC: %.2f " % (k, acc_sum_dict[k] / cnt_dict[k] * 100))
    # 打印总平均准确率
    print("AVERAGE ACC:%.2f " % (acc_sum / cnt * 100))


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main(args):
    # 1. 加载模型
    model, tokenizer = load_models_tokenizer(args)

    dev_result = {}
    # 2. 遍历 57 个学科
    for subject_name in tqdm(SUBJECTS):
        # dev 集：用作 few-shot 示例（每个学科 5 道题）
        dev_file_path = os.path.join(
            args.eval_data_path, "dev", f"{subject_name}_dev.csv"
        )
        # test 集：实际评测的题目
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}_test.csv"
        )
        dev_df = pd.read_csv(
            dev_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )

        # 3. 评估当前学科
        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            test_df,
            dev_df=dev_df,
            k=5,                 # 5-shot
            few_shot=True,       # 启用 few-shot
            save_result_dir=f"outs/mmlu_eval_result",
            batch_size=args.batch_size
        )
        dev_result[subject_name] = score

    # 4. 计算总分
    cal_mmlu(dev_result)


# ── MMLU 学科分类（57 个学科按 4 大类划分）─────────────────────────────────────

TASK_NAME_MAPPING = {
    # STEM：科学、技术、工程、数学（19 个学科）
    "stem": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    # 人文学科（13 个）
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    # 其他（13 个）
    "other": [
        "business_ethics", "college_medicine", "human_aging", "management",
        "marketing", "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
        "global_facts", "clinical_knowledge",
    ],
    # 社会科学（12 个）
    "social": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ],
}
# 把所有学科平铺成一个列表（共 57 个）
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
# 选项标签
choices = ["A", "B", "C", "D"]


# ── 命令行入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c", "--checkpoint-path", type=str, help="Checkpoint path",
        default="results/pt",
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")

    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, help="Path to eval data")
    group.add_argument(
        "--max-seq-len", type=int, default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--debug", action="store_true", default=False, help="Print infos."
    )
    group.add_argument(
        "--batch-size", type=int, default=1, help="batch size",
    )

    args = parser.parse_args()
    set_seed(args.seed)   # 固定随机种子，保证结果可复现

    main(args)
