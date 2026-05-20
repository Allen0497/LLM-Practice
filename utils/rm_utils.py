"""
奖励模型（Reward Model）工具类
==============================

本文件包含奖励模型训练所需的两个核心组件：

1. RewardDataCollatorWithPadding：偏好数据的 DataCollator
   - 将一个 batch 中的偏好对（response_j 和 response_k）分别 padding 到相同长度
   - 输出格式适配 RewardTrainer 的 compute_loss

2. RewardTrainer：自定义训练器
   - 继承自 HuggingFace Trainer
   - 重写 compute_loss 方法，实现 pairwise ranking loss（配对排序损失）
   - 核心思想：好回答的奖励分数应该高于差回答的奖励分数

损失函数来源：InstructGPT 论文 (https://huggingface.co/papers/2203.02155)
"""

from typing import Any, Optional, Union
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from dataclasses import dataclass, field
from transformers.utils import PaddingStrategy
import torch.nn as nn


# ============================================================
# RewardDataCollatorWithPadding：偏好数据的批处理器
# ============================================================
# 普通的 DataCollator 只需要处理一组 input_ids
# 但奖励模型的数据包含两组：input_ids_j（好回答）和 input_ids_k（差回答）
# 这个 DataCollator 需要分别对两组数据进行 padding，然后合并成一个 batch

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase                     # 分词器（用于 padding）
    padding: Union[bool, str, PaddingStrategy] = True      # padding 策略（True = 动态 padding 到 batch 内最长）
    pad_to_multiple_of: Optional[int] = None               # padding 到指定倍数（用于硬件优化）
    return_tensors: str = "pt"                             # 返回 PyTorch tensor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        将一个 batch 的偏好对数据整理成模型可以接受的格式。

        输入 features 的每个元素包含：
          - input_ids_j, attention_mask_j：好回答的 token 序列
          - input_ids_k, attention_mask_k：差回答的 token 序列

        处理流程：
          1. 把 j 和 k 分开，各自组成一个列表
          2. 分别用 tokenizer.pad() 进行 padding
          3. 合并成一个 dict 返回
        """
        # 分离好回答（j）和差回答（k）
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )

        # 分别对 j 和 k 进行 padding
        # padding=True 表示动态 padding 到当前 batch 内最长的序列长度
        # 这比固定长度 padding 更节省计算资源
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # 合并成一个 batch dict，供 RewardTrainer.compute_loss 使用
        batch = {
            "input_ids_j": batch_j["input_ids"],           # 好回答的 token ids
            "attention_mask_j": batch_j["attention_mask"],  # 好回答的 attention mask
            "input_ids_k": batch_k["input_ids"],            # 差回答的 token ids
            "attention_mask_k": batch_k["attention_mask"],  # 差回答的 attention mask
            "return_loss": True,                            # 告诉 Trainer 需要计算 loss
        }
        return batch


# ============================================================
# RewardTrainer：自定义奖励模型训练器
# ============================================================
# 继承自 HuggingFace Trainer，只重写了 compute_loss 方法
# 使用 InstructGPT 论文中的 pairwise ranking loss（配对排序损失）
#
# 核心思想：
#   对于同一个问题的两个回答（j=好, k=差），
#   奖励模型给好回答的分数应该高于差回答的分数
#
# 损失函数：
#   loss = -log(sigmoid(reward_j - reward_k))
#
#   当 reward_j > reward_k 时（模型判断正确）：
#     sigmoid(正数) → 接近 1 → log(1) → 0 → loss 小 ✅
#
#   当 reward_j < reward_k 时（模型判断错误）：
#     sigmoid(负数) → 接近 0 → log(0) → -∞ → loss 大 ❌
#
#   所以这个损失函数会驱动模型让 reward_j > reward_k

class RewardTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        # 分别计算好回答和差回答的奖励分数
        # model() 返回的 [0] 是 logits，形状为 (batch_size, 1)
        # 因为 num_labels=1，所以输出是一个标量分数
        rewards_j = model(
            input_ids=inputs["input_ids_j"],
            attention_mask=inputs["attention_mask_j"]
        )[0]  # 好回答的奖励分数

        rewards_k = model(
            input_ids=inputs["input_ids_k"],
            attention_mask=inputs["attention_mask_k"]
        )[0]  # 差回答的奖励分数

        # 计算 pairwise ranking loss
        # rewards_j - rewards_k：好回答和差回答的分数差
        # logsigmoid：将分数差映射到 (-∞, 0) 范围
        # 取负号：将最大化问题转为最小化问题
        # .mean()：对 batch 内所有样本取平均
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
