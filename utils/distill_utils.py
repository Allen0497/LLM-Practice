"""
知识蒸馏核心工具类：DistillConfig 和 DistillTrainer
======================================================

本模块实现了模型蒸馏的两个核心组件：

1. DistillConfig：蒸馏训练的配置类
   继承自 SFTConfig，增加了三个蒸馏专用参数：
   temperature、alpha、max_new_tokens

2. DistillTrainer：蒸馏训练器
   继承自 SFTTrainer，重写了 compute_loss 方法
   核心：在标准 SFT 交叉熵损失之外，增加 KL 散度蒸馏损失

损失函数公式：
  L_total = α × L_SFT + (1-α) × L_KL × T²

  L_SFT：标准的交叉熵损失（让 Student 生成正确文本）
  L_KL：KL 散度损失（让 Student 的 logits 分布接近 Teacher）
  T：温度系数（软化概率分布）
  α：权重系数（控制两种损失的比例）
  T²：温度补偿因子（保持梯度量级不变）
"""

import random
import warnings
from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, PreTrainedModel

from trl import SFTTrainer, SFTConfig
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import empty_cache, DataCollatorForChatML
from trl.models import PreTrainedModelWrapper

from dataclasses import dataclass
from accelerate.utils import is_deepspeed_available

if is_deepspeed_available():
    import deepspeed


@dataclass
class DistillConfig(SFTConfig):
    """
    蒸馏训练配置类，继承自 SFTConfig，增加三个蒸馏专用参数。

    参数说明：
        temperature (float, 默认 0.9)：
            Softmax 温度系数，用于软化 logits 概率分布。
            T > 1：分布更平滑，小概率 token 的信息被放大（"暗知识"更丰富）
            T < 1：分布更尖锐，接近 one-hot（退化为普通 SFT）
            T = 1：原始分布，不做软化
            Hinton 原论文建议 T=4~10；本脚本用 0.9（接近原始分布）

        alpha (float, 默认 1.0)：
            损失权重系数，控制 SFT 损失和蒸馏损失的比例。
            L_total = alpha × L_SFT + (1-alpha) × L_KL
            alpha=1.0 → 纯 SFT（忽略 Teacher logits）
            alpha=0.0 → 纯蒸馏（只对齐 Teacher 分布）
            alpha=0.5 → 两者各半（Hinton 原论文推荐）
            取值范围：[0.0, 1.0]

        max_new_tokens (int, 默认 1024)：
            Teacher 模型生成时允许的最大 token 数。
            数学推理链可能很长，需要设置足够大的值。
    """
    temperature: float = 0.9
    alpha: float = 1
    max_new_tokens: int = 1024

    def __post_init__(self):
        super().__post_init__()
        # 校验 alpha 必须在 [0, 1] 范围内
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("alpha must be in the range [0.0, 1.0].")


class DistillTrainer(SFTTrainer):
    """
    蒸馏训练器，继承自 SFTTrainer。

    与 SFTTrainer 的唯一区别：重写了 compute_loss 方法，
    在标准 SFT 损失之外增加了 KL 散度蒸馏损失。

    训练流程：
      每个 batch：
        1. Student 前向传播 → 得到 student_logits 和 SFT 损失
        2. Teacher 前向传播（no_grad）→ 得到 teacher_logits
        3. 只取回答部分的 logits（跳过 prompt 部分）
        4. 用温度 T 软化两个 logits
        5. 计算 KL 散度：KL(Student || Teacher)
        6. 加权合并：L = α × L_SFT + (1-α) × L_KL × T²
    """

    def __init__(
        self,
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[DistillConfig] = None,
        *sft_args,
        **kwargs,
    ):
        # 强制保留所有数据列（DataCollatorForChatML 需要访问 prompts 列）
        args.remove_unused_columns = False
        # 使用 ChatML 格式的数据整理器，处理 messages 格式的数据集
        kwargs["data_collator"] = DataCollatorForChatML(
            tokenizer=kwargs["tokenizer"], max_length=args.max_seq_length
        )
        super().__init__(*sft_args, args=args, **kwargs)

        # 保存蒸馏超参数
        self.alpha = args.alpha
        self.temperature = args.temperature

        # 准备 Teacher 模型：
        # evaluation_mode=True 确保 Teacher 始终处于 eval 模式（不更新参数）
        if self.is_deepspeed_enabled:
            # DeepSpeed 多卡训练时需要特殊处理 Teacher 模型
            self.teacher_model = self._prepare_deepspeed(teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(
                teacher_model, evaluation_mode=True
            )

        # Teacher 的生成配置（用于 max_new_tokens 等参数的记录，实际蒸馏中不调用 generate）
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_k=0,
            # gradient_checkpointing 开启时禁用 KV Cache（两者不兼容）
            use_cache=False if args.gradient_checkpointing else True,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        蒸馏损失计算的核心方法，重写自 SFTTrainer.compute_loss。

        inputs 字段说明（由 DataCollatorForChatML 生成）：
          input_ids:      完整序列的 token ids（prompt + response）
          attention_mask: 注意力掩码
          labels:         只有 response 部分有标签，prompt 部分为 -100（不计算损失）
          prompts:        只包含 prompt 部分的 token ids（用于确定 prompt 长度）

        损失计算流程：
          ┌─────────────────────────────────────────────────────────┐
          │  输入序列：[prompt tokens] [response tokens]             │
          │                                                          │
          │  Student logits：对整个序列做前向传播                    │
          │  Teacher logits：对整个序列做前向传播（no_grad）          │
          │                                                          │
          │  截取回答部分：只取 response 对应的 logits               │
          │  （从 prompt_length-1 到 -1，因为 LM 是预测下一个 token）│
          │                                                          │
          │  SFT 损失：CrossEntropy(student_logits, labels)          │
          │  KL 损失：KL(softmax(student/T) || softmax(teacher/T))   │
          │                                                          │
          │  总损失：α × SFT + (1-α) × KL × T²                     │
          └─────────────────────────────────────────────────────────┘
        """
        # ── 步骤 1：Student 前向传播 ──
        # 同时计算 SFT 损失（通过 labels 参数）
        outputs_student = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],   # labels 中 prompt 部分为 -100，不参与损失计算
        )
        student_loss = outputs_student.loss  # 标准交叉熵损失

        # ── 步骤 2：Teacher 前向传播（不计算梯度）──
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                # 不传 labels：Teacher 只需要 logits，不需要计算损失
            )

        # ── 步骤 3：截取回答部分的 logits ──
        # prompt_lengths：prompt 部分的 token 数量
        prompt_lengths = inputs["prompts"].shape[1]

        # LM 的 logits 是"预测下一个 token"的分布
        # logits[i] 预测的是 input_ids[i+1]
        # 所以要取 response 部分的 logits，需要从 prompt_length-1 开始
        # 例如：input = [p1, p2, p3, r1, r2, r3]（p=prompt, r=response）
        #        logits = [l1, l2, l3, l4, l5, l6]
        #        l3 预测 r1，l4 预测 r2，l5 预测 r3
        #        所以取 logits[prompt_length-1:-1] = [l3, l4, l5]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1: -1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1: -1, :]

        # ── 步骤 4：计算 KL 散度蒸馏损失 ──
        # KLDivLoss(reduction="batchmean")：对 batch 内所有位置取平均
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                # Student：用 log_softmax（KLDivLoss 要求输入是对数概率）
                F.log_softmax(shifted_student_logits / self.temperature, dim=-1),
                # Teacher：用 softmax（KLDivLoss 要求目标是概率）
                F.softmax(shifted_teacher_logits / self.temperature, dim=-1),
            )
            # T² 补偿因子：
            # 用温度 T 软化后，梯度会缩小 T² 倍
            # 乘以 T² 可以保持梯度量级与原始损失一致（Hinton 2015 论文中的技巧）
            * (self.temperature ** 2)
        )

        # ── 步骤 5：加权合并两种损失 ──
        # alpha=1.0 时：只有 SFT 损失（纯监督学习）
        # alpha=0.0 时：只有 KL 损失（纯蒸馏）
        # alpha=0.5 时：两者各半
        loss = self.alpha * student_loss + (1.0 - self.alpha) * loss_logits

        # 清理显存缓存（蒸馏需要同时保存两个模型的激活值，显存压力大）
        empty_cache()

        return (loss, outputs_student) if return_outputs else loss

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        """
        在 DeepSpeed ZeRO 环境下准备 Teacher 模型。

        DeepSpeed ZeRO-3 会将模型参数分片到多个 GPU 上，
        Teacher 模型需要特殊处理才能在这种环境下正常推理。

        处理逻辑：
          - ZeRO-3：Teacher 也参与分片（与 Student 共享显存）
          - 非 ZeRO-3：Teacher 使用 ZeRO-0（不分片，每个 GPU 保留完整副本）
        """
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # ZeRO-3 的分桶参数，根据 hidden_size 动态计算
                    # 这些参数影响 ZeRO-3 的通信效率
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # 非 ZeRO-3 时，Teacher 使用 stage=0（不分片）
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0

        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
