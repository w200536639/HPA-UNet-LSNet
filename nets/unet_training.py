import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    """
    Compute cross-entropy loss for semantic segmentation.
    计算语义分割任务的交叉熵损失。

    Args:
        inputs (Tensor): Model logits with shape [N, C, H, W].
            模型输出的 logits，形状为 [N, C, H, W]。
        target (Tensor): Ground-truth class index map with shape [N, H, W].
            真实标签类别索引图，形状为 [N, H, W]。
        cls_weights (Tensor): Class weights for loss balancing.
            类别权重，用于类别不平衡处理。
        num_classes (int): Number of valid classes.
            有效类别数，同时 num_classes 也被用作 ignore_index。

    Returns:
        Tensor: Cross-entropy loss value.
            交叉熵损失值。
    """
    batch_size, num_channels, pred_h, pred_w = inputs.size()
    target_batch, target_h, target_w = target.size()

    if pred_h != target_h or pred_w != target_w:
        inputs = F.interpolate(
            inputs,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True
        )

    flattened_inputs = (
        inputs.transpose(1, 2)
        .transpose(2, 3)
        .contiguous()
        .view(-1, num_channels)
    )
    flattened_target = target.view(-1)

    ce_loss = nn.CrossEntropyLoss(
        weight=cls_weights,
        ignore_index=num_classes
    )(flattened_inputs, flattened_target)

    return ce_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    """
    Compute focal loss for semantic segmentation.
    计算语义分割任务的 Focal Loss。

    Args:
        inputs (Tensor): Model logits with shape [N, C, H, W].
            模型输出 logits，形状为 [N, C, H, W]。
        target (Tensor): Ground-truth class index map with shape [N, H, W].
            真实标签类别索引图，形状为 [N, H, W]。
        cls_weights (Tensor): Class weights for balancing.
            类别权重。
        num_classes (int): Number of valid classes / ignore index.
            有效类别数，同时也作为忽略标签索引。
        alpha (float): Weighting coefficient for focal loss.
            Focal Loss 的类别平衡系数。
        gamma (float): Focusing parameter for focal loss.
            Focal Loss 的聚焦参数。

    Returns:
        Tensor: Focal loss value.
            Focal 损失值。
    """
    batch_size, num_channels, pred_h, pred_w = inputs.size()
    target_batch, target_h, target_w = target.size()

    if pred_h != target_h or pred_w != target_w:
        inputs = F.interpolate(
            inputs,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True
        )

    flattened_inputs = (
        inputs.transpose(1, 2)
        .transpose(2, 3)
        .contiguous()
        .view(-1, num_channels)
    )
    flattened_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(
        weight=cls_weights,
        ignore_index=num_classes,
        reduction="none"
    )(flattened_inputs, flattened_target)

    pt = torch.exp(logpt)

    if alpha is not None:
        logpt = logpt * alpha

    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    """
    Compute Dice loss for semantic segmentation.
    计算语义分割任务的 Dice Loss。

    Args:
        inputs (Tensor): Model logits with shape [N, C, H, W].
            模型输出 logits，形状为 [N, C, H, W]。
        target (Tensor): One-hot target with shape [N, H, W, C+1].
            one-hot 标签，形状为 [N, H, W, C+1]。
            最后一个通道通常表示 ignore 区域。
        beta (float): Beta coefficient used in Dice/F-score style formula.
            Dice/F-score 风格公式中的 beta 系数。
        smooth (float): Small value to avoid division by zero.
            防止除零的小常数。

    Returns:
        Tensor: Dice loss value.
            Dice 损失值。
    """
    batch_size, num_channels, pred_h, pred_w = inputs.size()
    target_batch, target_h, target_w, target_channels = target.size()

    if pred_h != target_h or pred_w != target_w:
        inputs = F.interpolate(
            inputs,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True
        )

    softmax_inputs = torch.softmax(
        inputs.transpose(1, 2)
        .transpose(2, 3)
        .contiguous()
        .view(batch_size, -1, num_channels),
        dim=-1
    )
    flattened_target = target.view(batch_size, -1, target_channels)

    # ----------------------------------------------------- #
    # Compute TP / FP / FN on valid classes only.
    # 仅在有效类别通道上计算 TP / FP / FN。
    # The last channel is treated as ignore region.
    # 最后一个通道视为忽略区域。
    # ----------------------------------------------------- #
    true_positive = torch.sum(flattened_target[..., :-1] * softmax_inputs, dim=[0, 1])
    false_positive = torch.sum(softmax_inputs, dim=[0, 1]) - true_positive
    false_negative = torch.sum(flattened_target[..., :-1], dim=[0, 1]) - true_positive

    score = ((1 + beta ** 2) * true_positive + smooth) / (
        (1 + beta ** 2) * true_positive + beta ** 2 * false_negative + false_positive + smooth
    )
    dice_loss = 1 - torch.mean(score)

    return dice_loss


def weights_init(net, init_type="normal", init_gain=0.02):
    """
    Initialize network weights.
    初始化网络权重。

    Args:
        net (nn.Module): Network to be initialized.
            待初始化的网络。
        init_type (str): Initialization type.
            初始化方式，可选 normal / xavier / kaiming / orthogonal。
        init_gain (float): Gain value for initialization.
            初始化增益系数。
    """
    def init_func(module):
        class_name = module.__class__.__name__

        if hasattr(module, "weight") and class_name.find("Conv") != -1:
            if init_type == "normal":
                torch.nn.init.normal_(module.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(module.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"initialization method [{init_type}] is not implemented"
                )

            if getattr(module, "bias", None) is not None:
                torch.nn.init.constant_(module.bias.data, 0.0)

        elif class_name.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(module.bias.data, 0.0)

    print(f"initialize network with {init_type} type")
    net.apply(init_func)


def get_lr_scheduler(
    lr_decay_type,
    lr,
    min_lr,
    total_iters,
    warmup_iters_ratio=0.05,
    warmup_lr_ratio=0.1,
    no_aug_iter_ratio=0.05,
    step_num=10
):
    """
    Build learning rate scheduler function.
    构建学习率调度函数。

    Args:
        lr_decay_type (str): Scheduler type, e.g. "cos" or "step".
            学习率衰减类型，例如 "cos" 或 "step"。
        lr (float): Initial learning rate.
            初始学习率。
        min_lr (float): Minimum learning rate.
            最小学习率。
        total_iters (int): Total training epochs/iterations.
            总训练轮数或总迭代数。
        warmup_iters_ratio (float): Warmup ratio for cosine scheduler.
            cosine 调度中 warmup 占比。
        warmup_lr_ratio (float): Initial warmup LR ratio.
            warmup 初始学习率比例。
        no_aug_iter_ratio (float): Ratio for final no-augmentation stage.
            最后 no-aug 阶段占比。
        step_num (int): Number of steps for step scheduler.
            step 衰减的总阶段数。

    Returns:
        callable: A function that maps current epoch/iter to LR.
            一个根据当前 epoch/iter 返回学习率的函数。
    """
    def yolox_warm_cos_lr(
        base_lr,
        min_lr_value,
        total_steps,
        warmup_total_steps,
        warmup_lr_start,
        no_aug_steps,
        current_step
    ):
        """
        Cosine LR with warmup and final no-augmentation stage.
        带 warmup 和末尾 no-aug 阶段的 cosine 学习率策略。
        """
        if current_step <= warmup_total_steps:
            current_lr = (
                (base_lr - warmup_lr_start)
                * pow(current_step / float(warmup_total_steps), 2)
                + warmup_lr_start
            )
        elif current_step >= total_steps - no_aug_steps:
            current_lr = min_lr_value
        else:
            current_lr = min_lr_value + 0.5 * (base_lr - min_lr_value) * (
                1.0 + math.cos(
                    math.pi * (current_step - warmup_total_steps)
                    / (total_steps - warmup_total_steps - no_aug_steps)
                )
            )
        return current_lr

    def step_lr(base_lr, decay_rate, step_size, current_step):
        """
        Step learning rate decay.
        分段式学习率衰减。
        """
        if step_size < 1:
            raise ValueError("step_size must above 1.")

        decay_times = current_step // step_size
        current_lr = base_lr * decay_rate ** decay_times
        return current_lr

    if lr_decay_type == "cos":
        warmup_total_steps = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_steps = min(max(no_aug_iter_ratio * total_iters, 1), 15)

        scheduler_func = partial(
            yolox_warm_cos_lr,
            lr,
            min_lr,
            total_iters,
            warmup_total_steps,
            warmup_lr_start,
            no_aug_steps
        )
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num

        scheduler_func = partial(
            step_lr,
            lr,
            decay_rate,
            step_size
        )

    return scheduler_func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """
    Update optimizer learning rate according to scheduler.
    根据学习率调度器更新优化器学习率。

    Args:
        optimizer (Optimizer): PyTorch optimizer.
            PyTorch 优化器。
        lr_scheduler_func (callable): Scheduler function returned by get_lr_scheduler.
            由 get_lr_scheduler 返回的调度函数。
        epoch (int): Current epoch index.
            当前 epoch 编号。
    """
    current_lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr