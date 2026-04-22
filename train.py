import datetime
import os
from functools import partial

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

from nets.unet import HPAUNetLSNet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch


# =========================================================
# Manual configuration / 手动配置区
# =========================================================
USE_CUDA = True
RANDOM_SEED = 11
USE_DISTRIBUTED = False
USE_SYNC_BN = True
USE_FP16 = False
EARLY_STOP_PATIENCE = None

# Dataset and model / 数据集与模型
NUM_CLASSES = 2
BACKBONE_NAME = "lsnet_b"          # "lsnet_t" / "lsnet_s" / "lsnet_b"
USE_PRETRAINED = False
CHECKPOINT_PATH = ""               # Fill in a checkpoint path for resume training / 若继续训练可填已有权重
INPUT_SIZE = [640, 640]

# Ablation switch / 消融开关
ENABLE_HPA = 0                     # 0: off, 1: on / 0关闭，1开启

# Training strategy / 训练策略
START_EPOCH = 0
FREEZE_END_EPOCH = 50
TOTAL_EPOCHS = 300
ENABLE_FREEZE_TRAIN = True
FREEZE_BATCH_SIZE = 4
UNFREEZE_BATCH_SIZE = 4

# Optimizer and learning rate / 优化器与学习率
INITIAL_LR = 1e-4
MIN_LR_RATIO = 0.01
OPTIMIZER_NAME = "adamw"           # adam / adamw / sgd
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_SCHEDULER_TYPE = "cos"          # cos / step

# Logging and evaluation / 日志与评估
SAVE_PERIOD = 300
LOG_ROOT_DIR = "logs"
ENABLE_EVAL = True
EVAL_PERIOD = 1

# Loss settings / 损失函数设置
USE_DICE_LOSS = True
USE_FOCAL_LOSS = False

# Data loading / 数据读取
NUM_WORKERS = 0
VOC_ROOT = "VOCdevkit"

# Optional run tag / 可选运行标签
RUN_TAG = ""

# Precision-recall evaluation settings / PR评估设置
SAVE_PR_TXT = True
PR_SCORE_THRESHOLDS = [0.5]
INSTANCE_IOU_THRESHOLDS = [0.5]
INSTANCE_MIN_AREA = 5
FOREGROUND_CLASS_ID = 1


# =========================================================
# Utility functions / 工具函数
# =========================================================
def validate_input_size(input_size):
    """
    Validate whether the input size is valid.
    检查输入尺寸是否合法。
    """
    if not (isinstance(input_size, (list, tuple)) and len(input_size) == 2):
        raise ValueError("INPUT_SIZE must be a list/tuple with length 2, e.g. [H, W]. / INPUT_SIZE 必须是长度为2的 [H, W]。")
    if input_size[0] % 32 != 0 or input_size[1] % 32 != 0:
        raise ValueError("The height and width of INPUT_SIZE must be multiples of 32, e.g. [512, 512] or [640, 640]. / INPUT_SIZE 的高宽必须为 32 的倍数。")


def validate_backbone_name(backbone_name):
    """
    Validate backbone name.
    检查主干网络名称是否合法。
    """
    valid_backbones = ["lsnet_t", "lsnet_s", "lsnet_b"]
    if backbone_name.lower() not in valid_backbones:
        raise ValueError(f"BACKBONE_NAME must be one of {valid_backbones}. / BACKBONE_NAME 仅支持 {valid_backbones}。")


def append_row_to_txt(txt_path, row_dict, header=None):
    """
    Append one row to a txt file.
    向文本文件追加一行记录。
    """
    file_exists = os.path.exists(txt_path)
    with open(txt_path, "a", encoding="utf-8") as f:
        if (not file_exists) and (header is not None):
            f.write(header + "\n")
        line = "\t".join([str(row_dict[k]) for k in row_dict.keys()])
        f.write(line + "\n")


def get_connected_components(binary_mask, min_area=1):
    """
    Compute connected components and keep only components larger than min_area.
    计算连通域，并保留面积不小于 min_area 的目标。
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    valid_component_ids = []
    for component_id in range(1, num_labels):
        area = stats[component_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_component_ids.append(component_id)

    return len(valid_component_ids), labels, valid_component_ids


def compute_pixel_level_pr(prob_map, gt_mask, thresholds):
    """
    Compute pixel-level precision and recall under different thresholds.
    计算不同阈值下的像素级 precision 和 recall。
    """
    results = []
    gt_binary = (gt_mask == 1)

    for threshold in thresholds:
        pred_binary = prob_map >= threshold

        tp = np.logical_and(pred_binary, gt_binary).sum()
        fp = np.logical_and(pred_binary, np.logical_not(gt_binary)).sum()
        fn = np.logical_and(np.logical_not(pred_binary), gt_binary).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        results.append({
            "threshold": float(threshold),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
        })
    return results


def compute_instance_level_pr(prob_map, gt_mask, thresholds, iou_threshold=0.5, min_area=1):
    """
    Compute instance-level precision and recall under different score thresholds.
    计算不同分数阈值下的实例级 precision 和 recall。
    """
    results = []

    gt_binary = (gt_mask == 1).astype(np.uint8)
    _, gt_labels, gt_ids = get_connected_components(gt_binary, min_area=min_area)

    for threshold in thresholds:
        pred_binary = (prob_map >= threshold).astype(np.uint8)
        _, pred_labels, pred_ids = get_connected_components(pred_binary, min_area=min_area)

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            results.append({
                "threshold": float(threshold),
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 1.0,
                "recall": 1.0,
            })
            continue

        matched_gt_ids = set()
        matched_pred_ids = set()

        for pred_id in pred_ids:
            pred_object = (pred_labels == pred_id)
            best_gt_id = None
            best_iou = 0.0

            for gt_id in gt_ids:
                if gt_id in matched_gt_ids:
                    continue

                gt_object = (gt_labels == gt_id)
                intersection = np.logical_and(pred_object, gt_object).sum()
                union = np.logical_or(pred_object, gt_object).sum()
                iou = intersection / max(union, 1)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_id = gt_id

            if best_gt_id is not None and best_iou >= iou_threshold:
                matched_pred_ids.add(pred_id)
                matched_gt_ids.add(best_gt_id)

        tp = len(matched_pred_ids)
        fp = len(pred_ids) - tp
        fn = len(gt_ids) - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        results.append({
            "threshold": float(threshold),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
        })

    return results


@torch.no_grad()
def evaluate_pixel_pr_curve(model, dataloader, device, num_classes, thresholds, foreground_class_id=1):
    """
    Evaluate pixel-level PR curve on a dataset.
    在数据集上评估像素级 PR 曲线。
    """
    model.eval()

    accumulators = {
        float(threshold): {"tp": 0, "fp": 0, "fn": 0}
        for threshold in thresholds
    }

    for batch in dataloader:
        images, png_masks, seg_labels = batch
        images = images.to(device)

        outputs = model(images)

        if num_classes == 1:
            probs = torch.sigmoid(outputs)
            foreground_probs = probs[:, 0]
        else:
            probs = torch.softmax(outputs, dim=1)
            foreground_probs = probs[:, foreground_class_id]

        foreground_probs = foreground_probs.detach().cpu().numpy()
        png_masks = png_masks.detach().cpu().numpy()

        for batch_idx in range(foreground_probs.shape[0]):
            prob_map = foreground_probs[batch_idx]
            gt_mask = (png_masks[batch_idx] == foreground_class_id).astype(np.uint8)

            pixel_results = compute_pixel_level_pr(prob_map, gt_mask, thresholds)
            for item in pixel_results:
                threshold = float(item["threshold"])
                accumulators[threshold]["tp"] += item["tp"]
                accumulators[threshold]["fp"] += item["fp"]
                accumulators[threshold]["fn"] += item["fn"]

    pr_curve = []
    for threshold in thresholds:
        threshold = float(threshold)
        tp = accumulators[threshold]["tp"]
        fp = accumulators[threshold]["fp"]
        fn = accumulators[threshold]["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        pr_curve.append({
            "threshold": threshold,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
        })

    model.train()
    return pr_curve


@torch.no_grad()
def evaluate_instance_pr_curve(
    model,
    dataloader,
    device,
    num_classes,
    thresholds,
    foreground_class_id=1,
    instance_iou_threshold=0.5,
    instance_min_area=1,
):
    """
    Evaluate instance-level PR curve on a dataset.
    在数据集上评估实例级 PR 曲线。
    """
    model.eval()

    accumulators = {
        float(threshold): {"tp": 0, "fp": 0, "fn": 0}
        for threshold in thresholds
    }

    for batch in dataloader:
        images, png_masks, seg_labels = batch
        images = images.to(device)

        outputs = model(images)

        if num_classes == 1:
            probs = torch.sigmoid(outputs)
            foreground_probs = probs[:, 0]
        else:
            probs = torch.softmax(outputs, dim=1)
            foreground_probs = probs[:, foreground_class_id]

        foreground_probs = foreground_probs.detach().cpu().numpy()
        png_masks = png_masks.detach().cpu().numpy()

        for batch_idx in range(foreground_probs.shape[0]):
            prob_map = foreground_probs[batch_idx]
            gt_mask = (png_masks[batch_idx] == foreground_class_id).astype(np.uint8)

            instance_results = compute_instance_level_pr(
                prob_map,
                gt_mask,
                thresholds,
                iou_threshold=instance_iou_threshold,
                min_area=instance_min_area
            )

            for item in instance_results:
                threshold = float(item["threshold"])
                accumulators[threshold]["tp"] += item["tp"]
                accumulators[threshold]["fp"] += item["fp"]
                accumulators[threshold]["fn"] += item["fn"]

    pr_curve = []
    for threshold in thresholds:
        threshold = float(threshold)
        tp = accumulators[threshold]["tp"]
        fp = accumulators[threshold]["fp"]
        fn = accumulators[threshold]["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        pr_curve.append({
            "threshold": threshold,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
        })

    model.train()
    return pr_curve


def format_iou_threshold_for_filename(iou_threshold):
    """
    Convert IoU threshold to a filename-friendly string.
    将 IoU 阈值转换为适合文件名的字符串。
    """
    return str(iou_threshold).replace(".", "p")


def compute_class_weights(annotation_lines, mask_dir, sample_cap=800, beta=0.999):
    """
    Automatically compute class weights from masks.
    根据掩膜自动计算类别权重。
    """
    class_counts = np.zeros(2, np.float64)
    num_samples = min(sample_cap, len(annotation_lines))
    sample_names = [line.strip() for line in annotation_lines[:num_samples]]

    for sample_name in sample_names:
        mask_path = os.path.join(mask_dir, f"{sample_name}.png")
        if not os.path.exists(mask_path):
            continue

        mask_array = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        class_counts[0] += (mask_array == 0).sum()
        class_counts[1] += (mask_array == 1).sum()

    effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta + 1e-12)
    class_weights = 1.0 / np.maximum(effective_num, 1e-12)
    class_weights = class_weights / class_weights.sum()
    class_weights = np.clip(class_weights, 0.2, 0.8).astype(np.float32)
    return class_weights


# =========================================================
# Main training procedure / 主训练流程
# =========================================================
if __name__ == "__main__":
    validate_input_size(INPUT_SIZE)
    validate_backbone_name(BACKBONE_NAME)

    auto_run_tag = f"{BACKBONE_NAME}_hpa{ENABLE_HPA}"
    run_tag = RUN_TAG.strip() or auto_run_tag
    save_dir = os.path.join(LOG_ROOT_DIR, run_tag)
    os.makedirs(save_dir, exist_ok=True)

    pixel_pr_txt_path = os.path.join(save_dir, "pixel_pr_curve.txt")
    instance_pr_txt_path_map = {
        float(iou_threshold): os.path.join(
            save_dir,
            f"instance_pr_iou_{format_iou_threshold_for_filename(iou_threshold)}.txt"
        )
        for iou_threshold in INSTANCE_IOU_THRESHOLDS
    }

    seed_everything(RANDOM_SEED)

    num_gpus = torch.cuda.device_count()
    if USE_DISTRIBUTED:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)

        if local_rank == 0:
            print(f"[{os.getpid()}] (rank={global_rank}, local_rank={local_rank}) training...")
            print("Gpu Device Count:", num_gpus)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
        local_rank = 0
        global_rank = 0

    model = HPAUNetLSNet(
        num_classes=NUM_CLASSES,
        pretrained=USE_PRETRAINED,
        backbone=BACKBONE_NAME,
        use_hpa=bool(ENABLE_HPA),
    ).train()

    if not USE_PRETRAINED:
        weights_init(model)

    # -----------------------------------------------------
    # Load checkpoint if provided / 若提供权重则加载
    # -----------------------------------------------------
    if CHECKPOINT_PATH not in ["", None]:
        if local_rank == 0:
            print(f"Load weights {CHECKPOINT_PATH}.")

        model_state_dict = model.state_dict()

        try:
            pretrained_state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        except TypeError:
            pretrained_state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")

        if isinstance(pretrained_state_dict, dict) and "state_dict" in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict["state_dict"]

        loaded_keys, skipped_keys, compatible_state_dict = [], [], {}

        for key, value in pretrained_state_dict.items():
            clean_key = key[7:] if key.startswith("module.") else key
            if clean_key in model_state_dict and np.shape(model_state_dict[clean_key]) == np.shape(value):
                compatible_state_dict[clean_key] = value
                loaded_keys.append(clean_key)
            else:
                skipped_keys.append(key)

        model_state_dict.update(compatible_state_dict)
        model.load_state_dict(model_state_dict, strict=False)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(loaded_keys)[:500], "……\nSuccessful Load Key Num:", len(loaded_keys))
            print("\nFail To Load Key:", str(skipped_keys)[:500], "……\nFail To Load Key Num:", len(skipped_keys))
            print("\n\033[1;33;44mReminder: it is normal if the head is not loaded; if many backbone parameters are not loaded, the checkpoint may be incompatible. / 温馨提示，head部分没有载入是正常现象；如果 backbone 大量没有载入，就说明权重不匹配。\033[0m")

    # -----------------------------------------------------
    # Initialize final bias with foreground prior
    # 使用前景先验初始化输出层偏置
    # -----------------------------------------------------
    try:
        with torch.no_grad():
            foreground_prior = 0.01
            logit_prior = float(np.log(foreground_prior / max(1e-8, 1.0 - foreground_prior)))
            model.final.weight.zero_()
            if model.final.bias is None:
                model.final.bias = torch.nn.Parameter(torch.zeros(NUM_CLASSES))
            model.final.bias.data[:] = 0.0
            if NUM_CLASSES >= 2:
                model.final.bias.data[1] = logit_prior

            if local_rank == 0:
                print(f"[Init head prior] set foreground bias to {logit_prior:.3f} (prior={foreground_prior})")
    except Exception as exc:
        if local_rank == 0:
            print(f"[Init head prior] skip: {exc}")

    # -----------------------------------------------------
    # Logger / 日志器
    # -----------------------------------------------------
    if local_rank == 0:
        time_stamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        loss_log_dir = os.path.join(save_dir, "loss_" + str(time_stamp))
        loss_history = LossHistory(loss_log_dir, model, input_shape=INPUT_SIZE)
    else:
        loss_history = None

    scaler = None
    if USE_FP16:
        try:
            from torch.amp import GradScaler as TorchGradScaler
            scaler = TorchGradScaler()
        except Exception:
            from torch.cuda.amp import GradScaler as TorchGradScaler
            scaler = TorchGradScaler()

    model_train = model.train()

    if USE_SYNC_BN and num_gpus > 1 and USE_DISTRIBUTED:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif USE_SYNC_BN:
        if local_rank == 0:
            print("SyncBN is only valid in distributed multi-GPU training and will be ignored now. / SyncBN 仅在分布式多卡时有效；当前已忽略。")

    if USE_CUDA:
        if USE_DISTRIBUTED:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = False
            cudnn.deterministic = True
            model_train = model_train.cuda()

    # -----------------------------------------------------
    # Load train/val split / 读取训练集与验证集
    # -----------------------------------------------------
    train_txt_path = os.path.join(VOC_ROOT, "VOC2007/ImageSets/Segmentation/train.txt")
    val_txt_path = os.path.join(VOC_ROOT, "VOC2007/ImageSets/Segmentation/val.txt")

    with open(train_txt_path, "r") as f:
        train_lines = f.readlines()
    with open(val_txt_path, "r") as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=NUM_CLASSES,
            backbone=BACKBONE_NAME,
            model_path=CHECKPOINT_PATH,
            input_shape=INPUT_SIZE,
            Init_Epoch=START_EPOCH,
            Freeze_Epoch=FREEZE_END_EPOCH,
            UnFreeze_Epoch=TOTAL_EPOCHS,
            Freeze_batch_size=FREEZE_BATCH_SIZE,
            Unfreeze_batch_size=UNFREEZE_BATCH_SIZE,
            Freeze_Train=ENABLE_FREEZE_TRAIN,
            Init_lr=INITIAL_LR,
            Min_lr=INITIAL_LR * MIN_LR_RATIO,
            optimizer_type=OPTIMIZER_NAME,
            momentum=MOMENTUM,
            lr_decay_type=LR_SCHEDULER_TYPE,
            save_period=SAVE_PERIOD,
            save_dir=save_dir,
            num_workers=NUM_WORKERS,
            num_train=num_train,
            num_val=num_val
        )
        print(f"[Run Tag] {run_tag}")
        print(f"[Ablation] ENABLE_HPA={ENABLE_HPA}")
        print(f"[PR thresholds] {PR_SCORE_THRESHOLDS}")
        print(f"[Instance IoU thresholds] {INSTANCE_IOU_THRESHOLDS}")

    # -----------------------------------------------------
    # Compute class weights / 计算类别权重
    # -----------------------------------------------------
    mask_dir = os.path.join(VOC_ROOT, "VOC2007/SegmentationClass")
    class_weights = compute_class_weights(train_lines, mask_dir)

    if local_rank == 0:
        print(f"[Auto class weights] -> {class_weights}")

    is_unfrozen = False

    if ENABLE_FREEZE_TRAIN:
        model.freeze_backbone()

    batch_size = FREEZE_BATCH_SIZE if ENABLE_FREEZE_TRAIN else UNFREEZE_BATCH_SIZE

    init_lr_fit = INITIAL_LR
    min_lr_fit = INITIAL_LR * MIN_LR_RATIO

    optimizer = {
        "adam": optim.Adam(model.parameters(), init_lr_fit, betas=(MOMENTUM, 0.999), weight_decay=WEIGHT_DECAY),
        "adamw": optim.AdamW(model.parameters(), init_lr_fit, betas=(MOMENTUM, 0.999), weight_decay=WEIGHT_DECAY),
        "sgd": optim.SGD(model.parameters(), init_lr_fit, momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY),
    }[OPTIMIZER_NAME]

    lr_scheduler_func = get_lr_scheduler(LR_SCHEDULER_TYPE, init_lr_fit, min_lr_fit, TOTAL_EPOCHS)

    train_steps_per_epoch = num_train // batch_size
    val_steps_per_epoch = num_val // batch_size
    if train_steps_per_epoch == 0 or val_steps_per_epoch == 0:
        raise ValueError("The dataset is too small for training. / 数据集过小，无法继续进行训练，请扩充数据集。")

    train_dataset = UnetDataset(train_lines, INPUT_SIZE, NUM_CLASSES, True, VOC_ROOT)
    val_dataset = UnetDataset(val_lines, INPUT_SIZE, NUM_CLASSES, False, VOC_ROOT)

    if USE_DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // num_gpus
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=shuffle_train,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=unet_dataset_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=global_rank, seed=RANDOM_SEED)
    )

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=unet_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=global_rank, seed=RANDOM_SEED)
    )

    # -----------------------------------------------------
    # Evaluation callback / 验证回调
    # -----------------------------------------------------
    if local_rank == 0:
        eval_callback = EvalCallback(
            model=model,
            input_shape=INPUT_SIZE,
            num_classes=NUM_CLASSES,
            val_lines=val_lines,
            VOCdevkit_path=VOC_ROOT,
            log_dir=save_dir,
            Cuda=USE_CUDA,
            eval_flag=ENABLE_EVAL,
            period=EVAL_PERIOD,
            target_label_value=None,
            area_thr=0,
            perim_thr=0,
            circ_thr=0.0,
            save_visual_topk=0,
            debug_first_n=5,
            autodetect_scan_max=200,
            iou_thr=0.5
        )
    else:
        eval_callback = None

    best_f1_seen = -1.0
    no_improvement_epochs = 0

    # =====================================================
    # Training loop / 训练主循环
    # =====================================================
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        if epoch >= FREEZE_END_EPOCH and not is_unfrozen and ENABLE_FREEZE_TRAIN:
            batch_size = UNFREEZE_BATCH_SIZE

            init_lr_fit = INITIAL_LR
            min_lr_fit = INITIAL_LR * MIN_LR_RATIO
            lr_scheduler_func = get_lr_scheduler(LR_SCHEDULER_TYPE, init_lr_fit, min_lr_fit, TOTAL_EPOCHS)

            model.unfreeze_backbone()

            train_steps_per_epoch = num_train // batch_size
            val_steps_per_epoch = num_val // batch_size
            if train_steps_per_epoch == 0 or val_steps_per_epoch == 0:
                raise ValueError("The dataset is too small for training. / 数据集过小，无法继续进行训练，请扩充数据集。")

            if USE_DISTRIBUTED:
                batch_size = batch_size // num_gpus

            train_dataloader = DataLoader(
                train_dataset,
                shuffle=shuffle_train,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
                collate_fn=unet_dataset_collate,
                sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=global_rank, seed=RANDOM_SEED)
            )

            val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=False,
                collate_fn=unet_dataset_collate,
                sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=global_rank, seed=RANDOM_SEED)
            )

            is_unfrozen = True

        if USE_DISTRIBUTED and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            train_steps_per_epoch,
            val_steps_per_epoch,
            train_dataloader,
            val_dataloader,
            TOTAL_EPOCHS,
            USE_CUDA,
            USE_DICE_LOSS,
            USE_FOCAL_LOSS,
            class_weights,
            NUM_CLASSES,
            USE_FP16,
            scaler,
            SAVE_PERIOD,
            save_dir,
            local_rank
        )

        if local_rank == 0 and SAVE_PR_TXT:
            pixel_pr_curve = evaluate_pixel_pr_curve(
                model=model,
                dataloader=val_dataloader,
                device=device,
                num_classes=NUM_CLASSES,
                thresholds=PR_SCORE_THRESHOLDS,
                foreground_class_id=FOREGROUND_CLASS_ID
            )

            print(f"\n[Epoch {epoch + 1}] Pixel-level PR:")
            for item in pixel_pr_curve:
                print(
                    f"  thr={item['threshold']:.1f} | "
                    f"P={item['precision']:.6f}, R={item['recall']:.6f}, "
                    f"TP={item['tp']}, FP={item['fp']}, FN={item['fn']}"
                )

            for item in pixel_pr_curve:
                row = {
                    "epoch": epoch + 1,
                    "threshold": item["threshold"],
                    "tp": item["tp"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "precision": item["precision"],
                    "recall": item["recall"],
                }
                append_row_to_txt(
                    pixel_pr_txt_path,
                    row,
                    header="epoch\tthreshold\ttp\tfp\tfn\tprecision\trecall"
                )

            for iou_threshold in INSTANCE_IOU_THRESHOLDS:
                instance_pr_curve = evaluate_instance_pr_curve(
                    model=model,
                    dataloader=val_dataloader,
                    device=device,
                    num_classes=NUM_CLASSES,
                    thresholds=PR_SCORE_THRESHOLDS,
                    foreground_class_id=FOREGROUND_CLASS_ID,
                    instance_iou_threshold=float(iou_threshold),
                    instance_min_area=INSTANCE_MIN_AREA
                )

                print(f"[Epoch {epoch + 1}] Instance-level PR (IoU={iou_threshold}):")
                for item in instance_pr_curve:
                    print(
                        f"  thr={item['threshold']:.1f} | "
                        f"P={item['precision']:.6f}, R={item['recall']:.6f}, "
                        f"TP={item['tp']}, FP={item['fp']}, FN={item['fn']}"
                    )

                txt_path = instance_pr_txt_path_map[float(iou_threshold)]
                for item in instance_pr_curve:
                    row = {
                        "epoch": epoch + 1,
                        "iou_threshold": float(iou_threshold),
                        "score_threshold": item["threshold"],
                        "tp": item["tp"],
                        "fp": item["fp"],
                        "fn": item["fn"],
                        "precision": item["precision"],
                        "recall": item["recall"],
                    }
                    append_row_to_txt(
                        txt_path,
                        row,
                        header="epoch\tiou_threshold\tscore_threshold\ttp\tfp\tfn\tprecision\trecall"
                    )
            print("")

        if local_rank == 0 and loss_history is not None:
            last_train_loss = loss_history.losses[-1] if len(loss_history.losses) > 0 else None
            last_val_loss = (
                loss_history.val_loss[-1]
                if hasattr(loss_history, "val_loss") and len(loss_history.val_loss) > 0
                else None
            )

            if (last_train_loss is not None and (np.isnan(last_train_loss) or np.isinf(last_train_loss))) or \
               (last_val_loss is not None and (np.isnan(last_val_loss) or np.isinf(last_val_loss))):
                print("⚠️ NaN/Inf detected in loss. Training will stop early. / 检测到 Loss 为 NaN/Inf，提前停止训练。")
                break

        if local_rank == 0 and eval_callback is not None and EARLY_STOP_PATIENCE is not None:
            current_best_f1 = getattr(eval_callback, "best_f1", None)
            if current_best_f1 is not None:
                if current_best_f1 > best_f1_seen + 1e-8:
                    best_f1_seen = current_best_f1
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                    if no_improvement_epochs >= EARLY_STOP_PATIENCE:
                        print(f"⏹ Early stopping triggered: F1 has not improved for {EARLY_STOP_PATIENCE} epochs. / 早停触发：F1 在 {EARLY_STOP_PATIENCE} 个 epoch 内未提升。")
                        break

        if USE_DISTRIBUTED:
            dist.barrier()

    if local_rank == 0 and loss_history is not None and hasattr(loss_history, "writer"):
        loss_history.writer.close()