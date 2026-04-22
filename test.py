# test_hpaunet_lsnet_eval.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.utils import seed_everything
from utils.dataloader import UnetDataset, unet_dataset_collate
from skimage.measure import label
from skimage.filters import sobel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nets.unet import HPAUNetLSNet


# =========================================================
# Configuration / 配置区
# =========================================================
SEED = 12
CUDA = True
FP16 = False
DETERMINISTIC = True

NUM_CLASSES = 2
BACKBONE = "lsnet_b"                  # lsnet_t / lsnet_s / lsnet_b
PRETRAINED = False
MODEL_PATH = r"logs/lsnet_b_bie0_hpa1/12/best_f1_epoch_199.pth"
INPUT_SHAPE = [640, 640]

VOC_ROOT = r"VOCdevkit"
IMAGE_SET = r"VOC2007/ImageSets/Segmentation/test.txt"   # Test set / 测试集

# Foreground class id / 前景类别编号
FG_CLASS_ID = 1

# Evaluation and visualization / 评估与可视化
VISUALIZATION_FOLDER = r"results"
METRICS_FILE = r"results/metrics_test.txt"
MASK_ALPHA = 0.12
MATCH_IOU_THR = 0.5

# Foreground probability threshold / 前景概率阈值
PRED_SCORE_THR = 0.5

# DataLoader / 数据加载
BATCH_SIZE = 1
NUM_WORKERS = 0

# HPA ablation switch / HPA 消融开关
USE_HPA = 1

# Optional run tag / 可选运行标签
RUN_TAG = ""


# =========================================================
# Utility functions / 工具函数
# =========================================================
def assert_input_shape(shape):
    """
    Validate input shape.
    检查输入尺寸是否合法。
    """
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        raise ValueError("INPUT_SHAPE must be [H, W]. / INPUT_SHAPE 必须是长度为2的 [H, W]。")
    if shape[0] % 32 != 0 or shape[1] % 32 != 0:
        raise ValueError("INPUT_SHAPE height and width must be multiples of 32. / INPUT_SHAPE 的高宽必须为 32 的倍数。")


def assert_backbone(backbone):
    """
    Validate backbone name.
    检查主干网络名称是否合法。
    """
    valid_backbones = ["lsnet_t", "lsnet_s", "lsnet_b"]
    if backbone.lower() not in valid_backbones:
        raise ValueError(f"BACKBONE must be one of {valid_backbones}. / BACKBONE 仅支持 {valid_backbones}。")


def load_image_set_robust(path):
    """
    Read image id list with multiple encodings.
    使用多种编码方式稳健读取图像编号列表。
    """
    encodings = ["utf-8", "gbk", "gb2312", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                ids = []
                for line in f:
                    line = line.strip().replace("\ufeff", "")
                    if line:
                        ids.append(line)
            print(f"成功读取图像列表文件: {path} | 编码={enc} | 数量={len(ids)}")
            return ids
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(f"无法读取文件 {path}，请检查编码。最后一次错误: {last_error}")


def connected_components_count(binary_mask):
    """
    Count connected components in a binary mask.
    统计二值掩膜中的连通域个数。
    """
    labeled_mask = label(binary_mask.astype(np.uint8), connectivity=2)
    return labeled_mask, int(labeled_mask.max())


def calculate_metrics(predicted_bin, ground_truth):
    """
    Compute instance-level TP / FP / FN using greedy matching.
    使用贪心匹配计算实例级 TP / FP / FN。

    Args:
        predicted_bin: binary prediction map / 0-1二值预测图
        ground_truth: ground-truth label map / 原始标签图
    """
    gt_binary = (ground_truth == FG_CLASS_ID).astype(np.uint8)
    pred_bin = (predicted_bin > 0).astype(np.uint8)

    gt_labels, num_gt_objects = connected_components_count(gt_binary)
    pred_labels, num_pred_objects = connected_components_count(pred_bin)

    if num_gt_objects == 0 and num_pred_objects == 0:
        return dict(
            tp=0, fp=0, fn=0,
            num_gt_objects=0, num_pred_objects=0,
            gt_binary=gt_binary, pred_bin=pred_bin
        )

    if num_gt_objects == 0 and num_pred_objects > 0:
        return dict(
            tp=0, fp=num_pred_objects, fn=0,
            num_gt_objects=num_gt_objects, num_pred_objects=num_pred_objects,
            gt_binary=gt_binary, pred_bin=pred_bin
        )

    if num_gt_objects > 0 and num_pred_objects == 0:
        return dict(
            tp=0, fp=0, fn=num_gt_objects,
            num_gt_objects=num_gt_objects, num_pred_objects=num_pred_objects,
            gt_binary=gt_binary, pred_bin=pred_bin
        )

    gt_masks, gt_areas = {}, {}
    for gid in range(1, num_gt_objects + 1):
        gt_obj = (gt_labels == gid)
        gt_area = int(gt_obj.sum())
        if gt_area > 0:
            gt_masks[gid] = gt_obj
            gt_areas[gid] = gt_area

    pred_masks, pred_areas = {}, {}
    for pid in range(1, num_pred_objects + 1):
        pred_obj = (pred_labels == pid)
        pred_area = int(pred_obj.sum())
        if pred_area > 0:
            pred_masks[pid] = pred_obj
            pred_areas[pid] = pred_area

    candidate_pairs = []
    for gid, gt_obj in gt_masks.items():
        gt_area = gt_areas[gid]
        for pid, pred_obj in pred_masks.items():
            inter = int(np.logical_and(gt_obj, pred_obj).sum())
            if inter == 0:
                continue

            union = gt_area + pred_areas[pid] - inter
            iou = inter / union if union > 0 else 0.0

            if iou >= MATCH_IOU_THR:
                candidate_pairs.append((gid, pid, iou))

    candidate_pairs.sort(key=lambda x: x[2], reverse=True)

    matched_gt = set()
    matched_pred = set()
    tp = 0

    for gid, pid, _ in candidate_pairs:
        if gid in matched_gt or pid in matched_pred:
            continue
        matched_gt.add(gid)
        matched_pred.add(pid)
        tp += 1

    fn = num_gt_objects - tp
    fp = num_pred_objects - tp
    fp = max(fp, 0)

    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        num_gt_objects=num_gt_objects,
        num_pred_objects=num_pred_objects,
        gt_binary=gt_binary,
        pred_bin=pred_bin
    )


def calculate_iou(pred_mask, gt_mask):
    """
    Compute pixel-level intersection and union.
    计算像素级交集和并集。
    """
    inter = int(np.logical_and(pred_mask == 1, gt_mask == 1).sum())
    union = int(np.logical_or(pred_mask == 1, gt_mask == 1).sum())
    return inter, union


def visualize_and_save(original_image, metrics, save_path):
    """
    Save overlay visualization.
    保存叠加可视化结果图。
    """
    gt_binary = metrics["gt_binary"]
    pred_bin = metrics["pred_bin"]

    pred_color = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
    pred_color[pred_bin == 1] = [135, 206, 250]

    edge_mask = sobel(pred_bin) > 0
    edge_color = np.zeros_like(pred_color)
    edge_color[edge_mask] = [255, 0, 0]

    gt_color = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
    gt_color[gt_binary == 1] = [255, 215, 0]

    blended = original_image.astype(np.float32)
    blended[pred_bin == 1] = blended[pred_bin == 1] * (1 - MASK_ALPHA) + pred_color[pred_bin == 1] * MASK_ALPHA
    blended[gt_binary == 1] = blended[gt_binary == 1] * (1 - MASK_ALPHA) + gt_color[gt_binary == 1] * MASK_ALPHA
    blended[edge_mask] = edge_color[edge_mask]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(blended)
    ax.set_title(
        f"GT: {metrics['num_gt_objects']} | Pred: {metrics['num_pred_objects']} | "
        f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_checkpoint_strict_flexible(model: torch.nn.Module, ckpt_path: str):
    """
    Load model weights with flexible key matching.
    采用灵活键匹配方式加载模型权重。
    """
    if not (ckpt_path and os.path.exists(ckpt_path)):
        print(f"⚠️ 权重文件不存在：{ckpt_path}，跳过加载。")
        return

    print(f"Load weights {ckpt_path}.")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[clean_key] = value

    model_dict = model.state_dict()
    load_keys, miss_keys = [], []

    for key, value in cleaned_state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            model_dict[key] = value
            load_keys.append(key)
        else:
            miss_keys.append(key)

    model.load_state_dict(model_dict, strict=False)

    print("\nSuccessful Load Key:", str(load_keys)[:500], "……")
    print("Successful Load Key Num:", len(load_keys))
    if miss_keys:
        print("\nFail To Load Key:", str(miss_keys)[:500], "……")
        print("Fail To Load Key Num:", len(miss_keys))
    print("\n\033[1;33;44m温馨提示：head部分没有载入通常是正常的；如果 backbone 大量没有载入，则说明权重可能不匹配。\033[0m")


def tensor_to_vis_image(img_tensor_chw):
    """
    Convert CHW tensor image to HWC uint8 image.
    将 CHW 张量图像转换为 HWC 的 uint8 图像。
    """
    img = img_tensor_chw.transpose(1, 2, 0)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return img


# =========================================================
# Main procedure / 主流程
# =========================================================
if __name__ == "__main__":
    seed_everything(SEED)
    assert_input_shape(INPUT_SHAPE)
    assert_backbone(BACKBONE)

    if DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True

    device = torch.device("cuda" if (torch.cuda.is_available() and CUDA) else "cpu")
    print(f"使用设备: {device}")

    auto_suffix = f"{BACKBONE}_hpa{USE_HPA}"
    run_tag = RUN_TAG.strip() or auto_suffix
    print(f"[Run Tag] {run_tag}")

    net = HPAUNetLSNet(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        use_hpa=bool(USE_HPA),
    ).to(device).eval()

    load_checkpoint_strict_flexible(net, MODEL_PATH)

    image_set_file = os.path.join(VOC_ROOT, IMAGE_SET)
    if not os.path.exists(image_set_file):
        raise FileNotFoundError(f"路径不存在: {image_set_file}")

    test_ids = load_image_set_robust(image_set_file)
    test_lines = [image_id + "\n" for image_id in test_ids]

    test_dataset = UnetDataset(test_lines, INPUT_SHAPE, NUM_CLASSES, False, VOC_ROOT)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=unet_dataset_collate
    )
    print(f"测试集加载完成，样本数: {len(test_dataset)}")

    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    metrics_dir = os.path.dirname(METRICS_FILE)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_intersection = 0
    total_union = 0
    total_gt_objects = 0
    image_metrics = []

    print(f"开始测试，共 {len(test_loader)} 个 batch（batch_size={BATCH_SIZE}）")
    print(f"FG_CLASS_ID = {FG_CLASS_ID}, PRED_SCORE_THR = {PRED_SCORE_THR}")

    use_amp = FP16 and (device.type == "cuda")
    sample_index = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, pngs, seg_labels = batch
            images = images.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = net(images)
            else:
                outputs = net(images)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if NUM_CLASSES == 1:
                fg_probs = torch.sigmoid(outputs)[:, 0]
            else:
                probs = torch.softmax(outputs, dim=1)
                fg_probs = probs[:, FG_CLASS_ID]

            fg_probs_np = fg_probs.detach().cpu().numpy()
            images_np = images.detach().cpu().numpy()
            pngs_np = pngs.detach().cpu().numpy()

            batch_size = fg_probs_np.shape[0]
            for batch_idx in range(batch_size):
                img_name = test_ids[sample_index]
                prob_map = fg_probs_np[batch_idx]
                gt = pngs_np[batch_idx].astype(np.uint8)

                pred_bin = (prob_map >= PRED_SCORE_THR).astype(np.uint8)

                metrics = calculate_metrics(pred_bin, gt)
                total_tp += metrics["tp"]
                total_fp += metrics["fp"]
                total_fn += metrics["fn"]
                total_gt_objects += metrics["num_gt_objects"]

                inter, uni = calculate_iou(metrics["pred_bin"], metrics["gt_binary"])
                total_intersection += inter
                total_union += uni

                img_iou = (inter / uni) if uni != 0 else 0.0
                fg_prob_mean = float(prob_map.mean())
                fg_prob_max = float(prob_map.max())

                image_metrics.append({
                    "image_id": sample_index,
                    "image_name": img_name,
                    "num_gt_objects": metrics["num_gt_objects"],
                    "num_pred_objects": metrics["num_pred_objects"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "iou": img_iou,
                    "fg_prob_mean": fg_prob_mean,
                    "fg_prob_max": fg_prob_max,
                    "pred_fg_pixels": int(pred_bin.sum()),
                    "gt_fg_pixels": int((gt == FG_CLASS_ID).sum()),
                })

                if sample_index < 5 or (sample_index + 1) % 10 == 0:
                    print(
                        f"图像 {sample_index + 1}/{len(test_dataset)}: "
                        f"名称={img_name}, "
                        f"GT={metrics['num_gt_objects']}, Pred={metrics['num_pred_objects']}, "
                        f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, "
                        f"IOU={img_iou:.4f}, "
                        f"prob_mean={fg_prob_mean:.6f}, prob_max={fg_prob_max:.6f}"
                    )

                original = tensor_to_vis_image(images_np[batch_idx])
                save_path = os.path.join(VISUALIZATION_FOLDER, f"{img_name}.png")
                visualize_and_save(original, metrics, save_path)

                sample_index += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    overall_iou = (total_intersection / total_union) if total_union else 0.0
    miou = float(np.mean([m["iou"] for m in image_metrics])) if image_metrics else 0.0
    mean_prob = float(np.mean([m["fg_prob_mean"] for m in image_metrics])) if image_metrics else 0.0
    max_prob_mean = float(np.mean([m["fg_prob_max"] for m in image_metrics])) if image_metrics else 0.0

    print("\n==== 测试集分割指标 / Test Metrics ====")
    print(f"真实前景实例总数: {total_gt_objects}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")
    print(f"mIoU: {miou:.4f}")
    print(f"IoU 匹配阈值 / IoU matching threshold: {MATCH_IOU_THR}")
    print(f"HPA 开关 / HPA switch: {USE_HPA}")
    print(f"FG_CLASS_ID: {FG_CLASS_ID}")
    print(f"PRED_SCORE_THR: {PRED_SCORE_THR}")
    print(f"平均前景概率均值: {mean_prob:.6f}")
    print(f"平均前景概率最大值: {max_prob_mean:.6f}")

    if max_prob_mean < 0.1:
        print("⚠️ 提示：前景概率整体偏低，可尝试把 PRED_SCORE_THR 从 0.5 调到 0.3 或 0.2。")

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(f"[Run Tag] {run_tag}\n")
        f.write("Dataset=Test Set / 测试集\n")
        f.write(f"真实前景实例总数={total_gt_objects}\n")
        f.write(f"TP={total_tp}, FP={total_fp}, FN={total_fn}\n")
        f.write(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}\n")
        f.write(f"Overall_IoU={overall_iou:.4f}\n")
        f.write(f"mIoU={miou:.4f}\n")
        f.write(f"MATCH_IOU_THR={MATCH_IOU_THR}\n")
        f.write(f"HPA={USE_HPA}\n")
        f.write(f"FG_CLASS_ID={FG_CLASS_ID}\n")
        f.write(f"PRED_SCORE_THR={PRED_SCORE_THR}\n")
        f.write(f"Mean_FG_Prob={mean_prob:.6f}\n")
        f.write(f"Mean_FG_Prob_Max={max_prob_mean:.6f}\n\n")

        f.write("==== Per-image metrics / 每张图像指标 ====\n")
        f.write("image_id,image_name,num_gt_objects,num_pred_objects,tp,fp,fn,iou,fg_prob_mean,fg_prob_max,pred_fg_pixels,gt_fg_pixels\n")
        for item in image_metrics:
            f.write(
                f"{item['image_id']},{item['image_name']},{item['num_gt_objects']},{item['num_pred_objects']},"
                f"{item['tp']},{item['fp']},{item['fn']},{item['iou']:.4f},"
                f"{item['fg_prob_mean']:.6f},{item['fg_prob_max']:.6f},"
                f"{item['pred_fg_pixels']},{item['gt_fg_pixels']}\n"
            )

        f.write("\n==== Configuration / 配置参数 ====\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"CUDA={CUDA}, FP16={FP16}\n")
        f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
        f.write(f"BACKBONE={BACKBONE}\n")
        f.write(f"MODEL_PATH={MODEL_PATH}\n")
        f.write(f"INPUT_SHAPE={INPUT_SHAPE}\n")
        f.write(f"VOC_ROOT={VOC_ROOT}\n")
        f.write(f"IMAGE_SET={IMAGE_SET}\n")
        f.write(f"VISUALIZATION_FOLDER={VISUALIZATION_FOLDER}\n")

    print(f"\n测试集详细结果已保存至: {METRICS_FILE}")
    print(f"测试集可视化结果已保存至: {VISUALIZATION_FOLDER}")