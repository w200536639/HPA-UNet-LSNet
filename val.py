# test_ablation_eval_fixed.py
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nets.unet import HPAUNetLSNet


# =======================

# =======================
SEED = 12
CUDA = True
FP16 = False
DETERMINISTIC = True

NUM_CLASSES = 2
BACKBONE = "lsnet_b"            # lsnet_t / lsnet_s / lsnet_b
PRETRAINED = False
MODEL_PATH = r"logs/lsnet_b_bie0_hpa1/12/best_f1_epoch_199.pth"
INPUT_SHAPE = [640, 640]

VOC_ROOT = r'VOCdevkit'
IMAGE_SET = r'VOC2007/ImageSets/Segmentation/val.txt'

# 与训练代码一致
FG_CLASS_ID = 1

# 评估与可视化
VISUALIZATION_FOLDER = r"results"
METRICS_FILE = "results/metrics.txt"
MASK_ALPHA = 0.12
MATCH_IOU_THR = 0.5

# 二分类前景阈值
PRED_SCORE_THR = 0.5

# DataLoader
BATCH_SIZE = 1
NUM_WORKERS = 0

# ===== 消融开关（仅保留 HPA）=====
USE_HPA = 1

# 可选：自定义运行标签
RUN_TAG = ""


# =======================
# 工具函数
# =======================
def assert_input_shape(shape):
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        raise ValueError("INPUT_SHAPE 必须是长度为2的 [H, W]。")
    if shape[0] % 32 != 0 or shape[1] % 32 != 0:
        raise ValueError("INPUT_SHAPE 的高宽必须为 32 的倍数，例如 [512, 512]、[640, 640]。")


def assert_backbone(backbone):
    valid_backbones = ["lsnet_t", "lsnet_s", "lsnet_b"]
    if backbone.lower() not in valid_backbones:
        raise ValueError(f"BACKBONE 仅支持 {valid_backbones}。")


def load_image_set_robust(path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
    last_error = None

    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                ids = []
                for line in f:
                    line = line.strip().replace('\ufeff', '')
                    if line:
                        ids.append(line)
            print(f"成功读取图像列表文件: {path} | 编码={enc} | 数量={len(ids)}")
            return ids
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(f"无法读取文件 {path}，请检查编码。最后一次错误: {last_error}")


def connected_components_count(binary_mask):
    labels_ = label(binary_mask.astype(np.uint8), connectivity=2)
    return labels_, int(labels_.max())


def calculate_metrics(predicted_bin, ground_truth):
    """
    predicted_bin: 0/1 二值预测图
    ground_truth: 原始标签图，前景定义为 FG_CLASS_ID
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
        g = (gt_labels == gid)
        a = int(g.sum())
        if a > 0:
            gt_masks[gid] = g
            gt_areas[gid] = a

    pred_masks, pred_areas = {}, {}
    for pid in range(1, num_pred_objects + 1):
        p = (pred_labels == pid)
        a = int(p.sum())
        if a > 0:
            pred_masks[pid] = p
            pred_areas[pid] = a

    pairs = []
    for gid, g in gt_masks.items():
        g_area = gt_areas[gid]
        for pid, p in pred_masks.items():
            inter = int(np.logical_and(g, p).sum())
            if inter == 0:
                continue
            union = g_area + pred_areas[pid] - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= MATCH_IOU_THR:
                pairs.append((gid, pid, iou))

    pairs.sort(key=lambda x: x[2], reverse=True)
    matched_g, matched_p = set(), set()
    tp = 0
    for gid, pid, _ in pairs:
        if gid in matched_g or pid in matched_p:
            continue
        matched_g.add(gid)
        matched_p.add(pid)
        tp += 1

    fn = num_gt_objects - tp
    fp = num_pred_objects - tp
    fp = max(fp, 0)

    return dict(
        tp=tp, fp=fp, fn=fn,
        num_gt_objects=num_gt_objects, num_pred_objects=num_pred_objects,
        gt_binary=gt_binary, pred_bin=pred_bin
    )


def calculate_iou(pred_mask, gt_mask):
    inter = int(np.logical_and(pred_mask == 1, gt_mask == 1).sum())
    union = int(np.logical_or(pred_mask == 1, gt_mask == 1).sum())
    return inter, union


def visualize_and_save(original_image, metrics, save_path):
    gt_binary = metrics['gt_binary']
    pred_bin = metrics['pred_bin']

    mask_color = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
    mask_color[pred_bin == 1] = [135, 206, 250]

    edges = sobel(pred_bin) > 0
    edge_color = np.zeros_like(mask_color)
    edge_color[edges] = [255, 0, 0]

    gt_color = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
    gt_color[gt_binary == 1] = [255, 215, 0]

    blended = original_image.astype(np.float32)
    blended[pred_bin == 1] = blended[pred_bin == 1] * (1 - MASK_ALPHA) + mask_color[pred_bin == 1] * MASK_ALPHA
    blended[gt_binary == 1] = blended[gt_binary == 1] * (1 - MASK_ALPHA) + gt_color[gt_binary == 1] * MASK_ALPHA
    blended[edges] = edge_color[edges]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(blended)
    ax.set_title(
        f"GT: {metrics['num_gt_objects']} | Pred: {metrics['num_pred_objects']} | "
        f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}"
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_checkpoint_strict_flexible(model: torch.nn.Module, ckpt_path: str):
    if not (ckpt_path and os.path.exists(ckpt_path)):
        print(f"⚠️ 权重文件不存在：{ckpt_path}，跳过加载。")
        return

    print(f"Load weights {ckpt_path}.")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    model_dict = model.state_dict()
    load_keys, miss_keys = [], []

    for k, v in new_state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            load_keys.append(k)
        else:
            miss_keys.append(k)

    model.load_state_dict(model_dict, strict=False)

    print("\nSuccessful Load Key:", str(load_keys)[:500], "……")
    print("Successful Load Key Num:", len(load_keys))
    if miss_keys:
        print("\nFail To Load Key:", str(miss_keys)[:500], "……")
        print("Fail To Load Key Num:", len(miss_keys))
    print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象；如果 backbone 大量没有载入，就说明权重不匹配。\033[0m")


def tensor_to_vis_image(img_tensor_chw):
    img = img_tensor_chw.transpose(1, 2, 0)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return img


if __name__ == "__main__":
    seed_everything(SEED)
    assert_input_shape(INPUT_SHAPE)
    assert_backbone(BACKBONE)

    if DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True

    device = torch.device('cuda' if (torch.cuda.is_available() and CUDA) else 'cpu')
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

    val_ids = load_image_set_robust(image_set_file)
    val_lines = [x + "\n" for x in val_ids]

    val_dataset = UnetDataset(val_lines, INPUT_SHAPE, NUM_CLASSES, False, VOC_ROOT)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=unet_dataset_collate
    )
    print(f"验证集加载完成，样本数: {len(val_dataset)}")

    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    metrics_dir = os.path.dirname(METRICS_FILE)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    total_tp = total_fp = total_fn = 0
    total_intersection = 0
    total_union = 0
    image_metrics = []
    total_gt_objects = 0

    print(f"开始验证，共 {len(val_loader)} 个 batch（batch_size={BATCH_SIZE}）")
    print(f"FG_CLASS_ID = {FG_CLASS_ID}, PRED_SCORE_THR = {PRED_SCORE_THR}")

    use_amp = FP16 and (device.type == 'cuda')
    sample_index = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, pngs, seg_labels = batch
            images = images.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = net(images)
            else:
                outputs = net(images)

            if NUM_CLASSES == 1:
                fg_probs = torch.sigmoid(outputs)[:, 0]
            else:
                probs = torch.softmax(outputs, dim=1)
                fg_probs = probs[:, FG_CLASS_ID]

            fg_probs_np = fg_probs.detach().cpu().numpy()
            images_np = images.detach().cpu().numpy()
            pngs_np = pngs.detach().cpu().numpy()

            bs = fg_probs_np.shape[0]
            for b in range(bs):
                img_name = val_ids[sample_index]
                prob_map = fg_probs_np[b]
                gt = pngs_np[b].astype(np.uint8)

                pred_bin = (prob_map >= PRED_SCORE_THR).astype(np.uint8)

                m = calculate_metrics(pred_bin, gt)
                total_tp += m['tp']
                total_fp += m['fp']
                total_fn += m['fn']
                total_gt_objects += m['num_gt_objects']

                inter, uni = calculate_iou(m['pred_bin'], m['gt_binary'])
                total_intersection += inter
                total_union += uni

                img_iou = (inter / uni) if uni != 0 else 0.0
                fg_prob_mean = float(prob_map.mean())
                fg_prob_max = float(prob_map.max())

                image_metrics.append({
                    'image_id': sample_index,
                    'image_name': img_name,
                    'num_gt_objects': m['num_gt_objects'],
                    'num_pred_objects': m['num_pred_objects'],
                    'tp': m['tp'],
                    'fp': m['fp'],
                    'fn': m['fn'],
                    'iou': img_iou,
                    'fg_prob_mean': fg_prob_mean,
                    'fg_prob_max': fg_prob_max,
                    'pred_fg_pixels': int(pred_bin.sum()),
                    'gt_fg_pixels': int((gt == FG_CLASS_ID).sum()),
                })

                if sample_index < 5 or (sample_index + 1) % 10 == 0:
                    print(
                        f"图像 {sample_index + 1}/{len(val_dataset)}: "
                        f"名称={img_name}, "
                        f"GT={m['num_gt_objects']}, Pred={m['num_pred_objects']}, "
                        f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, IOU={img_iou:.4f}, "
                        f"prob_mean={fg_prob_mean:.6f}, prob_max={fg_prob_max:.6f}"
                    )

                original = tensor_to_vis_image(images_np[b])
                save_path = os.path.join(VISUALIZATION_FOLDER, f"{img_name}.png")
                visualize_and_save(original, m, save_path)

                sample_index += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    overall_iou = (total_intersection / total_union) if total_union else 0.0
    miou = float(np.mean([m['iou'] for m in image_metrics])) if image_metrics else 0.0
    mean_prob = float(np.mean([m['fg_prob_mean'] for m in image_metrics])) if image_metrics else 0.0
    max_prob_mean = float(np.mean([m['fg_prob_max'] for m in image_metrics])) if image_metrics else 0.0

    print("\n==== 分割指标 ====")
    print(f"真实前景实例总数: {total_gt_objects}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")
    print(f"mIoU: {miou:.4f}")
    print(f"IoU 匹配阈值: {MATCH_IOU_THR}")
    print(f"消融开关: HPA={USE_HPA}")
    print(f"FG_CLASS_ID: {FG_CLASS_ID}")
    print(f"PRED_SCORE_THR: {PRED_SCORE_THR}")
    print(f"平均前景概率均值: {mean_prob:.6f}")
    print(f"平均前景概率最大值: {max_prob_mean:.6f}")

    if max_prob_mean < 0.1:
        print("⚠️ 提示：前景概率整体偏低，可尝试把 PRED_SCORE_THR 从 0.5 调到 0.3 或 0.2。")

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(f"[Run Tag] {run_tag}\n")
        f.write(f"真实前景实例总数: {total_gt_objects}\n")
        f.write(f"TP={total_tp}, FP={total_fp}, FN={total_fn}\n")
        f.write(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}\n")
        f.write(f"Overall_IoU={overall_iou:.4f}\n")
        f.write(f"mIoU={miou:.4f}\n")
        f.write(f"匹配阈值(MATCH_IOU_THR)={MATCH_IOU_THR}\n")
        f.write(f"消融: HPA={USE_HPA}\n")
        f.write(f"FG_CLASS_ID={FG_CLASS_ID}\n")
        f.write(f"PRED_SCORE_THR={PRED_SCORE_THR}\n")
        f.write(f"平均前景概率均值={mean_prob:.6f}\n")
        f.write(f"平均前景概率最大值={max_prob_mean:.6f}\n\n")

        f.write("==== 每张图像的指标 ====\n")
        f.write("图像ID,图像名称,真实目标数,预测目标数,TP,FP,FN,IOU,fg_prob_mean,fg_prob_max,pred_fg_pixels,gt_fg_pixels\n")
        for m in image_metrics:
            f.write(
                f"{m['image_id']},{m['image_name']},{m['num_gt_objects']},{m['num_pred_objects']},"
                f"{m['tp']},{m['fp']},{m['fn']},{m['iou']:.4f},"
                f"{m['fg_prob_mean']:.6f},{m['fg_prob_max']:.6f},"
                f"{m['pred_fg_pixels']},{m['gt_fg_pixels']}\n"
            )

        f.write("\n配置参数:\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"CUDA={CUDA}, FP16={FP16}\n")
        f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
        f.write(f"BACKBONE={BACKBONE}\n")
        f.write(f"MODEL_PATH={MODEL_PATH}\n")
        f.write(f"INPUT_SHAPE={INPUT_SHAPE}\n")
        f.write(f"VOC_ROOT={VOC_ROOT}\n")
        f.write(f"IMAGE_SET={IMAGE_SET}\n")
        f.write(f"VISUALIZATION_FOLDER={VISUALIZATION_FOLDER}\n")

    print(f"\n详细结果已保存至 {METRICS_FILE}")
    print(f"可视化结果已保存至 {VISUALIZATION_FOLDER}")