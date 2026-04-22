# test_ablation_eval_fixed.py  (Unet LSNet + HPA ablation + size-stratified eval w/ mIoU per bin)
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from utils.utils import seed_everything
from skimage.measure import label
from skimage.filters import sobel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nets.unet import HPAUNetLSNet

# =======================
# 配置区（含消融开关）
# =======================
SEED = 11
CUDA = True
FP16 = False
DETERMINISTIC = True

NUM_CLASSES = 2
BACKBONE = "lsnet_b"
PRETRAINED = False
MODEL_PATH = r"logs/lsnet_b_bie0_hpa1/12/best_f1_epoch_199.pth"
INPUT_SHAPE = [640, 640]  # [H,W]

VOC_ROOT = r'VOCdevkit'
IMAGE_SET = r'VOC2007/ImageSets/Segmentation/val.txt'

VISUALIZATION_FOLDER = r"results"
METRICS_FILE = "results/metrics.txt"

MASK_ALPHA = 0.12
MATCH_IOU_THR = 0.5
TARGET_LABEL_VALUE = 38

BATCH_SIZE = 1
NUM_WORKERS = 0

USE_HPA = 1
RUN_TAG = ""

# ===== mIoU 空图策略（总体 mIoU 用）=====
# "skip": union=0 的图不计入 mIoU（遥感 patch 常用）
# "one" : union=0 且 pred/gt 都无前景 -> IoU=1，否则0
MIoU_EMPTY_POLICY = "skip"  # "skip" or "one"

# 分层阈值：用 GT 实例面积分位数切分
Q_LOW = 0.33
Q_HIGH = 0.66


# =======================
# 工具函数
# =======================
def letterbox_image(image: Image.Image, size_hw):
    h, w = size_hw
    iw, ih = image.size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new('RGB', (w, h), (128, 128, 128))
    top = (h - nh) // 2
    left = (w - nw) // 2
    canvas.paste(image, (left, top))
    return canvas, nw, nh, top, left


def calculate_iou_binary(pred_mask, gt_mask):
    inter = int(np.logical_and(pred_mask == 1, gt_mask == 1).sum())
    union = int(np.logical_or(pred_mask == 1, gt_mask == 1).sum())
    return inter, union


def iou_for_miou(inter, union, pred_has_fg, gt_has_fg, policy="skip"):
    if union > 0:
        return inter / union
    if policy == "skip":
        return None
    if (not pred_has_fg) and (not gt_has_fg):
        return 1.0
    return 0.0


def get_area_bins_thresholds_from_gt(mask_dir, ids, input_shape_hw, q_low=0.33, q_high=0.66):
    areas = []
    H, W = input_shape_hw
    for name in tqdm(ids, desc="Pre-scan GT areas"):
        p = os.path.join(mask_dir, f"{name}.png")
        if not os.path.exists(p):
            continue
        mask = Image.open(p).convert('L')
        mask = mask.resize((W, H), Image.NEAREST)
        arr = np.array(mask, dtype=np.uint8)

        gt_binary = (arr == TARGET_LABEL_VALUE).astype(np.uint8)
        gt_labels = label(gt_binary, connectivity=2)
        n = int(gt_labels.max())
        for gid in range(1, n + 1):
            a = int((gt_labels == gid).sum())
            if a > 0:
                areas.append(a)

    if len(areas) == 0:
        return (0, 0), dict(count=0, min=0, max=0, mean=0, q33=0, q66=0)

    areas = np.array(areas, dtype=np.int64)
    t1 = int(np.quantile(areas, q_low))
    t2 = int(np.quantile(areas, q_high))
    if t2 < t1:
        t2 = t1

    stats = dict(
        count=int(len(areas)),
        min=int(areas.min()),
        max=int(areas.max()),
        mean=float(areas.mean()),
        q33=float(np.quantile(areas, 0.33)),
        q66=float(np.quantile(areas, 0.66)),
    )
    return (t1, t2), stats


def area_to_bin(area_px, t1, t2):
    if area_px <= t1:
        return "Small"
    elif area_px <= t2:
        return "Medium"
    else:
        return "Large"


def calculate_metrics_with_matching(predicted, ground_truth, t1, t2):
    gt_binary = (ground_truth == TARGET_LABEL_VALUE).astype(np.uint8)
    gt_labels = label(gt_binary, connectivity=2)
    num_gt_objects = int(gt_labels.max())

    # ✅ 二分类：前景严格 pred==1
    pred_bin = (predicted == 1).astype(np.uint8)
    pred_labels = label(pred_bin, connectivity=2)
    num_pred_objects = int(pred_labels.max())

    gt_areas, pred_areas = {}, {}
    gt_bins, pred_bins = {}, {}

    for gid in range(1, num_gt_objects + 1):
        g = (gt_labels == gid)
        a = int(g.sum())
        if a > 0:
            gt_areas[gid] = a
            gt_bins[gid] = area_to_bin(a, t1, t2)

    for pid in range(1, num_pred_objects + 1):
        p = (pred_labels == pid)
        a = int(p.sum())
        if a > 0:
            pred_areas[pid] = a
            pred_bins[pid] = area_to_bin(a, t1, t2)

    if num_gt_objects == 0 and num_pred_objects == 0:
        return dict(tp=0, fp=0, fn=0,
                    num_gt_objects=0, num_pred_objects=0,
                    gt_binary=gt_binary, pred_bin=pred_bin,
                    matched_pairs=[], unmatched_gt=set(), unmatched_pred=set(),
                    gt_bins=gt_bins, pred_bins=pred_bins)

    if num_gt_objects == 0 and num_pred_objects > 0:
        return dict(tp=0, fp=num_pred_objects, fn=0,
                    num_gt_objects=num_gt_objects, num_pred_objects=num_pred_objects,
                    gt_binary=gt_binary, pred_bin=pred_bin,
                    matched_pairs=[], unmatched_gt=set(),
                    unmatched_pred=set(pred_areas.keys()),
                    gt_bins=gt_bins, pred_bins=pred_bins)

    if num_gt_objects > 0 and num_pred_objects == 0:
        return dict(tp=0, fp=0, fn=num_gt_objects,
                    num_gt_objects=num_gt_objects, num_pred_objects=num_pred_objects,
                    gt_binary=gt_binary, pred_bin=pred_bin,
                    matched_pairs=[], unmatched_gt=set(gt_areas.keys()),
                    unmatched_pred=set(),
                    gt_bins=gt_bins, pred_bins=pred_bins)

    gt_masks = {gid: (gt_labels == gid) for gid in gt_areas.keys()}
    pred_masks = {pid: (pred_labels == pid) for pid in pred_areas.keys()}

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
    matched_pairs = []
    for gid, pid, iou in pairs:
        if gid in matched_g or pid in matched_p:
            continue
        matched_g.add(gid)
        matched_p.add(pid)
        matched_pairs.append((gid, pid, iou))

    tp = len(matched_g)
    fn = num_gt_objects - tp
    fp = num_pred_objects - tp
    fp = max(fp, 0)

    unmatched_gt = set(range(1, num_gt_objects + 1)) - matched_g
    unmatched_pred = set(range(1, num_pred_objects + 1)) - matched_p

    return dict(tp=tp, fp=fp, fn=fn,
                num_gt_objects=num_gt_objects, num_pred_objects=num_pred_objects,
                gt_binary=gt_binary, pred_bin=pred_bin,
                matched_pairs=matched_pairs,
                unmatched_gt=unmatched_gt, unmatched_pred=unmatched_pred,
                gt_bins=gt_bins, pred_bins=pred_bins)


def visualize_and_save(original_image, metrics, save_path):
    gt_binary = metrics['gt_binary']
    pred_bin  = metrics['pred_bin']

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


class VOCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, input_shape, image_set_file):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_shape = input_shape
        self.ids = self._load_image_set(image_set_file)
        print(f"加载数据集: {image_set_file}, 样本数量: {len(self.ids)}")
        self._count_total_gt_objects()

    def _load_image_set(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _count_total_gt_objects(self):
        total_gt_objects = 0
        for idx, name in enumerate(self.ids):
            mask_path = os.path.join(self.mask_dir, f'{name}.png')
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((self.input_shape[1], self.input_shape[0]), Image.NEAREST)
            mask = np.array(mask, dtype=np.uint8)
            gt_binary = (mask == TARGET_LABEL_VALUE).astype(np.uint8)
            gt_labels = label(gt_binary, connectivity=2)
            num_gt_objects = int(gt_labels.max())
            total_gt_objects += num_gt_objects
            if idx < 5:
                print(f"图像 {idx + 1}/{len(self.ids)}: 目标实例数 = {num_gt_objects}")
        print(f"所有图像中的真实目标实例总数 = {total_gt_objects}")
        self.total_gt_objects = total_gt_objects

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path  = os.path.join(self.image_dir, f'{name}.jpg')
        mask_path = os.path.join(self.mask_dir,  f'{name}.png')

        image = Image.open(img_path).convert('RGB')
        canvas, nw, nh, top, left = letterbox_image(image, (self.input_shape[0], self.input_shape[1]))
        image = np.array(canvas, dtype=np.float32) / 255.0
        image = image.transpose(2, 0, 1)

        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.input_shape[1], self.input_shape[0]), Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)

        return torch.tensor(image), torch.tensor(mask), nh, nw, top, left


def load_checkpoint_strict_flexible(model: torch.nn.Module, ckpt_path: str, device):
    if not (ckpt_path and os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"权重文件不存在或路径为空：{ckpt_path}")
    print(f"Load weights {ckpt_path}.")
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
    new_state = {}
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
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
    print("Successful Load Key Num:", len(load_keys))
    if miss_keys:
        print("Fail To Load Key Num:", len(miss_keys))
        print("Fail keys (first 50):", miss_keys[:50])
    print("\n提示：head 没载入通常正常，Backbone 大量没载入才需要排查。\n")


# =======================
# 主流程
# =======================
if __name__ == "__main__":
    seed_everything(SEED)
    if DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True

    device = torch.device('cuda' if (torch.cuda.is_available() and CUDA) else 'cpu')
    print(f"使用设备: {device}")

    auto_suffix = f"{BACKBONE}_hpa{USE_HPA}"
    run_tag = RUN_TAG.strip() or auto_suffix
    print(f"[Run Tag] {run_tag}")

    if not (MODEL_PATH and os.path.exists(MODEL_PATH)):
        raise FileNotFoundError(f"MODEL_PATH 不存在: {MODEL_PATH}")

    net = HPAUNetLSNet(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        use_hpa=bool(USE_HPA),
    ).to(device).eval()

    load_checkpoint_strict_flexible(net, MODEL_PATH, device)

    voc2007_dir = os.path.join(VOC_ROOT, 'VOC2007')
    image_dir = os.path.join(voc2007_dir, 'JPEGImages')
    mask_dir  = os.path.join(voc2007_dir, 'SegmentationClass')
    image_set_file = os.path.join(VOC_ROOT, IMAGE_SET)
    for p in [voc2007_dir, image_dir, mask_dir, image_set_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"路径不存在: {p}")

    with open(image_set_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    (t1, t2), area_stats = get_area_bins_thresholds_from_gt(
        mask_dir=mask_dir,
        ids=ids,
        input_shape_hw=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        q_low=Q_LOW,
        q_high=Q_HIGH
    )
    print("\n==== Crown-size thresholds (GT area in pixels) ====")
    print(f"Total GT instances scanned: {area_stats['count']}")
    print(f"Area(px) min={area_stats['min']}, mean={area_stats['mean']:.1f}, max={area_stats['max']}")
    print(f"Thresholds: Small<= {t1} px | Medium<= {t2} px | Large> {t2} px")

    val_dataset = VOCDataset(image_dir, mask_dir, INPUT_SHAPE, image_set_file)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"验证集加载完成，样本数: {len(val_dataset)}")

    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_FILE) or ".", exist_ok=True)

    # ===== 总体实例 =====
    total_tp = total_fp = total_fn = 0

    # ===== 总体像素：Overall IoU（micro）=====
    total_intersection = 0
    total_union = 0

    # ===== 总体像素：mIoU（逐图）=====
    per_image_ious_for_miou = []

    # ===== 分层统计 =====
    bins = ["Small", "Medium", "Large"]

    # 分层实例
    bin_tp = {b: 0 for b in bins}
    bin_fp = {b: 0 for b in bins}
    bin_fn = {b: 0 for b in bins}

    # 分层像素：OverallIoU_bin = sum(inter)/sum(union)
    bin_inter = {b: 0 for b in bins}
    bin_union = {b: 0 for b in bins}

    # ✅ 分层像素：mIoU_bin（逐图平均，规则：该图无该bin的GT则跳过）
    bin_per_image_ious = {b: [] for b in bins}

    image_metrics = []

    use_amp = FP16 and (device.type == 'cuda')
    print(f"开始验证，共 {len(val_loader)} 个 batch（batch_size={BATCH_SIZE}）")

    with torch.no_grad():
        for i, (images, masks, nh, nw, top, left) in enumerate(tqdm(val_loader)):
            images = images.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = net(images)
            else:
                outputs = net(images)

            pred_full = torch.softmax(outputs, dim=1).argmax(dim=1).cpu().numpy()[0]  # 0/1

            nh_val = int(nh[0].item()) if isinstance(nh, torch.Tensor) else int(nh)
            nw_val = int(nw[0].item()) if isinstance(nw, torch.Tensor) else int(nw)
            top_val = int(top[0].item()) if isinstance(top, torch.Tensor) else int(top)
            left_val = int(left[0].item()) if isinstance(left, torch.Tensor) else int(left)

            pred = pred_full[top_val: top_val + nh_val, left_val: left_val + nw_val]
            gt_full = masks.numpy()[0]
            gt = gt_full[top_val: top_val + nh_val, left_val: left_val + nw_val]

            original = images.cpu().numpy()[0].transpose(1, 2, 0)
            original = (original * 255).astype(np.uint8)
            original = original[top_val: top_val + nh_val, left_val: left_val + nw_val]

            m = calculate_metrics_with_matching(pred, gt, t1, t2)

            # ===== 实例总体 =====
            total_tp += m['tp']
            total_fp += m['fp']
            total_fn += m['fn']

            # ===== 像素总体 =====
            inter_all, uni_all = calculate_iou_binary(m['pred_bin'], m['gt_binary'])
            total_intersection += inter_all
            total_union += uni_all

            mi = iou_for_miou(
                inter=inter_all,
                union=uni_all,
                pred_has_fg=bool(m['pred_bin'].sum() > 0),
                gt_has_fg=bool(m['gt_binary'].sum() > 0),
                policy=MIoU_EMPTY_POLICY
            )
            if mi is not None:
                per_image_ious_for_miou.append(float(mi))

            # ===== 分层实例 =====
            for (gid, pid, _) in m['matched_pairs']:
                b = m['gt_bins'].get(gid, None)
                if b is not None:
                    bin_tp[b] += 1
            for gid in m['unmatched_gt']:
                b = m['gt_bins'].get(gid, None)
                if b is not None:
                    bin_fn[b] += 1
            for pid in m['unmatched_pred']:
                b = m['pred_bins'].get(pid, None)
                if b is not None:
                    bin_fp[b] += 1

            # ===== 分层像素：构建每个bin的 GT union mask 与 Pred union mask =====
            gt_labels_full = label((gt == TARGET_LABEL_VALUE).astype(np.uint8), connectivity=2)
            pred_labels_full = label((pred == 1).astype(np.uint8), connectivity=2)

            pred_ids_by_bin = {b: set() for b in bins}
            # matched pred：继承 GT bin
            for (gid, pid, _) in m["matched_pairs"]:
                gb = m["gt_bins"].get(gid, None)
                if gb is not None:
                    pred_ids_by_bin[gb].add(pid)
            # unmatched pred：按 pred 自己面积 bin
            for pid in m["unmatched_pred"]:
                pb = m["pred_bins"].get(pid, None)
                if pb is not None:
                    pred_ids_by_bin[pb].add(pid)

            for b in bins:
                gt_bin_mask = np.zeros_like(gt_labels_full, dtype=np.uint8)
                for gid, bin_name in m["gt_bins"].items():
                    if bin_name == b:
                        gt_bin_mask[gt_labels_full == gid] = 1

                # ✅ 该图该bin无GT -> 跳过（不计入 mIoU_bin）
                if gt_bin_mask.sum() == 0:
                    continue

                pred_bin_mask = np.zeros_like(pred_labels_full, dtype=np.uint8)
                for pid in pred_ids_by_bin[b]:
                    pred_bin_mask[pred_labels_full == pid] = 1

                inter_b, uni_b = calculate_iou_binary(pred_bin_mask, gt_bin_mask)

                # OverallIoU_bin 累积（micro）
                bin_inter[b] += inter_b
                bin_union[b] += uni_b

                # ✅ mIoU_bin：逐图 IoU（注意：这里 union=0 理论上不会发生，因为 gt_bin_mask>0）
                iou_b_img = (inter_b / uni_b) if uni_b > 0 else 0.0
                bin_per_image_ious[b].append(float(iou_b_img))

            overall_iou_img = (inter_all / uni_all) if uni_all != 0 else 0.0

            image_metrics.append({
                'image_id': i,
                'num_gt_objects': m['num_gt_objects'],
                'num_pred_objects': m['num_pred_objects'],
                'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn'],
                'overall_iou_img': float(overall_iou_img),
                'miou_img': float(mi) if mi is not None else -1.0
            })

            if i < 5 or (i + 1) % 10 == 0:
                print(f"图像 {i + 1}/{len(val_loader)}: "
                      f"GT={m['num_gt_objects']}, Pred={m['num_pred_objects']}, "
                      f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, "
                      f"OverallIoU(img)={overall_iou_img:.4f}, "
                      f"mIoU(img)={(mi if mi is not None else 'skip')}")

            save_path = os.path.join(VISUALIZATION_FOLDER, f"{run_tag}_result_{i}.png")
            visualize_and_save(original, m, save_path)

    # ===== 总体汇总 =====
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1_score  = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    overall_iou = (total_intersection / total_union) if total_union else 0.0
    miou = float(np.mean(per_image_ious_for_miou)) if len(per_image_ious_for_miou) else 0.0

    # ===== 分层汇总 =====
    bin_summary = {}
    for b in bins:
        p = bin_tp[b] / (bin_tp[b] + bin_fp[b]) if (bin_tp[b] + bin_fp[b]) else 0.0
        r = bin_tp[b] / (bin_tp[b] + bin_fn[b]) if (bin_tp[b] + bin_fn[b]) else 0.0
        f1b = (2 * p * r / (p + r)) if (p + r) else 0.0

        overall_iou_bin = (bin_inter[b] / bin_union[b]) if bin_union[b] else 0.0
        miou_bin = float(np.mean(bin_per_image_ious[b])) if len(bin_per_image_ious[b]) else 0.0

        bin_summary[b] = dict(
            precision=p, recall=r, f1=f1b,
            overall_iou=overall_iou_bin,
            miou=miou_bin,
            miou_count=len(bin_per_image_ious[b])  # 记录该bin参与mIoU的图数
        )

    print("\n==== Overall Segmentation Metrics ====")
    print(f"真实目标实例总数: {val_dataset.total_gt_objects}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")
    print(f"mIoU: {miou:.4f} (empty_policy={MIoU_EMPTY_POLICY})")
    print(f"IoU 匹配阈值: {MATCH_IOU_THR}")
    print(f"消融开关: HPA={USE_HPA}")

    print("\n==== Size-stratified Metrics (by GT instance area in pixels) ====")
    print(f"Thresholds: Small<= {t1} px | Medium<= {t2} px | Large> {t2} px")
    for b in bins:
        s = bin_summary[b]
        print(f"[{b}] TP={bin_tp[b]}, FP={bin_fp[b]}, FN={bin_fn[b]} | "
              f"P={s['precision']:.4f}, R={s['recall']:.4f}, F1={s['f1']:.4f}, "
              f"OverallIoU(bin)={s['overall_iou']:.4f}, mIoU(bin)={s['miou']:.4f} (n={s['miou_count']})")

    # ===== 保存 =====
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(f"[Run Tag] {run_tag}\n")
        f.write(f"MODEL_PATH={MODEL_PATH}\n")
        f.write(f"MIoU_EMPTY_POLICY={MIoU_EMPTY_POLICY}\n\n")

        f.write("==== Crown-size thresholds (GT area in pixels) ====\n")
        f.write(f"Total GT instances scanned: {area_stats['count']}\n")
        f.write(f"Area(px) min={area_stats['min']}, mean={area_stats['mean']:.1f}, max={area_stats['max']}\n")
        f.write(f"Thresholds: Small<= {t1} px | Medium<= {t2} px | Large> {t2} px\n\n")

        f.write("==== Overall Metrics ====\n")
        f.write(f"真实目标实例总数: {val_dataset.total_gt_objects}\n")
        f.write(f"TP={total_tp}, FP={total_fp}, FN={total_fn}\n")
        f.write(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}\n")
        f.write(f"[Pixel] Overall_IoU={overall_iou:.4f}\n")
        f.write(f"[Pixel] mIoU={miou:.4f}\n")
        f.write(f"匹配阈值(MATCH_IOU_THR)={MATCH_IOU_THR}\n")
        f.write(f"消融: HPA={USE_HPA}\n\n")

        f.write("==== Size-stratified Metrics ====\n")
        f.write("Bin,TP,FP,FN,Precision,Recall,F1,OverallIoU_bin,mIoU_bin,mIoU_bin_n\n")
        for b in bins:
            s = bin_summary[b]
            f.write(f"{b},{bin_tp[b]},{bin_fp[b]},{bin_fn[b]},"
                    f"{s['precision']:.4f},{s['recall']:.4f},{s['f1']:.4f},"
                    f"{s['overall_iou']:.4f},{s['miou']:.4f},{s['miou_count']}\n")
        f.write("\n")

        f.write("==== Per-image Metrics ====\n")
        f.write("image_id,gt_num,pred_num,tp,fp,fn,overall_iou_img,miou_img\n")
        for mm in image_metrics:
            f.write(f"{mm['image_id']},{mm['num_gt_objects']},{mm['num_pred_objects']},"
                    f"{mm['tp']},{mm['fp']},{mm['fn']},"
                    f"{mm['overall_iou_img']:.4f},{mm['miou_img']:.4f}\n")

        f.write("\n配置参数:\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"CUDA={CUDA}, FP16={FP16}, DETERMINISTIC={DETERMINISTIC}\n")
        f.write(f"NUM_CLASSES={NUM_CLASSES}\n")
        f.write(f"BACKBONE={BACKBONE}\n")
        f.write(f"INPUT_SHAPE={INPUT_SHAPE}\n")
        f.write(f"VOC_ROOT={VOC_ROOT}\n")
        f.write(f"IMAGE_SET={IMAGE_SET}\n")
        f.write(f"Q_LOW={Q_LOW}, Q_HIGH={Q_HIGH}\n")

    print(f"\n详细结果已保存至 {METRICS_FILE}")
    print(f"可视化结果已保存至 {VISUALIZATION_FOLDER}")