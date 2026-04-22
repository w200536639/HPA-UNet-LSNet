from collections import Counter
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import sobel
from skimage.measure import label, regionprops
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.utils import preprocess_input


# =========================================================
# Loss history recorder / 损失历史记录器
# =========================================================
class LossHistory:
    """
    Record training and validation losses, and save loss curves.
    记录训练损失与验证损失，并保存损失曲线。
    """

    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        """
        Args:
            log_dir (str): TensorBoard and log file directory.
                           TensorBoard 与日志文件保存目录。
            model (torch.nn.Module): Training model.
                                     训练模型。
            input_shape (tuple or list): Input size in (H, W).
                                         输入尺寸，格式为 (H, W)。
            val_loss_flag (bool): Whether to record validation loss.
                                  是否记录验证损失。
        """
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag
        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception:
            # Ignore graph export errors to avoid interrupting training.
            # 忽略模型结构图导出错误，避免中断训练。
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        """
        Append one epoch of loss values and update plots.
        追加一个 epoch 的损失值，并更新损失曲线。

        Args:
            epoch (int): Current epoch index.
                         当前 epoch 索引。
            loss (float): Training loss.
                          训练损失。
            val_loss (float, optional): Validation loss.
                                        验证损失。
        """
        os.makedirs(self.log_dir, exist_ok=True)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), "a") as f:
            f.write(str(loss) + "\n")

        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), "a") as f:
                f.write(str(val_loss) + "\n")

        self.writer.add_scalar("loss", loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar("val_loss", val_loss, epoch)

        self.loss_plot()

    def loss_plot(self):
        """
        Plot and save the loss curve.
        绘制并保存损失曲线。
        """
        import scipy.signal as signal

        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, "red", linewidth=2, label="train loss")

        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, "coral", linewidth=2, label="val loss")

        try:
            smooth_window = 5 if len(self.losses) < 25 else 15

            plt.plot(
                iters,
                signal.savgol_filter(self.losses, smooth_window, 3),
                "green",
                linestyle="--",
                linewidth=2,
                label="smooth train loss",
            )

            if self.val_loss_flag:
                plt.plot(
                    iters,
                    signal.savgol_filter(self.val_loss, smooth_window, 3),
                    "#8B4513",
                    linestyle="--",
                    linewidth=2,
                    label="smooth val loss",
                )
        except Exception:
            # Ignore smoothing failure for short sequences.
            # 对于过短序列，忽略平滑失败。
            pass

        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")


# =========================================================
# Mask reading utilities / 掩膜读取工具
# =========================================================
def _read_mask_array(mask_path):
    """
    Read mask as a class-index numpy array, compatible with both P and L modes.
    将掩膜读取为类别索引数组，兼容 P 模式和 L 模式。

    Args:
        mask_path (str): Path to mask image.
                         掩膜图像路径。

    Returns:
        tuple:
            - arr (np.ndarray): Mask array.
                                掩膜数组。
            - mode (str): Original PIL mode.
                          原始 PIL 模式。
    """
    mask = Image.open(mask_path)
    if mask.mode == "P":
        arr = np.array(mask, dtype=np.uint8)
    else:
        arr = np.array(mask.convert("L"), dtype=np.uint8)
    return arr, mask.mode


def _resize_mask_preserve_index(mask_pil, size_wh):
    """
    Resize mask with nearest interpolation while preserving class indices.
    使用最近邻插值缩放掩膜，并保持类别索引不变。

    Args:
        mask_pil (PIL.Image): Mask image.
                              掩膜图像。
        size_wh (tuple): Target size in (W, H).
                         目标尺寸，格式为 (W, H)。

    Returns:
        np.ndarray: Resized mask array.
                    缩放后的掩膜数组。
    """
    if mask_pil.mode == "P":
        return np.array(mask_pil.resize(size_wh, Image.NEAREST), dtype=np.uint8)
    return np.array(mask_pil.convert("L").resize(size_wh, Image.NEAREST), dtype=np.uint8)


# =========================================================
# Instance-level metric utilities / 实例级指标工具
# =========================================================
def _filter_pred_instances(pred_bin, area_thr, per_thr, circ_thr, debug=False):
    """
    Filter predicted connected components by area, perimeter and circularity.
    根据面积、周长和圆形度过滤预测连通域。

    Args:
        pred_bin (np.ndarray): Binary prediction mask.
                               二值预测掩膜。
        area_thr (float): Minimum area threshold.
                          最小面积阈值。
        per_thr (float): Minimum perimeter threshold.
                         最小周长阈值。
        circ_thr (float): Minimum circularity threshold.
                          最小圆形度阈值。
        debug (bool): Whether to print debug information.
                      是否打印调试信息。

    Returns:
        tuple:
            - filtered (np.ndarray): Filtered binary mask.
                                     过滤后的二值掩膜。
            - num_before (int): Number of components before filtering.
                                过滤前连通域数量。
            - num_after (int): Number of components after filtering.
                               过滤后连通域数量。
    """
    labeled_mask = label(pred_bin)
    props = regionprops(labeled_mask)
    filtered = np.zeros_like(pred_bin)

    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        if (area >= area_thr) and (perimeter >= per_thr) and (circularity >= circ_thr):
            filtered[labeled_mask == prop.label] = 1

    num_before = int(labeled_mask.max())
    num_after = int(label(filtered).max())

    if debug:
        print(
            f"    - 预测连通域：过滤前 {num_before}，过滤后 {num_after} "
            f"(area>={area_thr}, per>={per_thr}, circ>={circ_thr})"
        )

    return filtered, num_before, num_after


def _greedy_one_to_one_match_iou(pred_labels, gt_labels, iou_thr=0.5):
    """
    Perform greedy one-to-one matching between predicted and GT instances by IoU.
    基于 IoU 对预测实例与 GT 实例进行贪心式一对一匹配。

    Returns:
        tuple:
            - tp (int): True positives.
                        真阳性数量。
            - fp (int): False positives.
                        假阳性数量。
            - fn (int): False negatives.
                        假阴性数量。
    """
    num_pred = int(pred_labels.max())
    num_gt = int(gt_labels.max())

    if num_pred == 0 and num_gt == 0:
        return 0, 0, 0
    if num_pred == 0:
        return 0, 0, num_gt
    if num_gt == 0:
        return 0, num_pred, 0

    pred_areas = np.bincount(pred_labels.ravel(), minlength=num_pred + 1)[1:]
    gt_areas = np.bincount(gt_labels.ravel(), minlength=num_gt + 1)[1:]

    pred_flat = pred_labels.ravel()
    gt_flat = gt_labels.ravel()
    valid_mask = (pred_flat > 0) & (gt_flat > 0)

    if valid_mask.any():
        pred_ids = pred_flat[valid_mask] - 1
        gt_ids = gt_flat[valid_mask] - 1
        flat_index = pred_ids * num_gt + gt_ids
        inter_matrix = np.bincount(flat_index, minlength=num_pred * num_gt).reshape(num_pred, num_gt)
    else:
        inter_matrix = np.zeros((num_pred, num_gt), dtype=np.int64)

    union_matrix = pred_areas[:, None] + gt_areas[None, :] - inter_matrix

    with np.errstate(divide="ignore", invalid="ignore"):
        iou_matrix = inter_matrix / np.maximum(1, union_matrix)

    candidate_pairs = np.argwhere(iou_matrix >= iou_thr)

    tp = 0
    if candidate_pairs.size > 0:
        sorted_indices = np.argsort(iou_matrix[candidate_pairs[:, 0], candidate_pairs[:, 1]])[::-1]
        used_pred = np.zeros(num_pred, dtype=bool)
        used_gt = np.zeros(num_gt, dtype=bool)

        for idx in sorted_indices:
            pred_idx, gt_idx = candidate_pairs[idx]
            if not used_pred[pred_idx] and not used_gt[gt_idx]:
                used_pred[pred_idx] = True
                used_gt[gt_idx] = True
                tp += 1

    fp = num_pred - tp
    fn = num_gt - tp

    return int(tp), int(max(0, fp)), int(max(0, fn))


def _instance_metrics(
    pred_mask,
    gt_mask,
    target_label=38,
    area_thr=0,
    per_thr=0,
    circ_thr=0.0,
    iou_thr=0.5,
    debug=False,
):
    """
    Compute instance-level TP/FP/FN using IoU-based greedy one-to-one matching.
    使用基于 IoU 的贪心一对一匹配计算实例级 TP/FP/FN。

    Args:
        pred_mask (np.ndarray): Predicted label map.
                                预测标签图。
        gt_mask (np.ndarray): Ground-truth label map.
                              真实标签图。
        target_label (int): Target class label value in GT.
                            GT 中目标类别标签值。
        area_thr (float): Prediction area filter threshold.
                          预测面积过滤阈值。
        per_thr (float): Prediction perimeter filter threshold.
                         预测周长过滤阈值。
        circ_thr (float): Prediction circularity filter threshold.
                          预测圆形度过滤阈值。
        iou_thr (float): IoU matching threshold.
                         IoU 匹配阈值。
        debug (bool): Whether to print debug information.
                      是否打印调试信息。

    Returns:
        tuple:
            - tp (int)
            - fp (int)
            - fn (int)
            - dbg (dict): Debug information dictionary.
                          调试信息字典。
    """
    dbg = {}

    # Ground-truth target mask / 目标类别 GT 二值图
    gt_bin = (gt_mask == target_label).astype(np.uint8)
    dbg["gt_unique"] = np.unique(gt_mask).tolist()
    dbg["gt_target_pixels"] = int(gt_bin.sum())

    if debug:
        print(
            f"    - GT 唯一值: {dbg['gt_unique']}, "
            f"目标像素数(=={target_label}): {dbg['gt_target_pixels']}"
        )

    # Predicted foreground = non-zero / 预测前景定义为非0
    pred_bin = (pred_mask != 0).astype(np.uint8)
    dbg["pred_fore_pixels"] = int(pred_bin.sum())

    if debug:
        print(f"    - 预测前景像素数: {dbg['pred_fore_pixels']}")

    # Filter predicted instances / 过滤预测实例
    filtered_pred, num_before, num_after = _filter_pred_instances(
        pred_bin,
        area_thr,
        per_thr,
        circ_thr,
        debug=debug,
    )
    dbg["pred_cc_before"] = num_before
    dbg["pred_cc_after"] = num_after

    # Fallback if all components are removed after filtering
    # 若过滤后全部被删掉，则自动降级为不过滤
    if (num_after == 0) and (dbg["pred_fore_pixels"] > 0) and (area_thr > 0 or per_thr > 0 or circ_thr > 0):
        if debug:
            print("    🔁 过滤后实例=0，但前景像素>0，自动降级为 '不过滤' 再评估一次。")
        filtered_pred = pred_bin
        num_after = int(label(filtered_pred).max())
        dbg["pred_cc_after"] = num_after

    # Connected component labeling / 连通域标记
    pred_labels = label(filtered_pred, connectivity=2)
    gt_labels = label(gt_bin, connectivity=2)

    num_pred_instances = int(pred_labels.max())
    num_gt_instances = int(gt_labels.max())

    dbg["num_pred_instances"] = num_pred_instances
    dbg["num_gt_instances"] = num_gt_instances

    # Greedy IoU matching / 贪心 IoU 匹配
    tp, fp, fn = _greedy_one_to_one_match_iou(pred_labels, gt_labels, iou_thr=iou_thr)
    dbg["tp_fp_fn"] = (tp, fp, fn)
    dbg["all_bg"] = (dbg["pred_fore_pixels"] == 0)

    if debug:
        print(f"    - IoU阈值: {iou_thr}, 匹配后 TP={tp}, FP={fp}, FN={fn}")

    return tp, fp, fn, dbg


# =========================================================
# Evaluation callback / 验证回调
# =========================================================
class EvalCallback:
    """
    Validation callback based on instance-level F1 score.
    基于实例级 F1 分数的验证回调。
    """

    def __init__(
        self,
        model,
        input_shape,
        num_classes,
        val_lines,
        VOCdevkit_path,
        log_dir,
        Cuda,
        eval_flag=True,
        period=1,
        target_label_value=None,
        area_thr=0,
        perim_thr=0,
        circ_thr=0.0,
        save_visual_topk=0,
        debug_first_n=0,
        autodetect_scan_max=200,
        iou_thr=0.5,
    ):
        """
        Args:
            model: Model to evaluate.
                   待验证模型。
            input_shape (tuple): Input size (H, W).
                                 输入尺寸 (H, W)。
            num_classes (int): Number of classes.
                               类别数。
            val_lines (list): Validation sample ids.
                              验证集样本列表。
            VOCdevkit_path (str): VOC root directory.
                                  VOC 数据根目录。
            log_dir (str): Directory to save logs and best model.
                           日志和最优模型保存目录。
            Cuda (bool): Whether CUDA is enabled.
                         是否启用 CUDA。
            eval_flag (bool): Whether to enable periodic evaluation.
                              是否启用周期性验证。
            period (int): Evaluation interval in epochs.
                          验证周期（按 epoch）。
            target_label_value (int or None): Foreground label value in GT.
                                              GT 中的前景标签值。
            area_thr (float): Minimum prediction area threshold.
                              最小预测面积阈值。
            perim_thr (float): Minimum prediction perimeter threshold.
                               最小预测周长阈值。
            circ_thr (float): Minimum prediction circularity threshold.
                              最小预测圆形度阈值。
            save_visual_topk (int): Save visualization for top-k samples.
                                    保存前 k 个样本可视化。
            debug_first_n (int): Print debug info for first n samples.
                                 对前 n 个样本打印调试信息。
            autodetect_scan_max (int): Max samples used to auto-detect label.
                                       自动识别前景标签时最多扫描样本数。
            iou_thr (float): IoU threshold for instance matching.
                             实例匹配的 IoU 阈值。
        """
        self.model = model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.voc_root = VOCdevkit_path
        self.log_dir = log_dir
        self.cuda = Cuda
        self.eval_flag = eval_flag
        self.period = period

        self.area_thr = area_thr
        self.perim_thr = perim_thr
        self.circ_thr = circ_thr
        self.iou_thr = iou_thr

        self.best_f1 = 0.0
        self.metrics_log = os.path.join(self.log_dir, "best_f1_summary.txt")

        self.save_visual_topk = int(save_visual_topk)
        self.vis_dir = os.path.join(self.log_dir, "vis")
        if self.save_visual_topk > 0:
            os.makedirs(self.vis_dir, exist_ok=True)

        self.debug_first_n = int(debug_first_n)
        self.autodetect_scan_max = int(autodetect_scan_max)

        if target_label_value is None:
            self.target_label_value = self._auto_detect_target_label()
        else:
            self.target_label_value = int(target_label_value)
            print(f"[EvalCallback] 使用指定 target_label_value = {self.target_label_value}")

    def _auto_detect_target_label(self):
        """
        Automatically detect the dominant non-zero foreground label in validation masks.
        自动识别验证集标注中最主要的非零前景标签值。
        """
        sample_ids = [line.strip() for line in self.val_lines]
        mask_dir = os.path.join(self.voc_root, "VOC2007/SegmentationClass")
        counts = Counter()
        scanned = 0

        for name in sample_ids[:self.autodetect_scan_max]:
            mask_path = os.path.join(mask_dir, f"{name}.png")
            if not os.path.exists(mask_path):
                continue

            mask_array, _ = _read_mask_array(mask_path)
            unique_vals, counts_per_val = np.unique(mask_array, return_counts=True)

            for label_value, pixel_count in zip(unique_vals, counts_per_val):
                if label_value != 0:
                    counts[int(label_value)] += int(pixel_count)

            scanned += 1

        if scanned == 0:
            print("[EvalCallback] ⚠️ 自动识别失败：未能扫描到有效标注文件，回退 target_label_value=1")
            return 1

        if not counts:
            print("[EvalCallback] ⚠️ 自动识别：非 0 像素不存在，怀疑你的标注只有背景；回退 target_label_value=1")
            return 1

        target = counts.most_common(1)[0][0]
        print(f"[EvalCallback] 自动识别 target_label_value = {target}（统计自前 {scanned} 张标注）")
        print(f"[EvalCallback] 非 0 像素 Top 值及计数（最多 5 个）：{counts.most_common(5)}")
        return int(target)

    def _letterbox(self, img_pil):
        """
        Resize image with unchanged aspect ratio using gray padding.
        使用灰边填充方式对图像做等比例缩放。

        Returns:
            tuple:
                - canvas (PIL.Image): Letterboxed image.
                                      灰边填充后的图像。
                - (cx, cy, nw, nh): Placement info.
                                    贴图位置信息。
        """
        src_w, src_h = img_pil.size
        dst_w = self.input_shape[1]
        dst_h = self.input_shape[0]

        scale = min(dst_w / src_w, dst_h / src_h)
        new_w, new_h = int(src_w * scale), int(src_h * scale)

        resized = img_pil.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("RGB", (dst_w, dst_h), (128, 128, 128))

        offset_x = (dst_w - new_w) // 2
        offset_y = (dst_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))

        return canvas, (offset_x, offset_y, new_w, new_h)

    def _visualize(self, original_np, pred_bin, gt_bin, save_path, tp, fp, fn):
        """
        Save prediction-vs-GT visualization.
        保存预测结果与 GT 的可视化对比图。
        """
        edges = sobel(pred_bin) > 0

        gt_color = np.zeros((*gt_bin.shape, 3), dtype=np.uint8)
        gt_color[gt_bin == 1] = [255, 215, 0]

        pred_color = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
        pred_color[pred_bin == 1] = [135, 206, 250]

        alpha = 0.2
        blended = original_np.astype(np.float32).copy()
        blended[pred_bin == 1] = blended[pred_bin == 1] * (1 - alpha) + pred_color[pred_bin == 1] * alpha
        blended[gt_bin == 1] = blended[gt_bin == 1] * (1 - alpha) + gt_color[gt_bin == 1] * alpha
        blended[edges] = [255, 0, 0]
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        plt.figure(figsize=(8, 8))
        plt.imshow(blended)
        plt.title(f"TP={tp}, FP={fp}, FN={fn}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def on_epoch_end(self, epoch, model_eval=None):
        """
        Run evaluation at the end of an epoch.
        在每个 epoch 结束时执行验证。
        """
        if (not self.eval_flag) or ((epoch + 1) % self.period != 0):
            return

        print(f"\n[EvalCallback] 🚀 开始验证 Epoch {epoch + 1}（target_label_value={self.target_label_value}）")

        model = self.model if model_eval is None else model_eval
        model.eval()

        sample_ids = [line.strip() for line in self.val_lines]
        img_dir = os.path.join(self.voc_root, "VOC2007/JPEGImages")
        mask_dir = os.path.join(self.voc_root, "VOC2007/SegmentationClass")

        tp_sum = 0
        fp_sum = 0
        fn_sum = 0
        saved_vis = 0

        input_h, input_w = self.input_shape[0], self.input_shape[1]

        with torch.inference_mode():
            for idx, name in enumerate(tqdm(sample_ids, desc="Validating", ncols=100)):
                img_path = os.path.join(img_dir, f"{name}.jpg")
                mask_path = os.path.join(mask_dir, f"{name}.png")

                if (not os.path.exists(img_path)) or (not os.path.exists(mask_path)):
                    continue

                img = Image.open(img_path).convert("RGB")
                mask_pil = Image.open(mask_path)

                canvas, (offset_x, offset_y, new_w, new_h) = self._letterbox(img)

                img_np = np.array(canvas, dtype=np.float32)
                img_np = preprocess_input(img_np)

                x = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).contiguous()
                if self.cuda:
                    x = x.cuda(non_blocking=True)

                out = model(x)
                pred_full = torch.softmax(out, dim=1).argmax(dim=1).cpu().numpy()[0]

                gt_resized = _resize_mask_preserve_index(mask_pil, (new_w, new_h))
                gt_full = np.zeros((input_h, input_w), dtype=np.uint8)
                gt_full[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = gt_resized

                pred = pred_full
                gt = gt_full

                debug_this = (idx < self.debug_first_n)
                if debug_this:
                    pred_crop = pred_full[offset_y:offset_y + new_h, offset_x:offset_x + new_w]
                    print(f"\n[DEBUG] 样本 {idx + 1}: {name}")
                    print(
                        f"    - pred_full_fore = {int((pred_full != 0).sum())}, "
                        f"pred_crop_fore = {int((pred_crop != 0).sum())}, "
                        f"gt_crop_fore = {int((gt_resized == self.target_label_value).sum())}"
                    )

                tp, fp, fn, dbg = _instance_metrics(
                    pred,
                    gt,
                    target_label=self.target_label_value,
                    area_thr=self.area_thr,
                    per_thr=self.perim_thr,
                    circ_thr=self.circ_thr,
                    iou_thr=self.iou_thr,
                    debug=debug_this,
                )

                if debug_this and dbg["all_bg"]:
                    print("    ⚠️ 预测几乎全背景（前景像素数为 0）。")

                tp_sum += tp
                fp_sum += fp
                fn_sum += fn

                if self.save_visual_topk > 0 and saved_vis < self.save_visual_topk:
                    original_np = np.array(canvas)
                    pred_bin = (pred != 0).astype(np.uint8)
                    gt_bin = (gt == self.target_label_value).astype(np.uint8)
                    save_path = os.path.join(self.vis_dir, f"epoch_{epoch + 1:03d}_{name}.png")
                    self._visualize(original_np, pred_bin, gt_bin, save_path, tp, fp, fn)
                    saved_vis += 1

        precision = tp_sum / max(1, tp_sum + fp_sum)
        recall = tp_sum / max(1, tp_sum + fn_sum)
        f1 = (2 * precision * recall) / max(1e-12, (precision + recall))

        print(
            f"📈 Epoch {epoch + 1} → TP={tp_sum}, FP={fp_sum}, FN={fn_sum}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

        with open(self.metrics_log, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch + 1}: TP={tp_sum}, FP={fp_sum}, FN={fn_sum}, "
                f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n"
            )

        if f1 > self.best_f1:
            self.best_f1 = f1
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            save_path = os.path.join(self.log_dir, f"best_f1_epoch_{epoch + 1:03d}.pth")
            torch.save(state_dict, save_path)
            print(f"🎯 发现更优模型 (F1={f1:.4f})，已保存：{save_path}")

        model.train()

    def on_train_end(self):
        """
        Callback at the end of training.
        训练结束时的回调。
        """
        print("✅ 训练完成，EvalCallback 已结束。")