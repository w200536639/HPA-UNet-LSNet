# utils/instance_metrics.py
import numpy as np
from PIL import Image
import os
from skimage.measure import label, regionprops

def binarize_gt(mask_np: np.ndarray, target_label_value: int) -> np.ndarray:
    # 将标签中等于 target_label_value 的像元置为 1，其余为 0
    return (mask_np == target_label_value).astype(np.uint8)

def filter_pred_instances(pred_mask: np.ndarray,
                          area_thr: int = 0,
                          perim_thr: int = 0,
                          circ_thr: float = 0.0) -> np.ndarray:
    """对预测二值掩码做实例过滤，保留满足面积/周长/圆度的连通域"""
    labeled = label(pred_mask, connectivity=2)
    props = regionprops(labeled)
    keep = np.zeros_like(pred_mask, dtype=np.uint8)
    for p in props:
        peri = p.perimeter if p.perimeter > 0 else 0.0
        circ = (4.0 * np.pi * p.area) / (peri ** 2) if peri > 0 else 0.0
        if (p.area >= area_thr) and (peri >= perim_thr) and (circ >= circ_thr):
            keep[labeled == p.label] = 1
    return keep

def tp_fp_fn_from_instances(pred_mask_bin: np.ndarray, gt_mask_bin: np.ndarray):
    """按实例（gt 为基准）统计 TP/FP/FN"""
    gt_labels = label(gt_mask_bin, connectivity=2)
    num_gt = int(np.max(gt_labels))

    pred_labels = label(pred_mask_bin, connectivity=2)
    num_pred = int(np.max(pred_labels))

    gt_detected = [False] * (num_gt + 1)  # 1..num_gt 生效
    for pid in range(1, num_pred + 1):
        pred_obj = (pred_labels == pid)
        # 只要与任一 gt 实例有重叠，就认为该 gt 被检测到
        overlap_gts = np.unique(gt_labels[pred_obj])
        for gid in overlap_gts:
            if gid > 0:
                gt_detected[gid] = True

    tp = sum(gt_detected[1:])
    fn = num_gt - tp
    fp = num_pred - tp
    return tp, fp, fn, num_gt, num_pred

def intersection_union(pred_bin: np.ndarray, gt_bin: np.ndarray):
    inter = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    uni   = int(np.sum((pred_bin == 1) | (gt_bin == 1)))
    return inter, uni
