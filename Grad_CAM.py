import os
import gc
import traceback
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from nets.unet import HPAUNetLSNet


# =========================================================
# Configuration / 配置区
# =========================================================
VOC_ROOT = r"VOCdevkit"
TEST_TXT = r"VOC2007/ImageSets/Segmentation/test.txt"   # Test image list / 测试集图像列表
OUTPUT_DIR = r"HPAUNetLSNet_heatmap"                    # Output directory / 输出目录
MODEL_PATH = r"logs/lsnet_b_bie0_hpa1/12/best_f1_epoch_199.pth"  # Best checkpoint / 最优权重路径

IMG_SIZE = (640, 640)                                  # Inference image size / 推理图像尺寸
DEBUG_MODE = True                                      # Whether to print model structure / 是否打印模型结构
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 2
BACKBONE = "lsnet_b"                                   # lsnet_t / lsnet_s / lsnet_b
PRETRAINED = False
USE_HPA = 1                                            # Whether HPA is enabled / 是否启用 HPA

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# Grad-CAM implementation / Grad-CAM 实现
# =========================================================
class GradCAM:
    """
    Basic Grad-CAM for semantic segmentation models.
    面向语义分割模型的基础 Grad-CAM 实现。
    """

    def __init__(self, model, target_layer, layer_name):
        """
        Args:
            model: segmentation model / 分割模型
            target_layer: layer used for Grad-CAM / 用于 Grad-CAM 的目标层
            layer_name: readable layer name / 可读层名
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.layer_name = layer_name
        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        """
        Save forward activations.
        保存前向传播特征图。
        """
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """
        Save backward gradients.
        保存反向传播梯度。
        """
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        """
        Generate Grad-CAM heatmap for one class.
        为指定类别生成 Grad-CAM 热力图。

        Args:
            input_tensor: input image tensor / 输入图像张量
            class_idx: target class index / 目标类别索引
        """
        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.clone().requires_grad_(True)

        with torch.enable_grad():
            output = self.model(input_tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]

            # For segmentation, sum logits of one class over all pixels
            # 对语义分割任务，对某一类别整幅 logit 图求和
            target = output[:, class_idx, :, :].sum()

            self.model.zero_grad(set_to_none=True)
            target.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(f"[{self.layer_name}] 未捕获到激活值或梯度，请检查目标层。")

        # Global average pooling of gradients
        # 对梯度做全局平均池化
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        # 对特征图做加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=False)
        cam = F.relu(cam)

        cam = cam[0].detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()

        if cam_max - cam_min < 1e-8:
            cam = np.full_like(cam, 0.1, dtype=np.float32)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)
            cam = np.clip(cam, 0.1, 1.0)

        del weights
        gc.collect()
        return cam

    def remove_hooks(self):
        """
        Remove hooks after use.
        使用结束后移除 hook。
        """
        self.forward_hook.remove()
        self.backward_hook.remove()


# =========================================================
# Utility functions / 工具函数
# =========================================================
def load_weights_flexible(model, model_path, device="cpu"):
    """
    Flexibly load model weights.
    灵活加载模型权重，兼容 DataParallel 和 state_dict 格式。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"权重文件不存在: {model_path}")

    print(f"Load weights from: {model_path}")

    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[new_key] = v

    model_dict = model.state_dict()
    load_keys, miss_keys = [], []

    for k, v in cleaned_state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            load_keys.append(k)
        else:
            miss_keys.append(k)

    model.load_state_dict(model_dict, strict=False)

    print("Successful Load Key Num:", len(load_keys))
    print("Fail To Load Key Num:", len(miss_keys))
    if len(miss_keys) > 0:
        print("部分未加载参数（通常 head 不匹配是正常的）:")
        print(str(miss_keys[:20]))


def disable_inplace_relu(module: nn.Module):
    """
    Disable inplace ReLU to avoid Grad-CAM backward errors.
    关闭 inplace ReLU，避免 Grad-CAM 反向传播报错。
    """
    for m in module.modules():
        if isinstance(m, nn.ReLU) and m.inplace:
            m.inplace = False


def load_txt_lines_robust(path):
    """
    Read txt file robustly with multiple encodings.
    使用多种编码方式稳健读取 txt 文件。
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
            print(f"成功读取 test.txt: {path} | 编码={enc} | 数量={len(ids)}")
            return ids
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(f"无法读取文件 {path}，最后一次错误: {last_error}")


def resolve_voc_image_path(voc_root, image_id):
    """
    Search image file in VOC JPEGImages with multiple suffixes.
    在 VOC 的 JPEGImages 目录下搜索图像文件，兼容多种后缀。
    """
    jpeg_dir = os.path.join(voc_root, "VOC2007", "JPEGImages")
    candidate_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    for ext in candidate_exts:
        p = os.path.join(jpeg_dir, image_id + ext)
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"未找到图像: {image_id}，已尝试目录 {jpeg_dir} 下的 {candidate_exts}")


def print_model_structure(model):
    """
    Print model structure for debugging target layer paths.
    打印模型结构，用于调试目标层路径。
    """
    print("\n===== 模型结构 / Model Structure =====")
    print(model)

    print("\n===== 重点模块 / Key Modules =====")
    if hasattr(model, "encoder"):
        print("\n[encoder]")
        print(model.encoder)
    if hasattr(model, "up4"):
        print("\n[up4]")
        print(model.up4)
    if hasattr(model, "up3"):
        print("\n[up3]")
        print(model.up3)
    if hasattr(model, "up2"):
        print("\n[up2]")
        print(model.up2)
    if hasattr(model, "up1"):
        print("\n[up1]")
        print(model.up1)
    if hasattr(model, "out_head"):
        print("\n[out_head]")
        print(model.out_head)
    if hasattr(model, "final"):
        print("\n[final]")
        print(model.final)


def get_layer_by_path(model, path: str):
    """
    Get a submodule by dotted path.
    按点号路径获取子模块。

    Example / 示例:
        out_head.3
        up1.conv.3
    """
    obj = model
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def get_target_layers(model, debug_mode=False):
    """
    Get candidate target layers for Grad-CAM.
    获取 Grad-CAM 目标层候选列表。

    Notes:
        The current network renamed `output_head` to `out_head`,
        and decoder modules are now named `up4 / up3 / up2 / up1`.
        当前网络已将 `output_head` 改名为 `out_head`，
        解码器模块名称为 `up4 / up3 / up2 / up1`。
    """
    if debug_mode:
        print_model_structure(model)

    # Priority: final decoder head first, then decoder stages
    # 优先级：先输出头，再逐级解码器
    candidate_paths = [
        ("out_head_conv2", "out_head.3"),
        ("up1_conv2", "up1.conv.3"),
        ("up2_conv2", "up2.conv.3"),
        ("up3_conv2", "up3.conv.3"),
        ("up4_conv2", "up4.conv.3"),
    ]

    target_layers = []
    for alias, path in candidate_paths:
        try:
            layer = get_layer_by_path(model, path)
            if isinstance(layer, nn.Conv2d):
                target_layers.append((layer, alias, path))
                print(f"✅ 找到层: {alias} -> {path}")
            else:
                print(f"❌ {path} 不是 Conv2d，实际类型: {type(layer)}")
        except Exception as e:
            print(f"❌ 未找到层 {path}: {e}")

    if len(target_layers) == 0:
        raise ValueError("未找到任何可用目标层，请检查网络结构。")

    print(f"\n最终找到 {len(target_layers)} 个目标层：")
    for _, alias, path in target_layers:
        print(f"- {alias}: {path}")

    return target_layers


def visualize_cam(cam, rgb_img, class_idx):
    """
    Overlay CAM heatmap on the original image.
    将 CAM 热力图叠加到原始图像上。
    """
    cam = np.squeeze(cam)
    if cam.ndim != 2:
        cam = np.full((rgb_img.shape[0], rgb_img.shape[1]), 0.1, dtype=np.float32)

    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (rgb_img.shape[1], rgb_img.shape[0]),
            Image.LANCZOS
        )
    ).astype(np.float32) / 255.0

    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay_weight = 0.6 if class_idx == 1 else 0.4
    overlay = (rgb_img * (1 - overlay_weight) + heatmap * overlay_weight).astype(np.uint8)

    del cam_resized, heatmap
    gc.collect()
    return overlay


def safe_layer_name(name: str):
    """
    Convert layer name to filesystem-safe string.
    将层名转换为适合文件保存的安全字符串。
    """
    return name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


# =========================================================
# Main pipeline / 主流程
# =========================================================
def process_voc_test_images(
    voc_root,
    test_txt,
    output_dir,
    model_path,
    img_size=(640, 640),
    debug_mode=False,
    device="cpu",
):
    """
    Run Grad-CAM visualization for all images listed in VOC test.txt.
    对 VOC test.txt 中列出的全部图像运行 Grad-CAM 可视化。
    """
    os.makedirs(output_dir, exist_ok=True)

    heatmaps_root = os.path.join(output_dir, "heatmaps_decoder_layers")
    comparisons_root = os.path.join(output_dir, "comparisons_decoder_layers")
    os.makedirs(heatmaps_root, exist_ok=True)
    os.makedirs(comparisons_root, exist_ok=True)

    model = HPAUNetLSNet(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        use_hpa=bool(USE_HPA),
    )

    load_weights_flexible(model, model_path, device=device)
    model.to(device)
    model.eval()

    disable_inplace_relu(model)
    target_layers = get_target_layers(model, debug_mode=debug_mode)

    layer_save_dirs = {}
    for _, layer_alias, _ in target_layers:
        layer_dir_name = safe_layer_name(layer_alias)

        heatmap_layer_dir = os.path.join(heatmaps_root, layer_dir_name)
        comparison_layer_dir = os.path.join(comparisons_root, layer_dir_name)

        os.makedirs(heatmap_layer_dir, exist_ok=True)
        os.makedirs(comparison_layer_dir, exist_ok=True)

        layer_save_dirs[layer_alias] = {
            "heatmap_dir": heatmap_layer_dir,
            "comparison_dir": comparison_layer_dir,
        }

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    test_txt_path = os.path.join(voc_root, test_txt)
    image_ids = load_txt_lines_robust(test_txt_path)

    print(f"\n开始使用 VOC test.txt 生成热力图，共 {len(image_ids)} 张图像...")

    for img_idx, image_id in enumerate(tqdm(image_ids, desc="处理图像")):
        try:
            img_path = resolve_voc_image_path(voc_root, image_id)
            img_base = image_id
            img_ext = ".png"

            with Image.open(img_path) as img_file:
                img = img_file.convert("RGB")
                rgb_img = np.array(img)

            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]

                prob = F.softmax(output, dim=1)[0]

                print(f"\n[{img_idx + 1}/{len(image_ids)}] {image_id}")
                print(f"图像路径: {img_path}")
                print(f"背景概率均值: {prob[0].mean().item():.6f}, 前景概率均值: {prob[1].mean().item():.6f}")

            del output, prob
            gc.collect()

            for layer, layer_alias, layer_path in target_layers:
                print(f"生成层 {layer_alias} ({layer_path}) 的热力图...")

                heatmap_dir = layer_save_dirs[layer_alias]["heatmap_dir"]
                comparison_dir = layer_save_dirs[layer_alias]["comparison_dir"]

                cam_gen = GradCAM(model, layer, layer_alias)

                cam_bg = cam_gen.generate(input_tensor, class_idx=0)
                vis_bg = visualize_cam(cam_bg, rgb_img, class_idx=0)
                bg_save = os.path.join(heatmap_dir, f"{img_base}_bg{img_ext}")
                Image.fromarray(vis_bg).save(bg_save)

                cam_fg = cam_gen.generate(input_tensor, class_idx=1)
                vis_fg = visualize_cam(cam_fg, rgb_img, class_idx=1)
                fg_save = os.path.join(heatmap_dir, f"{img_base}_fg{img_ext}")
                Image.fromarray(vis_fg).save(fg_save)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                axes[0].imshow(rgb_img)
                axes[0].set_title("原始图像 / Original Image")

                axes[1].imshow(vis_bg)
                axes[1].set_title(f"背景热力图 / Background CAM\n{layer_alias}")

                axes[2].imshow(vis_fg)
                axes[2].set_title(f"前景热力图 / Foreground CAM\n{layer_alias}")

                for ax in axes:
                    ax.axis("off")

                plt.suptitle(
                    f"Layer={layer_alias} | Backbone={BACKBONE} | HPA={USE_HPA}",
                    fontsize=14
                )
                plt.tight_layout(rect=[0, 0, 1, 0.95])

                comp_save = os.path.join(comparison_dir, f"{img_base}_comp{img_ext}")
                plt.savefig(comp_save, dpi=300, bbox_inches="tight")
                plt.close(fig)

                cam_gen.remove_hooks()
                del cam_gen, cam_bg, cam_fg, vis_bg, vis_fg, fig, axes
                gc.collect()

            del input_tensor, rgb_img, img
            gc.collect()

        except Exception as e:
            print(f"处理 {image_id} 出错: {e}")
            traceback.print_exc()
            gc.collect()
            continue

    print("\n所有图像处理完成！")


# =========================================================
# Script entry / 脚本入口
# =========================================================
if __name__ == "__main__":
    process_voc_test_images(
        voc_root=VOC_ROOT,
        test_txt=TEST_TXT,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH,
        img_size=IMG_SIZE,
        debug_mode=DEBUG_MODE,
        device=DEVICE,
    )