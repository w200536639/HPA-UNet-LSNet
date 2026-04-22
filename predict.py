import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from nets.unet import HPAUNetLSNet
from utils.utils import cvtColor, preprocess_input, seed_everything


# =========================================================
# Configuration / 配置区
# =========================================================
SEED = 12
CUDA = True
FP16 = False
DETERMINISTIC = True

# Model settings / 模型设置
NUM_CLASSES = 2
BACKBONE = "lsnet_b"               # lsnet_t / lsnet_s / lsnet_b
PRETRAINED = False
USE_HPA = 1
MODEL_PATH = r"logs/lsnet_b_hpa1/12/best_f1_epoch_199.pth"

# Input settings / 输入设置
INPUT_SHAPE = [640, 640]          # [H, W]
IMAGE_DIR = r"predict_images"     # Folder containing unknown images / 待预测图像文件夹

# Output settings / 输出设置
OUTPUT_DIR = r"predict_results"
MASK_DIR = os.path.join(OUTPUT_DIR, "pred_masks")
BINARY_DIR = os.path.join(OUTPUT_DIR, "pred_binary")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "pred_overlay")

# Prediction settings / 预测设置
FG_CLASS_ID = 1
PRED_SCORE_THR = 0.5
MASK_ALPHA = 0.35

# Supported image suffixes / 支持的图像后缀
IMAGE_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


# =========================================================
# Utility functions / 工具函数
# =========================================================
def assert_input_shape(shape):
    """
    Check whether the input shape is valid.
    检查输入尺寸是否合法。
    """
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        raise ValueError("INPUT_SHAPE must be [H, W]. / INPUT_SHAPE 必须是长度为2的 [H, W]。")
    if shape[0] % 32 != 0 or shape[1] % 32 != 0:
        raise ValueError("INPUT_SHAPE height and width must be multiples of 32. / INPUT_SHAPE 的高宽必须为 32 的倍数。")


def assert_backbone(backbone):
    """
    Check whether the backbone name is valid.
    检查主干网络名称是否合法。
    """
    valid_backbones = ["lsnet_t", "lsnet_s", "lsnet_b"]
    if backbone.lower() not in valid_backbones:
        raise ValueError(f"BACKBONE must be one of {valid_backbones}. / BACKBONE 仅支持 {valid_backbones}。")


def load_checkpoint_strict_flexible(model: torch.nn.Module, ckpt_path: str):
    """
    Load weights flexibly while allowing partial key mismatch.
    灵活加载权重，允许部分键不匹配。
    """
    if not (ckpt_path and os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"Weight file does not exist: {ckpt_path} / 权重文件不存在：{ckpt_path}")

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
        new_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    model_dict = model.state_dict()
    loaded_keys, failed_keys = [], []

    for key, value in cleaned_state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            model_dict[key] = value
            loaded_keys.append(key)
        else:
            failed_keys.append(key)

    model.load_state_dict(model_dict, strict=False)

    print("\nSuccessful Load Key Num:", len(loaded_keys))
    print("Fail To Load Key Num:", len(failed_keys))
    if len(failed_keys) > 0:
        print("Part of unmatched keys / 部分未匹配键：")
        print(str(failed_keys[:30]))


def collect_image_paths(image_dir):
    """
    Collect image paths from the input folder.
    收集输入文件夹中的所有图像路径。
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image folder does not exist: {image_dir} / 图像文件夹不存在：{image_dir}")

    image_paths = []
    for name in os.listdir(image_dir):
        lower_name = name.lower()
        if any(lower_name.endswith(suffix) for suffix in IMAGE_SUFFIX):
            image_paths.append(os.path.join(image_dir, name))

    image_paths.sort()
    return image_paths


def resize_image_with_letterbox(image, size):
    """
    Resize image with unchanged aspect ratio using gray padding.
    采用灰边填充的方式缩放图像，同时保持原始长宽比。
    """
    image = cvtColor(image)
    iw, ih = image.size
    h, w = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    resized_image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", (w, h), (128, 128, 128))
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image.paste(resized_image, (dx, dy))

    return new_image, nw, nh, dx, dy


def preprocess_image(image, input_shape):
    """
    Preprocess image for model inference.
    对图像进行网络推理前预处理。
    """
    image, nw, nh, dx, dy = resize_image_with_letterbox(image, input_shape)
    image_array = np.array(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, 0)
    return image_array, nw, nh, dx, dy


def postprocess_prediction(prob_map, original_size, input_shape, nw, nh, dx, dy):
    """
    Remove gray padding and resize prediction back to original image size.
    去除灰边区域，并将预测结果恢复到原图大小。
    """
    input_h, input_w = input_shape
    original_w, original_h = original_size

    prob_map = prob_map[dy:dy + nh, dx:dx + nw]
    prob_map = np.array(
        Image.fromarray((prob_map * 255).astype(np.uint8)).resize(
            (original_w, original_h), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    return prob_map


def build_color_mask(pred_binary):
    """
    Build RGB mask image from binary prediction.
    根据二值预测结果构造彩色掩膜图。
    """
    color_mask = np.zeros((pred_binary.shape[0], pred_binary.shape[1], 3), dtype=np.uint8)
    color_mask[pred_binary == 1] = [255, 0, 0]
    return color_mask


def build_binary_mask(pred_binary):
    """
    Build black-red binary visualization image.
    构造黑底红色的二值可视化图。
    """
    binary_rgb = np.zeros((pred_binary.shape[0], pred_binary.shape[1], 3), dtype=np.uint8)
    binary_rgb[pred_binary == 1] = [255, 0, 0]
    return binary_rgb


def build_overlay_image(original_image, pred_binary, alpha=0.35):
    """
    Overlay the predicted mask on the original RGB image.
    将预测前景掩膜叠加到原始 RGB 图像上。
    """
    original_np = np.array(original_image, dtype=np.uint8)
    overlay = original_np.astype(np.float32).copy()

    mask_color = np.zeros_like(original_np, dtype=np.uint8)
    mask_color[pred_binary == 1] = [255, 0, 0]

    overlay[pred_binary == 1] = (
        overlay[pred_binary == 1] * (1 - alpha) +
        mask_color[pred_binary == 1] * alpha
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def save_prediction_results(image_path, original_image, pred_binary):
    """
    Save mask, binary map and overlay visualization.
    保存彩色掩膜图、二值图和叠加可视化图。
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    color_mask = build_color_mask(pred_binary)
    binary_mask = build_binary_mask(pred_binary)
    overlay_image = build_overlay_image(original_image, pred_binary, alpha=MASK_ALPHA)

    Image.fromarray(color_mask).save(os.path.join(MASK_DIR, f"{base_name}.png"))
    Image.fromarray(binary_mask).save(os.path.join(BINARY_DIR, f"{base_name}.png"))
    Image.fromarray(overlay_image).save(os.path.join(OVERLAY_DIR, f"{base_name}.png"))


# =========================================================
# Main inference function / 主预测函数
# =========================================================
def predict_folder():
    """
    Predict all images in a folder.
    对文件夹中的全部图像进行分割预测。
    """
    seed_everything(SEED)
    assert_input_shape(INPUT_SHAPE)
    assert_backbone(BACKBONE)

    if DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True

    device = torch.device("cuda" if (torch.cuda.is_available() and CUDA) else "cpu")
    print(f"Using device: {device} / 使用设备: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(BINARY_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    image_paths = collect_image_paths(IMAGE_DIR)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in folder: {IMAGE_DIR} / 文件夹中未找到图像：{IMAGE_DIR}")

    print(f"Found {len(image_paths)} images. / 共找到 {len(image_paths)} 张图像。")

    model = HPAUNetLSNet(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        use_hpa=bool(USE_HPA),
    ).to(device).eval()

    load_checkpoint_strict_flexible(model, MODEL_PATH)

    use_amp = FP16 and (device.type == "cuda")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Predicting / 正在预测"):
            try:
                image = Image.open(image_path).convert("RGB")
                original_size = image.size

                input_tensor, nw, nh, dx, dy = preprocess_image(image, INPUT_SHAPE)
                input_tensor = torch.from_numpy(input_tensor).float().to(device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_tensor)
                else:
                    outputs = model(input_tensor)

                if NUM_CLASSES == 1:
                    foreground_prob = torch.sigmoid(outputs)[0, 0].detach().cpu().numpy()
                else:
                    prob = torch.softmax(outputs, dim=1)[0, FG_CLASS_ID].detach().cpu().numpy()
                    foreground_prob = prob

                foreground_prob = postprocess_prediction(
                    foreground_prob,
                    original_size=original_size,
                    input_shape=INPUT_SHAPE,
                    nw=nw,
                    nh=nh,
                    dx=dx,
                    dy=dy,
                )

                pred_binary = (foreground_prob >= PRED_SCORE_THR).astype(np.uint8)
                save_prediction_results(image_path, image, pred_binary)

            except Exception as e:
                print(f"Failed on image: {image_path} / 处理失败: {image_path}")
                print(f"Error: {e}")
                continue

    print("\nPrediction completed. / 预测完成。")
    print(f"Color masks saved to: {MASK_DIR} / 彩色掩膜已保存到: {MASK_DIR}")
    print(f"Binary masks saved to: {BINARY_DIR} / 二值图已保存到: {BINARY_DIR}")
    print(f"Overlay images saved to: {OVERLAY_DIR} / 叠加图已保存到: {OVERLAY_DIR}")


if __name__ == "__main__":
    predict_folder()