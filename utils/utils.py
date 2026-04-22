import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


#---------------------------------------------------#
#   对输入图像进行resize（保持长宽比，灰边填充）
#---------------------------------------------------#
def resize_image(image: Image.Image, size: Sequence[int]):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w / iw, h / ih)
    nw      = int(iw * scale)
    nh      = int(ih * scale)

    image   = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


#---------------------------------------------------#
#   设置种子（尽量可复现）
#---------------------------------------------------#
def seed_everything(seed: int = 11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#---------------------------------------------------#
#   设置Dataloader的种子（与rank组合，避免各worker相同）
#---------------------------------------------------#
def worker_init_fn(worker_id: int, rank: int, seed: int):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image,
                     mean: Optional[Sequence[float]] = None,
                     std: Optional[Sequence[float]] = None):
    """
    统一的图像预处理函数：
    - 支持 PIL.Image / numpy(HWC) / torch.Tensor(CHW/NCHW/HWC/NHWC)
    - 自动把 0~255 转成 0~1 的 float32
    - 默认不做均值方差（保持你当前训练的行为，即 /255.0）
    - 需要时可传入 ImageNet 统计量：mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]

    返回与输入同“布局”的数据（numpy 保持 HWC，tensor 保持原始维度）
    """
    # ---------- torch.Tensor ----------
    if isinstance(image, torch.Tensor):
        x = image.float()
        # 如果是 0~255，先归一化到 0~1
        if x.max() > 1.0:
            x = x / 255.0

        # 可选：均值方差
        if mean is not None and std is not None:
            if x.dim() == 4:  # NCHW 或 NHWC
                if x.shape[1] == 3:  # NCHW
                    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
                    s = torch.tensor(std,  dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
                    x = (x - m) / s
                elif x.shape[-1] == 3:  # NHWC
                    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 1, 1, 3)
                    s = torch.tensor(std,  dtype=x.dtype, device=x.device).view(1, 1, 1, 3)
                    x = (x - m) / s
            elif x.dim() == 3:  # CHW 或 HWC
                if x.shape[0] == 3:  # CHW
                    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
                    s = torch.tensor(std,  dtype=x.dtype, device=x.device).view(3, 1, 1)
                    x = (x - m) / s
                elif x.shape[-1] == 3:  # HWC
                    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 1, 3)
                    s = torch.tensor(std,  dtype=x.dtype, device=x.device).view(1, 1, 3)
                    x = (x - m) / s
        return x

    # ---------- PIL.Image / numpy ----------
    # 统一成 numpy.float32, HWC
    if isinstance(image, Image.Image):
        x = np.asarray(image, dtype=np.float32)
    else:
        x = np.array(image, dtype=np.float32, copy=False)

    # 如果是 0~255，先归一化到 0~1
    if x.max() > 1.0:
        x = x / 255.0

    # 可选：均值方差（numpy 走 HWC）
    if mean is not None and std is not None and (x.ndim == 3 and x.shape[-1] == 3):
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std

    return x


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


#---------------------------------------------------#
#   模型权重下载到项目根目录的 ./model_data
#   - 支持 vgg / resnet50 / mobilenetv3 / mobilenet
#   - 已存在则跳过
#---------------------------------------------------#
def download_weights(backbone: str, model_dir: str = "./model_data") -> Optional[str]:
    """
    下载指定 backbone 的预训练权重到项目根目录下的 ./model_data。
    参数:
        backbone: 'vgg' | 'resnet50' | 'mobilenetv3' | 'mobilenet'
        model_dir: 目标相对目录（会被解析为相对项目根目录）

    返回:
        权重文件的本地绝对路径（存在/下载成功时），否则 None
    """
    backbone = str(backbone).lower()

    download_urls = {
        'vgg'        : 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50'   : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
        # 两个键都映射到 MobileNetV3 Large
        'mobilenetv3': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
        'mobilenet'  : 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
    }

    if backbone not in download_urls:
        print(f"[download_weights] 未配置 {backbone} 的下载URL，跳过。")
        return None

    # 将相对路径锚定到项目根目录（utils/ 的上一级）
    # utils/utils.py -> 上一级即项目根
    base_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if os.path.isabs(model_dir):
        target_dir = model_dir
    else:
        target_dir = os.path.abspath(os.path.join(base_root, model_dir))

    os.makedirs(target_dir, exist_ok=True)

    url = download_urls[backbone]
    filename = os.path.basename(url)
    dst_path = os.path.join(target_dir, filename)

    if os.path.exists(dst_path):
        print(f"[download_weights] 已存在：{dst_path}（跳过下载）")
        return dst_path

    try:
        print(f"[download_weights] 正在下载 {backbone} 权重：{url}")
        print(f"[download_weights] 目标路径：{dst_path}")
        download_url_to_file(url, dst_path, progress=True)
        print(f"[download_weights] 已保存到：{dst_path}")
        return dst_path
    except Exception as e:
        print(f"[download_weights] 下载失败：{e}")
        return None
