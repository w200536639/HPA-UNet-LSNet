import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    """
    Segmentation dataset for VOC-style directory structure.
    用于 VOC 风格目录结构的语义分割数据集。

    Expected folder structure / 期望目录结构:
        dataset_path/
            VOC2007/
                JPEGImages/
                SegmentationClass/
    """

    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        """
        Args:
            annotation_lines (list[str]): Sample id list, one id per line.
                样本编号列表，每行一个样本名。
            input_shape (list[int] | tuple[int]): Target input size [H, W].
                目标输入尺寸 [高, 宽]。
            num_classes (int): Number of valid semantic classes.
                有效类别数。
            train (bool): Whether to enable random augmentation.
                是否启用训练时随机增强。
            dataset_path (str): Root path of VOC dataset.
                VOC 数据集根目录。
        """
        super().__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

        self.image_dir = os.path.join(self.dataset_path, "VOC2007", "JPEGImages")
        self.mask_dir = os.path.join(self.dataset_path, "VOC2007", "SegmentationClass")

    def __len__(self):
        """
        Return dataset length.
        返回数据集长度。
        """
        return self.length

    def __getitem__(self, index):
        """
        Read one sample and apply preprocessing / augmentation.
        读取单个样本并执行预处理 / 数据增强。
        """
        annotation_line = self.annotation_lines[index]
        image_id = annotation_line.split()[0]

        # ------------------------------------------------- #
        # Load image and mask from disk.
        # 从磁盘读取原始图像与标签。
        # ------------------------------------------------- #
        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        mask_path = os.path.join(self.mask_dir, image_id + ".png")

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # ------------------------------------------------- #
        # Apply augmentation or deterministic resize.
        # 执行数据增强或确定性缩放。
        # ------------------------------------------------- #
        image, mask = self.get_random_data(
            image=image,
            label=mask,
            input_shape=self.input_shape,
            random=self.train
        )

        # ------------------------------------------------- #
        # Normalize image and convert to CHW.
        # 图像归一化并转为 CHW 排列。
        # ------------------------------------------------- #
        image = np.array(image, dtype=np.float64)
        image = preprocess_input(image)
        image = np.transpose(image, [2, 0, 1])

        # ------------------------------------------------- #
        # Convert mask to numpy array.
        # 将标签图转为 numpy 数组。
        # ------------------------------------------------- #
        mask = np.array(mask, dtype=np.uint8)

        # ------------------------------------------------- #
        # Clamp invalid labels to ignore index = num_classes.
        # 将越界标签裁剪到忽略类索引 num_classes。
        #
        # Note:
        # VOC-like masks may contain boundary/ignore values.
        # VOC 类数据中可能包含边界或忽略区域像素值。
        # ------------------------------------------------- #
        mask[mask >= self.num_classes] = self.num_classes

        # ------------------------------------------------- #
        # Convert mask to one-hot format.
        # 将标签转换为 one-hot 格式。
        #
        # The extra +1 channel is reserved for ignored pixels.
        # 多出的 +1 通道用于存放忽略像素。
        # ------------------------------------------------- #
        seg_labels = np.eye(self.num_classes + 1, dtype=np.float32)[mask.reshape([-1])]
        seg_labels = seg_labels.reshape(
            (int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1)
        )

        return image, mask, seg_labels

    @staticmethod
    def rand(a=0.0, b=1.0):
        """
        Uniform random number in [a, b).
        返回 [a, b) 区间内的均匀随机数。
        """
        return np.random.rand() * (b - a) + a

    def get_random_data(
        self,
        image,
        label,
        input_shape,
        jitter=0.3,
        hue=0.1,
        sat=0.7,
        val=0.3,
        random=True
    ):
        """
        Apply resize / letterbox / random augmentation.
        执行缩放、灰边填充及随机增强。

        Args:
            image (PIL.Image): RGB image.
                输入图像。
            label (PIL.Image): Segmentation mask.
                分割标签图。
            input_shape (tuple/list): Target size [H, W].
                目标尺寸 [高, 宽]。
            jitter (float): Aspect ratio jitter factor.
                宽高比扰动系数。
            hue (float): HSV hue jitter factor.
                HSV 色调扰动系数。
            sat (float): HSV saturation jitter factor.
                HSV 饱和度扰动系数。
            val (float): HSV brightness jitter factor.
                HSV 明度扰动系数。
            random (bool): Whether to use random augmentation.
                是否启用随机增强。
        """
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        # ------------------------------------------------- #
        # Original image size and target size.
        # 原始尺寸与目标尺寸。
        # ------------------------------------------------- #
        original_w, original_h = image.size
        target_h, target_w = input_shape

        # ------------------------------------------------- #
        # Validation / inference branch:
        # Keep aspect ratio with gray padding.
        # 验证 / 推理阶段：
        # 保持长宽比并使用灰边填充。
        # ------------------------------------------------- #
        if not random:
            scale = min(target_w / original_w, target_h / original_h)
            resized_w = int(original_w * scale)
            resized_h = int(original_h * scale)

            resized_image = image.resize((resized_w, resized_h), Image.BICUBIC)
            padded_image = Image.new("RGB", [target_w, target_h], (128, 128, 128))
            padded_image.paste(
                resized_image,
                ((target_w - resized_w) // 2, (target_h - resized_h) // 2)
            )

            resized_label = label.resize((resized_w, resized_h), Image.NEAREST)
            padded_label = Image.new("L", [target_w, target_h], 0)
            padded_label.paste(
                resized_label,
                ((target_w - resized_w) // 2, (target_h - resized_h) // 2)
            )
            return padded_image, padded_label

        # ------------------------------------------------- #
        # Random resize with aspect ratio jitter.
        # 随机缩放并加入宽高比扰动。
        # ------------------------------------------------- #
        new_aspect_ratio = (
            original_w / original_h
            * self.rand(1 - jitter, 1 + jitter)
            / self.rand(1 - jitter, 1 + jitter)
        )
        scale = self.rand(0.25, 2.0)

        if new_aspect_ratio < 1:
            resized_h = int(scale * target_h)
            resized_w = int(resized_h * new_aspect_ratio)
        else:
            resized_w = int(scale * target_w)
            resized_h = int(resized_w / new_aspect_ratio)

        image = image.resize((resized_w, resized_h), Image.BICUBIC)
        label = label.resize((resized_w, resized_h), Image.NEAREST)

        # ------------------------------------------------- #
        # Random horizontal flip.
        # 随机水平翻转。
        # ------------------------------------------------- #
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------------- #
        # Paste resized image onto padded canvas.
        # 将缩放后的图像粘贴到带灰边的画布中。
        # ------------------------------------------------- #
        dx = int(self.rand(0, target_w - resized_w))
        dy = int(self.rand(0, target_h - resized_h))

        padded_image = Image.new("RGB", (target_w, target_h), (128, 128, 128))
        padded_label = Image.new("L", (target_w, target_h), 0)

        padded_image.paste(image, (dx, dy))
        padded_label.paste(label, (dx, dy))

        image = padded_image
        label = padded_label

        # ------------------------------------------------- #
        # Convert image to numpy for HSV augmentation.
        # 将图像转为 numpy 以执行 HSV 增强。
        # ------------------------------------------------- #
        image_data = np.array(image, np.uint8)

        # ------------------------------------------------- #
        # Random HSV perturbation coefficients.
        # 随机生成 HSV 扰动系数。
        # ------------------------------------------------- #
        random_factors = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        # ------------------------------------------------- #
        # Split HSV channels.
        # 转到 HSV 空间并拆分通道。
        # ------------------------------------------------- #
        hue_channel, sat_channel, val_channel = cv2.split(
            cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
        )
        dtype = image_data.dtype

        # ------------------------------------------------- #
        # Build LUTs and apply color jitter.
        # 构造查找表并应用颜色扰动。
        # ------------------------------------------------- #
        lut_x = np.arange(0, 256, dtype=random_factors.dtype)
        lut_hue = ((lut_x * random_factors[0]) % 180).astype(dtype)
        lut_sat = np.clip(lut_x * random_factors[1], 0, 255).astype(dtype)
        lut_val = np.clip(lut_x * random_factors[2], 0, 255).astype(dtype)

        image_data = cv2.merge(
            (
                cv2.LUT(hue_channel, lut_hue),
                cv2.LUT(sat_channel, lut_sat),
                cv2.LUT(val_channel, lut_val),
            )
        )
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def unet_dataset_collate(batch):
    """
    Collate function for segmentation batches.
    语义分割任务的批量拼接函数。

    Returns:
        images (torch.FloatTensor): [B, C, H, W]
        pngs (torch.LongTensor): [B, H, W]
        seg_labels (torch.FloatTensor): [B, H, W, num_classes + 1]
    """
    images = []
    pngs = []
    seg_labels = []

    for image, png, labels in batch:
        images.append(image)
        pngs.append(png)
        seg_labels.append(labels)

    images = torch.from_numpy(np.array(images)).float()
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).float()

    return images, pngs, seg_labels