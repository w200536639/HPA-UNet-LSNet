import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =========================================================
# Identity helper module for ablation / 消融实验恒等辅助模块
# =========================================================
class AddZero(nn.Module):
    """
    Return an all-zeros tensor with the same shape as input.
    返回与输入同形状的全0张量。
    """

    def forward(self, x):
        return torch.zeros_like(x)


# =========================================================
# Generic convolution block / 通用卷积模块
# Kept for compatibility and future extension / 保留以兼容旧工程及后续扩展
# =========================================================
class ConvModule(nn.Module):
    """
    Convolution + optional normalization + optional activation.
    卷积 + 可选归一化 + 可选激活函数。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2 if padding is None else padding,
                groups=groups,
                bias=(norm_cfg is None),
            )
        ]

        if norm_cfg:
            if norm_cfg["type"] == "BN":
                layers.append(
                    nn.BatchNorm2d(
                        out_channels,
                        momentum=norm_cfg.get("momentum", 0.1),
                        eps=norm_cfg.get("eps", 1e-5),
                    )
                )
            else:
                raise NotImplementedError(f"Norm type {norm_cfg['type']} is not implemented.")

        if act_cfg:
            if act_cfg["type"] == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif act_cfg["type"] == "SiLU":
                layers.append(nn.SiLU(inplace=True))
            else:
                raise NotImplementedError(f"Act type {act_cfg['type']} is not implemented.")

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# =========================================================
# HPA: Hybrid Pooling Attention / HPA：混合池化注意力模块
# Parameter names are kept close to the old version / 参数命名尽量靠近旧版
# =========================================================
class HPA(nn.Module):
    """
    Hybrid Pooling Attention (HPA).
    混合池化注意力模块（HPA）。
    """

    def __init__(self, channels, factor=32):
        super().__init__()
        self.groups = factor
        assert channels % self.groups == 0, (
            f"HPA: channels {channels} must be divisible by factor {factor}"
        )

        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.AdaptiveMaxPool2d((1, 1))

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))
        self.max_w = nn.AdaptiveMaxPool2d((1, None))

        channels_per_group = channels // self.groups
        self.gn = nn.GroupNorm(
            num_groups=channels_per_group,
            num_channels=channels_per_group
        )
        self.conv1x1 = nn.Conv2d(channels_per_group, channels_per_group, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        num_groups = self.groups

        grouped_x = x.reshape(batch_size * num_groups, -1, height, width)

        xh = self.pool_h(grouped_x)
        xw = self.pool_w(grouped_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([xh, xw], dim=2))
        xh, xw = torch.split(hw, [height, width], dim=2)

        x1 = self.gn(
            grouped_x * xh.sigmoid() * xw.permute(0, 1, 3, 2).sigmoid()
        )

        yh = self.max_h(grouped_x)
        yw = self.max_w(grouped_x).permute(0, 1, 3, 2)
        yhw = self.conv1x1(torch.cat([yh, yw], dim=2))
        yh, yw = torch.split(yhw, [height, width], dim=2)

        y1 = self.gn(
            grouped_x * yh.sigmoid() * yw.permute(0, 1, 3, 2).sigmoid()
        )

        x11 = x1.reshape(batch_size * num_groups, -1, height * width)
        x12 = self.softmax(
            self.agp(x1).reshape(batch_size * num_groups, -1, 1).permute(0, 2, 1)
        )

        y11 = y1.reshape(batch_size * num_groups, -1, height * width)
        y12 = self.softmax(
            self.map(y1).reshape(batch_size * num_groups, -1, 1).permute(0, 2, 1)
        )

        weights = (
            torch.matmul(x12, y11) +
            torch.matmul(y12, x11)
        ).reshape(batch_size * num_groups, 1, height, width)

        return (grouped_x * weights.sigmoid()).reshape(batch_size, channels, height, width)


# =========================================================
# LSNet encoder blocks / LSNet编码器模块
# Key names are reverted toward the old implementation / 键名回退到旧版风格
# =========================================================
class Conv2d_BN(nn.Sequential):
    """
    Convolution followed by BatchNorm.
    卷积 + BatchNorm。
    """

    def __init__(self, in_channels, out_channels, ks=1, stride=1, pad=0, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, ks, stride, pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )


class LSBasicBlock(nn.Module):
    """
    Basic lightweight block used in LSNet.
    LSNet 中使用的基础轻量块。
    """

    def __init__(self, channels, expand=2.0):
        super().__init__()
        hidden_channels = int(channels * expand)

        self.branch = nn.Sequential(
            Conv2d_BN(channels, channels, ks=3, stride=1, pad=1, groups=channels),
            nn.ReLU(inplace=True),
            Conv2d_BN(channels, hidden_channels, ks=1),
            nn.ReLU(inplace=True),
            Conv2d_BN(hidden_channels, channels, ks=1),
        )

    def forward(self, x):
        return x + self.branch(x)


class LSDown(nn.Module):
    """
    Downsampling block in LSNet.
    LSNet 下采样模块。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Sequential(
            Conv2d_BN(in_channels, in_channels, ks=3, stride=2, pad=1, groups=in_channels),
            nn.ReLU(inplace=True),
            Conv2d_BN(in_channels, out_channels, ks=1),
        )

    def forward(self, x):
        return self.op(x)


class LSStem(nn.Module):
    """
    Stem block of LSNet.
    LSNet 的 stem 模块。

    Notes:
        The attribute names s1/s2/s3 are intentionally kept to match old checkpoints.
        这里故意保留 s1/s2/s3 命名，以尽量兼容旧权重键名。
    """

    def __init__(self, out_c):
        super().__init__()
        c1, c2, c3 = out_c // 4, out_c // 2, out_c

        self.s1 = nn.Sequential(Conv2d_BN(3, c1, 3, 2, 1), nn.ReLU(inplace=True))   # /2
        self.s2 = nn.Sequential(Conv2d_BN(c1, c2, 3, 2, 1), nn.ReLU(inplace=True))  # /4
        self.s3 = nn.Sequential(Conv2d_BN(c2, c3, 3, 2, 1))                         # /8

        self.out_c = (c1, c2, c3)

    def forward(self, x):
        f2 = self.s1(x)
        f4 = self.s2(f2)
        f8 = self.s3(f4)
        return f2, f4, f8


class LSNetEncoder(nn.Module):
    """
    LSNet encoder.
    LSNet 编码器。
    """

    PRESETS = {
        "lsnet_t": dict(embed_dim=[64, 128, 256, 384], depth=[0, 2, 8, 10]),
        "lsnet_s": dict(embed_dim=[96, 192, 320, 448], depth=[1, 2, 8, 10]),
        "lsnet_b": dict(embed_dim=[128, 256, 384, 512], depth=[4, 6, 8, 10]),
    }

    def __init__(self, variant="lsnet_b"):
        super().__init__()
        variant = variant.lower()
        assert variant in self.PRESETS, f"Unsupported LSNet variant: {variant}"

        cfg = self.PRESETS[variant]
        embed_dims, depths = cfg["embed_dim"], cfg["depth"]

        self.stem = LSStem(embed_dims[0])

        self.stage1 = nn.Sequential(*[LSBasicBlock(embed_dims[0]) for _ in range(depths[0])])

        self.down12 = LSDown(embed_dims[0], embed_dims[1])
        self.stage2 = nn.Sequential(*[LSBasicBlock(embed_dims[1]) for _ in range(depths[1])])

        self.down23 = LSDown(embed_dims[1], embed_dims[2])
        self.stage3 = nn.Sequential(*[LSBasicBlock(embed_dims[2]) for _ in range(depths[2])])

        self.down34 = LSDown(embed_dims[2], embed_dims[3])
        self.stage4 = nn.Sequential(*[LSBasicBlock(embed_dims[3]) for _ in range(depths[3])])

        # output channels: /2, /4, /8, /16, /32
        # 输出通道：/2, /4, /8, /16, /32
        self.channels = (
            embed_dims[0] // 4,
            embed_dims[0] // 2,
            embed_dims[0],
            embed_dims[1],
            embed_dims[2],
        )

    @torch.no_grad()
    def out_channels(self):
        return self.channels

    def forward(self, x):
        f2, f4, f8 = self.stem(x)
        f8 = self.stage1(f8)
        f16 = self.stage2(self.down12(f8))
        f32 = self.stage3(self.down23(f16))
        _ = self.stage4(self.down34(f32))
        return [f2, f4, f8, f16, f32]


# =========================================================
# Decoder blocks / 解码器模块
# Use old-style outer names while keeping current no-BIE functionality
# 使用旧版外层命名，但保持当前“无 BIE”功能
# =========================================================
class UnetUp(nn.Module):
    """
    Standard decoder upsampling block.
    标准解码上采样块。

    This block keeps the old module name `UnetUp`.
    该模块保留旧名称 `UnetUp`。
    """

    def __init__(self, in_size, out_size):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip, lowres):
        x = torch.cat([skip, self.up(lowres)], dim=1)
        return self.conv(x)


class UpNoBIE(nn.Module):
    """
    Decoder fusion block without BIE, but with old-style attribute names.
    无 BIE 的解码融合块，但内部属性尽量采用旧版风格命名。

    Notes:
        This is NOT the old BIE block.
        这不是旧版 BIE 模块。
    """

    def __init__(self, c_skip, c_low, out_ch):
        super().__init__()
        self.align_low = nn.Conv2d(c_low, c_skip, 1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * c_skip, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip, lowres):
        low = F.interpolate(
            lowres,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        low = self.align_low(low)
        fused = torch.cat([skip, low], dim=1)
        return self.conv(fused)


# =========================================================
# Main network / 主网络
# Keep current functionality, but revert parameter key style
# 保持当前功能，但尽量回退参数键名风格
# =========================================================
class HPAUNetLSNet(nn.Module):
    """
    HPA-UNet-LSNet without BIE for semantic segmentation.
    去除 BIE 后的 HPA-UNet-LSNet 语义分割网络。
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=False,
        backbone="lsnet_b",
        use_hpa: bool = True,
        **kwargs,
    ):
        """
        Args:
            num_classes: number of segmentation classes / 分割类别数
            pretrained: reserved for compatibility / 为兼容性保留
            backbone: LSNet backbone variant / LSNet 主干型号
            use_hpa: whether to enable HPA in encoder / 是否在编码器中启用 HPA
        """
        super().__init__()

        self.backbone_name = backbone.lower()
        assert self.backbone_name in LSNetEncoder.PRESETS, f"Unsupported backbone `{backbone}`"

        if pretrained:
            print("[Warning] pretrained=True is reserved for compatibility and is currently ignored. / pretrained=True 目前仅为兼容保留，暂未启用。")

        if len(kwargs) > 0:
            for key in kwargs.keys():
                print(f"[Warning] HPAUNetLSNet: argument `{key}` is ignored.")

        self.use_hpa = use_hpa

        # -------------------------------------------------
        # Encoder / 编码器
        # -------------------------------------------------
        self.encoder = LSNetEncoder(self.backbone_name)
        c2, c4, c8, c16, c32 = self.encoder.out_channels()

        # -------------------------------------------------
        # HPA modules / HPA 模块
        # Revert attribute names to old style / 属性名回退为旧版风格
        # -------------------------------------------------
        if use_hpa:
            self.hpa4 = HPA(c4, factor=32)
            self.hpa8 = HPA(c8, factor=32)
            self.hpa16 = HPA(c16, factor=32)
            self.hpa32 = HPA(c32, factor=32)
        else:
            self.hpa4 = AddZero()
            self.hpa8 = AddZero()
            self.hpa16 = AddZero()
            self.hpa32 = AddZero()

        # -------------------------------------------------
        # Decoder channel settings / 解码器通道设置
        # -------------------------------------------------
        dec = [64, 128, 256, 512]  # /2, /4, /8, /16

        # Keep old outer attribute names: up4/up3/up2/up1
        # 保留旧版外层模块名：up4/up3/up2/up1
        self.up4 = UnetUp(c16 + c32, dec[3])
        self.up3 = UpNoBIE(c_skip=c8, c_low=dec[3], out_ch=dec[2])
        self.up2 = UpNoBIE(c_skip=c4, c_low=dec[2], out_ch=dec[1])
        self.up1 = UpNoBIE(c_skip=c2, c_low=dec[1], out_ch=dec[0])

        # -------------------------------------------------
        # Output head / 输出头
        # Revert name to out_head / 名称回退为 out_head
        # -------------------------------------------------
        self.out_head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(dec[0], dec[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec[0], dec[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(dec[0], num_classes, 1)

    def forward(self, x):
        f2, f4, f8, f16, f32 = self.encoder(x)

        # Residual HPA enhancement / 残差式 HPA 增强
        f4 = f4 + self.hpa4(f4)
        f8 = f8 + self.hpa8(f8)
        f16 = f16 + self.hpa16(f16)
        f32 = f32 + self.hpa32(f32)

        # Decoder / 解码器
        u4 = self.up4(f16, f32)   # -> 1/16
        u3 = self.up3(f8, u4)     # -> 1/8
        u2 = self.up2(f4, u3)     # -> 1/4
        u1 = self.up1(f2, u2)     # -> 1/2

        logits = self.out_head(u1)
        logits = self.final(logits)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return logits

    def freeze_backbone(self):
        """
        Freeze encoder parameters.
        冻结编码器参数。
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze encoder parameters.
        解冻编码器参数。
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
