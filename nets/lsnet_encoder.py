# nets/lsnet_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import SqueezeExcite
from timm.models.vision_transformer import trunc_normal_

# 如果你的 LSNet 项目里有自定义的 SKA，请确保同目录有 `ska.py`
# 没有的话先把那份文件也拷过来。
from .ska import SKA


# ------------------------------
# 基础积木（与 LSNet 一致）
# ------------------------------
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups, w.size(0), w.shape[2:], stride=self.c.stride,
            padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups, bias=True, device=c.weight.device
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(a))
        self.add_module('l', nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            # 随机残差丢弃
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class Attention(nn.Module):
    """与原 LSNet 一致的相对位置偏置注意力（按像素展平做 self-attn）"""
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.qkv = Conv2d_BN(dim, self.dh + nh_kd * 2, ks=1)
        self.proj = nn.Sequential(nn.ReLU(inplace=True), Conv2d_BN(self.dh, dim, bn_weight_init=0))
        self.dw = Conv2d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)

        # 相对位置偏置索引
        points = [(i, j) for i in range(resolution) for j in range(resolution)]
        N = len(points)
        attention_offsets, idxs = {}, []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode: bool = True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        q = q.view(B, self.num_heads, -1, N)
        k = k.view(B, self.num_heads, -1, N)
        v = v.view(B, self.num_heads, -1, N)
        attn = (q.transpose(-2, -1) @ k) * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x


class RepVGGDW(nn.Module):
    """深度可分离的 RepVGG 单元，用作 mixer"""
    def __init__(self, ed):
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        conv1_w = F.pad(conv1.weight, [1, 1, 1, 1])
        identity = F.pad(torch.ones_like(conv1_w[:, :, :1, :1]), [1, 1, 1, 1])
        final_w = conv.weight + conv1_w + identity
        final_b = conv.bias + conv1.bias
        conv.weight.data.copy_(final_w)
        conv.bias.data.copy_(final_b)
        return conv


class LKP(nn.Module):
    """Large Kernel Prior + 生成空间可变卷积权重"""
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU(inplace=True)
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, w_ = w.size()
        return w.view(b, self.dim // self.groups, self.sks ** 2, h, w_)


class LSConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class Block(nn.Module):
    """与原 LSNet 的 stage block 一致"""
    def __init__(self, ed, kd, nh=8, ar=4, resolution=14, stage=-1, depth=-1):
        super().__init__()
        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se = SqueezeExcite(ed, 0.25)
        else:
            self.se = nn.Identity()
            self.mixer = Attention(ed, kd, nh, ar, resolution=resolution) if stage == 3 else LSConv(ed)
        self.ffn = Residual(FFN(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))


# ------------------------------
# Encoder：返回 5 个尺度 (1/2,1/4,1/8,1/16,1/32)
# ------------------------------
class LSNetEncoder(nn.Module):
    """
    仅保留 LSNet 的“下采样 + 主干”，去掉分类 head。
    输出:
        feat1: 1/2   (channels = embed_dim[0]//4)
        feat2: 1/4   (channels = embed_dim[0]//2)
        feat3: 1/8   (channels = embed_dim[0])
        feat4: 1/16  (channels = embed_dim[1])
        feat5: 1/32  (channels = embed_dim[2])  ← 注意：我们**不使用** stage4 的再一次 2× 下采样，
                                                   这样就与 U-Net 的 2×金字塔完全对齐
    """
    def __init__(
        self,
        pretrained: bool = False,   # 这里保留接口占位，不在此文件内加载权重
        img_size: int = 224,
        patch_size: int = 8,        # 与 LSNet 注册模型一致
        in_chans: int = 3,
        embed_dim = (64, 128, 256, 384),
        key_dim   = (16, 16, 16, 16),
        depth     = (0, 2, 8, 10),
        num_heads = (3, 3, 3, 4),
    ):
        super().__init__()
        self.embed_dim = list(embed_dim)
        # ---- stem：三次 stride=2（/2, /4, /8），与 LSNet 原实现一致 ----
        self.stem1 = Conv2d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1)
        self.act1  = nn.ReLU(inplace=True)
        self.stem2 = Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1)
        self.act2  = nn.ReLU(inplace=True)
        self.stem3 = Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)

        # ---- 四个 stage，与 LSNet 一致（但我们只取到 stage3 作为 1/32）----
        self.blocks1 = nn.Sequential(*[
            Block(embed_dim[0], key_dim[0], num_heads[0], embed_dim[0] / (key_dim[0] * num_heads[0]), img_size // patch_size, stage=0, depth=d)
            for d in range(depth[0])
        ])

        # 下采样到 stage2 起始（DW 3x3 s=2 + PW 1x1）
        self.down12 = nn.Sequential(
            Conv2d_BN(embed_dim[0], embed_dim[0], ks=3, stride=2, pad=1, groups=embed_dim[0]),
            Conv2d_BN(embed_dim[0], embed_dim[1], ks=1, stride=1, pad=0),
        )
        self.blocks2 = nn.Sequential(*[
            Block(embed_dim[1], key_dim[1], num_heads[1], embed_dim[1] / (key_dim[1] * num_heads[1]), (img_size // patch_size) // 2, stage=1, depth=d)
            for d in range(depth[1])
        ])

        # 下采样到 stage3 起始
        self.down23 = nn.Sequential(
            Conv2d_BN(embed_dim[1], embed_dim[1], ks=3, stride=2, pad=1, groups=embed_dim[1]),
            Conv2d_BN(embed_dim[1], embed_dim[2], ks=1, stride=1, pad=0),
        )
        self.blocks3 = nn.Sequential(*[
            Block(embed_dim[2], key_dim[2], num_heads[2], embed_dim[2] / (key_dim[2] * num_heads[2]), (img_size // patch_size) // 4, stage=2, depth=d)
            for d in range(depth[2])
        ])

        # 下采样到 stage4（原 LSNet 里这里还有一次 2×，会到 1/64）
        # 为了和 U-Net 的金字塔对齐，我们保留 stage4 的“block 深度”，
        # 但不做这次下采样，直接在 1/32 上继续堆叠（可选）。
        # 如需严格复刻 LSNet，请把下面两行注释取消，并在 forward 中使用它。
        self.down34 = nn.Sequential(
            Conv2d_BN(embed_dim[2], embed_dim[2], ks=3, stride=2, pad=1, groups=embed_dim[2]),
            Conv2d_BN(embed_dim[2], embed_dim[3], ks=1, stride=1, pad=0),
        )
        self.blocks4 = nn.Sequential(*[
            Block(embed_dim[3], key_dim[3], num_heads[3], embed_dim[3] / (key_dim[3] * num_heads[3]), (img_size // patch_size) // 8, stage=3, depth=d)
            for d in range(depth[3])
        ])

        # 给 U-Net 使用的各尺度通道数（1/2, 1/4, 1/8, 1/16, 1/32）
        self.stage_channels = (
            embed_dim[0] // 4,   # C1
            embed_dim[0] // 2,   # C2
            embed_dim[0],        # C3
            embed_dim[1],        # C4
            embed_dim[2],        # C5（注意：取 stage3 输出，保持 1/32）
        )

    def forward(self, x):
        # 1/2
        x1 = self.act1(self.stem1(x))
        # 1/4
        x2 = self.act2(self.stem2(x1))
        # 1/8
        x3_in = self.stem3(x2)
        x3 = self.blocks1(x3_in)

        # 1/16
        x4_in = self.down12(x3)
        x4 = self.blocks2(x4_in)

        # 1/32  —— 这里我们把 stage3 的输出作为金字塔最顶层 (C5)
        x5_in = self.down23(x4)
        x5 = self.blocks3(x5_in)

        # 如需“完全还原 LSNet 的 stage4（到 1/64）”，可以打开下面两行并返回 x6：
        # x6_in = self.down34(x5)
        # x6 = self.blocks4(x6_in)

        return x1, x2, x3, x4, x5  # C1..C5

    # 与你现有 Unet 使用方式一致
    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True
