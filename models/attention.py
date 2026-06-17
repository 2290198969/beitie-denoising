"""
CBAM 注意力模块
===============
论文: CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)

包含两个串联的子模块:
    1. ChannelAttention  —— 通道注意力，学"哪张特征图重要"
    2. SpatialAttention   —— 空间注意力，学"图里哪些位置重要"

数据流:
    输入 X (B, C, H, W)
        ↓
    Mc = ChannelAttention(X)    形状 (B, C, 1, 1)
    X1 = X * Mc                 通道加权
        ↓
    Ms = SpatialAttention(X1)   形状 (B, 1, H, W)
    X2 = X1 * Ms                空间加权
        ↓
    输出 X2 (B, C, H, W)        和输入同尺寸，即插即用

为什么对碑帖去噪有用？
    通道注意力：去噪时不同特征图作用不同——有的捕获笔画边缘，有的捕获噪声纹理。
                CA 让网络放大"边缘特征"、抑制"噪声特征"。
    空间注意力：碑帖图里 80% 是空白背景，注意力让网络把算力集中在笔画区域，
                避免在背景上过度去噪导致背景"假平滑"。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力 (Squeeze-and-Excitation 的强化版)。

    流程:
        1. 对 (B, C, H, W) 做空间 GAP 和 GMP，分别压成 (B, C, 1, 1)
        2. 两路共享一个 MLP（参数共享是 CBAM 的关键设计），输出 (B, C, 1, 1)
        3. 相加 → Sigmoid → 通道权重 Mc

    为什么 GAP + GMP 都要？
        GAP 关注"全局平均强度"，GMP 关注"最显著激活"。两者互补，单用 SE Block 只有 GAP。

    参数:
        in_channels: 输入通道数
        ratio:       MLP 中间层压缩比，原论文 16
    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        # 防止通道数太小时 in_channels // ratio 变成 0
        hidden = max(in_channels // ratio, 4)

        # 共享 MLP（用 1x1 Conv 实现等价于 Linear，但形状好处理）
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # 全局平均池化 → MLP
        avg_pool = F.adaptive_avg_pool2d(x, 1)      # (B, C, 1, 1)
        avg_out = self.mlp(avg_pool)
        # 全局最大池化 → MLP（共享同一个 mlp）
        max_pool = F.adaptive_max_pool2d(x, 1)      # (B, C, 1, 1)
        max_out = self.mlp(max_pool)
        # 相加 + Sigmoid 得到 [0,1] 的通道权重
        weight = torch.sigmoid(avg_out + max_out)   # (B, C, 1, 1)
        return x * weight                            # 广播乘法


class SpatialAttention(nn.Module):
    """空间注意力。

    流程:
        1. 沿通道维度做 mean 和 max，分别得到 (B, 1, H, W)
        2. concat → (B, 2, H, W)
        3. 7x7 卷积 → (B, 1, H, W) → Sigmoid → 空间权重 Ms

    为什么用 7x7 大卷积？
        空间注意力要"看大局"决定哪里重要，大卷积有大感受野。
        通道维只有 2 通道，参数也不多。

    参数:
        kernel_size: 7 (原论文)，必须奇数
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须奇数"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)

    def forward(self, x):
        # 沿通道维度统计
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)
        cat = torch.cat([avg_out, max_out], dim=1)     # (B, 2, H, W)
        weight = torch.sigmoid(self.conv(cat))         # (B, 1, H, W)
        return x * weight                               # 广播乘法


class CBAM(nn.Module):
    """CBAM = ChannelAttention + SpatialAttention 串联。

    用法:
        cbam = CBAM(channels=64)
        out = cbam(feature_map)   # 输入输出同形状

    参数:
        channels:    输入通道数
        ratio:       通道注意力的压缩比
        spatial_ks:  空间注意力的卷积核大小
    """
    def __init__(self, channels, ratio=16, spatial_ks=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=spatial_ks)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


if __name__ == "__main__":
    """自测"""
    cbam = CBAM(channels=64)
    x = torch.randn(2, 64, 32, 32)
    y = cbam(x)
    n_params = sum(p.numel() for p in cbam.parameters())
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    print(f"CBAM 参数量: {n_params} 个 (~{n_params/1024:.2f}K)")
