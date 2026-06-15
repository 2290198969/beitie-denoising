"""
U-Net 用于碑帖图像去噪
----------------------
经典 U-Net 改造版：输入带噪图，输出去噪图。
原 U-Net 是做分割（输出mask），这里做回归（输出图像），最后一层不接 Sigmoid，
让网络自由输出每个像素的灰度/RGB值。

结构（"U"形）：
    输入(H×W)
        ↓ Down1 → 跳连 ────────────────┐
        ↓ Down2 → 跳连 ──────────────┐ │
        ↓ Down3 → 跳连 ────────────┐ │ │
        ↓ Down4 → 跳连 ──────────┐ │ │ │
        ↓ Bottleneck            │ │ │ │
        ↑ Up4   ← concat ───────┘ │ │ │
        ↑ Up3   ← concat ─────────┘ │ │
        ↑ Up2   ← concat ───────────┘ │
        ↑ Up1   ← concat ─────────────┘
    输出(H×W)

为什么需要"跳连(skip connection)"？
    下采样会丢空间细节（笔画边缘），上采样恢复不回来。
    把编码器的特征直接"拷贝"到解码器，等于把细节"抄"过来。
    去噪任务里这一点尤其重要——我们要保住笔画的锋芒。

    上采样就只是把图片给模糊的放大了，
    没有提取信息，
    下采样提取了模糊的大边界，
    但是图片越来越小越来越模糊了。
    当我上采样后，利用跳跃连接送来的原图，
    把原图的信息一点一点加进去，
    此时，小边界就被提取出来了，
    因为在大边界里可以再找小边界

up放大完跳连
我跳跃连接以后的通道数变成两倍
所以再双卷积就可以把它变小
第一层卷积：512 → 256（压缩）
第二层卷积：256 → 256（精炼）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """两次 [Conv3x3 → BN → ReLU]，U-Net 的基本积木。
    
    为什么是两次而不是一次？
        两次卷积感受野更大（5x5），表达能力更强，是原论文的设计。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """下采样块：MaxPool2x2 + DoubleConv。空间尺寸减半，通道数翻倍。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """上采样块：上采样 + 与编码器特征 concat + DoubleConv。
    
    上采样有两种选择：
        1. nn.ConvTranspose2d (反卷积) —— 参数多，可能棋盘伪影
        2. F.interpolate (双线性插值) + Conv —— 更轻量、更稳
    这里选 2，去噪任务更合适。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels = 上层特征通道 + 跳连特征通道
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        # x: 来自下层的特征  skip: 来自编码器的跳连特征
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # 如果尺寸有偏差（奇数尺寸时会差1像素），裁剪对齐
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)  # 沿通道维度拼接
        return self.conv(x)


class UNet(nn.Module):
    """U-Net 去噪网络。
    
    参数:
        in_channels:  输入通道（灰度=1，RGB=3）
        out_channels: 输出通道（同输入）
        base_ch:      第一层的通道数，越大模型越大。32够用，64标准。

        输入：你有 1 个矩阵（灰度图），大小假设是 [1, 512, 512]（1个通道）。
卷积层：这一层不是只有一个卷积核，而是有 32 个并行的卷积核。每个卷积核的大小是 3x3
输出通道数 = 你用了多少个卷积核。
想要输出 32 个矩阵？这一层就定义 32 个卷积核。
想要输出 64 个矩阵？这一层就定义 64 个卷积核。

        residual:     是否用残差学习（学"噪声"而不是"干净图"）
    
    残差学习是什么？
        正常网络: 输入带噪图 → 输出干净图
        残差学习: 输入带噪图 → 输出"噪声" → 带噪图 - 噪声 = 干净图
        DnCNN 论文证明残差学习收敛更快、效果更好。
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, residual=True):
        super().__init__()
        self.residual = residual

        # 编码器（下采样）（尺寸减半通道翻倍）
        self.inc   = DoubleConv(in_channels, base_ch)         # 1   → 32
        self.down1 = Down(base_ch,    base_ch * 2)            # 32  → 64
        self.down2 = Down(base_ch * 2, base_ch * 4)           # 64  → 128
        self.down3 = Down(base_ch * 4, base_ch * 8)           # 128 → 256
        self.down4 = Down(base_ch * 8, base_ch * 16)          # 256 → 512  (bottleneck)

        # 解码器（上采样 + 跳连）
        # 输入通道 = 上层通道 + 跳连通道
        self.up1 = Up(base_ch * 16 + base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8  + base_ch * 4, base_ch * 4)
        self.up3 = Up(base_ch * 4  + base_ch * 2, base_ch * 2)
        self.up4 = Up(base_ch * 2  + base_ch,     base_ch)

        # 输出层：1x1 卷积把通道数压回去
        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码
        x1 = self.inc(x)      #  [B,  32, H,    W]
        x2 = self.down1(x1)   #  [B,  64, H/2,  W/2]
        x3 = self.down2(x2)   #  [B, 128, H/4,  W/4]
        x4 = self.down3(x3)   #  [B, 256, H/8,  W/8]
        x5 = self.down4(x4)   #  [B, 512, H/16, W/16]

        # 解码（带跳连）
        y = self.up1(x5, x4)
        y = self.up2(y,  x3)
        y = self.up3(y,  x2)
        y = self.up4(y,  x1)
        y = self.outc(y)

        if self.residual:
            # 学噪声: 网络预测噪声 y，干净图 = 输入 - 噪声
            return x - y
        else:
            return y


if __name__ == "__main__":
    # 自测：随便造个 batch 看看能不能跑通
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=1, out_channels=1, base_ch=32, residual=True).to(device)
    dummy = torch.randn(2, 1, 256, 256).to(device)  # batch=2, 灰度, 256x256
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"输入: {dummy.shape}")
    print(f"输出: {out.shape}")
    print(f"参数量: {n_params/1e6:.2f} M")
    print(f"设备: {device}")
