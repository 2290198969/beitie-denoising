"""
DnCNN —— Denoising Convolutional Neural Network
================================================
论文: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
       (Zhang et al., IEEE TIP 2017)
首次系统性证明：深度 CNN + 残差学习 全面超越 BM3D。

为什么要拿它当对比？
    它是"经典基线"，毕设/论文里写去噪几乎必比 DnCNN，证明你的 U-Net 不是瞎搞，
    是在公认的高水位线之上还有提升。

结构（很纯粹的"深而薄"）：
    输入 (1, H, W)
        ↓ Conv3x3 + ReLU                 ← 第1层（无 BN）
        ↓ [Conv3x3 + BN + ReLU] × 15     ← 中间15层
        ↓ Conv3x3                         ← 最后1层（无 BN、无 ReLU）
    输出残差噪声 (1, H, W)
    最终 干净图 = 输入 - 输出残差

和 U-Net 的区别（也是答辩可能问的）：
    DnCNN: 全程同分辨率 → 算力低、感受野靠堆深度凑
    U-Net: 多尺度下采样 → 大感受野、能看清"整个字"的结构
    对碑帖这种"笔画结构很重要"的任务，U-Net 天然更合适。

参数量：约 0.56M（17层、64通道），比 U-Net (7.85M) 小一个数量级，
       说明效果差距不能简单归因于"模型大就好"。
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """DnCNN 去噪网络。

    参数:
        in_channels:  输入通道（灰度=1）
        out_channels: 输出通道（=输入）
        num_layers:   总层数，原论文用 17。盲去噪用 20。
        num_features: 每层通道数，原论文用 64。
        residual:     残差学习（学噪声而不是学干净图）。原论文必开。
    """

    def __init__(self, in_channels=1, out_channels=1,
                 num_layers=17, num_features=64, residual=True):
        super().__init__()
        self.residual = residual

        layers = []

        # 第 1 层：Conv + ReLU（不带 BN）
        layers.append(nn.Conv2d(in_channels, num_features,
                                kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # 中间层：Conv + BN + ReLU，重复 num_layers-2 次
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features,
                                    kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # 最后 1 层：单独一个 Conv，输出残差噪声（不接激活，因为噪声有正有负）
        layers.append(nn.Conv2d(num_features, out_channels,
                                kernel_size=3, padding=1, bias=True))

        self.net = nn.Sequential(*layers)

        # 初始化（DnCNN 原论文用了 kaiming，能加速收敛）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            # 学噪声：网络输出"噪声估计"，干净图 = 输入 - 噪声
            return x - out
        return out


if __name__ == "__main__":
    """自测：看下能不能跑通 + 参数量"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DnCNN(in_channels=1, out_channels=1,
                  num_layers=17, num_features=64, residual=True).to(device)

    dummy = torch.randn(2, 1, 128, 128).to(device)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"输入: {dummy.shape}")
    print(f"输出: {out.shape}")
    print(f"参数量: {n_params/1e6:.3f} M")
    print(f"层数: 17 (Conv x 17 + BN x 15 + ReLU x 16)")
    print(f"设备: {device}")
