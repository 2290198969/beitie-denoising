"""
评价指标
--------
去噪效果怎么量化？两个核心指标（论文必写）：

1. PSNR (Peak Signal-to-Noise Ratio，峰值信噪比)
    - 单位: dB（分贝）
    - 越高越好，一般 >30dB 说明效果不错
    - 原理: 比较去噪图和真实干净图的像素差异
    - 公式: PSNR = 10 * log10(MAX² / MSE)
      其中 MSE = mean((去噪图 - 干净图)²)

2. SSIM (Structural Similarity，结构相似性)
    - 范围: [0, 1]，越接近1越好
    - 比 PSNR 更符合人眼感受（PSNR 只看像素差，SSIM 看结构）
    - 考虑了亮度、对比度、结构三个维度
    - 碑帖去噪中 SSIM 特别重要——笔画结构不能变形！

答辩时老师可能问：
    Q: "PSNR 和 SSIM 有什么区别？"
    A: "PSNR 衡量逐像素误差，SSIM 衡量结构保持程度。
        一张图整体变亮了 PSNR 很低，但笔画结构没变 SSIM 还很高。
        去噪任务中两个都要报告，SSIM 更能反映视觉质量。"
"""

import numpy as np
import torch


def calc_psnr(pred, target, data_range=1.0):
    """计算 PSNR。
    
    参数:
        pred: 去噪后的图 (numpy 或 torch tensor)
        target: 干净的 ground truth
        data_range: 像素值范围，归一化后是1.0，uint8是255
    
    返回:
        PSNR 值 (dB)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')  # 完全一样
    return 10 * np.log10(data_range ** 2 / mse)


def calc_ssim(pred, target, data_range=1.0, win_size=11):
    """计算 SSIM（简化版，单通道）。
    
    参数:
        pred: 去噪图, shape (H, W) 或 (1, H, W)
        target: 干净图
        win_size: 滑动窗口大小
    
    完整 SSIM 公式:
        SSIM(x,y) = (2*μx*μy + C1)(2*σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
        其中 C1, C2 是防除零的小常数
    
    这里用 scikit-image 的实现（学术界标准）。
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # 去掉 batch/channel 维度
    pred = pred.squeeze()
    target = target.squeeze()
    
    from skimage.metrics import structural_similarity
    return structural_similarity(pred, target, data_range=data_range, win_size=win_size)


class AverageMeter:
    """跟踪训练过程中指标的平均值。
    
    用法:
        meter = AverageMeter()
        for batch in dataloader:
            loss = ...
            meter.update(loss.item())
        print(f"平均loss: {meter.avg:.4f}")
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0    # 最新值
        self.avg = 0    # 平均值
        self.sum = 0    # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    """自测"""
    # 造两张测试图
    clean = np.random.rand(256, 256).astype(np.float32)
    noisy = clean + np.random.randn(256, 256).astype(np.float32) * 0.1  # sigma=0.1
    noisy = np.clip(noisy, 0, 1)
    
    psnr = calc_psnr(noisy, clean)
    ssim = calc_ssim(noisy, clean)
    
    print(f"带噪图 vs 干净图:")
    print(f"  PSNR = {psnr:.2f} dB")
    print(f"  SSIM = {ssim:.4f}")
    print()
    print(f"干净图 vs 自己:")
    print(f"  PSNR = {calc_psnr(clean, clean):.2f} dB (应该是 inf)")
    print(f"  SSIM = {calc_ssim(clean, clean):.4f} (应该是 1.0)")
