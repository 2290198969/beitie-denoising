"""
传统去噪方法封装（baselines）
============================
非深度学习方法，不需要训练，直接调库即可。
作为 U-Net / DnCNN 的对比基准。

四种方法：
    1. median  - 中值滤波  (cv2.medianBlur)
    2. nlm     - 非局部均值  (cv2.fastNlMeansDenoising)
    3. bm3d    - Block-Matching 3D (bm3d 库, 业界传统方法天花板)
    4. bilat   - 双边滤波  (cv2.bilateralFilter, 备选)

接口统一：
    输入:  numpy float32, shape (H, W), 范围 [0, 1]
    输出:  numpy float32, shape (H, W), 范围 [0, 1]

为什么要做这一层封装？
    每个方法的 API、输入范围都不一样：
        cv2.medianBlur 要 uint8
        cv2.fastNlMeansDenoising 要 uint8
        bm3d.bm3d 接受 float
    封装后，对比脚本里就能像调函数一样统一调用，代码干净。
"""

import numpy as np
import cv2


# ---------------- 1. 中值滤波 ----------------
def denoise_median(img, ksize=3):
    """中值滤波：取窗口内像素的中位数。

    参数:
        img:   numpy float32, [0, 1], (H, W)
        ksize: 窗口大小，必须是奇数。3 比 5 保细节，碑帖去噪 3 够用。

    适合: 椒盐噪声（中位数无视极端值）
    不适合: 高斯噪声（会糊掉笔画）
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out_u8 = cv2.medianBlur(img_u8, ksize)
    return out_u8.astype(np.float32) / 255.0


# ---------------- 2. 非局部均值 NLM ----------------
def denoise_nlm(img, h=10, template_window=7, search_window=21):
    """Non-Local Means：找全图相似 patch 加权平均。

    参数:
        img:             numpy float32, [0, 1]
        h:               滤波强度。越大去噪越狠、越糊。10 是 OpenCV 推荐起点。
        template_window: patch 大小。
        search_window:   搜索范围。越大效果越好、越慢。

    适合: 中等高斯噪声、纹理类噪声
    不适合: 强噪声（找不到"干净"的相似 patch 当参考）
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out_u8 = cv2.fastNlMeansDenoising(
        img_u8,
        h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )
    return out_u8.astype(np.float32) / 255.0


# ---------------- 3. BM3D ----------------
def denoise_bm3d(img, sigma_psd=25 / 255.0):
    """BM3D：Block-Matching + 3D 协同滤波。

    参数:
        img:       numpy float32, [0, 1]
        sigma_psd: 噪声标准差估计。值要和真实噪声匹配，太大太小都伤效果。
                   碑帖训练数据用 sigma=15-30，这里取 25/255 ≈ 0.098。

    适合: 几乎所有平稳噪声，长期是传统方法的天花板
    不适合: 速度慢（一张 256x256 大约 1-3 秒）
    """
    import bm3d
    # bm3d 要求 (H, W) 或 (H, W, C) 都行，float
    img_clean = np.clip(img.astype(np.float64), 0, 1)
    out = bm3d.bm3d(img_clean, sigma_psd=sigma_psd)
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------- 4. 双边滤波（备选） ----------------
def denoise_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    """双边滤波：考虑空间距离 + 像素值相似度的加权平均。

    参数:
        img:         numpy float32, [0, 1]
        d:           邻域直径
        sigma_color: 颜色空间 sigma，越大越糊
        sigma_space: 坐标空间 sigma，越大邻域越大

    特点: 保边（笔画边缘）较好，但去噪强度不如 BM3D。
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out_u8 = cv2.bilateralFilter(img_u8, d, sigma_color, sigma_space)
    return out_u8.astype(np.float32) / 255.0


# ---------------- 统一调用入口 ----------------
DENOISERS = {
    'median':    denoise_median,
    'nlm':       denoise_nlm,
    'bm3d':      denoise_bm3d,
    'bilateral': denoise_bilateral,
}


def denoise(img, method='bm3d', **kwargs):
    """通用调用入口。

    用法:
        out = denoise(noisy, method='bm3d')
        out = denoise(noisy, method='median', ksize=5)
    """
    if method not in DENOISERS:
        raise ValueError(f"未知方法 {method}, 可选: {list(DENOISERS.keys())}")
    return DENOISERS[method](img, **kwargs)


if __name__ == "__main__":
    """自测：用一张测试图跑一遍所有方法"""
    import os
    import time
    from utils.metrics import calc_psnr, calc_ssim

    # 找一张测试图
    test_path = "data/test/noisy/test_0000.png"
    gt_path = "data/test/clean/test_0000.png"
    if not os.path.exists(test_path):
        print(f"测试图不存在: {test_path}")
        exit(1)

    noisy = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    clean = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    print(f"测试图: {test_path}, shape={noisy.shape}")
    print(f"原始 (noisy vs clean): "
          f"PSNR={calc_psnr(noisy, clean):.2f}dB  "
          f"SSIM={calc_ssim(noisy, clean):.4f}")
    print()

    for name in DENOISERS:
        t0 = time.time()
        out = denoise(noisy, method=name)
        dt = time.time() - t0
        psnr = calc_psnr(out, clean)
        ssim = calc_ssim(out, clean)
        print(f"  {name:10s} | PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  "
              f"耗时={dt*1000:.0f}ms")
