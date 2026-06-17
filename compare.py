"""
对比实验脚本 —— 第 2 周核心产出
================================
在 data/test/ 测试集（50 对图）上跑全部去噪方法，输出统一对比表 + CSV。

支持的方法：
    1. noisy      - 不去噪（基线，看带噪图本身的 PSNR/SSIM）
    2. median     - 中值滤波
    3. nlm        - 非局部均值 (OpenCV)
    4. bilateral  - 双边滤波
    5. bm3d       - BM3D (传统天花板)
    6. dncnn      - DnCNN (深度学习经典基线)，需要 checkpoint
    7. unet       - U-Net (我们的方法)，需要 checkpoint

用法:
    # 跑所有传统方法 + U-Net
    python compare.py --methods median nlm bilateral bm3d unet \
        --unet_ckpt checkpoints/best.pth

    # 加上 DnCNN
    python compare.py --methods median nlm bm3d dncnn unet \
        --unet_ckpt checkpoints/best.pth \
        --dncnn_ckpt checkpoints/dncnn_best.pth

    # 同时保存可视化对比图（每张测试图一张大图）
    python compare.py --methods bm3d unet --unet_ckpt checkpoints/best.pth \
        --save_vis

输出:
    results/compare/<method>/test_XXXX.png    去噪结果
    results/compare/summary.csv               逐图指标
    results/compare/summary.md                平均指标表（贴报告用）
    results/compare/vis/test_XXXX.png         可视化对比图（开 --save_vis）
"""

import os
import csv
import argparse
import time
from collections import defaultdict

import numpy as np
import cv2
import torch

from utils.metrics import calc_psnr, calc_ssim
from utils.baselines import denoise_median, denoise_nlm, denoise_bilateral, denoise_bm3d


# ============== 深度模型推理（带 tiling，复用 test.py 的逻辑）==============

def _pad_to_16(img):
    """U-Net/DnCNN 输入需要尺寸能被 16 整除（4 次下采样）。
    DnCNN 其实没下采样，但保持一致逻辑无所谓。"""
    h, w = img.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    return img, h, w


@torch.no_grad()
def _model_infer(model, img, device):
    """模型推理 1 张图。img: numpy (H, W) [0,1]"""
    model.eval()
    img_pad, h, w = _pad_to_16(img)
    tensor = torch.from_numpy(img_pad[np.newaxis, np.newaxis]).to(device)
    out = model(tensor).squeeze().cpu().numpy()
    return np.clip(out[:h, :w], 0, 1).astype(np.float32)


def load_unet(ckpt_path, device, base_ch=32):
    from models.unet import UNet
    model = UNet(in_channels=1, out_channels=1, base_ch=base_ch, residual=True).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    print(f"[UNet] 加载 {ckpt_path}")
    return model


def load_dncnn(ckpt_path, device, num_layers=17, num_features=64):
    from models.dncnn import DnCNN
    model = DnCNN(in_channels=1, out_channels=1,
                  num_layers=num_layers, num_features=num_features,
                  residual=True).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    print(f"[DnCNN] 加载 {ckpt_path}")
    return model


# ============== 主流程 ==============

def make_method_callable(method, args, device):
    """把字符串方法名变成一个统一接口的函数 fn(noisy_img) -> denoised_img。"""
    if method == 'noisy':
        return lambda img: img.copy()
    if method == 'median':
        return lambda img: denoise_median(img, ksize=args.median_ksize)
    if method == 'nlm':
        return lambda img: denoise_nlm(img, h=args.nlm_h)
    if method == 'bilateral':
        return lambda img: denoise_bilateral(img)
    if method == 'bm3d':
        return lambda img: denoise_bm3d(img, sigma_psd=args.bm3d_sigma / 255.0)
    if method == 'unet':
        if not args.unet_ckpt:
            raise ValueError("使用 unet 必须传 --unet_ckpt")
        model = load_unet(args.unet_ckpt, device, base_ch=args.unet_base_ch)
        return lambda img: _model_infer(model, img, device)
    if method == 'dncnn':
        if not args.dncnn_ckpt:
            raise ValueError("使用 dncnn 必须传 --dncnn_ckpt")
        model = load_dncnn(args.dncnn_ckpt, device,
                           num_layers=args.dncnn_layers,
                           num_features=args.dncnn_features)
        return lambda img: _model_infer(model, img, device)
    raise ValueError(f"未知方法 {method}")


def save_vis(out_path, fname, clean, noisy, results_per_method):
    """把所有方法的结果横排拼成一张大图保存。"""
    cols = [('clean', clean), ('noisy', noisy)] + list(results_per_method.items())
    h, w = clean.shape

    # 每张图上方留 32 像素写文字
    label_h = 28
    panel_h = h + label_h
    panel_w = w
    canvas = np.ones((panel_h, panel_w * len(cols)), dtype=np.uint8) * 255

    for idx, (name, im) in enumerate(cols):
        x0 = idx * panel_w
        canvas[label_h:label_h + h, x0:x0 + w] = (im * 255).astype(np.uint8)
        cv2.putText(canvas, name, (x0 + 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)


def main():
    parser = argparse.ArgumentParser(description="碑帖去噪 - 多方法对比实验")
    parser.add_argument('--test_dir', type=str, default='data/test',
                        help='测试集根目录，含 noisy/ 和 clean/ 子目录')
    parser.add_argument('--out_dir', type=str, default='results/compare',
                        help='输出目录')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['noisy', 'median', 'nlm', 'bilateral', 'bm3d', 'unet'],
                        help='要跑的方法')
    parser.add_argument('--save_vis', action='store_true',
                        help='额外保存可视化对比大图')
    parser.add_argument('--save_each', action='store_true', default=True,
                        help='每个方法的去噪结果也单独保存')

    # U-Net
    parser.add_argument('--unet_ckpt', type=str, default='checkpoints/best.pth')
    parser.add_argument('--unet_base_ch', type=int, default=32)

    # DnCNN
    parser.add_argument('--dncnn_ckpt', type=str, default=None)
    parser.add_argument('--dncnn_layers', type=int, default=17)
    parser.add_argument('--dncnn_features', type=int, default=64)

    # 传统方法超参
    parser.add_argument('--median_ksize', type=int, default=3)
    parser.add_argument('--nlm_h', type=float, default=10)
    parser.add_argument('--bm3d_sigma', type=float, default=25,
                        help='BM3D 假设的噪声 sigma (0~255 范围)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    noisy_dir = os.path.join(args.test_dir, 'noisy')
    clean_dir = os.path.join(args.test_dir, 'clean')
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    fnames = sorted(f for f in os.listdir(noisy_dir) if f.lower().endswith(valid_ext))
    print(f"测试集: {len(fnames)} 张图")
    print(f"方法: {args.methods}")
    print()

    # 准备所有方法的可调用函数
    fns = {m: make_method_callable(m, args, device) for m in args.methods}

    # 收集结果
    per_image_rows = []                        # CSV 用
    psnr_by_method = defaultdict(list)
    ssim_by_method = defaultdict(list)
    time_by_method = defaultdict(list)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_each:
        for m in args.methods:
            os.makedirs(os.path.join(args.out_dir, m), exist_ok=True)

    print(f"{'image':<18}" + ''.join(f"{m:>14}" for m in args.methods))

    for fname in fnames:
        noisy = cv2.imread(os.path.join(noisy_dir, fname),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        clean = cv2.imread(os.path.join(clean_dir, fname),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        row = {'filename': fname}
        results_for_vis = {}

        line = f"{fname:<18}"
        for m in args.methods:
            t0 = time.time()
            out = fns[m](noisy)
            dt = time.time() - t0

            psnr = calc_psnr(out, clean)
            ssim = calc_ssim(out, clean)

            row[f'{m}_psnr'] = psnr
            row[f'{m}_ssim'] = ssim
            row[f'{m}_time'] = dt

            psnr_by_method[m].append(psnr)
            ssim_by_method[m].append(ssim)
            time_by_method[m].append(dt)

            if args.save_each and m != 'noisy':
                cv2.imwrite(os.path.join(args.out_dir, m, fname),
                            (out * 255).astype(np.uint8))

            results_for_vis[m] = out
            line += f"  P{psnr:5.2f}/S{ssim:.3f}"

        per_image_rows.append(row)
        print(line)

        if args.save_vis:
            # 可视化里去掉 'noisy' 重复（已在第二列），保留其它方法
            vis = {k: v for k, v in results_for_vis.items() if k != 'noisy'}
            save_vis(os.path.join(args.out_dir, 'vis', fname),
                     fname, clean, noisy, vis)

    # ===== CSV =====
    csv_path = os.path.join(args.out_dir, 'summary.csv')
    fieldnames = ['filename']
    for m in args.methods:
        fieldnames += [f'{m}_psnr', f'{m}_ssim', f'{m}_time']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    # ===== Markdown 平均表 =====
    md_path = os.path.join(args.out_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 碑帖去噪 - 方法对比 ({len(fnames)} 张测试图)\n\n")
        f.write("| 方法 | PSNR (dB) ↑ | SSIM ↑ | 平均耗时 (ms) |\n")
        f.write("|------|-------------|--------|---------------|\n")
        for m in args.methods:
            psnr = np.mean(psnr_by_method[m])
            ssim = np.mean(ssim_by_method[m])
            t_ms = np.mean(time_by_method[m]) * 1000
            f.write(f"| {m} | {psnr:.2f} | {ssim:.4f} | {t_ms:.1f} |\n")

    # ===== 终端汇总 =====
    print()
    print("=" * 60)
    print(f"{'方法':<12}{'PSNR(dB)':>12}{'SSIM':>10}{'耗时(ms)':>14}")
    print("-" * 60)
    for m in args.methods:
        psnr = np.mean(psnr_by_method[m])
        ssim = np.mean(ssim_by_method[m])
        t_ms = np.mean(time_by_method[m]) * 1000
        print(f"{m:<12}{psnr:>12.2f}{ssim:>10.4f}{t_ms:>14.1f}")
    print("=" * 60)
    print(f"\n[Done] CSV: {csv_path}")
    print(f"[Done] MD:  {md_path}")
    if args.save_vis:
        print(f"[Done] Vis: {os.path.join(args.out_dir, 'vis')}")


if __name__ == "__main__":
    main()
