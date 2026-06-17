"""
对比结果可视化（柱状图）
========================
读取 results/compare/summary.csv，画出 PSNR / SSIM / 耗时柱状图。
论文/答辩 PPT 里直接贴。

用法:
    python plot_results.py
    python plot_results.py --csv results/compare/summary.csv --out_dir results/figures
"""

import os
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 中文字体（论文/答辩多半要中文标题）
plt.rcParams['axes.unicode_minus'] = False
for cf in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS']:
    try:
        plt.rcParams['font.sans-serif'] = [cf]
        break
    except Exception:
        pass


# 方法的"友好显示名"
DISPLAY_NAME = {
    'noisy':     'Noisy (input)',
    'median':    'Median',
    'nlm':       'NLM',
    'bilateral': 'Bilateral',
    'bm3d':      'BM3D',
    'dncnn':     'DnCNN',
    'unet':      'U-Net',
    'unet_cbam': 'U-Net+CBAM (Ours)',
}

# 颜色：传统冷色调，深度方法暖色调，本方法红色突出
COLOR_MAP = {
    'noisy':     '#9e9e9e',
    'median':    '#bdbdbd',
    'nlm':       '#90a4ae',
    'bilateral': '#78909c',
    'bm3d':      '#607d8b',
    'dncnn':     '#42a5f5',
    'unet':      '#26a69a',
    'unet_cbam': '#e53935',
}


def parse_csv(csv_path):
    """读取 summary.csv -> {method: {'psnr': [...], 'ssim': [...], 'time': [...]}}"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{csv_path} 是空的")

    fields = list(rows[0].keys())
    methods = []
    seen = set()
    for f in fields:
        if f.endswith('_psnr'):
            m = f[:-len('_psnr')]
            if m not in seen:
                methods.append(m)
                seen.add(m)

    data = {m: {'psnr': [], 'ssim': [], 'time': []} for m in methods}
    for r in rows:
        for m in methods:
            data[m]['psnr'].append(float(r[f'{m}_psnr']))
            data[m]['ssim'].append(float(r[f'{m}_ssim']))
            data[m]['time'].append(float(r[f'{m}_time']))
    return methods, data


def bar_chart(methods, values, ylabel, title, out_path,
              show_values=True, ylim=None, errs=None):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    labels = [DISPLAY_NAME.get(m, m) for m in methods]
    colors = [COLOR_MAP.get(m, '#666') for m in methods]

    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
    if errs is not None:
        ax.errorbar(x, values, yerr=errs, fmt='none',
                    ecolor='black', capsize=3, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_values:
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f'{v:.2f}' if v >= 1 else f'{v:.4f}',
                    ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


def grouped_bar(methods, psnr_vals, ssim_vals, out_path):
    """PSNR / SSIM 双轴对比图。"""
    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=150)
    labels = [DISPLAY_NAME.get(m, m) for m in methods]
    x = np.arange(len(methods))
    width = 0.4

    bars1 = ax1.bar(x - width / 2, psnr_vals, width,
                    color='#42a5f5', edgecolor='black', linewidth=0.5,
                    label='PSNR (dB)')
    ax1.set_ylabel('PSNR (dB)', color='#1565c0')
    ax1.tick_params(axis='y', labelcolor='#1565c0')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, ssim_vals, width,
                    color='#ef5350', edgecolor='black', linewidth=0.5,
                    label='SSIM')
    ax2.set_ylabel('SSIM', color='#c62828')
    ax2.tick_params(axis='y', labelcolor='#c62828')
    ax2.set_ylim(0, 1.02)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right')
    ax1.set_title('Denoising Methods Comparison')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    for b, v in zip(bars1, psnr_vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f'{v:.1f}', ha='center', va='bottom', fontsize=8,
                 color='#1565c0')
    for b, v in zip(bars2, ssim_vals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8,
                 color='#c62828')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


def loss_curve_from_log(log_path, out_path):
    """可选：从训练日志解析 loss 曲线。
    匹配 'Loss: 0.xxx' / 'Train Loss: 0.xxx' 等。
    """
    if not os.path.exists(log_path):
        return False
    import re
    pat = re.compile(r'Loss[:\s]+([0-9]*\.?[0-9]+)')
    losses = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 只抓每个 epoch 的总 loss 行（含 -> 或 →）
            if '->' in line or '→' in line:
                m = pat.search(line)
                if m:
                    losses.append(float(m.group(1)))
    if not losses:
        return False
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(range(1, len(losses) + 1), losses, marker='o', markersize=3,
            color='#1976d2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Loss')
    ax.set_title('Training Loss Curve')
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='results/compare/summary.csv')
    parser.add_argument('--out_dir', type=str, default='results/figures')
    parser.add_argument('--log', type=str, default=None,
                        help='可选: 训练日志文件，画 loss 曲线')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    methods, data = parse_csv(args.csv)
    print(f"读取 {args.csv}，方法: {methods}")

    psnr_avg = [np.mean(data[m]['psnr']) for m in methods]
    ssim_avg = [np.mean(data[m]['ssim']) for m in methods]
    psnr_std = [np.std(data[m]['psnr']) for m in methods]
    ssim_std = [np.std(data[m]['ssim']) for m in methods]
    time_avg = [np.mean(data[m]['time']) * 1000 for m in methods]  # ms

    # 1. PSNR 柱状图
    bar_chart(methods, psnr_avg,
              ylabel='PSNR (dB) ↑',
              title='Denoising Methods - PSNR Comparison',
              out_path=os.path.join(args.out_dir, 'psnr_bar.png'),
              errs=psnr_std,
              ylim=(min(psnr_avg) - 3, max(psnr_avg) + 3))

    # 2. SSIM 柱状图
    bar_chart(methods, ssim_avg,
              ylabel='SSIM ↑',
              title='Denoising Methods - SSIM Comparison',
              out_path=os.path.join(args.out_dir, 'ssim_bar.png'),
              errs=ssim_std,
              ylim=(0, 1.05))

    # 3. PSNR + SSIM 双轴对比
    grouped_bar(methods, psnr_avg, ssim_avg,
                out_path=os.path.join(args.out_dir, 'psnr_ssim_combined.png'))

    # 4. 推理耗时（log 轴）
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    labels = [DISPLAY_NAME.get(m, m) for m in methods]
    colors = [COLOR_MAP.get(m, '#666') for m in methods]
    bars = ax.bar(np.arange(len(methods)), time_avg, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Inference Time per Image (ms, log scale)')
    ax.set_title('Inference Speed Comparison')
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.4)
    for b, v in zip(bars, time_avg):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f'{v:.1f}ms' if v >= 1 else f'{v:.2f}ms',
                ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'inference_time.png'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {os.path.join(args.out_dir, 'inference_time.png')}")

    # 5. 可选: 训练 loss 曲线
    if args.log:
        loss_curve_from_log(args.log,
                            os.path.join(args.out_dir, 'loss_curve.png'))

    print(f"\n[Done] 全部图片保存到: {args.out_dir}")


if __name__ == "__main__":
    main()
