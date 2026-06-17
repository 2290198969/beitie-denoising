"""
CBAM 注意力可视化
==================
通过 PyTorch hook 抽出 U-Net+CBAM 中各层的空间注意力权重，
用伪彩色（热力图）叠加在原图上，可视化"网络在关注哪里"。

为什么要做这个？
    论文/答辩里光说"加了注意力"不够，要拿出图来证明：
        网络确实把注意力放在了笔画区域，没浪费在空白背景上。
    这是非常有说服力的论据，也是答辩老师最容易问的。

输出（默认 results/attention_vis/）:
    test_XXXX_overlay.png   原图 + 注意力热力图叠加
    test_XXXX_panel.png     [原图 | 带噪 | 去噪 | 注意力] 并排
    test_XXXX_layers.png    多层注意力对比（看不同深度学到了什么）

用法:
    python visualize_attention.py \
        --ckpt checkpoints/unet_cbam_best.pth \
        --num 6
"""

import os
import argparse
import numpy as np
import cv2
import torch

from models.unet_cbam import UNetCBAM
from models.attention import SpatialAttention


# ------------ Hook 工具 ------------

class AttentionHook:
    """在 SpatialAttention 模块上注册 forward hook，抓取空间注意力 mask。

    SpatialAttention.forward 返回的是 x * weight (经过加权后的特征)，
    我们要的是 weight 本身（形状 (B, 1, H, W)）。
    所以拦截 SpatialAttention 内部 conv → sigmoid 的中间结果。
    """
    def __init__(self):
        self.maps = {}      # {layer_name: tensor}
        self.handles = []

    def _make_hook(self, name):
        def hook(module, inp, out):
            # SpatialAttention.forward 返回 x * weight
            # weight = sigmoid(conv(cat([mean, max])))
            # 这里我们再算一次 weight 存下来（不影响前向）
            x = inp[0]
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            cat = torch.cat([avg_out, max_out], dim=1)
            weight = torch.sigmoid(module.conv(cat))   # (B, 1, H, W)
            self.maps[name] = weight.detach().cpu()
        return hook

    def attach(self, model):
        """在所有 SpatialAttention 上挂 hook，按层名记录。"""
        # 按 U-Net 的层级命名（编码 inc/down1..4，解码 up1..4）
        # 我们遍历 model 的 named_modules，找出所有 SpatialAttention，
        # 用它在 model 中的 path 做名字。
        for path, mod in model.named_modules():
            if isinstance(mod, SpatialAttention):
                # path 形如 "inc.cbam.spatial_att"，简化成 "inc"/"down1"/...
                short = path.replace('.cbam.spatial_att', '')
                h = mod.register_forward_hook(self._make_hook(short))
                self.handles.append(h)
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ------------ 可视化工具 ------------

def overlay_heatmap(gray_img, heat, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """把单通道热力图（[0,1]）叠加到灰度图上。

    gray_img: (H, W) [0,1]
    heat:     (H, W) [0,1]
    alpha:    热力图占比
    """
    base = (np.stack([gray_img] * 3, axis=-1) * 255).astype(np.uint8)
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, colormap)
    overlay = cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0)
    return overlay


def normalize_map(m):
    """把 attention map 拉到 [0,1] 区间方便可视化。"""
    m = m.astype(np.float32)
    lo, hi = m.min(), m.max()
    if hi - lo < 1e-8:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def put_label(canvas, text, x, y, scale=0.5):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, 0, 1, cv2.LINE_AA)


@torch.no_grad()
def infer_with_attention(model, hook, noisy_np, device):
    """跑一次推理，同时把所有层的 attention map 存到 hook.maps。
    返回去噪结果 + attention dict（已 resize 到原图大小）。
    """
    h, w = noisy_np.shape
    # padding 到 16 整数倍
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    img_pad = np.pad(noisy_np, ((0, pad_h), (0, pad_w)), mode='reflect')
    tensor = torch.from_numpy(img_pad[np.newaxis, np.newaxis]).to(device)

    out = model(tensor).squeeze().cpu().numpy()
    out = np.clip(out[:h, :w], 0, 1).astype(np.float32)

    # attention map 全部 resize 到 (h, w)
    att_dict = {}
    for name, m in hook.maps.items():
        # m: (1, 1, h_layer, w_layer)
        a = m.squeeze().numpy()
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        att_dict[name] = a
    return out, att_dict


# ------------ 主流程 ------------

LAYER_ORDER = ['inc', 'down1', 'down2', 'down3', 'down4',
               'up1', 'up2', 'up3', 'up4']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/unet_cbam_best.pth')
    parser.add_argument('--test_dir', type=str, default='data/test')
    parser.add_argument('--out_dir', type=str, default='results/attention_vis')
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--num', type=int, default=6,
                        help='挑前几张测试图做可视化')
    parser.add_argument('--alpha', type=float, default=0.55,
                        help='热力图叠加透明度')
    parser.add_argument('--main_layer', type=str, default='inc',
                        help='主可视化采用哪层注意力（inc 最浅、和原图同分辨率，效果最直观）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    model = UNetCBAM(in_channels=1, out_channels=1,
                     base_ch=args.base_ch, residual=True).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()
    print(f"加载: {args.ckpt}")

    # 挂 hook
    hook = AttentionHook().attach(model)
    print(f"已挂载 {len(hook.handles)} 个 SpatialAttention hook")

    # 找测试图
    noisy_dir = os.path.join(args.test_dir, 'noisy')
    clean_dir = os.path.join(args.test_dir, 'clean')
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    fnames = sorted(f for f in os.listdir(noisy_dir) if f.lower().endswith(valid_ext))
    fnames = fnames[:args.num]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"将处理 {len(fnames)} 张图")
    print()

    for fname in fnames:
        noisy = cv2.imread(os.path.join(noisy_dir, fname),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        clean_path = os.path.join(clean_dir, fname)
        clean = (cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
                 .astype(np.float32) / 255.0) if os.path.exists(clean_path) else None

        denoised, att = infer_with_attention(model, hook, noisy, device)

        h, w = noisy.shape

        # ===== 1. 主叠加图：用 main_layer 的注意力 =====
        if args.main_layer in att:
            att_main = normalize_map(att[args.main_layer])
            # 用干净图做底图（演示效果最清楚），如果没干净图就用带噪图
            base_img = clean if clean is not None else noisy
            overlay = overlay_heatmap(base_img, att_main, alpha=args.alpha)
            cv2.imwrite(os.path.join(args.out_dir, f'{os.path.splitext(fname)[0]}_overlay.png'),
                        overlay)

        # ===== 2. 综合 panel: clean | noisy | denoised | attention overlay =====
        panel_cols = []
        labels = []
        if clean is not None:
            panel_cols.append((clean * 255).astype(np.uint8))
            labels.append('Clean')
        panel_cols.append((noisy * 255).astype(np.uint8))
        labels.append('Noisy')
        panel_cols.append((denoised * 255).astype(np.uint8))
        labels.append('Denoised')

        # attention 主层叠加（彩色）
        if args.main_layer in att:
            att_main = normalize_map(att[args.main_layer])
            overlay = overlay_heatmap(noisy, att_main, alpha=args.alpha)
            panel_cols.append(overlay)
            labels.append(f'Att({args.main_layer})')

        # 拼图
        label_h = 24
        # 灰度的转 BGR 才能和彩色 overlay 拼
        bgr_cols = [cv2.cvtColor(c, cv2.COLOR_GRAY2BGR) if c.ndim == 2 else c
                    for c in panel_cols]
        canvas = np.ones((h + label_h, w * len(bgr_cols), 3), dtype=np.uint8) * 255
        for i, (im, lab) in enumerate(zip(bgr_cols, labels)):
            canvas[label_h:label_h + h, i * w:(i + 1) * w] = im
            cv2.putText(canvas, lab, (i * w + 5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out_dir, f'{os.path.splitext(fname)[0]}_panel.png'),
                    canvas)

        # ===== 3. 多层注意力对比（看不同深度学到了什么）=====
        ordered = [n for n in LAYER_ORDER if n in att]
        if ordered:
            cols = []
            for name in ordered:
                a = normalize_map(att[name])
                ov = overlay_heatmap(noisy, a, alpha=args.alpha)
                cols.append((name, ov))
            label_h = 24
            canvas = np.ones((h + label_h, w * len(cols), 3), dtype=np.uint8) * 255
            for i, (name, im) in enumerate(cols):
                canvas[label_h:label_h + h, i * w:(i + 1) * w] = im
                cv2.putText(canvas, name, (i * w + 5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.out_dir, f'{os.path.splitext(fname)[0]}_layers.png'),
                        canvas)

        print(f"  {fname} 已生成: overlay / panel / layers")

    hook.detach()
    print(f"\n[Done] 输出: {args.out_dir}/")
    print("提示：")
    print("  *_panel.png  贴论文对比图")
    print("  *_overlay.png 答辩讲'网络在看笔画'用")
    print("  *_layers.png  讲不同深度学到不同尺度结构（浅层=笔锋，深层=整字）")


if __name__ == "__main__":
    main()
