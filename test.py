"""
测试脚本
--------
用法:
    python test.py --checkpoint checkpoints/best.pth --input data/test/noisy --output results/

功能:
    1. 加载训练好的模型
    2. 对测试图逐张去噪
    3. 计算 PSNR / SSIM（如果有 ground truth）
    4. 保存去噪结果
"""

import os
import argparse
import numpy as np
import cv2
import torch

from models.unet import UNet
from utils.metrics import calc_psnr, calc_ssim


def denoise_image(model, img, device, tile_size=256):
    """对一张完整图片做去噪。
    
    如果图太大（显存放不下），用 tiling 策略：
        把大图切成小块，分别去噪，再拼回来。
        块与块之间有 overlap，拼接时取中间部分，避免边缘不连续。
    
    参数:
        model: 训练好的模型
        img: numpy (H, W), 范围[0, 1]
        device: cuda/cpu
        tile_size: 切块大小
    """
    model.eval()
    h, w = img.shape
    
    # 小图直接整张推理
    if h <= tile_size and w <= tile_size:
        # padding 到能被16整除（U-Net 有4次下采样 = 缩小16倍）
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        img_pad = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        tensor = torch.from_numpy(img_pad[np.newaxis, np.newaxis, :, :]).to(device)
        with torch.no_grad():
            output = model(tensor)
        result = output.squeeze().cpu().numpy()
        return result[:h, :w]
    
    # 大图 tiling
    overlap = 32
    stride = tile_size - overlap * 2
    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            # 计算 tile 坐标
            bottom = min(top + tile_size, h)
            right = min(left + tile_size, w)
            top_actual = bottom - tile_size if bottom - top < tile_size else top
            left_actual = right - tile_size if right - left < tile_size else left
            
            tile = img[top_actual:top_actual+tile_size, left_actual:left_actual+tile_size]
            
            # padding
            th, tw = tile.shape
            pad_h = (16 - th % 16) % 16
            pad_w = (16 - tw % 16) % 16
            tile_pad = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            tensor = torch.from_numpy(tile_pad[np.newaxis, np.newaxis, :, :]).to(device)
            with torch.no_grad():
                out_tile = model(tensor)
            out_tile = out_tile.squeeze().cpu().numpy()[:th, :tw]
            
            # 写回（overlap 区域会被多次写入，最后除以 weight 取平均）
            output[top_actual:top_actual+tile_size, left_actual:left_actual+tile_size] += out_tile
            weight[top_actual:top_actual+tile_size, left_actual:left_actual+tile_size] += 1
    
    output /= np.maximum(weight, 1)
    return np.clip(output, 0, 1)


def main():
    parser = argparse.ArgumentParser(description='碑帖去噪 - 测试')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径')
    parser.add_argument('--input', type=str, required=True, help='输入图片/文件夹')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth文件夹（可选，有的话算指标）')
    parser.add_argument('--output', type=str, default='results/', help='输出文件夹')
    parser.add_argument('--base_ch', type=int, default=32, help='模型base_ch（必须与训练一致）')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = UNet(in_channels=1, out_channels=1, base_ch=args.base_ch, residual=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载完成: {args.checkpoint}")
    
    # 收集测试图
    if os.path.isfile(args.input):
        input_files = [args.input]
    else:
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        input_files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                      if f.lower().endswith(valid_ext)]
    
    os.makedirs(args.output, exist_ok=True)
    print(f"共 {len(input_files)} 张待处理\n")
    
    psnr_list = []
    ssim_list = []
    
    for fpath in input_files:
        fname = os.path.basename(fpath)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # 去噪
        denoised = denoise_image(model, img, device)
        
        # 保存
        out_path = os.path.join(args.output, f"denoised_{fname}")
        cv2.imwrite(out_path, (denoised * 255).astype(np.uint8))
        
        # 算指标（如果有 GT）
        if args.gt:
            gt_path = os.path.join(args.gt, fname)
            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                psnr = calc_psnr(denoised, gt)
                ssim = calc_ssim(denoised, gt)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                print(f"  {fname}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
            else:
                print(f"  {fname}: 已去噪 (无GT)")
        else:
            print(f"  {fname}: 已去噪")
    
    # 汇总
    if psnr_list:
        print(f"\n{'='*40}")
        print(f"平均 PSNR: {np.mean(psnr_list):.2f} dB")
        print(f"平均 SSIM: {np.mean(ssim_list):.4f}")
    
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
