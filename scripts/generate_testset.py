"""
生成测试集
----------
测试集需要 (noisy, clean) 配对，这样才能算 PSNR/SSIM。

输出结构:
    data/test/
        clean/   ← 干净原图
        noisy/   ← 对应的带噪版
"""

import os
import random
import numpy as np
import cv2
from PIL import Image

# 复用生成脚本的函数
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_data import generate_single_char_image, generate_multi_char_image, generate_calligraphy_style, invert_image
from utils.noise import add_composite_noise


def main():
    random.seed(123)
    np.random.seed(123)
    
    num = 50  # 测试集50张够了
    size = 256
    
    clean_dir = "data/test/clean"
    noisy_dir = "data/test/noisy"
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    
    print(f"生成 {num} 对测试图 (clean + noisy)...")
    
    for i in range(num):
        # 生成干净图
        if random.random() < 0.6:
            img, _ = generate_single_char_image(size)
        else:
            img = generate_multi_char_image(size)
        
        img = generate_calligraphy_style(img)
        if random.random() > 0.5:
            img = invert_image(img)
        
        # 转 numpy
        clean = np.array(img).astype(np.float32) / 255.0
        
        # 加混合噪声
        noisy = add_composite_noise(clean.copy(), 'medium')
        
        # 保存
        fname = f"test_{i:04d}.png"
        cv2.imwrite(os.path.join(clean_dir, fname), (clean * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(noisy_dir, fname), (noisy * 255).astype(np.uint8))
    
    print(f"[Done] {num} 对图片已保存")
    print(f"  clean: {clean_dir}/")
    print(f"  noisy: {noisy_dir}/")


if __name__ == "__main__":
    main()
