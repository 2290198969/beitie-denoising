"""
数据集类
--------
PyTorch 训练需要 Dataset 类来"喂数据"。

工作流程:
    1. 读取 data/raw/ 下的干净书法/碑帖图片
    2. 在线(on-the-fly)加噪声生成训练对 (noisy, clean)
    3. 随机裁剪 patch（不用整张图训练，太大了显存放不下）
    4. 数据增强（翻转、旋转）

为什么用"在线加噪"而不是"提前生成好存硬盘"？
    - 每次 epoch 加的噪声不一样 → 数据多样性无限大 → 不容易过拟合
    - 不占硬盘空间
    - 这是学术界主流做法（DnCNN、FFDNet、SwinIR 都这样）
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.noise import add_composite_noise, add_gaussian_noise


class DenoisingDataset(Dataset):
    """碑帖去噪训练数据集。
    
    参数:
        data_dir: 存放干净图的文件夹路径
        patch_size: 裁剪的 patch 大小（默认128，显存够可以开256）
        noise_level: 噪声等级 'light'/'medium'/'heavy'/'random'
        augment: 是否做数据增强
    """
    def __init__(self, data_dir, patch_size=128, noise_level='random', augment=True):
        super().__init__()
        self.patch_size = patch_size
        self.noise_level = noise_level
        self.augment = augment
        
        # 扫描所有图片文件
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_paths = []
        for f in sorted(os.listdir(data_dir)):
            if f.lower().endswith(valid_ext):
                self.image_paths.append(os.path.join(data_dir, f))
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"在 {data_dir} 下没找到任何图片！请先放入训练数据。")
        
        print(f"[Dataset] 找到 {len(self.image_paths)} 张训练图片 in {data_dir}")

    def __len__(self):
        # 每张图可以裁出很多 patch，所以虚拟放大数据量
        # 实际一个 epoch 过多少 patch 取决于这里的返回值
        return len(self.image_paths) * 20  # 每张图每 epoch 采样20次

    def __getitem__(self, idx):
        # 取真实图片索引（取模循环）
        img_idx = idx % len(self.image_paths)
        
        # 读图 → 灰度 → 归一化到[0,1]
        img = cv2.imread(self.image_paths[img_idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"图片读取失败: {self.image_paths[img_idx]}")
        img = img.astype(np.float32) / 255.0
        
        # 随机裁剪 patch
        img = self._random_crop(img)
        
        # 数据增强
        if self.augment:
            img = self._augment(img)
        
        # 添加噪声
        if self.noise_level == 'random':
            level = random.choice(['light', 'medium', 'heavy'])
        else:
            level = self.noise_level
        noisy = add_composite_noise(img.copy(), level)
        
        # numpy → torch tensor, shape: (1, H, W)
        clean_tensor = torch.from_numpy(img[np.newaxis, :, :])   # (1, H, W)
        noisy_tensor = torch.from_numpy(noisy[np.newaxis, :, :]) # (1, H, W)
        
        return noisy_tensor, clean_tensor

    def _random_crop(self, img):
        """随机裁剪 patch_size × patch_size 区域"""
        h, w = img.shape[:2]
        ps = self.patch_size
        
        # 如果图比 patch 小，先 resize 放大
        if h < ps or w < ps:
            scale = max(ps / h, ps / w) * 1.1
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]
        
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return img[top:top+ps, left:left+ps]

    def _augment(self, img):
        """简单数据增强：随机翻转 + 旋转90°"""
        # 水平翻转 (50%概率)
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
        # 垂直翻转 (50%概率)
        if random.random() > 0.5:
            img = np.flipud(img).copy()
        # 旋转90° (50%概率)
        if random.random() > 0.5:
            img = np.rot90(img).copy()
        return img


class TestDataset(Dataset):
    """测试数据集 —— 用成对的 (noisy, clean) 图测评。
    
    文件夹结构:
        test_dir/
            noisy/   ← 带噪声图
            clean/   ← 对应的干净图（文件名必须一样！）
    """
    def __init__(self, test_dir):
        self.noisy_dir = os.path.join(test_dir, 'noisy')
        self.clean_dir = os.path.join(test_dir, 'clean')
        
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.filenames = [f for f in sorted(os.listdir(self.noisy_dir))
                         if f.lower().endswith(valid_ext)]
        print(f"[TestDataset] 找到 {len(self.filenames)} 张测试图片")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        noisy = cv2.imread(os.path.join(self.noisy_dir, fname), cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(os.path.join(self.clean_dir, fname), cv2.IMREAD_GRAYSCALE)
        
        noisy = noisy.astype(np.float32) / 255.0
        clean = clean.astype(np.float32) / 255.0
        
        noisy_tensor = torch.from_numpy(noisy[np.newaxis, :, :])
        clean_tensor = torch.from_numpy(clean[np.newaxis, :, :])
        
        return noisy_tensor, clean_tensor, fname


if __name__ == "__main__":
    """自测"""
    print("=== 数据集类自测 ===")
    print("这个文件定义了 DenoisingDataset 和 TestDataset")
    print("当 data/raw/ 有图片后，可以这样用:")
    print()
    print("  from utils.dataset import DenoisingDataset")
    print("  ds = DenoisingDataset('data/raw', patch_size=128)")
    print("  noisy, clean = ds[0]")
    print("  print(noisy.shape)  # torch.Size([1, 128, 128])")
    print()
    print("⚠️ 目前 data/raw/ 还没图片，下一步我们要准备数据集！")
