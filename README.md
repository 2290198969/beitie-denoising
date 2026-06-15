# 碑帖图像去噪 (Rubbing Image Denoising)

基于深度学习的碑帖/拓片图像去噪系统。本科毕业设计项目。

## 方法概述

使用改进的 U-Net 网络，通过残差学习策略去除碑帖图像中的噪声（高斯噪声、椒盐噪声、拓印纹理噪声）。

## 项目结构

```
beitie-denoising/
├── data/                # 数据集
│   ├── raw/             # 干净训练图（书法/碑帖扫描件）
│   ├── noisy/           # 带噪图（合成生成，或真实拓片）
│   └── test/            # 测试集 (noisy/ + clean/ 子文件夹)
├── models/              # 网络模型
│   └── unet.py          # U-Net 去噪网络
├── utils/               # 工具函数
│   ├── noise.py         # 噪声生成（高斯/椒盐/拓印纹理）
│   ├── dataset.py       # PyTorch Dataset 类
│   └── metrics.py       # PSNR / SSIM 评价指标
├── train.py             # 训练脚本
├── test.py              # 测试/推理脚本
├── configs/
│   └── default.yaml     # 训练配置
├── checkpoints/         # 保存的模型权重
├── results/             # 去噪结果图
└── README.md
```

## 环境配置

```bash
conda create -n beitie python=3.10 -y
conda activate beitie
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy<2 pillow opencv-python matplotlib tqdm pyyaml scikit-image tensorboard
```

## 快速开始

### 1. 准备数据
将干净的书法/碑帖图片放入 `data/raw/`

### 2. 训练
```bash
python train.py --config configs/default.yaml
```

### 3. 测试
```bash
python test.py --checkpoint checkpoints/best.pth --input data/test/noisy --gt data/test/clean --output results/
```

## 评价指标

| 指标 | 含义 | 越高越好 |
|------|------|---------|
| PSNR | 峰值信噪比 (dB) | ✓ |
| SSIM | 结构相似性 [0,1] | ✓ |

## 硬件要求

- GPU: NVIDIA RTX 4060 或以上 (8GB显存)
- CUDA: 12.1+
- RAM: 16GB+
