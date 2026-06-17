"""
DnCNN 训练脚本
==============
和 train.py（U-Net 训练）逻辑几乎一样，把模型换成 DnCNN。

用法（服务器上跑）：
    python train_dncnn.py --config configs/dncnn.yaml

本机冒烟测试：
    python train_dncnn.py --config configs/dncnn_smoke.yaml
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.dncnn import DnCNN
from utils.dataset import DenoisingDataset
from utils.metrics import AverageMeter, calc_psnr


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=50):
    model.train()
    loss_meter = AverageMeter()
    for i, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        loss = criterion(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), noisy.size(0))
        if (i + 1) % log_every == 0:
            print(f"    [Batch {i+1}/{len(loader)}] Loss: {loss_meter.val:.6f} "
                  f"(avg: {loss_meter.avg:.6f})")
    return loss_meter.avg


def main():
    parser = argparse.ArgumentParser(description='碑帖去噪 - DnCNN 训练')
    parser.add_argument('--config', type=str, default='configs/dncnn.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 数据集（和 U-Net 共用）
    ds = DenoisingDataset(
        data_dir=cfg['data']['train_dir'],
        patch_size=cfg['data']['patch_size'],
        noise_level=cfg['data']['noise_level'],
        augment=True,
    )
    loader = DataLoader(
        ds, batch_size=cfg['train']['batch_size'],
        shuffle=True, num_workers=cfg['data']['num_workers'],
        pin_memory=True, drop_last=True,
    )
    print(f"训练: {len(ds)} patches/epoch, batch={cfg['train']['batch_size']}")

    # DnCNN 模型
    model = DnCNN(
        in_channels=cfg['model']['in_channels'],
        out_channels=cfg['model']['out_channels'],
        num_layers=cfg['model']['num_layers'],
        num_features=cfg['model']['num_features'],
        residual=cfg['model']['residual'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DnCNN 参数量: {n_params:.3f} M ({cfg['model']['num_layers']} layers)")

    # L1 Loss + Adam + Cosine LR（和 U-Net 训练保持一致，公平对比）
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg['train']['epochs'], eta_min=1e-6
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"从 epoch {start_epoch} 恢复")

    ckpt_dir = cfg['output']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float('inf')
    print(f"\n{'='*50}")
    print(f"DnCNN 训练开始! 共 {cfg['train']['epochs']} epoch")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, cfg['train']['epochs']):
        lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch+1}/{cfg['train']['epochs']}] lr={lr:.6f}")
        train_loss = train_one_epoch(
            model, loader, optimizer, criterion, device,
            log_every=cfg['train'].get('log_every', 50)
        )
        scheduler.step()
        print(f"  -> Loss: {train_loss:.6f}")

        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss

        if (epoch + 1) % cfg['train']['save_every'] == 0 or is_best:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }
            if (epoch + 1) % cfg['train']['save_every'] == 0:
                path = os.path.join(ckpt_dir, f'dncnn_epoch_{epoch+1}.pth')
                torch.save(ckpt, path)
                print(f"  -> Saved: {path}")
            if is_best:
                path = os.path.join(ckpt_dir, 'dncnn_best.pth')
                torch.save(ckpt, path)
                print(f"  -> Best! (loss={best_loss:.6f})")
        print()

    print("DnCNN 训练完成!")


if __name__ == "__main__":
    main()
