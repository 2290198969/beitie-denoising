"""
训练脚本
--------
用法:
    python train.py --config configs/default.yaml

功能:
    1. 加载配置
    2. 创建 Dataset + DataLoader
    3. 创建 U-Net 模型
    4. 训练循环（每 epoch 跑一遍所有 patch）
    5. 定期保存 checkpoint
    6. TensorBoard 记录 loss 曲线
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from models.unet import UNet
from utils.dataset import DenoisingDataset
from utils.metrics import AverageMeter, calc_psnr


def load_config(path):
    """加载 YAML 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, criterion, device, log_every=50):
    """训练一个 epoch。
    
    返回: 平均 loss
    """
    model.train()
    loss_meter = AverageMeter()
    
    for i, (noisy, clean) in enumerate(dataloader):
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        # 前向传播
        output = model(noisy)
        loss = criterion(output, clean)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), noisy.size(0))
        
        if (i + 1) % log_every == 0:
            print(f"    [Batch {i+1}/{len(dataloader)}] Loss: {loss_meter.val:.6f} (avg: {loss_meter.avg:.6f})")
    
    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    """验证集评估。返回: (avg_loss, avg_psnr)"""
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    
    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            output = model(noisy)
            loss = criterion(output, clean)
            
            loss_meter.update(loss.item(), noisy.size(0))
            
            # 逐张算 PSNR
            for j in range(output.size(0)):
                psnr = calc_psnr(output[j], clean[j])
                psnr_meter.update(psnr)
    
    return loss_meter.avg, psnr_meter.avg


def main():
    parser = argparse.ArgumentParser(description='碑帖去噪 - 训练')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 数据集
    train_dataset = DenoisingDataset(
        data_dir=cfg['data']['train_dir'],
        patch_size=cfg['data']['patch_size'],
        noise_level=cfg['data']['noise_level'],
        augment=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"训练集: {len(train_dataset)} patches/epoch, batch_size={cfg['train']['batch_size']}")
    
    # 模型
    model = UNet(
        in_channels=cfg['model']['in_channels'],
        out_channels=cfg['model']['out_channels'],
        base_ch=cfg['model']['base_ch'],
        residual=cfg['model']['residual'],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {n_params:.2f} M")
    
    # 损失函数（L1 Loss，对去噪任务比 MSE 好）
    # 为什么用 L1 不用 MSE(L2)？
    #   MSE 对大误差惩罚重 → 模型倾向输出"模糊"的平均值
    #   L1 对各种误差一视同仁 → 输出更锐利
    #   碑帖去噪要保持笔画锐度，L1 更合适
    criterion = nn.L1Loss()
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )
    
    # 学习率调度器
    if cfg['train']['lr_scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'], eta_min=1e-6)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从 epoch {start_epoch} 恢复训练")
    
    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(cfg['output']['log_dir'])
        use_tb = True
        print(f"TensorBoard 日志: {cfg['output']['log_dir']}/")
    except ImportError:
        use_tb = False
        print("未安装 tensorboard，跳过日志记录")
    
    # 创建输出目录
    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['output']['result_dir'], exist_ok=True)
    
    # ========== 训练循环 ==========
    best_loss = float('inf')
    print(f"\n{'='*50}")
    print(f"开始训练! 共 {cfg['train']['epochs']} 个 epoch")
    print(f"{'='*50}\n")
    
    for epoch in range(start_epoch, cfg['train']['epochs']):
        print(f"[Epoch {epoch+1}/{cfg['train']['epochs']}] lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            log_every=cfg['train']['log_every']
        )
        
        # 学习率衰减
        scheduler.step()
        
        print(f"  → Train Loss: {train_loss:.6f}")
        
        # TensorBoard
        if use_tb:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存 checkpoint
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
            # 定期存
            if (epoch + 1) % cfg['train']['save_every'] == 0:
                path = os.path.join(cfg['output']['checkpoint_dir'], f'epoch_{epoch+1}.pth')
                torch.save(ckpt, path)
                print(f"  → Checkpoint saved: {path}")
            # 最佳模型单独存
            if is_best:
                path = os.path.join(cfg['output']['checkpoint_dir'], 'best.pth')
                torch.save(ckpt, path)
                print(f"  → Best model updated! (loss={best_loss:.6f})")
        
        print()
    
    print("训练完成!")
    if use_tb:
        writer.close()


if __name__ == "__main__":
    main()
