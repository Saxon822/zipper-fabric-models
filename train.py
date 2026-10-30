#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zipper Fabric Segmentation Training & Fine-Tuning Pipeline
通用布带分割模型训练与微调脚本，支持从头训练、加载预训练权重微调，以及自动导出工业级 ONNX 模型。
"""

import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.fabric_unet import FabricUNet

class FabricDataset(Dataset):
    def __init__(self, data_dir, split="train", img_size=(512, 320), augment=True):
        self.img_dir = os.path.join(data_dir, split, "images")
        self.mask_dir = os.path.join(data_dir, split, "masks")
        self.img_size = img_size  # (height, width)
        self.augment = augment and (split == "train")

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"目录不存在: {self.img_dir}")

        self.filenames = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros(self.img_size, dtype=np.uint8)
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)

        # 调整至训练尺寸 (width, height)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        # 数据增强 (训练模式)
        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            if np.random.rand() > 0.5:
                # 随机亮度与对比度微调
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-15, 15)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # 归一化 [0, 1]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_tensor = (torch.from_numpy(mask).float().unsqueeze(0) > 127).float()

        return img_tensor, mask_tensor

class DiceBCELoss(nn.Module):
    """工业分割常用的复合损失函数: BCE + Dice Loss"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        return bce_loss + dice_loss

def calculate_metrics(pred, target, threshold=0.5):
    """计算 Dice 系数与 IoU"""
    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum().item()
    total = pred_flat.sum().item() + target_flat.sum().item()
    union = total - intersection

    dice = (2.0 * intersection) / (total + 1e-6)
    iou = intersection / (union + 1e-6)
    return dice, iou

def export_to_onnx(model, onnx_path, img_size=(512, 320), device="cpu"):
    """导出为通用工业级 ONNX 模型"""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 1, img_size[0], img_size[1], device=device)

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    try:
        # 优先使用经典兼容 exporter (打包单文件 .onnx，不分离 .data)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"}
            }
        )
    except Exception as e:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            dynamo=False,
            input_names=["input"],
            output_names=["output"]
        )
    print(f"[✓] 成功导出 ONNX 模型至: {onnx_path}")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[*] 训练设备: {device}")

    # 1. 准备数据集
    train_dataset = FabricDataset(args.data_dir, split="train", img_size=(args.img_height, args.img_width), augment=True)
    val_dataset = FabricDataset(args.data_dir, split="val", img_size=(args.img_height, args.img_width), augment=False)

    print(f"[*] 数据集加载: 训练集 {len(train_dataset)} 张, 验证集 {len(val_dataset)} 张")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 2. 构建模型
    model = FabricUNet(n_channels=1, n_classes=1, base_channels=args.base_channels)
    
    # 支持加载旧权重继续微调
    if args.weights and os.path.exists(args.weights):
        print(f"[*] 正在加载预训练权重进行微调: {args.weights}")
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state)

    model.to(device)

    # 3. 损失与优化器
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_dice = 0.0

    print("[*] 开始训练 / 微调循环...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # 验证评估
        model.eval()
        val_dice_sum, val_iou_sum = 0.0, 0.0
        with torch.no_grad():
            for val_imgs, val_masks in val_loader:
                val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)
                val_preds = model(val_imgs)
                d, i = calculate_metrics(val_preds, val_masks)
                val_dice_sum += d
                val_iou_sum += i

        val_dice = val_dice_sum / max(len(val_loader), 1)
        val_iou = val_iou_sum / max(len(val_loader), 1)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | 耗时: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        # 保存最优模型
        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(args.checkpoint_dir, "best_fabric_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  --> [★] 新最佳模型已保存: {best_path} (Dice: {best_dice:.4f})")

    # 导出 ONNX 模型
    if args.export_onnx:
        onnx_path = os.path.join(args.checkpoint_dir, "fabric_segmentation.onnx")
        # 加载最佳权重导出
        best_path = os.path.join(args.checkpoint_dir, "best_fabric_model.pth")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location="cpu"))
        export_to_onnx(model, onnx_path, (args.img_height, args.img_width), device="cpu")

    print("[✓] 训练流程全部顺利完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="布带检测通用模型训练与微调")
    parser.add_argument("--data_dir", type=str, default="dataset", help="标准数据集根目录")
    parser.add_argument("--weights", type=str, default=None, help="旧模型权重路径 (用于微调)")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--img_height", type=int, default=512, help="训练图像高度")
    parser.add_argument("--img_width", type=int, default=320, help="训练图像宽度")
    parser.add_argument("--base_channels", type=int, default=16, help="网络初始通道数 (轻量化)")
    parser.add_argument("--workers", type=int, default=0, help="数据读取多线程数")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="权重保存目录")
    parser.add_argument("--export_onnx", action="store_true", default=True, help="训练完成后自动导出 ONNX")
    args = parser.parse_args()

    train(args)
