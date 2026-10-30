#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zipper Fabric (布带) Dataset Preparation Tool
将原始拉链工程中的采样图与布带热力图提取、清洗并转换为标准分割数据集格式。
"""

import os
import random
import shutil
import argparse
import cv2
import numpy as np

def prepare_dataset(raw_dir, output_dir, val_ratio=0.2, max_samples=None, threshold=0, min_mask_ratio=0.15):
    source_dir = os.path.join(raw_dir, "source")
    heatmaps_dir = os.path.join(raw_dir, "heatmaps")

    if not os.path.exists(source_dir) or not os.path.exists(heatmaps_dir):
        raise FileNotFoundError(f"未在 {raw_dir} 中找到 source 或 heatmaps 目录")

    source_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    print(f"[*] 扫描到原始图像文件数: {len(source_files)}")

    pairs = []
    skipped_empty = 0
    for sf in source_files:
        stem = os.path.splitext(sf)[0]
        # 布带标注对应类别 1 (_1.png)
        hf = f"{stem}_1.png"
        hf_path = os.path.join(heatmaps_dir, hf)
        if os.path.exists(hf_path):
            # 检查掩码有效性 (过滤掉没有标注布带的假空样本)
            heat = cv2.imread(hf_path, cv2.IMREAD_GRAYSCALE)
            if heat is not None:
                mask_ratio = np.mean(heat > threshold)
                if mask_ratio >= min_mask_ratio:
                    pairs.append((os.path.join(source_dir, sf), hf_path, sf))
                else:
                    skipped_empty += 1

    print(f"[*] 过滤空白/底噪样本: {skipped_empty} 张，保留高质量布带标注对: {len(pairs)} 张")

    random.seed(42)
    random.shuffle(pairs)

    if max_samples is not None and max_samples > 0:
        pairs = pairs[:max_samples]
        print(f"[*] 按限制截取样本数: {len(pairs)}")

    val_count = int(len(pairs) * val_ratio)
    train_pairs = pairs[val_count:]
    val_pairs = pairs[:val_count]

    splits = {
        "train": train_pairs,
        "val": val_pairs
    }

    for split_name, split_list in splits.items():
        img_dir = os.path.join(output_dir, split_name, "images")
        mask_dir = os.path.join(output_dir, split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        print(f"[*] 正在生成 {split_name} 集 ({len(split_list)} 张)...")
        for src_path, heat_path, filename in split_list:
            # 复制/转存原图 (以灰度图读取并保存)
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            heat = cv2.imread(heat_path, cv2.IMREAD_GRAYSCALE)

            if img is None or heat is None:
                continue

            # 二值化掩码：大于阈值的区域为布带 (255)
            mask = (heat > threshold).astype(np.uint8) * 255

            base_name = os.path.splitext(filename)[0] + ".png"
            cv2.imwrite(os.path.join(img_dir, base_name), img)
            cv2.imwrite(os.path.join(mask_dir, base_name), mask)

    print(f"[✓] 数据集提取完成！已保存至: {output_dir}")
    print(f"    - 训练集: {len(train_pairs)} 张")
    print(f"    - 验证集: {len(val_pairs)} 张")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取布带分割数据集")
    parser.add_argument("--raw_dir", type=str, default="拉链5号", help="原始拉链数据目录")
    parser.add_argument("--output_dir", type=str, default="dataset", help="输出标准数据集目录")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--max_samples", type=int, default=None, help="最大提取样本数 (空为全量提取)")
    parser.add_argument("--threshold", type=int, default=0, help="热力图二值化判定阈值")
    parser.add_argument("--min_mask_ratio", type=float, default=0.15, help="保留最小布带掩码比例 (过滤空样本)")
    args = parser.parse_args()

    prepare_dataset(args.raw_dir, args.output_dir, args.val_ratio, args.max_samples, args.threshold, args.min_mask_ratio)
