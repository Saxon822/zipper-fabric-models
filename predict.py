#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zipper Fabric Universal Predictor (通用布带推理检测脚本)
支持 ONNX Runtime 与 PyTorch 双引擎推理，输出布带掩码、轮廓提取、面积占比与可视化图。
"""

import os
import argparse
import time
import cv2
import numpy as np

class FabricPredictor:
    def __init__(self, model_path, input_size=(512, 320), device="cpu"):
        self.model_path = model_path
        self.input_size = input_size  # (height, width)
        self.device = device
        self.is_onnx = model_path.endswith(".onnx")

        if self.is_onnx:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            if device == "cuda":
                providers.insert(0, 'CUDAExecutionProvider')
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"[*] 已初始化 ONNX Runtime 推理会话: {model_path} (Providers: {self.session.get_providers()})")
        else:
            import torch
            from models.fabric_unet import FabricUNet
            self.model = FabricUNet(n_channels=1, n_classes=1)
            state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state)
            self.model.to(device)
            self.model.eval()
            print(f"[*] 已初始化 PyTorch 推理模型: {model_path}")

    def preprocess(self, img_bgr_or_gray):
        if len(img_bgr_or_gray.shape) == 3:
            gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr_or_gray

        orig_h, orig_w = gray.shape[:2]
        # 缩放到模型输入尺寸
        resized = cv2.resize(gray, (self.input_size[1], self.input_size[0]))
        # 归一化至 [0, 1]
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=(0, 1))  # [1, 1, H, W]
        return tensor, (orig_h, orig_w)

    def predict(self, image_input, threshold=0.5):
        """
        image_input: 可以是图片路径，也可以是已加载的 numpy 图像
        """
        if isinstance(image_input, str):
            orig_img = cv2.imread(image_input)
            if orig_img is None:
                raise FileNotFoundError(f"无法读取图片: {image_input}")
        else:
            orig_img = image_input

        t0 = time.time()
        input_tensor, (orig_h, orig_w) = self.preprocess(orig_img)

        # 执行推理
        if self.is_onnx:
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            prob_map = outputs[0][0, 0]  # [H, W]
        else:
            import torch
            with torch.no_grad():
                tensor_torch = torch.from_numpy(input_tensor).to(self.device)
                probs = self.model(tensor_torch)
                prob_map = probs.cpu().numpy()[0, 0]

        latency_ms = (time.time() - t0) * 1000.0

        # 还原回原图尺寸
        prob_orig = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (prob_orig > threshold).astype(np.uint8) * 255

        # 计算几何特征与轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fabric_area_pixels = int(np.sum(binary_mask > 0))
        total_pixels = orig_h * orig_w
        area_ratio = fabric_area_pixels / float(total_pixels)

        # 生成可视化渲染图
        vis_img = orig_img.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

        # 绿色透明掩膜覆盖布带
        overlay = vis_img.copy()
        overlay[binary_mask > 0] = [0, 255, 0]
        vis_img = cv2.addWeighted(overlay, 0.35, vis_img, 0.65, 0)
        # 红色绘制外边界轮廓
        cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)

        # 标注文本信息
        info_text = f"Fabric Area: {area_ratio * 100:.1f}% | Time: {latency_ms:.1f}ms"
        cv2.putText(vis_img, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return {
            "mask": binary_mask,
            "probability_map": prob_orig,
            "fabric_area_ratio": area_ratio,
            "latency_ms": latency_ms,
            "contour_count": len(contours),
            "visualization": vis_img
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通用布带检测推理")
    parser.add_argument("--model", type=str, required=True, help="模型路径 (.onnx 或 .pth)")
    parser.add_argument("--image", type=str, required=True, help="输入测试图片路径")
    parser.add_argument("--output", type=str, default="output_result.png", help="输出可视化图保存路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化阈值")
    args = parser.parse_args()

    predictor = FabricPredictor(args.model)
    result = predictor.predict(args.image, threshold=args.threshold)

    print("\n=== 推理结果分析 ===")
    print(f"推理耗时: {result['latency_ms']:.2f} ms")
    print(f"布带面积占比: {result['fabric_area_ratio'] * 100:.2f}%")
    print(f"检测到轮廓数: {result['contour_count']}")

    cv2.imwrite(args.output, result["visualization"])
    print(f"[✓] 可视化结果已保存至: {args.output}")
