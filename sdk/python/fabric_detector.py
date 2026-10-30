#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zipper Fabric Detector Python SDK
轻量级 Python 调用接口，供工业视觉检测软件、自动化脚本直接导入使用。

使用示例:
    from sdk.python.fabric_detector import FabricDetector
    detector = FabricDetector("checkpoints/fabric_segmentation.onnx")
    result = detector.detect("sample.jpg")
    print("布带面积占比:", result["fabric_area_ratio"])
"""

import cv2
import numpy as np
import time

class FabricDetector:
    def __init__(self, onnx_model_path: str, input_height: int = 512, input_width: int = 320, use_gpu: bool = False):
        import onnxruntime as ort
        self.input_size = (input_height, input_width)
        providers = ['CPUExecutionProvider']
        if use_gpu:
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, image_input, threshold: float = 0.5):
        """
        输入: 图片路径 str 或 cv2 读取的 numpy 数组 (BGR/灰度)
        输出: dict 包含 mask (二值掩码), fabric_area_ratio (面积占比), latency_ms (耗时), contours (轮廓)
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {image_input}")
        else:
            img = image_input

        t0 = time.time()
        orig_h, orig_w = img.shape[:2]

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        resized = cv2.resize(gray, (self.input_size[1], self.input_size[0]))
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=(0, 1))

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        prob_map = outputs[0][0, 0]

        prob_orig = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (prob_orig > threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fabric_pixels = int(np.sum(binary_mask > 0))
        area_ratio = fabric_pixels / float(orig_h * orig_w)
        latency_ms = (time.time() - t0) * 1000.0

        return {
            "mask": binary_mask,
            "fabric_area_ratio": area_ratio,
            "latency_ms": latency_ms,
            "contour_count": len(contours),
            "contours": contours
        }
