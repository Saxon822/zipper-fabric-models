# Zipper Fabric Inspection & Segmentation (工业拉链布带区域检测通用模型)

本项目基于工业拉链生产线质检数据构建，提供了一套**开源、通用、支持二次微调，且专为工业机器视觉检测软件集成设计**的布带区域分割系统。

---

## 🌟 核心特性

* **通用标准化架构**：摒弃原有专有软件格式，采用轻量级 `FabricUNet` 架构，模型大小仅 **4.2 MB**，CPU 单帧推理仅需 **~20-30 ms**，GPU 毫秒级。
* **支持免环境二次微调（Fine-Tuning）**：提供从原始采样数据清洗、数据集划分、训练微调到权重保存的完整 Pipeline，支持接入新布带与缺陷样本继续微调。
* **标准 ONNX 格式部署**：支持一键导出为通用 `ONNX` 模型，脱离 Python 与 PyTorch 庞大运行时。
* **许可工业软件直接调用（商用友好）**：采用 **Apache 2.0 开源许可**，附带专有工业软件集成许可条款，允许闭源商业上位机（C# / C++ / Python）自由打包、嵌入与动态调用。
* **现成多语言 SDK**：开箱即用提供 C# (.NET WinForms/WPF) 与 C++ 源码级接入模块。

---

## 📂 项目结构

```text
zipper-fabric-models/
├── models/
│   └── fabric_unet.py           # 通用布带轻量级分割模型架构
├── sdk/                         # 工业检测软件调用集成接口
│   ├── csharp/FabricDetector.cs # C# .NET / ONNX Runtime 上位机检测类
│   ├── cpp/fabric_detector.hpp  # C++ / OpenCV / ONNX Runtime 检测头文件
│   └── python/fabric_detector.py# Python 纯净调用 SDK
├── checkpoints/                 # 模型产物
│   ├── best_fabric_model.pth    # PyTorch 训练权重
│   └── fabric_segmentation.onnx # 导出的通用 ONNX 部署模型
├── dataset/                     # 标准训练与验证数据集 (由 prepare_dataset 生成)
│   ├── train/ (images, masks)
│   └── val/   (images, masks)
├── prepare_dataset.py           # 原始工程数据提取与二值化清洗脚本
├── train.py                     # 模型训练、增量微调与自动 ONNX 导出
├── predict.py                   # 通用推理测试与可视化渲染脚本
├── requirements.txt             # Python 依赖清单
├── LICENSE                      # Apache 2.0 开源与商用授权许可
└── README.md                    # 本文档
```

---

## 🚀 快速上手

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 提取与清洗布带标注数据
运行脚本将原始拉链采样图与布带热力图转为标准的图像与掩码（二值 Mask）结构：
```bash
# 全量提取 2,599 对布带标注（默认 80% 训练，20% 验证）
python prepare_dataset.py --raw_dir "拉链5号" --output_dir "dataset" --val_ratio 0.2
```

### 3. 模型重新微调 (Fine-Tuning)
支持从零训练，或加载已有的最佳模型权重进行迁移学习微调：
```bash
# 加载现有权重进行 15 轮微调，并自动导出 ONNX 模型
python train.py \
  --data_dir dataset \
  --weights checkpoints/best_fabric_model.pth \
  --epochs 15 \
  --batch_size 8 \
  --lr 0.0005 \
  --img_height 512 \
  --img_width 320 \
  --export_onnx
```

### 4. 推理检测与可视化验证
使用导出的 ONNX 模型直接对测试图像进行检测：
```bash
python predict.py \
  --model checkpoints/fabric_segmentation.onnx \
  --image "拉链5号/source/160.9b941ee7.JPEG" \
  --output "output_result.png"
```
终端将输出：
* 推理耗时（ms）
* 布带有效面积占比（%）
* 轮廓外接几何属性
* 绿色半透明掩码与红色边缘的可视化效果图

---

## 🏭 工业检测软件集成调用指南

本模型专为嵌入工业上位机设计，上位机**无需安装 Python**，只需引入 ONNX Runtime 即可极速调用。

### 场景 A：C# .NET 上位机软件（WPF / WinForms）

1. 在 Visual Studio 中为项目安装 NuGet 包：
   * `Microsoft.ML.OnnxRuntime`（或支持显卡加速的 `Microsoft.ML.OnnxRuntime.DirectML` / `Gpu`）
   * `OpenCvSharp4` 与 `OpenCvSharp4.runtime.win`
2. 直接将 `sdk/csharp/FabricDetector.cs` 复制到工程中调用：

```csharp
using OpenCvSharp;
using ZipperInspection.SDK;

// 初始化检测引擎 (只需加载一次)
using var detector = new FabricDetector("checkpoints/fabric_segmentation.onnx");

// 读取相机拍摄的拉链图像
using var mat = Cv2.ImRead("zipper_sample.jpg");

// 执行布带检测
var result = detector.Detect(mat, threshold: 0.5f);

Console.WriteLine($"检测耗时: {result.LatencyMs:F2} ms");
Console.WriteLine($"布带面积占比: {result.FabricAreaRatio * 100:F2}%");

// 判断布带缺陷 (例如面积异常、边缘轮廓断裂等)
if (result.FabricAreaRatio < 0.3) {
    Console.WriteLine("[NG] 布带缺失或边缘严重破损！");
} else {
    Console.WriteLine("[OK] 布带正常");
}
```

### 场景 B：C++ 高性能视觉流水线

包含 `sdk/cpp/fabric_detector.hpp`，链接 `onnxruntime.lib` 与 `opencv_world.lib`：

```cpp
#include "sdk/cpp/fabric_detector.hpp"

int main() {
    FabricDetector detector("checkpoints/fabric_segmentation.onnx");
    cv::Mat img = cv::imread("zipper_sample.jpg");
    
    auto result = detector.detect(img, 0.5f);
    std::cout << "Latency: " << result.latency_ms << " ms" << std::endl;
    std::cout << "Fabric Ratio: " << result.fabric_area_ratio * 100 << "%" << std::endl;
    return 0;
}
```

### 场景 C：Python 自动化质检脚本

```python
from sdk.python.fabric_detector import FabricDetector

detector = FabricDetector("checkpoints/fabric_segmentation.onnx")
res = detector.detect("sample.jpg", threshold=0.5)

print(f"布带占比: {res['fabric_area_ratio'] * 100:.1f}%, 耗时: {res['latency_ms']:.2f}ms")
```

---

## 📄 软件授权与商业使用许可 (License)

本项目采用 **Apache License 2.0** 协议开源。
* **闭源与商业友好**：明确许可任何工业机器视觉检测软件、工厂 MES 系统、生产线 PLC 联动上位机在保持商业闭源的前提下直接加载、打包并调用本模型的权重与 SDK。
* **自由微调与分发**：允许在内部生产环境根据不同的布带规格二次训练与微调新模型。
