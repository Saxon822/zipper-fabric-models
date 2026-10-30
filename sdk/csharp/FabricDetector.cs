/**
 * Zipper Fabric Detector (C# .NET / ONNX Runtime SDK)
 * 工业检测软件调用集成类
 * 
 * 依赖 NuGet 包:
 *   - Microsoft.ML.OnnxRuntime (或 Microsoft.ML.OnnxRuntime.Gpu / DirectML)
 *   - OpenCvSharp4
 *   - OpenCvSharp4.runtime.win
 * 
 * 适用场景:
 *   任何 C# WPF / WinForms / 控制台工业机器视觉上位机软件
 *   支持免 Python 环境、低延迟实时调用
 */

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ZipperInspection.SDK
{
    public class DetectionResult
    {
        public Mat BinaryMask { get; set; }
        public double FabricAreaRatio { get; set; }
        public double LatencyMs { get; set; }
        public Point[][] Contours { get; set; }
    }

    public class FabricDetector : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly int _inputWidth;
        private readonly int _inputHeight;

        public FabricDetector(string onnxModelPath, bool useGpu = false, int inputHeight = 512, int inputWidth = 320)
        {
            var options = new SessionOptions();
            if (useGpu)
            {
                // 可配置 CUDA 或 DirectML 加速
                options.AppendExecutionProvider_DML(0);
            }
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            _session = new InferenceSession(onnxModelPath, options);
            _inputName = _session.InputMetadata.Keys.First();
            _inputHeight = inputHeight;
            _inputWidth = inputWidth;
        }

        /// <summary>
        /// 对输入图像进行布带区域检测
        /// </summary>
        /// <param name="srcImage">OpenCV Mat 原图 (灰度或彩色)</param>
        /// <param name="threshold">置信度分割阈值 (默认 0.5)</param>
        /// <returns>检测结果 (二值掩码、面积占比、耗时等)</returns>
        public DetectionResult Detect(Mat srcImage, float threshold = 0.5f)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            int origWidth = srcImage.Width;
            int origHeight = srcImage.Height;

            // 1. 前处理: 转灰度并缩放
            using var gray = new Mat();
            if (srcImage.Channels() == 3)
                Cv2.CvtColor(srcImage, gray, ColorConversionCodes.BGR2GRAY);
            else
                srcImage.CopyTo(gray);

            using var resized = new Mat();
            Cv2.Resize(gray, resized, new Size(_inputWidth, _inputHeight));

            // 2. 构造 Float Tensor 归一化到 [0, 1]
            var tensor = new DenseTensor<float>(new[] { 1, 1, _inputHeight, _inputWidth });
            unsafe
            {
                byte* ptr = (byte*)resized.Data;
                for (int y = 0; y < _inputHeight; y++)
                {
                    for (int x = 0; x < _inputWidth; x++)
                    {
                        tensor[0, 0, y, x] = (*ptr++) / 255.0f;
                    }
                }
            }

            // 3. 执行 ONNX 推理
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };

            using var outputs = _session.Run(inputs);
            var outputTensor = outputs.First().AsTensor<float>();

            // 4. 后处理: 提取概率图并阈值化
            using var probMapSmall = new Mat(_inputHeight, _inputWidth, MatType.CV_32FC1);
            unsafe
            {
                float* pData = (float*)probMapSmall.Data;
                for (int y = 0; y < _inputHeight; y++)
                {
                    for (int x = 0; x < _inputWidth; x++)
                    {
                        *pData++ = outputTensor[0, 0, y, x];
                    }
                }
            }

            // 还原到原图大小
            using var probMapOrig = new Mat();
            Cv2.Resize(probMapSmall, probMapOrig, new Size(origWidth, origHeight));

            var binaryMask = new Mat();
            Cv2.Threshold(probMapOrig, binaryMask, threshold, 255, ThresholdTypes.Binary);
            binaryMask.ConvertTo(binaryMask, MatType.CV_8UC1);

            // 5. 统计特征指标
            int fabricPixels = Cv2.CountNonZero(binaryMask);
            double areaRatio = (double)fabricPixels / (origWidth * origHeight);

            Cv2.FindContours(binaryMask, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            sw.Stop();

            return new DetectionResult
            {
                BinaryMask = binaryMask,
                FabricAreaRatio = areaRatio,
                LatencyMs = sw.Elapsed.TotalMilliseconds,
                Contours = contours
            };
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
