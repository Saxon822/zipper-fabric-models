/**
 * Zipper Fabric Detector (C++ / ONNX Runtime SDK)
 * 工业机器视觉检测软件 C++ 集成头文件
 * 
 * 依赖:
 *   - onnxruntime (C++ API)
 *   - OpenCV (core, imgproc, highgui)
 */

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct FabricDetectionResult {
    cv::Mat binary_mask;
    double fabric_area_ratio;
    double latency_ms;
    std::vector<std::vector<cv::Point>> contours;
};

class FabricDetector {
public:
    FabricDetector(const std::string& onnx_model_path, int input_height = 512, int input_width = 320)
        : env_(ORT_LOGGING_LEVEL_WARNING, "FabricDetector"),
          input_height_(input_height),
          input_width_(input_width) {
        
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, onnx_model_path.c_str(), session_options_);
    }

    FabricDetectionResult detect(const cv::Mat& src_image, float threshold = 0.5f) {
        auto t_start = std::chrono::high_resolution_clock::now();

        int orig_w = src_image.cols;
        int orig_h = src_image.rows;

        // 1. 前处理
        cv::Mat gray, resized;
        if (src_image.channels() == 3) {
            cv::cvtColor(src_image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = src_image;
        }
        cv::resize(gray, resized, cv::Size(input_width_, input_height_));

        // 2. 构造 Float Tensor 归一化 [0, 1]
        std::vector<float> input_tensor_values(1 * 1 * input_height_ * input_width_);
        for (int i = 0; i < input_height_ * input_width_; ++i) {
            input_tensor_values[i] = resized.data[i] / 255.0f;
        }

        std::vector<int64_t> input_shape = {1, 1, input_height_, input_width_};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()
        );

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        // 3. 执行推理
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
        );

        float* out_data = output_tensors.front().GetTensorMutableData<float>();

        // 4. 后处理
        cv::Mat prob_small(input_height_, input_width_, CV_32FC1, out_data);
        cv::Mat prob_orig;
        cv::resize(prob_small, prob_orig, cv::Size(orig_w, orig_h));

        cv::Mat binary_mask;
        cv::threshold(prob_orig, binary_mask, threshold, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8UC1);

        int fabric_pixels = cv::countNonZero(binary_mask);
        double area_ratio = static_cast<double>(fabric_pixels) / (orig_w * orig_h);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        FabricDetectionResult res;
        res.binary_mask = binary_mask;
        res.fabric_area_ratio = area_ratio;
        res.latency_ms = elapsed_ms;
        res.contours = contours;
        return res;
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    int input_height_;
    int input_width_;
};
