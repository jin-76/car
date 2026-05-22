#pragma once

#include "detection.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <array>
#include <string>
#include <vector>

struct YoloOnnxOptions {
  std::string provider = "cuda";
  bool trt_fp16 = true;
  std::string trt_cache_dir;
};

class YoloOnnx {
 public:
  YoloOnnx(const std::string& model_path, int input_size, const YoloOnnxOptions& options);

  std::vector<Detection> infer(const cv::Mat& bgr, float conf_threshold, float iou_threshold);

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  int input_size_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;
  std::array<int64_t, 4> input_shape_;
  std::vector<float> input_buffer_;
};
