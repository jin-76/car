#include "yolo_onnx.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {

struct LetterboxResult {
  cv::Mat image;
  float scale = 1.0f;
  int pad_x = 0;
  int pad_y = 0;
};

LetterboxResult letterbox(const cv::Mat& bgr, int size) {
  const float scale = std::min(size / static_cast<float>(bgr.cols), size / static_cast<float>(bgr.rows));
  const int new_w = static_cast<int>(std::round(bgr.cols * scale));
  const int new_h = static_cast<int>(std::round(bgr.rows * scale));
  LetterboxResult out;
  out.scale = scale;
  out.pad_x = (size - new_w) / 2;
  out.pad_y = (size - new_h) / 2;

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(new_w, new_h));
  out.image = cv::Mat(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(out.image(cv::Rect(out.pad_x, out.pad_y, new_w, new_h)));
  return out;
}

}  // namespace

YoloOnnx::YoloOnnx(const std::string& model_path, int input_size, const YoloOnnxOptions& options)
    : env_(ORT_LOGGING_LEVEL_WARNING, "car_speedup"),
      session_options_(),
      session_(nullptr),
      input_size_(input_size),
      input_shape_{1, 3, input_size, input_size},
      input_buffer_(static_cast<size_t>(3 * input_size * input_size)) {
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (options.provider == "tensorrt") {
    if (!options.trt_cache_dir.empty()) {
      std::filesystem::create_directories(options.trt_cache_dir);
    }
    Ort::TensorRTProviderOptions trt_options;
    std::unordered_map<std::string, std::string> trt_values{
        {"device_id", "0"},
        {"trt_fp16_enable", options.trt_fp16 ? "1" : "0"},
        {"trt_engine_cache_enable", options.trt_cache_dir.empty() ? "0" : "1"},
        {"trt_engine_cache_path", options.trt_cache_dir},
        {"trt_min_subgraph_size", "1"},
    };
    trt_options.Update(trt_values);
    session_options_.AppendExecutionProvider_TensorRT_V2(*trt_options);
    std::cout << "YOLO TensorRT EP: fp16=" << (options.trt_fp16 ? "on" : "off")
              << ", cache=" << (options.trt_cache_dir.empty() ? "off" : options.trt_cache_dir) << "\n";
  }
  if (options.provider == "cuda" || options.provider == "tensorrt") {
    OrtCUDAProviderOptions cuda_options{};
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  } else if (options.provider != "cpu") {
    throw std::runtime_error("Unsupported YOLO provider: " + options.provider);
  }

  session_ = Ort::Session(env_, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options_);

  const size_t input_count = session_.GetInputCount();
  const size_t output_count = session_.GetOutputCount();
  input_names_.reserve(input_count);
  output_names_.reserve(output_count);
  input_name_ptrs_.reserve(input_count);
  output_name_ptrs_.reserve(output_count);

  for (size_t i = 0; i < input_count; ++i) {
    auto name = session_.GetInputNameAllocated(i, allocator_);
    input_names_.emplace_back(name.get());
  }
  for (size_t i = 0; i < output_count; ++i) {
    auto name = session_.GetOutputNameAllocated(i, allocator_);
    output_names_.emplace_back(name.get());
  }
  for (auto& name : input_names_) input_name_ptrs_.push_back(name.c_str());
  for (auto& name : output_names_) output_name_ptrs_.push_back(name.c_str());
}

std::vector<Detection> YoloOnnx::infer(const cv::Mat& bgr, float conf_threshold, float iou_threshold) {
  if (bgr.empty()) return {};

  const auto lb = letterbox(bgr, input_size_);
  cv::Mat rgb;
  cv::cvtColor(lb.image, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] = cv::Mat(input_size_, input_size_, CV_32F,
                          input_buffer_.data() + c * input_size_ * input_size_);
  }
  cv::split(rgb, channels);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_buffer_.data(), input_buffer_.size(), input_shape_.data(), input_shape_.size());

  auto outputs = session_.Run(
      Ort::RunOptions{nullptr},
      input_name_ptrs_.data(),
      &tensor,
      1,
      output_name_ptrs_.data(),
      output_name_ptrs_.size());

  if (outputs.empty()) return {};

  auto& output = outputs[0];
  auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
  const float* data = output.GetTensorData<float>();

  // Ultralytics ONNX usually returns [1, 84, N] or [1, N, 84].
  int64_t rows = 0;
  int64_t dims = 0;
  bool transposed = false;
  if (shape.size() == 3 && shape[1] < shape[2]) {
    dims = shape[1];
    rows = shape[2];
    transposed = true;
  } else if (shape.size() == 3) {
    rows = shape[1];
    dims = shape[2];
  } else {
    throw std::runtime_error("Unsupported YOLO output shape");
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;
  for (int64_t i = 0; i < rows; ++i) {
    auto at = [&](int64_t d) -> float {
      return transposed ? data[d * rows + i] : data[i * dims + d];
    };
    float best_score = 0.0f;
    int best_class = -1;
    for (int64_t c = 4; c < dims; ++c) {
      const float score = at(c);
      if (score > best_score) {
        best_score = score;
        best_class = static_cast<int>(c - 4);
      }
    }
    if (best_score < conf_threshold) continue;

    const float cx = at(0);
    const float cy = at(1);
    const float w = at(2);
    const float h = at(3);
    float x1 = (cx - w / 2.0f - lb.pad_x) / lb.scale;
    float y1 = (cy - h / 2.0f - lb.pad_y) / lb.scale;
    float x2 = (cx + w / 2.0f - lb.pad_x) / lb.scale;
    float y2 = (cy + h / 2.0f - lb.pad_y) / lb.scale;
    x1 = std::clamp(x1, 0.0f, static_cast<float>(bgr.cols - 1));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(bgr.rows - 1));
    x2 = std::clamp(x2, 0.0f, static_cast<float>(bgr.cols - 1));
    y2 = std::clamp(y2, 0.0f, static_cast<float>(bgr.rows - 1));
    boxes.emplace_back(cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                cv::Point(static_cast<int>(x2), static_cast<int>(y2))));
    scores.push_back(best_score);
    class_ids.push_back(best_class);
  }

  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, keep);

  std::vector<Detection> detections;
  detections.reserve(keep.size());
  for (int index : keep) {
    Detection det;
    det.class_id = class_ids[index];
    det.confidence = scores[index];
    det.box = boxes[index];
    detections.push_back(det);
  }
  return detections;
}
