#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace paddle_infer {
class Predictor;
}

struct OcrResult {
  std::string text;
  float confidence = 0.0f;
};

class PaddleOcrRecognizer {
 public:
  PaddleOcrRecognizer(const std::string& model_dir, bool use_gpu, int cpu_threads);

  std::string recognize(const cv::Mat& plate_bgr);
  OcrResult recognize_with_score(const cv::Mat& plate_bgr);

 private:
  std::vector<std::string> characters_;
  std::shared_ptr<paddle_infer::Predictor> predictor_;
};
