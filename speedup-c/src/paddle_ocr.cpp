#include "paddle_ocr.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4244 4251)
#endif
#include <paddle_inference_api.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string trim(std::string value) {
  while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' || value.back() == '\t')) {
    value.pop_back();
  }
  size_t start = 0;
  while (start < value.size() && (value[start] == ' ' || value[start] == '\t')) {
    ++start;
  }
  return value.substr(start);
}

bool env_flag_enabled(const char* name) {
#ifdef _MSC_VER
  char* value = nullptr;
  size_t len = 0;
  if (_dupenv_s(&value, &len, name) != 0 || value == nullptr) return false;
  const std::string flag(value);
  std::free(value);
#else
  const char* value = std::getenv(name);
  if (value == nullptr) return false;
  const std::string flag(value);
#endif
  return !flag.empty() && flag != "0" && flag != "false" && flag != "FALSE";
}

std::vector<std::string> load_character_dict(const std::string& model_dir) {
  namespace fs = std::filesystem;
  const fs::path dir(model_dir);
  std::vector<fs::path> candidates = {
      dir / "inference.yml",
  };
  const std::string dir_name = dir.filename().string();
  if (dir_name.find("OCRv4") != std::string::npos || dir_name.find("ocrv4") != std::string::npos) {
    candidates.push_back(fs::current_path() / ".paddlex-cache" / "official_models" / "PP-OCRv4_mobile_rec" / "inference.yml");
    candidates.push_back(dir.parent_path().parent_path() / ".paddlex-cache" / "official_models" / "PP-OCRv4_mobile_rec" / "inference.yml");
  }
  candidates.push_back(dir.parent_path() / "inference.yml");

  std::ifstream in;
  fs::path selected;
  for (const auto& candidate : candidates) {
    in.open(candidate, std::ios::binary);
    if (in.is_open()) {
      selected = candidate;
      break;
    }
    in.clear();
  }
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open OCR inference.yml for character_dict");
  }

  std::vector<std::string> characters;
  std::string line;
  bool in_dict = false;
  while (std::getline(in, line)) {
    if (!in_dict) {
      if (line.find("character_dict:") != std::string::npos) {
        in_dict = true;
      }
      continue;
    }

    const std::string stripped = trim(line);
    if (stripped.rfind("- ", 0) != 0) {
      if (!characters.empty()) break;
      continue;
    }

    std::string token = trim(stripped.substr(2));
    if (token == "''''") {
      token = "'";
    }
    if (token.size() >= 2 && ((token.front() == '"' && token.back() == '"') || (token.front() == '\'' && token.back() == '\''))) {
      token = token.substr(1, token.size() - 2);
    }
    characters.push_back(token);
  }
  if (characters.empty()) {
    throw std::runtime_error("OCR character_dict is empty");
  }
  if (env_flag_enabled("OCR_DEBUG")) {
    std::cerr << "OCR character dict: " << selected.string() << " entries=" << characters.size() << "\n";
  }
  return characters;
}

cv::Mat resize_norm_img(const cv::Mat& plate_bgr, int img_h, int img_w) {
  const double ratio = plate_bgr.cols / static_cast<double>(std::max(1, plate_bgr.rows));
  int resized_w = static_cast<int>(std::ceil(img_h * ratio));
  resized_w = std::max(1, std::min(img_w, resized_w));

  cv::Mat resized;
  cv::resize(plate_bgr, resized, cv::Size(resized_w, img_h));
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  resized.convertTo(resized, CV_32F, 1.0 / 255.0);
  resized = (resized - 0.5f) / 0.5f;

  cv::Mat padded(img_h, img_w, CV_32FC3, cv::Scalar(-1.0f, -1.0f, -1.0f));
  resized.copyTo(padded(cv::Rect(0, 0, resized_w, img_h)));
  return padded;
}

OcrResult ctc_decode(const std::vector<float>& output, const std::vector<int>& shape, const std::vector<std::string>& characters) {
  if (shape.size() < 2 || output.empty()) return {};

  int seq_len = 0;
  int class_count = 0;
  size_t offset = 0;
  if (shape.size() == 3) {
    seq_len = shape[1];
    class_count = shape[2];
  } else {
    seq_len = shape[0];
    class_count = shape[1];
  }
  if (seq_len <= 0 || class_count <= 1) return {};
  if (static_cast<size_t>(seq_len) * static_cast<size_t>(class_count) > output.size()) return {};

  OcrResult result;
  int previous_index = -1;
  float score_sum = 0.0f;
  int score_count = 0;
  for (int t = 0; t < seq_len; ++t) {
    const float* row = output.data() + offset + static_cast<size_t>(t) * static_cast<size_t>(class_count);
    int best_index = 0;
    float best_score = row[0];
    for (int c = 1; c < class_count; ++c) {
      if (row[c] > best_score) {
        best_score = row[c];
        best_index = c;
      }
    }

    const bool is_blank = best_index == 0;
    const bool is_repeat = best_index == previous_index;
    if (!is_blank && !is_repeat) {
      const int char_index = best_index - 1;
      if (char_index >= 0 && char_index < static_cast<int>(characters.size())) {
        result.text += characters[char_index];
        score_sum += best_score;
        ++score_count;
      } else if (char_index == static_cast<int>(characters.size()) && class_count == static_cast<int>(characters.size()) + 2) {
        result.text += ' ';
        score_sum += best_score;
        ++score_count;
      }
    }
    previous_index = best_index;
  }
  result.confidence = score_count > 0 ? score_sum / static_cast<float>(score_count) : 0.0f;
  return result;
}

}  // namespace

PaddleOcrRecognizer::PaddleOcrRecognizer(const std::string& model_dir, bool use_gpu, int cpu_threads) {
  characters_ = load_character_dict(model_dir);

  paddle_infer::Config config;
  config.SetModel(model_dir + "/inference.pdmodel", model_dir + "/inference.pdiparams");
  if (use_gpu) {
    config.EnableUseGpu(512, 0);
  } else {
    config.DisableGpu();
    config.SetCpuMathLibraryNumThreads(std::max(1, cpu_threads));
  }
  config.SwitchIrOptim(true);
  predictor_ = paddle_infer::CreatePredictor(config);
  if (!predictor_) {
    throw std::runtime_error("Failed to create Paddle OCR predictor");
  }
}

OcrResult PaddleOcrRecognizer::recognize_with_score(const cv::Mat& plate_bgr) {
  if (plate_bgr.empty()) return {};

  constexpr int img_h = 48;
  constexpr int img_w = 320;
  cv::Mat normalized = resize_norm_img(plate_bgr, img_h, img_w);

  std::vector<float> input(static_cast<size_t>(3 * img_h * img_w));
  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] = cv::Mat(img_h, img_w, CV_32F, input.data() + c * img_h * img_w);
  }
  cv::split(normalized, channels);

  auto input_names = predictor_->GetInputNames();
  auto input_handle = predictor_->GetInputHandle(input_names[0]);
  input_handle->Reshape({1, 3, img_h, img_w});
  input_handle->CopyFromCpu(input.data());

  predictor_->Run();

  auto output_names = predictor_->GetOutputNames();
  auto output_handle = predictor_->GetOutputHandle(output_names[0]);
  std::vector<int> shape = output_handle->shape();
  int64_t count = 1;
  for (int dim : shape) count *= dim;
  std::vector<float> output(static_cast<size_t>(count));
  output_handle->CopyToCpu(output.data());

  if (env_flag_enabled("OCR_DEBUG")) {
    static bool printed = false;
    if (!printed) {
      printed = true;
      std::cerr << "OCR output shape:";
      for (int dim : shape) std::cerr << " " << dim;
      std::cerr << " dict_entries=" << characters_.size() << "\n";
    }
  }

  return ctc_decode(output, shape, characters_);
}

std::string PaddleOcrRecognizer::recognize(const cv::Mat& plate_bgr) {
  return recognize_with_score(plate_bgr).text;
}
