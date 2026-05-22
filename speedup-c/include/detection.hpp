#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct Detection {
  int class_id = -1;
  float confidence = 0.0f;
  cv::Rect2f box;
  std::string label;
};

