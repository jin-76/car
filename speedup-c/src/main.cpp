#include "paddle_ocr.hpp"
#include "yolo_onnx.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincrypt.h>
#include <winhttp.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <deque>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Line {
  cv::Point p1;
  cv::Point p2;
};

struct SpeedSample {
  int frame = 0;
  cv::Point2f image_point;
  cv::Point2f metric_point;
};

struct OrientedRect {
  cv::Point2f center;
  cv::Point2f heading;
  cv::Point2f right;
  float length = 0.0f;
  float width = 0.0f;
  std::vector<cv::Point2f> corners;
};

struct Track {
  int id = 0;
  int class_id = -1;
  cv::Rect2f box;
  int last_seen = 0;
  cv::Point2f still_anchor;
  int still_start_frame = 0;
  bool parking_reported = false;
  bool was_between = false;
  bool pending_between = false;
  int pending_count = 0;
  int entry_frame = 0;
  cv::Point2f entry_point;
  double last_speed_kmh = 0.0;
  double last_speed_mps = 0.0;
  int last_parking_scratch_frame = -1000000;
  int collision_speed_frames = 0;
  cv::Point2f collision_direction_anchor{0.0f, 0.0f};
  bool has_collision_direction_anchor = false;
  std::string last_speed_status;
  std::string plate_text;
  bool plate_locked = false;
  std::string plate_lock_reason;
  std::map<std::string, int> plate_votes;
  int plate_valid_observations = 0;
  std::string last_plate_candidate;
  int consecutive_plate_count = 0;
  std::deque<SpeedSample> metric_history;
};

struct PersonTrack {
  int id = 0;
  int class_id = 0;
  cv::Rect2f box;
  int last_seen = 0;
  int previous_last_seen = 0;
  std::string collision_stage = "normal";
  int near_count = 0;
  int contact_frame = 0;
  int contact_vehicle_id = 0;
  bool reported = false;
  std::deque<cv::Point2f> post_contact_points;
  cv::Point2f last_metric_point{0.0f, 0.0f};
  bool has_last_metric_point = false;
  cv::Point2f disappear_metric_point{0.0f, 0.0f};
  bool has_disappear_metric_point = false;
  bool waiting_reappear_after_disappear = false;
  int disappear_start_frame = 0;
  double last_speed_kmh = 0.0;
  double last_speed_mps = 0.0;
  std::deque<SpeedSample> metric_history;
};

struct ParkingCollisionState {
  std::string stage = "normal";
  int contact_frame = 0;
  int contact_count = 0;
  double previous_distance_m = 0.0;
  double max_recent_distance_m = 0.0;
  double min_after_contact_m = 1e9;
  bool near_zero_after_contact = false;
  bool reported_contact = false;
  bool reported_scratch = false;
  double speed_a_at_contact = 0.0;
  double speed_b_at_contact = 0.0;
  cv::Point2f heading_a_at_contact{0.0f, -1.0f};
  cv::Point2f heading_b_at_contact{0.0f, -1.0f};
  std::deque<double> recent_distances;
};

struct NotifyConfig {
  bool enabled = false;
  std::string provider;
  std::string webhook;
};

struct SnapshotBox {
  cv::Rect2f box;
  std::string label;
  cv::Scalar color;
};

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
  std::unordered_map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key.rfind("--", 0) != 0) continue;
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      args[key] = argv[++i];
    } else {
      args[key] = "true";
    }
  }
  return args;
}

std::string get_arg(const std::unordered_map<std::string, std::string>& args,
                    const std::string& key,
                    const std::string& fallback) {
  auto it = args.find(key);
  return it == args.end() ? fallback : it->second;
}

bool get_bool_arg(const std::unordered_map<std::string, std::string>& args,
                  const std::string& key,
                  bool fallback) {
  const auto value = get_arg(args, key, fallback ? "1" : "0");
  return value == "1" || value == "true" || value == "True" || value == "yes";
}

std::string get_env_string(const std::string& key) {
  if (key.empty()) return "";
  const DWORD needed = GetEnvironmentVariableA(key.c_str(), nullptr, 0);
  if (needed == 0) return "";
  std::string value(needed, '\0');
  const DWORD written = GetEnvironmentVariableA(key.c_str(), value.data(), needed);
  if (written == 0) return "";
  value.resize(written);
  return value;
}

std::string beijing_time_now() {
  const auto beijing_now = std::chrono::system_clock::now() + std::chrono::hours(8);
  const std::time_t timestamp = std::chrono::system_clock::to_time_t(beijing_now);
  std::tm time_info{};
#ifdef _WIN32
  gmtime_s(&time_info, &timestamp);
#else
  gmtime_r(&timestamp, &time_info);
#endif
  char buffer[32] = {};
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &time_info);
  return buffer;
}

std::wstring webhook_utf8_to_wide(const std::string& value) {
  if (value.empty()) return L"";
  const int size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
  std::wstring wide(static_cast<size_t>(std::max(0, size - 1)), L'\0');
  if (size > 1) {
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, wide.data(), size);
  }
  return wide;
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::uppercase << static_cast<int>(ch);
        } else {
          out << ch;
        }
    }
  }
  return out.str();
}

std::string make_notify_payload(const NotifyConfig& notify, const std::string& message) {
  const std::string escaped = json_escape(message);
  if (notify.provider == "wecom" || notify.provider == "wechat" || notify.provider == "qywx") {
    return "{\"msgtype\":\"text\",\"text\":{\"content\":\"" + escaped + "\"}}";
  }
  return "{\"msg_type\":\"text\",\"content\":{\"text\":\"" + escaped + "\"}}";
}

bool post_json_webhook(const std::string& url, const std::string& body, std::string& error) {
  const std::wstring wide_url = webhook_utf8_to_wide(url);
  URL_COMPONENTS parts{};
  parts.dwStructSize = sizeof(parts);
  parts.dwSchemeLength = static_cast<DWORD>(-1);
  parts.dwHostNameLength = static_cast<DWORD>(-1);
  parts.dwUrlPathLength = static_cast<DWORD>(-1);
  parts.dwExtraInfoLength = static_cast<DWORD>(-1);
  if (!WinHttpCrackUrl(wide_url.c_str(), 0, 0, &parts)) {
    error = "invalid webhook url";
    return false;
  }

  const std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
  std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
  if (parts.dwExtraInfoLength > 0) {
    path.append(parts.lpszExtraInfo, parts.dwExtraInfoLength);
  }
  const bool https = parts.nScheme == INTERNET_SCHEME_HTTPS;
  HINTERNET session = WinHttpOpen(L"SII-Traffic/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                  WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!session) {
    error = "WinHttpOpen failed";
    return false;
  }
  HINTERNET connect = WinHttpConnect(session, host.c_str(), parts.nPort, 0);
  if (!connect) {
    WinHttpCloseHandle(session);
    error = "WinHttpConnect failed";
    return false;
  }
  HINTERNET request = WinHttpOpenRequest(connect, L"POST", path.c_str(), nullptr,
                                         WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                         https ? WINHTTP_FLAG_SECURE : 0);
  if (!request) {
    WinHttpCloseHandle(connect);
    WinHttpCloseHandle(session);
    error = "WinHttpOpenRequest failed";
    return false;
  }
  const std::wstring headers = L"Content-Type: application/json; charset=utf-8\r\n";
  const BOOL ok = WinHttpSendRequest(request, headers.c_str(), static_cast<DWORD>(headers.size()),
                                     const_cast<char*>(body.data()), static_cast<DWORD>(body.size()),
                                     static_cast<DWORD>(body.size()), 0) &&
                  WinHttpReceiveResponse(request, nullptr);
  DWORD status = 0;
  DWORD status_size = sizeof(status);
  if (ok) {
    WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size, WINHTTP_NO_HEADER_INDEX);
  }
  WinHttpCloseHandle(request);
  WinHttpCloseHandle(connect);
  WinHttpCloseHandle(session);
  if (!ok || status < 200 || status >= 300) {
    error = "webhook HTTP status " + std::to_string(status);
    return false;
  }
  return true;
}

std::string base64_encode(const std::vector<unsigned char>& data) {
  static constexpr char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((data.size() + 2) / 3) * 4);
  for (size_t i = 0; i < data.size(); i += 3) {
    const unsigned int b0 = data[i];
    const unsigned int b1 = (i + 1 < data.size()) ? data[i + 1] : 0;
    const unsigned int b2 = (i + 2 < data.size()) ? data[i + 2] : 0;
    out.push_back(table[(b0 >> 2) & 0x3F]);
    out.push_back(table[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]);
    out.push_back(i + 1 < data.size() ? table[((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)] : '=');
    out.push_back(i + 2 < data.size() ? table[b2 & 0x3F] : '=');
  }
  return out;
}

std::string md5_hex(const std::vector<unsigned char>& data) {
#ifdef _WIN32
  HCRYPTPROV provider = 0;
  HCRYPTHASH hash = 0;
  BYTE digest[16] = {};
  DWORD digest_size = sizeof(digest);
  if (!CryptAcquireContextW(&provider, nullptr, nullptr, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) return "";
  if (!CryptCreateHash(provider, CALG_MD5, 0, 0, &hash)) {
    CryptReleaseContext(provider, 0);
    return "";
  }
  const BOOL hashed = CryptHashData(hash,
                                   data.empty() ? nullptr : data.data(),
                                   static_cast<DWORD>(data.size()),
                                   0);
  const BOOL got_hash = hashed && CryptGetHashParam(hash, HP_HASHVAL, digest, &digest_size, 0);
  CryptDestroyHash(hash);
  CryptReleaseContext(provider, 0);
  if (!got_hash) return "";
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (DWORD i = 0; i < digest_size; ++i) {
    out << std::setw(2) << static_cast<int>(digest[i]);
  }
  return out.str();
#else
  (void)data;
  return "";
#endif
}

std::string make_wecom_image_payload(const std::vector<unsigned char>& jpeg) {
  const std::string md5 = md5_hex(jpeg);
  if (md5.empty()) return "";
  return "{\"msgtype\":\"image\",\"image\":{\"base64\":\"" + base64_encode(jpeg) +
         "\",\"md5\":\"" + md5 + "\"}}";
}

void send_notification(const NotifyConfig& notify, const std::string& message) {
  if (!notify.enabled) return;
  std::string error;
  if (post_json_webhook(notify.webhook, make_notify_payload(notify, message), error)) {
    std::cout << "Notify sent: " << message << "\n";
  } else {
    std::cout << "Notify failed: " << error << "\n";
  }
}

cv::Rect2f union_rect(const cv::Rect2f& a, const cv::Rect2f& b) {
  if (a.width <= 0.0f || a.height <= 0.0f) return b;
  if (b.width <= 0.0f || b.height <= 0.0f) return a;
  const float x1 = std::min(a.x, b.x);
  const float y1 = std::min(a.y, b.y);
  const float x2 = std::max(a.x + a.width, b.x + b.width);
  const float y2 = std::max(a.y + a.height, b.y + b.height);
  return cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
}

cv::Rect snapshot_rect(const cv::Mat& frame, const cv::Rect2f& target) {
  if (frame.empty()) return {};
  cv::Rect2f box = target;
  if (box.width <= 1.0f || box.height <= 1.0f) {
    return cv::Rect(0, 0, frame.cols, frame.rows);
  }
  const float pad = std::max({box.width, box.height, 80.0f}) * 0.45f;
  box.x -= pad;
  box.y -= pad;
  box.width += pad * 2.0f;
  box.height += pad * 2.0f;
  cv::Rect roi(static_cast<int>(std::floor(box.x)),
               static_cast<int>(std::floor(box.y)),
               static_cast<int>(std::ceil(box.width)),
               static_cast<int>(std::ceil(box.height)));
  return roi & cv::Rect(0, 0, frame.cols, frame.rows);
}

void draw_snapshot_box(cv::Mat& crop, const SnapshotBox& item, const cv::Rect& roi, double scale) {
  cv::Rect2f box((item.box.x - static_cast<float>(roi.x)) * static_cast<float>(scale),
                 (item.box.y - static_cast<float>(roi.y)) * static_cast<float>(scale),
                 item.box.width * static_cast<float>(scale),
                 item.box.height * static_cast<float>(scale));
  box = box & cv::Rect2f(0.0f, 0.0f, static_cast<float>(crop.cols), static_cast<float>(crop.rows));
  if (box.width <= 1.0f || box.height <= 1.0f) return;
  const double draw_scale = std::clamp(static_cast<double>(std::max(1, std::min(crop.cols, crop.rows))) / 1080.0,
                                       0.45,
                                       1.15);
  const int thickness = std::max(1, static_cast<int>(std::lround(2.0 * draw_scale)));
  cv::rectangle(crop, box, item.color, thickness, cv::LINE_AA);
  if (item.label.empty()) return;
  int baseline = 0;
  const double font_scale = std::max(0.38, 0.62 * draw_scale);
  const auto size = cv::getTextSize(item.label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
  const int x = std::clamp(static_cast<int>(box.x), 0, std::max(0, crop.cols - size.width - 4));
  const int y = std::clamp(static_cast<int>(box.y) - 4, size.height + 4, std::max(size.height + 4, crop.rows - 4));
  cv::rectangle(crop,
                cv::Point(x - 2, y - size.height - baseline - 2),
                cv::Point(x + size.width + 2, y + baseline + 2),
                cv::Scalar(0, 0, 0),
                -1);
  cv::putText(crop, item.label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, font_scale, item.color, thickness, cv::LINE_AA);
}

bool encode_snapshot_jpeg(const cv::Mat& frame,
                          const cv::Rect2f& target,
                          const std::vector<SnapshotBox>& highlight_boxes,
                          std::vector<unsigned char>& jpeg) {
  jpeg.clear();
  if (frame.empty()) return false;
  const cv::Rect roi = snapshot_rect(frame, target);
  if (roi.width <= 0 || roi.height <= 0) return false;
  cv::Mat crop = frame(roi).clone();
  double resize_scale = 1.0;
  const int max_side = 1280;
  const int side = std::max(crop.cols, crop.rows);
  if (side > max_side) {
    resize_scale = static_cast<double>(max_side) / static_cast<double>(side);
    cv::resize(crop, crop, cv::Size(), resize_scale, resize_scale, cv::INTER_AREA);
  }
  for (const auto& item : highlight_boxes) {
    draw_snapshot_box(crop, item, roi, resize_scale);
  }
  for (int quality : {85, 75, 65}) {
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    if (cv::imencode(".jpg", crop, jpeg, params) && !jpeg.empty() && jpeg.size() <= 1900U * 1024U) {
      return true;
    }
  }
  return !jpeg.empty();
}

void send_notification_snapshot(const NotifyConfig& notify,
                                const std::string& message,
                                const cv::Mat& frame,
                                const cv::Rect2f& target,
                                const std::vector<SnapshotBox>& highlight_boxes = {}) {
  send_notification(notify, message);
  if (!notify.enabled) return;
  if (!(notify.provider == "wecom" || notify.provider == "wechat" || notify.provider == "qywx")) return;
  std::vector<unsigned char> jpeg;
  if (!encode_snapshot_jpeg(frame, target, highlight_boxes, jpeg)) {
    std::cout << "Notify image failed: snapshot encode failed\n";
    return;
  }
  const std::string payload = make_wecom_image_payload(jpeg);
  if (payload.empty()) {
    std::cout << "Notify image failed: md5 failed\n";
    return;
  }
  std::string error;
  if (post_json_webhook(notify.webhook, payload, error)) {
    std::cout << "Notify image sent: " << jpeg.size() << " bytes\n";
  } else {
    std::cout << "Notify image failed: " << error << "\n";
  }
}

class FpsMeter {
 public:
  void add(double seconds) {
    total_seconds_ += seconds;
    ++count_;
  }

  double fps() const {
    return total_seconds_ > 0.0 ? static_cast<double>(count_) / total_seconds_ : 0.0;
  }

  double ms() const {
    return count_ > 0 ? total_seconds_ * 1000.0 / static_cast<double>(count_) : 0.0;
  }

  int count() const {
    return count_;
  }

 private:
  double total_seconds_ = 0.0;
  int count_ = 0;
};

template <typename Func>
auto timed(FpsMeter& meter, Func&& func) {
  const auto t0 = std::chrono::steady_clock::now();
  auto result = func();
  const auto t1 = std::chrono::steady_clock::now();
  meter.add(std::chrono::duration<double>(t1 - t0).count());
  return result;
}

double rect_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const cv::Rect2f inter = a & b;
  const double inter_area = inter.area();
  const double union_area = a.area() + b.area() - inter_area;
  return union_area > 0.0 ? inter_area / union_area : 0.0;
}

cv::Point2f box_center(const cv::Rect2f& box) {
  return cv::Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
}

cv::Point2f box_bottom_center(const cv::Rect2f& box) {
  return cv::Point2f(box.x + box.width * 0.5f, box.y + box.height);
}

double point_distance(const cv::Point2f& a, const cv::Point2f& b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

cv::Point2f normalize_or_default(const cv::Point2f& vector, const cv::Point2f& fallback) {
  const double length = std::sqrt(vector.x * vector.x + vector.y * vector.y);
  if (length < 1e-6) return fallback;
  return cv::Point2f(static_cast<float>(vector.x / length), static_cast<float>(vector.y / length));
}

double heading_angle_deg(const cv::Point2f& a, const cv::Point2f& b) {
  const cv::Point2f na = normalize_or_default(a, cv::Point2f(0.0f, -1.0f));
  const cv::Point2f nb = normalize_or_default(b, cv::Point2f(0.0f, -1.0f));
  const double dot = std::clamp(static_cast<double>(na.x * nb.x + na.y * nb.y), -1.0, 1.0);
  return std::acos(dot) * 180.0 / CV_PI;
}

bool point_in_rect(const cv::Point2f& p, const cv::Rect2f& r) {
  return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height;
}

double line_side(const Line& line, const cv::Point2f& p) {
  return (line.p2.x - line.p1.x) * (p.y - line.p1.y) - (line.p2.y - line.p1.y) * (p.x - line.p1.x);
}

bool point_between_two_lines(const cv::Point2f& point, const std::vector<Line>& lines) {
  if (lines.size() < 2) return false;
  const double side_a = line_side(lines[0], point);
  const double side_b = line_side(lines[1], point);
  if (side_a == 0.0 || side_b == 0.0) return true;
  return (side_a > 0.0) != (side_b > 0.0);
}

std::string speed_status(double speed_kmh, double warning_kmh, double violation_kmh) {
  if (speed_kmh < warning_kmh) return "ok";
  if (speed_kmh <= violation_kmh) return "warning";
  return "violation";
}

void add_reason(std::map<int, std::vector<std::string>>& reasons, int track_id, const std::string& reason) {
  if (track_id <= 0) return;
  auto& items = reasons[track_id];
  if (std::find(items.begin(), items.end(), reason) == items.end()) {
    items.push_back(reason);
  }
}

std::string join_reasons(const std::vector<std::string>& reasons) {
  std::ostringstream out;
  for (size_t i = 0; i < reasons.size(); ++i) {
    if (i) out << ";";
    out << reasons[i];
  }
  return out.str();
}

bool is_valid_plate_text(const std::string& text) {
  return !text.empty() && text != "[ocr-decoder-pending]";
}

std::string report_plate_text(const Track& track) {
  if (is_valid_plate_text(track.plate_text)) return track.plate_text;
  return "<\xE6\x9C\xAA\xE8\xAF\x86\xE5\x88\xAB\xE8\xBD\xA6\xE7\x89\x8C>";
}

std::string normalize_plate_text(const std::string& text) {
  std::string normalized;
  for (unsigned char ch : text) {
    if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == '-') continue;
    if (ch == 'I' || ch == 'i') {
      normalized.push_back('1');
    } else if (ch >= 'a' && ch <= 'z') {
      normalized.push_back(static_cast<char>(std::toupper(ch)));
    } else {
      normalized.push_back(static_cast<char>(ch));
    }
  }
  return normalized;
}

bool starts_with_bytes(const std::string& text, const std::string& prefix) {
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

const std::vector<std::string>& province_prefixes() {
  static const std::vector<std::string> prefixes = {
      std::string("\xE4\xBA\xAC", 3), std::string("\xE6\xB4\xA5", 3), std::string("\xE6\xB2\xAA", 3),
      std::string("\xE6\xB8\x9D", 3), std::string("\xE5\x86\x80", 3), std::string("\xE8\xB1\xAB", 3),
      std::string("\xE4\xBA\x91", 3), std::string("\xE8\xBE\xBD", 3), std::string("\xE9\xBB\x91", 3),
      std::string("\xE6\xB9\x98", 3), std::string("\xE7\x9A\x96", 3), std::string("\xE9\xB2\x81", 3),
      std::string("\xE6\x96\xB0", 3), std::string("\xE8\x8B\x8F", 3), std::string("\xE6\xB5\x99", 3),
      std::string("\xE8\xB5\xA3", 3), std::string("\xE9\x84\x82", 3), std::string("\xE6\xA1\x82", 3),
      std::string("\xE7\x94\x98", 3), std::string("\xE6\x99\x8B", 3), std::string("\xE8\x92\x99", 3),
      std::string("\xE9\x99\x95", 3), std::string("\xE5\x90\x89", 3), std::string("\xE9\x97\xBD", 3),
      std::string("\xE8\xB4\xB5", 3), std::string("\xE7\xB2\xA4", 3), std::string("\xE9\x9D\x92", 3),
      std::string("\xE8\x97\x8F", 3), std::string("\xE5\xB7\x9D", 3), std::string("\xE5\xAE\x81", 3),
      std::string("\xE7\x90\xBC", 3)};
  return prefixes;
}

bool is_legal_plate_text(const std::string& text) {
  const std::string plate = normalize_plate_text(text);
  size_t province_len = 0;
  for (const auto& province : province_prefixes()) {
    if (starts_with_bytes(plate, province)) {
      province_len = province.size();
      break;
    }
  }
  if (province_len == 0 || plate.size() <= province_len) return false;

  const unsigned char area = static_cast<unsigned char>(plate[province_len]);
  if (area < 'A' || area > 'Z') return false;

  int ascii_tail = 0;
  for (size_t i = province_len; i < plate.size(); ++i) {
    const unsigned char ch = static_cast<unsigned char>(plate[i]);
    if (!std::isalnum(ch)) return false;
    ++ascii_tail;
  }
  return ascii_tail == 6 || ascii_tail == 7;
}

bool update_plate_lock(Track& track, const std::string& raw_text, float ocr_confidence, std::string& lock_reason) {
  lock_reason.clear();
  if (track.plate_locked) return true;

  const std::string plate = normalize_plate_text(raw_text);
  if (!is_legal_plate_text(plate)) return false;

  track.plate_text = plate;
  ++track.plate_valid_observations;
  const int votes = ++track.plate_votes[plate];
  if (plate == track.last_plate_candidate) {
    ++track.consecutive_plate_count;
  } else {
    track.last_plate_candidate = plate;
    track.consecutive_plate_count = 1;
  }

  const double vote_ratio = track.plate_valid_observations > 0
                                ? static_cast<double>(votes) / static_cast<double>(track.plate_valid_observations)
                                : 0.0;
  if (track.consecutive_plate_count >= 3) {
    lock_reason = "consecutive";
  } else if (ocr_confidence > 0.90f) {
    lock_reason = "high_conf";
  } else if (track.plate_valid_observations >= 3 && vote_ratio > 0.70) {
    lock_reason = "vote";
  }

  if (lock_reason.empty()) return false;
  track.plate_locked = true;
  track.plate_lock_reason = lock_reason;
  track.plate_text = plate;
  return true;
}

std::vector<int> extract_ints(const std::string& text) {
  std::vector<int> values;
  std::string current;
  for (char ch : text) {
    if ((ch >= '0' && ch <= '9') || ch == '-') {
      current.push_back(ch);
    } else if (!current.empty()) {
      values.push_back(std::stoi(current));
      current.clear();
    }
  }
  if (!current.empty()) values.push_back(std::stoi(current));
  return values;
}

std::vector<double> extract_numbers(const std::string& text) {
  std::vector<double> values;
  std::string current;
  for (char ch : text) {
    const bool number_char = (ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E';
    if (number_char) {
      current.push_back(ch);
    } else if (!current.empty()) {
      try {
        values.push_back(std::stod(current));
      } catch (...) {
      }
      current.clear();
    }
  }
  if (!current.empty()) {
    try {
      values.push_back(std::stod(current));
    } catch (...) {
    }
  }
  return values;
}

std::vector<Line> load_lines(const std::string& path) {
  if (path.empty()) return {};
  std::ifstream in(path);
  if (!in) return {};
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  const size_t lines_pos = text.find("\"lines\"");
  if (lines_pos == std::string::npos) return {};
  const size_t homography_pos = text.find("\"homography\"", lines_pos);
  const std::string lines_text = text.substr(lines_pos, homography_pos == std::string::npos ? std::string::npos : homography_pos - lines_pos);
  const auto values = extract_ints(lines_text);
  if (values.size() < 8) return {};
  return {
      Line{cv::Point(values[0], values[1]), cv::Point(values[2], values[3])},
      Line{cv::Point(values[4], values[5]), cv::Point(values[6], values[7])},
  };
}

cv::Mat load_homography(const std::string& path) {
  if (path.empty()) return {};
  std::ifstream in(path);
  if (!in) return {};
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  const size_t homography_pos = text.find("\"homography\"");
  if (homography_pos == std::string::npos) return {};
  const size_t image_pos = text.find("\"image_points\"", homography_pos);
  const size_t world_pos = text.find("\"world_points\"", homography_pos);
  if (image_pos == std::string::npos || world_pos == std::string::npos) return {};

  const std::string image_text = text.substr(image_pos, world_pos - image_pos);
  const auto image_values = extract_numbers(image_text);
  const auto world_values = extract_numbers(text.substr(world_pos));
  if (image_values.size() < 8 || world_values.size() < 8) return {};

  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point2f> world_points;
  for (int i = 0; i < 4; ++i) {
    image_points.emplace_back(static_cast<float>(image_values[i * 2]), static_cast<float>(image_values[i * 2 + 1]));
    world_points.emplace_back(static_cast<float>(world_values[i * 2]), static_cast<float>(world_values[i * 2 + 1]));
  }
  return cv::findHomography(image_points, world_points, 0);
}

bool project_ground_point(const cv::Point2f& point, const cv::Mat& homography, cv::Point2f& metric_point) {
  if (homography.empty()) return false;
  std::vector<cv::Point2f> src{point};
  std::vector<cv::Point2f> dst;
  cv::perspectiveTransform(src, dst, homography);
  if (dst.empty()) return false;
  metric_point = dst[0];
  return true;
}

bool update_continuous_speed(std::deque<SpeedSample>& history,
                             const cv::Point2f& image_point,
                             const cv::Point2f& metric_point,
                             int frame_index,
                             double fps,
                             int window_frames,
                             double& speed_mps,
                             double& speed_kmh) {
  history.push_back(SpeedSample{frame_index, image_point, metric_point});
  const int min_frame = frame_index - window_frames;
  while (history.size() > 2 && history.front().frame < min_frame) {
    history.pop_front();
  }
  if (history.size() < 2 || fps <= 0.0) return false;
  const auto& first = history.front();
  const double elapsed_sec = (frame_index - first.frame) / fps;
  if (elapsed_sec <= 0.0) return false;
  const double dx = metric_point.x - first.metric_point.x;
  const double dy = metric_point.y - first.metric_point.y;
  const double distance_m = std::sqrt(dx * dx + dy * dy);
  speed_mps = distance_m / elapsed_sec;
  speed_kmh = speed_mps * 3.6;
  return true;
}

bool metric_direction_from_history(const std::deque<SpeedSample>& history, cv::Point2f& direction) {
  if (history.size() < 2) return false;
  const cv::Point2f delta = history.back().metric_point - history.front().metric_point;
  const double norm = std::sqrt(delta.x * delta.x + delta.y * delta.y);
  if (norm < 0.15) return false;
  direction = cv::Point2f(static_cast<float>(delta.x / norm), static_cast<float>(delta.y / norm));
  return true;
}

double direction_angle_deg(const cv::Point2f& a, const cv::Point2f& b) {
  double dot = static_cast<double>(a.x) * b.x + static_cast<double>(a.y) * b.y;
  dot = std::max(-1.0, std::min(1.0, dot));
  return std::acos(dot) * 180.0 / CV_PI;
}

bool is_vehicle_class(int class_id) {
  return class_id == 2 || class_id == 3 || class_id == 5 || class_id == 7;
}

bool is_collision_victim_class(int class_id) {
  return class_id == 0 || class_id == 1 || class_id == 3;
}

bool is_collision_vehicle_class(int class_id) {
  return class_id == 2 || class_id == 5 || class_id == 7;
}

std::string collision_victim_name(int class_id) {
  switch (class_id) {
    case 0:
      return "person";
    case 1:
      return "bicycle";
    case 3:
      return "motorcycle";
    default:
      return "victim";
  }
}

bool has_vehicle_detection(const std::vector<Detection>& detections) {
  return std::any_of(detections.begin(), detections.end(), [](const Detection& det) {
    return is_vehicle_class(det.class_id);
  });
}

bool overlaps_known_parked_vehicle(const Detection& det, const std::map<int, Track>& tracks) {
  for (const auto& [id, track] : tracks) {
    (void)id;
    if (!track.parking_reported) continue;
    if (rect_iou(det.box, track.box) >= 0.25) return true;
    if (point_distance(box_center(det.box), box_center(track.box)) <= std::max(track.box.width, track.box.height) * 0.5) {
      return true;
    }
  }
  return false;
}

bool has_new_vehicle_detection(const std::vector<Detection>& detections, const std::map<int, Track>& tracks) {
  return std::any_of(detections.begin(), detections.end(), [&](const Detection& det) {
    return is_vehicle_class(det.class_id) && !overlaps_known_parked_vehicle(det, tracks);
  });
}

std::vector<Detection> infer_standby_vehicle_rois(YoloOnnx& vehicle_detector,
                                                  FpsMeter& vehicle_fps,
                                                  const cv::Mat& frame) {
  std::vector<Detection> detections;
  const int quarter_h = frame.rows / 4;
  if (quarter_h <= 8 || frame.cols <= 8) return detections;

  const cv::Rect top_roi(0, 0, frame.cols, quarter_h);
  const cv::Rect bottom_roi(0, quarter_h * 3, frame.cols, frame.rows - quarter_h * 3);
  for (const cv::Rect& roi : {top_roi, bottom_roi}) {
    if (roi.width <= 8 || roi.height <= 8) continue;
    auto roi_detections = timed(vehicle_fps, [&]() {
      return vehicle_detector.infer(frame(roi), 0.35f, 0.45f);
    });
    for (auto& det : roi_detections) {
      det.box.x += static_cast<float>(roi.x);
      det.box.y += static_cast<float>(roi.y);
      detections.push_back(det);
    }
  }
  return detections;
}

std::string vehicle_name(int class_id) {
  switch (class_id) {
    case 2:
      return "car";
    case 3:
      return "motorcycle";
    case 5:
      return "bus";
    case 7:
      return "truck";
    default:
      return "vehicle";
  }
}

std::map<int, Track> update_vehicle_tracks(std::map<int, Track>& tracks,
                                           const std::vector<Detection>& detections,
                                           int frame_index,
                                           int& next_id) {
  std::map<int, Track> visible;
  std::set<int> used_tracks;
  for (const auto& det : detections) {
    if (!is_vehicle_class(det.class_id)) continue;
    int best_id = 0;
    double best_score = -1.0;
    for (const auto& [id, track] : tracks) {
      if (used_tracks.count(id)) continue;
      const double iou = rect_iou(det.box, track.box);
      const double dist = point_distance(box_center(det.box), box_center(track.box));
      if (iou < 0.20 && dist > 130.0) continue;
      const double score = iou * 1000.0 - dist;
      if (score > best_score) {
        best_score = score;
        best_id = id;
      }
    }
    if (best_id == 0) {
      best_id = next_id++;
      tracks[best_id].id = best_id;
      tracks[best_id].still_anchor = box_center(det.box);
      tracks[best_id].still_start_frame = frame_index;
    }
    used_tracks.insert(best_id);
    auto& track = tracks[best_id];
    track.class_id = det.class_id;
    track.box = det.box;
    track.last_seen = frame_index;
    visible[best_id] = track;
  }

  for (auto it = tracks.begin(); it != tracks.end();) {
    if (frame_index - it->second.last_seen > 30) {
      it = tracks.erase(it);
    } else {
      ++it;
    }
  }
  return visible;
}

std::map<int, PersonTrack> update_person_tracks(std::map<int, PersonTrack>& tracks,
                                                const std::vector<Detection>& detections,
                                                int frame_index,
                                                int& next_id) {
  std::map<int, PersonTrack> visible;
  std::set<int> used_tracks;
  for (const auto& det : detections) {
    if (!is_collision_victim_class(det.class_id)) continue;
    int best_id = 0;
    double best_score = -1.0;
    for (const auto& [id, track] : tracks) {
      if (used_tracks.count(id)) continue;
      const double iou = rect_iou(det.box, track.box);
      const double dist = point_distance(box_center(det.box), box_center(track.box));
      const bool active_collision_track = track.collision_stage != "normal" || track.waiting_reappear_after_disappear;
      const double max_transition_dist = active_collision_track ? 180.0 : 100.0;
      const double min_transition_iou = active_collision_track ? 0.05 : 0.15;
      if (iou < min_transition_iou && dist > max_transition_dist) continue;
      const double score = iou * 1000.0 - dist;
      if (score > best_score) {
        best_score = score;
        best_id = id;
      }
    }
    if (best_id == 0) {
      best_id = next_id++;
      tracks[best_id].id = best_id;
      tracks[best_id].previous_last_seen = frame_index;
    }
    used_tracks.insert(best_id);
    auto& track = tracks[best_id];
    track.previous_last_seen = track.last_seen == 0 ? frame_index : track.last_seen;
    track.class_id = det.class_id;
    track.box = det.box;
    track.last_seen = frame_index;
    visible[best_id] = track;
  }
  for (auto it = tracks.begin(); it != tracks.end();) {
    if (frame_index - it->second.last_seen > 30) {
      it = tracks.erase(it);
    } else {
      ++it;
    }
  }
  return visible;
}

int track_id_for_plate(const cv::Rect2f& plate_box, const std::map<int, Track>& visible_tracks) {
  const cv::Point2f plate_center = box_center(plate_box);
  int best_id = 0;
  double best_score = -1.0;
  int fallback_id = 0;
  double fallback_distance = 1e9;
  for (const auto& [id, track] : visible_tracks) {
    const cv::Point2f vehicle_center = box_center(track.box);
    const double distance = point_distance(vehicle_center, plate_center);
    const double max_fallback_distance = std::max(track.box.width, track.box.height) * 0.75;
    if (distance < fallback_distance && distance <= max_fallback_distance) {
      fallback_distance = distance;
      fallback_id = id;
    }
    if (!point_in_rect(plate_center, track.box)) continue;
    const double score = track.box.area();
    if (score > best_score) {
      best_score = score;
      best_id = id;
    }
  }
  return best_id != 0 ? best_id : fallback_id;
}

cv::Rect2f vehicle_contact_zone(const cv::Rect2f& box) {
  return cv::Rect2f(box.x, box.y + box.height * 0.65f, box.width, box.height * 0.35f);
}

cv::Rect2f expand_rect(const cv::Rect2f& box, float padding) {
  return cv::Rect2f(box.x - padding,
                    box.y - padding,
                    box.width + padding * 2.0f,
                    box.height + padding * 2.0f);
}

double point_to_rect_distance(const cv::Point2f& point, const cv::Rect2f& rect) {
  const double dx = std::max({rect.x - point.x, 0.0f, point.x - (rect.x + rect.width)});
  const double dy = std::max({rect.y - point.y, 0.0f, point.y - (rect.y + rect.height)});
  return std::sqrt(dx * dx + dy * dy);
}

cv::Point2f vehicle_heading_from_history(const Track& track) {
  if (track.metric_history.size() >= 2) {
    const auto& first = track.metric_history.front();
    const auto& last = track.metric_history.back();
    const cv::Point2f delta = last.metric_point - first.metric_point;
    if (point_distance(first.metric_point, last.metric_point) >= 0.15) {
      return normalize_or_default(delta, cv::Point2f(0.0f, -1.0f));
    }
  }
  return cv::Point2f(0.0f, -1.0f);
}

OrientedRect make_vehicle_footprint(const cv::Point2f& center,
                                    const cv::Point2f& heading,
                                    double length_m,
                                    double width_m) {
  OrientedRect rect;
  rect.center = center;
  rect.heading = normalize_or_default(heading, cv::Point2f(0.0f, -1.0f));
  rect.right = cv::Point2f(rect.heading.y, -rect.heading.x);
  rect.length = static_cast<float>(length_m);
  rect.width = static_cast<float>(width_m);
  const cv::Point2f half_forward = rect.heading * (rect.length * 0.5f);
  const cv::Point2f half_right = rect.right * (rect.width * 0.5f);
  rect.corners = {
      rect.center + half_forward + half_right,
      rect.center + half_forward - half_right,
      rect.center - half_forward - half_right,
      rect.center - half_forward + half_right,
  };
  return rect;
}

double point_segment_distance(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
  const cv::Point2f ab = b - a;
  const double denom = ab.x * ab.x + ab.y * ab.y;
  if (denom <= 1e-9) return point_distance(p, a);
  const double t = std::clamp(((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / denom, 0.0, 1.0);
  const cv::Point2f projection(a.x + static_cast<float>(t * ab.x), a.y + static_cast<float>(t * ab.y));
  return point_distance(p, projection);
}

double cross2d(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool segments_intersect(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c, const cv::Point2f& d) {
  const double c1 = cross2d(a, b, c);
  const double c2 = cross2d(a, b, d);
  const double c3 = cross2d(c, d, a);
  const double c4 = cross2d(c, d, b);
  return ((c1 >= 0.0 && c2 <= 0.0) || (c1 <= 0.0 && c2 >= 0.0)) &&
         ((c3 >= 0.0 && c4 <= 0.0) || (c3 <= 0.0 && c4 >= 0.0));
}

bool point_in_polygon(const cv::Point2f& point, const std::vector<cv::Point2f>& polygon) {
  bool inside = false;
  for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
    const auto& pi = polygon[i];
    const auto& pj = polygon[j];
    if (((pi.y > point.y) != (pj.y > point.y)) &&
        (point.x < (pj.x - pi.x) * (point.y - pi.y) / std::max(1e-6f, pj.y - pi.y) + pi.x)) {
      inside = !inside;
    }
  }
  return inside;
}

double polygon_distance(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    const cv::Point2f a1 = a[i];
    const cv::Point2f a2 = a[(i + 1) % a.size()];
    for (size_t j = 0; j < b.size(); ++j) {
      if (segments_intersect(a1, a2, b[j], b[(j + 1) % b.size()])) return 0.0;
    }
  }
  if (point_in_polygon(a.front(), b) || point_in_polygon(b.front(), a)) return 0.0;
  double best = 1e9;
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < b.size(); ++j) {
      best = std::min(best, point_segment_distance(a[i], b[j], b[(j + 1) % b.size()]));
      best = std::min(best, point_segment_distance(b[j], a[i], a[(i + 1) % a.size()]));
    }
  }
  return best;
}

bool project_metric_point(const cv::Point2f& point, const cv::Mat& inverse_homography, cv::Point2f& image_point) {
  if (inverse_homography.empty()) return false;
  std::vector<cv::Point2f> src{point};
  std::vector<cv::Point2f> dst;
  cv::perspectiveTransform(src, dst, inverse_homography);
  if (dst.empty()) return false;
  image_point = dst.front();
  return true;
}

cv::Rect expanded_clamped_roi(const cv::Rect2f& box, const cv::Size& frame_size, float scale_x, float scale_y) {
  const float pad_x = box.width * scale_x;
  const float pad_y = box.height * scale_y;
  cv::Rect2f expanded(box.x - pad_x,
                      box.y - pad_y,
                      box.width + pad_x * 2.0f,
                      box.height + pad_y * 2.0f);
  return expanded & cv::Rect2f(0.0f, 0.0f, static_cast<float>(frame_size.width), static_cast<float>(frame_size.height));
}

std::vector<Detection> infer_plates_in_vehicle_roi(YoloOnnx& plate_detector,
                                                   FpsMeter& plate_fps,
                                                   const cv::Mat& frame,
                                                   const cv::Rect2f& vehicle_box) {
  const cv::Rect roi = expanded_clamped_roi(vehicle_box, frame.size(), 0.08f, 0.10f);
  if (roi.width <= 8 || roi.height <= 8) return {};

  std::vector<Detection> detections = timed(plate_fps, [&]() {
    return plate_detector.infer(frame(roi), 0.20f, 0.45f);
  });
  for (auto& det : detections) {
    det.box.x += static_cast<float>(roi.x);
    det.box.y += static_cast<float>(roi.y);
  }
  return detections;
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string& text) {
  if (text.empty()) return {};
  const int len = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  if (len <= 0) return {};
  std::wstring wide(static_cast<size_t>(len), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(), len);
  return wide;
}

double annotation_scale(const cv::Mat& frame) {
  const int short_side = std::max(1, std::min(frame.cols, frame.rows));
  return std::clamp(static_cast<double>(short_side) / 1080.0, 0.45, 1.15);
}

int annotation_thickness(const cv::Mat& frame) {
  return std::max(1, static_cast<int>(std::lround(2.0 * annotation_scale(frame))));
}

int annotation_marker_radius(const cv::Mat& frame) {
  return std::max(2, static_cast<int>(std::lround(4.0 * annotation_scale(frame))));
}

int annotation_font_px(const cv::Mat& frame) {
  return std::max(12, static_cast<int>(std::lround(24.0 * annotation_scale(frame))));
}

void draw_label_gdi(cv::Mat& frame, const std::string& label, cv::Point origin, const cv::Scalar& color) {
  const std::wstring wide = utf8_to_wide(label);
  if (wide.empty()) return;

  HDC dc = CreateCompatibleDC(nullptr);
  if (!dc) return;
  const int font_px = annotation_font_px(frame);
  const int pad = std::max(3, static_cast<int>(std::lround(4.0 * annotation_scale(frame))));
  HFONT font = CreateFontW(-font_px, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                           OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                           DEFAULT_PITCH | FF_DONTCARE, L"SimHei");
  HGDIOBJ old_font = SelectObject(dc, font);
  SIZE text_size{};
  GetTextExtentPoint32W(dc, wide.c_str(), static_cast<int>(wide.size()), &text_size);

  const int chip_w = std::max(1L, text_size.cx + static_cast<LONG>(pad * 2));
  const int chip_h = std::max(1L, text_size.cy + static_cast<LONG>(pad * 2));
  const int x = std::clamp(origin.x, 0, std::max(0, frame.cols - chip_w));
  const int y = std::clamp(origin.y - chip_h + 2, 0, std::max(0, frame.rows - chip_h));

  BITMAPINFO bmi{};
  bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  bmi.bmiHeader.biWidth = chip_w;
  bmi.bmiHeader.biHeight = -chip_h;
  bmi.bmiHeader.biPlanes = 1;
  bmi.bmiHeader.biBitCount = 32;
  bmi.bmiHeader.biCompression = BI_RGB;

  void* bits = nullptr;
  HBITMAP bitmap = CreateDIBSection(dc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
  if (!bitmap || !bits) {
    SelectObject(dc, old_font);
    DeleteObject(font);
    DeleteDC(dc);
    return;
  }
  HGDIOBJ old_bitmap = SelectObject(dc, bitmap);
  RECT rect{0, 0, chip_w, chip_h};
  HBRUSH bg = CreateSolidBrush(RGB(0, 0, 0));
  FillRect(dc, &rect, bg);
  DeleteObject(bg);
  SetBkMode(dc, TRANSPARENT);
  SetTextColor(dc, RGB(static_cast<int>(color[2]), static_cast<int>(color[1]), static_cast<int>(color[0])));
  TextOutW(dc, pad, pad, wide.c_str(), static_cast<int>(wide.size()));
  GdiFlush();

  const auto* src = static_cast<const unsigned char*>(bits);
  for (int row = 0; row < chip_h; ++row) {
    cv::Vec3b* dst = frame.ptr<cv::Vec3b>(y + row) + x;
    const unsigned char* src_row = src + static_cast<size_t>(row) * static_cast<size_t>(chip_w) * 4U;
    for (int col = 0; col < chip_w; ++col) {
      dst[col][0] = src_row[col * 4 + 0];
      dst[col][1] = src_row[col * 4 + 1];
      dst[col][2] = src_row[col * 4 + 2];
    }
  }

  SelectObject(dc, old_bitmap);
  SelectObject(dc, old_font);
  DeleteObject(bitmap);
  DeleteObject(font);
  DeleteDC(dc);
}
#endif

void draw_label(cv::Mat& frame, const std::string& label, cv::Point origin, const cv::Scalar& color) {
#ifdef _WIN32
  draw_label_gdi(frame, label, origin, color);
#else
  int baseline = 0;
  const double scale = annotation_scale(frame);
  const double font_scale = std::max(0.38, 0.65 * scale);
  const int thickness = annotation_thickness(frame);
  const int pad = std::max(2, static_cast<int>(std::lround(3.0 * scale)));
  const auto size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
  const int x = std::clamp(origin.x, 0, std::max(0, frame.cols - size.width - 4));
  const int y = std::clamp(origin.y, size.height + 4, std::max(size.height + 4, frame.rows - baseline - 4));
  cv::rectangle(frame,
                cv::Point(x - pad, y - size.height - baseline - pad),
                cv::Point(x + size.width + pad, y + baseline + pad),
                cv::Scalar(0, 0, 0),
                -1);
  cv::putText(frame, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv::LINE_AA);
#endif
}

void draw_box(cv::Mat& frame,
              const cv::Rect2f& box,
              const std::string& label,
              const cv::Scalar& color,
              bool label_right = false) {
  const double scale = annotation_scale(frame);
  const int font_px = annotation_font_px(frame);
  cv::rectangle(frame, box, color, annotation_thickness(frame));
  cv::Point origin(static_cast<int>(box.x),
                   std::max(font_px, static_cast<int>(box.y) - static_cast<int>(std::lround(8.0 * scale))));
  if (label_right) {
    origin = cv::Point(static_cast<int>(box.x + box.width + std::lround(6.0 * scale)),
                       std::max(font_px, static_cast<int>(box.y + font_px)));
  }
  draw_label(frame, label, origin, color);
}

void draw_lines(cv::Mat& frame, const std::vector<Line>& lines) {
  const cv::Scalar colors[] = {cv::Scalar(0, 0, 255), cv::Scalar(0, 180, 255)};
  for (size_t i = 0; i < lines.size(); ++i) {
    cv::line(frame, lines[i].p1, lines[i].p2, colors[i % 2],
             std::max(1, annotation_thickness(frame) + 1), cv::LINE_AA);
  }
}

std::string fire_smoke_name(int class_id) {
  if (class_id == 0) return "fire";
  if (class_id == 1) return "smoke";
  return "fire_smoke_" + std::to_string(class_id);
}

cv::Scalar fire_smoke_color(int class_id) {
  if (class_id == 0) return cv::Scalar(0, 80, 255);
  if (class_id == 1) return cv::Scalar(180, 180, 180);
  return cv::Scalar(255, 255, 0);
}

std::string csv_escape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"') escaped += "\"\"";
    else escaped += ch;
  }
  escaped += "\"";
  return escaped;
}

void write_utf8_bom(std::ofstream& file) {
  file << "\xEF\xBB\xBF";
}

}  // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  const auto args = parse_args(argc, argv);
  const std::string source = get_arg(args, "--source", "");
  const std::string vehicle_model = get_arg(args, "--vehicle-model", "speedup-c/models/vehicle_yolo.onnx");
  const std::string plate_model = get_arg(args, "--plate-model", "speedup-c/models/plate_yolo.onnx");
  const std::string ocr_model_dir = get_arg(args, "--ocr-model-dir", "speedup-c/models/ocr/ch_PP-OCRv4_rec_infer");
  const std::string line_json = get_arg(args, "--line-json", "");
  const std::string plate_log_path = get_arg(args, "--plate-log", "");
  const std::string speed_log_path = get_arg(args, "--speed-log", "");
  const std::string parking_log_path = get_arg(args, "--parking-log", "");
  const std::string collision_log_path = get_arg(args, "--collision-log", "");
  const std::string output_video_path = get_arg(args, "--output-video", "");
  const std::string fire_model = get_arg(args, "--fire-model", "");
  const std::string fire_log_path = get_arg(args, "--fire-log", "");
  const std::string notify_provider = get_arg(args, "--notify-provider", "wecom");
  const std::string notify_webhook = get_arg(args, "--notify-webhook", "");
  const std::string notify_webhook_env = get_arg(args, "--notify-webhook-env", "");
  const std::string ocr_trigger = get_arg(args, "--ocr-trigger", "event");
  const std::string yolo_provider = get_arg(args, "--yolo-provider", "cuda");
  const std::string fire_provider = get_arg(args, "--fire-provider", "cuda");
  const std::string trt_cache_dir = get_arg(args, "--trt-cache-dir", "speedup-c/trt-cache");
  const bool trt_fp16 = get_bool_arg(args, "--trt-fp16", true);
  const int imgsz = std::stoi(get_arg(args, "--imgsz", "1280"));
  const int fire_imgsz = std::stoi(get_arg(args, "--fire-imgsz", "640"));
  const double fire_conf = std::stod(get_arg(args, "--fire-conf", "0.25"));
  const double fire_every_sec = std::stod(get_arg(args, "--fire-every-sec", "1.0"));
  const int fire_event_min_hits = std::max(1, std::stoi(get_arg(args, "--fire-event-min-hits", "2")));
  const int fire_event_clear_hits = std::max(1, std::stoi(get_arg(args, "--fire-event-clear-hits", "5")));
  const int vehicle_every = std::max(1, std::stoi(get_arg(args, "--vehicle-every", "3")));
  const bool standby_enable = get_bool_arg(args, "--standby-enable", true);
  const int vehicle_standby_every = std::max(1, std::stoi(get_arg(args, "--vehicle-standby-every", "10")));
  const bool ocr_gpu = get_bool_arg(args, "--ocr-gpu", false);
  const int ocr_every = std::max(1, std::stoi(get_arg(args, "--ocr-every", "10")));
  const int max_ocr_per_frame = std::max(0, std::stoi(get_arg(args, "--max-ocr-per-frame", "4")));
  const double parking_still_sec = std::stod(get_arg(args, "--parking-still-sec", "3.0"));
  const double parking_pixel_thresh = std::stod(get_arg(args, "--parking-pixel-thresh", "5.0"));
  const double preview_scale = std::stod(get_arg(args, "--preview-scale", "1.0"));
  const int preview_every = std::max(1, std::stoi(get_arg(args, "--preview-every", "1")));
  const bool preview_enabled = !get_bool_arg(args, "--no-preview", false);
  const int cpu_threads = std::max(1, std::stoi(get_arg(args, "--cpu-threads", "0")));
  const int continuous_speed_window = std::max(2, std::stoi(get_arg(args, "--continuous-speed-window", "10")));
  const double speed_violation_kmh = std::stod(get_arg(args, "--speed-limit-kmh", "45.0"));
  const double speed_warning_kmh = std::stod(get_arg(args, "--speed-warning-kmh", std::to_string(speed_violation_kmh * 0.875)));
  const int max_frames = std::max(0, std::stoi(get_arg(args, "--max-frames", "0")));
  const bool collision_detect = get_bool_arg(args, "--collision-detect", false);
  const bool parking_collision_mode = get_bool_arg(args, "--parking-collision-mode", false);
  const double vehicle_length_m = std::max(0.1, std::stod(get_arg(args, "--vehicle-length-m", "4.5")));
  const double vehicle_width_m = std::max(0.1, std::stod(get_arg(args, "--vehicle-width-m", "1.8")));
  const double parking_contact_m = std::max(0.0, std::stod(get_arg(args, "--parking-contact-m", "0.15")));
  const double parking_near_m = std::max(parking_contact_m, std::stod(get_arg(args, "--parking-near-m", "0.35")));
  const double parking_approach_from_m = std::max(parking_near_m, std::stod(get_arg(args, "--parking-approach-from-m", "0.9")));
  const double parking_watch_sec = std::max(0.1, std::stod(get_arg(args, "--parking-watch-sec", "2.0")));
  const double parking_still_kmh = std::max(0.0, std::stod(get_arg(args, "--parking-still-kmh", "1.0")));
  const double parking_min_moving_kmh = std::max(parking_still_kmh, std::stod(get_arg(args, "--parking-min-moving-kmh", "3.0")));
  const int parking_contact_frames = std::max(1, std::stoi(get_arg(args, "--parking-contact-frames", "5")));
  const double parking_separate_max_m = std::max(parking_near_m, std::stod(get_arg(args, "--parking-separate-max-m", "0.8")));
  const int parking_scratch_min_evidence = std::max(1, std::stoi(get_arg(args, "--parking-scratch-min-evidence", "3")));
  const double parking_scratch_cooldown_sec = std::max(0.0, std::stod(get_arg(args, "--parking-scratch-cooldown-sec", "3.0")));
  const double collision_vehicle_min_kmh = std::stod(get_arg(args, "--collision-vehicle-min-kmh", "30.0"));
  const double collision_continuous_min_kmh = std::stod(get_arg(args, "--collision-continuous-min-kmh", "20.0"));
  const int collision_vehicle_speed_frames = std::max(1, std::stoi(get_arg(args, "--collision-vehicle-speed-frames", "8")));
  const double collision_direction_max_deg = std::max(0.0, std::stod(get_arg(args, "--collision-direction-max-deg", "25.0")));
  const double collision_contact_px = std::max(0.0, std::stod(get_arg(args, "--collision-contact-px", "35.0")));
  const int collision_contact_frames = std::max(1, std::stoi(get_arg(args, "--collision-contact-frames", "1")));
  const double collision_watch_sec = std::max(0.1, std::stod(get_arg(args, "--collision-watch-sec", "1.5")));
  const double collision_still_px = std::max(0.0, std::stod(get_arg(args, "--collision-still-px", "15.0")));
  const double collision_aspect = std::max(0.1, std::stod(get_arg(args, "--collision-aspect", "1.1")));
  const int collision_disappear_frames = std::max(1, std::stoi(get_arg(args, "--collision-disappear-frames", "3")));
  const double collision_reappear_min_m = std::max(0.0, std::stod(get_arg(args, "--collision-reappear-min-m", "1.0")));
  const double stats_every_sec = std::max(0.1, std::stod(get_arg(args, "--stats-every-sec", "0.5")));

  if (source.empty()) {
    std::cerr << "Missing --source\n";
    return 2;
  }

  try {
    std::cout << std::unitbuf;
    if (cpu_threads > 0) {
      cv::setNumThreads(cpu_threads);
    }
    YoloOnnxOptions yolo_options;
    yolo_options.provider = yolo_provider;
    yolo_options.trt_fp16 = trt_fp16;
    yolo_options.trt_cache_dir = trt_cache_dir;
    YoloOnnx vehicle_detector(vehicle_model, imgsz, yolo_options);
    YoloOnnx plate_detector(plate_model, imgsz, yolo_options);
    std::unique_ptr<YoloOnnx> fire_detector;
    if (!fire_model.empty()) {
      YoloOnnxOptions fire_options = yolo_options;
      fire_options.provider = fire_provider;
      fire_options.trt_cache_dir = trt_cache_dir + "-fire";
      fire_detector = std::make_unique<YoloOnnx>(fire_model, fire_imgsz, fire_options);
    }
    PaddleOcrRecognizer ocr(ocr_model_dir, ocr_gpu, cpu_threads > 0 ? cpu_threads : 8);
    NotifyConfig notify;
    notify.provider = notify_provider;
    notify.webhook = !notify_webhook.empty() ? notify_webhook : get_env_string(notify_webhook_env);
    notify.enabled = !notify.webhook.empty() && (notify.provider == "feishu" || notify.provider == "wecom" ||
                                                notify.provider == "wechat" || notify.provider == "qywx");
    const auto lines = load_lines(line_json);
    const auto homography = load_homography(line_json);
    const cv::Mat inverse_homography = homography.empty() ? cv::Mat() : homography.inv();

    std::ofstream plate_log;
    std::ofstream speed_log;
    std::ofstream parking_log;
    std::ofstream collision_log;
    std::ofstream fire_log;
    if (!plate_log_path.empty()) {
      plate_log.open(plate_log_path);
      write_utf8_bom(plate_log);
      plate_log << "frame,time_sec,track_id,reason,text,plate_confidence,ocr_confidence,lock_status,x1,y1,x2,y2\n";
    }
    if (!speed_log_path.empty()) {
      speed_log.open(speed_log_path);
      write_utf8_bom(speed_log);
      speed_log << "frame,time_sec,object_type,track_id,plate_text,speed_mps,speed_kmh,status,image_x,image_y,world_x_m,world_y_m\n";
    }
    if (!parking_log_path.empty()) {
      parking_log.open(parking_log_path);
      write_utf8_bom(parking_log);
      parking_log << "track_id,start_frame,parking_frame,start_time_sec,parking_time_sec,elapsed_sec,anchor_x,anchor_y,current_x,current_y,plate_text,status\n";
    }
    if (!collision_log_path.empty()) {
      collision_log.open(collision_log_path);
      write_utf8_bom(collision_log);
      if (parking_collision_mode) {
        collision_log << "frame,time_sec,event,vehicle_id_a,vehicle_id_b,distance_m,reason\n";
      } else {
        collision_log << "frame,time_sec,event,person_id,vehicle_id,vehicle_plate,reason\n";
      }
    }
    if (!fire_log_path.empty()) {
      fire_log.open(fire_log_path);
      write_utf8_bom(fire_log);
      fire_log << "frame,time_sec,class_id,class_name,confidence,x1,y1,x2,y2\n";
    }

    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
      std::cerr << "Failed to open source: " << source << "\n";
      return 3;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const int source_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int source_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int fire_every_frames = std::max(1, static_cast<int>(std::round(fps * std::max(0.001, fire_every_sec))));

    cv::VideoWriter output_video;
    if (!output_video_path.empty()) {
      const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
      output_video.open(output_video_path, fourcc, fps, cv::Size(source_width, source_height));
      if (!output_video.isOpened()) {
        std::cerr << "Failed to open output video: " << output_video_path << "\n";
        return 4;
      }
    }

    std::cout << "Input: " << source_width << "x" << source_height << ", " << fps << " FPS, "
              << total_frames << " frames\n";
    std::cout << "YOLO provider: " << yolo_provider;
    if (yolo_provider == "tensorrt") {
      std::cout << ", fp16=" << (trt_fp16 ? "on" : "off") << ", cache=" << trt_cache_dir;
    }
    std::cout << "\n";
    std::cout << "Vehicle YOLO every " << vehicle_every << " frame(s), imgsz=" << imgsz << "\n";
    if (fire_detector) {
      std::cout << "Fire/smoke YOLO every " << fire_every_sec << " sec, imgsz=" << fire_imgsz
                << ", conf=" << fire_conf << ", provider=" << fire_provider << "\n";
    } else {
      std::cout << "Fire/smoke YOLO: disabled\n";
    }
    std::cout << "Vehicle standby: " << (standby_enable ? "enabled" : "disabled");
    if (standby_enable) {
      std::cout << ", scan p1+p4 every " << vehicle_standby_every << " frame(s)";
    }
    std::cout << "\n";
    std::cout << "C++ OCR trigger: " << ocr_trigger << ", every " << ocr_every << " frame(s), max "
              << max_ocr_per_frame << "\n";
    std::cout << "CPU threads: " << (cpu_threads > 0 ? std::to_string(cpu_threads) : std::string("OpenCV default"))
              << "\n";
    std::cout << "Detection lines: " << lines.size() << "\n";
    std::cout << "Homography speed: " << (homography.empty() ? "disabled" : "annotation homography")
              << ", window=" << continuous_speed_window << " frames, limit=" << speed_violation_kmh << " km/h\n";
    if (parking_collision_mode) {
      std::cout << "Parking collision mode: enabled, footprint=" << vehicle_length_m << "x" << vehicle_width_m
                << "m, contact=" << parking_contact_m << "m, near=" << parking_near_m
                << "m, contact_frames=" << parking_contact_frames
                << ", separate_max=" << parking_separate_max_m << "m"
                << ", min_evidence=" << parking_scratch_min_evidence
                << ", scratch_cooldown=" << parking_scratch_cooldown_sec << "s\n";
    }
    std::cout << "Notify: " << (notify.enabled ? notify.provider : "disabled") << "\n";

    FpsMeter vehicle_fps;
    FpsMeter plate_fps;
    FpsMeter fire_fps;
    FpsMeter ocr_fps;
    FpsMeter total_fps;
    std::map<int, Track> vehicle_tracks;
    std::map<int, PersonTrack> person_tracks;
    std::map<std::pair<int, int>, ParkingCollisionState> parking_collision_states;
    int next_vehicle_id = 1;
    int next_person_id = 1;
    int speed_events = 0;
    int parking_events = 0;
    int collision_events = 0;
    int plate_records = 0;
    int fire_records = 0;
    int fire_events = 0;
    bool fire_event_active = false;
    int fire_positive_hits = 0;
    int fire_clear_hits = 0;

    if (preview_enabled) {
      cv::namedWindow("car_speedup", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    }
    if (preview_enabled && preview_scale > 0.0) {
      const int window_w = std::max(320, static_cast<int>(source_width * preview_scale));
      const int window_h = std::max(180, static_cast<int>(source_height * preview_scale));
      cv::resizeWindow("car_speedup", window_w, window_h);
    }

    cv::Mat frame;
    int frame_index = 0;
    auto print_runtime_stats = [&]() {
      std::cout << "Processed " << frame_index << "/" << total_frames << " frames\n";
      std::cout << "Events: parking=" << parking_events << ", speed=" << speed_events
                << ", collision=" << collision_events << ", plate_ocr=" << plate_records
                << ", fire=" << fire_events << ", fire_smoke=" << fire_records << "\n";
      std::cout << "Vehicle YOLO: " << vehicle_fps.fps() << " FPS, " << vehicle_fps.ms() << " ms/call, calls="
                << vehicle_fps.count() << "\n";
      std::cout << "Plate YOLO: " << plate_fps.fps() << " FPS, " << plate_fps.ms() << " ms/call, calls="
                << plate_fps.count() << "\n";
      std::cout << "Fire/smoke YOLO: " << fire_fps.fps() << " FPS, " << fire_fps.ms() << " ms/call, calls="
                << fire_fps.count() << "\n";
      std::cout << "PaddleOCR: " << ocr_fps.fps() << " FPS, " << ocr_fps.ms() << " ms/call, calls="
                << ocr_fps.count() << "\n";
      std::cout << "Total pipeline: " << total_fps.fps() << " FPS, " << total_fps.ms() << " ms/frame, frames="
                << total_fps.count() << "\n";
    };
    auto last_stats_time = std::chrono::steady_clock::now();
    std::vector<Detection> last_objects;
    std::vector<Detection> last_fire_smoke;
    std::map<int, Track> last_visible_vehicles;
    std::map<int, PersonTrack> last_visible_persons;
    bool vehicle_active = !standby_enable;
    int vehicle_active_start_frame = 1;
    while (cap.read(frame)) {
      ++frame_index;
      const auto frame_t0 = std::chrono::steady_clock::now();
      bool vehicle_infer_frame = false;
      std::vector<Detection> objects;
      if (!standby_enable) {
        vehicle_infer_frame = ((frame_index - 1) % vehicle_every) == 0;
        if (vehicle_infer_frame) {
          objects = timed(vehicle_fps, [&]() {
            return vehicle_detector.infer(frame, 0.35f, 0.45f);
          });
          last_objects = objects;
        } else {
          objects = last_objects;
        }
      } else if (!vehicle_active) {
        const bool standby_scan_frame = ((frame_index - 1) % vehicle_standby_every) == 0;
        if (standby_scan_frame) {
          const auto standby_objects = infer_standby_vehicle_rois(vehicle_detector, vehicle_fps, frame);
          if (has_new_vehicle_detection(standby_objects, vehicle_tracks)) {
            vehicle_active = true;
            vehicle_active_start_frame = frame_index;
            vehicle_infer_frame = true;
            objects = timed(vehicle_fps, [&]() {
              return vehicle_detector.infer(frame, 0.35f, 0.45f);
            });
            last_objects = objects;
          } else {
            objects.clear();
            last_objects.clear();
            last_visible_vehicles.clear();
            last_visible_persons.clear();
          }
        } else {
          objects.clear();
        }
      } else {
        vehicle_infer_frame = ((frame_index - vehicle_active_start_frame) % vehicle_every) == 0;
        if (vehicle_infer_frame) {
          objects = timed(vehicle_fps, [&]() {
            return vehicle_detector.infer(frame, 0.35f, 0.45f);
          });
          if (!has_vehicle_detection(objects)) {
            vehicle_active = false;
            objects.clear();
            last_objects.clear();
            last_visible_vehicles.clear();
            last_visible_persons.clear();
          } else {
            last_objects = objects;
          }
        } else {
          objects = last_objects;
        }
      }
      std::vector<Detection> plates;

      std::map<int, Track> visible_vehicles;
      std::map<int, PersonTrack> visible_persons;
      if (vehicle_infer_frame) {
        visible_vehicles = update_vehicle_tracks(vehicle_tracks, objects, frame_index, next_vehicle_id);
        visible_persons = update_person_tracks(person_tracks, objects, frame_index, next_person_id);
        last_visible_vehicles = visible_vehicles;
        last_visible_persons = visible_persons;
      } else {
        visible_vehicles = last_visible_vehicles;
        visible_persons = last_visible_persons;
      }
      std::map<int, std::vector<std::string>> ocr_vehicle_reasons;
      std::map<int, OrientedRect> parking_footprints;
      std::map<std::pair<int, int>, double> parking_pair_distances;

      for (auto& [id, track] : visible_vehicles) {
        auto& stored = vehicle_tracks[id];
        const cv::Point2f center = box_center(track.box);
        if (point_distance(center, stored.still_anchor) <= parking_pixel_thresh) {
          const double elapsed_sec = (frame_index - stored.still_start_frame) / fps;
          if (elapsed_sec >= parking_still_sec && !stored.parking_reported) {
            stored.parking_reported = true;
            ++parking_events;
            std::cout << "Parking id=" << id << " still=" << elapsed_sec << "s point=("
                      << static_cast<int>(center.x) << "," << static_cast<int>(center.y) << ")\n";
            if (parking_log) {
              parking_log << id << "," << stored.still_start_frame << "," << frame_index << ","
                          << stored.still_start_frame / fps << "," << frame_index / fps << ","
                          << elapsed_sec << "," << static_cast<int>(stored.still_anchor.x) << ","
                          << static_cast<int>(stored.still_anchor.y) << "," << static_cast<int>(center.x) << ","
                          << static_cast<int>(center.y) << "," << csv_escape(report_plate_text(stored)) << ",parking\n";
            }
          }
        } else {
          stored.still_anchor = center;
          stored.still_start_frame = frame_index;
          stored.parking_reported = false;
        }
        if (stored.parking_reported) {
          add_reason(ocr_vehicle_reasons, id, "parking");
        }

        if (vehicle_infer_frame && !homography.empty() && track.box.width >= 40.0f && track.box.height >= 40.0f) {
          const cv::Point2f speed_point = box_bottom_center(track.box);
          cv::Point2f metric_point;
          if (project_ground_point(speed_point, homography, metric_point) &&
              update_continuous_speed(stored.metric_history,
                                      speed_point,
                                      metric_point,
                                      frame_index,
                                      fps,
                                      continuous_speed_window,
                                      stored.last_speed_mps,
                                      stored.last_speed_kmh)) {
            const std::string previous_speed_status = stored.last_speed_status;
            stored.last_speed_status = speed_status(stored.last_speed_kmh, speed_warning_kmh, speed_violation_kmh);
            if (stored.last_speed_status == "violation") {
              if (previous_speed_status != "violation") {
                ++speed_events;
                std::cout << "Speeding id=" << id << " speed=" << stored.last_speed_kmh
                          << "km/h plate=" << report_plate_text(stored) << "\n";
              }
              add_reason(ocr_vehicle_reasons, id, "speeding_rt");
            }
            if (speed_log) {
              speed_log << frame_index << "," << frame_index / fps << ",vehicle," << id << ","
                        << csv_escape(report_plate_text(stored)) << ","
                        << stored.last_speed_mps << "," << stored.last_speed_kmh << "," << stored.last_speed_status << ","
                        << static_cast<int>(speed_point.x) << "," << static_cast<int>(speed_point.y) << ","
                        << metric_point.x << "," << metric_point.y << "\n";
            }
          }
        }
        if (stored.last_speed_status == "violation") {
          add_reason(ocr_vehicle_reasons, id, "speeding_rt");
        }
        cv::Point2f collision_direction;
        const bool has_collision_direction = metric_direction_from_history(stored.metric_history, collision_direction);
        if (stored.last_speed_kmh > collision_continuous_min_kmh && has_collision_direction) {
          if (!stored.has_collision_direction_anchor) {
            stored.collision_direction_anchor = collision_direction;
            stored.has_collision_direction_anchor = true;
            stored.collision_speed_frames = 1;
          } else if (direction_angle_deg(stored.collision_direction_anchor, collision_direction) <= collision_direction_max_deg) {
            ++stored.collision_speed_frames;
          } else {
            stored.collision_direction_anchor = collision_direction;
            stored.collision_speed_frames = 1;
          }
        } else {
          stored.collision_speed_frames = 0;
          stored.has_collision_direction_anchor = false;
        }
      }

      if (standby_enable && vehicle_active && !visible_vehicles.empty()) {
        const bool has_non_parked_vehicle = std::any_of(
            visible_vehicles.begin(), visible_vehicles.end(), [&](const auto& item) {
              const auto track_it = vehicle_tracks.find(item.first);
              return track_it == vehicle_tracks.end() || !track_it->second.parking_reported;
            });
        if (!has_non_parked_vehicle) {
          vehicle_active = false;
          last_objects.clear();
          last_visible_vehicles.clear();
          last_visible_persons.clear();
        }
      }

      if (parking_collision_mode && !homography.empty()) {
        for (const auto& [id, vehicle] : visible_vehicles) {
          const auto track_it = vehicle_tracks.find(id);
          if (track_it == vehicle_tracks.end() || track_it->second.metric_history.empty()) continue;
          const auto& stored = track_it->second;
          const cv::Point2f metric_point = stored.metric_history.back().metric_point;
          parking_footprints[id] = make_vehicle_footprint(
              metric_point, vehicle_heading_from_history(stored), vehicle_length_m, vehicle_width_m);
        }

        const int parking_watch_frames = std::max(1, static_cast<int>(parking_watch_sec * fps));
        for (auto it_a = parking_footprints.begin(); it_a != parking_footprints.end(); ++it_a) {
          auto it_b = it_a;
          ++it_b;
          for (; it_b != parking_footprints.end(); ++it_b) {
            const int id_a = it_a->first;
            const int id_b = it_b->first;
            const auto key = std::minmax(id_a, id_b);
            auto& state = parking_collision_states[key];
            const double distance_m = polygon_distance(it_a->second.corners, it_b->second.corners);
            parking_pair_distances[key] = distance_m;
            state.recent_distances.push_back(distance_m);
            while (state.recent_distances.size() > 12) state.recent_distances.pop_front();
            state.max_recent_distance_m = *std::max_element(state.recent_distances.begin(), state.recent_distances.end());

            auto& track_a = vehicle_tracks[id_a];
            auto& track_b = vehicle_tracks[id_b];
            const bool a_moving = track_a.last_speed_kmh >= parking_min_moving_kmh;
            const bool b_moving = track_b.last_speed_kmh >= parking_min_moving_kmh;
            const bool any_moving = a_moving || b_moving;
            const bool contact = distance_m <= parking_contact_m;
            const bool fast_approach = state.max_recent_distance_m >= parking_approach_from_m &&
                                       distance_m <= parking_near_m &&
                                       state.max_recent_distance_m - distance_m >=
                                           (parking_approach_from_m - parking_near_m) * 0.7 &&
                                       any_moving;
            if (contact && any_moving) {
              ++state.contact_count;
            } else if (!contact) {
              state.contact_count = 0;
            }
            const bool confirmed_contact = state.contact_count >= parking_contact_frames;

            if (state.stage == "normal") {
              if (fast_approach) {
                state.stage = "approaching";
              }
            } else if (state.stage == "approaching") {
              if (confirmed_contact) {
                state.stage = "contact";
              } else if (!fast_approach && distance_m > parking_approach_from_m) {
                state.stage = "normal";
                state.contact_count = 0;
              }
            }

            if (state.stage == "contact" && !state.reported_contact) {
              state.reported_contact = true;
              state.stage = "impact_watch";
              state.contact_frame = frame_index;
              state.min_after_contact_m = distance_m;
              state.near_zero_after_contact = distance_m <= parking_contact_m;
              state.speed_a_at_contact = track_a.last_speed_kmh;
              state.speed_b_at_contact = track_b.last_speed_kmh;
              state.heading_a_at_contact = it_a->second.heading;
              state.heading_b_at_contact = it_b->second.heading;
              add_reason(ocr_vehicle_reasons, id_a, "parking_contact");
              add_reason(ocr_vehicle_reasons, id_b, "parking_contact");
              std::cout << "ParkingCollision event=parking_contact frame=" << frame_index
                        << " vehicle_a=" << id_a << " vehicle_b=" << id_b
                        << " distance_m=" << distance_m << " reason=fast_approach_contact\n";
              if (collision_log) {
                collision_log << frame_index << "," << frame_index / fps << ",parking_contact,"
                              << id_a << "," << id_b << "," << distance_m << ",fast_approach_contact\n";
              }
            }

            if (state.stage == "impact_watch" || state.stage == "suspected_scratch") {
              state.min_after_contact_m = std::min(state.min_after_contact_m, distance_m);
              if (distance_m <= parking_contact_m) state.near_zero_after_contact = true;

              std::vector<std::string> scratch_reasons;
              int strong_evidence = 0;
              int weak_evidence = 0;
              if (state.speed_a_at_contact <= parking_still_kmh && track_a.last_speed_kmh >= parking_min_moving_kmh) {
                scratch_reasons.push_back("vehicle_a_passive_move");
                ++strong_evidence;
              }
              if (state.speed_b_at_contact <= parking_still_kmh && track_b.last_speed_kmh >= parking_min_moving_kmh) {
                scratch_reasons.push_back("vehicle_b_passive_move");
                ++strong_evidence;
              }
              if (state.speed_a_at_contact >= parking_min_moving_kmh && track_a.last_speed_kmh <= parking_still_kmh) {
                scratch_reasons.push_back("vehicle_a_sudden_stop");
                ++strong_evidence;
              }
              if (state.speed_b_at_contact >= parking_min_moving_kmh && track_b.last_speed_kmh <= parking_still_kmh) {
                scratch_reasons.push_back("vehicle_b_sudden_stop");
                ++strong_evidence;
              }
              if (heading_angle_deg(state.heading_a_at_contact, it_a->second.heading) >= 35.0) {
                scratch_reasons.push_back("vehicle_a_heading_change");
                ++weak_evidence;
              }
              if (heading_angle_deg(state.heading_b_at_contact, it_b->second.heading) >= 35.0) {
                scratch_reasons.push_back("vehicle_b_heading_change");
                ++weak_evidence;
              }
              if (state.near_zero_after_contact &&
                  distance_m >= parking_near_m &&
                  distance_m <= parking_separate_max_m) {
                scratch_reasons.push_back("near_zero_then_separate");
                ++strong_evidence;
              }
              if (frame_index - state.contact_frame >= parking_watch_frames &&
                  (track_a.last_speed_kmh <= parking_still_kmh || track_b.last_speed_kmh <= parking_still_kmh)) {
                scratch_reasons.push_back("post_contact_long_stop");
                ++weak_evidence;
              }

              const bool enough_evidence =
                  distance_m <= parking_separate_max_m &&
                  (strong_evidence >= parking_scratch_min_evidence ||
                   (strong_evidence >= 1 && strong_evidence + weak_evidence >= parking_scratch_min_evidence));
              const int parking_scratch_cooldown_frames =
                  std::max(1, static_cast<int>(std::lround(parking_scratch_cooldown_sec * fps)));
              const bool scratch_cooldown_active =
                  frame_index - track_a.last_parking_scratch_frame <= parking_scratch_cooldown_frames ||
                  frame_index - track_b.last_parking_scratch_frame <= parking_scratch_cooldown_frames;

              if (!state.reported_scratch && enough_evidence && !scratch_cooldown_active) {
                state.reported_scratch = true;
                state.stage = "suspected_scratch";
                ++collision_events;
                track_a.last_parking_scratch_frame = frame_index;
                track_b.last_parking_scratch_frame = frame_index;
                add_reason(ocr_vehicle_reasons, id_a, "suspected_scratch");
                add_reason(ocr_vehicle_reasons, id_b, "suspected_scratch");
                std::ostringstream reason;
                for (size_t i = 0; i < scratch_reasons.size(); ++i) {
                  if (i > 0) reason << ";";
                  reason << scratch_reasons[i];
                }
                std::cout << "ParkingCollision event=suspected_scratch frame=" << frame_index
                          << " vehicle_a=" << id_a << " vehicle_b=" << id_b
                          << " distance_m=" << distance_m << " reason=" << reason.str() << "\n";
                std::ostringstream message;
                message << "[SII告警] 疑似停车场剐蹭事件"
                        << "\n告警时间(北京时间): " << beijing_time_now()
                        << "\n视频位置: " << cv::format("%.1f", frame_index / fps) << "s"
                        << "\n车辆A: " << id_a
                        << "\n车辆B: " << id_b
                        << "\n两车距离: " << cv::format("%.2f", distance_m) << " m"
                        << "\n原因: " << reason.str()
                        << "\n视频: " << source;
                const std::vector<SnapshotBox> boxes = {
                    {track_a.box, "vehicle " + std::to_string(id_a), cv::Scalar(0, 255, 0)},
                    {track_b.box, "vehicle " + std::to_string(id_b), cv::Scalar(0, 255, 0)}};
                send_notification_snapshot(notify, message.str(), frame, union_rect(track_a.box, track_b.box), boxes);
                if (collision_log) {
                  collision_log << frame_index << "," << frame_index / fps << ",suspected_scratch,"
                                << id_a << "," << id_b << "," << distance_m << ","
                                << csv_escape(reason.str()) << "\n";
                }
              } else if (!state.reported_scratch && enough_evidence && scratch_cooldown_active) {
                state = ParkingCollisionState{};
              } else if (!state.reported_scratch &&
                         distance_m > parking_separate_max_m) {
                state = ParkingCollisionState{};
              }
            }
            state.previous_distance_m = distance_m;
          }
        }

        for (auto it = parking_collision_states.begin(); it != parking_collision_states.end();) {
          if (visible_vehicles.count(it->first.first) == 0 || visible_vehicles.count(it->first.second) == 0) {
            it = parking_collision_states.erase(it);
          } else {
            ++it;
          }
        }
      }

      if (vehicle_infer_frame && !homography.empty()) {
        for (auto& [person_id, person] : visible_persons) {
          auto& stored_person = person_tracks[person_id];
          if (person.box.width < 20.0f || person.box.height < 20.0f) continue;
          const cv::Point2f speed_point = box_bottom_center(person.box);
          cv::Point2f metric_point;
          if (project_ground_point(speed_point, homography, metric_point) &&
              update_continuous_speed(stored_person.metric_history,
                                      speed_point,
                                      metric_point,
                                      frame_index,
                                      fps,
                                      continuous_speed_window,
                                      stored_person.last_speed_mps,
                                      stored_person.last_speed_kmh)) {
            if (speed_log) {
              speed_log << frame_index << "," << frame_index / fps << ","
                        << collision_victim_name(stored_person.class_id) << "," << person_id << ","
                        << ","
                        << stored_person.last_speed_mps << "," << stored_person.last_speed_kmh << ",walking,"
                        << static_cast<int>(speed_point.x) << "," << static_cast<int>(speed_point.y) << ","
                        << metric_point.x << "," << metric_point.y << "\n";
            }
          }
        }
      }

      if (collision_detect) {
        const int collision_watch_frames = std::max(1, static_cast<int>(collision_watch_sec * fps));
        for (auto& [person_id, person] : visible_persons) {
          const cv::Point2f person_point = box_bottom_center(person.box);
          auto& stored_person = person_tracks[person_id];
          const int missed_frames = frame_index - stored_person.previous_last_seen;
          const bool reappeared_after_disappear =
              stored_person.waiting_reappear_after_disappear &&
              missed_frames >= collision_disappear_frames;
          cv::Point2f person_metric_point;
          const bool has_person_metric_point =
              project_ground_point(person_point, homography, person_metric_point);
          stored_person.box = person.box;
          stored_person.last_seen = frame_index;
          int best_vehicle_id = 0;
          double best_distance = 1e9;
          for (const auto& [vehicle_id, vehicle] : visible_vehicles) {
            const auto vehicle_track_it = vehicle_tracks.find(vehicle_id);
            if (vehicle_track_it == vehicle_tracks.end() ||
                !is_collision_vehicle_class(vehicle_track_it->second.class_id)) {
              continue;
            }
            const double vehicle_speed_kmh = vehicle_track_it == vehicle_tracks.end()
                                                 ? 0.0
                                                 : vehicle_track_it->second.last_speed_kmh;
            if (vehicle_speed_kmh <= collision_vehicle_min_kmh) continue;
            if (vehicle_track_it->second.collision_speed_frames < collision_vehicle_speed_frames) continue;
            const cv::Rect2f zone = expand_rect(vehicle_contact_zone(vehicle.box),
                                                static_cast<float>(collision_contact_px));
            const double distance = point_to_rect_distance(person_point, zone);
            if (distance < best_distance) {
              best_distance = distance;
              best_vehicle_id = vehicle_id;
            }
          }
          const bool near_contact = best_vehicle_id > 0 && best_distance <= collision_contact_px;
          if (stored_person.collision_stage == "normal") {
            stored_person.near_count = near_contact ? 1 : 0;
            stored_person.collision_stage = near_contact ? "near_collision" : "normal";
          } else if (stored_person.collision_stage == "near_collision") {
            stored_person.near_count = near_contact ? stored_person.near_count + 1 : 0;
            if (!near_contact) {
              stored_person.collision_stage = "normal";
              stored_person.contact_vehicle_id = 0;
              stored_person.post_contact_points.clear();
              stored_person.waiting_reappear_after_disappear = false;
              stored_person.has_disappear_metric_point = false;
            } else if (stored_person.near_count >= collision_contact_frames) {
              stored_person.collision_stage = "watch_after_contact";
              stored_person.contact_frame = frame_index;
              stored_person.contact_vehicle_id = best_vehicle_id;
              stored_person.post_contact_points.clear();
              stored_person.post_contact_points.push_back(person_point);
              stored_person.waiting_reappear_after_disappear = false;
              stored_person.has_disappear_metric_point = false;
              add_reason(ocr_vehicle_reasons, best_vehicle_id, "collision:contact");
              std::cout << "Collision event=contact frame=" << frame_index << " person=" << person_id
                        << " vehicle=" << best_vehicle_id
                        << " reason=near_vehicle_front;vehicle_speed_kmh="
                        << vehicle_tracks[best_vehicle_id].last_speed_kmh << "\n";
              const auto vehicle_track_it = vehicle_tracks.find(best_vehicle_id);
              const std::string vehicle_plate = vehicle_track_it == vehicle_tracks.end()
                                                    ? "<\xE6\x9C\xAA\xE8\xAF\x86\xE5\x88\xAB\xE8\xBD\xA6\xE7\x89\x8C>"
                                                    : report_plate_text(vehicle_track_it->second);
              if (collision_log) {
                collision_log << frame_index << "," << frame_index / fps << ",contact," << person_id << ","
                              << best_vehicle_id << "," << csv_escape(vehicle_plate)
                              << ",near_vehicle_front;vehicle_speed_kmh="
                              << vehicle_tracks[best_vehicle_id].last_speed_kmh << "\n";
              }
            }
          } else if (stored_person.collision_stage == "watch_after_contact" ||
                     stored_person.collision_stage == "suspected_accident") {
            stored_person.post_contact_points.push_back(person_point);
            while (static_cast<int>(stored_person.post_contact_points.size()) > collision_watch_frames) {
              stored_person.post_contact_points.pop_front();
            }
            const double aspect = person.box.height <= 1.0f ? person.box.width : person.box.width / person.box.height;
            double point_span = 0.0;
            if (stored_person.post_contact_points.size() >= 2) {
              const cv::Point2f first_point = stored_person.post_contact_points.front();
              for (const auto& point : stored_person.post_contact_points) {
                point_span = std::max(point_span, point_distance(first_point, point));
              }
            }
            const bool still_after_contact =
                static_cast<int>(stored_person.post_contact_points.size()) >= std::min(5, collision_watch_frames) &&
                point_span <= collision_still_px;
            const bool horizontal_after_contact = aspect >= collision_aspect;
            const bool reappear_moved_after_disappear =
                reappeared_after_disappear &&
                stored_person.has_disappear_metric_point &&
                has_person_metric_point &&
                point_distance(stored_person.disappear_metric_point, person_metric_point) >= collision_reappear_min_m;
            std::ostringstream reason;
            bool should_report_accident = false;
            bool need_sep = false;
            if (stored_person.waiting_reappear_after_disappear) {
              if (reappear_moved_after_disappear) {
                reason << "person_reappeared_move_m="
                       << point_distance(stored_person.disappear_metric_point, person_metric_point)
                       << ";min_m=" << collision_reappear_min_m;
                should_report_accident = true;
              }
            } else {
              if (horizontal_after_contact) {
                reason << "horizontal_aspect=" << aspect;
                need_sep = true;
                should_report_accident = true;
              }
              if (still_after_contact) {
                if (need_sep) reason << ";";
                reason << "still_span_px=" << point_span;
                should_report_accident = true;
              }
            }
            if (!stored_person.reported && should_report_accident) {
              stored_person.reported = true;
              stored_person.collision_stage = "suspected_accident";
              ++collision_events;
              const int vehicle_id = stored_person.contact_vehicle_id;
              add_reason(ocr_vehicle_reasons, vehicle_id, "collision:suspected_accident");
              std::cout << "Collision event=suspected_accident frame=" << frame_index << " person=" << person_id
                        << " vehicle=" << vehicle_id << " reason=" << reason.str() << "\n";
              const auto vehicle_track_it = vehicle_tracks.find(vehicle_id);
              const std::string vehicle_plate = vehicle_track_it == vehicle_tracks.end()
                                                    ? "<\xE6\x9C\xAA\xE8\xAF\x86\xE5\x88\xAB\xE8\xBD\xA6\xE7\x89\x8C>"
                                                    : report_plate_text(vehicle_track_it->second);
              std::ostringstream message;
              message << "[SII鍛婅] 鐤戜技鎾炰汉浜嬩欢"
                      << "\n鍛婅鏃堕棿(鍖椾含鏃堕棿): " << beijing_time_now()
                      << "\n瑙嗛浣嶇疆: " << cv::format("%.1f", frame_index / fps) << "s"
                      << "\n琛屼汉ID: " << person_id
                      << "\n杞﹁締ID: " << vehicle_id
                      << "\n杞︾墝: " << vehicle_plate
                      << "\n瑙嗛: " << source;
              message.str("");
              message.clear();
              message << "[SII告警] 疑似撞人事件"
                      << "\n告警时间(北京时间): " << beijing_time_now()
                      << "\n视频位置: " << cv::format("%.1f", frame_index / fps) << "s"
                      << "\n行人ID: " << person_id
                      << "\n车辆ID: " << vehicle_id
                      << "\n车牌: " << vehicle_plate
                      << "\n视频: " << source;
              const cv::Rect2f snapshot_box = vehicle_track_it == vehicle_tracks.end()
                                                  ? person.box
                                                  : union_rect(person.box, vehicle_track_it->second.box);
              std::vector<SnapshotBox> boxes;
              if (vehicle_track_it != vehicle_tracks.end()) {
                boxes.push_back({vehicle_track_it->second.box, "vehicle " + std::to_string(vehicle_id), cv::Scalar(0, 255, 0)});
              }
              send_notification_snapshot(notify, message.str(), frame, snapshot_box, boxes);
              if (collision_log) {
                collision_log << frame_index << "," << frame_index / fps << ",suspected_accident," << person_id
                              << "," << vehicle_id << "," << csv_escape(vehicle_plate) << ","
                              << csv_escape(reason.str()) << "\n";
              }
            }
          }
          if constexpr (false) {
            ++stored_person.near_count;
            stored_person.contact_vehicle_id = best_vehicle_id;
            if (stored_person.near_count >= 2 && !stored_person.reported) {
              stored_person.reported = true;
              ++collision_events;
              add_reason(ocr_vehicle_reasons, best_vehicle_id, "collision:contact");
              std::cout << "Collision event=contact frame=" << frame_index << " person=" << person_id
                        << " vehicle=" << best_vehicle_id << " reason=near_vehicle_front\n";
              const auto vehicle_track_it = vehicle_tracks.find(best_vehicle_id);
              const std::string vehicle_plate = vehicle_track_it == vehicle_tracks.end()
                                                    ? "<\xE6\x9C\xAA\xE8\xAF\x86\xE5\x88\xAB\xE8\xBD\xA6\xE7\x89\x8C>"
                                                    : report_plate_text(vehicle_track_it->second);
              std::ostringstream message;
              message << "[SII告警] 疑似撞人事件"
                      << "\n告警时间(北京时间): " << beijing_time_now()
                      << "\n视频位置: " << cv::format("%.1f", frame_index / fps) << "s"
                      << "\n行人ID: " << person_id
                      << "\n车辆ID: " << best_vehicle_id
                      << "\n车牌: " << vehicle_plate
                      << "\n视频: " << source;
              send_notification(notify, message.str());
              if (collision_log) {
                collision_log << frame_index << "," << frame_index / fps << ",contact," << person_id << ","
                              << best_vehicle_id << "," << csv_escape(vehicle_plate) << ",near_vehicle_front\n";
              }
            }
          } else if constexpr (false) {
            stored_person.near_count = 0;
          }
          if (has_person_metric_point) {
            stored_person.last_metric_point = person_metric_point;
            stored_person.has_last_metric_point = true;
          }
          if (reappeared_after_disappear && !stored_person.reported) {
            stored_person.collision_stage = "normal";
            stored_person.contact_vehicle_id = 0;
            stored_person.post_contact_points.clear();
            stored_person.waiting_reappear_after_disappear = false;
            stored_person.has_disappear_metric_point = false;
          }
        }
        for (auto& [person_id, stored_person] : person_tracks) {
          if (visible_persons.count(person_id) > 0) continue;
          if (stored_person.collision_stage != "watch_after_contact" ||
              stored_person.reported ||
              stored_person.waiting_reappear_after_disappear ||
              frame_index - stored_person.last_seen < collision_disappear_frames) {
            continue;
          }
          stored_person.waiting_reappear_after_disappear = true;
          stored_person.disappear_start_frame = frame_index;
          stored_person.has_disappear_metric_point = stored_person.has_last_metric_point;
          if (stored_person.has_last_metric_point) {
            stored_person.disappear_metric_point = stored_person.last_metric_point;
          }
          if (collision_log) {
            collision_log << frame_index << "," << frame_index / fps << ",contact," << person_id
                          << "," << stored_person.contact_vehicle_id << ",,"
                          << "person_disappeared_waiting_reappear;missing_frames="
                          << (frame_index - stored_person.last_seen)
                          << ";reappear_min_m=" << collision_reappear_min_m << "\n";
          }
          continue;
          stored_person.reported = true;
          stored_person.collision_stage = "suspected_accident";
          ++collision_events;
          const int vehicle_id = stored_person.contact_vehicle_id;
          add_reason(ocr_vehicle_reasons, vehicle_id, "collision:suspected_accident");
          const auto vehicle_track_it = vehicle_tracks.find(vehicle_id);
          const std::string vehicle_plate = vehicle_track_it == vehicle_tracks.end()
                                                ? "<\xE6\x9C\xAA\xE8\xAF\x86\xE5\x88\xAB\xE8\xBD\xA6\xE7\x89\x8C>"
                                                : report_plate_text(vehicle_track_it->second);
          std::ostringstream reason;
          reason << "person_disappeared_" << (frame_index - stored_person.last_seen)
                 << "_frames_after_contact";
          std::cout << "Collision event=suspected_accident frame=" << frame_index << " person=" << person_id
                    << " vehicle=" << vehicle_id << " reason=" << reason.str() << "\n";
          std::ostringstream message;
          message << "[SII鍛婅] 鐤戜技鎾炰汉浜嬩欢"
                  << "\n鍛婅鏃堕棿(鍖椾含鏃堕棿): " << beijing_time_now()
                  << "\n瑙嗛浣嶇疆: " << cv::format("%.1f", frame_index / fps) << "s"
                  << "\n琛屼汉ID: " << person_id
                  << "\n杞﹁締ID: " << vehicle_id
                  << "\n杞︾墝: " << vehicle_plate
                  << "\n瑙嗛: " << source;
          message.str("");
          message.clear();
          message << "[SII告警] 疑似撞人事件"
                  << "\n告警时间(北京时间): " << beijing_time_now()
                  << "\n视频位置: " << cv::format("%.1f", frame_index / fps) << "s"
                  << "\n行人ID: " << person_id
                  << "\n车辆ID: " << vehicle_id
                  << "\n车牌: " << vehicle_plate
                  << "\n视频: " << source;
          const cv::Rect2f snapshot_box = vehicle_track_it == vehicle_tracks.end()
                                              ? cv::Rect2f()
                                              : vehicle_track_it->second.box;
          std::vector<SnapshotBox> boxes;
          if (vehicle_track_it != vehicle_tracks.end()) {
            boxes.push_back({vehicle_track_it->second.box, "vehicle " + std::to_string(vehicle_id), cv::Scalar(0, 255, 0)});
          }
          send_notification_snapshot(notify, message.str(), frame, snapshot_box, boxes);
          if (collision_log) {
            collision_log << frame_index << "," << frame_index / fps << ",suspected_accident," << person_id
                          << "," << vehicle_id << "," << csv_escape(vehicle_plate) << ","
                          << csv_escape(reason.str()) << "\n";
          }
        }
      }

      if (frame_index % ocr_every == 0 && max_ocr_per_frame > 0) {
        int ocr_count = 0;
        std::vector<int> plate_candidate_ids;
        for (const auto& [id, track] : visible_vehicles) {
          const auto& stored = vehicle_tracks[id];
          if (stored.plate_locked) continue;
          const bool has_event_reason = ocr_vehicle_reasons.count(id) > 0;
          if (ocr_trigger != "all" && !has_event_reason) continue;
          plate_candidate_ids.push_back(id);
        }

        for (int candidate_id : plate_candidate_ids) {
          if (ocr_count >= max_ocr_per_frame) break;
          const auto visible_it = visible_vehicles.find(candidate_id);
          if (visible_it == visible_vehicles.end()) continue;
          auto roi_plates = infer_plates_in_vehicle_roi(plate_detector, plate_fps, frame, visible_it->second.box);
          plates.insert(plates.end(), roi_plates.begin(), roi_plates.end());

          const std::string reason = ocr_vehicle_reasons.count(candidate_id)
                                         ? join_reasons(ocr_vehicle_reasons[candidate_id])
                                         : "all";
          for (const auto& plate : roi_plates) {
            if (ocr_count >= max_ocr_per_frame) break;
            const int vehicle_id = track_id_for_plate(plate.box, visible_vehicles);
            if (vehicle_id != candidate_id) continue;
            const cv::Rect roi = (plate.box & cv::Rect2f(0, 0, static_cast<float>(frame.cols), static_cast<float>(frame.rows)));
            if (roi.width <= 0 || roi.height <= 0) continue;
            const OcrResult ocr_result = timed(ocr_fps, [&]() {
              return ocr.recognize_with_score(frame(roi).clone());
            });
            ++ocr_count;
            ++plate_records;
            std::string lock_reason;
            bool locked = false;
            if (is_valid_plate_text(ocr_result.text)) {
              locked = update_plate_lock(vehicle_tracks[vehicle_id], ocr_result.text, ocr_result.confidence, lock_reason);
            }
            std::cout << "Plate frame=" << frame_index << " time=" << frame_index / fps << "s id="
                      << vehicle_id << " reason=" << reason
                      << " conf=" << plate.confidence << " ocr_conf=" << ocr_result.confidence
                      << " text=" << ocr_result.text;
            if (locked) {
              std::cout << " lock=" << lock_reason;
            }
            std::cout << "\n";
            if (plate_log) {
              plate_log << frame_index << "," << frame_index / fps << "," << vehicle_id << ","
                        << reason << "," << csv_escape(ocr_result.text) << ","
                        << plate.confidence << "," << ocr_result.confidence << ","
                        << (locked ? lock_reason : "pending") << ","
                        << static_cast<int>(roi.x) << "," << static_cast<int>(roi.y) << ","
                        << static_cast<int>(roi.x + roi.width) << "," << static_cast<int>(roi.y + roi.height) << "\n";
            }
          }
        }
      }

      if (fire_detector && ((frame_index - 1) % fire_every_frames) == 0) {
        last_fire_smoke = timed(fire_fps, [&]() {
          return fire_detector->infer(frame, static_cast<float>(fire_conf), 0.45f);
        });
        int fire_count = 0;
        int smoke_count = 0;
        for (const auto& det : last_fire_smoke) {
          ++fire_records;
          if (det.class_id == 0) ++fire_count;
          if (det.class_id == 1) ++smoke_count;
          if (fire_log) {
            const cv::Rect roi = det.box & cv::Rect2f(0, 0, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
            fire_log << frame_index << "," << frame_index / fps << ","
                     << det.class_id << "," << fire_smoke_name(det.class_id) << ","
                     << det.confidence << ","
                     << roi.x << "," << roi.y << "," << roi.x + roi.width << "," << roi.y + roi.height << "\n";
          }
        }
        if (!last_fire_smoke.empty()) {
          std::cout << "Fire/smoke frame=" << frame_index << " time=" << frame_index / fps
                    << "s fire=" << fire_count << " smoke=" << smoke_count
                    << " count=" << last_fire_smoke.size() << "\n";
          ++fire_positive_hits;
          fire_clear_hits = 0;
          if (!fire_event_active && fire_positive_hits >= fire_event_min_hits) {
            fire_event_active = true;
            ++fire_events;
            std::cout << "Fire incident start frame=" << frame_index << " time=" << frame_index / fps
                      << "s fire=" << fire_count << " smoke=" << smoke_count
                      << " event=" << fire_events << "\n";
            std::ostringstream message;
            message << "[SII告警] 起火/烟雾事件 #" << fire_events
                    << "\n告警时间(北京时间): " << beijing_time_now()
                    << "\n视频位置: " << cv::format("%.1f", frame_index / fps) << "s"
                    << "\n火焰: " << fire_count << "  烟雾: " << smoke_count
                    << "\n视频: " << source;
            cv::Rect2f snapshot_box;
            for (const auto& det : last_fire_smoke) {
              snapshot_box = union_rect(snapshot_box, det.box);
            }
            std::vector<SnapshotBox> boxes;
            if (!visible_vehicles.empty() && snapshot_box.width > 0.0f && snapshot_box.height > 0.0f) {
              const cv::Point2f incident_center = box_center(snapshot_box);
              int best_vehicle_id = 0;
              double best_distance = 1e9;
              for (const auto& [vehicle_id, vehicle] : visible_vehicles) {
                const double distance = point_distance(incident_center, box_center(vehicle.box));
                if (distance < best_distance) {
                  best_distance = distance;
                  best_vehicle_id = vehicle_id;
                }
              }
              if (best_vehicle_id > 0 && best_distance <= std::max(frame.cols, frame.rows) * 0.45) {
                snapshot_box = union_rect(snapshot_box, visible_vehicles[best_vehicle_id].box);
                boxes.push_back({visible_vehicles[best_vehicle_id].box,
                                 "vehicle " + std::to_string(best_vehicle_id),
                                 cv::Scalar(0, 255, 0)});
              }
            }
            send_notification_snapshot(notify, message.str(), frame, snapshot_box, boxes);
          }
        } else {
          fire_positive_hits = 0;
          if (fire_event_active) {
            ++fire_clear_hits;
            if (fire_clear_hits >= fire_event_clear_hits) {
              fire_event_active = false;
              fire_clear_hits = 0;
              std::cout << "Fire incident clear frame=" << frame_index << " time=" << frame_index / fps << "s\n";
            }
          }
        }
      }

      for (const auto& [id, track] : visible_vehicles) {
        const auto& stored = vehicle_tracks[id];
        std::ostringstream label;
        label << vehicle_name(stored.class_id) << " id:" << id;
        if (stored.last_speed_kmh > 0.0) {
          label << " " << cv::format("%.1f", stored.last_speed_kmh) << "km/h " << stored.last_speed_status;
        }
        if (is_valid_plate_text(stored.plate_text)) {
          label << " plate:" << stored.plate_text;
          if (stored.plate_locked) {
            label << " locked";
          }
        }
        if (stored.parking_reported) {
          label << " parking";
        }
        if (collision_detect) {
          for (const auto& [person_id, stored_person] : person_tracks) {
            if (stored_person.contact_vehicle_id == id && stored_person.collision_stage != "normal") {
              label << " " << stored_person.collision_stage;
              break;
            }
          }
        }
        draw_box(frame, track.box, label.str(), cv::Scalar(0, 255, 0));
        cv::circle(frame, box_center(track.box), annotation_marker_radius(frame), cv::Scalar(0, 255, 0), -1);
        if (collision_detect && is_collision_vehicle_class(stored.class_id)) {
          cv::rectangle(frame, vehicle_contact_zone(track.box), cv::Scalar(0, 220, 255),
                        std::max(1, annotation_thickness(frame) - 1));
        }
      }
      if (parking_collision_mode && !inverse_homography.empty()) {
        for (const auto& [id, footprint] : parking_footprints) {
          std::vector<cv::Point> image_corners;
          for (const auto& corner : footprint.corners) {
            cv::Point2f image_point;
            if (project_metric_point(corner, inverse_homography, image_point)) {
              image_corners.emplace_back(static_cast<int>(std::round(image_point.x)),
                                         static_cast<int>(std::round(image_point.y)));
            }
          }
          if (image_corners.size() == 4) {
            const cv::Scalar color = cv::Scalar(255, 180, 0);
            cv::polylines(frame, image_corners, true, color, annotation_thickness(frame), cv::LINE_AA);
          }
        }
        for (const auto& [pair_ids, distance_m] : parking_pair_distances) {
          if (distance_m > parking_approach_from_m) continue;
          const auto fp_a = parking_footprints.find(pair_ids.first);
          const auto fp_b = parking_footprints.find(pair_ids.second);
          if (fp_a == parking_footprints.end() || fp_b == parking_footprints.end()) continue;
          const cv::Point2f metric_mid = (fp_a->second.center + fp_b->second.center) * 0.5f;
          cv::Point2f image_mid;
          if (project_metric_point(metric_mid, inverse_homography, image_mid)) {
            draw_label_gdi(frame,
                           cv::format("%d-%d %.2fm", pair_ids.first, pair_ids.second, distance_m),
                           cv::Point(static_cast<int>(image_mid.x), static_cast<int>(image_mid.y)),
                           cv::Scalar(255, 220, 0));
          }
        }
      }
      for (const auto& [id, person] : visible_persons) {
        const auto& stored_person = person_tracks[id];
        std::ostringstream label;
        label << collision_victim_name(stored_person.class_id) << " id:" << id;
        if (stored_person.last_speed_kmh > 0.0) {
          label << " walk:" << cv::format("%.1f", stored_person.last_speed_kmh) << "km/h";
        }
        if (collision_detect && stored_person.collision_stage != "normal") {
          label << " " << stored_person.collision_stage;
        }
        draw_box(frame, person.box, label.str(), cv::Scalar(0, 165, 255));
        cv::circle(frame, box_bottom_center(person.box), annotation_marker_radius(frame), cv::Scalar(0, 165, 255), -1);
      }
      for (const auto& det : objects) {
        if (det.class_id == 9) {
          draw_box(frame, det.box, "light", cv::Scalar(0, 255, 255));
        }
      }
      for (const auto& plate : plates) {
        const int vehicle_id = track_id_for_plate(plate.box, visible_vehicles);
        const Track* track = vehicle_id > 0 ? &vehicle_tracks[vehicle_id] : nullptr;
        const std::string text = track ? track->plate_text : "";
        std::ostringstream label;
        if (is_valid_plate_text(text)) {
          label << "plate:" << text;
          if (track && track->plate_locked) {
            label << " locked";
          }
        } else {
          label << "plate " << cv::format("%.2f", plate.confidence);
        }
        draw_box(frame, plate.box, label.str(), cv::Scalar(255, 0, 0), true);
      }
      for (const auto& det : last_fire_smoke) {
        std::ostringstream label;
        label << fire_smoke_name(det.class_id) << " " << cv::format("%.2f", det.confidence);
        draw_box(frame, det.box, label.str(), fire_smoke_color(det.class_id));
      }
      const auto frame_t1 = std::chrono::steady_clock::now();
      total_fps.add(std::chrono::duration<double>(frame_t1 - frame_t0).count());

      if (output_video.isOpened()) {
        output_video.write(frame);
      }

      if (preview_enabled && frame_index % preview_every == 0) {
        if (preview_scale > 0.0 && preview_scale != 1.0) {
          cv::Mat preview;
          cv::resize(frame, preview, cv::Size(), preview_scale, preview_scale, cv::INTER_AREA);
          cv::imshow("car_speedup", preview);
        } else {
          cv::imshow("car_speedup", frame);
        }
        if (cv::waitKey(1) == 27) break;
      }
      const auto stats_now = std::chrono::steady_clock::now();
      if (std::chrono::duration<double>(stats_now - last_stats_time).count() >= stats_every_sec) {
        print_runtime_stats();
        last_stats_time = stats_now;
      }
      if (max_frames > 0 && frame_index >= max_frames) {
        break;
      }
    }

    std::cout << "Done. Processed " << frame_index << " frames.\n";
    std::cout << "Events: parking=" << parking_events << ", speed=" << speed_events
              << ", collision=" << collision_events << ", plate_ocr=" << plate_records
              << ", fire=" << fire_events << ", fire_smoke=" << fire_records << "\n";
    std::cout << "Vehicle YOLO: " << vehicle_fps.fps() << " FPS, " << vehicle_fps.ms() << " ms/call, calls="
              << vehicle_fps.count() << "\n";
    std::cout << "Plate YOLO: " << plate_fps.fps() << " FPS, " << plate_fps.ms() << " ms/call, calls="
              << plate_fps.count() << "\n";
    std::cout << "Fire/smoke YOLO: " << fire_fps.fps() << " FPS, " << fire_fps.ms() << " ms/call, calls="
              << fire_fps.count() << "\n";
    std::cout << "PaddleOCR: " << ocr_fps.fps() << " FPS, " << ocr_fps.ms() << " ms/call, calls="
              << ocr_fps.count() << "\n";
    std::cout << "Total pipeline: " << total_fps.fps() << " FPS, " << total_fps.ms() << " ms/frame, frames="
              << total_fps.count() << "\n";
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }

  return 0;
}
