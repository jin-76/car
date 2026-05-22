cmake_minimum_required(VERSION 3.20)
project(car_speedup LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime root directory")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=path/to/onnxruntime")
endif()

set(PADDLE_INFERENCE_ROOT "" CACHE PATH "Paddle Inference root directory")
if(NOT PADDLE_INFERENCE_ROOT)
  message(FATAL_ERROR "Set -DPADDLE_INFERENCE_ROOT=path/to/paddle_inference")
endif()

add_executable(car_speedup
  src/main.cpp
  src/yolo_onnx.cpp
  src/paddle_ocr.cpp
)

target_include_directories(car_speedup PRIVATE
  include
  "${ONNXRUNTIME_ROOT}/include"
  "${PADDLE_INFERENCE_ROOT}/paddle/include"
)

target_link_directories(car_speedup PRIVATE
  "${ONNXRUNTIME_ROOT}/lib"
  "${PADDLE_INFERENCE_ROOT}/paddle/lib"
)

target_link_libraries(car_speedup PRIVATE
  ${OpenCV_LIBS}
  onnxruntime
  paddle_inference
  winhttp
  advapi32
)

if(MSVC)
  target_compile_options(car_speedup PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(car_speedup PRIVATE -Wall -Wextra -Wpedantic)
endif()
