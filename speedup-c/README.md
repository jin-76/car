# C++ inference speedup path

This folder is the C++ inference replacement path for the Python runtime.

Target layout:

- `vehicle_yolo.onnx`: vehicle/person detector exported from `model/obj-det-model/yolov8n.pt`
- `plate_yolo.onnx`: plate detector exported from `model/ocr-model/plate_yolo.pt`
- `ocr/`: PaddleOCR text recognition inference model directory

The intended runtime split is:

- Vehicle/person detection: ONNX Runtime CUDA, or ONNX Runtime TensorRT EP with FP16
- License plate detection: ONNX Runtime CUDA, or ONNX Runtime TensorRT EP with FP16
- OCR text recognition: Paddle Inference GPU

The original Python project stays intact. Generated model files should be placed under `speedup-c/models/`, which is ignored by Git.

## Export models

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File speedup-c\scripts\export_models.ps1
```

This exports the two YOLO `.pt` files to ONNX. PaddleOCR C++ needs Paddle inference model files; place them under:

```text
speedup-c/models/ocr/
```

Expected OCR files:

```text
inference.pdmodel
inference.pdiparams
```

## Build

Install or unpack dependencies locally, then point CMake to them:

```powershell
cmake -S speedup-c -B speedup-c\build `
  -DOpenCV_DIR=C:\opencv\build `
  -DONNXRUNTIME_ROOT=C:\onnxruntime-win-x64-gpu `
  -DPADDLE_INFERENCE_ROOT=C:\paddle_inference_root

cmake --build speedup-c\build --config Release
```

After running `scripts/download_deps.ps1`, the local values are:

```powershell
speedup-c\third_party\cmake-4.2.4-windows-x86_64\bin\cmake.exe -S speedup-c -B speedup-c\build `
  -DOpenCV_DIR="speedup-c\third_party\opencv\build" `
  -DONNXRUNTIME_ROOT="speedup-c\third_party\onnxruntime-win-x64-gpu-1.23.2" `
  -DPADDLE_INFERENCE_ROOT="speedup-c\third_party"
```

## Run

```powershell
speedup-c\build-release\Release\car_speedup.exe `
  --source "movie\Traffic IP Camera video [Gr0HpDM8Ki8].mp4" `
  --vehicle-model speedup-c\models\vehicle_yolo.onnx `
  --plate-model speedup-c\models\plate_yolo.onnx `
  --ocr-model-dir speedup-c\models\ocr\ch_PP-OCRv4_rec_infer `
  --imgsz 1280 `
  --no-preview
```

Fire/smoke detection can run as an extra YOLO stream. It is sampled once per
second by default here, while the last fire/smoke boxes stay visible on all
intermediate frames so saved video playback keeps the original speed:

```powershell
speedup-c\build-release\Release\car_speedup.exe `
  --source "movie\Car Catches on Fire at Witchita Apartment Parking Lot [oQmCTmGlwgA].mp4" `
  --vehicle-model speedup-c\models\vehicle_yolo.onnx `
  --plate-model speedup-c\models\plate_yolo.onnx `
  --ocr-model-dir speedup-c\models\ocr\ch_PP-OCRv4_rec_infer `
  --fire-model speedup-c\models\fire_smoke.onnx `
  --fire-imgsz 640 `
  --fire-conf 0.25 `
  --fire-every-sec 1 `
  --fire-log runs\speedup-c\fire_smoke_realtime\fire_smoke.csv `
  --output-video runs\speedup-c\fire_smoke_realtime\annotated.mp4 `
  --imgsz 1280 `
  --vehicle-every 30 `
  --standby-enable 0 `
  --max-ocr-per-frame 0
```

`speedup-c/models/fire_smoke.onnx` is exported from
`model/fire-smoth-model/model/best.pt`. Keep `--imgsz 1280` for the existing
vehicle model and use `--fire-imgsz 640` for the fire/smoke model.

TensorRT FP16 path:

```powershell
$env:Path = "E:\TensorRT-10.9.0.34\lib;E:\TensorRT-10.9.0.34\bin;" `
  + "$PWD\.venv-gpu\Lib\site-packages\torch\lib;" `
  + "$PWD\speedup-c\third_party\opencv\build\x64\vc16\bin;" `
  + "$PWD\speedup-c\build-release\Release;$env:Path"

speedup-c\build-release\Release\car_speedup.exe `
  --source "movie\Traffic IP Camera video [Gr0HpDM8Ki8].mp4" `
  --vehicle-model speedup-c\models\vehicle_yolo.onnx `
  --plate-model speedup-c\models\plate_yolo.onnx `
  --ocr-model-dir speedup-c\models\ocr\ch_PP-OCRv4_rec_infer `
  --imgsz 1280 `
  --yolo-provider tensorrt `
  --trt-fp16 1 `
  --trt-cache-dir speedup-c\trt-cache `
  --no-preview
```

Use the fp32 ONNX exports for this path. TensorRT builds FP16 engines from them when
`--trt-fp16 1` is enabled. The first run is slower because TensorRT builds and caches
engines; run the same command once to warm `--trt-cache-dir`, then compare the second
run. `onnxruntime_providers_tensorrt.dll` is included in the ONNX Runtime GPU package,
but TensorRT runtime DLLs such as `nvinfer*.dll`, `nvinfer_plugin*.dll`, and
`nvonnxparser*.dll` are under TensorRT `lib`, so `E:\TensorRT-10.9.0.34\lib` must be on
`PATH`. `bin` alone is enough for `trtexec.exe`, but not enough for ONNX Runtime's
TensorRT EP. OpenCV's `opencv_videoio_ffmpeg*.dll` must also be reachable, otherwise
`VideoCapture` can open metadata but fail to read frames.

Current status: C++ scaffolding and model export path are in place. The next implementation step is to port the Python post-processing and tracking logic into `src/main.cpp`.
