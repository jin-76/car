$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$OutDir = Join-Path $Root "speedup-c\models"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Python = Join-Path $Root ".venv-gpu\python.exe"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

& $Python (Join-Path $PSScriptRoot "export_yolo_onnx.py") `
  --input (Join-Path $Root "model\obj-det-model\yolov8n.pt") `
  --output (Join-Path $OutDir "vehicle_yolo.onnx") `
  --imgsz 1280

& $Python (Join-Path $PSScriptRoot "export_yolo_onnx.py") `
  --input (Join-Path $Root "model\ocr-model\plate_yolo.pt") `
  --output (Join-Path $OutDir "plate_yolo.onnx") `
  --imgsz 1280

Write-Host "YOLO ONNX models exported to $OutDir"
Write-Host "Place exported PaddleOCR recognition inference files in $OutDir\ocr"
