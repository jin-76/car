$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Exe = Join-Path $Root "speedup-c\build\Release\car_speedup.exe"

& $Exe `
  --source (Join-Path $Root "movie\Traffic IP Camera video [Gr0HpDM8Ki8].mp4") `
  --vehicle-model (Join-Path $Root "speedup-c\models\vehicle_yolo.onnx") `
  --plate-model (Join-Path $Root "speedup-c\models\plate_yolo.onnx") `
  --ocr-model-dir (Join-Path $Root "speedup-c\models\ocr") `
  --imgsz 1280
