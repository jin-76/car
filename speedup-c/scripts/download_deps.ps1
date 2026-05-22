$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$ThirdParty = Join-Path $Root "speedup-c\third_party"
$Archives = Join-Path $ThirdParty "_archives"
New-Item -ItemType Directory -Force -Path $Archives | Out-Null

$Deps = @(
  @{
    Name = "cmake"
    Url = "https://github.com/Kitware/CMake/releases/download/v4.2.4/cmake-4.2.4-windows-x86_64.zip"
    File = "cmake-4.2.4-windows-x86_64.zip"
    Target = "cmake-4.2.4-windows-x86_64"
  },
  @{
    Name = "onnxruntime"
    Url = "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip"
    File = "onnxruntime-win-x64-gpu-1.23.2.zip"
    Target = "onnxruntime-win-x64-gpu-1.23.2"
  },
  @{
    Name = "paddle_inference"
    Url = "https://paddle-inference-lib.bj.bcebos.com/2.6.2/cxx_c/Windows/GPU/x86-64_cuda12.0_cudnn8.9.1_trt8.6.1.6_mkl_avx_vs2019/paddle_inference.zip"
    File = "paddle_inference-win-gpu-cuda12.0-2.6.2.zip"
    Target = "paddle_inference"
  }
)

foreach ($dep in $Deps) {
  $archive = Join-Path $Archives $dep.File
  $target = Join-Path $ThirdParty $dep.Target
  if (-not (Test-Path -LiteralPath $archive)) {
    Write-Host "Downloading $($dep.Name)..."
    Invoke-WebRequest -Uri $dep.Url -OutFile $archive
  } else {
    Write-Host "Archive exists: $archive"
  }
  if (-not (Test-Path -LiteralPath $target)) {
    Write-Host "Extracting $($dep.Name)..."
    Expand-Archive -LiteralPath $archive -DestinationPath $ThirdParty -Force
  } else {
    Write-Host "Extracted exists: $target"
  }
}

$opencvArchive = Join-Path $Archives "opencv-4.13.0-windows.exe"
$opencvTarget = Join-Path $ThirdParty "opencv"
if (-not (Test-Path -LiteralPath $opencvArchive)) {
  Write-Host "Downloading opencv..."
  Invoke-WebRequest -Uri "https://github.com/opencv/opencv/releases/download/4.13.0/opencv-4.13.0-windows.exe" -OutFile $opencvArchive
} else {
  Write-Host "Archive exists: $opencvArchive"
}

if (-not (Test-Path -LiteralPath $opencvTarget)) {
  Write-Host "Extracting opencv..."
  $args = @("-o$ThirdParty", "-y")
  $process = Start-Process -FilePath $opencvArchive -ArgumentList $args -Wait -PassThru -WindowStyle Hidden
  if ($process.ExitCode -ne 0) {
    throw "OpenCV extractor failed with exit code $($process.ExitCode)"
  }
} else {
  Write-Host "Extracted exists: $opencvTarget"
}

Write-Host ""
Write-Host "Done. Dependencies are under:"
Write-Host $ThirdParty
Write-Host ""
Write-Host "CMake example:"
Write-Host "`"$ThirdParty\cmake-4.2.4-windows-x86_64\bin\cmake.exe`" -S speedup-c -B speedup-c\build -DOpenCV_DIR=`"$ThirdParty\opencv\build`" -DONNXRUNTIME_ROOT=`"$ThirdParty\onnxruntime-win-x64-gpu-1.23.2`" -DPADDLE_INFERENCE_ROOT=`"$ThirdParty`""
