@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "PADDLE_PDX_CACHE_HOME=%PROJECT_DIR%.paddlex-cache"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
call "C:\ProgramData\miniconda3\Scripts\activate.bat" "%PROJECT_DIR%.venv-gpu"
python "%PROJECT_DIR%traffic_mvp.py" --device 0 --imgsz 512 --half --process-every 5 --output-scale 0.5 --ocr-every 60 %*
