# Traffic MVP 项目说明

这个项目用于视频或摄像头画面中的车辆、交通灯检测，并支持可选的车牌检测和 OCR 识别。

## 软件工作台

现在可以用桌面工作台启动项目：

```bat
cd /d E:\pywork\x-ii
run_workbench.bat
```

工作台入口文件是 `traffic_workbench.py`。它会复用现有 `traffic_mvp.py` 算法能力，提供：

- 左侧视频预览区。
- 顶部按钮：选择视频、选择摄像头、选择模型、绘制标定线、开始播放、暂停播放、关闭播放、导出结果。
- 右侧参数面板：检测 FPS、OCR FPS、视频 FPS、违停数、超速数、疑似事故数，以及模型、测速、违停、事故检测等开关。
- 底部记录页：运行日志、违停记录、超速记录、疑似事故记录、OCR 记录。
- 每次分析会自动写入 `runs/workbench/时间戳/`，包含 `annotated.mp4`、`speed.csv`、`parking.csv`、`collision.csv`、`ocr.csv` 和车牌截图目录。

当前版本的“暂停播放”先暂停左侧预览；分析进程仍按现有批处理逻辑运行。真正暂停/恢复分析需要下一步把 `traffic_mvp.py` 拆成可被 GUI 调用的逐帧引擎。

## 文件作用

| 文件/目录 | 作用 |
| --- | --- |
| `traffic_workbench.py` | 桌面工作台。负责选择视频/摄像头、配置模型参数、启动分析进程、查看日志与记录。 |
| `run_workbench.bat` | Windows 一键启动工作台脚本。会激活 `.venv-gpu` 环境。 |
| `traffic_mvp.py` | 主程序。读取视频或摄像头，使用 YOLO 检测车辆和交通灯；如果提供车牌 YOLO 模型，还会检测车牌并调用 PaddleOCR 识别文字。 |
| `run_gpu.bat` | Windows 一键运行脚本。会自动激活 `.venv-gpu` 环境，并设置 PaddleOCR/PaddleX 的项目内缓存目录。推荐用它启动程序。 |
| `requirements-gpu.txt` | GPU 环境依赖清单。记录 PyTorch CUDA、Ultralytics、OpenCV、PaddleOCR 等版本，方便以后重装环境。 |
| `.venv-gpu/` | 当前项目专用 Python/Conda 虚拟环境。PyTorch 已配置 CUDA，可使用 RTX 3060 Laptop GPU。 |
| `.paddlex-cache/` | PaddleOCR/PaddleX 的缓存目录。放在项目内，避免写入用户目录时遇到权限问题。 |
| `yolov8n.pt` | Ultralytics YOLOv8 nano 预训练模型。默认用于车辆和交通灯检测。 |
| `output.mp4` | 程序输出的视频结果文件。运行时可用 `--save` 指定其他输出路径。 |
| `__pycache__/` | Python 自动生成的缓存目录。可删除，不影响源码。 |

## 环境状态

当前已配置的环境路径：

```bat
E:\pywork\x-ii\.venv-gpu
```

已验证：

```text
torch 2.11.0+cu128
CUDA 可用: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
ultralytics 8.4.50
PaddleOCR 可导入
```

说明：YOLO/PyTorch 检测走 GPU；当前 PaddleOCR 使用 CPU 版 Paddle，车牌 OCR 会走 CPU。

## 常用命令

使用摄像头：

```bat
cd /d E:\pywork\x-ii
run_gpu.bat --source 0
```

处理视频：

```bat
cd /d E:\pywork\x-ii
run_gpu.bat --source video.mp4 --save output.mp4
```

指定检测模型：

```bat
run_gpu.bat --source video.mp4 --model model\obj-det-model\yolov8n.pt --save output.mp4
```

如果以后有车牌检测模型：

```bat
run_gpu.bat --source video.mp4 --model model\obj-det-model\yolov8n.pt --plate-model model\ocr-model\plate_yolo.pt --save output.mp4
```

查看参数说明：

```bat
run_gpu.bat --help
```

## 重新安装环境

如果 `.venv-gpu` 损坏，可以按下面思路重建：

```bat
C:\ProgramData\miniconda3\Scripts\conda.exe create -p E:\pywork\x-ii\.venv-gpu python=3.11 -y
E:\pywork\x-ii\.venv-gpu\python.exe -m pip install -r E:\pywork\x-ii\requirements-gpu.txt
```

如果下载很慢，这是网络问题，不是依赖冲突。重复执行安装命令通常会继续利用 pip 缓存。

## 注意事项

`traffic_mvp.py` 里的中文提示在当前终端可能显示乱码，但程序参数和逻辑能正常解析。

PowerShell 里出现 `profile.ps1` 执行策略警告时，可以忽略；推荐直接使用 `run_gpu.bat` 运行。

$env:Path += ";C:\Program Files\Labcenter Electronics\Proteus 9 Professional\Tools\Python"---》爬视频用的

  - 绿色框：车辆，比如 car / bus / truck / motorcycle
  - 黄色框：红绿灯，并尝试判断 red / yellow / green
  - 蓝色框：车牌模型检测到的区域
  - 蓝色框文字：

查看视频有哪些下载选项
yt-dlp --js-runtimes node -F https://www.youtube.com/watch?v=Gr0HpDM8Ki8
下载视频
yt-dlp --js-runtimes node -f "137" https://www.youtube.com/watch?v=Gr0HpDM8Ki8

opencv_bicubic_x4加1280提取分辨率效果最好

python traffic_mvp.py --source "movie\Traffic IP Camera video [Gr0HpDM8Ki8].mp4" --model model\obj-det-model\yolov8n.pt --plate-model model\ocr-model\plate_yolo.pt --line-json runs\traffic_gr0_line.json --speed-log runs\traffic_gr0_speed.csv --plate-log runs\traffic_gr0_ocr.csv --save runs\traffic_gr0_1280_ocr_speed.mp4 --no-preview

model\obj-det-model\yolov8n.pt模型能够检测以下对象
person          → 行人
bicycle         → 自行车
car             → 小汽车
motorcycle      → 摩托车
bus             → 公交车 / 大巴车
truck           → 卡车 / 货车
traffic light   → 红绿灯 / 交通信号灯
stop sign       → 停车标志
parking meter   → 停车计时器
train           → 火车

model\ocr-model\plate_yolo.pt模型能够检测以下对象
车牌<--only one object
