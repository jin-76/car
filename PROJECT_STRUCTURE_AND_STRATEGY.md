# 项目结构与车牌识别策略说明

## 项目目录结构

- `traffic_mvp.py`
  - 当前项目的主处理脚本。
  - 使用 YOLO + ByteTrack 做车辆检测和目标 ID 跟踪。
  - 可选使用 `model/ocr-model/plate_yolo.pt` 做车牌检测。
  - 对检测到的车牌区域调用 PaddleOCR 做文字识别。
  - 在视频上绘制车辆 ID、车牌文本、速度状态、违停状态和检测线。
  - 可输出标注后的视频和 CSV 日志。
  - 当前新增功能：
    - `--plate-crop-dir`：按车辆目标 ID 保存车牌裁剪图。
    - `--plate-crop-every`：控制同一个目标 ID 每隔多少帧保存一张车牌图。
    - 图片命名格式为 `track_id--index.jpg`，例如 `9--1.jpg`。
    - 保存目录中会同时生成 `manifest.csv`，记录每张图的帧号、时间、坐标、OCR 文本等信息。

- `multi_frame_plate_ocr_fusion.py`
  - 对已经保存的车牌图片做离线多帧 OCR 融合。
  - 输入示例：`runs/sysvideo_first10s_plate_tracks/*.jpg`。
  - 根据文件名中的目标 ID 分组，例如 `9--3.jpg` 属于目标 ID `9`。
  - 输出每张图片的 OCR 明细 CSV。
  - 输出每个目标 ID 的融合车牌号 CSV。
  - 当前默认 OCR 图像增强方式是 `bicubic_x4`。
  - 当前融合方式是：过滤空结果后，对同一个 ID 的多个 OCR 候选做相似度加权投票。

- `annotate_video_line.py`
  - 交互式标线工具。
  - 用来在视频画面中手动画检测线。
  - 输出 JSON 文件，例如 `runs/annotations/traffic_gr0_line.json`。

- `plate_sr_compare.py`
  - 从视频中采样车牌小图。
  - 生成原始裁剪图和 OpenCV 双三次放大图。
  - 可用于对比超分辨率模型的车牌增强效果。

- `ocr_sr_variants.py`
  - 对不同版本的车牌图片运行 PaddleOCR。
  - 用来比较原图、放大图、增强图、超分图的 OCR 效果。

- `bytetrack_custom.yaml`
  - Ultralytics YOLO 跟踪器配置。
  - 用于车辆目标 ID 跟踪。

- `run_gpu.bat` / `run_light_gpu.bat`
  - GPU 环境启动脚本。
  - 会激活项目里的 `.venv-gpu` 环境后运行主脚本。

- `model/ocr-model/plate_yolo.pt`
  - 车牌检测模型。

- `yolov8n.pt`
  - 车辆和通用目标检测模型。

- `movie/`
  - 存放输入视频和中间视频。
  - 当前相关视频：
    - `Sysvideo 4K 8 Megapixel IP Camera Demo traffic car first10s noaudio.mp4`

- `runs/`
  - 存放程序输出结果。
  - 包括标注视频、CSV 日志、车牌裁剪图、最终结果包和实验文件。

## 重要输出目录

- `runs/final/gr0_traffic/`
  - `movie/Traffic IP Camera video [Gr0HpDM8Ki8].mp4` 的最终结果目录。
  - 包含：
    - `traffic_gr0_1280_ocr_speed_parking_plate.mp4`
    - `traffic_gr0_ocr_status_plate.csv`
    - `traffic_gr0_speed_status_plate.csv`
    - `traffic_gr0_parking_plate.csv`

- `runs/sysvideo_first10s_plate_tracks/`
  - 从 10 秒 Sysvideo 视频中按目标 ID 保存的车牌图片。
  - 图片命名格式：
    - `9--1.jpg`
    - `9--2.jpg`
    - `25--12.jpg`
  - `manifest.csv` 记录每张车牌图对应的目标 ID、帧号、时间、OCR 文本、置信度和坐标。

- `runs/sysvideo_first10s_plate_fusion.csv`
  - 10 秒 Sysvideo 视频的多帧 OCR 融合结果。
  - 每一行对应一个车辆目标 ID。

- `runs/sysvideo_first10s_plate_fusion_detail.csv`
  - 多帧 OCR 融合使用的逐图 OCR 明细。
  - 可用于排查某个 ID 为什么融合成某个车牌。

## 之前 `gr0_traffic` 的车牌识别策略

`runs/final/gr0_traffic` 里的最终结果，并不是严格意义上的“多帧 OCR 投票融合”。
它之所以能比较稳定地输出真实车牌，主要依靠主流程里的几层抗噪策略。

### 1. 不是每一帧都做 OCR

默认参数是：

```bat
--ocr-every 10
```

也就是每 10 帧才对车牌做一次 OCR。

这意味着中间很多模糊帧根本不会参与 OCR，因此不会影响车牌文本。

### 2. OCR 有置信度过滤

`ocr_plate()` 里只接受置信度大于 `0.5` 的识别文本。

如果某一帧很模糊，OCR 结果通常为空。
空结果不会覆盖之前已经识别到的车牌。

### 3. 车牌框会继承上一轮文本

`carry_plate_text()` 会把当前车牌框和上一轮车牌框做 IoU 匹配。

如果两个框重叠程度足够高：

```python
IoU >= 0.25
```

就把之前识别到的车牌文本继续带到当前车牌框上。

这样即使某几帧 OCR 没识别出来，画面上也不会立刻丢失车牌号。

### 4. 每个车辆目标 ID 会保存自己的车牌文本

车辆跟踪时，每个车辆都有一个 `track_id`。

主流程会把识别到的车牌文本写入：

```python
track_state[track_id]["plate_text"]
```

后续显示、违停日志、测速事件都可以继续使用这个状态里的车牌文本。

因此，一辆车只要某个时刻识别到了正确车牌，短时间的模糊帧或空 OCR 不会马上把它覆盖掉。

### 5. 车牌和车辆通过空间关系绑定

`plate_text_for_vehicle()` 会优先选择车牌中心点落在车辆框内部的车牌。

如果没有完全落在车辆框内，才会用“距离最近”的车牌作为兜底。

这能减少把旁边车辆的车牌错误绑定到当前车辆 ID 上的情况。

## 为什么之前结果不容易被模糊帧误导

核心逻辑可以概括为：

```text
非空才更新
+ 低置信度过滤
+ 车牌框文本继承
+ 车辆 ID 持久保存车牌
+ 车牌和车辆空间绑定
```

所以模糊帧通常有两种情况：

1. OCR 识别为空。
   - 不会覆盖已有车牌。

2. 当前帧没做 OCR。
   - 继续沿用之前的文本。

之前 `gr0_traffic` 视频中，真实车牌出现得比较稳定，清晰帧比较多，所以这个策略足够有效。

但它并不是绝对防错。
如果某一帧模糊图被 OCR 误识别成一个“非空且置信度较高”的错误文本，理论上仍然可能覆盖旧车牌。

## 当前新增的多帧 OCR 融合实验

针对 10 秒 Sysvideo 视频，目前流程是：

1. 用 `traffic_mvp.py` 按车辆目标 ID 保存多帧车牌图。
2. 用 `multi_frame_plate_ocr_fusion.py` 对同一个 ID 的多张图分别 OCR。
3. 对同一 ID 的多个 OCR 结果做相似度加权融合。
4. 输出每个目标 ID 最终融合车牌。

当前多帧融合结果示例：

- ID `9`：`粤B4UV99`
- ID `25`：`粤BU317G`
- ID `31`：`B3J9U3`
- ID `21`：`4BR167F`

多帧融合对有多张清晰车牌图的目标效果更明显。
如果某个目标只有 1 张图，或者多张图都很糊，仍然需要人工复核。

## 常用重跑命令

### 从 Sysvideo 10 秒视频中按目标 ID 保存车牌图

```bat
python traffic_mvp.py --source "movie\Sysvideo 4K 8 Megapixel IP Camera Demo traffic car first10s noaudio.mp4" --model yolov8n.pt --plate-model "model\ocr-model\plate_yolo.pt" --plate-crop-dir "runs\sysvideo_first10s_plate_tracks" --plate-crop-every 5 --plate-log "runs\sysvideo_first10s_plate_tracks.csv" --no-save --no-preview --device 0 --imgsz 1280 --ocr-every 10
```

### 对保存好的车牌图重新做多帧 OCR 融合

```bat
python multi_frame_plate_ocr_fusion.py --crop-dir runs\sysvideo_first10s_plate_tracks --out-csv runs\sysvideo_first10s_plate_fusion.csv --detail-csv runs\sysvideo_first10s_plate_fusion_detail.csv --variants bicubic_x4
```

### 不重新跑 OCR，只根据已有明细 CSV 重算融合结果

```bat
python multi_frame_plate_ocr_fusion.py --from-detail-csv "runs\sysvideo_first10s_plate_fusion_detail.csv" --out-csv "runs\sysvideo_first10s_plate_fusion.csv"
```
