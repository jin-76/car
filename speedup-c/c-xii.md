
现在主要有两套识别策略：车辆检测调度策略 和 车牌识别触发策略。下面按当前 C++ 实时版 speedup-c/src/main.cpp 说明，也会补充 Python 版默认值。

  1. 车辆 Vehicle YOLO 策略

  当前是“待机态 / 活跃态”两态。

  待机态：

  - 程序读取输入视频尺寸，例如 1920x1080。
  - 按高度从上到下分成 4 段：
      - p1: 顶部 1/4
      - p2: 中上 1/4
      - p3: 中下 1/4
      - p4: 底部 1/4
  - 待机时只扫描 p1 + p4。
  - p2/p3 在待机态完全不跑 Vehicle YOLO。
  - 待机扫描频率现在默认是 每 10 帧一次：
      - 参数：--vehicle-standby-every
      - 默认值：10
  - 待机扫描时，p1 和 p4 是两个 ROI，所以一次待机扫描会调用两次 Vehicle YOLO。
  - 如果 p1/p4 发现“新车辆”，系统立刻切换到活跃态，并在当前帧跑一次整帧 Vehicle YOLO。

  对应代码在 speedup-c/src/main.cpp:448 和 speedup-c/src/main.cpp:866。

  活跃态：

  - 回到原策略：整帧 Vehicle YOLO。
  - 默认每 3 帧一次：
      - 参数：--vehicle-every
      - 默认值：3
  - 非检测帧复用上一次车辆框和行人框。
  - 活跃态如果整帧检测不到车辆，会清空可见目标并回到待机态。
  - 如果当前只剩已判定为违停的车辆，没有其他非违停车辆，也会回到待机态。
  - 回到待机后，已违停车辆不会继续让系统保持活跃；只有 p1/p4 再发现新车辆才重新激活。

  2. Vehicle YOLO 检测类别

  当前车辆类别按 COCO 类别判断：

  - 2: car
  - 3: motorcycle
  - 5: bus
  - 7: truck

  行人类别：

  - 0: person

  红绿灯绘制用：

  - 9: traffic light

  车辆判断在 speedup-c/src/main.cpp:420。

  3. 跟踪策略

  当前 C++ 版没有用 ByteTrack，而是一个轻量自定义跟踪：

  - 每次 Vehicle YOLO 检测后，用检测框和已有 track 做匹配。
  - 匹配依据：
      - IoU
      - 框中心距离
  - 车辆匹配条件大致是：
      - IoU 至少有一定重叠，或者中心距离不太远
  - 新车辆分配新的 track_id。
  - 超过 30 帧没看到的 track 会删除。

  车辆跟踪在 speedup-c/src/main.cpp:439。

  4. 违停策略

  每个车辆 track 都有一个静止锚点 still_anchor。

  逻辑是：

  - 取车辆框中心点。
  - 如果当前中心点距离锚点不超过 parking_pixel_thresh，认为还在原地。
  - 默认阈值：
      - --parking-pixel-thresh 5.0
  - 如果持续静止时间超过 parking_still_sec，判定违停。
  - 默认时间：
      - --parking-still-sec 3.0
  - 判定违停后：
      - 写入 parking.csv
      - 给该车辆添加 OCR 触发原因 parking
      - 画面标签加 parking

  对应代码在 speedup-c/src/main.cpp:934。

  5. 超速策略

  如果提供了 --line-json 且里面有 homography，程序会做连续测速。

  逻辑是：

  - 对车辆取底部中心点。
  - 用 homography 投影到地面坐标。
  - 在一个滑动窗口内计算位移和速度。
  - 默认窗口：
      - --continuous-speed-window 10
  - 速度状态：
      - < 35 km/h: ok
      - 35-40 km/h: warning
      - > 40 km/h: violation
  - 如果是 violation，给该车辆添加 OCR 触发原因 speeding_rt。

  速度判断在 speedup-c/src/main.cpp:416 和 speedup-c/src/main.cpp:960。

  6. 撞行人策略

  碰撞检测默认是关的：

  - 参数：--collision-detect
  - 默认：false

  打开后逻辑是：

  - 对每个行人，取行人底部中心点。
  - 对每辆车，取车辆下部接触区域。
  - 扩大接触区域约 35 像素。
  - 如果行人点靠近车辆接触区，并连续满足一定条件，就记录碰撞事件。
  - 触发后给相关车辆添加 OCR 原因：
      - collision:contact

  对应代码在 speedup-c/src/main.cpp:1027。

  7. 车牌 Plate YOLO 策略

  现在已经恢复为事件触发。

  默认：

  - --ocr-trigger event

  也就是说，默认不会对所有车辆做车牌检测和 OCR。

  只有车辆有以下事件原因之一，才会进入车牌检测候选：

  - parking
  - speeding_rt
  - collision:contact
  - 以后如果扩展其他事件原因，也可以进这个 map

  车牌检测不是整帧扫，而是：

  - 对事件车辆的车辆框做一个轻微扩大 ROI。
  - 在车辆 ROI 内跑 Plate YOLO。
  - Plate YOLO 检出的框再映射回整帧坐标。

  ROI 扩大比例：

  - 横向约 8%
  - 纵向约 10%

  对应代码在 speedup-c/src/main.cpp:616。

  8. OCR 文字识别策略

  PaddleOCR 只对 Plate YOLO 检出的车牌框做文字识别。

  当前 OCR 触发条件：

  - 当前帧号满足：
      - frame_index % ocr_every == 0
  - 默认：
      - --ocr-every 10
  - 每帧最多 OCR 数：
      - --max-ocr-per-frame
      - 当前运行常用是 4
  - 车辆没有锁定车牌文本：
      - stored.plate_locked == false
  - 车辆必须有事件原因，除非你显式传：
      - --ocr-trigger all

  关键点：
  现在默认状态下，普通车辆不会做车辆 ROI 内 Plate YOLO，也不会 OCR。

  候选筛选在 speedup-c/src/main.cpp:1064。

  9. 车牌锁定策略

  OCR 出来的文本不是直接永久采用，会做合法性和锁定判断。

  当前逻辑：

  - 先规范化车牌文本：
      - 去空格、换行、短横线
      - 小写转大写
      - I/i 归一成 1
  - 检查是否像中国车牌：
      - 省份汉字前缀
      - 后面字母数字长度符合规则
  - 满足以下任一条件会锁定：
      - 连续 3 次识别到同一个合法车牌
      - OCR 置信度 > 0.90
      - 至少 3 次合法观测，且同一车牌投票比例 > 0.70

  锁定后：

  - 该车辆后续不再进入 OCR 候选。
  - 画面标签会显示 locked。

  对应代码在 speedup-c/src/main.cpp:284。

  10. 当前默认策略总结

  默认现在是：

  待机：
    每 10 帧扫一次 p1 + p4
    p2/p3 不扫
    发现新车辆 -> 活跃

  活跃：
    每 3 帧整帧 Vehicle YOLO
    没车 -> 待机
    只剩违停车 -> 待机

  事件判断：
    车辆静止 3 秒 -> parking
    速度 > 40 km/h -> speeding_rt
    撞行人打开后触发 -> collision:contact

  车牌检测：
    默认只对事件车辆做车辆 ROI 内 Plate YOLO

  文字 OCR：
    默认只对事件车辆 ROI 内检测到的车牌框做 OCR
    普通车辆不做车牌检测，也不做 OCR

  如果你想临时恢复“所有车辆都识别车牌”，可以显式传：

  --ocr-trigger all

  否则默认就是事件触发。 看上去合理吗