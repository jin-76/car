from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


COLORS = {
    0: (0, 80, 255),   # fire
    1: (180, 180, 180),  # smoke
}


def draw_detections(frame, detections, names):
    for cls_id, conf, xyxy in detections:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        color = COLORS.get(cls_id, (0, 255, 0))
        label = f"{names.get(cls_id, cls_id)} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(th + baseline + 4, y1)
        cv2.rectangle(frame, (x1, y_text - th - baseline - 4), (x1 + tw + 6, y_text), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, y_text - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description="Run fire/smoke YOLO once per second while saving a real-time video.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    source = Path(args.source)
    save_path = Path(args.save)
    csv_path = Path(args.csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(fps))

    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {save_path}")

    model = YOLO(args.model)
    names = model.names
    active_detections = []
    rows = []

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % stride == 0:
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            active_detections = []
            second = frame_index / fps
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                active_detections.append((cls_id, conf, xyxy))
                rows.append(
                    {
                        "frame": frame_index,
                        "second": f"{second:.2f}",
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "conf": f"{conf:.4f}",
                        "x1": int(xyxy[0]),
                        "y1": int(xyxy[1]),
                        "x2": int(xyxy[2]),
                        "y2": int(xyxy[3]),
                    }
                )

        draw_detections(frame, active_detections, names)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "second", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"fps={fps:.3f} stride={stride} frames={frame_index}/{total_frames}")
    print(f"video={save_path}")
    print(f"csv={csv_path}")
    print(f"detections={len(rows)}")


if __name__ == "__main__":
    main()
