import os

os.environ.setdefault("YOLO_CONFIG_DIR", r"E:\pywork\x-ii\.ultralytics")

from ultralytics import YOLO


def main() -> None:
    model = YOLO(r"E:\pywork\x-ii\runs\detect\fire_smoke_yolov8n_quick_5e\weights\best.pt")
    model.train(
        data=r"E:\pywork\x-ii\model\fire-smoth-model\data.yaml",
        epochs=20,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        amp=False,
        project=r"E:\pywork\x-ii\runs\detect",
        name="fire_smoke_yolov8n_20e_from_quick",
        exist_ok=True,
        patience=8,
        save_period=5,
        verbose=False,
    )


if __name__ == "__main__":
    main()
