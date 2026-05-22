from ultralytics import YOLO


def main() -> None:
    model = YOLO(r"E:\pywork\x-ii\model\obj-det-model\yolov8n.pt")
    model.train(
        data=r"E:\pywork\x-ii\model\fire-smoth-model\data.yaml",
        epochs=5,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        amp=False,
        project=r"E:\pywork\x-ii\runs\detect",
        name="fire_smoke_yolov8n_quick_5e",
        exist_ok=True,
        patience=5,
        save_period=1,
        verbose=False,
    )


if __name__ == "__main__":
    main()
