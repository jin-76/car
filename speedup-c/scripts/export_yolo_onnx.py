import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX weights/input.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(input_path))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=False,
            simplify=False,
            opset=12,
            half=args.half,
        )
    )
    if exported.resolve() != output_path.resolve():
        shutil.move(str(exported), str(output_path))


if __name__ == "__main__":
    main()
