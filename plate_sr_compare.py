import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_source(source: str) -> str:
    path = Path(source)
    if path.exists():
        return str(path.resolve())
    for candidate in Path.cwd().glob("**/*.mp4"):
        if candidate.name == path.name:
            return str(candidate.resolve())
    return source


def crop_with_pad(frame, xyxy, pad_ratio: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad_ratio
    py = bh * pad_ratio
    x1 = max(0, int(round(x1 - px)))
    y1 = max(0, int(round(y1 - py)))
    x2 = min(w, int(round(x2 + px)))
    y2 = min(h, int(round(y2 + py)))
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def write_comparison(inputs_dir: Path, bicubic_dir: Path, hit_dir: Path, out_path: Path):
    rows = []
    for input_path in sorted(inputs_dir.glob("*.png")):
        stem = input_path.stem
        hit_candidates = sorted(hit_dir.glob(f"{stem}*.png"))
        if not hit_candidates:
            continue
        original = cv2.imread(str(input_path))
        bicubic = cv2.imread(str(bicubic_dir / f"{stem}_bicubic_x4.png"))
        hit = cv2.imread(str(hit_candidates[0]))
        if original is None or bicubic is None or hit is None:
            continue

        h, w = hit.shape[:2]
        original_x4 = cv2.resize(original, (w, h), interpolation=cv2.INTER_NEAREST)
        bicubic = cv2.resize(bicubic, (w, h), interpolation=cv2.INTER_AREA)

        row = np.hstack([original_x4, bicubic, hit])
        label_h = 34
        label = np.full((label_h, row.shape[1], 3), 255, dtype=np.uint8)
        labels = ["original x4 nearest", "bicubic x4", "HiT-SRF x4"]
        for idx, text in enumerate(labels):
            x = idx * w + 8
            cv2.putText(label, text, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
        rows.append(np.vstack([label, row]))

    if not rows:
        raise RuntimeError(f"No matching SR outputs found in {hit_dir}")

    max_w = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.full((row.shape[0], max_w - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(padded))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="movie/xx-ii.mp4")
    parser.add_argument("--plate-model", default="model/ocr-model/plate_yolo.pt")
    parser.add_argument("--out-dir", default="runs/plate_sr")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--max-crops", type=int, default=8)
    parser.add_argument("--pad-ratio", type=float, default=0.35)
    parser.add_argument("--sr-dir", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    inputs_dir = out_dir / "inputs"
    bicubic_dir = out_dir / "bicubic_x4"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    bicubic_dir.mkdir(parents=True, exist_ok=True)

    if args.sr_dir:
        write_comparison(inputs_dir, bicubic_dir, Path(args.sr_dir), out_dir / "plate_sr_comparison.jpg")
        print(f"Wrote {out_dir / 'plate_sr_comparison.jpg'}")
        return

    source = parse_source(args.source)
    detector = YOLO(args.plate_model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    saved = 0
    frame_index = 0
    try:
        while saved < args.max_crops:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index == 1 or frame_index % max(1, args.sample_every) == 0:
                result = detector.predict(
                    frame,
                    conf=args.conf,
                    device=args.device,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]
                boxes = sorted(result.boxes, key=lambda b: float(b.conf[0]), reverse=True)
                for box in boxes:
                    crop, coords = crop_with_pad(frame, box.xyxy[0].tolist(), args.pad_ratio)
                    if crop.size == 0 or min(crop.shape[:2]) < 6:
                        continue
                    name = f"frame_{frame_index:06d}_plate_{saved + 1:02d}"
                    cv2.imwrite(str(inputs_dir / f"{name}.png"), crop)
                    h, w = crop.shape[:2]
                    bicubic = cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(bicubic_dir / f"{name}_bicubic_x4.png"), bicubic)
                    print(f"saved {name}: conf={float(box.conf[0]):.3f} coords={coords} size={w}x{h}")
                    saved += 1
                    if saved >= args.max_crops:
                        break
    finally:
        cap.release()

    print(f"Saved {saved} crops to {inputs_dir}")


if __name__ == "__main__":
    main()
