import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
from paddleocr import PaddleOCR

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PROJECT_DIR / ".paddlex-cache"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def ocr_image(ocr, image):
    result = ocr.predict(image)
    texts = []
    scores = []
    for page in result or []:
        rec_texts = page.get("rec_texts", []) if isinstance(page, dict) else []
        rec_scores = page.get("rec_scores", []) if isinstance(page, dict) else []
        for text, score in zip(rec_texts, rec_scores):
            if score > 0.5:
                texts.append(text)
                scores.append(float(score))
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return "".join(texts), mean_score


def enhance_for_plate_ocr(image):
    image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    image = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(image, (0, 0), 0.55)
    image = cv2.addWeighted(image, 1.45, blurred, -0.45, 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.12, beta=4)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_nearest_images(inputs_dir, nearest_dir):
    nearest_dir.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(inputs_dir.glob("*.png")):
        image = cv2.imread(str(input_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        nearest = cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(nearest_dir / f"{input_path.stem}_nearest_x4.png"), nearest)


def matching_hit_path(hit_dir, stem):
    candidates = sorted(hit_dir.glob(f"{stem}*hit_x4.png"))
    return candidates[0] if candidates else None


def frame_from_stem(stem):
    match = re.search(r"frame_(\d+)", stem)
    return int(match.group(1)) if match else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="runs/plate_sr_1280")
    parser.add_argument(
        "--hit-dir",
        default="model/hat/HiT-SR/results/test_plate_crops_1280_x4/visualization/PlateCrops1280",
    )
    parser.add_argument("--out-csv", default="runs/csv/plate_sr_ocr_1280_basic.csv")
    parser.add_argument(
        "--post-enhance",
        choices=["none", "clarity"],
        default="none",
        help="Apply uniform plate clarity enhancement before OCR.",
    )
    parser.add_argument("--save-enhanced-dir", default="", help="Optional directory for OCR input images after enhancement.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    inputs_dir = base_dir / "inputs"
    nearest_dir = base_dir / "nearest_x4"
    bicubic_dir = base_dir / "bicubic_x4"
    hit_dir = Path(args.hit_dir)
    save_enhanced_dir = Path(args.save_enhanced_dir) if args.save_enhanced_dir else None
    make_nearest_images(inputs_dir, nearest_dir)

    ocr = PaddleOCR(
        lang="ch",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    rows = []
    variants = [
        ("original_crop", lambda stem: inputs_dir / f"{stem}.png"),
        ("nearest_x4", lambda stem: nearest_dir / f"{stem}_nearest_x4.png"),
        ("opencv_bicubic_x4", lambda stem: bicubic_dir / f"{stem}_bicubic_x4.png"),
        ("hit_srf_x4", lambda stem: matching_hit_path(hit_dir, stem)),
    ]

    for input_path in sorted(inputs_dir.glob("*.png")):
        stem = input_path.stem
        for variant, resolve_path in variants:
            image_path = resolve_path(stem)
            if image_path is None or not image_path.exists():
                rows.append([stem, frame_from_stem(stem), variant, "", "", 0, 0, 0.0, "missing"])
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                rows.append([stem, frame_from_stem(stem), variant, str(image_path), "", 0, 0, 0.0, "unreadable"])
                continue
            if args.post_enhance == "clarity":
                image = enhance_for_plate_ocr(image)
                if save_enhanced_dir is not None:
                    enhanced_variant_dir = save_enhanced_dir / variant
                    enhanced_variant_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(enhanced_variant_dir / f"{stem}_{variant}_clarity.png"), image)
            text, score = ocr_image(ocr, image)
            h, w = image.shape[:2]
            rows.append([stem, frame_from_stem(stem), variant, str(image_path), text, w, h, f"{score:.4f}", "ok"])
            print(f"{stem} {variant}: {text or '<empty>'} score={score:.3f}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_id", "frame", "variant", "path", "text", "width", "height", "mean_score", "status"])
        writer.writerows(rows)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
