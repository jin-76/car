import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PROJECT_DIR / ".paddlex-cache"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
from paddleocr import PaddleOCR


def parse_crop_name(path):
    match = re.fullmatch(r"(\d+)--(\d+)", path.stem)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def normalize_text(text):
    text = text.upper()
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def enhance_basic(image):
    h, w = image.shape[:2]
    scale = max(2, min(4, 240 // max(1, min(h, w))))
    image = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.10, beta=5)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def enhance_aggressive(image):
    h, w = image.shape[:2]
    scale = max(3, min(8, 480 // max(1, min(h, w))))
    image = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(6, 6)).apply(l_channel)
    image = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(image, (0, 0), 0.8)
    image = cv2.addWeighted(image, 2.2, blurred, -1.2, 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_variants(image, selected_variants):
    h, w = image.shape[:2]
    variants = {
        "original": image,
        "bicubic_x4": cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC),
        "basic": enhance_basic(image),
        "aggressive": enhance_aggressive(image),
    }
    return [(name, variants[name]) for name in selected_variants]


def ocr_image(ocr, image):
    result = ocr.predict(image)
    texts = []
    scores = []
    for page in result or []:
        rec_texts = page.get("rec_texts", []) if isinstance(page, dict) else []
        rec_scores = page.get("rec_scores", []) if isinstance(page, dict) else []
        for text, score in zip(rec_texts, rec_scores):
            if score > 0.35:
                texts.append(text)
                scores.append(float(score))
    return "".join(texts), (sum(scores) / len(scores) if scores else 0.0)


def choose_best(candidates):
    nonempty = [row for row in candidates if row["norm_text"]]
    if not nonempty:
        return max(candidates, key=lambda row: row["score"], default=None)
    return max(nonempty, key=lambda row: (row["score"] * max(1, len(row["norm_text"])), row["score"]))


def fuse_track(rows):
    valid = [row for row in rows if row["norm_text"]]
    if not valid:
        return "", 0.0, "", 0

    exact_votes = defaultdict(float)
    examples = {}
    for row in valid:
        weight = row["score"] * max(1, len(row["norm_text"]))
        exact_votes[row["norm_text"]] += weight
        examples.setdefault(row["norm_text"], row["text"])

    exact_text, exact_weight = max(exact_votes.items(), key=lambda item: item[1])
    consensus_scores = {}
    for candidate in valid:
        candidate_text = candidate["norm_text"]
        score = 0.0
        for other in valid:
            ratio = SequenceMatcher(None, candidate_text, other["norm_text"]).ratio()
            score += other["score"] * ratio
        score *= max(0.5, min(1.25, len(candidate_text) / 7.0))
        consensus_scores[candidate_text] = max(consensus_scores.get(candidate_text, 0.0), score)
    consensus_text, consensus_weight = max(consensus_scores.items(), key=lambda item: item[1])

    length_votes = Counter()
    for row in valid:
        length_votes[len(row["norm_text"])] += row["score"]
    target_len = length_votes.most_common(1)[0][0]

    chars = []
    char_conf = []
    for index in range(target_len):
        votes = defaultdict(float)
        for row in valid:
            text = row["norm_text"]
            if len(text) != target_len:
                continue
            votes[text[index]] += row["score"]
        if not votes:
            continue
        char, weight = max(votes.items(), key=lambda item: item[1])
        chars.append(char)
        char_conf.append(weight)

    char_vote_text = "".join(chars)
    if consensus_weight >= exact_weight / max(1, len(exact_text)):
        fused = consensus_text
        method = "similarity_vote"
        confidence = consensus_weight / max(1, len(valid))
    elif len(char_vote_text) >= 4:
        fused = char_vote_text
        method = "char_vote"
        confidence = sum(char_conf) / max(1, target_len)
    else:
        fused = exact_text
        method = "exact_vote"
        confidence = exact_weight / max(1, len(exact_text))

    return fused, confidence, method, len(valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-dir", default="runs/sysvideo_first10s_plate_tracks")
    parser.add_argument("--out-csv", default="runs/sysvideo_first10s_plate_fusion.csv")
    parser.add_argument("--detail-csv", default="runs/sysvideo_first10s_plate_fusion_detail.csv")
    parser.add_argument(
        "--variants",
        default="bicubic_x4",
        help="Comma-separated OCR variants: original,bicubic_x4,basic,aggressive.",
    )
    parser.add_argument("--from-detail-csv", default="", help="Recompute fusion summary from an existing detail CSV.")
    args = parser.parse_args()

    if args.from_detail_csv:
        best_by_image = {}
        with Path(args.from_detail_csv).open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row["track_id"] = int(row["track_id"])
                row["crop_index"] = int(row["crop_index"])
                row["score"] = float(row["score"])
                key = (row["track_id"], row["crop_index"], row["image"])
                current = best_by_image.get(key)
                if current is None or (row["score"] * max(1, len(row["norm_text"]))) > (
                    current["score"] * max(1, len(current["norm_text"]))
                ):
                    best_by_image[key] = row
        best_rows = list(best_by_image.values())
        write_summary(best_rows, Path(args.out_csv))
        return

    crop_dir = Path(args.crop_dir)
    image_paths = sorted(crop_dir.glob("*.jpg"), key=lambda p: parse_crop_name(p))
    selected_variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    allowed_variants = {"original", "bicubic_x4", "basic", "aggressive"}
    invalid_variants = [name for name in selected_variants if name not in allowed_variants]
    if invalid_variants:
        raise ValueError(f"Invalid OCR variants: {invalid_variants}")

    ocr = PaddleOCR(
        lang="ch",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    detail_rows = []
    best_rows = []
    for image_path in image_paths:
        track_id, crop_index = parse_crop_name(image_path)
        if track_id is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        candidates = []
        for variant, variant_image in make_variants(image, selected_variants):
            text, score = ocr_image(ocr, variant_image)
            row = {
                "track_id": track_id,
                "crop_index": crop_index,
                "image": image_path.name,
                "variant": variant,
                "text": text,
                "norm_text": normalize_text(text),
                "score": score,
            }
            candidates.append(row)
            detail_rows.append(row)

        best = choose_best(candidates)
        if best is not None:
            best_rows.append(best)
            print(
                f"id={track_id} crop={crop_index} "
                f"best={best['norm_text'] or '<empty>'} "
                f"variant={best['variant']} score={best['score']:.3f}",
                flush=True,
            )

    grouped = defaultdict(list)
    for row in best_rows:
        grouped[row["track_id"]].append(row)

    detail_csv = Path(args.detail_csv)
    detail_csv.parent.mkdir(parents=True, exist_ok=True)

    with detail_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "crop_index", "image", "variant", "text", "norm_text", "score"])
        writer.writeheader()
        writer.writerows(detail_rows)

    write_summary(best_rows, Path(args.out_csv))
    print(f"Wrote {detail_csv}", flush=True)


def write_summary(best_rows, out_csv):
    grouped = defaultdict(list)
    for row in best_rows:
        grouped[int(row["track_id"])].append(row)

    summary_rows = []
    for track_id in sorted(grouped):
        rows = sorted(grouped[track_id], key=lambda row: int(row["crop_index"]))
        fused, confidence, method, valid_frames = fuse_track(rows)
        frame_count = len(rows)
        best_texts = ";".join(
            f"{row['crop_index']}:{row['norm_text'] or '<empty>'}@{float(row['score']):.2f}"
            for row in rows
        )
        summary_rows.append([track_id, fused, f"{confidence:.4f}", method, valid_frames, frame_count, best_texts])
        print(f"FUSED id={track_id}: {fused or '<empty>'} method={method} valid={valid_frames}/{frame_count}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "fused_text", "fused_confidence", "method", "valid_frames", "frame_count", "best_frame_texts"])
        writer.writerows(summary_rows)

    print(f"Wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
