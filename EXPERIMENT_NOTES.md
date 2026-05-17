# Traffic OCR Experiment Notes

## Current Takeaway

For this video, `imgsz=1280` improves plate-box confidence compared with `imgsz=960`, but the very aggressive plate-crop enhancement does not clearly improve OCR text accuracy.

Best next direction:

- Keep YOLO inference at `--imgsz 1280` if the GPU can tolerate the runtime.
- Reduce the plate-crop enhancement back toward the previous 960-run strength.
- Add multi-frame voting and de-duplication instead of pushing sharpening harder.
- Treat per-frame OCR as noisy candidates, not final plate results.

## Compared Runs

### 960 + Aggressive Plate Enhancement

- OCR total rows: 74
- Rough plate-like candidates: 48
- Weak or dirty rows: 26
- Observed behavior: more detections and decent candidates, but still many OCR substitutions.

### 1280 + Very Aggressive Plate Enhancement

- OCR total rows: 68
- Rough plate-like candidates: 47
- Weak or dirty rows: 21
- Observed behavior: detection confidence was generally higher, but OCR text often shifted in the wrong direction.

Examples where 1280 looked useful:

- `QB4UV99`
- `BBI44S`
- `88JZ963`
- `BV066V`
- `粤BP0540`
- `CB897RT`
- `B7480Z`

Examples where aggressive enhancement likely hurt OCR:

- `4BR167F` became closer to `LER167F`
- `粤BXA895` became variants like `EB XA895` / `1B XA895`

## Recommendation

Do not continue increasing sharpness as the main strategy. The next useful improvement is a post-processing stage:

1. Normalize OCR strings by removing spaces and punctuation noise.
2. Filter obvious non-plate strings such as `car`, `bus`, decimal confidences, and pure digits.
3. Group results by time window or nearby plate-box position.
4. Vote across repeated candidates and keep the most stable result.
5. Save a final clean CSV separate from the raw OCR CSV.
