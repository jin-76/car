import argparse
import json
from pathlib import Path

import cv2


def parse_source(source):
    path = Path(source)
    if path.exists():
        return str(path.resolve())
    return source


LINE_COLORS = [(0, 0, 255), (0, 180, 255), (255, 0, 255), (255, 255, 0)]
QUAD_COLOR = (0, 255, 255)


def draw_overlay(frame, lines, current_points, scale, target_lines):
    preview = frame.copy()
    for index, line in enumerate(lines):
        color = LINE_COLORS[index % len(LINE_COLORS)]
        p1, p2 = line
        cv2.line(preview, p1, p2, color, 3, cv2.LINE_AA)
        cv2.circle(preview, p1, 7, color, -1)
        cv2.circle(preview, p2, 7, color, -1)
        cv2.putText(preview, f"line {index + 1}", (p1[0] + 8, max(24, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if current_points:
        color = LINE_COLORS[len(lines) % len(LINE_COLORS)]
        cv2.circle(preview, current_points[0], 7, color, -1)

    help_lines = [
        f"Left click: set endpoints ({len(lines)}/{target_lines} lines)",
        "Right click or C: clear",
        "Backspace/U: undo last line",
        "S: save after all lines",
        "Esc/Q: quit",
    ]
    y = 28
    for text in help_lines:
        cv2.putText(preview, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30

    if scale != 1.0:
        h, w = preview.shape[:2]
        preview = cv2.resize(preview, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return preview


def draw_quad_overlay(frame, points, scale):
    preview = frame.copy()
    for index, point in enumerate(points):
        cv2.circle(preview, point, 7, QUAD_COLOR, -1)
        cv2.putText(
            preview,
            str(index + 1),
            (point[0] + 8, max(24, point[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            QUAD_COLOR,
            2,
            cv2.LINE_AA,
        )
    if len(points) >= 2:
        for index in range(len(points) - 1):
            cv2.line(preview, points[index], points[index + 1], QUAD_COLOR, 2, cv2.LINE_AA)
    if len(points) == 4:
        cv2.line(preview, points[3], points[0], QUAD_COLOR, 2, cv2.LINE_AA)

    help_lines = [
        f"Left click: road quad corner ({len(points)}/4)",
        "Order: near-left, near-right, far-left, far-right",
        "Right click or C: clear",
        "Backspace/U: undo last point",
        "S: save image_points",
        "Esc/Q: quit",
    ]
    y = 28
    for text in help_lines:
        cv2.putText(preview, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 30

    if scale != 1.0:
        h, w = preview.shape[:2]
        preview = cv2.resize(preview, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return preview


def save_annotation(out_path, source, frame, lines, target_lines):
    if len(lines) != target_lines:
        print(f"Need {target_lines} lines before saving; current: {len(lines)}.")
        return False

    h, w = frame.shape[:2]
    data = {
        "source": source,
        "frame_size": [w, h],
        "lines": [
            [[int(line[0][0]), int(line[0][1])], [int(line[1][0]), int(line[1][1])]]
            for line in lines
        ],
    }
    if target_lines == 1:
        data["line"] = data["lines"][0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved annotation: {out_path}")
    print(f"Lines: {data['lines']}")
    return True


def save_homography_quad(out_path, source, frame, points):
    if len(points) != 4:
        print(f"Need 4 road-quad points before saving; current: {len(points)}.")
        return False

    h, w = frame.shape[:2]
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        data = {"source": source, "frame_size": [w, h]}

    data["source"] = source
    data["frame_size"] = [w, h]
    data.setdefault("homography", {})
    data["homography"]["image_points"] = [[int(point[0]), int(point[1])] for point in points]
    data["homography"].pop("world_points", None)
    data["homography"]["point_order"] = "near-left, near-right, far-left, far-right"
    data["homography"]["needs_world_points"] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved homography image points: {out_path}")
    print(f"Image points: {data['homography']['image_points']}")
    print("Next: provide road width and length/height in meters to fill world_points.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video path.")
    parser.add_argument("--out", default="runs/traffic_line.json", help="Output annotation JSON.")
    parser.add_argument("--frame", type=int, default=1, help="Frame number to annotate, 1-based.")
    parser.add_argument("--preview-scale", type=float, default=0.5, help="Preview scale.")
    parser.add_argument("--lines", type=int, default=2, help="Number of detection lines to annotate.")
    parser.add_argument("--homography-quad", action="store_true", help="Annotate 4 road-plane points for homography.")
    args = parser.parse_args()

    source = parse_source(args.source)
    out_path = Path(args.out)
    scale = args.preview_scale if args.preview_scale > 0 else 1.0
    target_lines = max(1, args.lines)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    frame_number = max(1, args.frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_number} from source: {args.source}")

    lines = []
    quad_points = []
    current_points = []
    window_name = "Annotate Homography Quad" if args.homography_quad else "Annotate Detection Line"

    def on_mouse(event, x, y, flags, param):
        del flags, param
        real_point = (int(x / scale), int(y / scale))
        real_point = (
            max(0, min(frame.shape[1] - 1, real_point[0])),
            max(0, min(frame.shape[0] - 1, real_point[1])),
        )
        if event == cv2.EVENT_LBUTTONDOWN:
            if args.homography_quad:
                if len(quad_points) >= 4:
                    print("All 4 points are already set. Press S to save, C to clear, or U to undo.")
                    return
                quad_points.append(real_point)
                print(f"Quad point {len(quad_points)}/4: {real_point}")
                return
            if len(lines) >= target_lines:
                print("All lines are already set. Press S to save, C to clear, or U to undo.")
                return
            current_points.append(real_point)
            print(f"Line {len(lines) + 1} point {len(current_points)}: {real_point}")
            if len(current_points) == 2:
                lines.append((current_points[0], current_points[1]))
                current_points.clear()
                print(f"Completed line {len(lines)}/{target_lines}.")
        elif event == cv2.EVENT_RBUTTONDOWN:
            lines.clear()
            quad_points.clear()
            current_points.clear()
            print("Cleared all points.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    if args.homography_quad:
        print("Left click 4 road-plane corners: near-left, near-right, far-left, far-right. Press S to save.")
    else:
        print(f"Left click {target_lines * 2} points to mark {target_lines} detection lines. Press S to save.")
    while True:
        if args.homography_quad:
            cv2.imshow(window_name, draw_quad_overlay(frame, quad_points, scale))
        else:
            cv2.imshow(window_name, draw_overlay(frame, lines, current_points, scale, target_lines))
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("c"):
            lines.clear()
            quad_points.clear()
            current_points.clear()
            print("Cleared all points.")
        elif key in (8, ord("u")):
            if args.homography_quad and quad_points:
                removed = quad_points.pop()
                print(f"Removed last quad point: {removed}")
            elif current_points:
                current_points.clear()
                print("Cleared current unfinished line.")
            elif lines:
                removed = lines.pop()
                print(f"Removed last line: {removed}")
        elif key == ord("s"):
            if args.homography_quad:
                saved = save_homography_quad(out_path, source, frame, quad_points)
            else:
                saved = save_annotation(out_path, source, frame, lines, target_lines)
            if saved:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
