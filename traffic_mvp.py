import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PROJECT_DIR / ".paddlex-cache"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    from paddleocr import PaddleOCR

    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
PERSON_CLASS = "person"
LIGHT_CLASS = "traffic light"
PIL_FONT = None


def get_pil_font(size=24):
    global PIL_FONT
    if PIL_FONT is not None:
        return PIL_FONT

    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            PIL_FONT = ImageFont.truetype(font_path, size)
            return PIL_FONT

    PIL_FONT = ImageFont.load_default()
    return PIL_FONT


def draw_text(frame, text, origin, color, bg_color=None):
    if not text:
        return

    x, y = origin
    if all(ord(char) < 128 for char in text):
        if bg_color is not None:
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                frame,
                (x - 2, y - text_h - baseline - 2),
                (x + text_w + 2, y + baseline + 2),
                bg_color,
                -1,
            )
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return

    if not PIL_AVAILABLE:
        cv2.putText(frame, text.encode("ascii", "ignore").decode("ascii"), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = get_pil_font()
    bbox = draw.textbbox((x, y), text, font=font)
    if bg_color is not None:
        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
        draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=bg_rgb)
    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_box(frame, box, label, color=(0, 255, 0), label_position="above"):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label_position == "right":
        text_x = min(frame.shape[1] - 1, x2 + 6)
        text_y = max(24, y1 + 24)
        if text_x > frame.shape[1] - 160:
            text_x = max(0, x1 - 160)
    else:
        text_x = x1
        text_y = max(25, y1 - 8)
    draw_text(frame, label, (text_x, text_y), color, (0, 0, 0))


LINE_COLORS = [(0, 0, 255), (0, 180, 255), (255, 0, 255), (255, 255, 0)]


def load_detection_lines(path):
    if not path:
        return []

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_lines = data.get("lines")
    if raw_lines is None and data.get("line") is not None:
        raw_lines = [data["line"]]
    if not raw_lines:
        if data.get("homography") or data.get("calibration"):
            return []
        raise ValueError(f"Invalid line annotation file: {path}")

    lines = []
    for line in raw_lines:
        if len(line) != 2:
            raise ValueError(f"Invalid line annotation file: {path}")
        lines.append(tuple((int(point[0]), int(point[1])) for point in line))
    return lines


def load_homography(path, lines, distance_m, road_width_m):
    if not path:
        return None, ""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    calibration = data.get("homography") or data.get("calibration") or {}
    image_points = calibration.get("image_points")
    world_points = calibration.get("world_points")

    if image_points is not None or world_points is not None:
        if not image_points or not world_points or len(image_points) != 4 or len(world_points) != 4:
            raise ValueError("Homography calibration needs 4 image_points and 4 world_points.")
        src = np.asarray(image_points, dtype=np.float32)
        dst = np.asarray(world_points, dtype=np.float32)
        matrix, _ = cv2.findHomography(src, dst, 0)
        if matrix is None:
            raise ValueError(f"Failed to compute homography from {path}")
        return matrix, "annotation homography"

    if len(lines) < 2:
        return None, ""

    src = np.asarray(
        [
            lines[0][0],
            lines[0][1],
            lines[1][0],
            lines[1][1],
        ],
        dtype=np.float32,
    )
    dst = np.asarray(
        [
            [road_width_m, 0.0],
            [0.0, 0.0],
            [road_width_m, distance_m],
            [0.0, distance_m],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return matrix, "two-line homography"


def project_ground_point(point, homography):
    if homography is None:
        return None

    src = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)[0][0]
    return float(dst[0]), float(dst[1])


def draw_detection_lines(frame, lines):
    if not lines:
        return

    for index, line in enumerate(lines):
        color = LINE_COLORS[index % len(LINE_COLORS)]
        p1, p2 = line
        cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)
        cv2.circle(frame, p1, 7, color, -1)
        cv2.circle(frame, p2, 7, color, -1)
        draw_text(frame, f"line {index + 1}", (p1[0] + 8, max(24, p1[1] - 8)), color, (0, 0, 0))


def line_side(line, point):
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def point_between_two_lines(point, lines):
    if len(lines) < 2:
        return False

    side_a = line_side(lines[0], point)
    side_b = line_side(lines[1], point)
    if side_a == 0 or side_b == 0:
        return True
    return (side_a > 0) != (side_b > 0)


def box_center(box):
    x1, y1, x2, y2 = map(float, box)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def box_bottom_center(box):
    x1, _y1, x2, y2 = map(float, box)
    return int((x1 + x2) / 2), int(y2)


def box_size(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def center_distance(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def rect_center(rect):
    x1, y1, x2, y2 = map(float, rect)
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_to_rect_distance(point, rect):
    px, py = map(float, point)
    x1, y1, x2, y2 = map(float, rect)
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def expand_box(box, padding):
    x1, y1, x2, y2 = map(float, box)
    return [x1 - padding, y1 - padding, x2 + padding, y2 + padding]


def vehicle_contact_zone(box):
    x1, y1, x2, y2 = map(float, box)
    height = max(1.0, y2 - y1)
    return [x1, y2 - 0.35 * height, x2, y2]


def speed_status(speed_kmh):
    if speed_kmh < 35:
        return "合规"
    if speed_kmh <= 40:
        return "注意"
    return "违规"


def default_track_state():
    return {
        "was_between": False,
        "pending_between": None,
        "pending_count": 0,
        "entry_frame": None,
        "entry_point": None,
        "entry_metric_point": None,
        "last_speed_kmh": None,
        "last_speed_status": "",
        "last_elapsed_sec": None,
        "metric_history": [],
        "continuous_speed_kmh": None,
        "continuous_speed_mps": None,
        "events": 0,
        "rejected_events": 0,
        "plate_text": "",
        "still_anchor": None,
        "still_start_frame": None,
        "parking_reported": False,
        "parking_frame": None,
    }


def speed_status(speed_kmh):
    if speed_kmh < 35:
        return "\u5408\u89c4"
    if speed_kmh <= 40:
        return "\u6ce8\u610f"
    return "\u8fdd\u89c4"


def update_speed_state(
    track_state,
    track_id,
    center,
    metric_point,
    in_between,
    frame_index,
    fps,
    distance_m,
    use_homography_distance,
    confirm_frames,
    min_speed_kmh,
    max_speed_kmh,
):
    state = track_state.setdefault(track_id, default_track_state())

    event = None
    rejected_event = None
    confirmed_change = False
    if in_between == state["was_between"]:
        state["pending_between"] = None
        state["pending_count"] = 0
    else:
        if state["pending_between"] != in_between:
            state["pending_between"] = in_between
            state["pending_count"] = 1
        else:
            state["pending_count"] += 1
        if state["pending_count"] >= confirm_frames:
            state["was_between"] = in_between
            state["pending_between"] = None
            state["pending_count"] = 0
            confirmed_change = True

    if confirmed_change and in_between:
        state["entry_frame"] = frame_index
        state["entry_point"] = center
        state["entry_metric_point"] = metric_point
    elif confirmed_change and not in_between and state["entry_frame"] is not None:
        elapsed_sec = (frame_index - state["entry_frame"]) / fps if fps else 0
        if elapsed_sec > 0:
            measured_distance_m = distance_m
            if use_homography_distance and state["entry_metric_point"] is not None and metric_point is not None:
                measured_distance_m = abs(metric_point[1] - state["entry_metric_point"][1])
            speed_mps = measured_distance_m / elapsed_sec
            speed_kmh = speed_mps * 3.6
            event_data = {
                "track_id": track_id,
                "entry_frame": state["entry_frame"],
                "exit_frame": frame_index,
                "entry_point": state["entry_point"],
                "exit_point": center,
                "entry_metric_point": state["entry_metric_point"],
                "exit_metric_point": metric_point,
                "elapsed_sec": elapsed_sec,
                "distance_m": measured_distance_m,
                "speed_mps": speed_mps,
                "speed_kmh": speed_kmh,
            }
            if min_speed_kmh <= speed_kmh <= max_speed_kmh:
                state["last_speed_kmh"] = speed_kmh
                state["last_speed_status"] = speed_status(speed_kmh)
                state["last_elapsed_sec"] = elapsed_sec
                state["events"] += 1
                event_data["event_index"] = state["events"]
                event_data["status"] = state["last_speed_status"]
                event = event_data
            else:
                state["rejected_events"] += 1
                event_data["event_index"] = state["rejected_events"]
                rejected_event = event_data
        state["entry_frame"] = None
        state["entry_point"] = None
        state["entry_metric_point"] = None

    return state, event, rejected_event


def update_continuous_speed_state(track_state, track_id, image_point, metric_point, frame_index, fps, window_frames):
    state = track_state.setdefault(track_id, default_track_state())
    if metric_point is None or fps <= 0:
        return state

    history = state["metric_history"]
    history.append((frame_index, image_point, metric_point))
    min_frame = frame_index - window_frames
    while len(history) > 2 and history[0][0] < min_frame:
        history.pop(0)

    if len(history) >= 2:
        start_frame, _start_image_point, start_metric_point = history[0]
        elapsed_sec = (frame_index - start_frame) / fps
        if elapsed_sec > 0:
            dx = metric_point[0] - start_metric_point[0]
            dy = metric_point[1] - start_metric_point[1]
            distance_m = (dx * dx + dy * dy) ** 0.5
            state["continuous_speed_mps"] = distance_m / elapsed_sec
            state["continuous_speed_kmh"] = state["continuous_speed_mps"] * 3.6

    return state


def default_collision_state(frame_index, person_box, person_point):
    return {
        "stage": "normal",
        "contact_frame": None,
        "contact_vehicle_id": None,
        "last_seen_frame": frame_index,
        "last_box": person_box,
        "last_point": person_point,
        "post_contact_points": [],
        "reported": False,
    }


def update_collision_states(
    collision_state,
    persons,
    vehicles,
    frame_index,
    fps,
    contact_px,
    contact_frames,
    watch_sec,
    still_px,
    aspect_thresh,
    disappear_frames,
):
    events = []
    labels = {"person": {}, "vehicle": {}}
    current_person_keys = set()
    watch_frames = max(1, int(watch_sec * fps))

    for person in persons:
        person_key = person["key"]
        current_person_keys.add(person_key)
        state = collision_state.setdefault(person_key, default_collision_state(frame_index, person["box"], person["point"]))
        state["last_seen_frame"] = frame_index
        state["last_box"] = person["box"]
        state["last_point"] = person["point"]

        best_vehicle = None
        best_distance = float("inf")
        for vehicle in vehicles:
            zone = expand_box(vehicle_contact_zone(vehicle["box"]), contact_px)
            distance = point_to_rect_distance(person["point"], zone)
            overlap = box_iou(person["box"], zone)
            score = 0.0 if overlap > 0 else distance
            if score < best_distance:
                best_distance = score
                best_vehicle = vehicle

        near_contact = best_vehicle is not None and best_distance <= contact_px
        if state["stage"] == "normal":
            state["stage"] = "near_collision" if near_contact else "normal"
            state["near_count"] = 1 if near_contact else 0
        elif state["stage"] == "near_collision":
            state["near_count"] = state.get("near_count", 0) + 1 if near_contact else 0
            if not near_contact:
                state["stage"] = "normal"
            elif state["near_count"] >= contact_frames:
                state["stage"] = "watch_after_contact"
                state["contact_frame"] = frame_index
                state["contact_vehicle_id"] = best_vehicle["track_id"]
                state["post_contact_points"] = [person["point"]]
                events.append(
                    {
                        "event": "contact",
                        "frame": frame_index,
                        "person_id": person["track_id"],
                        "vehicle_id": best_vehicle["track_id"],
                        "reason": "near_vehicle_front",
                    }
                )
        elif state["stage"] == "watch_after_contact":
            state["post_contact_points"].append(person["point"])
            if len(state["post_contact_points"]) > watch_frames:
                state["post_contact_points"].pop(0)

            x1, y1, x2, y2 = map(float, person["box"])
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            aspect = width / height
            point_span = 0.0
            if len(state["post_contact_points"]) >= 2:
                first_point = state["post_contact_points"][0]
                point_span = max(center_distance(first_point, point) for point in state["post_contact_points"])
            still_after_contact = len(state["post_contact_points"]) >= min(5, watch_frames) and point_span <= still_px
            horizontal_after_contact = aspect >= aspect_thresh
            if not state["reported"] and (horizontal_after_contact or still_after_contact):
                reasons = []
                if horizontal_after_contact:
                    reasons.append(f"horizontal_aspect={aspect:.2f}")
                if still_after_contact:
                    reasons.append(f"still_span_px={point_span:.1f}")
                state["reported"] = True
                state["stage"] = "suspected_accident"
                events.append(
                    {
                        "event": "suspected_accident",
                        "frame": frame_index,
                        "person_id": person["track_id"],
                        "vehicle_id": state["contact_vehicle_id"],
                        "reason": ";".join(reasons),
                    }
                )
        if state["stage"] != "normal":
            labels["person"][person["track_id"]] = state["stage"]
            if state.get("contact_vehicle_id") is not None:
                labels["vehicle"][state["contact_vehicle_id"]] = state["stage"]

    for person_key, state in list(collision_state.items()):
        if person_key in current_person_keys or state.get("stage") not in {"watch_after_contact", "suspected_accident"}:
            continue
        missing_frames = frame_index - state["last_seen_frame"]
        if missing_frames >= disappear_frames and not state["reported"]:
            state["reported"] = True
            state["stage"] = "suspected_accident"
            events.append(
                {
                    "event": "suspected_accident",
                    "frame": frame_index,
                    "person_id": person_key.replace("person:", ""),
                    "vehicle_id": state["contact_vehicle_id"],
                    "reason": f"person_disappeared_{missing_frames}_frames_after_contact",
                }
            )

    return events, labels


def update_parking_state(track_state, track_id, center, frame_index, fps, still_seconds, pixel_thresh):
    state = track_state.setdefault(track_id, default_track_state())
    if state["still_anchor"] is None:
        state["still_anchor"] = center
        state["still_start_frame"] = frame_index
        return state, None

    if center_distance(center, state["still_anchor"]) <= pixel_thresh:
        elapsed_sec = (frame_index - state["still_start_frame"]) / fps if fps else 0
        if elapsed_sec >= still_seconds and not state["parking_reported"]:
            state["parking_reported"] = True
            state["parking_frame"] = frame_index
            return state, {
                "track_id": track_id,
                "start_frame": state["still_start_frame"],
                "parking_frame": frame_index,
                "elapsed_sec": elapsed_sec,
                "anchor_point": state["still_anchor"],
                "current_point": center,
            }
    else:
        state["still_anchor"] = center
        state["still_start_frame"] = frame_index
        state["parking_reported"] = False
        state["parking_frame"] = None

    return state, None


def crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return frame[y1:y2, x1:x2]


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def carry_plate_text(new_box, previous_detections):
    best_text = ""
    best_iou = 0.0
    for previous in previous_detections:
        previous_text = previous.get("text", "")
        if not previous_text:
            continue
        current_iou = box_iou(new_box, previous["box"])
        if current_iou > best_iou:
            best_iou = current_iou
            best_text = previous_text
    return best_text if best_iou >= 0.25 else ""


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = map(float, box)
    return x1 <= x <= x2 and y1 <= y <= y2


def plate_text_for_vehicle(vehicle_box, plate_detections):
    best_text = ""
    best_score = -1.0
    vehicle_center = box_center(vehicle_box)
    vehicle_w, vehicle_h = box_size(vehicle_box)
    fallback_text = ""
    fallback_distance = float("inf")
    max_fallback_distance = max(vehicle_w, vehicle_h) * 0.75
    for plate_detection in plate_detections:
        plate_text = plate_detection.get("text", "")
        if not plate_text:
            continue
        plate_box = plate_detection.get("box")
        if not plate_box:
            continue
        plate_center = box_center(plate_box)
        distance = center_distance(vehicle_center, plate_center)
        if distance < fallback_distance and distance <= max_fallback_distance:
            fallback_distance = distance
            fallback_text = plate_text
        if not point_in_box(plate_center, vehicle_box):
            continue
        score = float(plate_detection.get("conf", 0.0))
        if score > best_score:
            best_score = score
            best_text = plate_text
    return best_text or fallback_text


def track_id_for_plate(plate_box, detections):
    plate_center = box_center(plate_box)
    best_track_id = None
    best_score = -1.0
    fallback_track_id = None
    fallback_distance = float("inf")

    for name, _conf, vehicle_box, track_id in detections:
        if name not in VEHICLE_CLASSES or track_id is None:
            continue

        vehicle_center = box_center(vehicle_box)
        vehicle_w, vehicle_h = box_size(vehicle_box)
        distance = center_distance(vehicle_center, plate_center)
        max_fallback_distance = max(vehicle_w, vehicle_h) * 0.75
        if distance < fallback_distance and distance <= max_fallback_distance:
            fallback_distance = distance
            fallback_track_id = track_id

        if not point_in_box(plate_center, vehicle_box):
            continue

        score = box_iou(plate_box, vehicle_box)
        if score > best_score:
            best_score = score
            best_track_id = track_id

    return best_track_id if best_track_id is not None else fallback_track_id


def save_plate_crop(frame, plate_box, track_id, output_dir, saved_counts):
    plate_img = crop(frame, plate_box)
    if plate_img.size == 0:
        return None

    saved_counts[track_id] = saved_counts.get(track_id, 0) + 1
    output_path = output_dir / f"{track_id}--{saved_counts[track_id]}.jpg"
    cv2.imwrite(str(output_path), plate_img)
    return output_path


def detect_light_color(frame, box):
    roi = crop(frame, box)
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    yellow = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    green = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))

    scores = {
        "red": cv2.countNonZero(red1) + cv2.countNonZero(red2),
        "yellow": cv2.countNonZero(yellow),
        "green": cv2.countNonZero(green),
    }

    color, score = max(scores.items(), key=lambda x: x[1])
    return color if score > 20 else "unknown"


def ocr_plate(ocr, plate_img, enhance_mode="basic"):
    if not OCR_AVAILABLE or ocr is None:
        return ""

    if plate_img.size == 0:
        return ""

    plate_img = prepare_plate_for_ocr(plate_img, enhance_mode)

    result = ocr.predict(plate_img)
    if not result:
        return ""

    texts = []
    for page in result:
        rec_texts = page.get("rec_texts", []) if isinstance(page, dict) else []
        rec_scores = page.get("rec_scores", []) if isinstance(page, dict) else []
        for text, score in zip(rec_texts, rec_scores):
            if score > 0.5:
                texts.append(text)

    return "".join(texts)


def prepare_plate_for_ocr(plate_img, mode):
    if mode == "none":
        return plate_img

    if mode == "aggressive":
        return enhance_plate_for_ocr(plate_img)

    h, w = plate_img.shape[:2]
    if mode == "opencv_bicubic_x4":
        return cv2.resize(plate_img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    scale = max(2, min(4, 240 // max(1, min(h, w))))
    if scale > 1:
        plate_img = cv2.resize(plate_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.08, beta=4)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def enhance_plate_for_ocr(plate_img):
    # Plate crops are tiny after traffic-video crop/resize; OCR benefits from
    # a stronger local enhancement pass than the full-frame video uses.
    h, w = plate_img.shape[:2]
    scale = max(3, min(8, 480 // max(1, min(h, w))))
    if scale > 1:
        plate_img = cv2.resize(plate_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    plate_img = cv2.fastNlMeansDenoisingColored(plate_img, None, 10, 10, 7, 21)

    lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(6, 6))
    l_channel = clahe.apply(l_channel)
    plate_img = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(plate_img, (0, 0), 0.8)
    plate_img = cv2.addWeighted(plate_img, 2.2, blurred, -1.2, 0)

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def ocr_plate_legacy(ocr, plate_img):
    result = ocr.ocr(plate_img)
    if not result or not result[0]:
        return ""

    texts = []
    for line in result[0]:
        text = line[1][0]
        score = line[1][1]
        if score > 0.5:
            texts.append(text)

    return "".join(texts)


def parse_source(source):
    if source.isdigit():
        return int(source)

    path = Path(source)
    if path.exists():
        return str(path.resolve())

    wanted_name = path.name
    mp4s = list(Path.cwd().glob("**/*.mp4"))
    for candidate in mp4s:
        if candidate.name == wanted_name:
            return str(candidate.resolve())

    if len(mp4s) == 1:
        print(f"Source path was not found; using the only MP4 in this project: {mp4s[0]}")
        return str(mp4s[0].resolve())

    return source


def make_preview(frame, scale):
    if scale <= 0 or scale == 1:
        return frame

    h, w = frame.shape[:2]
    preview_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(frame, preview_size, interpolation=cv2.INTER_AREA)


def write_preview_image(frame, path, max_width=1280):
    preview = frame
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        preview = cv2.resize(frame, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    temp_path = path.with_suffix(".tmp.jpg")
    try:
        if cv2.imwrite(str(temp_path), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 80]):
            os.replace(temp_path, path)
    except OSError:
        # The workbench may briefly hold the previous image while reading it on Windows.
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video path or camera index, for example 0.")
    parser.add_argument("--model", default="model/obj-det-model/yolov8n.pt", help="YOLO model path.")
    parser.add_argument("--device", default="0", help="YOLO device. Use 0 for NVIDIA GPU, cpu for CPU.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size. Lower saves memory and time.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA devices.")
    parser.add_argument("--process-every", type=int, default=1, help="Run detection every N frames and reuse boxes between runs.")
    parser.add_argument("--tracker", default="bytetrack_custom.yaml", help="Ultralytics tracker config for vehicle IDs.")
    parser.add_argument("--plate-model", default="", help="Optional license-plate YOLO model path.")
    parser.add_argument("--plate-conf", type=float, default=0.2, help="Confidence threshold for plate detection.")
    parser.add_argument("--no-ocr", action="store_true", help="Draw plate boxes without loading PaddleOCR.")
    parser.add_argument("--save", default="output.mp4", help="Output video path.")
    parser.add_argument("--no-save", action="store_true", help="Do not write an output video.")
    parser.add_argument("--output-scale", type=float, default=1.0, help="Scale saved video; 0.5 greatly reduces disk writes.")
    parser.add_argument("--plate-log", default="", help="Optional CSV path for recognized plates.")
    parser.add_argument("--plate-crop-dir", default="", help="Optional directory for plate crops named track_id--index.jpg.")
    parser.add_argument("--plate-crop-every", type=int, default=1, help="Save one plate crop every N frames per tracked vehicle.")
    parser.add_argument("--ocr-every", type=int, default=10, help="Run OCR on plate boxes every N frames.")
    parser.add_argument(
        "--plate-enhance",
        choices=["none", "basic", "aggressive", "opencv_bicubic_x4"],
        default="opencv_bicubic_x4",
        help="Plate crop preprocessing before OCR.",
    )
    parser.add_argument("--preview-scale", type=float, default=0.35, help="Preview window scale for large videos.")
    parser.add_argument("--no-preview", action="store_true", help="Do not show the preview window.")
    parser.add_argument("--preview-image", default="", help="Optional path for the latest annotated preview JPEG.")
    parser.add_argument("--preview-image-every", type=int, default=1, help="Write the preview JPEG every N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames; 0 means all frames.")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start processing at this video timestamp in seconds.")
    parser.add_argument("--end-sec", type=float, default=0.0, help="Stop processing at this video timestamp in seconds; 0 means no limit.")
    parser.add_argument("--cv-threads", type=int, default=1, help="OpenCV worker threads.")
    parser.add_argument("--line-json", default="", help="Optional detection-line annotation JSON.")
    parser.add_argument("--speed-distance-m", type=float, default=14.0, help="Real distance between the two detection lines in meters.")
    parser.add_argument(
        "--speed-homography",
        action="store_true",
        help="Use a ground-plane homography to measure speed from metric coordinates.",
    )
    parser.add_argument(
        "--speed-road-width-m",
        type=float,
        default=12.0,
        help="Road width used when deriving homography from the two speed lines.",
    )
    parser.add_argument(
        "--speed-point",
        choices=["center", "bottom-center"],
        default="center",
        help="Vehicle point used for speed tracking and homography projection.",
    )
    parser.add_argument("--speed-log", default="", help="Optional CSV path for line-to-line speed events.")
    parser.add_argument(
        "--continuous-speed",
        action="store_true",
        help="Estimate and display per-track sliding-window speed from homography coordinates.",
    )
    parser.add_argument("--continuous-speed-log", default="", help="Optional CSV path for per-frame homography speeds.")
    parser.add_argument(
        "--continuous-speed-window",
        type=int,
        default=10,
        help="Frame window used for continuous homography speed smoothing.",
    )
    parser.add_argument(
        "--pedestrian-speed",
        action="store_true",
        help="Detect tracked persons and display sliding-window homography speed.",
    )
    parser.add_argument(
        "--pedestrian-point",
        choices=["center", "bottom-center"],
        default="bottom-center",
        help="Person point used for homography projection.",
    )
    parser.add_argument("--pedestrian-speed-log", default="", help="Optional CSV path for per-frame pedestrian speeds.")
    parser.add_argument("--pedestrian-min-box", type=int, default=20, help="Minimum person box width and height for speed tracking.")
    parser.add_argument("--collision-detect", action="store_true", help="Enable person-vehicle suspected collision state machine.")
    parser.add_argument("--collision-log", default="", help="Optional CSV path for suspected collision events.")
    parser.add_argument("--collision-contact-px", type=float, default=35.0, help="Person-to-vehicle-front pixel distance threshold.")
    parser.add_argument("--collision-contact-frames", type=int, default=2, help="Consecutive near-contact frames before contact.")
    parser.add_argument("--collision-watch-sec", type=float, default=1.5, help="Seconds to watch a person after contact.")
    parser.add_argument("--collision-still-px", type=float, default=15.0, help="Post-contact point span below this counts as still.")
    parser.add_argument("--collision-aspect", type=float, default=1.1, help="Post-contact person width/height threshold for horizontal posture.")
    parser.add_argument("--collision-disappear-frames", type=int, default=3, help="Missing frames after contact before suspected accident.")
    parser.add_argument("--speed-min-kmh", type=float, default=5.0, help="Reject speed events below this value.")
    parser.add_argument("--speed-max-kmh", type=float, default=160.0, help="Reject speed events above this value.")
    parser.add_argument("--speed-confirm-frames", type=int, default=3, help="Frames required to confirm entry/exit of the speed zone.")
    parser.add_argument("--speed-min-box", type=int, default=40, help="Minimum vehicle box width and height for speed tracking.")
    parser.add_argument("--parking-still-sec", type=float, default=3.0, help="Seconds a vehicle center must stay still before parking violation.")
    parser.add_argument("--parking-pixel-thresh", type=float, default=5.0, help="Maximum center movement in pixels to count as still.")
    parser.add_argument("--parking-log", default="", help="Optional CSV path for parking violation events.")
    args = parser.parse_args()

    args.process_every = max(1, args.process_every)
    args.ocr_every = max(1, args.ocr_every)
    args.plate_crop_every = max(1, args.plate_crop_every)
    args.cv_threads = max(0, args.cv_threads)
    args.start_sec = max(0.0, args.start_sec)
    args.end_sec = max(0.0, args.end_sec)
    args.speed_distance_m = max(0.01, args.speed_distance_m)
    args.speed_road_width_m = max(0.01, args.speed_road_width_m)
    args.continuous_speed_window = max(2, args.continuous_speed_window)
    args.pedestrian_min_box = max(1, args.pedestrian_min_box)
    args.collision_contact_px = max(0.0, args.collision_contact_px)
    args.collision_contact_frames = max(1, args.collision_contact_frames)
    args.collision_watch_sec = max(0.1, args.collision_watch_sec)
    args.collision_still_px = max(0.0, args.collision_still_px)
    args.collision_aspect = max(0.1, args.collision_aspect)
    args.collision_disappear_frames = max(1, args.collision_disappear_frames)
    args.speed_confirm_frames = max(1, args.speed_confirm_frames)
    args.speed_min_box = max(1, args.speed_min_box)
    args.parking_still_sec = max(0.1, args.parking_still_sec)
    args.parking_pixel_thresh = max(0.0, args.parking_pixel_thresh)
    cv2.setNumThreads(args.cv_threads)

    source = parse_source(args.source)
    detection_lines = load_detection_lines(args.line_json)
    speed_homography = None
    speed_homography_source = ""
    if args.speed_homography:
        speed_homography, speed_homography_source = load_homography(
            args.line_json,
            detection_lines,
            args.speed_distance_m,
            args.speed_road_width_m,
        )
        if speed_homography is None:
            raise ValueError("--speed-homography needs either homography points or at least two detection lines.")
    if args.continuous_speed and speed_homography is None:
        raise ValueError("--continuous-speed requires --speed-homography.")
    if args.pedestrian_speed and speed_homography is None:
        raise ValueError("--pedestrian-speed requires --speed-homography.")

    detector = YOLO(args.model)

    plate_detector = None
    if args.plate_model:
        if Path(args.plate_model).exists():
            plate_detector = YOLO(args.plate_model)
            print(f"Loaded plate model: {args.plate_model}")
        else:
            print(f"Plate model not found, skipping plate detection: {args.plate_model}")

    ocr = None
    if plate_detector is not None and not args.no_ocr and OCR_AVAILABLE:
        ocr = PaddleOCR(
            lang="ch",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print("Loaded PaddleOCR.")
    elif plate_detector is not None and args.no_ocr:
        print("OCR disabled; plate boxes will be drawn without text.")
    elif plate_detector is not None:
        print("PaddleOCR is not available; plate boxes will be drawn without text.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = int(args.start_sec * fps) + 1 if args.start_sec > 0 else 1
    end_frame = int(args.end_sec * fps) if args.end_sec > 0 else 0
    if end_frame and end_frame < start_frame:
        raise ValueError("--end-sec must be greater than --start-sec.")
    if start_frame > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    writer = None
    save_width = max(1, int(width * args.output_scale))
    save_height = max(1, int(height * args.output_scale))
    if args.no_save:
        args.save = ""

    if args.save:
        writer = cv2.VideoWriter(
            args.save,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (save_width, save_height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open output writer: {args.save}")

    if not args.no_preview:
        cv2.namedWindow("Traffic MVP", cv2.WINDOW_NORMAL)

    print(f"Input: {width}x{height}, {fps:.2f} FPS, {total_frames or 'unknown'} frames")
    if start_frame > 1 or end_frame:
        end_text = f"{end_frame / fps:.2f}s" if end_frame else "end"
        print(f"Clip: {args.start_sec:.2f}s to {end_text} (frames {start_frame}-{end_frame or 'end'})")
    print(f"YOLO: device={args.device}, imgsz={args.imgsz}, half={args.half}, process_every={args.process_every}")
    print(f"Plate OCR enhancement: {args.plate_enhance}")
    if detection_lines:
        print(f"Detection lines: {len(detection_lines)}")
        for index, line in enumerate(detection_lines, start=1):
            print(f"  line {index}: {line[0]} -> {line[1]}")
        if len(detection_lines) >= 2:
            print(f"Speed distance: {args.speed_distance_m:.2f} m")
            print(
                "Speed filters: "
                f"{args.speed_min_kmh:.1f}-{args.speed_max_kmh:.1f} km/h, "
                f"confirm_frames={args.speed_confirm_frames}, min_box={args.speed_min_box}px"
            )
            if speed_homography is not None:
                print(
                    "Speed homography: "
                    f"{speed_homography_source}, road_width={args.speed_road_width_m:.2f}m, "
                    f"point={args.speed_point}"
                )
                if args.continuous_speed:
                    print(f"Continuous speed: window={args.continuous_speed_window} frames")
                if args.pedestrian_speed:
                    print(
                        "Pedestrian speed: "
                        f"window={args.continuous_speed_window} frames, "
                        f"min_box={args.pedestrian_min_box}px, point={args.pedestrian_point}"
                    )
                if args.collision_detect:
                    print(
                        "Collision detect: "
                        f"contact_px={args.collision_contact_px:.1f}, "
                        f"contact_frames={args.collision_contact_frames}, "
                        f"watch={args.collision_watch_sec:.1f}s"
                    )
    print(
        "Parking rule: "
        f"still for {args.parking_still_sec:.1f}s within {args.parking_pixel_thresh:.1f}px"
    )
    print(f"Output: {args.save or 'disabled'}")
    if args.plate_crop_dir:
        print(f"Plate crops: {args.plate_crop_dir}, every {args.plate_crop_every} frame(s) per ID")
    print("Press Esc in the preview window to stop early.")

    plate_log_file = None
    plate_log = None
    if args.plate_log:
        plate_log_file = open(args.plate_log, "w", newline="", encoding="utf-8-sig")
        plate_log = csv.writer(plate_log_file)
        plate_log.writerow(["frame", "time_sec", "text", "confidence", "x1", "y1", "x2", "y2"])

    speed_log_file = None
    speed_log = None
    if args.speed_log:
        speed_log_file = open(args.speed_log, "w", newline="", encoding="utf-8-sig")
        speed_log = csv.writer(speed_log_file)
        speed_log.writerow(
            [
                "track_id",
                "event_index",
                "entry_frame",
                "exit_frame",
                "entry_time_sec",
                "exit_time_sec",
                "elapsed_sec",
                "distance_m",
                "speed_mps",
                "speed_kmh",
                "status",
                "entry_x",
                "entry_y",
                "exit_x",
                "exit_y",
                "entry_world_x_m",
                "entry_world_y_m",
                "exit_world_x_m",
                "exit_world_y_m",
            ]
        )

    continuous_speed_log_file = None
    continuous_speed_log = None
    if args.continuous_speed_log:
        continuous_speed_log_file = open(args.continuous_speed_log, "w", newline="", encoding="utf-8-sig")
        continuous_speed_log = csv.writer(continuous_speed_log_file)
        continuous_speed_log.writerow(
            [
                "frame",
                "time_sec",
                "object_type",
                "track_id",
                "speed_mps",
                "speed_kmh",
                "status",
                "image_x",
                "image_y",
                "world_x_m",
                "world_y_m",
            ]
        )

    pedestrian_speed_log_file = None
    pedestrian_speed_log = None
    if args.pedestrian_speed_log:
        pedestrian_speed_log_file = open(args.pedestrian_speed_log, "w", newline="", encoding="utf-8-sig")
        pedestrian_speed_log = csv.writer(pedestrian_speed_log_file)
        pedestrian_speed_log.writerow(
            [
                "frame",
                "time_sec",
                "track_id",
                "speed_mps",
                "speed_kmh",
                "image_x",
                "image_y",
                "world_x_m",
                "world_y_m",
            ]
        )

    collision_log_file = None
    collision_log = None
    if args.collision_log:
        collision_log_file = open(args.collision_log, "w", newline="", encoding="utf-8-sig")
        collision_log = csv.writer(collision_log_file)
        collision_log.writerow(["frame", "time_sec", "event", "person_id", "vehicle_id", "reason"])

    parking_log_file = None
    parking_log = None
    if args.parking_log:
        parking_log_file = open(args.parking_log, "w", newline="", encoding="utf-8-sig")
        parking_log = csv.writer(parking_log_file)
        parking_log.writerow(
            [
                "track_id",
                "start_frame",
                "parking_frame",
                "start_time_sec",
                "parking_time_sec",
                "elapsed_sec",
                "anchor_x",
                "anchor_y",
                "current_x",
                "current_y",
                "plate_text",
                "status",
            ]
        )

    frame_index = start_frame - 1
    processed_frames = 0
    preview_image_path = Path(args.preview_image) if args.preview_image else None
    preview_image_every = max(1, args.preview_image_every)
    if preview_image_path is not None:
        preview_image_path.parent.mkdir(parents=True, exist_ok=True)
    cached_detections = []
    cached_plate_detections = []
    track_state = {}
    collision_state = {}
    plate_crop_dir = Path(args.plate_crop_dir) if args.plate_crop_dir else None
    if plate_crop_dir is not None:
        plate_crop_dir.mkdir(parents=True, exist_ok=True)
    plate_crop_counts = {}
    plate_crop_last_frame = {}
    plate_crop_log_file = None
    plate_crop_log = None
    if plate_crop_dir is not None:
        plate_crop_log_file = open(plate_crop_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig")
        plate_crop_log = csv.writer(plate_crop_log_file)
        plate_crop_log.writerow(["track_id", "crop_index", "image", "frame", "time_sec", "text", "confidence", "x1", "y1", "x2", "y2"])
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            processed_frames += 1
            raw_frame = frame.copy()
            should_process = frame_index == 1 or frame_index % args.process_every == 0
            if should_process:
                results = detector.track(
                    frame,
                    conf=0.35,
                    device=args.device,
                    imgsz=args.imgsz,
                    half=args.half,
                    tracker=args.tracker,
                    persist=True,
                    verbose=False,
                )[0]
                cached_detections = []
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = detector.names[cls_id]
                    xyxy = box.xyxy[0].tolist()
                    track_id = int(box.id[0]) if box.id is not None else None
                    cached_detections.append((name, conf, xyxy, track_id))

            collision_labels = {"person": {}, "vehicle": {}}
            if args.collision_detect:
                collision_persons = []
                collision_vehicles = []
                for detection_name, _detection_conf, detection_box, detection_track_id in cached_detections:
                    if detection_name in VEHICLE_CLASSES:
                        collision_vehicles.append(
                            {
                                "track_id": detection_track_id,
                                "box": detection_box,
                            }
                        )
                    elif detection_name == PERSON_CLASS and detection_track_id is not None:
                        person_w, person_h = box_size(detection_box)
                        if person_w >= args.pedestrian_min_box and person_h >= args.pedestrian_min_box:
                            person_point = box_bottom_center(detection_box) if args.pedestrian_point == "bottom-center" else box_center(detection_box)
                            collision_persons.append(
                                {
                                    "key": f"person:{detection_track_id}",
                                    "track_id": detection_track_id,
                                    "box": detection_box,
                                    "point": person_point,
                                }
                            )
                collision_events, collision_labels = update_collision_states(
                    collision_state,
                    collision_persons,
                    collision_vehicles,
                    frame_index,
                    fps,
                    args.collision_contact_px,
                    args.collision_contact_frames,
                    args.collision_watch_sec,
                    args.collision_still_px,
                    args.collision_aspect,
                    args.collision_disappear_frames,
                )
                for collision_event in collision_events:
                    print(
                        "Collision "
                        f"event={collision_event['event']} frame={collision_event['frame']} "
                        f"person={collision_event['person_id']} vehicle={collision_event['vehicle_id']} "
                        f"reason={collision_event['reason']}"
                    )
                    if collision_log is not None:
                        collision_log.writerow(
                            [
                                collision_event["frame"],
                                f"{collision_event['frame'] / fps:.2f}" if fps else "0.00",
                                collision_event["event"],
                                collision_event["person_id"],
                                collision_event["vehicle_id"],
                                collision_event["reason"],
                            ]
                        )

            for name, conf, xyxy, track_id in cached_detections:
                if name in VEHICLE_CLASSES:
                    center = box_center(xyxy)
                    speed_point = box_bottom_center(xyxy) if args.speed_point == "bottom-center" else center
                    box_w, box_h = box_size(xyxy)
                    speed_label = ""
                    parking_label = ""
                    collision_label = ""
                    if track_id is not None and track_id in collision_labels["vehicle"]:
                        collision_label = f" {collision_labels['vehicle'][track_id]}"
                    can_track_speed = box_w >= args.speed_min_box and box_h >= args.speed_min_box
                    metric_point = project_ground_point(speed_point, speed_homography)
                    if track_id is not None and can_track_speed:
                        state = track_state.setdefault(track_id, default_track_state())
                        vehicle_plate_text = plate_text_for_vehicle(xyxy, cached_plate_detections)
                        if vehicle_plate_text:
                            state["plate_text"] = vehicle_plate_text
                        parking_state, parking_event = update_parking_state(
                            track_state,
                            track_id,
                            center,
                            frame_index,
                            fps,
                            args.parking_still_sec,
                            args.parking_pixel_thresh,
                        )
                        if parking_state["parking_reported"]:
                            parking_label = " 违停"
                        if parking_state["parking_reported"]:
                            parking_label = f" \u8fdd\u505c:{parking_state['plate_text']}" if parking_state["plate_text"] else " \u8fdd\u505c"
                        if parking_event is not None:
                            print(
                                "Parking "
                                f"id={track_id} still={parking_event['elapsed_sec']:.2f}s "
                                f"point={parking_event['current_point']}"
                                f" plate={parking_state['plate_text']}"
                            )
                            if parking_log is not None:
                                parking_log.writerow(
                                    [
                                        track_id,
                                        parking_event["start_frame"],
                                        parking_event["parking_frame"],
                                        f"{parking_event['start_frame'] / fps:.2f}" if fps else "0.00",
                                        f"{parking_event['parking_frame'] / fps:.2f}" if fps else "0.00",
                                        f"{parking_event['elapsed_sec']:.3f}",
                                        parking_event["anchor_point"][0],
                                        parking_event["anchor_point"][1],
                                        parking_event["current_point"][0],
                                        parking_event["current_point"][1],
                                        parking_state["plate_text"],
                                        "违停",
                                    ]
                                )
                    if track_id is not None and args.continuous_speed and can_track_speed:
                        state = update_continuous_speed_state(
                            track_state,
                            track_id,
                            speed_point,
                            metric_point,
                            frame_index,
                            fps,
                            args.continuous_speed_window,
                        )
                        if state["continuous_speed_kmh"] is not None:
                            continuous_status = speed_status(state["continuous_speed_kmh"])
                            speed_label = f" rt:{state['continuous_speed_kmh']:.1f}km/h {continuous_status}"
                            if continuous_speed_log is not None:
                                continuous_speed_log.writerow(
                                    [
                                        frame_index,
                                        f"{frame_index / fps:.2f}" if fps else "0.00",
                                        name,
                                        track_id,
                                        f"{state['continuous_speed_mps']:.3f}",
                                        f"{state['continuous_speed_kmh']:.3f}",
                                        continuous_status,
                                        speed_point[0],
                                        speed_point[1],
                                        f"{metric_point[0]:.3f}" if metric_point else "",
                                        f"{metric_point[1]:.3f}" if metric_point else "",
                                    ]
                                )
                    if track_id is not None and len(detection_lines) >= 2 and can_track_speed:
                        in_between = point_between_two_lines(speed_point, detection_lines)
                        state, speed_event, rejected_speed_event = update_speed_state(
                            track_state,
                            track_id,
                            speed_point,
                            metric_point,
                            in_between,
                            frame_index,
                            fps,
                            args.speed_distance_m,
                            speed_homography is not None,
                            args.speed_confirm_frames,
                            args.speed_min_kmh,
                            args.speed_max_kmh,
                        )
                        if state["last_speed_kmh"] is not None and not args.continuous_speed:
                            line_speed_label = f" {state['last_speed_kmh']:.1f}km/h {state['last_speed_status']}"
                            speed_label = f"{speed_label}{line_speed_label}" if speed_label else line_speed_label
                        if speed_event is not None:
                            print(
                                "Speed "
                                f"id={track_id} elapsed={speed_event['elapsed_sec']:.2f}s "
                                f"speed={speed_event['speed_kmh']:.1f}km/h "
                                f"status={speed_event['status']}"
                            )
                            if speed_log is not None:
                                speed_log.writerow(
                                    [
                                        track_id,
                                        speed_event["event_index"],
                                        speed_event["entry_frame"],
                                        speed_event["exit_frame"],
                                        f"{speed_event['entry_frame'] / fps:.2f}" if fps else "0.00",
                                        f"{speed_event['exit_frame'] / fps:.2f}" if fps else "0.00",
                                        f"{speed_event['elapsed_sec']:.3f}",
                                        f"{speed_event['distance_m']:.3f}",
                                        f"{speed_event['speed_mps']:.3f}",
                                        f"{speed_event['speed_kmh']:.3f}",
                                        speed_event["status"],
                                        speed_event["entry_point"][0],
                                        speed_event["entry_point"][1],
                                        speed_event["exit_point"][0],
                                        speed_event["exit_point"][1],
                                        f"{speed_event['entry_metric_point'][0]:.3f}" if speed_event["entry_metric_point"] else "",
                                        f"{speed_event['entry_metric_point'][1]:.3f}" if speed_event["entry_metric_point"] else "",
                                        f"{speed_event['exit_metric_point'][0]:.3f}" if speed_event["exit_metric_point"] else "",
                                        f"{speed_event['exit_metric_point'][1]:.3f}" if speed_event["exit_metric_point"] else "",
                                    ]
                                )
                        if rejected_speed_event is not None:
                            print(
                                "Rejected speed "
                                f"id={track_id} elapsed={rejected_speed_event['elapsed_sec']:.2f}s "
                                f"speed={rejected_speed_event['speed_kmh']:.1f}km/h"
                            )
                    id_label = f" id:{track_id}" if track_id is not None else ""
                    draw_box(frame, xyxy, f"{name}{id_label} {conf:.2f}{speed_label}{parking_label}{collision_label}", (0, 255, 0))
                    if args.collision_detect:
                        zone = vehicle_contact_zone(xyxy)
                        zx1, zy1, zx2, zy2 = map(int, zone)
                        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 220, 255), 1)
                    cv2.circle(frame, center, 4, (0, 255, 0), -1)
                elif name == PERSON_CLASS:
                    center = box_center(xyxy)
                    speed_point = box_bottom_center(xyxy) if args.pedestrian_point == "bottom-center" else center
                    box_w, box_h = box_size(xyxy)
                    speed_label = ""
                    can_track_person = box_w >= args.pedestrian_min_box and box_h >= args.pedestrian_min_box
                    metric_point = project_ground_point(speed_point, speed_homography)
                    collision_label = ""
                    if track_id is not None and track_id in collision_labels["person"]:
                        collision_label = f" {collision_labels['person'][track_id]}"
                    if track_id is not None and args.pedestrian_speed and can_track_person:
                        person_state_key = f"person:{track_id}"
                        state = update_continuous_speed_state(
                            track_state,
                            person_state_key,
                            speed_point,
                            metric_point,
                            frame_index,
                            fps,
                            args.continuous_speed_window,
                        )
                        if state["continuous_speed_kmh"] is not None:
                            speed_label = f" walk:{state['continuous_speed_kmh']:.1f}km/h"
                            if pedestrian_speed_log is not None:
                                pedestrian_speed_log.writerow(
                                    [
                                        frame_index,
                                        f"{frame_index / fps:.2f}" if fps else "0.00",
                                        track_id,
                                        f"{state['continuous_speed_mps']:.3f}",
                                        f"{state['continuous_speed_kmh']:.3f}",
                                        speed_point[0],
                                        speed_point[1],
                                        f"{metric_point[0]:.3f}" if metric_point else "",
                                        f"{metric_point[1]:.3f}" if metric_point else "",
                                    ]
                                )
                    id_label = f" id:{track_id}" if track_id is not None else ""
                    draw_box(frame, xyxy, f"person{id_label} {conf:.2f}{speed_label}{collision_label}", (0, 165, 255))
                    cv2.circle(frame, speed_point, 4, (0, 165, 255), -1)
                elif name == LIGHT_CLASS:
                    light_color = detect_light_color(frame, xyxy)
                    draw_box(frame, xyxy, f"light:{light_color}", (0, 255, 255))

            if plate_detector is not None:
                if should_process:
                    plate_results = plate_detector.predict(
                        raw_frame,
                        conf=args.plate_conf,
                        device=args.device,
                        imgsz=args.imgsz,
                        half=args.half,
                        verbose=False,
                    )[0]
                    previous_plate_detections = cached_plate_detections
                    cached_plate_detections = [
                        {
                            "conf": float(box.conf[0]),
                            "box": box.xyxy[0].tolist(),
                            "text": carry_plate_text(box.xyxy[0].tolist(), previous_plate_detections),
                        }
                        for box in plate_results.boxes
                    ]
                for plate_detection in cached_plate_detections:
                    plate_conf = plate_detection["conf"]
                    xyxy = plate_detection["box"]
                    plate_text = plate_detection.get("text", "")
                    plate_track_id = track_id_for_plate(xyxy, cached_detections)
                    recognized_this_frame = False
                    if ocr is not None and frame_index % args.ocr_every == 0:
                        new_plate_text = ocr_plate(ocr, crop(raw_frame, xyxy), args.plate_enhance)
                        if new_plate_text:
                            plate_text = new_plate_text
                            plate_detection["text"] = plate_text
                            recognized_this_frame = True
                    if plate_track_id is not None and plate_crop_dir is not None:
                        last_saved_frame = plate_crop_last_frame.get(plate_track_id, -args.plate_crop_every)
                        if frame_index - last_saved_frame >= args.plate_crop_every:
                            saved_path = save_plate_crop(raw_frame, xyxy, plate_track_id, plate_crop_dir, plate_crop_counts)
                            if saved_path is not None:
                                plate_crop_last_frame[plate_track_id] = frame_index
                                if plate_crop_log is not None:
                                    time_sec = frame_index / fps if fps else 0
                                    coords = [int(v) for v in xyxy]
                                    plate_crop_log.writerow(
                                        [
                                            plate_track_id,
                                            plate_crop_counts[plate_track_id],
                                            saved_path.name,
                                            frame_index,
                                            f"{time_sec:.2f}",
                                            plate_text,
                                            f"{plate_conf:.3f}",
                                            *coords,
                                        ]
                                    )
                    label = f"plate:{plate_text}" if plate_text else f"plate {plate_conf:.2f}"
                    draw_box(frame, xyxy, label, (255, 0, 0), label_position="right")
                    if recognized_this_frame:
                        time_sec = frame_index / fps if fps else 0
                        coords = [int(v) for v in xyxy]
                        print(f"Plate frame={frame_index} time={time_sec:.2f}s conf={plate_conf:.2f} text={plate_text}")
                        if plate_log is not None:
                            plate_log.writerow([frame_index, f"{time_sec:.2f}", plate_text, f"{plate_conf:.3f}", *coords])

            draw_detection_lines(frame, detection_lines)

            if preview_image_path is not None and frame_index % preview_image_every == 0:
                write_preview_image(frame, preview_image_path)

            if writer is not None:
                if args.output_scale != 1.0:
                    writer.write(cv2.resize(frame, (save_width, save_height), interpolation=cv2.INTER_AREA))
                else:
                    writer.write(frame)

            if not args.no_preview:
                cv2.imshow("Traffic MVP", make_preview(frame, args.preview_scale))
                if cv2.waitKey(1) & 0xFF == 27:
                    print("Stopped early by Esc.")
                    break

            if frame_index % 50 == 0:
                print(f"Processed {frame_index}/{total_frames or '?'} frames")

            if args.max_frames and processed_frames >= args.max_frames:
                break
            if end_frame and frame_index >= end_frame:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if plate_log_file is not None:
            plate_log_file.close()
        if speed_log_file is not None:
            speed_log_file.close()
        if continuous_speed_log_file is not None:
            continuous_speed_log_file.close()
        if pedestrian_speed_log_file is not None:
            pedestrian_speed_log_file.close()
        if collision_log_file is not None:
            collision_log_file.close()
        if parking_log_file is not None:
            parking_log_file.close()
        if plate_crop_log_file is not None:
            plate_crop_log_file.close()
        cv2.destroyAllWindows()

    if args.save:
        print(f"Done. Wrote {processed_frames} frames to {args.save}")
    else:
        print(f"Done. Processed {processed_frames} frames without writing video.")


if __name__ == "__main__":
    main()
