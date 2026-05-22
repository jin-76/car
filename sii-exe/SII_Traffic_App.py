import ctypes
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Conda 打包后的 Tcl/Tk 在中文安装路径下可能自动探测失败，导致 pythonw 静默退出。
# 显式指定 Tcl/Tk 脚本目录，桌面快捷方式直接启动时也能找到 init.tcl。
TCL_LIBRARY = PROJECT_ROOT / ".venv-gpu" / "Library" / "lib" / "tcl8.6"
TK_LIBRARY = PROJECT_ROOT / ".venv-gpu" / "Library" / "lib" / "tk8.6"
if TCL_LIBRARY.exists():
    os.environ.setdefault("TCL_LIBRARY", str(TCL_LIBRARY))
if TK_LIBRARY.exists():
    os.environ.setdefault("TK_LIBRARY", str(TK_LIBRARY))

DEFAULT_VIDEO_CANDIDATES = [
    PROJECT_ROOT / "movie" / "Traffic IP Camera video[Gr0HpDM8Ki8].mp4",
    PROJECT_ROOT / "movie" / "Traffic IP Camera video [Gr0HpDM8Ki8].mp4",
]
DEFAULT_VIDEO = next((path for path in DEFAULT_VIDEO_CANDIDATES if path.exists()), DEFAULT_VIDEO_CANDIDATES[0])
DEFAULT_EXE = PROJECT_ROOT / "speedup-c" / "build-release" / "Release" / "car_speedup.exe"
DEFAULT_VEHICLE_MODEL = PROJECT_ROOT / "speedup-c" / "models" / "vehicle_yolo.onnx"
DEFAULT_PLATE_MODEL = PROJECT_ROOT / "speedup-c" / "models" / "plate_yolo.onnx"
DEFAULT_FIRE_MODEL = PROJECT_ROOT / "speedup-c" / "models" / "fire_smoke.onnx"
DEFAULT_OCR_MODEL = PROJECT_ROOT / "speedup-c" / "models" / "ocr" / "ch_PP-OCRv4_rec_infer"
DEFAULT_LINE_JSON = PROJECT_ROOT / "runs" / "annotations" / "traffic_gr0_line.json"
DEFAULT_TRT = Path("E:/TensorRT-10.9.0.34")
DEFAULT_CACHE = PROJECT_ROOT / "speedup-c" / "trt-cache-fp16"
DEFAULT_LOG_DIR = APP_DIR / "logs"
DEFAULT_WECOM_WEBHOOK = ""
APP_ICON = APP_DIR / "app.ico"

# 安装包会被复制到不同电脑，标定文件不能只依赖开发机上的绝对路径。
# 这里按视频文件名绑定预置矫正 JSON，用户选择随包测试视频时会自动切换。
PRESET_VIDEO_CONFIGS = {
    "Traffic IP Camera video [Gr0HpDM8Ki8].mp4": {
        "line_json": "preset_traffic_gr0.json",
    },
    "Traffic IP Camera video[Gr0HpDM8Ki8].mp4": {
        "line_json": "preset_traffic_gr0.json",
    },
    "Car Catches on Fire at Witchita Apartment Parking Lot [oQmCTmGlwgA].mp4": {
        "line_json": "preset_witchita_fire.json",
    },
    "Video Project 10.mp4": {
        "line_json": "preset_video_project_9.json",
    },
    "Car Accident in Parking Lot Captured by SentriForce [X4qMykwdxNo] 480p.mp4": {
        "line_json": "preset_sentri_force_collision.json",
        "parking_collision": True,
        # 该视频没有行人，若开启行人碰撞会把误检的 person 当成撞人事件。
        # 因此预设为停车场车辆接触/剐蹭检测，不启用行人碰撞检测。
        "parking_collision": True,
    },
}


def preset_config_for_source(source):
    return PRESET_VIDEO_CONFIGS.get(Path(source).name)


def preset_line_json_path(source):
    config = preset_config_for_source(source)
    if not config:
        return None
    path = PROJECT_ROOT / "runs" / "annotations" / config["line_json"]
    return path if path.exists() else None


def _same_path(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a).lower() == str(b).lower()


def latest_homography_for_source(source):
    annotation_dir = PROJECT_ROOT / "runs" / "annotations"
    if not annotation_dir.exists() or not source:
        return None
    candidates = []
    for path in annotation_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        # 开发机标定文件里可能保存了旧绝对路径；安装版优先允许按同名视频复用。
        if _same_path(data.get("source", ""), source) or Path(data.get("source", "")).name == Path(source).name:
            candidates.append(path)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return None


def default_line_json_path(source=DEFAULT_VIDEO):
    preset = preset_line_json_path(source)
    if preset:
        return preset
    matched = latest_homography_for_source(source)
    if matched:
        return matched
    return DEFAULT_LINE_JSON if DEFAULT_LINE_JSON.exists() else None


STAT_PATTERNS = {
    "vehicle": re.compile(r"Vehicle YOLO:\s+([0-9.]+) FPS,\s+([0-9.]+) ms/call,\s+calls=([0-9]+)"),
    "plate": re.compile(r"Plate YOLO:\s+([0-9.]+) FPS,\s+([0-9.]+) ms/call,\s+calls=([0-9]+)"),
    "ocr": re.compile(r"PaddleOCR:\s+([0-9.]+) FPS,\s+([0-9.]+) ms/call,\s+calls=([0-9]+)"),
    "total": re.compile(r"Total pipeline:\s+([0-9.]+) FPS,\s+([0-9.]+) ms/frame,\s+frames=([0-9]+)"),
    "progress": re.compile(r"Processed\s+([0-9]+)/([0-9]+)\s+frames"),
    "done": re.compile(r"Done\. Processed\s+([0-9]+)\s+frames"),
    "events": re.compile(r"Events:\s+parking=([0-9]+),\s+speed=([0-9]+),\s+collision=([0-9]+),\s+plate_ocr=([0-9]+)(?:,\s+fire=([0-9]+))?(?:,\s+fire_smoke=([0-9]+))?"),
    "fire": re.compile(r"Fire/smoke YOLO:\s+([0-9.]+) FPS,\s+([0-9.]+) ms/call,\s+calls=([0-9]+)"),
    "plate_event": re.compile(r"Plate frame=.*?id=([0-9]+).*?text=([^ \r\n]+)"),
    "parking_event": re.compile(r"Parking id=([0-9]+).*?point=\(([0-9]+),([0-9]+)\)"),
    "speeding_event": re.compile(r"Speeding id=([0-9]+) speed=([0-9.]+)km/h plate=([^\r\n]+)"),
    "parking_collision": re.compile(r"ParkingCollision event=(parking_contact|suspected_scratch) frame=([0-9]+) vehicle_a=([0-9]+) vehicle_b=([0-9]+) distance_m=([0-9.]+) reason=([^\r\n]+)"),
    "person_collision": re.compile(r"Collision event=suspected_accident frame=([0-9]+) person=([0-9]+) vehicle=([0-9]+) reason=([^\r\n]+)"),
    "fire_event": re.compile(r"Fire/smoke frame=.*?fire=([0-9]+)\s+smoke=([0-9]+)\s+count=([0-9]+)"),
    "fire_incident": re.compile(r"Fire incident start frame=.*?fire=([0-9]+)\s+smoke=([0-9]+)\s+event=([0-9]+)"),
}


GWL_STYLE = -16
WS_CHILD = 0x40000000
WS_VISIBLE = 0x10000000
WS_POPUP = 0x80000000
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
SW_SHOW = 5
WM_CLOSE = 0x0010


class TrafficApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI车辆监控哨兵")
        if APP_ICON.exists():
            self.iconbitmap(str(APP_ICON))
        self.geometry("1280x800")
        self.minsize(1120, 700)

        self.proc = None
        self.reader_thread = None
        self.output_queue = queue.Queue()
        self.current_log_file = None
        self.preview_hwnd = None
        self.embed_attempts = 0
        self.live_fire_count = 0
        self.event_counts = {"parking": 0, "speed": 0, "fire": 0, "scratch": 0, "injury": 0, "plate": 0}

        self.vars = {
            "source": tk.StringVar(value=str(DEFAULT_VIDEO)),
            "exe": tk.StringVar(value=str(DEFAULT_EXE)),
            "vehicle_model": tk.StringVar(value=str(DEFAULT_VEHICLE_MODEL)),
            "plate_model": tk.StringVar(value=str(DEFAULT_PLATE_MODEL)),
            "fire_model": tk.StringVar(value=str(DEFAULT_FIRE_MODEL)),
            "ocr_model": tk.StringVar(value=str(DEFAULT_OCR_MODEL)),
            "line_json": tk.StringVar(value=str(default_line_json_path() or "")),
            "trt_root": tk.StringVar(value=str(DEFAULT_TRT)),
            "cache_dir": tk.StringVar(value=str(DEFAULT_CACHE)),
            "provider": tk.StringVar(value="cuda"),
            "fp16": tk.BooleanVar(value=False),
            "standby_enable": tk.BooleanVar(value=False),
            "save_logs": tk.BooleanVar(value=True),
            "save_video": tk.BooleanVar(value=False),
            "ocr_gpu": tk.BooleanVar(value=False),
            "collision": tk.BooleanVar(value=True),
            "parking_collision": tk.BooleanVar(value=False),
            "notify_enable": tk.BooleanVar(value=False),
            "notify_provider": tk.StringVar(value="wecom"),
            "notify_webhook": tk.StringVar(value=DEFAULT_WECOM_WEBHOOK),
            "imgsz": tk.StringVar(value="1280"),
            "vehicle_every": tk.StringVar(value="3"),
            "preview_every": tk.StringVar(value="1"),
            "standby_every": tk.StringVar(value="10"),
            "ocr_every": tk.StringVar(value="10"),
            "max_ocr": tk.StringVar(value="4"),
            "fire_conf": tk.StringVar(value="0.25"),
            "fire_every_sec": tk.StringVar(value="1"),
            "speed_limit": tk.StringVar(value="45"),
            "max_frames": tk.StringVar(value="0"),
            "status": tk.StringVar(value="待机"),
            "source_name": tk.StringVar(value=DEFAULT_VIDEO.name),
        }

        self.metric_vars = {
            "overall": tk.StringVar(value="未开始"),
            "progress": tk.StringVar(value="0 / 0 帧"),
            "total": tk.StringVar(value="-- FPS"),
            "vehicle": tk.StringVar(value="-- FPS"),
            "plate": tk.StringVar(value="-- FPS"),
            "fire": tk.StringVar(value="-- FPS"),
            "ocr": tk.StringVar(value="-- ms"),
            "events_top": tk.StringVar(value="起火 0   剐蹭 0   伤人 0"),
            "events_bottom": tk.StringVar(value="停车 0   超速 0   车牌 0"),
            "latest_plate": tk.StringVar(value="暂无车牌"),
            "latest_alarm": tk.StringVar(value="暂无告警"),
        }

        self._configure_style()
        self._build_ui()
        self._bind_updates()

    def _configure_style(self):
        self.configure(bg="#e9eef3")
        style = ttk.Style(self)
        style.theme_use("vista" if "vista" in style.theme_names() else "clam")
        style.configure("TFrame", background="#e9eef3")
        style.configure("Panel.TFrame", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("Dark.TFrame", background="#111827")
        style.configure("TLabel", background="#e9eef3", foreground="#1f2937", font=("Microsoft YaHei UI", 10))
        style.configure("Panel.TLabel", background="#ffffff", foreground="#1f2937", font=("Microsoft YaHei UI", 10))
        style.configure("Dark.TLabel", background="#111827", foreground="#e5e7eb", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background="#e9eef3", foreground="#111827", font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background="#e9eef3", foreground="#4b5563", font=("Microsoft YaHei UI", 11))
        style.configure("Metric.TLabel", background="#ffffff", foreground="#111827", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Good.TLabel", background="#ffffff", foreground="#047857", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Warn.TLabel", background="#ffffff", foreground="#b45309", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 12, "bold"), padding=(18, 10))
        style.configure("Danger.TButton", font=("Microsoft YaHei UI", 12, "bold"), padding=(18, 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(10, 6))
        style.configure("TCheckbutton", background="#ffffff", font=("Microsoft YaHei UI", 10))
        style.configure("TRadiobutton", background="#ffffff", font=("Microsoft YaHei UI", 10))

    def _build_ui(self):
        root = ttk.Frame(self, padding=16)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X)
        title_box = ttk.Frame(header)
        title_box.pack(side=tk.LEFT)
        ttk.Label(title_box, text="AI车辆监控哨兵", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_box, text="实时交通与停车场智能分析系统", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.vars["status"], font=("Microsoft YaHei UI", 14, "bold")).pack(side=tk.RIGHT)

        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True, pady=(14, 0))

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(main, width=340)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        right.pack_propagate(False)

        self._build_video_panel(left)
        self._build_bottom_controls(left)
        self._build_guard_panel(right)

    def _panel(self, parent, title=None, padding=12):
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=padding)
        if title:
            ttk.Label(frame, text=title, style="Panel.TLabel", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        return frame

    def _build_video_panel(self, parent):
        panel = self._panel(parent, padding=0)
        panel.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(panel, style="Panel.TFrame", padding=(12, 10))
        top.pack(fill=tk.X)
        ttk.Label(top, text="实时监控画面", style="Panel.TLabel", font=("Microsoft YaHei UI", 13, "bold")).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.vars["source_name"], style="Panel.TLabel", foreground="#667085").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(top, textvariable=self.metric_vars["progress"], style="Panel.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(side=tk.RIGHT)

        self.video_host = tk.Frame(panel, bg="#0b1220", height=470)
        self.video_host.pack(fill=tk.BOTH, expand=True)
        self.video_host.bind("<Configure>", lambda _event: self._resize_embedded_preview())

        self.video_placeholder = tk.Label(
            self.video_host,
            text="点击“开始值守”后，实时识别画面会显示在这里",
            bg="#0b1220",
            fg="#cbd5e1",
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        self.video_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def _build_bottom_controls(self, parent):
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=(12, 0))
        self.start_btn = ttk.Button(actions, text="开始值守", style="Primary.TButton", command=self.start_run)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(actions, text="停止", style="Danger.TButton", command=self.stop_run, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(actions, text="选择视频", command=lambda: self._browse_path("source", [("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(actions, text="打开记录", command=self.open_logs_folder).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(actions, text="角度矫正", command=self.open_homography_calibrator).pack(side=tk.LEFT, padx=(10, 0))

        settings = self._panel(parent, "设置", padding=10)
        settings.pack(fill=tk.X, pady=(12, 0))

        row1 = ttk.Frame(settings, style="Panel.TFrame")
        row1.pack(fill=tk.X, pady=(6, 0))
        ttk.Radiobutton(row1, text="高性能 TensorRT", variable=self.vars["provider"], value="tensorrt").pack(side=tk.LEFT)
        ttk.Radiobutton(row1, text="普通 CUDA", variable=self.vars["provider"], value="cuda").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Checkbutton(row1, text="FP16 加速", variable=self.vars["fp16"]).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Checkbutton(row1, text="开启待机策略", variable=self.vars["standby_enable"]).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Checkbutton(row1, text="保存记录表", variable=self.vars["save_logs"]).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Checkbutton(row1, text="保存标注视频", variable=self.vars["save_video"]).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Checkbutton(row1, text="碰撞提醒", variable=self.vars["collision"]).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Checkbutton(row1, text="停车场剐蹭", variable=self.vars["parking_collision"]).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Checkbutton(row1, text="机器人通知", variable=self.vars["notify_enable"]).pack(side=tk.LEFT, padx=(16, 0))

        row2 = ttk.Frame(settings, style="Panel.TFrame")
        row2.pack(fill=tk.X, pady=(10, 0))
        for label, key, width in [
            ("清晰度", "imgsz", 8),
            ("车辆检测间隔", "vehicle_every", 8),
            ("预览刷新", "preview_every", 8),
            ("OCR间隔", "ocr_every", 8),
            ("每帧最多识别", "max_ocr", 8),
            ("限速km/h", "speed_limit", 8),
            ("最多处理帧数", "max_frames", 10),
        ]:
            ttk.Label(row2, text=label, style="Panel.TLabel").pack(side=tk.LEFT, padx=(0, 4))
            ttk.Entry(row2, textvariable=self.vars[key], width=width).pack(side=tk.LEFT, padx=(0, 12))

        row3 = ttk.Frame(settings, style="Panel.TFrame")
        row3.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(row3, text="高级路径", command=self._open_path_window).pack(side=tk.LEFT)
        ttk.Label(row3, text="通知平台：企业微信", style="Panel.TLabel").pack(side=tk.LEFT, padx=(14, 0))
        ttk.Label(row3, text="默认会自动加载 TensorRT、CUDA、OpenCV 依赖。", style="Panel.TLabel", foreground="#667085").pack(side=tk.LEFT, padx=(10, 0))

    def _build_guard_panel(self, parent):
        status = self._panel(parent, "当前状态")
        status.pack(fill=tk.X)
        ttk.Label(status, textvariable=self.metric_vars["overall"], style="Good.TLabel").pack(anchor="w", pady=(10, 0))
        ttk.Label(status, textvariable=self.metric_vars["events_top"], style="Panel.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(8, 0))
        ttk.Label(status, textvariable=self.metric_vars["events_bottom"], style="Panel.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(3, 0))

        alarm = self._panel(parent, "最新提醒")
        alarm.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(alarm, textvariable=self.metric_vars["latest_alarm"], style="Warn.TLabel", wraplength=290).pack(anchor="w", pady=(8, 0))
        ttk.Label(alarm, textvariable=self.metric_vars["latest_plate"], style="Metric.TLabel", wraplength=290).pack(anchor="w", pady=(10, 0))

        speed = self._panel(parent, "运行速度")
        speed.pack(fill=tk.X, pady=(12, 0))
        for title, key in [
            ("整体处理", "total"),
            ("车辆检测", "vehicle"),
            ("车牌检测", "plate"),
            ("起火检测", "fire"),
            ("车牌识别", "ocr"),
        ]:
            card = ttk.Frame(speed, style="Panel.TFrame", padding=(0, 8))
            card.pack(fill=tk.X)
            ttk.Label(card, text=title, style="Panel.TLabel", foreground="#667085").pack(anchor="w")
            ttk.Label(card, textvariable=self.metric_vars[key], style="Metric.TLabel").pack(anchor="w")

        log_panel = self._panel(parent, "系统记录")
        log_panel.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.console = tk.Text(
            log_panel,
            height=8,
            wrap=tk.WORD,
            bg="#111827",
            fg="#d1d5db",
            font=("Consolas", 9),
            relief=tk.FLAT,
        )
        self.console.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.console.configure(state=tk.DISABLED)

    def _bind_updates(self):
        self.vars["source"].trace_add("write", lambda *_: self._update_source_name())

    def _update_source_name(self):
        source = self.vars["source"].get()
        matched = preset_line_json_path(source) or latest_homography_for_source(source)
        if matched:
            self.vars["line_json"].set(str(matched))
        preset = preset_config_for_source(source)
        if preset:
            if "collision" in preset:
                self.vars["collision"].set(bool(preset["collision"]))
            if "parking_collision" in preset:
                self.vars["parking_collision"].set(bool(preset["parking_collision"]))
        self.vars["source_name"].set(Path(source).name if source else "未选择视频")

    def _open_path_window(self):
        win = tk.Toplevel(self)
        win.title("高级路径设置")
        win.geometry("820x365")
        win.configure(bg="#f4f6f8")
        box = ttk.Frame(win, padding=14)
        box.pack(fill=tk.BOTH, expand=True)
        for row, (label, key, directory, filetypes) in enumerate([
            ("运行程序", "exe", False, [("Executable", "*.exe"), ("All files", "*.*")]),
            ("车辆模型", "vehicle_model", False, [("ONNX", "*.onnx"), ("All files", "*.*")]),
            ("车牌模型", "plate_model", False, [("ONNX", "*.onnx"), ("All files", "*.*")]),
            ("起火模型", "fire_model", False, [("ONNX", "*.onnx"), ("All files", "*.*")]),
            ("OCR模型目录", "ocr_model", True, None),
            ("测速标定文件", "line_json", False, [("JSON", "*.json"), ("All files", "*.*")]),
            ("TensorRT目录", "trt_root", True, None),
            ("引擎缓存目录", "cache_dir", True, None),
            ("机器人Webhook", "notify_webhook", False, [("All files", "*.*")]),
        ]):
            box.columnconfigure(1, weight=1)
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(box, textvariable=self.vars[key], show="*" if key == "notify_webhook" else "").grid(row=row, column=1, sticky="ew", padx=8, pady=5)
            if key == "notify_webhook":
                ttk.Label(box, text="粘贴群机器人地址").grid(row=row, column=2, pady=5)
            else:
                ttk.Button(box, text="选择", command=lambda k=key, d=directory, f=filetypes: self._browse_path(k, f, d)).grid(row=row, column=2, pady=5)

    def _browse_path(self, key, filetypes, directory=False):
        current = self.vars[key].get()
        initial = str(Path(current).parent if current else PROJECT_ROOT)
        if directory:
            selected = filedialog.askdirectory(initialdir=initial)
        else:
            selected = filedialog.askopenfilename(initialdir=initial, filetypes=filetypes)
        if selected:
            self.vars[key].set(selected)

    def _timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _runtime_path(self, value):
        """传给 C++ 程序的路径尽量使用相对路径，避免中文安装目录在 char* argv 中乱码。"""
        if not value:
            return value
        path = Path(value)
        try:
            resolved = path.resolve()
            root = PROJECT_ROOT.resolve()
            if resolved == root:
                return "."
            return str(resolved.relative_to(root))
        except (OSError, ValueError):
            return str(value)

    def _make_env(self):
        env = os.environ.copy()
        trt_root = Path(self.vars["trt_root"].get())
        paths = [
            trt_root / "lib",
            trt_root / "bin",
            PROJECT_ROOT / ".venv-gpu" / "Lib" / "site-packages" / "torch" / "lib",
            PROJECT_ROOT / "speedup-c" / "third_party" / "opencv" / "build" / "x64" / "vc16" / "bin",
            PROJECT_ROOT / "speedup-c" / "build-release" / "Release",
        ]
        env["PATH"] = os.pathsep.join(str(p) for p in paths if p.exists()) + os.pathsep + env.get("PATH", "")
        if self.vars["notify_enable"].get() and self.vars["notify_webhook"].get().strip():
            env["SII_NOTIFY_WEBHOOK"] = self.vars["notify_webhook"].get().strip()
        return env

    def _build_command(self):
        # 基础参数保持通用；针对随包测试视频的差异化矫正/碰撞阈值在末尾追加。
        cmd = [
            self.vars["exe"].get(),
            "--source", self._runtime_path(self.vars["source"].get()),
            "--vehicle-model", self._runtime_path(self.vars["vehicle_model"].get()),
            "--plate-model", self._runtime_path(self.vars["plate_model"].get()),
            "--ocr-model-dir", self._runtime_path(self.vars["ocr_model"].get()),
            "--fire-model", self._runtime_path(self.vars["fire_model"].get()),
            "--fire-imgsz", "640",
            "--fire-conf", self.vars["fire_conf"].get(),
            "--fire-every-sec", self.vars["fire_every_sec"].get(),
            "--fire-provider", "cuda",
            "--fire-event-min-hits", "2",
            "--fire-event-clear-hits", "5",
            "--imgsz", self.vars["imgsz"].get(),
            "--yolo-provider", self.vars["provider"].get(),
            "--trt-fp16", "1" if self.vars["fp16"].get() else "0",
            "--trt-cache-dir", self._runtime_path(self.vars["cache_dir"].get()),
            "--vehicle-every", self.vars["vehicle_every"].get(),
            "--standby-enable", "1" if self.vars["standby_enable"].get() else "0",
            "--vehicle-standby-every", self.vars["standby_every"].get(),
            "--ocr-every", self.vars["ocr_every"].get(),
            "--max-ocr-per-frame", self.vars["max_ocr"].get(),
            "--speed-limit-kmh", self.vars["speed_limit"].get(),
            "--preview-scale", "1.0",
            "--preview-every", self.vars["preview_every"].get(),
            "--ocr-gpu", "1" if self.vars["ocr_gpu"].get() else "0",
        ]
        line_json = self.vars["line_json"].get().strip()
        if line_json:
            cmd.extend(["--line-json", self._runtime_path(line_json)])
        max_frames = self.vars["max_frames"].get().strip()
        if max_frames and max_frames != "0":
            cmd.extend(["--max-frames", max_frames])
        preset = preset_config_for_source(self.vars["source"].get())
        if self.vars["collision"].get():
            cmd.append("--collision-detect")
            if preset:
                cmd.extend(preset.get("collision_args", []))
        if self.vars["parking_collision"].get():
            cmd.append("--parking-collision-mode")
        if self.vars["notify_enable"].get() and self.vars["notify_webhook"].get().strip():
            cmd.extend(["--notify-provider", "wecom"])
            cmd.extend(["--notify-webhook-env", "SII_NOTIFY_WEBHOOK"])
        return cmd

    def _command_text(self, cmd):
        return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)

    def _validate_notify_webhook(self):
        if not self.vars["notify_enable"].get():
            return True
        webhook = self.vars["notify_webhook"].get().strip()
        if not webhook:
            messagebox.showerror("通知配置缺失", "启用机器人通知后，需要在高级路径里填写 Webhook。")
            return False
        if "work.weixin.qq.com/wework_admin/common/openBotProfile" in webhook:
            messagebox.showerror(
                "企业微信地址错误",
                "当前填写的是机器人管理页地址，不是发送消息 Webhook。\n\n"
                "请复制这种格式：\n"
                "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=...",
            )
            return False
        if not webhook.startswith("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key="):
            messagebox.showerror(
                "企业微信地址错误",
                "企业微信机器人 Webhook 应为：\n"
                "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=...",
            )
            return False
        return True

    def _validate(self):
        checks = [
            ("运行程序", Path(self.vars["exe"].get()).is_file()),
            ("视频源", Path(self.vars["source"].get()).exists()),
            ("车辆模型", Path(self.vars["vehicle_model"].get()).is_file()),
            ("车牌模型", Path(self.vars["plate_model"].get()).is_file()),
            ("起火模型", Path(self.vars["fire_model"].get()).is_file()),
            ("OCR模型目录", Path(self.vars["ocr_model"].get()).is_dir()),
        ]
        if self.vars["line_json"].get().strip():
            checks.append(("测速标定文件", Path(self.vars["line_json"].get()).is_file()))
        missing = [name for name, ok in checks if not ok]
        if missing:
            messagebox.showerror("文件缺失", "请检查：" + "、".join(missing))
            return False
        for key in ["imgsz", "vehicle_every", "preview_every", "standby_every", "ocr_every", "max_ocr", "max_frames"]:
            try:
                int(self.vars[key].get())
            except ValueError:
                messagebox.showerror("数字错误", f"{key} 必须是整数。")
                return False
        preview_every = int(self.vars["preview_every"].get())
        if preview_every < 1 or preview_every > 120:
            messagebox.showerror("数字错误", "预览刷新必须是 1 到 120 之间的整数。")
            return False
        try:
            float(self.vars["speed_limit"].get())
        except ValueError:
            messagebox.showerror("数字错误", "限速必须是数字。")
            return False
        for key in ["fire_conf", "fire_every_sec"]:
            try:
                float(self.vars[key].get())
            except ValueError:
                messagebox.showerror("数字错误", f"{key} 必须是数字。")
                return False
        if not self._validate_notify_webhook():
            return False
        return True

    def _prepare_run_files(self):
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        run_id = self._timestamp()
        stdout_log = DEFAULT_LOG_DIR / f"run_{run_id}.txt"
        args = []
        if self.vars["save_logs"].get():
            args.extend(["--plate-log", self._runtime_path(DEFAULT_LOG_DIR / f"plates_{run_id}.csv")])
            args.extend(["--speed-log", self._runtime_path(DEFAULT_LOG_DIR / f"speed_{run_id}.csv")])
            args.extend(["--parking-log", self._runtime_path(DEFAULT_LOG_DIR / f"parking_{run_id}.csv")])
            args.extend(["--fire-log", self._runtime_path(DEFAULT_LOG_DIR / f"fire_{run_id}.csv")])
            args.extend(["--collision-log", self._runtime_path(DEFAULT_LOG_DIR / f"collision_{run_id}.csv")])
        if self.vars["save_video"].get():
            args.extend(["--output-video", self._runtime_path(DEFAULT_LOG_DIR / f"annotated_{run_id}.mp4")])
        return stdout_log, args

    def start_run(self):
        if self.proc is not None:
            return
        if not self._validate():
            return

        self._close_stale_preview_windows()
        self._reset_run_state()
        self.update_idletasks()
        stdout_log, extra_args = self._prepare_run_files()
        cmd = self._build_command() + extra_args
        self.current_log_file = stdout_log
        self._append_console("> " + self._command_text(cmd) + "\n")

        creationflags = 0
        startupinfo = None
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=self._make_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
        except OSError as exc:
            self.proc = None
            messagebox.showerror("启动失败", str(exc))
            return

        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.vars["status"].set("值守中")
        self.metric_vars["overall"].set("正在识别")
        self.reader_thread = threading.Thread(target=self._read_process_output, args=(stdout_log,), daemon=True)
        self.reader_thread.start()
        self.after(20, self._drain_output)
        self.after(500, self._try_embed_preview)

    def stop_run(self):
        if self.proc is None:
            return
        self.vars["status"].set("正在停止")
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.terminate()
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.after(1500, self._force_stop_if_running)

    def _force_stop_if_running(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _read_process_output(self, stdout_log):
        with stdout_log.open("w", encoding="utf-8", errors="replace") as log:
            if self.proc and self.proc.stdout:
                for line in self.proc.stdout:
                    log.write(line)
                    log.flush()
                    self.output_queue.put(line)
            code = self.proc.wait() if self.proc else -1
            self.output_queue.put(("__EXIT__", code))

    def _drain_output(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__EXIT__":
                    self._finish_process(item[1])
                    return
                self._append_console(item)
                self._parse_line(item)
        except queue.Empty:
            pass
        self.after(20, self._drain_output)

    def _finish_process(self, code):
        self.proc = None
        self.preview_hwnd = None
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.vars["status"].set("待机")
        self.metric_vars["overall"].set("本次结束")
        self.video_placeholder.configure(text="本次值守已结束，点击“开始值守”可再次运行")
        self.video_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        if self.current_log_file:
            self._append_console(f"记录已保存: {self.current_log_file}\n")
        if code != 0:
            self.metric_vars["latest_alarm"].set(f"程序异常退出，代码 {code}")

    def _append_console(self, text):
        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)

    def _update_events_display(self):
        self.metric_vars["events_top"].set(
            f"起火 {self.event_counts['fire']}   "
            f"剐蹭 {self.event_counts['scratch']}   "
            f"伤人 {self.event_counts['injury']}"
        )
        self.metric_vars["events_bottom"].set(
            f"停车 {self.event_counts['parking']}   "
            f"超速 {self.event_counts['speed']}   "
            f"车牌 {self.event_counts['plate']}"
        )

    def _parse_line(self, line):
        for key in ["vehicle", "plate", "fire", "ocr", "total"]:
            match = STAT_PATTERNS[key].search(line)
            if not match:
                continue
            fps, ms, calls = match.groups()
            if key == "ocr":
                self.metric_vars[key].set(f"{float(ms):.0f} ms")
            else:
                self.metric_vars[key].set(f"{float(fps):.1f} FPS")

        match = STAT_PATTERNS["progress"].search(line)
        if match:
            self.metric_vars["progress"].set(f"{match.group(1)} / {match.group(2)} 帧")

        match = STAT_PATTERNS["done"].search(line)
        if match:
            self.metric_vars["progress"].set(f"已完成 {match.group(1)} 帧")

        match = STAT_PATTERNS["events"].search(line)
        if match:
            p, s, c, o, f, _raw_fire_smoke = match.groups()
            self.event_counts["parking"] = int(p)
            self.event_counts["speed"] = int(s)
            self.event_counts["plate"] = int(o)
            self.event_counts["fire"] = int(f or 0)
            if self.vars["parking_collision"].get():
                self.event_counts["scratch"] = int(c)
            elif self.vars["collision"].get():
                self.event_counts["injury"] = int(c)
            self._update_events_display()

        match = STAT_PATTERNS["plate_event"].search(line)
        if match:
            vehicle_id, text = match.groups()
            self.event_counts["plate"] += 1
            self._update_events_display()
            self.metric_vars["latest_plate"].set(f"车辆 {vehicle_id}  车牌 {text}")

        match = STAT_PATTERNS["parking_event"].search(line)
        if match:
            vehicle_id, x, y = match.groups()
            self.event_counts["parking"] += 1
            self._update_events_display()
            self.metric_vars["latest_alarm"].set(f"车辆 {vehicle_id} 疑似停留，位置 {x},{y}")

        match = STAT_PATTERNS["speeding_event"].search(line)
        if match:
            vehicle_id, speed, plate = match.groups()
            self.event_counts["speed"] += 1
            self._update_events_display()
            self.metric_vars["latest_alarm"].set(f"车辆 {vehicle_id} 超速 {float(speed):.1f} km/h")
            self.metric_vars["latest_plate"].set(f"车牌 {plate.strip()}")

        match = STAT_PATTERNS["parking_collision"].search(line)
        if match:
            event, _frame, vehicle_a, vehicle_b, distance_m, reason = match.groups()
            if event == "suspected_scratch":
                self.event_counts["scratch"] += 1
                self._update_events_display()
                self.metric_vars["latest_alarm"].set(
                    f"疑似停车场剐蹭：车辆 {vehicle_a} / {vehicle_b}，距离 {float(distance_m):.2f} m"
                )
            else:
                self.metric_vars["latest_alarm"].set(
                    f"停车场接触观察：车辆 {vehicle_a} / {vehicle_b}，距离 {float(distance_m):.2f} m"
                )

        match = STAT_PATTERNS["person_collision"].search(line)
        if match:
            _frame, person_id, vehicle_id, reason = match.groups()
            self.event_counts["injury"] += 1
            self._update_events_display()
            self.metric_vars["latest_alarm"].set(f"疑似伤人：行人 {person_id} / 车辆 {vehicle_id}")

        match = STAT_PATTERNS["fire_event"].search(line)
        if match:
            fire_count, smoke_count, _total_count = match.groups()
            self.metric_vars["latest_alarm"].set(f"起火 {fire_count}  烟雾 {smoke_count}")

        match = STAT_PATTERNS["fire_incident"].search(line)
        if match:
            fire_count, smoke_count, event_count = match.groups()
            self.event_counts["fire"] = int(event_count)
            self._update_events_display()
            self.metric_vars["latest_alarm"].set(f"起火事件 {event_count}：火焰 {fire_count}  烟雾 {smoke_count}")

    def _reset_run_state(self):
        self.preview_hwnd = None
        self.embed_attempts = 0
        self.live_fire_count = 0
        self.event_counts = {"parking": 0, "speed": 0, "fire": 0, "scratch": 0, "injury": 0, "plate": 0}
        self.metric_vars["overall"].set("准备启动")
        self.metric_vars["progress"].set("0 / 0 帧")
        self.metric_vars["total"].set("-- FPS")
        self.metric_vars["vehicle"].set("-- FPS")
        self.metric_vars["plate"].set("-- FPS")
        self.metric_vars["fire"].set("-- FPS")
        self.metric_vars["ocr"].set("-- ms")
        self._update_events_display()
        self.metric_vars["latest_plate"].set("暂无车牌")
        self.metric_vars["latest_alarm"].set("暂无告警")
        self.video_placeholder.configure(text="正在启动识别画面...")
        self.video_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self.console.configure(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.configure(state=tk.DISABLED)

    def _try_embed_preview(self):
        if self.proc is None or os.name != "nt":
            return
        if self.proc.poll() is not None:
            self.video_placeholder.configure(text="识别进程已退出，实时画面未启动，请查看系统记录")
            return
        hwnd = ctypes.windll.user32.FindWindowW(None, "car_speedup")
        if hwnd:
            self._embed_preview_window(hwnd)
            return
        self.embed_attempts += 1
        if self.embed_attempts < 240:
            self.after(500, self._try_embed_preview)
        else:
            self.video_placeholder.configure(text="实时画面窗口未找到，请查看是否被系统阻止")

    def _close_stale_preview_windows(self):
        if os.name != "nt":
            return
        user32 = ctypes.windll.user32
        for _ in range(5):
            hwnd = user32.FindWindowW(None, "car_speedup")
            if not hwnd:
                break
            user32.SendMessageW(hwnd, WM_CLOSE, 0, 0)
            time.sleep(0.1)

    def _embed_preview_window(self, hwnd):
        self.preview_hwnd = hwnd
        self.video_placeholder.place_forget()
        parent = self.video_host.winfo_id()
        user32 = ctypes.windll.user32
        user32.SetParent(hwnd, parent)
        style = user32.GetWindowLongW(hwnd, GWL_STYLE)
        style = (style & ~(WS_POPUP | WS_CAPTION | WS_THICKFRAME)) | WS_CHILD | WS_VISIBLE
        user32.SetWindowLongW(hwnd, GWL_STYLE, style)
        user32.ShowWindow(hwnd, SW_SHOW)
        self._resize_embedded_preview()

    def _resize_embedded_preview(self):
        if not self.preview_hwnd:
            return
        width = max(320, self.video_host.winfo_width())
        height = max(240, self.video_host.winfo_height())
        ctypes.windll.user32.MoveWindow(self.preview_hwnd, 0, 0, width, height, True)

    def open_logs_folder(self):
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(DEFAULT_LOG_DIR))
        else:
            subprocess.Popen(["xdg-open", str(DEFAULT_LOG_DIR)])

    def open_homography_calibrator(self):
        source = Path(self.vars["source"].get())
        if not source.exists():
            messagebox.showerror("无法标定", "请先选择一个有效视频。")
            return
        try:
            import cv2
            from PIL import Image, ImageTk
        except Exception as exc:
            messagebox.showerror("无法标定", f"缺少图像组件：{exc}")
            return

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            messagebox.showerror("无法标定", "视频打开失败。")
            return
        frame_index = max(1, min(30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("无法标定", "读取视频画面失败。")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        max_w, max_h = 980, 560
        scale = min(max_w / width, max_h / height, 1.0)
        view_w = int(width * scale)
        view_h = int(height * scale)
        resized = cv2.resize(rgb, (view_w, view_h), interpolation=cv2.INTER_AREA)

        win = tk.Toplevel(self)
        win.title("角度矫正")
        win.geometry(f"{view_w + 40}x{view_h + 120}")
        win.configure(bg="#e9eef3")
        win.transient(self)

        ttk.Label(
            win,
            text="依次点击梯形四个点：1近左、2近右、3远左、4远右。点完后填写实际宽度和长度。",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 6))

        canvas = tk.Canvas(win, width=view_w, height=view_h, bg="#111827", highlightthickness=0)
        canvas.pack(padx=14, pady=4)
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        canvas.image = photo
        canvas.create_image(0, 0, anchor="nw", image=photo)

        points = []
        drawings = []

        def redraw():
            for item in drawings:
                canvas.delete(item)
            drawings.clear()
            scaled = [(int(x * scale), int(y * scale)) for x, y in points]
            if len(scaled) >= 2:
                for i in range(len(scaled) - 1):
                    drawings.append(canvas.create_line(*scaled[i], *scaled[i + 1], fill="#facc15", width=3))
            if len(scaled) == 4:
                drawings.append(canvas.create_line(*scaled[3], *scaled[0], fill="#facc15", width=3))
            for index, (x, y) in enumerate(scaled, start=1):
                drawings.append(canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#facc15", outline="#111827", width=2))
                drawings.append(canvas.create_text(x + 16, y - 12, text=str(index), fill="#facc15", font=("Segoe UI", 15, "bold")))

        def save_points():
            if len(points) != 4:
                return
            road_width = simpledialog.askfloat("填写宽度", "宽度：1点到2点的实际距离（米）", minvalue=0.01, parent=win)
            if road_width is None:
                return
            road_length = simpledialog.askfloat("填写长度", "长度：12边到34边的实际距离（米）", minvalue=0.01, parent=win)
            if road_length is None:
                return

            out_dir = PROJECT_ROOT / "runs" / "annotations"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"homography_{self._timestamp()}.json"
            data = {
                "source": str(source.resolve()),
                "frame_size": [width, height],
                "homography": {
                    "image_points": [[int(x), int(y)] for x, y in points],
                    "point_order": "near-left, near-right, far-left, far-right",
                    "world_points": [
                        [0, 0],
                        [road_width, 0],
                        [0, road_length],
                        [road_width, road_length],
                    ],
                    "needs_world_points": False,
                },
            }
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.vars["line_json"].set(str(out_path))
            messagebox.showinfo("角度矫正完成", f"已保存标定文件：\n{out_path}", parent=win)
            win.destroy()

        def on_click(event):
            if len(points) >= 4:
                return
            real_x = max(0, min(width - 1, int(event.x / scale)))
            real_y = max(0, min(height - 1, int(event.y / scale)))
            points.append((real_x, real_y))
            redraw()
            if len(points) == 4:
                save_points()

        def undo():
            if points:
                points.pop()
                redraw()

        def clear():
            points.clear()
            redraw()

        canvas.bind("<Button-1>", on_click)
        controls = ttk.Frame(win)
        controls.pack(fill=tk.X, padx=14, pady=(8, 0))
        ttk.Button(controls, text="撤销一点", command=undo).pack(side=tk.LEFT)
        ttk.Button(controls, text="重新选择", command=clear).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="关闭", command=win.destroy).pack(side=tk.RIGHT)

    def destroy(self):
        if self.proc is not None:
            if messagebox.askyesno("退出", "当前还在值守，是否停止并退出？"):
                self.stop_run()
            else:
                return
        super().destroy()


if __name__ == "__main__":
    app = TrafficApp()
    app.mainloop()
