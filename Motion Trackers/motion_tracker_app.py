"""
MOTION TRACKER — PyQt6 Desktop App
====================================
Real-time motion detection and object tracking using
OpenCV MOG2 background subtraction, wrapped in a native
PyQt6 desktop UI.

Requirements:
    pip install opencv-python PyQt6

To run:
    python motion_tracker_app.py

To build a standalone binary:
    pip install pyinstaller
    pyinstaller --onefile --windowed motion_tracker_app.py
"""

import sys
import time
import numpy as np
import cv2

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QComboBox, QCheckBox, QHBoxLayout, QVBoxLayout,
    QGroupBox, QStatusBar, QFileDialog, QSizePolicy,
    QSpinBox, QFrame, QTextEdit, QSplitter
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QIcon
)

# ── palette ────────────────────────────────────────────────────────────────────
COLORS_BGR = [
    (0, 255, 136), (0, 200, 255), (180, 0, 255),
    (0, 165, 255), (0, 255, 200), (100, 255, 0),
    (255, 100, 0),  (255, 0, 150),
]

DARK = {
    "bg":        "#0f0f0f",
    "panel":     "#181818",
    "border":    "#2a2a2a",
    "accent":    "#00ff88",
    "accent2":   "#4cc9f0",
    "danger":    "#ff4444",
    "text":      "#cccccc",
    "dim":       "#555555",
    "label_bg":  "#121212",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {DARK['bg']};
    color: {DARK['text']};
    font-family: 'Courier New', monospace;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid {DARK['border']};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 6px;
    font-size: 11px;
    color: {DARK['dim']};
    letter-spacing: 2px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
QPushButton {{
    background: transparent;
    color: {DARK['accent']};
    border: 1px solid {DARK['accent']};
    border-radius: 3px;
    padding: 6px 14px;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    min-width: 80px;
}}
QPushButton:hover  {{ background: rgba(0,255,136,0.08); }}
QPushButton:pressed {{ background: rgba(0,255,136,0.18); }}
QPushButton:checked {{ background: rgba(0,255,136,0.15); border-color: {DARK['accent']}; color: {DARK['accent']}; }}
QPushButton:disabled {{ color: {DARK['dim']}; border-color: {DARK['border']}; }}
QPushButton#danger {{
    color: {DARK['danger']};
    border-color: {DARK['danger']};
}}
QPushButton#danger:hover {{ background: rgba(255,68,68,0.08); }}
QSlider::groove:horizontal {{
    height: 4px;
    background: {DARK['border']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {DARK['accent']};
    width: 12px; height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
QSlider::sub-page:horizontal {{ background: {DARK['accent']}; border-radius: 2px; }}
QComboBox {{
    background: {DARK['panel']};
    border: 1px solid {DARK['border']};
    border-radius: 3px;
    padding: 4px 8px;
    color: {DARK['text']};
    min-width: 100px;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {DARK['panel']};
    border: 1px solid {DARK['border']};
    color: {DARK['text']};
    selection-background-color: rgba(0,255,136,0.15);
}}
QCheckBox {{
    spacing: 6px;
    color: {DARK['text']};
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {DARK['border']};
    border-radius: 2px;
    background: {DARK['panel']};
}}
QCheckBox::indicator:checked {{
    background: {DARK['accent']};
    border-color: {DARK['accent']};
}}
QLabel#value_label {{
    color: {DARK['accent']};
    font-size: 11px;
    min-width: 28px;
}}
QLabel#stat_val {{
    color: {DARK['accent']};
    font-size: 13px;
    font-weight: bold;
}}
QLabel#stat_lbl {{
    color: {DARK['dim']};
    font-size: 10px;
    letter-spacing: 2px;
}}
QTextEdit {{
    background: {DARK['label_bg']};
    border: 1px solid {DARK['border']};
    color: #4a8a4a;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    border-radius: 3px;
}}
QStatusBar {{
    background: {DARK['panel']};
    color: {DARK['dim']};
    font-size: 10px;
    border-top: 1px solid {DARK['border']};
}}
QSplitter::handle {{ background: {DARK['border']}; width: 1px; }}
QSpinBox {{
    background: {DARK['panel']};
    border: 1px solid {DARK['border']};
    border-radius: 3px;
    padding: 3px 6px;
    color: {DARK['text']};
}}
QFrame#divider {{
    background: {DARK['border']};
    max-height: 1px;
}}
"""


# ── tracked object ──────────────────────────────────────────────────────────────
@dataclass
class TrackedObject:
    id: int
    cx: float; cy: float
    x: int;    y: int
    w: int;    h: int
    area: float
    last_seen: float
    is_new: bool = True
    age: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=60))
    color: tuple = field(default_factory=lambda: (0, 255, 136))

    def update(self, cx, cy, x, y, w, h, area):
        self.cx, self.cy = cx, cy
        self.x, self.y, self.w, self.h = x, y, w, h
        self.area = area
        self.last_seen = time.time()
        self.is_new = False
        self.age += 1
        self.trail.append((int(cx), int(cy)))

    def freshness(self):
        return max(0.0, 1.0 - (time.time() - self.last_seen) / 0.6)


# ── worker thread ───────────────────────────────────────────────────────────────
class TrackerThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(int, float, int)   # active_objs, fps, total
    log_event     = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.source     = 0
        self.running    = False
        self.paused     = False
        self.sensitivity = 50
        self.min_area   = 800
        self.mode       = "BOX"
        self.show_trails = True
        self.objects: list[TrackedObject] = []
        self.next_id    = 1
        self.heatmap: Optional[np.ndarray] = None
        self._bg        = None
        self._fps_times = deque(maxlen=30)
        self._timeout   = 0.8
        self._palette   = COLORS_BGR
        self._kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_big= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._total     = 0

    def _make_bg(self):
        thresh = max(8, int(80 - self.sensitivity * 0.72))
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=thresh, detectShadows=False)

    def reset(self):
        self.objects.clear()
        self.next_id = 1
        self._total = 0
        if self.heatmap is not None:
            self.heatmap[:] = 0
        self._make_bg()
        self.log_event.emit("Tracker reset")

    def _color(self, obj_id):
        return self._palette[(obj_id - 1) % len(self._palette)]

    def _process(self, frame):
        mask = self._bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.dilate(mask, self._kernel_big, iterations=2)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return mask

    def _blobs(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            blobs.append({"x":x,"y":y,"w":w,"h":h,"cx":x+w/2,"cy":y+h/2,"area":area,"contour":c})
        return blobs

    def _match(self, blobs):
        MAX_D = 120
        now   = time.time()
        matched = set()
        for b in blobs:
            best, best_d = None, MAX_D
            for obj in self.objects:
                if obj.id in matched: continue
                d = np.hypot(b["cx"]-obj.cx, b["cy"]-obj.cy)
                if d < best_d: best_d, best = d, obj
            if best:
                best.update(b["cx"],b["cy"],b["x"],b["y"],b["w"],b["h"],b["area"])
                matched.add(best.id)
            else:
                nid = self.next_id; self.next_id += 1; self._total += 1
                obj = TrackedObject(id=nid,cx=b["cx"],cy=b["cy"],x=b["x"],y=b["y"],
                                    w=b["w"],h=b["h"],area=b["area"],last_seen=now,
                                    color=self._color(nid))
                obj.trail.append((int(b["cx"]),int(b["cy"])))
                self.objects.append(obj)
                matched.add(nid)
                self.log_event.emit(f"Object #{nid} detected  ({int(b['cx'])},{int(b['cy'])})  area={int(b['area'])}px²")
        self.objects = [o for o in self.objects if now-o.last_seen < self._timeout]

    def _heat(self, shape):
        if self.heatmap is None or self.heatmap.shape[:2] != shape[:2]:
            self.heatmap = np.zeros(shape[:2], dtype=np.float32)
        for obj in self.objects:
            if obj.freshness() > 0.3:
                cv2.circle(self.heatmap,(int(obj.cx),int(obj.cy)),
                           int(np.sqrt(obj.area)*1.5),6.0,-1)
        self.heatmap *= 0.97

    # ── drawing helpers ─────────────────────────────────────────────────────────
    def _draw_box(self, out, obj):
        f   = obj.freshness()
        col = tuple(int(c*max(0.4,f)) for c in obj.color)
        pad = 10; L = 18
        x1,y1 = obj.x-pad, obj.y-pad
        x2,y2 = obj.x+obj.w+pad, obj.y+obj.h+pad
        cv2.rectangle(out,(x1,y1),(x2,y2),tuple(c//4 for c in col),1)
        for p1,p2 in [
            [(x1,y1),(x1+L,y1)],[(x1,y1),(x1,y1+L)],
            [(x2,y1),(x2-L,y1)],[(x2,y1),(x2,y1+L)],
            [(x1,y2),(x1+L,y2)],[(x1,y2),(x1,y2-L)],
            [(x2,y2),(x2-L,y2)],[(x2,y2),(x2,y2-L)],
        ]:
            cv2.line(out,p1,p2,col,2,cv2.LINE_AA)
        scan = y1+int(((time.time()*60)%(obj.h+2*pad)))
        cv2.line(out,(x1,scan),(x2,scan),tuple(c//3 for c in col),1)
        cv2.rectangle(out,(x1,y1-18),(x1+140,y1),(8,8,8),-1)
        label = f"#{obj.id}  {'NEW' if obj.is_new else 'MOVE'}  {obj.w}x{obj.h}"
        cv2.putText(out,label,(x1+3,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)

    def _draw_dot(self, out, obj):
        f   = obj.freshness()
        col = tuple(int(c*max(0.4,f)) for c in obj.color)
        r   = max(6,int(np.sqrt(obj.area)*0.5))
        cv2.circle(out,(int(obj.cx),int(obj.cy)),r,tuple(c//3 for c in col),-1)
        cv2.circle(out,(int(obj.cx),int(obj.cy)),5,col,-1)
        cv2.circle(out,(int(obj.cx),int(obj.cy)),r,col,1,cv2.LINE_AA)
        cv2.putText(out,f"#{obj.id}",(int(obj.cx)+r+4,int(obj.cy)-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)

    def _draw_outline(self, out, obj, contour):
        f   = obj.freshness()
        col = tuple(int(c*max(0.4,f)) for c in obj.color)
        cv2.drawContours(out,[contour],-1,col,2,cv2.LINE_AA)
        cv2.circle(out,(int(obj.cx),int(obj.cy)),4,col,-1)
        cv2.putText(out,f"#{obj.id}",(int(obj.cx)+6,int(obj.cy)-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)

    def _draw_trails(self, out):
        for obj in self.objects:
            pts = list(obj.trail)
            for i in range(1, len(pts)):
                a = i/len(pts)
                col = tuple(int(c*a*0.7) for c in obj.color)
                cv2.line(out,pts[i-1],pts[i],col,2,cv2.LINE_AA)

    def _draw_heatmap(self, out):
        if self.heatmap is None: return
        norm    = np.clip(self.heatmap/80.0,0,1)
        heat_u8 = (norm*255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
        mask    = norm > 0.05
        out[mask] = cv2.addWeighted(out,0.4,colored,0.6,0)[mask]

    # ── main loop ───────────────────────────────────────────────────────────────
    def run(self):
        self._make_bg()
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.log_event.emit(f"ERROR: cannot open '{self.source}'")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.log_event.emit("Camera opened")
        self.running = True

        while self.running:
            if self.paused:
                self.msleep(30)
                continue

            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break

            now = time.time()
            self._fps_times.append(now)
            fps = 0.0
            if len(self._fps_times) >= 2:
                fps = (len(self._fps_times)-1)/(self._fps_times[-1]-self._fps_times[0])

            mask  = self._process(frame)
            blobs = self._blobs(mask)
            self._match(blobs)
            self._heat(frame.shape)

            out = frame.copy()
            if self.mode == "HEATMAP":
                self._draw_heatmap(out)
            if self.show_trails:
                self._draw_trails(out)

            active = 0
            for obj in self.objects:
                if now - obj.last_seen > 0.15: continue
                active += 1
                if   self.mode == "BOX":     self._draw_box(out, obj)
                elif self.mode == "DOT":     self._draw_dot(out, obj)
                elif self.mode == "OUTLINE":
                    best_c, best_d = None, 999
                    for b in blobs:
                        d = np.hypot(b["cx"]-obj.cx, b["cy"]-obj.cy)
                        if d < best_d: best_d,best_c = d,b["contour"]
                    if best_c is not None: self._draw_outline(out,obj,best_c)
                elif self.mode == "HEATMAP": self._draw_dot(out, obj)

            self.frame_ready.emit(out)
            self.stats_updated.emit(active, fps, self._total)

        cap.release()
        self.log_event.emit("Camera closed")

    def stop(self):
        self.running = False
        self.wait(2000)


# ── stat widget ─────────────────────────────────────────────────────────────────
class StatBox(QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8,4,8,4)
        lay.setSpacing(1)
        self.val = QLabel("0")
        self.val.setObjectName("stat_val")
        self.val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel(label)
        lbl.setObjectName("stat_lbl")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.val)
        lay.addWidget(lbl)

    def set(self, v): self.val.setText(str(v))


# ── main window ─────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Tracker")
        self.setMinimumSize(1100, 680)
        self._screenshots = 0
        self._last_frame: Optional[np.ndarray] = None

        self.thread = TrackerThread()
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.stats_updated.connect(self._on_stats)
        self.thread.log_event.connect(self._on_log)

        self._build_ui()
        self.setStyleSheet(STYLESHEET)

    # ── UI construction ─────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10,10,10,10)
        root.setSpacing(10)

        # ── left: video feed ──────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(f"background:#000; border:1px solid {DARK['border']}; border-radius:4px;")
        self.video_label.setText("No camera feed\nPress  START  to begin")
        self.video_label.setFont(QFont("Courier New", 13))

        # stat bar under video
        stat_row = QHBoxLayout()
        self.stat_objs  = StatBox("ACTIVE")
        self.stat_fps   = StatBox("FPS")
        self.stat_total = StatBox("TOTAL SEEN")
        for s in (self.stat_objs, self.stat_fps, self.stat_total):
            stat_row.addWidget(s)

        left.addWidget(self.video_label, stretch=1)
        left.addLayout(stat_row)

        # ── right: controls ──────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)
        right.setContentsMargins(0,0,0,0)

        # header
        title = QLabel("MOTION TRACKER")
        title.setFont(QFont("Courier New", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{DARK['accent']}; letter-spacing:4px;")
        right.addWidget(title)

        div = QFrame(); div.setObjectName("divider"); right.addWidget(div)

        # ── camera group ─────────────────────────────────────────────────────
        cam_grp = QGroupBox("CAMERA SOURCE")
        cam_lay = QVBoxLayout(cam_grp)
        cam_lay.setSpacing(6)

        cam_row = QHBoxLayout()
        self.cam_combo = QComboBox()
        for i in range(4): self.cam_combo.addItem(f"Camera {i}", i)
        cam_row.addWidget(QLabel("Device:"))
        cam_row.addWidget(self.cam_combo, 1)
        cam_lay.addLayout(cam_row)

        file_row = QHBoxLayout()
        self.file_btn = QPushButton("Open Video File…")
        self.file_btn.clicked.connect(self._pick_file)
        self.file_lbl = QLabel("(none)")
        self.file_lbl.setStyleSheet(f"color:{DARK['dim']}; font-size:10px;")
        self.file_lbl.setWordWrap(True)
        file_row.addWidget(self.file_btn)
        cam_lay.addLayout(file_row)
        cam_lay.addWidget(self.file_lbl)

        right.addWidget(cam_grp)

        # ── controls group ───────────────────────────────────────────────────
        ctrl_grp = QGroupBox("CONTROLS")
        ctrl_lay = QVBoxLayout(ctrl_grp)
        ctrl_lay.setSpacing(8)

        btn_row1 = QHBoxLayout()
        self.btn_start = QPushButton("▶  START")
        self.btn_start.clicked.connect(self._start)
        self.btn_pause = QPushButton("⏸  PAUSE")
        self.btn_pause.setCheckable(True)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        btn_row1.addWidget(self.btn_start)
        btn_row1.addWidget(self.btn_pause)
        btn_row1.addWidget(self.btn_stop)
        ctrl_lay.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_reset = QPushButton("↺  RESET")
        self.btn_reset.clicked.connect(self._reset)
        self.btn_save = QPushButton("📷  SCREENSHOT")
        self.btn_save.clicked.connect(self._screenshot)
        btn_row2.addWidget(self.btn_reset)
        btn_row2.addWidget(self.btn_save)
        ctrl_lay.addLayout(btn_row2)

        right.addWidget(ctrl_grp)

        # ── display group ─────────────────────────────────────────────────────
        disp_grp = QGroupBox("DISPLAY")
        disp_lay = QVBoxLayout(disp_grp)
        disp_lay.setSpacing(8)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        for m in ["BOX", "DOT", "HEATMAP", "OUTLINE"]:
            self.mode_combo.addItem(m)
        self.mode_combo.currentTextChanged.connect(self._set_mode)
        mode_row.addWidget(self.mode_combo, 1)
        disp_lay.addLayout(mode_row)

        self.chk_trails = QCheckBox("Motion Trails")
        self.chk_trails.setChecked(True)
        self.chk_trails.toggled.connect(lambda v: setattr(self.thread, 'show_trails', v))
        disp_lay.addWidget(self.chk_trails)

        right.addWidget(disp_grp)

        # ── tuning group ──────────────────────────────────────────────────────
        tune_grp = QGroupBox("TUNING")
        tune_lay = QVBoxLayout(tune_grp)
        tune_lay.setSpacing(8)

        # sensitivity
        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Sensitivity"))
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(5, 100)
        self.sens_slider.setValue(50)
        self.sens_val = QLabel("50"); self.sens_val.setObjectName("value_label")
        self.sens_slider.valueChanged.connect(self._set_sensitivity)
        sens_row.addWidget(self.sens_slider, 1)
        sens_row.addWidget(self.sens_val)
        tune_lay.addLayout(sens_row)

        # min area
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Min Area (px²)"))
        self.area_spin = QSpinBox()
        self.area_spin.setRange(50, 10000)
        self.area_spin.setSingleStep(100)
        self.area_spin.setValue(800)
        self.area_spin.valueChanged.connect(lambda v: setattr(self.thread, 'min_area', v))
        area_row.addWidget(self.area_spin)
        tune_lay.addLayout(area_row)

        # object timeout
        to_row = QHBoxLayout()
        to_row.addWidget(QLabel("Object Timeout (s)"))
        self.timeout_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeout_slider.setRange(1, 30)
        self.timeout_slider.setValue(8)
        self.timeout_val = QLabel("0.8s"); self.timeout_val.setObjectName("value_label")
        self.timeout_slider.valueChanged.connect(self._set_timeout)
        to_row.addWidget(self.timeout_slider, 1)
        to_row.addWidget(self.timeout_val)
        tune_lay.addLayout(to_row)

        right.addWidget(tune_grp)

        # ── log ───────────────────────────────────────────────────────────────
        log_grp = QGroupBox("EVENT LOG")
        log_lay = QVBoxLayout(log_grp)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        log_lay.addWidget(self.log_box)
        right.addWidget(log_grp)

        right.addStretch()

        # ── assemble ──────────────────────────────────────────────────────────
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(280)

        root.addLayout(left, 1)
        root.addWidget(right_widget)

        # status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — choose a camera source and press START")

    # ── slots ──────────────────────────────────────────────────────────────────
    def _on_frame(self, frame: np.ndarray):
        self._last_frame = frame
        h, w, ch = frame.shape
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pix   = QPixmap.fromImage(img)
        label_size = self.video_label.size()
        self.video_label.setPixmap(
            pix.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        )

    def _on_stats(self, active, fps, total):
        self.stat_objs.set(active)
        self.stat_fps.set(f"{fps:.1f}")
        self.stat_total.set(total)

    def _on_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.append(f'<span style="color:#4a8a4a">[{ts}]</span> {msg}')
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())
        self.status.showMessage(msg, 4000)

    def _start(self):
        if self.thread.isRunning(): return
        src = self.cam_combo.currentData()
        # if file was picked, thread.source already set
        if not hasattr(self, '_file_source'):
            self.thread.source = src
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.thread.running = True
        self.thread.paused  = False
        self.thread.start()
        self.status.showMessage("Tracking…")

    def _pause(self, checked):
        self.thread.paused = checked
        self.btn_pause.setText("▶  RESUME" if checked else "⏸  PAUSE")
        self.status.showMessage("Paused" if checked else "Tracking…")

    def _stop(self):
        self.thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setChecked(False)
        self.btn_pause.setText("⏸  PAUSE")
        self.btn_stop.setEnabled(False)
        self.video_label.setText("Stopped\nPress START to begin again")
        self.status.showMessage("Stopped")

    def _reset(self):
        self.thread.reset()

    def _set_mode(self, mode):
        self.thread.mode = mode

    def _set_sensitivity(self, val):
        self.sens_val.setText(str(val))
        self.thread.sensitivity = val
        thresh = max(8, int(80 - val * 0.72))
        if self.thread._bg:
            self.thread._bg.setVarThreshold(thresh)

    def _set_timeout(self, val):
        t = val / 10.0
        self.timeout_val.setText(f"{t:.1f}s")
        self.thread._timeout = t

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if path:
            self.thread.source = path
            self._file_source  = path
            self.file_lbl.setText(path.split("/")[-1])
            self.cam_combo.setEnabled(False)
            self.status.showMessage(f"Source: {path}")

    def _screenshot(self):
        if self._last_frame is None:
            self.status.showMessage("No frame to save"); return
        self._screenshots += 1
        fname = f"tracker_shot_{self._screenshots:03d}.png"
        cv2.imwrite(fname, self._last_frame)
        self.status.showMessage(f"Saved {fname}")
        self._on_log(f"Screenshot saved → {fname}")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


# ── entry ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Motion Tracker")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
