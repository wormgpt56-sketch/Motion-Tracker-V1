"""
MOTION TRACKER v2 — PyQt6 Desktop App
=======================================
Full upgrade with:
  - Object classification (Person / Vehicle / Animal / Unknown)
  - Motion recording — saves .avi clips on detection
  - Alert system — sound beep + Windows toast notification
  - Dashboard with live graphs (FPS, object count over time)
  - Settings page
  - IP camera / stream URL support

Requirements:
    pip install opencv-python PyQt6

Optional (Windows toast notifications):
    pip install win10toast

Run:
    python motion_tracker_v2.py

Build standalone:
    pip install pyinstaller
    pyinstaller --onefile --windowed motion_tracker_v2.py
"""

import sys, os, time, math, threading, subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QComboBox, QCheckBox, QHBoxLayout, QVBoxLayout,
    QGroupBox, QStatusBar, QFileDialog, QSizePolicy,
    QSpinBox, QFrame, QTextEdit, QTabWidget, QGridLayout,
    QLineEdit, QScrollArea, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QRectF
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QPainter, QPen, QColor,
    QBrush, QPolygonF, QPainterPath
)

# ── theme ──────────────────────────────────────────────────────────────────────
A = {
    "bg":      "#0a0a0f",
    "panel":   "#13131a",
    "panel2":  "#1a1a24",
    "border":  "#2a2a3a",
    "accent":  "#00ff88",
    "blue":    "#4cc9f0",
    "purple":  "#9d4edd",
    "orange":  "#f4a261",
    "red":     "#ff4444",
    "yellow":  "#f0c040",
    "text":    "#cccccc",
    "dim":     "#4a4a5a",
}

SS = f"""
QMainWindow, QWidget {{
    background:{A['bg']}; color:{A['text']};
    font-family:'Courier New',monospace; font-size:12px;
}}
QTabWidget::pane {{ border:1px solid {A['border']}; background:{A['panel']}; }}
QTabBar::tab {{
    background:{A['panel2']}; color:{A['dim']};
    padding:8px 18px; border:1px solid {A['border']};
    border-bottom:none; font-size:11px; letter-spacing:1px;
}}
QTabBar::tab:selected {{ background:{A['panel']}; color:{A['accent']}; border-bottom:2px solid {A['accent']}; }}
QGroupBox {{
    border:1px solid {A['border']}; border-radius:4px;
    margin-top:8px; padding-top:6px;
    font-size:10px; color:{A['dim']}; letter-spacing:2px;
}}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}
QPushButton {{
    background:transparent; color:{A['accent']};
    border:1px solid {A['accent']}; border-radius:3px;
    padding:6px 14px; font-family:'Courier New',monospace;
    font-size:11px; letter-spacing:1px; min-width:70px;
}}
QPushButton:hover  {{ background:rgba(0,255,136,.08); }}
QPushButton:pressed {{ background:rgba(0,255,136,.18); }}
QPushButton:checked {{ background:rgba(0,255,136,.15); }}
QPushButton:disabled {{ color:{A['dim']}; border-color:{A['border']}; }}
QPushButton#red   {{ color:{A['red']};    border-color:{A['red']};    }}
QPushButton#red:hover {{ background:rgba(255,68,68,.08); }}
QPushButton#blue  {{ color:{A['blue']};   border-color:{A['blue']};   }}
QPushButton#blue:hover {{ background:rgba(76,201,240,.08); }}
QPushButton#yellow {{ color:{A['yellow']}; border-color:{A['yellow']}; }}
QPushButton#yellow:hover {{ background:rgba(240,192,64,.08); }}
QSlider::groove:horizontal {{ height:4px; background:{A['border']}; border-radius:2px; }}
QSlider::handle:horizontal {{
    background:{A['accent']}; width:12px; height:12px;
    margin:-4px 0; border-radius:6px;
}}
QSlider::sub-page:horizontal {{ background:{A['accent']}; border-radius:2px; }}
QComboBox, QLineEdit, QSpinBox {{
    background:{A['panel2']}; border:1px solid {A['border']};
    border-radius:3px; padding:4px 8px; color:{A['text']};
}}
QComboBox::drop-down {{ border:none; width:20px; }}
QComboBox QAbstractItemView {{
    background:{A['panel2']}; border:1px solid {A['border']};
    color:{A['text']}; selection-background-color:rgba(0,255,136,.15);
}}
QCheckBox {{ spacing:6px; color:{A['text']}; }}
QCheckBox::indicator {{
    width:14px; height:14px; border:1px solid {A['border']};
    border-radius:2px; background:{A['panel2']};
}}
QCheckBox::indicator:checked {{ background:{A['accent']}; border-color:{A['accent']}; }}
QTextEdit {{
    background:{A['panel2']}; border:1px solid {A['border']};
    color:#4a8a4a; font-size:10px; border-radius:3px;
}}
QScrollBar:vertical {{
    background:{A['panel']}; width:6px; border-radius:3px;
}}
QScrollBar::handle:vertical {{ background:{A['border']}; border-radius:3px; }}
QStatusBar {{ background:{A['panel']}; color:{A['dim']}; font-size:10px; border-top:1px solid {A['border']}; }}
QFrame#div {{ background:{A['border']}; max-height:1px; }}
QLabel#val {{ color:{A['accent']}; font-size:11px; min-width:32px; }}
QLabel#bigval {{ color:{A['accent']}; font-size:20px; font-weight:bold; }}
QLabel#biglbl {{ color:{A['dim']}; font-size:10px; letter-spacing:2px; }}
QLabel#cls_person  {{ color:#4cc9f0; font-size:10px; }}
QLabel#cls_vehicle {{ color:#f4a261; font-size:10px; }}
QLabel#cls_unknown {{ color:#9d4edd; font-size:10px; }}
"""

# ── colours per class ──────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Person":  (240, 200, 76),   # blue-ish (BGR)
    "Vehicle": (97, 162, 244),   # orange
    "Animal":  (100, 220, 100),  # green
    "Unknown": (180, 80, 200),   # purple
}
OBJ_PALETTE = [
    (0,255,136),(0,200,255),(180,80,255),
    (97,162,244),(100,220,100),(255,160,0),
    (0,180,255),(200,80,180),
]

# ── HOG person detector (loaded once) ─────────────────────────────────────────
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def classify_blob(frame, x, y, w, h, area) -> str:
    """Lightweight classification: HOG for person, aspect+area heuristics for rest."""
    aspect = w / max(h, 1)
    # try HOG on the ROI
    pad = 10
    rx = max(0, x-pad); ry = max(0, y-pad)
    rw = min(frame.shape[1]-rx, w+pad*2)
    rh = min(frame.shape[0]-ry, h+pad*2)
    roi = frame[ry:ry+rh, rx:rx+rw]
    if roi.size > 0 and rw > 32 and rh > 64:
        rects, _ = _HOG.detectMultiScale(roi, winStride=(8,8), padding=(4,4), scale=1.05)
        if len(rects) > 0:
            return "Person"
    # heuristic: wide + large → vehicle
    if aspect > 1.4 and area > 4000:
        return "Vehicle"
    # tall-ish + medium → person fallback
    if aspect < 0.9 and area > 1500:
        return "Person"
    return "Unknown"


# ── tracked object ─────────────────────────────────────────────────────────────
@dataclass
class TrackedObject:
    id: int
    cx: float; cy: float
    x: int;    y: int
    w: int;    h: int
    area: float
    last_seen: float
    label: str = "Unknown"
    is_new: bool = True
    age: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=60))
    color: tuple = field(default_factory=lambda: (0,255,136))
    classify_countdown: int = 3   # classify after N frames of stability

    def update(self, cx, cy, x, y, w, h, area):
        self.cx,self.cy = cx,cy
        self.x,self.y,self.w,self.h = x,y,w,h
        self.area = area
        self.last_seen = time.time()
        self.is_new = False
        self.age += 1
        self.trail.append((int(cx),int(cy)))

    def freshness(self):
        return max(0.0, 1.0-(time.time()-self.last_seen)/0.6)


# ── alert system ───────────────────────────────────────────────────────────────
class AlertSystem:
    def __init__(self):
        self.sound_enabled = True
        self.notif_enabled = True
        self._last_alert   = 0
        self._cooldown     = 3.0   # seconds between alerts

    def trigger(self, label: str):
        now = time.time()
        if now - self._last_alert < self._cooldown:
            return
        self._last_alert = now
        if self.sound_enabled:
            threading.Thread(target=self._beep, daemon=True).start()
        if self.notif_enabled:
            threading.Thread(target=self._notify, args=(label,), daemon=True).start()

    def _beep(self):
        try:
            import winsound
            winsound.Beep(880, 120)
            time.sleep(0.08)
            winsound.Beep(1100, 80)
        except Exception:
            pass   # non-Windows: silent

    def _notify(self, label: str):
        try:
            import subprocess
            subprocess.Popen([
                'powershell', '-WindowStyle', 'Hidden', '-Command',
                f'''
                [Windows.UI.Notifications.ToastNotificationManager,
                 Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
                $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
                    [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                $xml.GetElementsByTagName("text")[0].AppendChild(
                    $xml.CreateTextNode("Motion Tracker")) | Out-Null
                $xml.GetElementsByTagName("text")[1].AppendChild(
                    $xml.CreateTextNode("{label} detected")) | Out-Null
                $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
                [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier(
                    "Motion Tracker").Show($toast)
                '''
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


# ── video recorder ────────────────────────────────────────────────────────────
class VideoRecorder:
    def __init__(self):
        self.active    = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._path     = ""
        self._timeout  = 3.0   # seconds of silence before closing clip
        self._last_motion = 0
        self._fps      = 20.0
        self._timer    = None
        self.save_dir  = "recordings"
        self._clip_num = 0

    def motion_detected(self, frame):
        self._last_motion = time.time()
        if not self.active:
            return
        if self._writer is None:
            self._start_clip(frame)
        if self._writer:
            self._writer.write(frame)

    def _start_clip(self, frame):
        os.makedirs(self.save_dir, exist_ok=True)
        self._clip_num += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(self.save_dir, f"clip_{ts}_{self._clip_num:03d}.avi")
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (w,h))

    def tick(self):
        """Call periodically — closes clip if motion has stopped."""
        if self._writer and time.time()-self._last_motion > self._timeout:
            self._writer.release()
            self._writer = None
            return self._path   # return path of finished clip
        return None

    def stop(self):
        if self._writer:
            self._writer.release()
            self._writer = None


# ── sparkline widget ──────────────────────────────────────────────────────────
class SparkLine(QWidget):
    def __init__(self, color="#00ff88", max_points=120, label="", parent=None):
        super().__init__(parent)
        self.color  = QColor(color)
        self.data   = deque(maxlen=max_points)
        self.label  = label
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def push(self, v):
        self.data.append(v)
        self.update()

    def paintEvent(self, e):
        if not self.data: return
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w,h = self.width(), self.height()
        mx  = max(self.data) or 1
        pts = list(self.data)
        n   = len(pts)

        # background
        p.fillRect(0,0,w,h, QColor(A['panel2']))

        # grid line
        p.setPen(QPen(QColor(A['border']), 1))
        p.drawLine(0, h//2, w, h//2)

        # fill
        path = QPainterPath()
        path.moveTo(0, h)
        for i,v in enumerate(pts):
            x = int(i/(n-1)*w) if n>1 else 0
            y = int(h - (v/mx)*(h-6) - 3)
            path.lineTo(x,y)
        path.lineTo(w,h)
        path.closeSubpath()
        fill = QColor(self.color); fill.setAlpha(40)
        p.fillPath(path, QBrush(fill))

        # line
        p.setPen(QPen(self.color, 1.5))
        for i in range(1,n):
            x1 = int((i-1)/(n-1)*w) if n>1 else 0
            x2 = int(i/(n-1)*w)     if n>1 else 0
            y1 = int(h-(pts[i-1]/mx)*(h-6)-3)
            y2 = int(h-(pts[i]/mx)*(h-6)-3)
            p.drawLine(x1,y1,x2,y2)

        # label + current value
        p.setPen(QColor(A['dim']))
        p.setFont(QFont("Courier New",8))
        p.drawText(4,12,self.label)
        p.setPen(self.color)
        p.setFont(QFont("Courier New",9,QFont.Weight.Bold))
        p.drawText(4,h-4,f"{pts[-1]:.1f}")


# ── tracker thread ────────────────────────────────────────────────────────────
class TrackerThread(QThread):
    frame_ready    = pyqtSignal(np.ndarray)
    stats_updated  = pyqtSignal(int, float, int, int)  # active,fps,total,recording
    log_event      = pyqtSignal(str)
    alert_signal   = pyqtSignal(str)
    clip_saved     = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.source       = 0
        self.running      = False
        self.paused       = False
        self.sensitivity  = 50
        self.min_area     = 800
        self.mode         = "BOX"
        self.show_trails  = True
        self.do_classify  = True
        self.objects: list[TrackedObject] = []
        self.next_id      = 1
        self.heatmap: Optional[np.ndarray] = None
        self._bg          = None
        self._fps_times   = deque(maxlen=30)
        self._timeout     = 0.8
        self._total       = 0
        self._kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self._kernel_big  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        self.recorder     = VideoRecorder()
        self.alert        = AlertSystem()

    def _make_bg(self):
        thresh = max(8, int(80-self.sensitivity*0.72))
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=thresh, detectShadows=False)

    def reset(self):
        self.objects.clear(); self.next_id=1; self._total=0
        if self.heatmap is not None: self.heatmap[:]=0
        self._make_bg()
        self.log_event.emit("Tracker reset")

    def _col(self, oid):
        return OBJ_PALETTE[(oid-1)%len(OBJ_PALETTE)]

    def _process(self, frame):
        mask = self._bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.dilate(mask, self._kernel_big, iterations=2)
        _, mask = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
        return mask

    def _blobs(self, mask):
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        out=[]
        for c in contours:
            area=cv2.contourArea(c)
            if area<self.min_area: continue
            x,y,w,h=cv2.boundingRect(c)
            out.append({"x":x,"y":y,"w":w,"h":h,"cx":x+w/2,"cy":y+h/2,"area":area,"contour":c})
        return out

    def _match(self, blobs, frame):
        MAX_D=120; now=time.time(); matched=set()
        for b in blobs:
            best,best_d=None,MAX_D
            for obj in self.objects:
                if obj.id in matched: continue
                d=np.hypot(b["cx"]-obj.cx,b["cy"]-obj.cy)
                if d<best_d: best_d,best=d,obj
            if best:
                best.update(b["cx"],b["cy"],b["x"],b["y"],b["w"],b["h"],b["area"])
                matched.add(best.id)
                if self.do_classify and best.classify_countdown>0:
                    best.classify_countdown-=1
                    if best.classify_countdown==0:
                        best.label=classify_blob(frame,best.x,best.y,best.w,best.h,best.area)
                        best.color=CLASS_COLORS.get(best.label,OBJ_PALETTE[0])
            else:
                nid=self.next_id; self.next_id+=1; self._total+=1
                obj=TrackedObject(id=nid,cx=b["cx"],cy=b["cy"],x=b["x"],y=b["y"],
                                  w=b["w"],h=b["h"],area=b["area"],last_seen=now,
                                  color=self._col(nid))
                obj.trail.append((int(b["cx"]),int(b["cy"])))
                self.objects.append(obj); matched.add(nid)
                self.log_event.emit(f"#{nid} detected  area={int(b['area'])}px²")
                self.alert_signal.emit(obj.label)
        self.objects=[o for o in self.objects if now-o.last_seen<self._timeout]

    def _heat(self, shape):
        if self.heatmap is None or self.heatmap.shape[:2]!=shape[:2]:
            self.heatmap=np.zeros(shape[:2],dtype=np.float32)
        for obj in self.objects:
            if obj.freshness()>0.3:
                cv2.circle(self.heatmap,(int(obj.cx),int(obj.cy)),
                           int(np.sqrt(obj.area)*1.5),6.0,-1)
        self.heatmap*=0.97

    # ── drawing ────────────────────────────────────────────────────────────────
    def _draw_box(self, out, obj):
        f=obj.freshness(); col=tuple(int(c*max(.4,f)) for c in obj.color)
        pad=10; L=18
        x1,y1=obj.x-pad,obj.y-pad; x2,y2=obj.x+obj.w+pad,obj.y+obj.h+pad
        cv2.rectangle(out,(x1,y1),(x2,y2),tuple(c//4 for c in col),1)
        for p1,p2 in [
            [(x1,y1),(x1+L,y1)],[(x1,y1),(x1,y1+L)],
            [(x2,y1),(x2-L,y1)],[(x2,y1),(x2,y1+L)],
            [(x1,y2),(x1+L,y2)],[(x1,y2),(x1,y2-L)],
            [(x2,y2),(x2-L,y2)],[(x2,y2),(x2,y2-L)],
        ]: cv2.line(out,p1,p2,col,2,cv2.LINE_AA)
        scan=y1+int(((time.time()*60)%(obj.h+2*pad)))
        cv2.line(out,(x1,scan),(x2,scan),tuple(c//3 for c in col),1)
        cv2.rectangle(out,(x1,y1-20),(x1+170,y1),(8,8,8),-1)
        tag="NEW" if obj.is_new else obj.label
        cv2.putText(out,f"#{obj.id} {tag}  {obj.w}x{obj.h}px",
                    (x1+3,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)

    def _draw_dot(self, out, obj):
        f=obj.freshness(); col=tuple(int(c*max(.4,f)) for c in obj.color)
        r=max(6,int(np.sqrt(obj.area)*0.5))
        cv2.circle(out,(int(obj.cx),int(obj.cy)),r,tuple(c//3 for c in col),-1)
        cv2.circle(out,(int(obj.cx),int(obj.cy)),5,col,-1)
        cv2.circle(out,(int(obj.cx),int(obj.cy)),r,col,1,cv2.LINE_AA)
        cv2.putText(out,f"#{obj.id}",(int(obj.cx)+r+4,int(obj.cy)-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)

    def _draw_outline(self, out, obj, contour):
        f=obj.freshness(); col=tuple(int(c*max(.4,f)) for c in obj.color)
        cv2.drawContours(out,[contour],-1,col,2,cv2.LINE_AA)
        cv2.circle(out,(int(obj.cx),int(obj.cy)),4,col,-1)
        cv2.putText(out,f"#{obj.id} {obj.label}",(int(obj.cx)+6,int(obj.cy)-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)

    def _draw_trails(self, out):
        for obj in self.objects:
            pts=list(obj.trail)
            for i in range(1,len(pts)):
                a=i/len(pts)
                col=tuple(int(c*a*.7) for c in obj.color)
                cv2.line(out,pts[i-1],pts[i],col,2,cv2.LINE_AA)

    def _draw_heat(self, out):
        if self.heatmap is None: return
        norm=np.clip(self.heatmap/80.,0,1)
        heat_u8=(norm*255).astype(np.uint8)
        colored=cv2.applyColorMap(heat_u8,cv2.COLORMAP_INFERNO)
        mask=norm>0.05
        out[mask]=cv2.addWeighted(out,.4,colored,.6,0)[mask]

    def _draw_rec_indicator(self, out):
        if self.recorder.active and self.recorder._writer:
            cv2.circle(out,(out.shape[1]-20,20),8,(0,0,255),-1)
            cv2.putText(out,"REC",(out.shape[1]-45,26),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

    def run(self):
        self._make_bg()
        cap=cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.log_event.emit(f"ERROR: cannot open '{self.source}'"); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        self.log_event.emit("Camera opened"); self.running=True
        rec_tick=0

        while self.running:
            if self.paused: self.msleep(30); continue
            ret,frame=cap.read()
            if not ret:
                if isinstance(self.source,str): cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
                break

            now=time.time()
            self._fps_times.append(now)
            fps=0.0
            if len(self._fps_times)>=2:
                fps=(len(self._fps_times)-1)/(self._fps_times[-1]-self._fps_times[0])

            mask=self._process(frame)
            blobs=self._blobs(mask)
            self._match(blobs,frame)
            self._heat(frame.shape)

            # recording
            if blobs:
                self.recorder.motion_detected(frame)
            rec_tick+=1
            if rec_tick%30==0:
                saved=self.recorder.tick()
                if saved: self.clip_saved.emit(saved)

            out=frame.copy()
            if self.mode=="HEATMAP": self._draw_heat(out)
            if self.show_trails:     self._draw_trails(out)

            active=0
            for obj in self.objects:
                if now-obj.last_seen>0.15: continue
                active+=1
                if   self.mode=="BOX":     self._draw_box(out,obj)
                elif self.mode=="DOT":     self._draw_dot(out,obj)
                elif self.mode=="OUTLINE":
                    bc,bd=None,999
                    for b in blobs:
                        d=np.hypot(b["cx"]-obj.cx,b["cy"]-obj.cy)
                        if d<bd: bd,bc=d,b["contour"]
                    if bc is not None: self._draw_outline(out,obj,bc)
                elif self.mode=="HEATMAP": self._draw_dot(out,obj)

            self._draw_rec_indicator(out)
            self.frame_ready.emit(out)
            is_recording=1 if (self.recorder.active and self.recorder._writer) else 0
            self.stats_updated.emit(active,fps,self._total,is_recording)

        cap.release(); self.recorder.stop()
        self.log_event.emit("Camera closed")

    def stop(self):
        self.running=False; self.wait(2000)


# ── stat box ──────────────────────────────────────────────────────────────────
class StatBox(QWidget):
    def __init__(self, label, color=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{A['panel2']};border:1px solid {A['border']};border-radius:4px;")
        lay=QVBoxLayout(self); lay.setContentsMargins(10,6,10,6); lay.setSpacing(1)
        self.val=QLabel("0"); self.val.setObjectName("bigval")
        if color: self.val.setStyleSheet(f"color:{color};font-size:20px;font-weight:bold;")
        self.val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl=QLabel(label); lbl.setObjectName("biglbl")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.val); lay.addWidget(lbl)
    def set(self,v): self.val.setText(str(v))


# ── main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Tracker  v2")
        self.setMinimumSize(1200,720)
        self._last_frame: Optional[np.ndarray]=None
        self._screenshots=0
        self.thread=TrackerThread()
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.stats_updated.connect(self._on_stats)
        self.thread.log_event.connect(self._on_log)
        self.thread.alert_signal.connect(self._on_alert)
        self.thread.clip_saved.connect(self._on_clip_saved)
        self._build_ui()
        self.setStyleSheet(SS)
        # graph update timer
        self._graph_timer=QTimer()
        self._graph_timer.timeout.connect(self._update_graphs)
        self._graph_timer.start(500)
        self._fps_buf=0.0; self._obj_buf=0

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central); root.setContentsMargins(10,10,10,10); root.setSpacing(10)

        # ── left: video + stats ───────────────────────────────────────────────
        left=QVBoxLayout(); left.setSpacing(8)

        self.video_label=QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(700,500)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(f"background:#000;border:1px solid {A['border']};border-radius:4px;")
        self.video_label.setText("No camera feed\nPress  START  to begin")
        self.video_label.setFont(QFont("Courier New",13))

        # stat row
        stat_row=QHBoxLayout(); stat_row.setSpacing(6)
        self.stat_active =StatBox("ACTIVE",  A['accent'])
        self.stat_fps    =StatBox("FPS",     A['blue'])
        self.stat_total  =StatBox("TOTAL",   A['yellow'])
        self.stat_rec    =StatBox("RECORDING",A['red'])
        for s in (self.stat_active,self.stat_fps,self.stat_total,self.stat_rec):
            stat_row.addWidget(s)

        left.addWidget(self.video_label,1)
        left.addLayout(stat_row)

        # ── right: tabs ───────────────────────────────────────────────────────
        self.tabs=QTabWidget()
        self.tabs.setFixedWidth(310)
        self.tabs.addTab(self._tab_control(),  "CONTROL")
        self.tabs.addTab(self._tab_dashboard(),"DASHBOARD")
        self.tabs.addTab(self._tab_settings(), "SETTINGS")

        root.addLayout(left,1)
        root.addWidget(self.tabs)

        self.status=QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    def _tab_control(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setSpacing(8); lay.setContentsMargins(8,8,8,8)

        # title
        t=QLabel("MOTION TRACKER  v2")
        t.setFont(QFont("Courier New",12,QFont.Weight.Bold))
        t.setStyleSheet(f"color:{A['accent']};letter-spacing:3px;")
        lay.addWidget(t)
        d=QFrame(); d.setObjectName("div"); lay.addWidget(d)

        # source
        src=QGroupBox("SOURCE"); sl=QVBoxLayout(src); sl.setSpacing(6)
        cr=QHBoxLayout(); cr.addWidget(QLabel("Camera:"))
        self.cam_combo=QComboBox()
        for i in range(4): self.cam_combo.addItem(f"Camera {i}",i)
        cr.addWidget(self.cam_combo,1); sl.addLayout(cr)
        # IP / URL
        ur=QHBoxLayout(); ur.addWidget(QLabel("URL:"))
        self.url_edit=QLineEdit(); self.url_edit.setPlaceholderText("rtsp://... or http://...")
        ur.addWidget(self.url_edit,1); sl.addLayout(ur)
        fb=QPushButton("Open Video File…"); fb.clicked.connect(self._pick_file)
        self.file_lbl=QLabel("(none)"); self.file_lbl.setStyleSheet(f"color:{A['dim']};font-size:10px;")
        sl.addWidget(fb); sl.addWidget(self.file_lbl)
        lay.addWidget(src)

        # playback controls
        ctrl=QGroupBox("CONTROLS"); cl=QVBoxLayout(ctrl); cl.setSpacing(6)
        r1=QHBoxLayout()
        self.btn_start=QPushButton("▶ START"); self.btn_start.clicked.connect(self._start)
        self.btn_pause=QPushButton("⏸ PAUSE"); self.btn_pause.setCheckable(True)
        self.btn_pause.clicked.connect(self._pause); self.btn_pause.setEnabled(False)
        self.btn_stop=QPushButton("■ STOP"); self.btn_stop.setObjectName("red")
        self.btn_stop.clicked.connect(self._stop); self.btn_stop.setEnabled(False)
        r1.addWidget(self.btn_start); r1.addWidget(self.btn_pause); r1.addWidget(self.btn_stop)
        cl.addLayout(r1)
        r2=QHBoxLayout()
        self.btn_reset=QPushButton("↺ RESET"); self.btn_reset.clicked.connect(self._reset)
        self.btn_shot=QPushButton("📷 SAVE"); self.btn_shot.clicked.connect(self._screenshot)
        self.btn_shot.setObjectName("blue")
        r2.addWidget(self.btn_reset); r2.addWidget(self.btn_shot)
        cl.addLayout(r2)
        lay.addWidget(ctrl)

        # display
        disp=QGroupBox("DISPLAY"); dl=QVBoxLayout(disp); dl.setSpacing(6)
        mr=QHBoxLayout(); mr.addWidget(QLabel("Mode:"))
        self.mode_combo=QComboBox()
        for m in ["BOX","DOT","HEATMAP","OUTLINE"]: self.mode_combo.addItem(m)
        self.mode_combo.currentTextChanged.connect(lambda v: setattr(self.thread,'mode',v))
        mr.addWidget(self.mode_combo,1); dl.addLayout(mr)
        self.chk_trails=QCheckBox("Motion Trails"); self.chk_trails.setChecked(True)
        self.chk_trails.toggled.connect(lambda v: setattr(self.thread,'show_trails',v))
        self.chk_classify=QCheckBox("Object Classification"); self.chk_classify.setChecked(True)
        self.chk_classify.toggled.connect(lambda v: setattr(self.thread,'do_classify',v))
        dl.addWidget(self.chk_trails); dl.addWidget(self.chk_classify)
        lay.addWidget(disp)

        # recording
        rec=QGroupBox("RECORDING"); rl=QVBoxLayout(rec); rl.setSpacing(6)
        self.chk_record=QCheckBox("Auto-record motion clips")
        self.chk_record.toggled.connect(lambda v: setattr(self.thread.recorder,'active',v))
        rl.addWidget(self.chk_record)
        rdr=QHBoxLayout(); rdr.addWidget(QLabel("Save to:"))
        self.rec_dir=QLineEdit("recordings"); self.rec_dir.setReadOnly(True)
        rdr_btn=QPushButton("…"); rdr_btn.setFixedWidth(30)
        rdr_btn.clicked.connect(self._pick_rec_dir)
        rdr.addWidget(self.rec_dir,1); rdr.addWidget(rdr_btn)
        rl.addLayout(rdr)
        lay.addWidget(rec)

        # log
        log=QGroupBox("EVENT LOG"); ll=QVBoxLayout(log)
        self.log_box=QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setFixedHeight(100)
        ll.addWidget(self.log_box)
        lay.addWidget(log)
        lay.addStretch()
        return w

    def _tab_dashboard(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setSpacing(10); lay.setContentsMargins(8,8,8,8)
        lay.addWidget(QLabel("LIVE GRAPHS"))

        fps_grp=QGroupBox("FPS"); fl=QVBoxLayout(fps_grp)
        self.spark_fps=SparkLine(A['blue'],label="FPS")
        fl.addWidget(self.spark_fps); lay.addWidget(fps_grp)

        obj_grp=QGroupBox("ACTIVE OBJECTS"); ol=QVBoxLayout(obj_grp)
        self.spark_obj=SparkLine(A['accent'],label="Objects")
        ol.addWidget(self.spark_obj); lay.addWidget(obj_grp)

        # class breakdown
        cls_grp=QGroupBox("CLASS BREAKDOWN"); cl=QGridLayout(cls_grp)
        self.cls_labels={}
        for i,(name,col) in enumerate([("Person",A['blue']),("Vehicle",A['orange']),
                                        ("Animal",A['accent']),("Unknown",A['purple'])]):
            lbl=QLabel(name); lbl.setStyleSheet(f"color:{col};")
            val=QLabel("0");  val.setStyleSheet(f"color:{col};font-weight:bold;")
            cl.addWidget(lbl,i,0); cl.addWidget(val,i,1)
            self.cls_labels[name]=val
        lay.addWidget(cls_grp)

        # clips list
        clip_grp=QGroupBox("SAVED CLIPS"); kl=QVBoxLayout(clip_grp)
        self.clips_log=QTextEdit(); self.clips_log.setReadOnly(True); self.clips_log.setFixedHeight(80)
        kl.addWidget(self.clips_log); lay.addWidget(clip_grp)
        lay.addStretch()
        return w

    def _tab_settings(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setSpacing(10); lay.setContentsMargins(8,8,8,8)
        lay.addWidget(QLabel("TUNING"))

        tune=QGroupBox("DETECTION"); tl=QVBoxLayout(tune); tl.setSpacing(8)
        # sensitivity
        sr=QHBoxLayout(); sr.addWidget(QLabel("Sensitivity"))
        self.sens_slider=QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(5,100); self.sens_slider.setValue(50)
        self.sens_val=QLabel("50"); self.sens_val.setObjectName("val")
        self.sens_slider.valueChanged.connect(self._set_sens)
        sr.addWidget(self.sens_slider,1); sr.addWidget(self.sens_val)
        tl.addLayout(sr)
        # min area
        ar=QHBoxLayout(); ar.addWidget(QLabel("Min Area px²"))
        self.area_spin=QSpinBox(); self.area_spin.setRange(50,10000)
        self.area_spin.setSingleStep(100); self.area_spin.setValue(800)
        self.area_spin.valueChanged.connect(lambda v: setattr(self.thread,'min_area',v))
        ar.addWidget(self.area_spin)
        tl.addLayout(ar)
        # timeout
        tor=QHBoxLayout(); tor.addWidget(QLabel("Object Timeout"))
        self.timeout_sl=QSlider(Qt.Orientation.Horizontal)
        self.timeout_sl.setRange(1,30); self.timeout_sl.setValue(8)
        self.timeout_val=QLabel("0.8s"); self.timeout_val.setObjectName("val")
        self.timeout_sl.valueChanged.connect(self._set_timeout)
        tor.addWidget(self.timeout_sl,1); tor.addWidget(self.timeout_val)
        tl.addLayout(tor)
        lay.addWidget(tune)

        alert=QGroupBox("ALERTS"); al=QVBoxLayout(alert); al.setSpacing(6)
        self.chk_sound=QCheckBox("Sound beep on detection"); self.chk_sound.setChecked(True)
        self.chk_sound.toggled.connect(lambda v: setattr(self.thread.alert,'sound_enabled',v))
        self.chk_notif=QCheckBox("Windows notification"); self.chk_notif.setChecked(True)
        self.chk_notif.toggled.connect(lambda v: setattr(self.thread.alert,'notif_enabled',v))
        # cooldown
        cr=QHBoxLayout(); cr.addWidget(QLabel("Alert cooldown (s)"))
        self.cooldown_spin=QSpinBox(); self.cooldown_spin.setRange(1,60)
        self.cooldown_spin.setValue(3)
        self.cooldown_spin.valueChanged.connect(lambda v: setattr(self.thread.alert,'_cooldown',float(v)))
        cr.addWidget(self.cooldown_spin)
        al.addWidget(self.chk_sound); al.addWidget(self.chk_notif); al.addLayout(cr)
        lay.addWidget(alert)

        rec=QGroupBox("RECORDING"); rl=QVBoxLayout(rec); rl.setSpacing(6)
        rr=QHBoxLayout(); rr.addWidget(QLabel("Clip timeout (s)"))
        self.rectimeout_spin=QSpinBox(); self.rectimeout_spin.setRange(1,30)
        self.rectimeout_spin.setValue(3)
        self.rectimeout_spin.valueChanged.connect(lambda v: setattr(self.thread.recorder,'_timeout',float(v)))
        rr.addWidget(self.rectimeout_spin); rl.addLayout(rr)
        lay.addWidget(rec)
        lay.addStretch()
        return w

    # ── slots ─────────────────────────────────────────────────────────────────
    def _on_frame(self, frame):
        self._last_frame=frame
        h,w,ch=frame.shape
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=QImage(rgb.data,w,h,ch*w,QImage.Format.Format_RGB888)
        pix=QPixmap.fromImage(img)
        sz=self.video_label.size()
        self.video_label.setPixmap(pix.scaled(sz,Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))

    def _on_stats(self, active, fps, total, is_rec):
        self.stat_active.set(active)
        self.stat_fps.set(f"{fps:.1f}")
        self.stat_total.set(total)
        self.stat_rec.set("● REC" if is_rec else "—")
        self._fps_buf=fps; self._obj_buf=active
        # class breakdown
        counts={"Person":0,"Vehicle":0,"Animal":0,"Unknown":0}
        for obj in self.thread.objects:
            if obj.label in counts: counts[obj.label]+=1
        for name,lbl in self.cls_labels.items():
            lbl.setText(str(counts[name]))

    def _on_log(self, msg):
        ts=time.strftime("%H:%M:%S")
        self.log_box.append(f'<span style="color:#3a6a3a">[{ts}]</span> {msg}')
        sb=self.log_box.verticalScrollBar(); sb.setValue(sb.maximum())
        self.status.showMessage(msg,4000)

    def _on_alert(self, label):
        self.thread.alert.trigger(label)

    def _on_clip_saved(self, path):
        self.clips_log.append(f'<span style="color:{A["accent"]}">{time.strftime("%H:%M:%S")}</span> {os.path.basename(path)}')
        self._on_log(f"Clip saved → {path}")

    def _update_graphs(self):
        self.spark_fps.push(self._fps_buf)
        self.spark_obj.push(float(self._obj_buf))

    def _start(self):
        if self.thread.isRunning(): return
        url=self.url_edit.text().strip()
        if url:
            self.thread.source=url
        elif hasattr(self,'_file_src'):
            self.thread.source=self._file_src
        else:
            self.thread.source=self.cam_combo.currentData()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.thread.running=True; self.thread.paused=False
        self.thread.start(); self.status.showMessage("Tracking…")

    def _pause(self, checked):
        self.thread.paused=checked
        self.btn_pause.setText("▶ RESUME" if checked else "⏸ PAUSE")

    def _stop(self):
        self.thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False); self.btn_pause.setChecked(False)
        self.btn_pause.setText("⏸ PAUSE"); self.btn_stop.setEnabled(False)
        self.video_label.setText("Stopped\nPress START to begin again")

    def _reset(self): self.thread.reset()

    def _set_sens(self, v):
        self.sens_val.setText(str(v)); self.thread.sensitivity=v
        thresh=max(8,int(80-v*0.72))
        if self.thread._bg: self.thread._bg.setVarThreshold(thresh)

    def _set_timeout(self, v):
        t=v/10.0; self.timeout_val.setText(f"{t:.1f}s"); self.thread._timeout=t

    def _pick_file(self):
        path,_=QFileDialog.getOpenFileName(self,"Open Video","",
            "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*)")
        if path:
            self._file_src=path; self.thread.source=path
            self.file_lbl.setText(path.split("/")[-1].split("\\")[-1])

    def _pick_rec_dir(self):
        d=QFileDialog.getExistingDirectory(self,"Select Recording Folder")
        if d: self.rec_dir.setText(d); self.thread.recorder.save_dir=d

    def _screenshot(self):
        if self._last_frame is None: self.status.showMessage("No frame"); return
        self._screenshots+=1
        fname=f"shot_{self._screenshots:03d}.png"
        cv2.imwrite(fname,self._last_frame)
        self.status.showMessage(f"Saved {fname}")
        self._on_log(f"Screenshot → {fname}")

    def closeEvent(self, e):
        self.thread.stop(); e.accept()


if __name__=="__main__":
    app=QApplication(sys.argv)
    app.setApplicationName("Motion Tracker v2")
    app.setStyle("Fusion")
    win=MainWindow(); win.show()
    sys.exit(app.exec())
