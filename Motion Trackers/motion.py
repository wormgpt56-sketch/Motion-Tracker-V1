#!/usr/bin/env python3
"""
MOTION TRACKER — Standalone OpenCV app
Detects and marks moving objects in real time using MOG2 background subtraction.

Requirements:
    pip install opencv-python numpy

Usage:
    python motion_tracker.py              # webcam (default)
    python motion_tracker.py 0            # specific camera index
    python motion_tracker.py video.mp4    # video file

Controls:
    SPACE   — pause / resume
    S       — cycle display style  (box / dot / heatmap / combined)
    T       — toggle trails
    H       — toggle HUD
    R       — reset background model + trails
    +/-     — increase / decrease sensitivity
    Q / ESC — quit
"""

import cv2
import numpy as np
import sys
import time
from collections import deque

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WINDOW_NAME   = "MOTION TRACKER"
MIN_AREA      = 800        # px² — blobs smaller than this are ignored
MAX_OBJECTS   = 30         # max simultaneous tracked objects
TRAIL_LEN     = 60         # frames of trail history per object
MATCH_DIST    = 120        # px — max centroid distance to match existing object
VANISH_FRAMES = 15         # frames before an unseen object is dropped
SENSITIVITY   = 40         # initial MOG2 varThreshold (lower = more sensitive)
DISPLAY_MODES = ["BOX", "DOT", "HEATMAP", "COMBINED"]

# Palette (BGR)
COL_GREEN  = (0, 255, 136)
COL_ORANGE = (0, 150, 255)
COL_RED    = (50, 50, 255)
COL_WHITE  = (220, 220, 220)
COL_DIM    = (80, 80, 80)
COL_BG     = (13, 13, 13)


# ─────────────────────────────────────────────
# TRACKED OBJECT
# ─────────────────────────────────────────────
class TrackedObject:
    _counter = 0

    def __init__(self, cx, cy, bbox):
        TrackedObject._counter += 1
        self.id          = TrackedObject._counter
        self.cx, self.cy = cx, cy
        self.bbox        = bbox          # (x, y, w, h)
        self.trail       = deque([(cx, cy)], maxlen=TRAIL_LEN)
        self.frames_seen = 1
        self.frames_lost = 0
        self.is_new      = True
        self.area        = bbox[2] * bbox[3]
        self.speed       = 0.0

    def update(self, cx, cy, bbox):
        self.speed       = np.hypot(cx - self.cx, cy - self.cy)
        self.cx, self.cy = cx, cy
        self.bbox        = bbox
        self.area        = bbox[2] * bbox[3]
        self.trail.append((cx, cy))
        self.frames_seen += 1
        self.frames_lost  = 0
        self.is_new       = self.frames_seen < 4


# ─────────────────────────────────────────────
# TRACKER
# ─────────────────────────────────────────────
class MotionTracker:
    def __init__(self, source=0, sensitivity=SENSITIVITY):
        self.cap        = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # MOG2 background subtractor
        self.sensitivity  = sensitivity
        self.subtractor   = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=self.sensitivity, detectShadows=True
        )

        # Morphological kernel for noise removal
        self.kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.objects      = []
        self.display_mode = 0
        self.show_trails  = True
        self.show_hud     = True
        self.paused       = False

        self.heatmap      = np.zeros((self.H, self.W), dtype=np.float32)
        self.fps_times    = deque(maxlen=30)
        self.frame_count  = 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(self.W, 1280), min(self.H, 720))

    # ── detection ────────────────────────────
    def detect_blobs(self, frame):
        mask = self.subtractor.apply(frame)
        # Remove shadows (gray, value 127) — keep only hard foreground (255)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # Clean up noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            blobs.append((cx, cy, (x, y, w, h), area, cnt))
        return blobs, mask

    # ── tracking ─────────────────────────────
    def match_objects(self, blobs):
        matched_ids = set()
        for cx, cy, bbox, area, cnt in blobs:
            best, best_dist = None, MATCH_DIST
            for obj in self.objects:
                if obj.id in matched_ids:
                    continue
                d = np.hypot(cx - obj.cx, cy - obj.cy)
                if d < best_dist:
                    best_dist, best = d, obj
            if best:
                best.update(cx, cy, bbox)
                matched_ids.add(best.id)
            elif len(self.objects) < MAX_OBJECTS:
                self.objects.append(TrackedObject(cx, cy, bbox))

        for obj in self.objects:
            if obj.id not in matched_ids:
                obj.frames_lost += 1

        self.objects = [o for o in self.objects if o.frames_lost <= VANISH_FRAMES]

    # ── heatmap ──────────────────────────────
    def update_heatmap(self, mask):
        self.heatmap += (mask.astype(np.float32) / 255.0) * 0.4
        self.heatmap *= 0.97  # decay

    # ── drawing ──────────────────────────────
    def draw_box(self, vis, obj):
        x, y, w, h = obj.bbox
        pad = 10
        bx, by, bw, bh = x - pad, y - pad, w + pad*2, h + pad*2
        alpha = 1.0 if obj.frames_lost == 0 else max(0, 1 - obj.frames_lost / VANISH_FRAMES)
        col   = COL_ORANGE if obj.is_new else COL_GREEN

        # Main bounding rect (dim)
        overlay = vis.copy()
        cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), col, 1)
        cv2.addWeighted(overlay, 0.5 * alpha, vis, 1 - 0.5 * alpha, 0, vis)

        # Corner brackets (bright)
        c = 16
        pts = [
            [(bx, by+c), (bx, by), (bx+c, by)],
            [(bx+bw-c, by), (bx+bw, by), (bx+bw, by+c)],
            [(bx+bw, by+bh-c), (bx+bw, by+bh), (bx+bw-c, by+bh)],
            [(bx+c, by+bh), (bx, by+bh), (bx, by+bh-c)],
        ]
        for tri in pts:
            for i in range(len(tri)-1):
                cv2.line(vis, tri[i], tri[i+1], col, 2, cv2.LINE_AA)

        # Label
        label  = f"#{obj.id}  {'NEW' if obj.is_new else 'MOVE'}  {int(obj.speed)}px/f"
        lx, ly = bx + 4, by - 8
        cv2.rectangle(vis, (lx - 2, ly - 14), (lx + len(label)*7 + 2, ly + 2), (0,0,0), -1)
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_PLAIN, 0.9,
                    COL_ORANGE if obj.is_new else COL_GREEN, 1, cv2.LINE_AA)

        # Scan line
        scan_y = by + int((time.time() * 120) % bh)
        cv2.line(vis, (bx, scan_y), (bx+bw, scan_y), (*col[:2], max(0, col[2]-100)), 1)

    def draw_dot(self, vis, obj):
        r = max(6, int(np.sqrt(obj.area) * 0.5))
        col = COL_ORANGE if obj.is_new else COL_GREEN
        cv2.circle(vis, (obj.cx, obj.cy), 5, col, -1, cv2.LINE_AA)
        cv2.circle(vis, (obj.cx, obj.cy), r, col, 1, cv2.LINE_AA)
        cv2.putText(vis, f"#{obj.id}", (obj.cx + 8, obj.cy - 8),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, col, 1, cv2.LINE_AA)

    def draw_trails(self, vis, obj):
        trail = list(obj.trail)
        if len(trail) < 2:
            return
        for i in range(1, len(trail)):
            t = i / len(trail)
            col = tuple(int(c * t) for c in COL_GREEN)
            cv2.line(vis, trail[i-1], trail[i], col, 1, cv2.LINE_AA)

    def draw_heatmap_overlay(self, vis):
        norm = np.clip(self.heatmap, 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        mask2d  = (norm > 0.05).astype(np.float32)
        mask3d  = np.stack([mask2d]*3, axis=-1)
        vis[:]  = (vis * (1 - mask3d * 0.55) + colored * mask3d * 0.55).astype(np.uint8)

    def draw_hud(self, vis, fps):
        mode_str = DISPLAY_MODES[self.display_mode]
        n_active = sum(1 for o in self.objects if o.frames_lost == 0)
        lines = [
            f"FPS: {fps:.0f}",
            f"OBJECTS: {n_active}",
            f"MODE: {mode_str}",
            f"TRAILS: {'ON' if self.show_trails else 'OFF'}",
            f"SENS: {self.sensitivity}",
            f"FRAME: {self.frame_count}",
        ]
        pad = 10
        for i, line in enumerate(lines):
            y = pad + 16 + i * 18
            cv2.putText(vis, line, (pad, y), cv2.FONT_HERSHEY_PLAIN, 1.0, COL_DIM, 1, cv2.LINE_AA)
            cv2.putText(vis, line, (pad, y), cv2.FONT_HERSHEY_PLAIN, 1.0, COL_GREEN, 1, cv2.LINE_AA)

        # Controls hint at bottom
        hints = "SPACE=pause  S=style  T=trails  H=hud  R=reset  +/-=sens  Q=quit"
        cv2.putText(vis, hints, (pad, self.H - 10),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, COL_DIM, 1, cv2.LINE_AA)

    # ── main render ──────────────────────────
    def render(self, frame, mask):
        mode = DISPLAY_MODES[self.display_mode]
        vis  = frame.copy()

        if mode == "HEATMAP":
            self.draw_heatmap_overlay(vis)
            for obj in self.objects:
                if self.show_trails: self.draw_trails(vis, obj)
                self.draw_dot(vis, obj)

        elif mode == "DOT":
            for obj in self.objects:
                if self.show_trails: self.draw_trails(vis, obj)
                self.draw_dot(vis, obj)

        elif mode == "COMBINED":
            self.draw_heatmap_overlay(vis)
            for obj in self.objects:
                if self.show_trails: self.draw_trails(vis, obj)
                self.draw_box(vis, obj)

        else:  # BOX (default)
            for obj in self.objects:
                if self.show_trails: self.draw_trails(vis, obj)
                self.draw_box(vis, obj)

        # Mask preview (small, bottom-right)
        mh, mw = self.H // 5, self.W // 5
        small_mask = cv2.resize(mask, (mw, mh))
        small_mask_bgr = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        vis[self.H-mh:self.H, self.W-mw:self.W] = small_mask_bgr
        cv2.rectangle(vis, (self.W-mw, self.H-mh), (self.W-1, self.H-1), COL_DIM, 1)
        cv2.putText(vis, "MASK", (self.W-mw+4, self.H-mh+14),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, COL_DIM, 1)

        return vis

    # ── main loop ────────────────────────────
    def run(self):
        print(f"\n{'='*50}")
        print("  MOTION TRACKER  |  OpenCV", cv2.__version__)
        print(f"  Source: {self.W}x{self.H}")
        print(f"{'='*50}")
        print("  SPACE  pause/resume")
        print("  S      cycle display mode")
        print("  T      toggle trails")
        print("  H      toggle HUD")
        print("  R      reset tracker")
        print("  +/-    sensitivity")
        print("  Q/ESC  quit")
        print(f"{'='*50}\n")

        t0 = time.time()

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Stream ended.")
                    break

                self.frame_count += 1
                self.fps_times.append(time.time())
                fps = len(self.fps_times) / max(1e-9, self.fps_times[-1] - self.fps_times[0]) if len(self.fps_times) > 1 else 0

                blobs, mask = self.detect_blobs(frame)
                self.match_objects(blobs)
                self.update_heatmap(mask)
                vis = self.render(frame, mask)
                if self.show_hud:
                    self.draw_hud(vis, fps)
            else:
                # When paused just redisplay last frame
                time.sleep(0.03)

            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print("PAUSED" if self.paused else "RESUMED")
            elif key in (ord('s'), ord('S')):
                self.display_mode = (self.display_mode + 1) % len(DISPLAY_MODES)
                print(f"Mode: {DISPLAY_MODES[self.display_mode]}")
            elif key in (ord('t'), ord('T')):
                self.show_trails = not self.show_trails
                print(f"Trails: {'ON' if self.show_trails else 'OFF'}")
            elif key in (ord('h'), ord('H')):
                self.show_hud = not self.show_hud
            elif key in (ord('r'), ord('R')):
                self.objects   = []
                self.heatmap   = np.zeros((self.H, self.W), dtype=np.float32)
                self.subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=300, varThreshold=self.sensitivity, detectShadows=True
                )
                TrackedObject._counter = 0
                print("Tracker reset.")
            elif key in (ord('+'), ord('='), 82):  # + or up arrow
                self.sensitivity = min(100, self.sensitivity + 5)
                self.subtractor.setVarThreshold(self.sensitivity)
                print(f"Sensitivity: {self.sensitivity}")
            elif key in (ord('-'), 84):  # - or down arrow
                self.sensitivity = max(5, self.sensitivity - 5)
                self.subtractor.setVarThreshold(self.sensitivity)
                print(f"Sensitivity: {self.sensitivity}")

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nTracked {TrackedObject._counter} objects over {self.frame_count} frames.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # keep as string (file path)

    try:
        tracker = MotionTracker(source=source)
        tracker.run()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)