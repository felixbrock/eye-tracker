#!/usr/bin/env python3
"""
Eye Gaze Tracker
================
Tracks eye movements via webcam using MediaPipe Face Mesh iris detection
and projects the estimated gaze point as a crosshair overlay on screen.

Webcam assumed centered on top of monitor (Logitech).

Controls (in the debug window):
    d       Toggle debug webcam view
    +/-     Increase/decrease smoothing
    q/Esc   Quit

Dependencies (managed by uv):
    opencv-python, mediapipe, numpy, screeninfo

i3 users: add these rules to your i3 config for best experience:
    for_window [title="^gaze$"] floating enable, border none, sticky enable
    for_window [title="^Eye Tracker Debug$"] floating enable
"""

import json
import os
import time
import sys
import tempfile
import shutil
import subprocess
from collections import deque

# OpenCV HighGUI on some Linux builds uses Qt and expects a font dir.
# Point Qt to system fonts early to avoid noisy QFontDatabase warnings.
if "QT_QPA_FONTDIR" not in os.environ and "OPENCV_QT_FONTDIR" not in os.environ:
    for font_dir in (
        "/usr/share/fonts/noto",
        "/usr/share/fonts/liberation",
        "/usr/share/fonts/gnu-free",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ):
        if os.path.isdir(font_dir):
            os.environ["QT_QPA_FONTDIR"] = font_dir
            os.environ["OPENCV_QT_FONTDIR"] = font_dir
            break

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from screeninfo import get_monitors
try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:
    Image = ImageDraw = ImageFont = ImageTk = None

# ─── Configuration ────────────────────────────────────────────────────────────

WEBCAM_INDEX = 0
WEBCAM_W = 640
WEBCAM_H = 480
SMOOTHING_FRAMES = 2       # number of frames to average for smoothing
SENSITIVITY_X = 1.3        # horizontal multiplier
SENSITIVITY_Y = 1.6        # vertical multiplier (usually needs to be higher)
CURVE_EXP = 1.0            # keep center-to-edge mapping linear to reduce center lock-in
DOT_SIZE = 48              # overlay marker window size in px
SETTINGS_FILE = os.path.expanduser("~/.config/gaze_settings.json")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
# Head-pose compensation coefficients (empirical).
# These offsets are applied to iris ratios after head-anchor capture.
HEAD_COMP_X_FROM_DX = 0.11
HEAD_COMP_Y_FROM_DY = 0.075
HEAD_COMP_Y_FROM_SCALE = 0.010
HEAD_COMP_MAX_X = 0.030
HEAD_COMP_MAX_Y = 0.035
UFO_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ufo.png")
RANGE_WINDOW = 220
RANGE_MIN_SAMPLES = 30
TARGET_IRIS_SPAN_X = 0.11
TARGET_IRIS_SPAN_Y = 0.12
MAX_AUTO_RANGE_BOOST = 1.45
MAX_CURSOR_STEP_RATIO = 0.20
MIN_DYNAMIC_SPAN_X = 0.120
MIN_DYNAMIC_SPAN_Y = 0.140
MAX_DYNAMIC_SPAN_X = 0.42
MAX_DYNAMIC_SPAN_Y = 0.46
AUTO_BOUND_BLEND = 0.20
AUTO_CENTER_BLEND = 0.10
MAX_DYNAMIC_CENTER_SHIFT_X = 0.02
MAX_DYNAMIC_CENTER_SHIFT_Y = 0.02
DYNAMIC_CENTER_MIN_SAMPLES = 60
DYNAMIC_CENTER_COVER_MARGIN_X = 0.05
DYNAMIC_CENTER_COVER_MARGIN_Y = 0.05
HEAD_ANCHOR_ALPHA = 0.035
HEAD_ANCHOR_WARMUP_FRAMES = 12
HEAD_ANCHOR_WARMUP_ALPHA = 0.30
HEAD_ANCHOR_EDGE_ADAPT_FLOOR = 0.45
RUNTIME_BIAS_UPDATE = 0.008
RUNTIME_BIAS_MAX = 0.06
RUNTIME_BIAS_DECAY = 0.97
RUNTIME_BIAS_MIN_SAMPLES = 90
RUNTIME_BIAS_MIN_SPAN_X = 0.12
RUNTIME_BIAS_MIN_SPAN_Y = 0.12
RUNTIME_BIAS_CENTER_MARGIN_X = 0.08
RUNTIME_BIAS_CENTER_MARGIN_Y = 0.08
MIN_DYNAMIC_CENTER_SPAN_X = 0.100
MIN_DYNAMIC_CENTER_SPAN_Y = 0.100
OUTLIER_MEDIAN_GUARD = 0.18
OUTLIER_EXTREME_LOW = 0.06
OUTLIER_EXTREME_HIGH = 0.94
FIXATION_DEADZONE_H = 0.0012
FIXATION_DEADZONE_V = 0.0015


# ─── MediaPipe Landmark Indices ───────────────────────────────────────────────
# FaceLandmarker produces 478 landmarks (468 face + 10 iris)

# Right eye (subject's right = camera's left) — matches RIGHT_EYE connections
R_OUTER  = 33    # outer corner
R_INNER  = 133   # inner corner
R_TOP    = 159   # upper lid
R_BOTTOM = 145   # lower lid
R_IRIS_RING = (469, 470, 471, 472)  # iris ring landmarks

# Left eye (subject's left = camera's right) — matches LEFT_EYE connections
L_INNER  = 362   # inner corner
L_OUTER  = 263   # outer corner
L_TOP    = 386   # upper lid
L_BOTTOM = 374   # lower lid
L_IRIS_RING = (474, 475, 476, 477)  # iris ring landmarks


class EyeTracker:
    """Extracts normalized iris position using MediaPipe FaceLandmarker Tasks API."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model not found at {MODEL_PATH}")
            print("Download it with:")
            print(f'  curl -L -o "{MODEL_PATH}" '
                  '"https://storage.googleapis.com/mediapipe-models/'
                  'face_landmarker/face_landmarker/float16/latest/face_landmarker.task"')
            sys.exit(1)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._frame_ts = 0
        self.head_anchor = None
        self._head_anchor_warmup_left = HEAD_ANCHOR_WARMUP_FRAMES

    def reset_head_anchor(self):
        self.head_anchor = None
        self._head_anchor_warmup_left = HEAD_ANCHOR_WARMUP_FRAMES

    def _head_features(self, lm):
        """Compact head-position features from stable landmarks."""
        eye_mid_x = (lm[33].x + lm[263].x) * 0.5
        eye_mid_y = (lm[33].y + lm[263].y + lm[133].y + lm[362].y) * 0.25
        inter_eye = np.hypot(lm[263].x - lm[33].x, lm[263].y - lm[33].y)
        return eye_mid_x, eye_mid_y, inter_eye

    def process(self, frame, apply_head_comp=True, return_meta=False):
        """Return ((h_ratio, v_ratio), landmarks) or None.

        h_ratio: 0 = iris at outer corner, 1 = iris at inner corner
        v_ratio: 0 = iris at top lid, 1 = iris at bottom lid
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33  # ~30fps in milliseconds
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]  # list of NormalizedLandmark

        def iris_center(ring_indices):
            xs = [lm[i].x for i in ring_indices]
            ys = [lm[i].y for i in ring_indices]
            return sum(xs) / len(xs), sum(ys) / len(ys)

        def ratio(iris_ring, outer, inner, top, bottom):
            ix, iy = iris_center(iris_ring)
            # Normalize horizontally in camera coordinates (left->right) for
            # both eyes so averaging does not cancel opposite inner/outer
            # movement directions.
            left_x = min(lm[outer].x, lm[inner].x)
            right_x = max(lm[outer].x, lm[inner].x)
            h = (ix - left_x) / (right_x - left_x + 1e-7)

            top_y = min(lm[top].y, lm[bottom].y)
            bottom_y = max(lm[top].y, lm[bottom].y)
            v = (iy - top_y) / (bottom_y - top_y + 1e-7)
            return h, v

        rh, rv = ratio(R_IRIS_RING, R_OUTER, R_INNER, R_TOP, R_BOTTOM)
        lh, lv = ratio(L_IRIS_RING, L_OUTER, L_INNER, L_TOP, L_BOTTOM)

        h = (lh + rh) / 2
        v = (lv + rv) / 2
        raw_h = h
        raw_v = v
        hx, hy, hs = self._head_features(lm)

        if self.head_anchor is None:
            self.head_anchor = (hx, hy, hs)

        if self._head_anchor_warmup_left > 0:
            ax, ay, ascale = self.head_anchor
            wa = HEAD_ANCHOR_WARMUP_ALPHA
            self.head_anchor = (
                (1.0 - wa) * ax + wa * hx,
                (1.0 - wa) * ay + wa * hy,
                (1.0 - wa) * ascale + wa * hs,
            )
            self._head_anchor_warmup_left -= 1

        ax, ay, ascale = self.head_anchor
        edge_blend = 0.0
        head_motion = 0.0
        if apply_head_comp and self._head_anchor_warmup_left <= 0:
            # Clamp compensation inputs to prevent short head-motion spikes from
            # over-correcting gaze and collapsing usable vertical range.
            dx_norm = float(np.clip((hx - ax) / (ascale + 1e-7), -0.30, 0.30))
            dy_norm = float(np.clip((hy - ay) / (ascale + 1e-7), -0.30, 0.30))
            dscale = float(np.clip(np.log((hs + 1e-7) / (ascale + 1e-7)), -0.25, 0.25))
            x_comp = float(np.clip(HEAD_COMP_X_FROM_DX * dx_norm, -HEAD_COMP_MAX_X, HEAD_COMP_MAX_X))
            y_comp = HEAD_COMP_Y_FROM_DY * dy_norm + HEAD_COMP_Y_FROM_SCALE * dscale
            y_comp = float(np.clip(y_comp, -HEAD_COMP_MAX_Y, HEAD_COMP_MAX_Y))
            # Preserve prior head-motion correction near center gaze, but
            # reduce cancellation when user intentionally looks toward edges.
            gaze_excursion = max(abs(raw_h - 0.5), abs(raw_v - 0.5))
            edge_blend = float(np.clip((gaze_excursion - 0.06) / 0.24, 0.0, 1.0))
            comp_scale = 1.0 - 0.45 * edge_blend
            x_comp *= comp_scale
            y_comp *= comp_scale
            h -= x_comp
            v -= y_comp
            head_motion = max(abs(dx_norm), abs(dy_norm), abs(dscale))

        # Keep anchor adaptive to slow posture changes while preserving
        # compensation against short-term head motion.
        if self._head_anchor_warmup_left <= 0:
            # Do not let prolonged off-center gaze rapidly absorb into the
            # anchor; that can decay compensation and cause drift away.
            adapt_scale = 1.0 - (1.0 - HEAD_ANCHOR_EDGE_ADAPT_FLOOR) * edge_blend
            if head_motion > 0.08:
                adapt_scale = max(adapt_scale, 0.70)
            a = HEAD_ANCHOR_ALPHA * adapt_scale
            self.head_anchor = (
                (1.0 - a) * ax + a * hx,
                (1.0 - a) * ay + a * hy,
                (1.0 - a) * ascale + a * hs,
            )
        h = float(np.clip(h, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))

        if return_meta:
            return (h, v), lm, (hx, hy, hs)
        return (h, v), lm

    def close(self):
        self.landmarker.close()


class GazeMapper:
    """Maps iris ratio → screen coordinates with optional persisted settings."""

    def __init__(
        self,
        screen_w,
        screen_h,
        smoothing=SMOOTHING_FRAMES,
        load_saved_settings=True,
        calibration_mode=False,
    ):
        self.sw = screen_w
        self.sh = screen_h
        self.calibration_mode = bool(calibration_mode)
        self.history = deque(maxlen=smoothing)
        self.sensitivity_x = SENSITIVITY_X
        self.sensitivity_y = SENSITIVITY_Y
        self.curve_exp = CURVE_EXP

        # Default mapping bounds — tighter range since actual iris movement
        # within the eye is very small (typically ~0.45-0.55 across screen).
        # Optional persisted settings may refine these.
        # Slightly recenter horizontal defaults to reduce persistent rightward
        # bias observed in repeated calibration runs while preserving span.
        self.x_min = 0.38
        self.x_max = 0.60
        self.y_min = 0.40
        self.y_max = 0.60
        self.flip_x = True
        self.flip_y = False
        self.x_offset = 0.0
        self.x_gain = 1.0
        self.y_offset = 0.0
        self.y_gain = 1.0
        self._raw_h_hist = deque(maxlen=3)
        self._raw_v_hist = deque(maxlen=3)
        self._range_h_hist = deque(maxlen=RANGE_WINDOW)
        self._range_v_hist = deque(maxlen=RANGE_WINDOW)
        self._norm_xy_hist = deque(maxlen=120)
        self._runtime_x_bias = 0.0
        self._runtime_y_bias = 0.0
        self._norm_x_lp = None
        self._norm_y_lp = None
        self._last_h = None
        self._last_v = None
        if load_saved_settings:
            self._load()

    def set_smoothing(self, n):
        old = list(self.history)
        self.history = deque(old[-n:], maxlen=max(1, n))

    def set_sensitivity(self, s):
        s = max(0.5, min(5.0, s))
        self.sensitivity_x = s
        self.sensitivity_y = s

    def set_vertical_sensitivity(self, s):
        self.sensitivity_y = max(0.5, min(8.0, s))

    def set_y_offset(self, off):
        self.y_offset = max(-0.45, min(0.45, off))

    def set_y_gain(self, gain):
        self.y_gain = max(0.6, min(3.0, gain))

    def toggle_flip_y(self):
        self.flip_y = not self.flip_y

    def set_x_offset(self, off):
        self.x_offset = max(-0.45, min(0.45, off))

    def set_x_gain(self, gain):
        self.x_gain = max(0.6, min(3.0, gain))

    def toggle_flip_x(self):
        self.flip_x = not self.flip_x

    def _ensure_min_span(self):
        """Prevent collapsed mapping ranges that pin cursor to one region."""
        min_x_span = 0.18
        min_y_span = 0.16

        x_span = self.x_max - self.x_min
        if x_span < min_x_span:
            mid = (self.x_min + self.x_max) / 2.0
            half = min_x_span / 2.0
            self.x_min = mid - half
            self.x_max = mid + half

        y_span = self.y_max - self.y_min
        if y_span < min_y_span:
            mid = (self.y_min + self.y_max) / 2.0
            half = min_y_span / 2.0
            self.y_min = mid - half
            self.y_max = mid + half

    def _filter_measurement(self, h, v):
        if not (np.isfinite(h) and np.isfinite(v)):
            return None
        raw_h = float(h)
        raw_v = float(v)
        h = raw_h
        v = raw_v

        # Ignore blink/landmark failures that briefly collapse an axis to
        # extreme values and can otherwise drag the cursor far off-target.
        if self._raw_h_hist:
            med_h = float(np.median(self._raw_h_hist))
            med_v = float(np.median(self._raw_v_hist))
            bad_h = ((raw_h < OUTLIER_EXTREME_LOW or raw_h > OUTLIER_EXTREME_HIGH) and abs(raw_h - med_h) > OUTLIER_MEDIAN_GUARD)
            bad_v = ((raw_v < OUTLIER_EXTREME_LOW or raw_v > OUTLIER_EXTREME_HIGH) and abs(raw_v - med_v) > OUTLIER_MEDIAN_GUARD)
            if bad_h:
                h = med_h
            if bad_v:
                v = med_v

        if self._last_h is not None:
            max_step_h = 0.100
            max_step_v = 0.105
            h = self._last_h + np.clip(h - self._last_h, -max_step_h, max_step_h)
            v = self._last_v + np.clip(v - self._last_v, -max_step_v, max_step_v)
        if self._raw_h_hist:
            med_h = float(np.median(self._raw_h_hist))
            med_v = float(np.median(self._raw_v_hist))
            mad_h = float(np.median(np.abs(np.asarray(self._raw_h_hist) - med_h))) + 1e-4
            mad_v = float(np.median(np.abs(np.asarray(self._raw_v_hist) - med_v))) + 1e-4
            lim_h = 0.028 + 4.0 * mad_h
            lim_v = 0.030 + 4.0 * mad_v
            h = med_h + float(np.clip(h - med_h, -lim_h, lim_h))
            v = med_v + float(np.clip(v - med_v, -lim_v, lim_v))
        h = float(np.clip(h, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        self._raw_h_hist.append(h)
        self._raw_v_hist.append(v)
        h = float(np.median(self._raw_h_hist))
        v = float(np.median(self._raw_v_hist))
        if self._last_h is not None:
            if abs(h - self._last_h) < FIXATION_DEADZONE_H:
                h = self._last_h
            if abs(v - self._last_v) < FIXATION_DEADZONE_V:
                v = self._last_v
        self._last_h = h
        self._last_v = v
        return h, v

    def clear_runtime_state(self, reset_bias=False):
        self.history.clear()
        self._raw_h_hist.clear()
        self._raw_v_hist.clear()
        self._range_h_hist.clear()
        self._range_v_hist.clear()
        self._norm_xy_hist.clear()
        if reset_bias:
            self._runtime_x_bias = 0.0
            self._runtime_y_bias = 0.0
        self._norm_x_lp = None
        self._norm_y_lp = None
        self._last_h = None
        self._last_v = None

    def _auto_range_boost(self):
        if len(self._range_h_hist) < RANGE_MIN_SAMPLES:
            return 1.0, 1.0
        hs = np.asarray(self._range_h_hist, dtype=float)
        vs = np.asarray(self._range_v_hist, dtype=float)
        h_lo, h_hi = np.percentile(hs, [10, 90])
        v_lo, v_hi = np.percentile(vs, [10, 90])
        h_span = max(1e-4, float(h_hi - h_lo))
        v_span = max(1e-4, float(v_hi - v_lo))
        # Only expand low-amplitude ranges here; avoid automatic shrink that can
        # make edge reach worse on users with already narrow iris motion.
        bx = float(np.clip(TARGET_IRIS_SPAN_X / h_span, 1.0, MAX_AUTO_RANGE_BOOST))
        by = float(np.clip(TARGET_IRIS_SPAN_Y / v_span, 1.0, MAX_AUTO_RANGE_BOOST))
        return bx, by

    def _effective_bounds(self):
        x_min = self.x_min
        x_max = self.x_max
        y_min = self.y_min
        y_max = self.y_max
        if len(self._range_h_hist) < max(6, RANGE_MIN_SAMPLES // 2):
            return x_min, x_max, y_min, y_max

        hs = np.asarray(self._range_h_hist, dtype=float)
        vs = np.asarray(self._range_v_hist, dtype=float)
        h_lo, h_hi = np.percentile(hs, [5, 95])
        v_lo, v_hi = np.percentile(vs, [5, 95])

        h_obs_span = float(h_hi - h_lo)
        v_obs_span = float(v_hi - v_lo)
        hx_span = float(np.clip(h_obs_span * 1.35, MIN_DYNAMIC_SPAN_X, MAX_DYNAMIC_SPAN_X))
        vy_span = float(np.clip(v_obs_span * 1.45, MIN_DYNAMIC_SPAN_Y, MAX_DYNAMIC_SPAN_Y))
        base_x_span = max(1e-4, self.x_max - self.x_min)
        base_y_span = max(1e-4, self.y_max - self.y_min)
        base_x_mid = (self.x_min + self.x_max) * 0.5
        base_y_mid = (self.y_min + self.y_max) * 0.5

        a = AUTO_BOUND_BLEND
        x_span = (1.0 - a) * base_x_span + a * hx_span
        y_span = (1.0 - a) * base_y_span + a * vy_span
        # Keep mapping from becoming tighter than base calibration bounds.
        # Tightening amplifies tiny iris jitter into large cursor drift.
        x_span = float(np.clip(x_span, base_x_span, MAX_DYNAMIC_SPAN_X))
        y_span = float(np.clip(y_span, base_y_span, MAX_DYNAMIC_SPAN_Y))

        x_mid = base_x_mid
        y_mid = base_y_mid
        center_ready = len(self._range_h_hist) >= DYNAMIC_CENTER_MIN_SAMPLES
        if center_ready and h_obs_span >= MIN_DYNAMIC_CENTER_SPAN_X:
            # Let center follow sustained user-specific neutral gaze with a
            # hard safety clamp so short glances do not drag the mapping.
            covered_both_x = h_lo <= (base_x_mid - DYNAMIC_CENTER_COVER_MARGIN_X) and h_hi >= (
                base_x_mid + DYNAMIC_CENTER_COVER_MARGIN_X
            )
            if covered_both_x:
                h_mid = float((h_lo + h_hi) * 0.5)
                x_shift = float(
                    np.clip(h_mid - base_x_mid, -MAX_DYNAMIC_CENTER_SHIFT_X, MAX_DYNAMIC_CENTER_SHIFT_X)
                )
                x_mid = float(base_x_mid + AUTO_CENTER_BLEND * x_shift)
        if center_ready and v_obs_span >= MIN_DYNAMIC_CENTER_SPAN_Y:
            covered_both_y = v_lo <= (base_y_mid - DYNAMIC_CENTER_COVER_MARGIN_Y) and v_hi >= (
                base_y_mid + DYNAMIC_CENTER_COVER_MARGIN_Y
            )
            if covered_both_y:
                v_mid = float((v_lo + v_hi) * 0.5)
                y_shift = float(
                    np.clip(v_mid - base_y_mid, -MAX_DYNAMIC_CENTER_SHIFT_Y, MAX_DYNAMIC_CENTER_SHIFT_Y)
                )
                y_mid = float(base_y_mid + AUTO_CENTER_BLEND * y_shift)

        x_min = x_mid - x_span * 0.5
        x_max = x_mid + x_span * 0.5
        y_min = y_mid - y_span * 0.5
        y_max = y_mid + y_span * 0.5
        return x_min, x_max, y_min, y_max

    def map(self, h, v):
        filtered = self._filter_measurement(h, v)
        if filtered is None:
            if self.history:
                return self.history[-1]
            return self.sw // 2, self.sh // 2
        h, v = filtered
        self._range_h_hist.append(h)
        self._range_v_hist.append(v)

        x_min, x_max, y_min, y_max = self._effective_bounds()

        # Normalize to 0-1 based on bounds
        x = (h - x_min) / (x_max - x_min + 1e-7)
        y = (v - y_min) / (y_max - y_min + 1e-7)

        # Expand tiny live iris spans so edge targets remain reachable even when
        # a user naturally has limited eye-lid aperture in one axis.
        bx, by = self._auto_range_boost()
        x = 0.5 + (x - 0.5) * bx
        y = 0.5 + (y - 0.5) * by

        if self.flip_x:
            x = 1.0 - x
        if self.flip_y:
            y = 1.0 - y

        # Independent horizontal range expansion/compression around center.
        x = 0.5 + (x - 0.5) * self.x_gain
        # Independent vertical range expansion/compression around center.
        y = 0.5 + (y - 0.5) * self.y_gain

        # Apply sensitivity: amplify deviation from center.
        # Calibration mode intentionally keeps this linear/low-gain to avoid
        # edge lock-in while collecting fitting samples.
        sx = min(self.sensitivity_x, 1.0) if self.calibration_mode else self.sensitivity_x
        sy = min(self.sensitivity_y, 1.0) if self.calibration_mode else self.sensitivity_y
        x = 0.5 + (x - 0.5) * sx
        y = 0.5 + (y - 0.5) * sy

        if self.calibration_mode:
            x = float(np.clip(x, 0.0, 1.0))
            y = float(np.clip(y, 0.0, 1.0))
        else:
            # Compress extremes before clipping so temporary outliers are less likely
            # to hard-pin the cursor to screen edges.
            x = self._soft_clip_unit(x)
            y = self._soft_clip_unit(y)

            # Apply power curve to push values toward edges.
            # This makes small movements near center less sticky and
            # helps the gaze actually reach screen edges.
            x = self._power_curve(x)
            y = self._power_curve(y)

        self._norm_xy_hist.append((x, y))
        if (not self.calibration_mode) and len(self._norm_xy_hist) >= RUNTIME_BIAS_MIN_SAMPLES:
            mx = float(np.mean([p[0] for p in self._norm_xy_hist]))
            my = float(np.mean([p[1] for p in self._norm_xy_hist]))
            sx_lo, sx_hi = np.percentile([p[0] for p in self._norm_xy_hist], [5, 95])
            sy_lo, sy_hi = np.percentile([p[1] for p in self._norm_xy_hist], [5, 95])
            span_x = float(sx_hi - sx_lo)
            span_y = float(sy_hi - sy_lo)
            enough_x_span = span_x >= max(RUNTIME_BIAS_MIN_SPAN_X, 2.0 * RUNTIME_BIAS_CENTER_MARGIN_X)
            enough_y_span = span_y >= max(RUNTIME_BIAS_MIN_SPAN_Y, 2.0 * RUNTIME_BIAS_CENTER_MARGIN_Y)
            covered_both_x = sx_lo <= (0.5 - RUNTIME_BIAS_CENTER_MARGIN_X) and sx_hi >= (0.5 + RUNTIME_BIAS_CENTER_MARGIN_X)
            covered_both_y = sy_lo <= (0.5 - RUNTIME_BIAS_CENTER_MARGIN_Y) and sy_hi >= (0.5 + RUNTIME_BIAS_CENTER_MARGIN_Y)
            if enough_x_span and covered_both_x and abs(0.5 - mx) > 0.10:
                self._runtime_x_bias = float(
                    np.clip(
                        self._runtime_x_bias + RUNTIME_BIAS_UPDATE * (0.5 - mx),
                        -RUNTIME_BIAS_MAX,
                        RUNTIME_BIAS_MAX,
                    )
                )
            else:
                self._runtime_x_bias *= RUNTIME_BIAS_DECAY
            if enough_y_span and covered_both_y and abs(0.5 - my) > 0.10:
                self._runtime_y_bias = float(
                    np.clip(
                        self._runtime_y_bias + RUNTIME_BIAS_UPDATE * (0.5 - my),
                        -RUNTIME_BIAS_MAX,
                        RUNTIME_BIAS_MAX,
                    )
                )
            else:
                self._runtime_y_bias *= RUNTIME_BIAS_DECAY

        x += self.x_offset + self._runtime_x_bias
        y += self.y_offset + self._runtime_y_bias

        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        # Adaptive normalized smoothing: settle strongly while fixating to
        # suppress drift/jitter, but stay responsive on intentional large moves.
        if self._norm_x_lp is None:
            self._norm_x_lp = x
            self._norm_y_lp = y
        delta = float(np.hypot(x - self._norm_x_lp, y - self._norm_y_lp))
        alpha = 0.85 if delta > 0.10 else 0.50
        self._norm_x_lp = (1.0 - alpha) * self._norm_x_lp + alpha * x
        self._norm_y_lp = (1.0 - alpha) * self._norm_y_lp + alpha * y
        x = self._norm_x_lp
        y = self._norm_y_lp

        sx = int(round(x * (self.sw - 1)))
        sy = int(round(y * (self.sh - 1)))

        # Bound frame-to-frame cursor velocity to suppress one-frame spikes.
        if self.history:
            prev_x, prev_y = self.history[-1]
            dx = float(sx - prev_x)
            dy = float(sy - prev_y)
            step = float(np.hypot(dx, dy))
            max_step_px = max(40.0, min(self.sw, self.sh) * MAX_CURSOR_STEP_RATIO)
            if step > max_step_px:
                scale = max_step_px / (step + 1e-7)
                sx = int(round(prev_x + dx * scale))
                sy = int(round(prev_y + dy * scale))

        # Large gaze shifts should not drag stale history (reduces long settling tails).
        if self.history:
            prev_x, prev_y = self.history[-1]
            jump_px = float(np.hypot(sx - prev_x, sy - prev_y))
            jump_reset_px = max(self.sw, self.sh) * 0.05
            if jump_px > jump_reset_px:
                self.history.clear()

        # Smooth with recency weighting to reduce jitter.
        self.history.append((sx, sy))
        # Exponential recency weighting lowers lag while still averaging noise.
        weights = np.geomspace(0.75, 1.0, len(self.history))
        hx = np.array([p[0] for p in self.history], dtype=float)
        hy = np.array([p[1] for p in self.history], dtype=float)
        sx = int(round(float(np.average(hx, weights=weights))))
        sy = int(round(float(np.average(hy, weights=weights))))

        return sx, sy

    def _soft_clip_unit(self, val):
        """Softly compress values into [0, 1] to reduce edge lock-in."""
        val = float(np.clip(val, -1.0, 2.0))
        centered = val - 0.5
        strength = 3.2
        compressed = np.tanh(centered * strength)
        return float(np.clip(0.5 + 0.5 * compressed, 0.0, 1.0))

    def _power_curve(self, val):
        """Apply a signed power curve around 0.5.

        Maps 0-1 through a curve that compresses movement near center
        and expands toward edges, giving a stable center region while
        still allowing full reach to screen corners.
        """
        val = float(np.clip(val, 0.0, 1.0))
        centered = val - 0.5
        sign = 1.0 if centered >= 0 else -1.0
        stretched = sign * (abs(centered) * 2) ** self.curve_exp / 2
        return stretched + 0.5

    def save_state(self):
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)

        def _json_safe(value):
            if isinstance(value, dict):
                return {k: _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            if isinstance(value, np.ndarray):
                return _json_safe(value.tolist())
            if isinstance(value, np.generic):
                return value.item()
            return value

        payload = {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "sensitivity_x": self.sensitivity_x,
            "sensitivity_y": self.sensitivity_y,
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
            "x_offset": self.x_offset,
            "x_gain": self.x_gain,
            "y_offset": self.y_offset,
            "y_gain": self.y_gain,
        }
        payload = _json_safe(payload)
        # Atomic write to avoid truncated/corrupt JSON when interrupted.
        cfg_dir = os.path.dirname(SETTINGS_FILE)
        fd, tmp_path = tempfile.mkstemp(prefix=".gaze_settings.", suffix=".tmp", dir=cfg_dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, SETTINGS_FILE)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def reset(self):
        self.x_min = 0.38
        self.x_max = 0.60
        self.y_min = 0.40
        self.y_max = 0.60
        self.flip_x = True
        self.flip_y = False
        self.x_offset = 0.0
        self.x_gain = 1.0
        self.y_offset = 0.0
        self.y_gain = 1.0
        self.clear_runtime_state(reset_bias=True)
        if os.path.exists(SETTINGS_FILE):
            os.remove(SETTINGS_FILE)
        print("[settings] reset to defaults")

    def _is_valid_bounds(self, x_min, x_max, y_min, y_max):
        if not all(np.isfinite(v) for v in (x_min, x_max, y_min, y_max)):
            return False
        if x_max - x_min < 0.02:
            return False
        if y_max - y_min < 0.02:
            return False
        # Normalized landmark coordinates should stay close to [0, 1].
        if min(x_min, y_min) < -0.5 or max(x_max, y_max) > 1.5:
            return False
        return True

    def _load(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"[settings] failed to load settings ({e}); using defaults")
                # Keep the bad file from crashing future runs.
                try:
                    bad_path = SETTINGS_FILE + ".corrupt"
                    os.replace(SETTINGS_FILE, bad_path)
                    print(f"[settings] moved corrupt file to {bad_path}")
                except OSError:
                    pass
                return
            x_min = float(d.get("x_min", self.x_min))
            x_max = float(d.get("x_max", self.x_max))
            y_min = float(d.get("y_min", self.y_min))
            y_max = float(d.get("y_max", self.y_max))

            if self._is_valid_bounds(x_min, x_max, y_min, y_max):
                self.x_min = x_min
                self.x_max = x_max
                self.y_min = y_min
                self.y_max = y_max
                self._ensure_min_span()
                self.flip_x = bool(d.get("flip_x", self.flip_x))
                self.flip_y = bool(d.get("flip_y", self.flip_y))
                self.x_offset = float(d.get("x_offset", self.x_offset))
                self.x_gain = float(d.get("x_gain", self.x_gain))
                self.y_offset = float(d.get("y_offset", self.y_offset))
                self.y_gain = float(d.get("y_gain", self.y_gain))
                self.set_x_offset(self.x_offset)
                self.set_x_gain(self.x_gain)
                self.set_y_offset(self.y_offset)
                self.set_y_gain(self.y_gain)
                self.sensitivity_x = float(d.get("sensitivity_x", self.sensitivity_x))
                self.sensitivity_y = float(d.get("sensitivity_y", self.sensitivity_y))
                loaded_sy = self.sensitivity_y
                self.set_sensitivity(self.sensitivity_x)
                self.set_vertical_sensitivity(loaded_sy)
                self.clear_runtime_state(reset_bias=True)
                print(f"[settings] loaded from {SETTINGS_FILE}")
            else:
                print("[settings] ignored invalid saved settings")


class Overlay:
    """Small always-on-top marker window that follows the gaze point."""

    def __init__(self, screen_w, screen_h):
        self.sw = screen_w
        self.sh = screen_h
        self.x = screen_w // 2
        self.y = screen_h // 2
        self.alive = True

        self.root = tk.Tk()
        self.root.title("gaze")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        self.s = DOT_SIZE
        self.r = self.s // 2
        self._last_geom = None
        self._click_fx_total = 7
        self._click_fx_left = 0
        self._hidden_until = 0.0
        self._is_withdrawn = False
        self._pending_click_fx = False

        self.canvas = tk.Canvas(self.root, width=self.s, height=self.s, bg="black", highlightthickness=0)
        self.canvas.pack()
        self._emoji_img = self._load_ufo_asset()
        if self._emoji_img is not None:
            self.canvas.create_image(self.r, self.r, image=self._emoji_img)
        else:
            # Fallback to rendered emoji or text when asset load fails.
            self._emoji_img = self._make_ufo_image()
            if self._emoji_img is not None:
                self.canvas.create_image(self.r, self.r, image=self._emoji_img)
            else:
                self.canvas.create_text(
                    self.r,
                    self.r,
                    text="🛸",
                    font=("Noto Color Emoji", int(self.s * 0.62)),
                    fill="#ffffff",
                )
        self._beam_id = self.canvas.create_polygon(
            self.r, self.r + 6, self.r - 5, self.s - 2, self.r + 5, self.s - 2,
            fill="#66e8ff",
            outline="",
            state="hidden",
        )
        self._ring_id = self.canvas.create_oval(
            self.r - 6, self.r + 6, self.r + 6, self.r + 18,
            outline="#9ff8ff",
            width=2,
            state="hidden",
        )
        self._pulse_id = self.canvas.create_oval(
            6, 6, self.s - 6, self.s - 6,
            outline="#7ff6ff",
            width=3,
            state="hidden",
        )
        self.tick()

    def _load_ufo_asset(self):
        """Load bundled UFO icon to avoid platform-specific emoji font issues."""
        if not os.path.exists(UFO_ICON_PATH):
            return None
        try:
            return tk.PhotoImage(file=UFO_ICON_PATH)
        except Exception:
            return None

    def _make_ufo_image(self):
        """Render a color UFO emoji into a Tk image to avoid font fallback issues."""
        if Image is None:
            return None

        font_candidates = [
            "/usr/share/fonts/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        ]
        font_path = next((p for p in font_candidates if os.path.exists(p)), None)
        if font_path is None:
            return None

        try:
            # Color emoji fonts often support only fixed bitmap strike sizes.
            strike_sizes = [self.s, 109, 128, 96, 72, 64]
            font = None
            strike = None
            for size in strike_sizes:
                try:
                    font = ImageFont.truetype(font_path, size)
                    strike = size
                    break
                except OSError:
                    continue
            if font is None:
                return None

            src = Image.new("RGBA", (strike, strike), (0, 0, 0, 0))
            draw = ImageDraw.Draw(src)
            bbox = draw.textbbox((0, 0), "🛸", font=font, embedded_color=True)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = (strike - w) // 2 - bbox[0]
            y = (strike - h) // 2 - bbox[1]
            draw.text((x, y), "🛸", font=font, embedded_color=True)

            if strike != self.s:
                src = src.resize((self.s, self.s), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(src)
        except Exception:
            return None

    def move(self, x, y):
        self.x = x
        self.y = y

    def trigger_click_fx(self):
        """Show a short UFO-themed click pulse for Enter-triggered clicks."""
        self._click_fx_left = self._click_fx_total

    def prepare_click_through(self, hide_seconds=0.10):
        """Temporarily hide overlay so OS click goes to the underlying app."""
        self._hidden_until = max(self._hidden_until, time.time() + hide_seconds)
        self._pending_click_fx = True
        try:
            if not self._is_withdrawn:
                self.root.withdraw()
                self._is_withdrawn = True
        except tk.TclError:
            self.alive = False

    def _update_click_fx(self):
        if self._click_fx_left <= 0:
            self.canvas.itemconfigure(self._beam_id, state="hidden")
            self.canvas.itemconfigure(self._ring_id, state="hidden")
            self.canvas.itemconfigure(self._pulse_id, state="hidden")
            self.canvas.configure(bg="black")
            return

        frame = self._click_fx_total - self._click_fx_left
        beam_half = 5 + frame * 2
        beam_top = self.r + 6
        beam_bottom = min(self.s - 2, beam_top + 10 + frame * 2)
        self.canvas.coords(
            self._beam_id,
            self.r, beam_top,
            self.r - beam_half, beam_bottom,
            self.r + beam_half, beam_bottom,
        )

        ring_r = 7 + frame * 3
        ring_y = min(self.s - 2, self.r + 12 + frame * 2)
        self.canvas.coords(
            self._ring_id,
            self.r - ring_r, ring_y - ring_r // 2,
            self.r + ring_r, ring_y + ring_r // 2,
        )
        ring_color = "#d4fdff" if frame < 3 else "#7ceeff"
        pulse_pad = 6 + frame
        self.canvas.itemconfigure(self._beam_id, fill="#66e8ff", state="normal")
        self.canvas.itemconfigure(self._ring_id, outline=ring_color, state="normal")
        self.canvas.coords(
            self._pulse_id,
            pulse_pad,
            pulse_pad,
            self.s - pulse_pad,
            self.s - pulse_pad,
        )
        self.canvas.itemconfigure(self._pulse_id, outline="#7ff6ff", state="normal")
        self.canvas.configure(bg="#04161a" if frame % 2 == 0 else "black")
        self.canvas.tag_raise(self._beam_id)
        self.canvas.tag_raise(self._ring_id)
        self.canvas.tag_raise(self._pulse_id)
        self._click_fx_left -= 1

    def tick(self):
        if not self.alive:
            return
        ox = max(0, min(self.x - self.r, self.sw - self.s))
        oy = max(0, min(self.y - self.r, self.sh - self.s))
        geom = f"{self.s}x{self.s}+{ox}+{oy}"
        try:
            now = time.time()
            if now < self._hidden_until:
                if not self._is_withdrawn:
                    self.root.withdraw()
                    self._is_withdrawn = True
                self.root.update_idletasks()
                self.root.update()
                return
            if self._is_withdrawn:
                self.root.deiconify()
                self.root.attributes("-topmost", True)
                self._is_withdrawn = False
            if self._pending_click_fx:
                self.trigger_click_fx()
                self._pending_click_fx = False
            if geom != self._last_geom:
                self.root.geometry(geom)
                self._last_geom = geom
            self._update_click_fx()
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.alive = False

    def stop(self):
        if not self.alive:
            return
        self.alive = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def draw_debug(frame, landmarks, h, v, sx, sy, mapper):
    """Annotate the webcam frame with tracking info."""
    fh, fw = frame.shape[:2]

    # Iris centers (average of ring landmarks)
    for ring in (R_IRIS_RING, L_IRIS_RING):
        cx = int(np.mean([landmarks[i].x for i in ring]) * fw)
        cy = int(np.mean([landmarks[i].y for i in ring]) * fh)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Iris ring points
    for ring in (R_IRIS_RING, L_IRIS_RING):
        for idx in ring:
            cx = int(landmarks[idx].x * fw)
            cy = int(landmarks[idx].y * fh)
            cv2.circle(frame, (cx, cy), 2, (0, 200, 0), -1)

    # Eye corners
    for idx in (L_OUTER, L_INNER, R_INNER, R_OUTER):
        cx = int(landmarks[idx].x * fw)
        cy = int(landmarks[idx].y * fh)
        cv2.circle(frame, (cx, cy), 2, (0, 150, 255), -1)

    # Eye top/bottom
    for idx in (L_TOP, L_BOTTOM, R_TOP, R_BOTTOM):
        cx = int(landmarks[idx].x * fw)
        cy = int(landmarks[idx].y * fh)
        cv2.circle(frame, (cx, cy), 2, (255, 150, 0), -1)

    # Text overlay
    cv2.putText(frame, f"iris h={h:.3f} v={v:.3f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"screen ({sx}, {sy})",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"sensX={mapper.sensitivity_x:.1f} sensY={mapper.sensitivity_y:.1f} xG={mapper.x_gain:.2f} xO={mapper.x_offset:+.2f} yG={mapper.y_gain:.2f} yO={mapper.y_offset:+.2f} sm={mapper.history.maxlen}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, "[r]reset [d]debug [o/l]Yg [u/j]Yo [v]flipY [q]quit",
                (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)


def click_at(x, y):
    """Left click at absolute screen coordinates using xdotool."""
    if shutil.which("xdotool") is None:
        print("Enter pressed, but xdotool is not installed; cannot click.")
        return
    try:
        subprocess.run(
            ["xdotool", "mousemove", str(x), str(y), "click", "1"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print("Enter pressed, but xdotool click failed.")


def ensure_opencv_qt_fonts():
    """Create OpenCV's expected Qt fonts path to suppress repeated warnings."""
    try:
        cv2_dir = os.path.dirname(cv2.__file__)
        qt_fonts_dir = os.path.join(cv2_dir, "qt", "fonts")
        if os.path.isdir(qt_fonts_dir):
            return
        os.makedirs(os.path.dirname(qt_fonts_dir), exist_ok=True)
        # Try a symlink first.
        for src in ("/usr/share/fonts/noto", "/usr/share/fonts/liberation", "/usr/share/fonts"):
            if os.path.isdir(src):
                try:
                    os.symlink(src, qt_fonts_dir)
                    return
                except FileExistsError:
                    return
                except OSError:
                    pass
        # Fallback: create empty dir to satisfy path checks.
        os.makedirs(qt_fonts_dir, exist_ok=True)
    except Exception:
        pass


def main():
    # Screen
    monitor = get_monitors()[0]
    sw, sh = monitor.width, monitor.height
    print(f"Screen: {sw}x{sh}")
    ensure_opencv_qt_fonts()

    # Webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    tracker = EyeTracker()
    mapper = GazeMapper(sw, sh)

    print("Launching tracker loop...")

    overlay = Overlay(sw, sh)

    show_debug = True
    smoothing = SMOOTHING_FRAMES

    print("Eye Gaze Tracker running.")
    print(f"  sensitivity_x={mapper.sensitivity_x:.1f} sensitivity_y={mapper.sensitivity_y:.1f} smoothing={smoothing}")
    print("  r = reset settings | d = toggle debug")
    print("  s/S = both sensitivity down/up")
    print("  v = toggle vertical flip | o/l = vertical gain up/down | u/j = vertical offset up/down")
    print("  b = toggle horizontal flip | p/; = horizontal gain up/down | h/n = horizontal offset left/right")
    print("  +/- = smoothing | Enter = click | q = quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            result = tracker.process(frame)

            if result is not None:
                (h, v), landmarks = result
                sx, sy = mapper.map(h, v)
                overlay.move(sx, sy)

                if show_debug:
                    draw_debug(frame, landmarks, h, v, sx, sy, mapper)
            overlay.tick()

            if show_debug:
                cv2.imshow("Eye Tracker Debug", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow("Eye Tracker Debug")
            elif key == ord("r"):
                mapper.reset()
            elif key == ord("S"):  # Shift+s = increase sensitivity
                mapper.set_sensitivity(mapper.sensitivity_x + 0.25)
                mapper.save_state()
                print(f"Sensitivity X/Y: {mapper.sensitivity_x:.2f}")
            elif key == ord("s"):  # s = decrease sensitivity
                mapper.set_sensitivity(mapper.sensitivity_x - 0.25)
                mapper.save_state()
                print(f"Sensitivity X/Y: {mapper.sensitivity_x:.2f}")
            elif key == ord("i"):  # increase vertical sensitivity
                mapper.set_vertical_sensitivity(mapper.sensitivity_y + 0.25)
                mapper.save_state()
                print(f"Vertical sensitivity: {mapper.sensitivity_y:.2f}")
            elif key == ord("k"):  # decrease vertical sensitivity
                mapper.set_vertical_sensitivity(mapper.sensitivity_y - 0.25)
                mapper.save_state()
                print(f"Vertical sensitivity: {mapper.sensitivity_y:.2f}")
            elif key == ord("u"):  # move mapped gaze upward
                mapper.set_y_offset(mapper.y_offset - 0.01)
                mapper.save_state()
                print(f"Y offset: {mapper.y_offset:+.2f}")
            elif key == ord("j"):  # move mapped gaze downward
                mapper.set_y_offset(mapper.y_offset + 0.01)
                mapper.save_state()
                print(f"Y offset: {mapper.y_offset:+.2f}")
            elif key == ord("o"):  # increase vertical range gain
                mapper.set_y_gain(mapper.y_gain + 0.05)
                mapper.save_state()
                print(f"Y gain: {mapper.y_gain:.2f}")
            elif key == ord("l"):  # decrease vertical range gain
                mapper.set_y_gain(mapper.y_gain - 0.05)
                mapper.save_state()
                print(f"Y gain: {mapper.y_gain:.2f}")
            elif key == ord("v"):  # toggle vertical direction
                mapper.toggle_flip_y()
                mapper.save_state()
                print(f"Vertical flip: {mapper.flip_y}")
            elif key == ord("b"):  # toggle horizontal direction
                mapper.toggle_flip_x()
                mapper.save_state()
                print(f"Horizontal flip: {mapper.flip_x}")
            elif key == ord("p"):  # increase horizontal range gain
                mapper.set_x_gain(mapper.x_gain + 0.05)
                mapper.save_state()
                print(f"X gain: {mapper.x_gain:.2f}")
            elif key == ord(";"):  # decrease horizontal range gain
                mapper.set_x_gain(mapper.x_gain - 0.05)
                mapper.save_state()
                print(f"X gain: {mapper.x_gain:.2f}")
            elif key == ord("h"):  # move mapped gaze left
                mapper.set_x_offset(mapper.x_offset - 0.01)
                mapper.save_state()
                print(f"X offset: {mapper.x_offset:+.2f}")
            elif key == ord("n"):  # move mapped gaze right
                mapper.set_x_offset(mapper.x_offset + 0.01)
                mapper.save_state()
                print(f"X offset: {mapper.x_offset:+.2f}")
            elif key in (ord("+"), ord("=")):
                smoothing = min(20, smoothing + 1)
                mapper.set_smoothing(smoothing)
                print(f"Smoothing: {smoothing}")
            elif key == ord("-"):
                smoothing = max(1, smoothing - 1)
                mapper.set_smoothing(smoothing)
                print(f"Smoothing: {smoothing}")
            elif key in (10, 13):
                overlay.prepare_click_through()
                overlay.tick()  # apply withdraw immediately before emitting click
                time.sleep(0.015)
                click_at(overlay.x, overlay.y)

    except KeyboardInterrupt:
        pass
    finally:
        overlay.stop()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
