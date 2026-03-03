#!/usr/bin/env python3
"""Shared webcam gaze extraction core used by cursor and tab trackers."""

from __future__ import annotations

import os
import sys

import cv2
import mediapipe as mp
import numpy as np

WEBCAM_INDEX = 0
WEBCAM_W = 640
WEBCAM_H = 480
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# Head-pose compensation coefficients (empirical).
# These offsets are applied to iris ratios after head-anchor capture.
HEAD_COMP_X_FROM_DX = 0.11
HEAD_COMP_Y_FROM_DY = 0.075
HEAD_COMP_Y_FROM_SCALE = 0.010
HEAD_COMP_MAX_X = 0.030
HEAD_COMP_MAX_Y = 0.035
HEAD_ANCHOR_ALPHA = 0.035
HEAD_ANCHOR_WARMUP_FRAMES = 12
HEAD_ANCHOR_WARMUP_ALPHA = 0.30
HEAD_ANCHOR_EDGE_ADAPT_FLOOR = 0.45

# MediaPipe Face Landmarker iris indices.
R_OUTER = 33
R_INNER = 133
R_TOP = 159
R_BOTTOM = 145
R_IRIS_RING = (469, 470, 471, 472)

L_INNER = 362
L_OUTER = 263
L_TOP = 386
L_BOTTOM = 374
L_IRIS_RING = (474, 475, 476, 477)


class EyeTracker:
    """Extract normalized iris position using MediaPipe FaceLandmarker Tasks API."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model not found at {MODEL_PATH}")
            print("Download it with:")
            print(
                f'  curl -L -o "{MODEL_PATH}" '
                '"https://storage.googleapis.com/mediapipe-models/'
                'face_landmarker/face_landmarker/float16/latest/face_landmarker.task"'
            )
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
        """Return ((h_ratio, v_ratio), landmarks) or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += 33
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]

        def iris_center(ring_indices):
            xs = [lm[i].x for i in ring_indices]
            ys = [lm[i].y for i in ring_indices]
            return sum(xs) / len(xs), sum(ys) / len(ys)

        def ratio(iris_ring, outer, inner, top, bottom):
            ix, iy = iris_center(iris_ring)
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
            dx_norm = float(np.clip((hx - ax) / (ascale + 1e-7), -0.30, 0.30))
            dy_norm = float(np.clip((hy - ay) / (ascale + 1e-7), -0.30, 0.30))
            dscale = float(np.clip(np.log((hs + 1e-7) / (ascale + 1e-7)), -0.25, 0.25))
            x_comp = float(np.clip(HEAD_COMP_X_FROM_DX * dx_norm, -HEAD_COMP_MAX_X, HEAD_COMP_MAX_X))
            y_comp = HEAD_COMP_Y_FROM_DY * dy_norm + HEAD_COMP_Y_FROM_SCALE * dscale
            y_comp = float(np.clip(y_comp, -HEAD_COMP_MAX_Y, HEAD_COMP_MAX_Y))
            gaze_excursion = max(abs(raw_h - 0.5), abs(raw_v - 0.5))
            edge_blend = float(np.clip((gaze_excursion - 0.06) / 0.24, 0.0, 1.0))
            comp_scale = 1.0 - 0.45 * edge_blend
            x_comp *= comp_scale
            y_comp *= comp_scale
            h -= x_comp
            v -= y_comp
            head_motion = max(abs(dx_norm), abs(dy_norm), abs(dscale))

        if self._head_anchor_warmup_left <= 0:
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
