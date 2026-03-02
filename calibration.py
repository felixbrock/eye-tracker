#!/usr/bin/env python3
"""Eye-tracker calibration sampler and mapper fitter.

Collects deterministic multi-point fixation data and fits mapper bounds.
"""

import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from screeninfo import get_monitors

from eye_tracker import (
    EyeTracker,
    GazeMapper,
    L_BOTTOM,
    L_INNER,
    L_OUTER,
    L_TOP,
    R_BOTTOM,
    R_INNER,
    R_OUTER,
    R_TOP,
    SETTINGS_FILE,
    SENSITIVITY_X,
    SENSITIVITY_Y,
    WEBCAM_H,
    WEBCAM_INDEX,
    WEBCAM_W,
)


ITERATION_TIMEOUT_SECONDS = 7.0
TARGET_DWELL_SECONDS = 0.35
TARGET_CAPTURE_SECONDS = 1.20
TARGET_SETTLE_SECONDS = 0.45
STABLE_MIN_FRAMES = 8
STABLE_MAX_IRIS_STEP = 0.020
BOX_W_RATIO = 0.14
BOX_H_RATIO = 0.18
OUT_DIR = "calibration_logs"
FLIP_EVAL_MIN_SAMPLES = 90
FLIP_EVAL_REL_MARGIN = 0.06
FLIP_ACTIVE_MIN_SAMPLES = 18
FLIP_ACTIVE_REL_MARGIN = 0.03
BND_MIN = -0.20
BND_MAX = 1.20
SPAN_MIN = 0.18
SPAN_MAX = 0.55
MIN_EYE_OPEN_RATIO = 0.075
MAX_HEAD_MOTION_NORM = 0.23
MAX_IRIS_STEP = 0.085
FIT_AXIS_MIN_TARGETS = 3
TARGET_MAX_RETRIES = 2
TARGET_STD_MAX_H = 0.0065
TARGET_STD_MAX_V = 0.0065
TARGET_MIN_SAMPLES = 20


def _fit_axis_bounds(features, targets, force_flip=None, weights=None):
    """Fit affine feature->target map and derive mapper bounds."""
    best = None
    flip_opts = [force_flip] if force_flip is not None else [False, True]
    x = np.asarray(features, dtype=float)
    t = np.asarray(targets, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    if x.size < 3:
        return None
    if not np.isfinite(x).all() or not np.isfinite(t).all():
        return None
    if w is not None:
        if w.size != x.size or not np.isfinite(w).all():
            return None
        w = np.clip(w, 1e-6, None)
    for flip in flip_opts:
        # Mapper applies flip after normalization. Fit target in that space.
        y = 1.0 - t if flip else t
        A = np.column_stack([x, np.ones_like(x)])
        if w is None:
            coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
        else:
            sw = np.sqrt(w)
            Aw = A * sw[:, None]
            yw = y * sw
            coeff, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        a = float(coeff[0])
        b = float(coeff[1])
        if abs(a) < 1e-6:
            continue
        pred = a * x + b
        abs_err = np.abs(pred - y)
        if w is None:
            mae = float(np.mean(abs_err))
        else:
            mae = float(np.sum(abs_err * w) / (np.sum(w) + 1e-7))
        p_lo, p_hi = np.percentile(x, [5, 95])
        span = float(np.clip(abs((1.0 - b) / a - (0.0 - b) / a), SPAN_MIN, SPAN_MAX))
        center = float(np.clip((p_lo + p_hi) * 0.5, BND_MIN, BND_MAX))
        x_min = center - span * 0.5
        x_max = center + span * 0.5
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        # Small margin prevents center-fit from clipping edge samples.
        margin = 0.04 * span
        x_min = float(np.clip(x_min - margin, BND_MIN, BND_MAX))
        x_max = float(np.clip(x_max + margin, BND_MIN, BND_MAX))
        if x_max - x_min < SPAN_MIN:
            continue
        cand = {
            "flip": bool(flip),
            "min": x_min,
            "max": x_max,
            "mae": mae,
        }
        if best is None or cand["mae"] < best["mae"]:
            best = cand
    return best


def derive_mapper_settings(payload):
    sw = float(payload["screen"]["width"])
    sh = float(payload["screen"]["height"])
    grouped = []
    for it in payload["iterations"]:
        box = it["target_box"]
        cx = (float(box["x1"]) + float(box["x2"])) * 0.5
        cy = (float(box["y1"]) + float(box["y2"])) * 0.5
        tx = float(np.clip(cx / max(sw - 1.0, 1.0), 0.0, 1.0))
        ty = float(np.clip(cy / max(sh - 1.0, 1.0), 0.0, 1.0))
        iter_samples = [
            s
            for s in it["samples"]
            if np.isfinite(float(s["iris_h"])) and np.isfinite(float(s["iris_v"]))
        ]
        if len(iter_samples) < 8:
            continue

        h_arr = np.asarray([float(s["iris_h"]) for s in iter_samples], dtype=float)
        v_arr = np.asarray([float(s["iris_v"]) for s in iter_samples], dtype=float)
        h_med = float(np.median(h_arr))
        v_med = float(np.median(v_arr))
        h_mad = float(np.median(np.abs(h_arr - h_med))) + 1e-4
        v_mad = float(np.median(np.abs(v_arr - v_med))) + 1e-4
        keep_mask = (
            (np.abs(h_arr - h_med) <= (3.5 * h_mad + 0.008))
            & (np.abs(v_arr - v_med) <= (3.5 * v_mad + 0.008))
        )
        keep_idx = np.where(keep_mask)[0]
        if keep_idx.size < 6:
            keep_idx = np.arange(h_arr.size)
        h_keep = h_arr[keep_idx]
        v_keep = v_arr[keep_idx]
        grouped.append(
            {
                "tx": tx,
                "ty": ty,
                "h_med": float(np.median(h_keep)),
                "v_med": float(np.median(v_keep)),
                "w": float(max(1.0, keep_idx.size) / (1.0 + 5.0 * (h_mad + v_mad))),
            }
        )
    if len(grouped) < 6:
        return None

    x_groups = {}
    y_groups = {}
    for g in grouped:
        x_groups.setdefault(round(g["tx"], 6), []).append(g)
        y_groups.setdefault(round(g["ty"], 6), []).append(g)

    if len(x_groups) < FIT_AXIS_MIN_TARGETS:
        return None
    if len(y_groups) < FIT_AXIS_MIN_TARGETS:
        return None

    hs = []
    txs = []
    wsx = []
    for tx_key in sorted(x_groups.keys()):
        gs = x_groups[tx_key]
        hs.append(float(np.median([g["h_med"] for g in gs])))
        txs.append(float(gs[0]["tx"]))
        wsx.append(float(np.sum([g["w"] for g in gs])))

    vs = []
    tys = []
    wsy = []
    h_for_y = []
    for ty_key in sorted(y_groups.keys()):
        gs = y_groups[ty_key]
        h_for_y.append(float(np.median([g["h_med"] for g in gs])))
        vs.append(float(np.median([g["v_med"] for g in gs])))
        tys.append(float(gs[0]["ty"]))
        wsy.append(float(np.sum([g["w"] for g in gs])))

    # Estimate vertical leakage from horizontal gaze and remove it before
    # fitting vertical bounds. This helps users where eyelid/pose geometry
    # makes v vary with x.
    h_pts = np.asarray([g["h_med"] for g in grouped], dtype=float)
    v_pts = np.asarray([g["v_med"] for g in grouped], dtype=float)
    ty_pts = np.asarray([g["ty"] for g in grouped], dtype=float)
    w_pts = np.asarray([g["w"] for g in grouped], dtype=float)
    h_center = float(np.median(h_pts))
    yx_coupling = 0.0
    if len(grouped) >= 6:
        A = np.column_stack([h_pts, ty_pts, np.ones_like(h_pts)])
        sw = np.sqrt(np.clip(w_pts, 1e-6, None))
        coeff, *_ = np.linalg.lstsq(A * sw[:, None], v_pts * sw, rcond=None)
        yx_coupling = float(np.clip(coeff[0], -1.0, 1.0))
        for i in range(len(vs)):
            vs[i] = float(vs[i] - yx_coupling * (h_for_y[i] - h_center))

    fx = _fit_axis_bounds(hs, txs, force_flip=None, weights=wsx)
    fy = _fit_axis_bounds(vs, tys, force_flip=None, weights=wsy)
    if fx is None or fy is None:
        return None

    settings = {
        "x_min": fx["min"],
        "x_max": fx["max"],
        "y_min": fy["min"],
        "y_max": fy["max"],
        "sensitivity_x": float(SENSITIVITY_X),
        "sensitivity_y": float(SENSITIVITY_Y),
        "flip_x": bool(fx["flip"]),
        "flip_y": bool(fy["flip"]),
        "x_offset": 0.0,
        "x_gain": 1.0,
        "y_offset": 0.0,
        "y_gain": 1.0,
        "y_x_coupling": float(yx_coupling),
        "y_x_center": float(h_center),
        "calibration_source_created_at": payload.get("created_at"),
    }
    return settings


def write_mapper_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    tmp_path = SETTINGS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, SETTINGS_FILE)


def calibration_targets(sw, sh):
    bw = max(120, int(sw * BOX_W_RATIO))
    bh = max(120, int(sh * BOX_H_RATIO))

    anchors_x = [0, max(0, (sw - bw) // 2), max(0, sw - bw)]
    anchors_y = [0, max(0, (sh - bh) // 2), max(0, sh - bh)]
    # Base 3x3 coverage phase.
    grid_points = [
        ("grid-center", 1, 1, "mixed"),
        ("grid-top-left", 0, 0, "mixed"),
        ("grid-top-right", 2, 0, "mixed"),
        ("grid-bottom-right", 2, 2, "mixed"),
        ("grid-bottom-left", 0, 2, "mixed"),
        ("grid-top-center", 1, 0, "mixed"),
        ("grid-right-center", 2, 1, "mixed"),
        ("grid-bottom-center", 1, 2, "mixed"),
        ("grid-left-center", 0, 1, "mixed"),
    ]
    # Dedicated vertical isolation phase on center column.
    vertical_points = [
        ("v-top-center-1", 1, 0, "vertical"),
        ("v-mid-center-1", 1, 1, "vertical"),
        ("v-bottom-center-1", 1, 2, "vertical"),
        ("v-top-center-2", 1, 0, "vertical"),
        ("v-mid-center-2", 1, 1, "vertical"),
        ("v-bottom-center-2", 1, 2, "vertical"),
    ]
    targets = []
    for name, ix, iy, phase in grid_points + vertical_points:
        x1 = anchors_x[ix]
        y1 = anchors_y[iy]
        targets.append(
            {
                "name": name,
                "phase": phase,
                "retry_index": 0,
                "target_box": (x1, y1, x1 + bw, y1 + bh),
            }
        )
    return targets


def distance_to_box(sx, sy, x1, y1, x2, y2):
    dx = max(x1 - sx, 0, sx - x2)
    dy = max(y1 - sy, 0, sy - y2)
    return float(np.hypot(dx, dy))


def _eye_open_ratio(lm):
    left_open = float(np.hypot(lm[L_BOTTOM].x - lm[L_TOP].x, lm[L_BOTTOM].y - lm[L_TOP].y))
    right_open = float(np.hypot(lm[R_BOTTOM].x - lm[R_TOP].x, lm[R_BOTTOM].y - lm[R_TOP].y))
    inter_eye = float(np.hypot(lm[L_OUTER].x - lm[R_OUTER].x, lm[L_OUTER].y - lm[R_OUTER].y))
    return ((left_open + right_open) * 0.5) / (inter_eye + 1e-7)


def _head_motion_norm(head_state, anchor_state):
    if anchor_state is None:
        return 0.0
    hx, hy, hs = head_state
    ax, ay, a_scale = anchor_state
    dx_norm = float((hx - ax) / (a_scale + 1e-7))
    dy_norm = float((hy - ay) / (a_scale + 1e-7))
    ds = float(np.log((hs + 1e-7) / (a_scale + 1e-7)))
    return max(abs(dx_norm), abs(dy_norm), abs(ds))


def _safe_corr(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2 or y.size != x.size:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_quality_report(iterations, sw, sh):
    point_tx = []
    point_ty = []
    point_h = []
    point_v = []
    per_target = []
    for it in iterations:
        box = it.get("target_box", {})
        samples = it.get("samples", [])
        if not box or not samples:
            continue
        hs = np.asarray([float(s["iris_h"]) for s in samples], dtype=float)
        vs = np.asarray([float(s["iris_v"]) for s in samples], dtype=float)
        if hs.size == 0 or vs.size == 0:
            continue
        cx = (float(box["x1"]) + float(box["x2"])) * 0.5
        cy = (float(box["y1"]) + float(box["y2"])) * 0.5
        tx = float(np.clip(cx / max(float(sw - 1), 1.0), 0.0, 1.0))
        ty = float(np.clip(cy / max(float(sh - 1), 1.0), 0.0, 1.0))
        point_tx.append(tx)
        point_ty.append(ty)
        point_h.append(float(np.median(hs)))
        point_v.append(float(np.median(vs)))
        per_target.append(
            {
                "name": it.get("target_name"),
                "phase": it.get("phase"),
                "retry_index": int(it.get("retry_index", 0)),
                "samples": int(hs.size),
                "h_std": float(np.std(hs)),
                "v_std": float(np.std(vs)),
            }
        )

    x_corr = abs(_safe_corr(point_h, point_tx))
    y_corr = abs(_safe_corr(point_v, point_ty))
    x_cross = abs(_safe_corr(point_h, point_ty))
    y_cross = abs(_safe_corr(point_v, point_tx))
    h_span = float((max(point_h) - min(point_h)) if point_h else 0.0)
    v_span = float((max(point_v) - min(point_v)) if point_v else 0.0)
    return {
        "points_used": int(len(point_h)),
        "x_corr_abs": float(x_corr),
        "y_corr_abs": float(y_corr),
        "x_cross_abs": float(x_cross),
        "y_cross_abs": float(y_cross),
        "axis_score": float(x_corr + y_corr - x_cross - y_cross),
        "h_span": h_span,
        "v_span": v_span,
        "per_target": per_target,
    }


def main():
    monitor = get_monitors()[0]
    sw, sh = monitor.width, monitor.height
    target_queue = calibration_targets(sw, sh)
    total_planned = len(target_queue)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    tracker = EyeTracker()
    # Keep calibration runs independent from any manually tuned persistent settings.
    # Evaluate both vertical-polarity options and lock to the better one based
    # on target-center distance during the run.
    probe_mappers = {}
    for flip_y in (False, True):
        mapper = GazeMapper(sw, sh, smoothing=1, load_saved_settings=False, calibration_mode=True)
        mapper.flip_y = flip_y
        probe_mappers[flip_y] = mapper
    active_flip_y = False
    flip_eval_samples = 0
    flip_eval_total_dist = {False: 0.0, True: 0.0}
    flip_eval_locked = False
    win = "Calibration Iteration"
    all_iterations = []

    try:
        tracker.reset_head_anchor()
        countdown_start = time.time()
        countdown_secs = 3
        while True:
            remaining = countdown_secs - int(time.time() - countdown_start)
            if remaining <= 0:
                break
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("Cancelled.")
                return
            disp = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.putText(
                disp,
                f"Calibration starts in {remaining}",
                (max(40, sw // 2 - 280), sh // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                disp,
                "Look at the screen and keep your head still",
                (max(40, sw // 2 - 340), sh // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (210, 210, 210),
                2,
            )
            cv2.imshow(win, disp)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        iteration_index = 0
        while target_queue:
            target = target_queue.pop(0)
            iteration_index += 1
            box = target["target_box"]
            prev_h = None
            prev_v = None
            for mapper in probe_mappers.values():
                mapper.clear_runtime_state(reset_bias=True)
                mapper.set_x_offset(0.0)
                mapper.set_y_offset(0.0)
                mapper.set_x_gain(1.0)
                mapper.set_y_gain(1.0)
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            samples = []
            rejected_quality = 0
            settled_since = None
            stable_frames = 0
            capture_started_at = None
            t0 = time.time()

            while True:
                elapsed = time.time() - t0
                if elapsed >= ITERATION_TIMEOUT_SECONDS:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("Cancelled.")
                    return

                ret, frame = cap.read()
                if not ret:
                    continue

                # Fit using the same compensated gaze stream used at runtime.
                result = tracker.process(frame, apply_head_comp=True, return_meta=True)
                disp = np.zeros((sh, sw, 3), dtype=np.uint8)

                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # Center marker to provide an explicit fixation point.
                cxi, cyi = int(round(cx)), int(round(cy))
                marker_len = 18
                cv2.line(disp, (cxi - marker_len, cyi), (cxi + marker_len, cyi), (255, 255, 255), 2)
                cv2.line(disp, (cxi, cyi - marker_len), (cxi, cyi + marker_len), (255, 255, 255), 2)
                cv2.circle(disp, (cxi, cyi), 6, (255, 255, 255), 2)
                cv2.putText(
                    disp,
                    "Move the eye-tracker cursor into the red box center",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    disp,
                    f"Iteration: {iteration_index} (planned {total_planned}, queue {len(target_queue)})",
                    (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (210, 210, 210),
                    2,
                )
                cv2.putText(
                    disp,
                    f"Timeout in: {max(0.0, ITERATION_TIMEOUT_SECONDS - elapsed):0.1f}s",
                    (40, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (210, 210, 210),
                    2,
                )
                cv2.putText(
                    disp,
                    "Esc/q = cancel",
                    (40, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (160, 160, 160),
                    1,
                )
                cv2.putText(
                    disp,
                    f"Target: {target['name']} ({target['phase']}) retry {target['retry_index']}",
                    (40, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (185, 185, 185),
                    2,
                )

                if result is not None:
                    (h, v), lm, head_state = result
                    eye_open = _eye_open_ratio(lm)
                    head_motion = _head_motion_norm(head_state, tracker.head_anchor)
                    iris_step = 0.0
                    if prev_h is not None and prev_v is not None:
                        iris_step = float(np.hypot(h - prev_h, v - prev_v))
                    prev_h = h
                    prev_v = v
                    quality_ok = (
                        eye_open >= MIN_EYE_OPEN_RATIO
                        and head_motion <= MAX_HEAD_MOTION_NORM
                        and iris_step <= MAX_IRIS_STEP
                    )
                    stable_ok = (
                        quality_ok
                        and head_motion <= (0.85 * MAX_HEAD_MOTION_NORM)
                        and iris_step <= STABLE_MAX_IRIS_STEP
                    )

                    mapped_points = {}
                    center_dists = {}
                    for flip_y, mapper in probe_mappers.items():
                        sx_i, sy_i = mapper.map(h, v)
                        mapped_points[flip_y] = (sx_i, sy_i)
                        center_dists[flip_y] = float(np.hypot(sx_i - cx, sy_i - cy))

                    if quality_ok and (not flip_eval_locked):
                        flip_eval_samples += 1
                        flip_eval_total_dist[False] += center_dists[False]
                        flip_eval_total_dist[True] += center_dists[True]
                        if flip_eval_samples >= FLIP_ACTIVE_MIN_SAMPLES:
                            mean_false = flip_eval_total_dist[False] / float(flip_eval_samples)
                            mean_true = flip_eval_total_dist[True] / float(flip_eval_samples)
                            best = min(mean_false, mean_true)
                            worst = max(mean_false, mean_true)
                            if (worst - best) / max(worst, 1.0) >= FLIP_ACTIVE_REL_MARGIN:
                                active_flip_y = mean_true < mean_false
                        if flip_eval_samples >= FLIP_EVAL_MIN_SAMPLES:
                            mean_false = flip_eval_total_dist[False] / float(flip_eval_samples)
                            mean_true = flip_eval_total_dist[True] / float(flip_eval_samples)
                            best = min(mean_false, mean_true)
                            worst = max(mean_false, mean_true)
                            if (worst - best) / max(worst, 1.0) >= FLIP_EVAL_REL_MARGIN:
                                active_flip_y = mean_true < mean_false
                                flip_eval_locked = True
                                print(
                                    "Auto flip-y locked to "
                                    f"{active_flip_y} after {flip_eval_samples} samples "
                                    f"(mean center distance false={mean_false:.1f}, true={mean_true:.1f})"
                                )

                    sx, sy = mapped_points[active_flip_y]
                    inside = x1 <= sx <= x2 and y1 <= sy <= y2
                    dist_box_px = distance_to_box(sx, sy, x1, y1, x2, y2)
                    dist_center_px = float(np.hypot(sx - cx, sy - cy))

                    if quality_ok:
                        if stable_ok:
                            stable_frames += 1
                            if settled_since is None:
                                settled_since = elapsed
                        else:
                            stable_frames = 0
                            settled_since = None
                    else:
                        stable_frames = 0
                        settled_since = None

                    if (
                        capture_started_at is None
                        and settled_since is not None
                        and stable_frames >= STABLE_MIN_FRAMES
                        and elapsed >= TARGET_SETTLE_SECONDS
                        and (elapsed - settled_since) >= TARGET_DWELL_SECONDS
                    ):
                        for mapper in probe_mappers.values():
                            mapper.lock_calibration_boost()
                        capture_started_at = elapsed

                    if capture_started_at is not None and quality_ok:
                        samples.append(
                            {
                                "iteration_index": iteration_index,
                                "t": elapsed,
                                "cursor_x": int(sx),
                                "cursor_y": int(sy),
                                "iris_h": float(h),
                                "iris_v": float(v),
                                "inside_box": bool(inside),
                                "distance_to_box_px": dist_box_px,
                                "distance_to_box_center_px": dist_center_px,
                                "eye_open_ratio": float(eye_open),
                                "head_motion_norm": float(head_motion),
                                "iris_step": float(iris_step),
                            }
                        )
                    elif not quality_ok:
                        rejected_quality += 1

                    color = (0, 220, 0) if inside else (0, 255, 255)
                    cv2.circle(disp, (int(sx), int(sy)), 10, color, -1)

                    hit_rate = (
                        sum(1 for s in samples if s["inside_box"]) / float(len(samples))
                        if samples
                        else 0.0
                    )
                    cv2.putText(
                        disp,
                        f"Live hit-rate: {hit_rate * 100.0:0.1f}%",
                        (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (110, 245, 140),
                        2,
                    )
                    phase = "seek fixation"
                    if settled_since is not None:
                        phase = "dwell"
                    if capture_started_at is not None:
                        phase = "capture"
                    cv2.putText(
                        disp,
                        f"Phase: {phase}",
                        (40, 275),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (220, 220, 220),
                        2,
                    )
                    cv2.putText(
                        disp,
                        f"Quality: {'OK' if quality_ok else 'REJECT'} stable={stable_frames}/{STABLE_MIN_FRAMES} eye={eye_open:.3f} head={head_motion:.3f}",
                        (40, 310),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.56,
                        (120, 240, 120) if quality_ok else (90, 150, 255),
                        2,
                    )

                    if capture_started_at is not None and (elapsed - capture_started_at) >= TARGET_CAPTURE_SECONDS:
                        break

                cv2.imshow(win, disp)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            valid = len(samples)
            inside_count = sum(1 for s in samples if s["inside_box"])
            hit_rate = float(inside_count) / float(valid) if valid else 0.0
            mean_dist_box = float(np.mean([s["distance_to_box_px"] for s in samples])) if samples else 0.0
            mean_dist_center = float(np.mean([s["distance_to_box_center_px"] for s in samples])) if samples else 0.0
            h_std = float(np.std([s["iris_h"] for s in samples])) if samples else None
            v_std = float(np.std([s["iris_v"] for s in samples])) if samples else None
            needs_retry = (
                (valid < TARGET_MIN_SAMPLES)
                or (h_std is not None and h_std > TARGET_STD_MAX_H)
                or (v_std is not None and v_std > TARGET_STD_MAX_V)
            )
            if needs_retry and int(target["retry_index"]) < TARGET_MAX_RETRIES:
                retry_target = dict(target)
                retry_target["retry_index"] = int(target["retry_index"]) + 1
                target_queue.append(retry_target)
                print(
                    f"Queued retry for {target['name']} (retry {retry_target['retry_index']}) "
                    f"because samples={valid} h_std={h_std if h_std is not None else float('nan'):.4f} "
                    f"v_std={v_std if v_std is not None else float('nan'):.4f}"
                )

            all_iterations.append(
                {
                    "iteration_index": iteration_index,
                    "target_name": target["name"],
                    "phase": target["phase"],
                    "retry_index": int(target["retry_index"]),
                    "target_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "summary": {
                        "valid_samples": valid,
                        "inside_count": inside_count,
                        "hit_rate": hit_rate,
                        "mean_distance_to_box_px": mean_dist_box,
                        "mean_distance_to_box_center_px": mean_dist_center,
                        "quality_rejects": int(rejected_quality),
                        "capture_started": bool(capture_started_at is not None),
                        "stable_min_frames": int(STABLE_MIN_FRAMES),
                        "h_std": h_std,
                        "v_std": v_std,
                        "retry_queued": bool(needs_retry and int(target["retry_index"]) < TARGET_MAX_RETRIES),
                    },
                    "samples": samples,
                }
            )

            print(
                f"Iteration {iteration_index}: "
                f"samples={valid} hit_rate={hit_rate:.3f} "
                f"mean_dist_box_px={mean_dist_box:.1f} "
                f"mean_dist_center_px={mean_dist_center:.1f} "
                f"quality_rejects={rejected_quality} "
                f"h_std={(h_std if h_std is not None else float('nan')):.4f} "
                f"v_std={(v_std if v_std is not None else float('nan')):.4f}"
            )

        quality_report = build_quality_report(all_iterations, sw, sh)
        payload = {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "screen": {"width": sw, "height": sh},
            "iteration_timeout_seconds": ITERATION_TIMEOUT_SECONDS,
            "target_dwell_seconds": TARGET_DWELL_SECONDS,
            "target_capture_seconds": TARGET_CAPTURE_SECONDS,
            "target_settle_seconds": TARGET_SETTLE_SECONDS,
            "stable_min_frames": STABLE_MIN_FRAMES,
            "total_iterations_planned": total_planned,
            "total_iterations_executed": len(all_iterations),
            "auto_flip_y_selected": active_flip_y,
            "auto_flip_y_locked": flip_eval_locked,
            "auto_flip_eval_samples": flip_eval_samples,
            "auto_flip_mean_center_distance_px": {
                "flip_y_false": (flip_eval_total_dist[False] / float(flip_eval_samples)) if flip_eval_samples else None,
                "flip_y_true": (flip_eval_total_dist[True] / float(flip_eval_samples)) if flip_eval_samples else None,
            },
            "quality_report": quality_report,
            "iterations": all_iterations,
        }

        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUT_DIR, f"iteration_{stamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        fitted_settings = derive_mapper_settings(payload)
        if fitted_settings is not None:
            write_mapper_settings(fitted_settings)
            print(f"Saved calibrated mapper settings to: {SETTINGS_FILE}")
            print(
                "Calibrated bounds "
                f"x=[{fitted_settings['x_min']:.3f}, {fitted_settings['x_max']:.3f}] "
                f"y=[{fitted_settings['y_min']:.3f}, {fitted_settings['y_max']:.3f}] "
                f"flip_x={fitted_settings['flip_x']} flip_y={fitted_settings['flip_y']}"
            )
        else:
            print("Calibration run completed, but not enough valid samples to fit mapper settings.")

        print(f"Saved calibration iteration data to: {out_path}")
        print(
            "Calibration quality report: "
            f"axis_score={quality_report['axis_score']:.3f} "
            f"x_corr={quality_report['x_corr_abs']:.3f} "
            f"y_corr={quality_report['y_corr_abs']:.3f} "
            f"y_cross={quality_report['y_cross_abs']:.3f} "
            f"v_span={quality_report['v_span']:.4f}"
        )
        print(f"Iterations saved: {len(all_iterations)} (planned {total_planned})")
    finally:
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
