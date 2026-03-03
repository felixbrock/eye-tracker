#!/usr/bin/env python3
"""Tab-focused calibration/training data collection and model fitting."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from screeninfo import get_monitors

from eye_tracker import (
    EyeTracker,
    L_BOTTOM,
    L_OUTER,
    L_TOP,
    R_BOTTOM,
    R_OUTER,
    R_TOP,
    WEBCAM_H,
    WEBCAM_INDEX,
    WEBCAM_W,
)
from tab_model import TAB_SETTINGS_FILE, fit_tab_model, predict_tab, save_tab_model, tab_rectangles


OUT_DIR = "calibration_logs"
TARGET_TOP_H = 72
DEFAULT_TABS = 8
DEFAULT_ROUNDS = 2
TARGET_CAPTURE_SECONDS = 1.1
TARGET_DWELL_SECONDS = 0.25
TARGET_TIMEOUT_SECONDS = 6.0
MIN_EYE_OPEN_RATIO = 0.075
MAX_IRIS_STEP = 0.090
MAX_HEAD_MOTION = 0.25


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


def _draw_tabs(canvas, rects, target_idx):
    for idx, (x1, y1, x2, y2) in enumerate(rects):
        active = idx == target_idx
        color = (40, 140, 240) if active else (70, 70, 70)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (240, 240, 240), 1)
        label = f"Tab {idx + 1}"
        cv2.putText(
            canvas,
            label,
            (x1 + 10, y1 + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (250, 250, 250),
            2,
        )


def _fit_from_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    samples_by_tab = defaultdict(list)
    for trial in payload.get("trials", []):
        tab_idx = int(trial["tab_index"])
        for s in trial.get("samples", []):
            samples_by_tab[tab_idx].append([float(s["iris_h"]), float(s["iris_v"])])
    model = fit_tab_model(samples_by_tab)
    if model is None:
        raise RuntimeError("not enough valid samples in log to fit tab model")
    save_tab_model(model)
    print(f"Saved tab model to: {TAB_SETTINGS_FILE}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabs", type=int, default=DEFAULT_TABS)
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.fit_only:
        if not args.log_file:
            raise SystemExit("--fit-only requires --log-file")
        _fit_from_log(args.log_file)
        return

    tab_count = max(2, min(20, int(args.tabs)))
    rounds = max(1, min(8, int(args.rounds)))

    monitor = get_monitors()[0]
    sw, sh = monitor.width, monitor.height
    rects = tab_rectangles(sw, TARGET_TOP_H, tab_count)
    trial_order = []
    for _ in range(rounds):
        arr = list(range(tab_count))
        random.shuffle(arr)
        trial_order.extend(arr)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    tracker = EyeTracker()
    win = "Tab Calibration"
    trials = []
    samples_by_tab = defaultdict(list)
    prev_h = None
    prev_v = None

    try:
        tracker.reset_head_anchor()
        for trial_idx, tab_idx in enumerate(trial_order, start=1):
            samples = []
            quality_rejects = 0
            settled_since = None
            capture_started = None
            t0 = time.time()

            while True:
                elapsed = time.time() - t0
                if elapsed >= TARGET_TIMEOUT_SECONDS:
                    break
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("Cancelled.")
                    return
                ret, frame = cap.read()
                if not ret:
                    continue
                result = tracker.process(frame, apply_head_comp=True, return_meta=True)

                disp = np.zeros((sh, sw, 3), dtype=np.uint8)
                _draw_tabs(disp, rects, tab_idx)
                cv2.putText(
                    disp,
                    f"Look at browser Tab {tab_idx + 1}",
                    (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    disp,
                    f"Trial {trial_idx}/{len(trial_order)}",
                    (40, 178),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (220, 220, 220),
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
                        and head_motion <= MAX_HEAD_MOTION
                        and iris_step <= MAX_IRIS_STEP
                    )
                    if quality_ok:
                        if settled_since is None:
                            settled_since = elapsed
                    else:
                        settled_since = None
                        quality_rejects += 1

                    if (
                        capture_started is None
                        and settled_since is not None
                        and (elapsed - settled_since) >= TARGET_DWELL_SECONDS
                    ):
                        capture_started = elapsed

                    if capture_started is not None and quality_ok:
                        samples.append(
                            {
                                "t": float(elapsed),
                                "iris_h": float(h),
                                "iris_v": float(v),
                                "eye_open_ratio": float(eye_open),
                                "head_motion_norm": float(head_motion),
                            }
                        )

                    cv2.putText(
                        disp,
                        f"Samples: {len(samples)}  Rejects: {quality_rejects}",
                        (40, 214),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (180, 255, 180),
                        2,
                    )
                    if capture_started is not None and (elapsed - capture_started) >= TARGET_CAPTURE_SECONDS:
                        break

                cv2.imshow(win, disp)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            for s in samples:
                samples_by_tab[tab_idx].append([s["iris_h"], s["iris_v"]])
            trials.append(
                {
                    "trial_index": trial_idx,
                    "tab_index": int(tab_idx),
                    "samples": samples,
                    "summary": {
                        "samples": len(samples),
                        "quality_rejects": int(quality_rejects),
                    },
                }
            )
            print(f"Trial {trial_idx}/{len(trial_order)} tab={tab_idx + 1} samples={len(samples)} rejects={quality_rejects}")

        model = fit_tab_model(samples_by_tab)
        if model is None:
            print("Not enough valid data to fit tab model.")
            return
        save_tab_model(model)

        # quick in-sample score for operator feedback
        correct = 0
        total = 0
        for tab_idx, pts in samples_by_tab.items():
            for h, v in pts:
                pred, _, _ = predict_tab(model, h, v)
                correct += int(pred == tab_idx)
                total += 1
        train_acc = float(correct) / float(total) if total else 0.0

        payload = {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "screen": {"width": sw, "height": sh},
            "tab_count": tab_count,
            "rounds": rounds,
            "target_capture_seconds": TARGET_CAPTURE_SECONDS,
            "target_dwell_seconds": TARGET_DWELL_SECONDS,
            "target_timeout_seconds": TARGET_TIMEOUT_SECONDS,
            "model": model,
            "train_accuracy_in_sample": train_acc,
            "trials": trials,
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUT_DIR, f"tab_calibration_{stamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved tab calibration log: {out_path}")
        print(f"Saved tab model: {TAB_SETTINGS_FILE}")
        print(f"In-sample training accuracy: {train_acc * 100.0:.1f}% ({correct}/{total})")
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
