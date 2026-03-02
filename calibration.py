#!/usr/bin/env python3
"""
Single-iteration eye-tracker calibration sampler.

Runs one 5-second trial with a random target box and logs cursor behavior.
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
    WEBCAM_H,
    WEBCAM_INDEX,
    WEBCAM_W,
)


ITERATION_SECONDS = 5.0
TOTAL_ITERATIONS = 5
BOX_W_RATIO = 0.14
BOX_H_RATIO = 0.18
OUT_DIR = "calibration_logs"


def random_box(sw, sh, rng):
    bw = max(120, int(sw * BOX_W_RATIO))
    bh = max(120, int(sh * BOX_H_RATIO))

    # Include strict edge placements (including corners) so calibration
    # repeatedly exercises the full screen range, not only interior regions.
    anchors_x = [0, max(0, (sw - bw) // 2), max(0, sw - bw)]
    anchors_y = [0, max(0, (sh - bh) // 2), max(0, sh - bh)]
    edge_boxes = [(ax, ay) for ay in anchors_y for ax in anchors_x]

    # Use edge placements most of the time, with occasional random placements
    # to avoid overfitting to a fixed grid.
    if float(rng.random()) < 0.8:
        x1, y1 = edge_boxes[int(rng.integers(0, len(edge_boxes)))]
    else:
        x1 = int(rng.integers(0, max(1, sw - bw + 1)))
        y1 = int(rng.integers(0, max(1, sh - bh + 1)))

    return x1, y1, x1 + bw, y1 + bh


def distance_to_box(sx, sy, x1, y1, x2, y2):
    dx = max(x1 - sx, 0, sx - x2)
    dy = max(y1 - sy, 0, sy - y2)
    return float(np.hypot(dx, dy))


def main():
    monitor = get_monitors()[0]
    sw, sh = monitor.width, monitor.height
    rng = np.random.default_rng()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    tracker = EyeTracker()
    # Keep calibration runs independent from any manually tuned persistent settings.
    mapper = GazeMapper(sw, sh, load_saved_settings=False)

    win = "Calibration Iteration"
    all_iterations = []

    try:
        for iteration_index in range(1, TOTAL_ITERATIONS + 1):
            mapper.clear_runtime_state(reset_bias=True)
            tracker.reset_head_anchor()
            box = random_box(sw, sh, rng)
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            samples = []
            t0 = time.time()

            while True:
                elapsed = time.time() - t0
                if elapsed >= ITERATION_SECONDS:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("Cancelled.")
                    return

                ret, frame = cap.read()
                if not ret:
                    continue

                result = tracker.process(frame, apply_head_comp=True)
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
                    f"Iteration: {iteration_index}/{TOTAL_ITERATIONS}",
                    (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (210, 210, 210),
                    2,
                )
                cv2.putText(
                    disp,
                    f"Time left: {max(0.0, ITERATION_SECONDS - elapsed):0.1f}s",
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

                if result is not None:
                    (h, v), _ = result
                    sx, sy = mapper.map(h, v)
                    inside = x1 <= sx <= x2 and y1 <= sy <= y2
                    dist_box_px = distance_to_box(sx, sy, x1, y1, x2, y2)
                    dist_center_px = float(np.hypot(sx - cx, sy - cy))

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
                        }
                    )

                    color = (0, 220, 0) if inside else (0, 255, 255)
                    cv2.circle(disp, (int(sx), int(sy)), 10, color, -1)

                    hit_rate = sum(1 for s in samples if s["inside_box"]) / float(len(samples))
                    cv2.putText(
                        disp,
                        f"Live hit-rate: {hit_rate * 100.0:0.1f}%",
                        (40, 205),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (110, 245, 140),
                        2,
                    )

                cv2.imshow(win, disp)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            valid = len(samples)
            inside_count = sum(1 for s in samples if s["inside_box"])
            hit_rate = float(inside_count) / float(valid) if valid else 0.0
            mean_dist_box = float(np.mean([s["distance_to_box_px"] for s in samples])) if samples else 0.0
            mean_dist_center = float(np.mean([s["distance_to_box_center_px"] for s in samples])) if samples else 0.0

            all_iterations.append(
                {
                    "iteration_index": iteration_index,
                    "target_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "summary": {
                        "valid_samples": valid,
                        "inside_count": inside_count,
                        "hit_rate": hit_rate,
                        "mean_distance_to_box_px": mean_dist_box,
                        "mean_distance_to_box_center_px": mean_dist_center,
                    },
                    "samples": samples,
                }
            )

            print(
                f"Iteration {iteration_index}/{TOTAL_ITERATIONS}: "
                f"samples={valid} hit_rate={hit_rate:.3f} "
                f"mean_dist_box_px={mean_dist_box:.1f} "
                f"mean_dist_center_px={mean_dist_center:.1f}"
            )

        payload = {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "screen": {"width": sw, "height": sh},
            "iteration_seconds": ITERATION_SECONDS,
            "total_iterations": TOTAL_ITERATIONS,
            "iterations": all_iterations,
        }

        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUT_DIR, f"iteration_{stamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved calibration iteration data to: {out_path}")
        print(f"Iterations saved: {TOTAL_ITERATIONS}")
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
