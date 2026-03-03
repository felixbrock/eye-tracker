#!/usr/bin/env python3
"""Shared model utilities for tab-focused gaze classification."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np


TAB_SETTINGS_FILE = os.path.expanduser("~/.config/gaze_tab_settings.json")


def fit_tab_model(samples_by_tab):
    """Fit centroid + pooled covariance model from per-tab iris samples."""
    centers = {}
    all_points = []
    total_samples = 0
    for tab_id, samples in samples_by_tab.items():
        pts = np.asarray(samples, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 6:
            continue
        center = np.median(pts, axis=0)
        centers[int(tab_id)] = [float(center[0]), float(center[1])]
        all_points.append(pts)
        total_samples += int(len(pts))

    if len(centers) < 2:
        return None

    joined = np.vstack(all_points)
    cov = np.cov(joined.T) + np.eye(2) * 1e-4
    inv_cov = np.linalg.pinv(cov)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tab_count": len(centers),
        "total_samples": total_samples,
        "centers": {str(k): v for k, v in sorted(centers.items())},
        "inv_cov": inv_cov.tolist(),
        "cov": cov.tolist(),
    }


def predict_tab(model, h, v):
    """Return (tab_id, confidence, score_map)."""
    centers = model.get("centers", {})
    if not centers:
        return None, 0.0, {}
    x = np.asarray([float(h), float(v)], dtype=float)
    inv_cov = np.asarray(model.get("inv_cov"), dtype=float)
    if inv_cov.shape != (2, 2):
        inv_cov = np.eye(2, dtype=float)

    scores = {}
    for tab_id, center in centers.items():
        c = np.asarray(center, dtype=float)
        d = x - c
        m2 = float(d.T @ inv_cov @ d)
        scores[int(tab_id)] = m2

    best_tab = min(scores.keys(), key=lambda k: scores[k])
    sorted_scores = sorted(scores.values())
    if len(sorted_scores) >= 2:
        margin = sorted_scores[1] - sorted_scores[0]
    else:
        margin = 0.0
    confidence = float(1.0 - np.exp(-max(0.0, margin)))
    return best_tab, float(np.clip(confidence, 0.0, 1.0)), scores


def save_tab_model(model, path=TAB_SETTINGS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_tab_model(path=TAB_SETTINGS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tab_rectangles(screen_w, top_h, tab_count, margin=24, gap=8):
    usable_w = max(1, screen_w - 2 * margin - gap * (tab_count - 1))
    tab_w = max(60, usable_w // max(tab_count, 1))
    rects = []
    x = margin
    for idx in range(tab_count):
        x1 = x
        x2 = min(screen_w - margin, x1 + tab_w)
        rects.append((x1, 18, x2, 18 + top_h))
        x = x2 + gap
    return rects
