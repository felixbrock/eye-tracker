# Eye Tracker User Guide

## Goal

Get the gaze pointer aligned to where you are looking.

## Start

Run:

```bash
uv run python eye_tracker.py
```

## Mandatory calibration flow

1. Close the tracker app if it is running.
2. Run:
```bash
uv run python calibration.py
```
3. Sit normally and look at each target box center as it appears.
4. Keep your head mostly still during each iteration.
5. Confirm terminal prints `Saved calibration iteration data`.
6. Start the tracker with `uv run python eye_tracker.py`.

## After calibration

- Move your eyes around the screen and verify the red gaze overlay follows.
- Press `Enter` to click where the overlay is.

## If tracking is inaccurate

- Press `r` to reset and calibrate again with `uv run python calibration.py`.
- Improve lighting on your face.
- Recenter the webcam at top-middle of monitor.
- Increase/decrease sensitivity with `S`/`s`.
- Increase/decrease smoothing with `+`/`-`.

## Controls reference

- `r` reset calibration
- `d` toggle debug view
- `s` / `S` sensitivity down/up
- `+` / `-` smoothing up/down
- `Enter` click
- `q` or `Esc` quit
