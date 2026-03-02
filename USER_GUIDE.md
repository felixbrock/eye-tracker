# Eye Tracker User Guide

## Goal

Get the gaze pointer aligned to where you are looking.

## Start

Run:

```bash
uv run python eye_tracker.py
```

## Mandatory calibration flow

1. Sit normally and look at center of screen.
2. Press `c`.
3. Look directly at each calibration dot until it completes:
   - Top-left
   - Top-right
   - Bottom-left
   - Bottom-right
4. Keep your head mostly still during each point.
5. Confirm you see `Calibration saved!`.

## After calibration

- Move your eyes around the screen and verify the red gaze overlay follows.
- Press `Enter` to click where the overlay is.

## If tracking is inaccurate

- Press `r` to reset and calibrate again with `c`.
- Improve lighting on your face.
- Recenter the webcam at top-middle of monitor.
- Increase/decrease sensitivity with `S`/`s`.
- Increase/decrease smoothing with `+`/`-`.

## Controls reference

- `c` calibrate
- `r` reset calibration
- `d` toggle debug view
- `s` / `S` sensitivity down/up
- `+` / `-` smoothing up/down
- `Enter` click
- `q` or `Esc` quit
