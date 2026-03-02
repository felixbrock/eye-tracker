# Eye Tracker

A webcam-based eye gaze tracker that shows a moving on-screen crosshair and lets you click with `Enter`.

For a short operator-focused walkthrough, see [USER_GUIDE.md](./USER_GUIDE.md).

## Requirements

- Python 3.11+
- A webcam centered near the top of your display
- Linux desktop (for `Enter` click support via `xdotool`)

## Install

This project uses `uv`.

```bash
uv sync
```

Optional but recommended for `Enter` = mouse click:

```bash
sudo apt install xdotool
```

## Run

```bash
uv run python eye_tracker.py
```

## Controls

- `r`: reset calibration to defaults
- `d`: toggle debug webcam window
- `s` / `S`: sensitivity down/up
- `+` / `-`: smoothing up/down
- `Enter`: left-click at current gaze point
- `q` or `Esc`: quit

## Calibration (Important)

You should calibrate every time your camera position, seat position, or monitor setup changes.

1. Close the tracker app if it is running.
2. Run calibration:
   ```bash
   uv run python calibration.py
   ```
3. Keep your head in a natural position and follow each red target box center.
4. Wait for `Saved calibration iteration data` in terminal.
5. Start the tracker:
   ```bash
   uv run python eye_tracker.py
   ```

Calibration values are stored at:

- `~/.config/gaze_settings.json`

## How to get accurate tracking

- Keep lighting stable (avoid strong backlight).
- Keep your face fully visible to the camera.
- Mount camera at the top-center of the monitor.
- Sit at a consistent distance from the screen.
- Re-run calibration if cursor mapping feels shifted.

## Quick alignment checklist

If gaze point is off:

1. Press `r` to reset.
2. Re-run `uv run python calibration.py`.
3. Adjust sensitivity (`s`/`S`) and smoothing (`+`/`-`).
4. Confirm camera is not tilted and is centered.

## i3 window manager (optional)

For better overlay behavior, add:

```i3
for_window [title="^gaze$"] floating enable, border none, sticky enable
for_window [title="^Calibration$"] floating enable, border none
for_window [title="^Eye Tracker Debug$"] floating enable
```
