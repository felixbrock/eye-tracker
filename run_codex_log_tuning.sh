#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_codex_log_tuning.sh --log-file <path> [--validation-log-file <path>] [--validation-runs <n>] [--codex-cmd <cmd>] [--history-count <n>] [--skip-validation] [--auto-retry|--no-auto-retry] [--max-attempts <n>] [--dry-run]

Description:
  Builds a prompt embedding the given log file and starts a new Codex CLI session
  to analyze poor eye-tracker performance and adjust the tracker logic accordingly.
  Then it validates performance using a post-change calibration run, applies
  objective acceptance gates focused on calibration-data quality, and auto-rolls
  back changes on regression.
  Passing runs are auto-committed with a context marker ("codex context" or
  "claude code context") in the commit message.

Options:
  --log-file   Path to calibration log JSON (required)
  --validation-log-file Path to post-change calibration log JSON (optional; if omitted, script runs calibration.py)
  --validation-runs Number of validation calibration runs to aggregate by median when auto-running validation (default: 2)
  --codex-cmd  Codex launcher command (default: codex)
  --history-count Number of recent relevant commits to include as context (default: 12)
  --skip-validation Skip performance gates and commit directly after syntax checks
  --auto-retry Automatically retry with the new validation log as baseline after rejection (default: on)
  --no-auto-retry Disable automatic retry after rejection
  --max-attempts Maximum total attempts including the first run (0 = unlimited, default: 0)
  --dry-run    Print generated prompt and exit (do not launch Codex)
  -h, --help   Show this help
EOF
}

log_file=""
validation_log_file=""
validation_runs=2
codex_cmd="codex"
history_count=12
skip_validation=0
dry_run=0
auto_retry=1
max_attempts=0
attempt_index=1
settings_file="${HOME}/.config/gaze_settings.json"
tmp_settings_backup=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-file)
      [[ $# -ge 2 ]] || { echo "ERROR: --log-file requires a value" >&2; exit 2; }
      log_file="$2"
      shift 2
      ;;
    --codex-cmd)
      [[ $# -ge 2 ]] || { echo "ERROR: --codex-cmd requires a value" >&2; exit 2; }
      codex_cmd="$2"
      shift 2
      ;;
    --validation-log-file)
      [[ $# -ge 2 ]] || { echo "ERROR: --validation-log-file requires a value" >&2; exit 2; }
      validation_log_file="$2"
      shift 2
      ;;
    --validation-runs)
      [[ $# -ge 2 ]] || { echo "ERROR: --validation-runs requires a value" >&2; exit 2; }
      [[ "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: --validation-runs must be a non-negative integer" >&2; exit 2; }
      validation_runs="$2"
      shift 2
      ;;
    --history-count)
      [[ $# -ge 2 ]] || { echo "ERROR: --history-count requires a value" >&2; exit 2; }
      [[ "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: --history-count must be a non-negative integer" >&2; exit 2; }
      history_count="$2"
      shift 2
      ;;
    --skip-validation)
      skip_validation=1
      shift
      ;;
    --auto-retry)
      auto_retry=1
      shift
      ;;
    --no-auto-retry)
      auto_retry=0
      shift
      ;;
    --max-attempts)
      [[ $# -ge 2 ]] || { echo "ERROR: --max-attempts requires a value" >&2; exit 2; }
      [[ "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: --max-attempts must be a non-negative integer" >&2; exit 2; }
      max_attempts="$2"
      shift 2
      ;;
    --attempt-index)
      [[ $# -ge 2 ]] || { echo "ERROR: --attempt-index requires a value" >&2; exit 2; }
      [[ "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: --attempt-index must be a non-negative integer" >&2; exit 2; }
      attempt_index="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$log_file" ]] || { echo "ERROR: --log-file is required" >&2; usage >&2; exit 2; }
[[ -f "$log_file" ]] || { echo "ERROR: log file not found: $log_file" >&2; exit 1; }
if [[ -n "$validation_log_file" ]]; then
  [[ -f "$validation_log_file" ]] || { echo "ERROR: validation log file not found: $validation_log_file" >&2; exit 1; }
fi
if [[ "$validation_runs" -lt 1 ]]; then
  echo "ERROR: --validation-runs must be >= 1" >&2
  exit 1
fi

if ! command -v "$codex_cmd" >/dev/null 2>&1; then
  echo "ERROR: codex command not found: $codex_cmd" >&2
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: must be run from inside a git repository" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: git working tree is not clean. Commit or stash existing changes first." >&2
  exit 1
fi
if [[ "$max_attempts" -gt 0 && "$attempt_index" -gt "$max_attempts" ]]; then
  echo "ERROR: attempt index (${attempt_index}) exceeded max attempts (${max_attempts})." >&2
  exit 1
fi

abs_log_file="$(realpath "$log_file")"
script_path="$(realpath "$0")"
if [[ -n "$validation_log_file" ]]; then
  abs_validation_log_file="$(realpath "$validation_log_file")"
else
  abs_validation_log_file=""
fi
tmp_prompt="$(mktemp)"
tmp_history="$(mktemp)"
tmp_baseline="$(mktemp)"
tmp_validation="$(mktemp)"
tuning_start_epoch="$(date +%s)"

cleanup() {
  rm -f "$tmp_prompt"
  rm -f "$tmp_history"
  rm -f "$tmp_baseline"
  rm -f "$tmp_validation"
  if [[ -n "${tmp_settings_backup:-}" ]]; then
    rm -f "$tmp_settings_backup"
  fi
}
trap cleanup EXIT

extract_metrics() {
  local in_file="$1"
  local out_file="$2"
  python3 - "$in_file" "$out_file" <<'PY'
import json
import math
import statistics
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]
with open(in_file, "r", encoding="utf-8") as f:
    payload = json.load(f)

iters = payload.get("iterations", [])
if not iters:
    raise SystemExit("no iterations found in calibration log")

def corr(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)

hit_rates = []
dist_box = []
dist_center = []
valid_total = 0
quality_rejects = []
point_tx = []
point_ty = []
point_h = []
point_v = []
for it in iters:
    s = it.get("summary", {})
    hit_rates.append(float(s.get("hit_rate", 0.0)))
    dist_box.append(float(s.get("mean_distance_to_box_px", 0.0)))
    dist_center.append(float(s.get("mean_distance_to_box_center_px", 0.0)))
    valid_total += int(s.get("valid_samples", 0))
    quality_rejects.append(int(s.get("quality_rejects", 0)))
    box = it.get("target_box", {})
    samples = it.get("samples", [])
    if samples and box:
        cx = (float(box["x1"]) + float(box["x2"])) * 0.5
        cy = (float(box["y1"]) + float(box["y2"])) * 0.5
        tx = max(0.0, min(1.0, cx / max(float(payload["screen"]["width"]) - 1.0, 1.0)))
        ty = max(0.0, min(1.0, cy / max(float(payload["screen"]["height"]) - 1.0, 1.0)))
        hs = [float(x["iris_h"]) for x in samples if "iris_h" in x and "iris_v" in x]
        vs = [float(x["iris_v"]) for x in samples if "iris_h" in x and "iris_v" in x]
        if hs and vs:
            point_tx.append(tx)
            point_ty.append(ty)
            point_h.append(float(statistics.median(hs)))
            point_v.append(float(statistics.median(vs)))

avg_hit = sum(hit_rates) / len(hit_rates)
avg_box = sum(dist_box) / len(dist_box)
avg_center = sum(dist_center) / len(dist_center)
avg_quality_rejects = sum(quality_rejects) / len(quality_rejects)
qr = payload.get("quality_report", {}) if isinstance(payload.get("quality_report"), dict) else {}
x_corr_abs = float(qr.get("x_corr_abs", abs(corr(point_h, point_tx))))
y_corr_abs = float(qr.get("y_corr_abs", abs(corr(point_v, point_ty))))
x_cross_abs = float(qr.get("x_cross_abs", abs(corr(point_h, point_ty))))
y_cross_abs = float(qr.get("y_cross_abs", abs(corr(point_v, point_tx))))
axis_score = float(qr.get("axis_score", x_corr_abs + y_corr_abs - x_cross_abs - y_cross_abs))
h_span = float(qr.get("h_span", (max(point_h) - min(point_h)) if point_h else 0.0))
v_span = float(qr.get("v_span", (max(point_v) - min(point_v)) if point_v else 0.0))
auto_flip_locked = bool(payload.get("auto_flip_y_locked", False))

with open(out_file, "w", encoding="utf-8") as f:
    f.write(f"avg_hit_rate={avg_hit:.6f}\n")
    f.write(f"avg_dist_box_px={avg_box:.6f}\n")
    f.write(f"avg_dist_center_px={avg_center:.6f}\n")
    f.write(f"total_valid_samples={valid_total}\n")
    f.write(f"avg_quality_rejects={avg_quality_rejects:.6f}\n")
    f.write(f"x_corr_abs={x_corr_abs:.6f}\n")
    f.write(f"y_corr_abs={y_corr_abs:.6f}\n")
    f.write(f"x_cross_abs={x_cross_abs:.6f}\n")
    f.write(f"y_cross_abs={y_cross_abs:.6f}\n")
    f.write(f"axis_score={axis_score:.6f}\n")
    f.write(f"h_span={h_span:.6f}\n")
    f.write(f"v_span={v_span:.6f}\n")
    f.write(f"auto_flip_locked={'true' if auto_flip_locked else 'false'}\n")
PY
}

rollback_repo_changes() {
  echo "Rolling back unaccepted tuning changes..."
  git restore --staged --worktree -- .
  git clean -fd -e calibration_logs/ -e calibration_logs/*
}

backup_settings_file() {
  if [[ -f "$settings_file" ]]; then
    tmp_settings_backup="$(mktemp)"
    cp -f "$settings_file" "$tmp_settings_backup"
  fi
}

restore_settings_file() {
  if [[ -n "${tmp_settings_backup:-}" && -f "$tmp_settings_backup" ]]; then
    mkdir -p "$(dirname "$settings_file")"
    cp -f "$tmp_settings_backup" "$settings_file"
    echo "Restored pre-run settings file: $settings_file"
  fi
}

write_rejection_report() {
  local report_path="$1"
  python3 - "$report_path" "$abs_log_file" "$validation_used_file" \
    "$base_hit_rate" "$base_dist_box" "$base_dist_center" "$base_axis_score" "$base_y_corr_abs" "$base_y_cross_abs" "$base_v_span" \
    "$new_hit_rate" "$new_dist_box" "$new_dist_center" "$new_axis_score" "$new_y_corr_abs" "$new_y_cross_abs" "$new_v_span" "$new_valid_samples" \
    "$hard_fail_reasons" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    report_path,
    baseline_log,
    validation_log,
    base_hit,
    base_box,
    base_center,
    base_axis_score,
    base_y_corr,
    base_y_cross,
    base_v_span,
    new_hit,
    new_box,
    new_center,
    new_axis_score,
    new_y_corr,
    new_y_cross,
    new_v_span,
    new_valid,
    reasons,
) = sys.argv[1:]

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "status": "rejected",
    "baseline_log_file": baseline_log,
    "validation_log_file": validation_log,
    "baseline_metrics": {
        "avg_hit_rate": float(base_hit),
        "avg_dist_box_px": float(base_box),
        "avg_dist_center_px": float(base_center),
        "axis_score": float(base_axis_score),
        "y_corr_abs": float(base_y_corr),
        "y_cross_abs": float(base_y_cross),
        "v_span": float(base_v_span),
    },
    "validation_metrics": {
        "avg_hit_rate": float(new_hit),
        "avg_dist_box_px": float(new_box),
        "avg_dist_center_px": float(new_center),
        "axis_score": float(new_axis_score),
        "y_corr_abs": float(new_y_corr),
        "y_cross_abs": float(new_y_cross),
        "v_span": float(new_v_span),
        "valid_samples": int(float(new_valid)),
    },
    "reasons": reasons,
}
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
}

extract_metrics "$abs_log_file" "$tmp_baseline"
source "$tmp_baseline"
base_hit_rate="$avg_hit_rate"
base_dist_box="$avg_dist_box_px"
base_dist_center="$avg_dist_center_px"
base_valid_samples="$total_valid_samples"
base_axis_score="$axis_score"
base_y_corr_abs="$y_corr_abs"
base_y_cross_abs="$y_cross_abs"
base_v_span="$v_span"
backup_settings_file

# Capture recent commit intent/history so the tuning pass can avoid
# fighting earlier adjustments and build from prior rationale.
git log --no-merges \
  --max-count "$history_count" \
  --date=iso-strict \
  --pretty=format:'---%ncommit %H%nDate: %ad%nSubject: %s%nBody:%n%b' \
  -- eye_tracker.py calibration.py run_codex_log_tuning.sh > "$tmp_history"

if [[ ! -s "$tmp_history" ]]; then
  printf 'No prior relevant commits were found for eye-tracker tuning files.\n' > "$tmp_history"
fi

{
  cat <<EOF
You are in /home/felix/repos/eye-tracker.

Task:
1. Analyze the calibration performance log below.
2. Review the previous commit history/context below, including commit body comments describing prior adjustment intent.
3. Identify how the eye tracker is performing poorly (bias, instability, lag, poor box hit-rate, distance errors, etc.) and what prior fixes have already been attempted.
4. Modify the eye-tracker logic in this repo to address remaining weaknesses while preserving or refining prior successful adjustments.
5. Run quick validation checks (at least syntax/compile checks and any cheap runtime sanity checks available).
6. Summarize exactly what changed and why.

Constraints:
- Make concrete code edits, not just recommendations.
- Prefer minimal, targeted changes in eye_tracker.py and/or calibration.py.
- Keep behavior robust for repeated calibration runs.
- Explicitly account for prior commit comments so new changes do not blindly overwrite previous tuning rationale.
- Optimize for measurable improvement against this baseline:
  - baseline_avg_hit_rate=${base_hit_rate}
  - baseline_avg_dist_to_box_px=${base_dist_box}
  - baseline_avg_dist_to_center_px=${base_dist_center}
  - baseline_valid_samples=${base_valid_samples}
  - baseline_axis_score=${base_axis_score}
  - baseline_y_corr_abs=${base_y_corr_abs}
  - baseline_y_cross_abs=${base_y_cross_abs}
  - baseline_v_span=${base_v_span}

Relevant recent git history (newest first):
EOF
  echo '```text'
  cat "$tmp_history"
  echo
  echo '```'
  cat <<EOF

Log file path: $abs_log_file
Log file content:
EOF
  echo '```json'
  cat "$abs_log_file"
  echo
  echo '```'
} > "$tmp_prompt"

if [[ "$dry_run" -eq 1 ]]; then
  cat "$tmp_prompt"
  exit 0
fi

"$codex_cmd" exec - < "$tmp_prompt"

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No repo changes detected; no commit created."
  exit 0
fi

python3 -m py_compile eye_tracker.py calibration.py

acceptance_note=""
validation_used_file=""
if [[ "$skip_validation" -eq 1 ]]; then
  acceptance_note="validation_skipped=true"
else
  if [[ -n "$abs_validation_log_file" ]]; then
    validation_used_file="$abs_validation_log_file"
    extract_metrics "$validation_used_file" "$tmp_validation"
    source "$tmp_validation"
    new_hit_rate="$avg_hit_rate"
    new_dist_box="$avg_dist_box_px"
    new_dist_center="$avg_dist_center_px"
    new_valid_samples="$total_valid_samples"
    new_axis_score="$axis_score"
    new_y_corr_abs="$y_corr_abs"
    new_y_cross_abs="$y_cross_abs"
    new_v_span="$v_span"
  else
    echo
    echo "Running post-change calibration validation (${validation_runs} runs, median aggregation)..."
    metrics_csv="$(mktemp)"
    validation_list_file="$(mktemp)"
    latest_seen_mtime=0
    for run_i in $(seq 1 "$validation_runs"); do
      echo "Validation run ${run_i}/${validation_runs}..."
      uv run python calibration.py
      latest_validation="$(ls -1t calibration_logs/iteration_*.json 2>/dev/null | head -n 1 || true)"
      [[ -n "$latest_validation" ]] || {
        rollback_repo_changes
        rm -f "$metrics_csv" "$validation_list_file"
        echo "ERROR: no validation calibration log found after running calibration.py." >&2
        exit 1
      }
      latest_validation_abs="$(realpath "$latest_validation")"
      latest_validation_mtime="$(stat -c %Y "$latest_validation_abs")"
      if [[ "$latest_validation_mtime" -lt "$tuning_start_epoch" || "$latest_validation_mtime" -lt "$latest_seen_mtime" ]]; then
        rollback_repo_changes
        rm -f "$metrics_csv" "$validation_list_file"
        echo "ERROR: validation log timestamp is invalid; cannot validate." >&2
        exit 1
      fi
      latest_seen_mtime="$latest_validation_mtime"
      echo "$latest_validation_abs" >> "$validation_list_file"
      extract_metrics "$latest_validation_abs" "$tmp_validation"
      source "$tmp_validation"
      echo "${avg_hit_rate},${avg_dist_box_px},${avg_dist_center_px},${axis_score},${y_corr_abs},${y_cross_abs},${v_span},${total_valid_samples}" >> "$metrics_csv"
    done
    validation_used_file="$(paste -sd, "$validation_list_file")"
    agg_tmp="$(mktemp)"
    python3 - "$metrics_csv" <<'PY' > "$agg_tmp"
import statistics
import sys

rows = []
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split(",")])

if not rows:
    raise SystemExit("no validation metrics rows")

cols = list(zip(*rows))
names = [
    "new_hit_rate",
    "new_dist_box",
    "new_dist_center",
    "new_axis_score",
    "new_y_corr_abs",
    "new_y_cross_abs",
    "new_v_span",
    "new_valid_samples",
]
for name, vals in zip(names, cols):
    if name == "new_valid_samples":
        print(f"{name}={int(round(statistics.median(vals)))}")
    else:
        print(f"{name}={statistics.median(vals):.6f}")
PY
    source "$agg_tmp"
    rm -f "$agg_tmp" "$metrics_csv" "$validation_list_file"
  fi

  gate_tmp="$(mktemp)"
  python3 - \
    "$base_axis_score" "$base_y_corr_abs" "$base_y_cross_abs" "$base_v_span" \
    "$new_axis_score" "$new_y_corr_abs" "$new_y_cross_abs" "$new_v_span" "$new_valid_samples" <<'PY' > "$gate_tmp"
import sys

base_axis = float(sys.argv[1])
base_y_corr = float(sys.argv[2])
base_y_cross = float(sys.argv[3])
base_v_span = float(sys.argv[4])
new_axis = float(sys.argv[5])
new_y_corr = float(sys.argv[6])
new_y_cross = float(sys.argv[7])
new_v_span = float(sys.argv[8])
new_valid = int(float(sys.argv[9]))

hard_fail = []
if new_valid < 180:
    hard_fail.append(f"not enough validation samples ({new_valid} < 180)")
if new_axis < base_axis - 0.08:
    hard_fail.append(f"axis-score regression too large ({new_axis:.3f} < {base_axis - 0.08:.3f})")
if new_y_corr < base_y_corr - 0.08:
    hard_fail.append(f"vertical-corr regression too large ({new_y_corr:.3f} < {base_y_corr - 0.08:.3f})")
if new_y_cross > base_y_cross + 0.08:
    hard_fail.append(f"vertical cross-axis leakage regression too large ({new_y_cross:.3f} > {base_y_cross + 0.08:.3f})")
if new_v_span < base_v_span * 0.85:
    hard_fail.append(f"vertical span regression too large ({new_v_span:.4f} < {base_v_span * 0.85:.4f})")

axis_improved = (new_axis >= base_axis + 0.05) or (new_axis >= base_axis * 1.08)
y_corr_improved = (new_y_corr >= base_y_corr + 0.05) or (new_y_corr >= base_y_corr * 1.08)
y_cross_improved = (new_y_cross <= base_y_cross - 0.05) or (new_y_cross <= base_y_cross * 0.90)
v_span_improved = (new_v_span >= base_v_span + 0.01) or (new_v_span >= base_v_span * 1.10)
soft_pass = sum([axis_improved, y_corr_improved, y_cross_improved, v_span_improved]) >= 2

passed = (not hard_fail) and soft_pass
print("passed=true" if passed else "passed=false")
print(f"axis_improved={str(axis_improved).lower()}")
print(f"y_corr_improved={str(y_corr_improved).lower()}")
print(f"y_cross_improved={str(y_cross_improved).lower()}")
print(f"v_span_improved={str(v_span_improved).lower()}")
if hard_fail:
    print("hard_fail_reasons=" + "; ".join(hard_fail))
else:
    print("hard_fail_reasons=")
PY
  passed="$(sed -n 's/^passed=//p' "$gate_tmp" | head -n 1)"
  axis_improved="$(sed -n 's/^axis_improved=//p' "$gate_tmp" | head -n 1)"
  y_corr_improved="$(sed -n 's/^y_corr_improved=//p' "$gate_tmp" | head -n 1)"
  y_cross_improved="$(sed -n 's/^y_cross_improved=//p' "$gate_tmp" | head -n 1)"
  v_span_improved="$(sed -n 's/^v_span_improved=//p' "$gate_tmp" | head -n 1)"
  hard_fail_reasons="$(sed -n 's/^hard_fail_reasons=//p' "$gate_tmp" | head -n 1)"
  rm -f "$gate_tmp"

  echo "Baseline metrics: axis_score=${base_axis_score}, y_corr_abs=${base_y_corr_abs}, y_cross_abs=${base_y_cross_abs}, v_span=${base_v_span}"
  echo "Validation metrics: axis_score=${new_axis_score}, y_corr_abs=${new_y_corr_abs}, y_cross_abs=${new_y_cross_abs}, v_span=${new_v_span}, valid_samples=${new_valid_samples}"
  echo "Gate checks: axis_improved=${axis_improved} y_corr_improved=${y_corr_improved} y_cross_improved=${y_cross_improved} v_span_improved=${v_span_improved}"
  if [[ "$passed" != "true" ]]; then
    rollback_repo_changes
    restore_settings_file
    reject_stamp="$(date +%Y%m%d_%H%M%S)"
    reject_report="calibration_logs/rejections/rejected_${reject_stamp}.json"
    write_rejection_report "$reject_report"
    echo "Tuning rejected by objective gates."
    if [[ -n "${hard_fail_reasons:-}" ]]; then
      echo "Reasons: ${hard_fail_reasons}"
    fi
    echo "Saved rejection report to: ${reject_report}"
    echo "Next action baseline would be: ${validation_used_file}"
    if [[ "$auto_retry" -eq 1 ]]; then
      if [[ "$max_attempts" -eq 0 || "$attempt_index" -lt "$max_attempts" ]]; then
        next_attempt=$((attempt_index + 1))
        echo "Auto-retrying tuning (attempt ${next_attempt})..."
        exec "$script_path" \
          --log-file "$validation_used_file" \
          --codex-cmd "$codex_cmd" \
          --history-count "$history_count" \
          --validation-runs "$validation_runs" \
          --auto-retry \
          --max-attempts "$max_attempts" \
          --attempt-index "$next_attempt"
      fi
      echo "Auto-retry stopped: reached max attempts (${max_attempts})."
    fi
    exit 1
  fi
  acceptance_note="validation_log=${validation_used_file}"
fi

context_desc="codex context"
case "$(basename "$codex_cmd" | tr '[:upper:]' '[:lower:]')" in
  *claude*)
    context_desc="claude code context"
    ;;
esac

git add -A
git commit -m "run log tuning (${context_desc})" \
  -m "Context: ${context_desc}" \
  -m "Log file: ${abs_log_file}" \
  -m "Baseline: axis_score=${base_axis_score} y_corr_abs=${base_y_corr_abs} y_cross_abs=${base_y_cross_abs} v_span=${base_v_span}" \
  -m "Validation: ${acceptance_note}"

echo "Committed run-log changes with ${context_desc}."

git push
