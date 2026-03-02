#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_codex_log_tuning.sh --log-file <path> [--validation-log-file <path>] [--codex-cmd <cmd>] [--history-count <n>] [--skip-validation] [--dry-run]

Description:
  Builds a prompt embedding the given log file and starts a new Codex CLI session
  to analyze poor eye-tracker performance and adjust the tracker logic accordingly.
  Then it validates performance using a post-change calibration run, applies
  objective acceptance gates, and auto-rolls back changes on regression.
  Passing runs are auto-committed with a context marker ("codex context" or
  "claude code context") in the commit message.

Options:
  --log-file   Path to calibration log JSON (required)
  --validation-log-file Path to post-change calibration log JSON (optional; if omitted, script runs calibration.py)
  --codex-cmd  Codex launcher command (default: codex)
  --history-count Number of recent relevant commits to include as context (default: 12)
  --skip-validation Skip performance gates and commit directly after syntax checks
  --dry-run    Print generated prompt and exit (do not launch Codex)
  -h, --help   Show this help
EOF
}

log_file=""
validation_log_file=""
codex_cmd="codex"
history_count=12
skip_validation=0
dry_run=0

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

abs_log_file="$(realpath "$log_file")"
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
}
trap cleanup EXIT

extract_metrics() {
  local in_file="$1"
  local out_file="$2"
  python3 - "$in_file" "$out_file" <<'PY'
import json
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]
with open(in_file, "r", encoding="utf-8") as f:
    payload = json.load(f)

iters = payload.get("iterations", [])
if not iters:
    raise SystemExit("no iterations found in calibration log")

hit_rates = []
dist_box = []
dist_center = []
valid_total = 0
for it in iters:
    s = it.get("summary", {})
    hit_rates.append(float(s.get("hit_rate", 0.0)))
    dist_box.append(float(s.get("mean_distance_to_box_px", 0.0)))
    dist_center.append(float(s.get("mean_distance_to_box_center_px", 0.0)))
    valid_total += int(s.get("valid_samples", 0))

avg_hit = sum(hit_rates) / len(hit_rates)
avg_box = sum(dist_box) / len(dist_box)
avg_center = sum(dist_center) / len(dist_center)

with open(out_file, "w", encoding="utf-8") as f:
    f.write(f"avg_hit_rate={avg_hit:.6f}\n")
    f.write(f"avg_dist_box_px={avg_box:.6f}\n")
    f.write(f"avg_dist_center_px={avg_center:.6f}\n")
    f.write(f"total_valid_samples={valid_total}\n")
PY
}

rollback_repo_changes() {
  echo "Rolling back unaccepted tuning changes..."
  git restore --staged --worktree -- .
  git clean -fd -e calibration_logs/ -e calibration_logs/*
}

extract_metrics "$abs_log_file" "$tmp_baseline"
source "$tmp_baseline"
base_hit_rate="$avg_hit_rate"
base_dist_box="$avg_dist_box_px"
base_dist_center="$avg_dist_center_px"
base_valid_samples="$total_valid_samples"

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
  else
    echo
    echo "Running post-change calibration validation..."
    uv run python calibration.py
    latest_validation="$(ls -1t calibration_logs/iteration_*.json 2>/dev/null | head -n 1 || true)"
    [[ -n "$latest_validation" ]] || {
      rollback_repo_changes
      echo "ERROR: no validation calibration log found after running calibration.py." >&2
      exit 1
    }
    latest_validation_abs="$(realpath "$latest_validation")"
    latest_validation_mtime="$(stat -c %Y "$latest_validation_abs")"
    if [[ "$latest_validation_mtime" -lt "$tuning_start_epoch" ]]; then
      rollback_repo_changes
      echo "ERROR: latest calibration log predates tuning run; cannot validate." >&2
      exit 1
    fi
    validation_used_file="$latest_validation_abs"
  fi

  extract_metrics "$validation_used_file" "$tmp_validation"
  source "$tmp_validation"
  new_hit_rate="$avg_hit_rate"
  new_dist_box="$avg_dist_box_px"
  new_dist_center="$avg_dist_center_px"
  new_valid_samples="$total_valid_samples"

  gate_tmp="$(mktemp)"
  python3 - "$base_hit_rate" "$base_dist_box" "$base_dist_center" "$new_hit_rate" "$new_dist_box" "$new_dist_center" "$new_valid_samples" <<'PY' > "$gate_tmp"
import sys

base_hit = float(sys.argv[1])
base_box = float(sys.argv[2])
base_center = float(sys.argv[3])
new_hit = float(sys.argv[4])
new_box = float(sys.argv[5])
new_center = float(sys.argv[6])
new_valid = int(float(sys.argv[7]))

hard_fail = []
if new_valid < 120:
    hard_fail.append(f"not enough validation samples ({new_valid} < 120)")
if new_hit < base_hit - 0.03:
    hard_fail.append(f"hit-rate regression too large ({new_hit:.3f} < {base_hit - 0.03:.3f})")
if new_box > base_box * 1.08:
    hard_fail.append(f"box-distance regression too large ({new_box:.1f} > {base_box * 1.08:.1f})")
if new_center > base_center * 1.08:
    hard_fail.append(f"center-distance regression too large ({new_center:.1f} > {base_center * 1.08:.1f})")

hit_improved = (new_hit >= base_hit + 0.02) or (new_hit >= base_hit * 1.12)
box_improved = (new_box <= base_box - 20.0) or (new_box <= base_box * 0.97)
center_improved = (new_center <= base_center - 25.0) or (new_center <= base_center * 0.97)
soft_pass = sum([hit_improved, box_improved, center_improved]) >= 2

passed = (not hard_fail) and soft_pass
print("passed=true" if passed else "passed=false")
print(f"hit_improved={str(hit_improved).lower()}")
print(f"box_improved={str(box_improved).lower()}")
print(f"center_improved={str(center_improved).lower()}")
if hard_fail:
    print("hard_fail_reasons=" + "; ".join(hard_fail))
else:
    print("hard_fail_reasons=")
PY
  source "$gate_tmp"
  rm -f "$gate_tmp"

  echo "Baseline metrics: hit_rate=${base_hit_rate}, dist_box_px=${base_dist_box}, dist_center_px=${base_dist_center}"
  echo "Validation metrics: hit_rate=${new_hit_rate}, dist_box_px=${new_dist_box}, dist_center_px=${new_dist_center}, valid_samples=${new_valid_samples}"
  echo "Gate checks: hit_improved=${hit_improved} box_improved=${box_improved} center_improved=${center_improved}"
  if [[ "$passed" != "true" ]]; then
    rollback_repo_changes
    echo "Tuning rejected by objective gates."
    if [[ -n "${hard_fail_reasons:-}" ]]; then
      echo "Reasons: ${hard_fail_reasons}"
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
  -m "Baseline: hit_rate=${base_hit_rate} dist_box_px=${base_dist_box} dist_center_px=${base_dist_center}" \
  -m "Validation: ${acceptance_note}"

echo "Committed run-log changes with ${context_desc}."

git push
