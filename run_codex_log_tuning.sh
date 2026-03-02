#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_codex_log_tuning.sh --log-file <path> [--codex-cmd <cmd>] [--history-count <n>] [--dry-run]

Description:
  Builds a prompt embedding the given log file and starts a new Codex CLI session
  to analyze poor eye-tracker performance and adjust the tracker logic accordingly.
  After the agent run, this script auto-commits any repo changes with a context
  marker ("codex context" or "claude code context") in the commit message.

Options:
  --log-file   Path to calibration log JSON (required)
  --codex-cmd  Codex launcher command (default: codex)
  --history-count Number of recent relevant commits to include as context (default: 12)
  --dry-run    Print generated prompt and exit (do not launch Codex)
  -h, --help   Show this help
EOF
}

log_file=""
codex_cmd="codex"
history_count=12
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
    --history-count)
      [[ $# -ge 2 ]] || { echo "ERROR: --history-count requires a value" >&2; exit 2; }
      [[ "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: --history-count must be a non-negative integer" >&2; exit 2; }
      history_count="$2"
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
tmp_prompt="$(mktemp)"
tmp_history="$(mktemp)"

cleanup() {
  rm -f "$tmp_prompt"
  rm -f "$tmp_history"
}
trap cleanup EXIT

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

context_desc="codex context"
case "$(basename "$codex_cmd" | tr '[:upper:]' '[:lower:]')" in
  *claude*)
    context_desc="claude code context"
    ;;
esac

git add -A
git commit -m "run log tuning (${context_desc})" \
  -m "Context: ${context_desc}" \
  -m "Log file: ${abs_log_file}"

echo "Committed run-log changes with ${context_desc}."

git push
