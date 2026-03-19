#!/usr/bin/env bash
set -euo pipefail

ITERATIONS=""
MAX_CONSECUTIVE_NONE=""
REPO=""
REPO_URL=""
REF=""
TARGET_WORKDIR=""
TARGET_REPORT_ROOT=""
MODEL=""
LAST_ITERATION_OUTPUT_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROMPT_PATH="${REPO_ROOT}/ai/agentic-bug-sweep.md"
SCHEMA_PATH="${REPO_ROOT}/ai/agentic-bug-sweep.schema.json"
STATE_ROOT="${REPO_ROOT}/target/agentic-bug-sweep"
REPORT_ROOT="${REPO_ROOT}/docs/test-reports/agentic-bug-sweep"
LOCK_PATH="${STATE_ROOT}/lock"

usage() {
  cat <<'EOF'
Usage: bash scripts/agentic-bug-sweep.sh [options]

Options:
  --iterations N              Maximum number of Codex iterations to run
  --max-consecutive-none N    Stop after N consecutive `none` results
  --repo OWNER/REPO           Target GitHub repository slug
  --repo-url URL              Remote repository URL to clone and analyze
  --ref REF                   Git ref to clone when using --repo-url
  --workdir PATH              Target repository working directory
  --model MODEL               Optional model override for `codex exec`
  --help                      Show this help text
EOF
}

log() {
  printf '%s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing required command: $1"
    exit 1
  fi
}

ensure_inputs() {
  if [[ -z "$ITERATIONS" || -z "$MAX_CONSECUTIVE_NONE" ]]; then
    log "missing required arguments"
    usage
    exit 1
  fi
  if [[ -z "$TARGET_WORKDIR" && -z "$REPO_URL" ]]; then
    log "either --workdir or --repo-url is required"
    usage
    exit 1
  fi
  if [[ -z "$REPO" && -z "$REPO_URL" ]]; then
    log "either --repo or --repo-url is required"
    usage
    exit 1
  fi
  if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || ! [[ "$MAX_CONSECUTIVE_NONE" =~ ^[0-9]+$ ]]; then
    log "--iterations and --max-consecutive-none must be non-negative integers"
    exit 1
  fi
  if [[ "$ITERATIONS" -eq 0 ]]; then
    log "--iterations must be greater than 0"
    exit 1
  fi
}

ensure_paths() {
  if [[ ! -f "$PROMPT_PATH" ]]; then
    log "missing prompt file: $PROMPT_PATH"
    exit 1
  fi
  if [[ ! -f "$SCHEMA_PATH" ]]; then
    log "missing schema file: $SCHEMA_PATH"
    exit 1
  fi
  if [[ ! -d "$TARGET_WORKDIR" ]]; then
    log "target workdir does not exist: $TARGET_WORKDIR"
    exit 1
  fi
  TARGET_REPORT_ROOT="${TARGET_WORKDIR}/docs/test-reports/agentic-bug-sweep"
  mkdir -p "$TARGET_REPORT_ROOT"
}

ensure_tools() {
  require_command codex
  require_command gh
  require_command python3
  if [[ -n "$REPO_URL" ]]; then
    require_command git
  fi
  gh auth status >/dev/null 2>&1
}

prepare_state_dirs() {
  mkdir -p "${STATE_ROOT}/context"
  mkdir -p "${STATE_ROOT}/output"
  mkdir -p "${STATE_ROOT}/repos"
  mkdir -p "${STATE_ROOT}/tmp"
  mkdir -p "$REPORT_ROOT"
}

release_lock() {
  rmdir "$LOCK_PATH" >/dev/null 2>&1 || true
}

acquire_lock() {
  if ! mkdir "$LOCK_PATH" >/dev/null 2>&1; then
    log "failed to acquire lock: $LOCK_PATH"
    exit 1
  fi
  trap release_lock EXIT
}

capture_open_issues() {
  gh issue list \
    --repo "$REPO" \
    --state open \
    --label bug \
    --limit 200 \
    --json number,title,body,labels,url >"${STATE_ROOT}/context/open-issues.json"
}

capture_prior_reports() {
  find "$TARGET_REPORT_ROOT" -maxdepth 1 -type f -name '*.md' | sort >"${STATE_ROOT}/context/prior-reports.txt"
}

parse_repo_slug_from_url() {
  python3 - "$1" <<'PY'
import re
import sys
from urllib.parse import urlparse

url = sys.argv[1]
if "://" in url:
    parsed = urlparse(url)
    path = parsed.path
else:
    match = re.match(r"^[^@]+@[^:]+:(.+)$", url)
    if not match:
        raise SystemExit(f"unsupported repo url: {url}")
    path = "/" + match.group(1)

parts = [part for part in path.split("/") if part]
if len(parts) < 2:
    raise SystemExit(f"unsupported repo url: {url}")
owner, repo = parts[-2], parts[-1]
if repo.endswith(".git"):
    repo = repo[:-4]
print(f"{owner}/{repo}")
PY
}

resolve_target_repo() {
  local clone_dir
  local -a clone_args

  if [[ -z "$REPO" && -n "$REPO_URL" ]]; then
    REPO="$(parse_repo_slug_from_url "$REPO_URL")"
  fi

  if [[ -n "$TARGET_WORKDIR" ]]; then
    TARGET_WORKDIR="$(cd "$TARGET_WORKDIR" && pwd)"
    return
  fi

  clone_dir="${STATE_ROOT}/repos/${REPO//\//-}"
  rm -rf "$clone_dir"
  clone_args=(clone --depth 1)
  if [[ -n "$REF" ]]; then
    clone_args+=(--branch "$REF")
  fi
  clone_args+=("$REPO_URL" "$clone_dir")

  if ! git "${clone_args[@]}"; then
    log "failed to clone repository: $REPO_URL"
    exit 1
  fi
  TARGET_WORKDIR="$clone_dir"
}

validate_json_file() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    json.load(handle)
PY
}

json_get_string() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
for part in key.split("."):
    data = data[part]
if isinstance(data, (list, dict)):
    raise SystemExit(f"{key} does not resolve to a scalar")
print(data)
PY
}

json_get_lines() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
for part in key.split("."):
    data = data[part]
if not isinstance(data, list):
    raise SystemExit(f"{key} is not a list")
for item in data:
    print(item)
PY
}

json_has_value() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
try:
    for part in key.split("."):
        data = data[part]
except (KeyError, TypeError):
    raise SystemExit(1)
if data in (None, "", []):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

write_text_with_related() {
  python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

path, key, dest = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
value = data
for part in key.split("."):
    value = value[part]
related = data.get("related_issue_numbers", [])

with open(dest, "w", encoding="utf-8") as handle:
    handle.write(value)
    if related:
        if not value.endswith("\n"):
            handle.write("\n")
        handle.write("\n## Related issues\n\n")
        for issue_number in related:
            handle.write(f"- #{issue_number}\n")
PY
}

write_text_exact() {
  python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

path, key, dest = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
value = data
for part in key.split("."):
    value = value[part]
with open(dest, "w", encoding="utf-8") as handle:
    handle.write(value)
PY
}

create_issue() {
  local result_path="$1"
  local title
  local body_file
  local -a labels
  local -a create_args

  title="$(json_get_string "$result_path" "issue.title")"
  body_file="$(mktemp "${STATE_ROOT}/output/create-body.XXXXXX.md")"
  write_text_with_related "$result_path" "issue.body" "$body_file"
  mapfile -t labels < <(json_get_lines "$result_path" "issue.labels")

  create_args=(issue create --repo "$REPO" --title "$title" --body-file "$body_file")
  for label in "${labels[@]}"; do
    create_args+=(--label "$label")
  done
  if ! gh "${create_args[@]}" >/dev/null; then
    fail_run "failed_github_mutation" "failed to create issue"
  fi
}

update_issue() {
  local result_path="$1"
  local issue_number
  local comment_file

  issue_number="$(json_get_string "$result_path" "canonical_issue_number")"
  comment_file="$(mktemp "${STATE_ROOT}/output/update-comment.XXXXXX.md")"
  write_text_with_related "$result_path" "issue_comment" "$comment_file"
  if ! gh issue comment "$issue_number" --repo "$REPO" --body-file "$comment_file" >/dev/null; then
    fail_run "failed_github_mutation" "failed to comment on issue ${issue_number}"
  fi
}

merge_issue() {
  local result_path="$1"
  local canonical_issue_number
  local canonical_comment_file
  local duplicate_comment_file
  local duplicate_issue_number

  canonical_issue_number="$(json_get_string "$result_path" "canonical_issue_number")"
  canonical_comment_file="$(mktemp "${STATE_ROOT}/output/merge-canonical-comment.XXXXXX.md")"
  duplicate_comment_file="$(mktemp "${STATE_ROOT}/output/merge-duplicate-comment.XXXXXX.md")"

  write_text_with_related "$result_path" "issue_comment" "$canonical_comment_file"
  write_text_exact "$result_path" "duplicate_comment" "$duplicate_comment_file"
  if ! gh issue comment "$canonical_issue_number" --repo "$REPO" --body-file "$canonical_comment_file" >/dev/null; then
    fail_run "failed_github_mutation" "failed to comment on canonical issue ${canonical_issue_number}"
  fi

  while IFS= read -r duplicate_issue_number; do
    if ! gh issue comment "$duplicate_issue_number" --repo "$REPO" --body-file "$duplicate_comment_file" >/dev/null; then
      fail_run "failed_github_mutation" "failed to comment on duplicate issue ${duplicate_issue_number}"
    fi
    if ! gh issue close "$duplicate_issue_number" --repo "$REPO" --reason not planned >/dev/null; then
      fail_run "failed_github_mutation" "failed to close duplicate issue ${duplicate_issue_number}"
    fi
  done < <(json_get_lines "$result_path" "duplicates_to_close")
}

apply_action() {
  local result_path="$1"
  local action="$2"
  case "$action" in
    create)
      create_issue "$result_path"
      ;;
    update)
      update_issue "$result_path"
      ;;
    merge)
      merge_issue "$result_path"
      ;;
    none)
      ;;
    *)
      log "unsupported action: $action"
      exit 1
      ;;
  esac
}

write_run_summary() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json
import sys

summary_path, iterations_run, consecutive_none_count, stop_reason = sys.argv[1:]
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "iterations_run": int(iterations_run),
            "consecutive_none_count": int(consecutive_none_count),
            "stop_reason": stop_reason,
        },
        handle,
        indent=2,
    )
PY
}

fail_run() {
  local stop_reason="$1"
  local message="$2"

  write_run_summary \
    "${STATE_ROOT}/output/run-summary.json" \
    "$iteration_number" \
    "$consecutive_none_count" \
    "$stop_reason"
  log "$message"
  exit 1
}

validate_result_contract() {
  local result_path="$1"
  local action

  action="$(json_get_string "$result_path" "action")"
  case "$action" in
    create)
      json_has_value "$result_path" "issue.title" || fail_run "failed_invalid_contract" "missing issue.title for create action"
      json_has_value "$result_path" "issue.body" || fail_run "failed_invalid_contract" "missing issue.body for create action"
      json_has_value "$result_path" "issue.labels" || fail_run "failed_invalid_contract" "missing issue.labels for create action"
      ;;
    update)
      json_has_value "$result_path" "canonical_issue_number" || fail_run "failed_invalid_contract" "missing canonical_issue_number for update action"
      json_has_value "$result_path" "issue_comment" || fail_run "failed_invalid_contract" "missing issue_comment for update action"
      ;;
    merge)
      json_has_value "$result_path" "canonical_issue_number" || fail_run "failed_invalid_contract" "missing canonical_issue_number for merge action"
      json_has_value "$result_path" "issue_comment" || fail_run "failed_invalid_contract" "missing issue_comment for merge action"
      json_has_value "$result_path" "duplicates_to_close" || fail_run "failed_invalid_contract" "missing duplicates_to_close for merge action"
      json_has_value "$result_path" "duplicate_comment" || fail_run "failed_invalid_contract" "missing duplicate_comment for merge action"
      ;;
    none)
      ;;
    *)
      fail_run "failed_invalid_contract" "unsupported action returned by codex: ${action}"
      ;;
  esac

  if json_has_value "$result_path" "related_issue_numbers"; then
    json_has_value "$result_path" "related_comment" || fail_run "failed_invalid_contract" "missing related_comment for related issues"
  fi
}

run_iteration() {
  local iteration_number="$1"
  local iteration_tag
  local output_path
  local prompt_text
  local -a codex_args

  printf -v iteration_tag '%03d' "$iteration_number"
  output_path="${STATE_ROOT}/output/iteration-${iteration_tag}.json"
  prompt_text="$(cat "$PROMPT_PATH")

Target repository: ${REPO}
Target workdir: ${TARGET_WORKDIR}
Open issues JSON: ${STATE_ROOT}/context/open-issues.json
Prior bug-sweep report index: ${STATE_ROOT}/context/prior-reports.txt
Target report root: ${TARGET_REPORT_ROOT}"

  codex_args=(exec --cd "$TARGET_WORKDIR" --sandbox workspace-write --output-schema "$SCHEMA_PATH" -o "$output_path")
  if [[ -n "$MODEL" ]]; then
    codex_args+=(--model "$MODEL")
  fi
  codex_args+=("$prompt_text")

  if ! TMPDIR="${STATE_ROOT}/tmp" codex "${codex_args[@]}"; then
    fail_run "failed_codex_exec" "codex exec failed on iteration ${iteration_number}"
  fi
  if ! validate_json_file "$output_path"; then
    fail_run "failed_invalid_json" "invalid JSON returned by codex on iteration ${iteration_number}"
  fi
  validate_result_contract "$output_path"
  LAST_ITERATION_OUTPUT_PATH="$output_path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --max-consecutive-none)
      MAX_CONSECUTIVE_NONE="$2"
      shift 2
      ;;
    --repo)
      REPO="$2"
      shift 2
      ;;
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --workdir)
      TARGET_WORKDIR="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      log "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

ensure_inputs
ensure_tools
prepare_state_dirs
iteration_number=0
consecutive_none_count=0
stop_reason=""
acquire_lock
resolve_target_repo
ensure_paths

while (( iteration_number < ITERATIONS )); do
  iteration_number=$((iteration_number + 1))
  capture_open_issues
  capture_prior_reports

  run_iteration "$iteration_number"
  iteration_output_path="$LAST_ITERATION_OUTPUT_PATH"
  iteration_action="$(json_get_string "$iteration_output_path" "action")"
  apply_action "$iteration_output_path" "$iteration_action"

  if [[ "$iteration_action" == "none" ]]; then
    consecutive_none_count=$((consecutive_none_count + 1))
    if (( MAX_CONSECUTIVE_NONE > 0 && consecutive_none_count >= MAX_CONSECUTIVE_NONE )); then
      stop_reason="completed_consecutive_none_threshold"
      break
    fi
  else
    consecutive_none_count=0
  fi
done

if [[ -z "$stop_reason" ]]; then
  stop_reason="completed_max_iterations"
fi

write_run_summary \
  "${STATE_ROOT}/output/run-summary.json" \
  "$iteration_number" \
  "$consecutive_none_count" \
  "$stop_reason"

log "agentic bug sweep iteration completed"
