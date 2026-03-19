#!/usr/bin/env bash
set -euo pipefail

ORG="tensor4all"
VISIBILITY="private"
REPO_NAME=""
DESCRIPTION=""
DEST_PATH=""
ROLLBACK_ON_FAILURE=0
DRY_RUN=0
TEMPLATE_REPO="tensor4all/template-rs"

REMOTE_CREATED=0
LOCAL_CLONED=0
SETTINGS_CONFIGURED=0
ASSETS_SYNCED=0
VERIFIED=0
FULL_REPO=""

usage() {
  cat <<'EOF'
Usage: bash scripts/new-repo.sh --repo NAME --description TEXT [options]

Options:
  --repo NAME               Repository name to create
  --description TEXT        Short repository description
  --org NAME                GitHub organization or owner (default: tensor4all)
  --public                  Create the repository as public
  --private                 Create the repository as private (default)
  --internal                Create the repository as internal
  --dest PATH               Clone destination (default: ./<repo>)
  --rollback-on-failure     Delete the remote repo if bootstrap fails after creation
  --dry-run                 Perform preflight only
  --help                    Show this help text
EOF
}

log() {
  printf '%s\n' "$*"
}

fail_with_summary() {
  local status="$1"
  local message="$2"

  log "$message"
  log "bootstrap-summary: remote_created=${REMOTE_CREATED} local_cloned=${LOCAL_CLONED} settings_configured=${SETTINGS_CONFIGURED} assets_synced=${ASSETS_SYNCED} verified=${VERIFIED}"

  if [[ "$ROLLBACK_ON_FAILURE" -eq 1 && "$REMOTE_CREATED" -eq 1 ]]; then
    log "rollback: deleting $FULL_REPO"
    gh repo delete "$FULL_REPO" --yes
  fi

  exit "$status"
}

replace_template_readme_text() {
  python3 - "$1" "$2" "$3" <<'PY'
from pathlib import Path
import sys

readme_path = Path(sys.argv[1])
repo_name = sys.argv[2]
description = sys.argv[3]
text = readme_path.read_text(encoding="utf-8")
text = text.replace("# template-rs", f"# {repo_name}", 1)
text = text.replace(
    "Template repository for Rust workspace projects in the tensor4all organization.",
    description,
    1,
)
readme_path.write_text(text, encoding="utf-8")
PY
}

run_in_repo() {
  (
    cd "$DEST_PATH"
    "$@"
  )
}

preflight() {
  gh auth status >/dev/null 2>&1 || fail_with_summary 1 "gh authentication is unavailable"

  if gh repo view "$FULL_REPO" >/dev/null 2>&1; then
    fail_with_summary 1 "repository already exists: $FULL_REPO"
  fi

  if [[ -e "$DEST_PATH" ]] && [[ -n "$(find "$DEST_PATH" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
    fail_with_summary 1 "destination path is not empty: $DEST_PATH"
  fi

  gh repo view "$TEMPLATE_REPO" >/dev/null 2>&1 || fail_with_summary 1 "template repository is unavailable: $TEMPLATE_REPO"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO_NAME="$2"
      shift 2
      ;;
    --description)
      DESCRIPTION="$2"
      shift 2
      ;;
    --org)
      ORG="$2"
      shift 2
      ;;
    --public)
      VISIBILITY="public"
      shift
      ;;
    --private)
      VISIBILITY="private"
      shift
      ;;
    --internal)
      VISIBILITY="internal"
      shift
      ;;
    --dest)
      DEST_PATH="$2"
      shift 2
      ;;
    --rollback-on-failure)
      ROLLBACK_ON_FAILURE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ -z "$REPO_NAME" || -z "$DESCRIPTION" ]]; then
  usage
  exit 1
fi

FULL_REPO="${ORG}/${REPO_NAME}"
if [[ -z "$DEST_PATH" ]]; then
  DEST_PATH="$PWD/$REPO_NAME"
fi

preflight

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "dry-run-ok: $FULL_REPO -> $DEST_PATH"
  exit 0
fi

mkdir -p "$(dirname "$DEST_PATH")"

if ! gh repo create "$FULL_REPO" "--${VISIBILITY}" --template "$TEMPLATE_REPO" --description "$DESCRIPTION"; then
  fail_with_summary 1 "failed to create repository: $FULL_REPO"
fi
REMOTE_CREATED=1

if ! gh repo clone "$FULL_REPO" "$DEST_PATH"; then
  fail_with_summary 1 "failed to clone repository: $FULL_REPO"
fi
LOCAL_CLONED=1

replace_template_readme_text "$DEST_PATH/README.md" "$REPO_NAME" "$DESCRIPTION"

if ! run_in_repo bash scripts/configure-repo-settings.sh --repo "$FULL_REPO"; then
  fail_with_summary 1 "failed to configure repository settings for $FULL_REPO"
fi
SETTINGS_CONFIGURED=1

if ! run_in_repo bash scripts/sync-agent-assets.sh; then
  fail_with_summary 1 "failed to sync agent assets in $DEST_PATH"
fi
ASSETS_SYNCED=1

if ! run_in_repo cargo fmt --all --check; then
  fail_with_summary 1 "cargo fmt --all --check failed in $DEST_PATH"
fi
if ! run_in_repo cargo nextest run --workspace --release --no-fail-fast; then
  fail_with_summary 1 "cargo nextest run failed in $DEST_PATH"
fi
if ! run_in_repo cargo test --doc --workspace --release; then
  fail_with_summary 1 "cargo test --doc failed in $DEST_PATH"
fi
if ! run_in_repo cargo llvm-cov nextest --workspace --release --json --output-path coverage.json; then
  fail_with_summary 1 "cargo llvm-cov failed in $DEST_PATH"
fi
if ! run_in_repo python3 scripts/check-coverage.py coverage.json; then
  fail_with_summary 1 "coverage thresholds failed in $DEST_PATH"
fi
if ! run_in_repo cargo doc --workspace --no-deps; then
  fail_with_summary 1 "cargo doc --workspace --no-deps failed in $DEST_PATH"
fi
if ! run_in_repo python3 scripts/check-docs-site.py; then
  fail_with_summary 1 "docs-site completeness checks failed in $DEST_PATH"
fi
VERIFIED=1

log "repo-created: https://github.com/$FULL_REPO"
log "clone-path: $DEST_PATH"
log "bootstrap-summary: remote_created=1 local_cloned=1 settings_configured=1 assets_synced=1 verified=1"
