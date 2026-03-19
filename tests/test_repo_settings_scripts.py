import json
import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check-repo-settings.sh"
CONFIGURE_SCRIPT = REPO_ROOT / "scripts" / "configure-repo-settings.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def write_repo_settings(root: Path) -> None:
    (root / "ai").mkdir()
    (root / "ai" / "repo-settings.json").write_text(
        json.dumps(
            {
                "default_branch": "main",
                "allow_auto_merge": True,
                "delete_branch_on_merge": True,
                "required_status_checks": {
                    "strict": True,
                    "contexts": [
                        "rustfmt",
                        "cargo test (ubuntu-latest)",
                        "cargo test (macos-latest)",
                        "coverage",
                        "docs-site",
                    ],
                },
                "pages": {
                    "enabled": True,
                    "build_type": "workflow",
                },
            }
        ),
        encoding="utf-8",
    )


def repo_json() -> str:
    return json.dumps(
        {
            "allow_auto_merge": True,
            "delete_branch_on_merge": True,
        }
    )


def protection_json() -> str:
    return json.dumps(
        {
            "required_status_checks": {
                "strict": True,
                "contexts": [
                    "rustfmt",
                    "cargo test (ubuntu-latest)",
                    "cargo test (macos-latest)",
                    "coverage",
                    "docs-site",
                ],
            }
        }
    )


class RepoSettingsScriptTests(unittest.TestCase):
    def run_script(self, script_name: str, gh_script: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        (root / "scripts").mkdir()
        shutil.copy2(CHECK_SCRIPT, root / "scripts" / "check-repo-settings.sh")
        shutil.copy2(CONFIGURE_SCRIPT, root / "scripts" / "configure-repo-settings.sh")
        write_repo_settings(root)

        bin_dir = root / "bin"
        bin_dir.mkdir()
        state_dir = root / "state"
        state_dir.mkdir()
        (state_dir / "gh-log.jsonl").write_text("", encoding="utf-8")

        write_executable(bin_dir / "gh", gh_script)

        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["FAKE_GH_STATE_DIR"] = str(state_dir)

        result = subprocess.run(
            ["bash", str(root / "scripts" / script_name)],
            cwd=root,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        return result, root

    def test_check_repo_settings_fails_when_pages_site_is_missing(self) -> None:
        result, _ = self.run_script(
            "check-repo-settings.sh",
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -euo pipefail

                if [[ "$1" == "repo" && "$2" == "view" ]]; then
                  printf 'example/repo\\n'
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo" ]]; then
                  printf '%s\\n' '{repo_json()}'
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo/branches/main/protection" ]]; then
                  printf '%s\\n' '{protection_json()}'
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo/pages" ]]; then
                  printf '{{"message":"Not Found","status":"404"}}\\n'
                  printf 'gh: Not Found (HTTP 404)\\n' >&2
                  exit 1
                fi

                printf 'unexpected gh invocation: %s\\n' "$*" >&2
                exit 1
                """
            ),
        )

        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"expected failure, got success:\\nstdout={result.stdout}\\nstderr={result.stderr}",
        )
        self.assertIn("pages is not enabled", result.stdout)

    def test_configure_repo_settings_creates_pages_site_for_workflow_builds(self) -> None:
        result, root = self.run_script(
            "configure-repo-settings.sh",
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -euo pipefail

                state_dir="${{FAKE_GH_STATE_DIR:?}}"
                log_file="$state_dir/gh-log.jsonl"
                pages_file="$state_dir/pages.json"

                log_call() {{
                  python3 - "$log_file" "$@" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
entry = sys.argv[2:]
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry) + "\\n")
PY
                }}

                if [[ "$1" == "repo" && "$2" == "view" ]]; then
                  log_call "$@"
                  printf 'example/repo\\n'
                  exit 0
                fi

                if [[ "$1" == "repo" && "$2" == "edit" ]]; then
                  log_call "$@"
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo" ]]; then
                  log_call "$@"
                  printf '%s\\n' '{repo_json()}'
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo/branches/main/protection" ]]; then
                  log_call "$@"
                  printf '%s\\n' '{protection_json()}'
                  exit 0
                fi

                if [[ "$1" == "api" && "${{4-}}" == "repos/example/repo/branches/main/protection" ]]; then
                  log_call "$@"
                  cat >/dev/null
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo/pages" ]]; then
                  log_call "$@"
                  if [[ -f "$pages_file" ]]; then
                    cat "$pages_file"
                    exit 0
                  fi

                  printf '{{"message":"Not Found","status":"404"}}\\n'
                  printf 'gh: Not Found (HTTP 404)\\n' >&2
                  exit 1
                fi

                if [[ "$1" == "api" && "${{4-}}" == "repos/example/repo/pages" ]]; then
                  log_call "$@"
                  printf '{{"build_type":"workflow"}}\\n' >"$pages_file"
                  printf '{{"build_type":"workflow"}}\\n'
                  exit 0
                fi

                printf 'unexpected gh invocation: %s\\n' "$*" >&2
                exit 1
                """
            ),
        )

        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\\nstderr={result.stderr}")
        self.assertIn("repo-settings-ok: example/repo", result.stdout)
        gh_log = [
            json.loads(line)
            for line in (root / "state" / "gh-log.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertIn(
            [
                "api",
                "-X",
                "POST",
                "repos/example/repo/pages",
                "-H",
                "Accept: application/vnd.github+json",
                "-f",
                "build_type=workflow",
            ],
            gh_log,
        )

    def test_check_repo_settings_fails_when_pages_build_type_is_wrong(self) -> None:
        result, _ = self.run_script(
            "check-repo-settings.sh",
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -euo pipefail

                if [[ "$1" == "repo" && "$2" == "view" ]]; then
                  printf 'example/repo\\n'
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo" ]]; then
                  printf '%s\\n' '{repo_json()}'
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo/branches/main/protection" ]]; then
                  printf '%s\\n' '{protection_json()}'
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo/pages" ]]; then
                  printf '{{"build_type":"legacy"}}\\n'
                  exit 0
                fi

                printf 'unexpected gh invocation: %s\\n' "$*" >&2
                exit 1
                """
            ),
        )

        self.assertNotEqual(
            result.returncode,
            0,
            msg=f"expected failure, got success:\\nstdout={result.stdout}\\nstderr={result.stderr}",
        )
        self.assertIn("pages.build_type mismatch", result.stdout)

    def test_configure_repo_settings_updates_existing_pages_site_to_workflow_builds(self) -> None:
        result, root = self.run_script(
            "configure-repo-settings.sh",
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -euo pipefail

                state_dir="${{FAKE_GH_STATE_DIR:?}}"
                log_file="$state_dir/gh-log.jsonl"
                pages_file="$state_dir/pages.json"
                if [[ ! -f "$pages_file" ]]; then
                  printf '{{"build_type":"legacy"}}\\n' >"$pages_file"
                fi

                log_call() {{
                  python3 - "$log_file" "$@" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
entry = sys.argv[2:]
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry) + "\\n")
PY
                }}

                if [[ "$1" == "repo" && "$2" == "view" ]]; then
                  log_call "$@"
                  printf 'example/repo\\n'
                  exit 0
                fi

                if [[ "$1" == "repo" && "$2" == "edit" ]]; then
                  log_call "$@"
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo" ]]; then
                  log_call "$@"
                  printf '%s\\n' '{repo_json()}'
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo/branches/main/protection" ]]; then
                  log_call "$@"
                  printf '%s\\n' '{protection_json()}'
                  exit 0
                fi

                if [[ "$1" == "api" && "${{4-}}" == "repos/example/repo/branches/main/protection" ]]; then
                  log_call "$@"
                  cat >/dev/null
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/example/repo/pages" ]]; then
                  log_call "$@"
                  cat "$pages_file"
                  exit 0
                fi

                if [[ "$1" == "api" && "${{4-}}" == "repos/example/repo/pages" ]]; then
                  log_call "$@"
                  printf '{{"build_type":"workflow"}}\\n' >"$pages_file"
                  printf '{{"build_type":"workflow"}}\\n'
                  exit 0
                fi

                printf 'unexpected gh invocation: %s\\n' "$*" >&2
                exit 1
                """
            ),
        )

        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\\nstderr={result.stderr}")
        gh_log = [
            json.loads(line)
            for line in (root / "state" / "gh-log.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertIn(
            [
                "api",
                "-X",
                "PUT",
                "repos/example/repo/pages",
                "-H",
                "Accept: application/vnd.github+json",
                "-f",
                "build_type=workflow",
            ],
            gh_log,
        )


if __name__ == "__main__":
    unittest.main()
