import json
import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "monitor-pr-checks.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class MonitorPrChecksTests(unittest.TestCase):
    def run_monitor(self, responses):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bin_dir = temp_path / "bin"
            state_dir = temp_path / "state"
            bin_dir.mkdir()
            state_dir.mkdir()

            for index, payload in enumerate(responses):
                (state_dir / f"checks-{index}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            (state_dir / "index.txt").write_text("0", encoding="utf-8")

            write_executable(
                bin_dir / "gh",
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail

                    if [[ "$1" == "pr" && "$2" == "checks" ]]; then
                      state_dir="${FAKE_GH_STATE_DIR:?}"
                      index="$(cat "$state_dir/index.txt")"
                      response="$state_dir/checks-${index}.json"
                      if [[ ! -f "$response" ]]; then
                        response="$state_dir/checks-$((index - 1)).json"
                      fi
                      cat "$response"
                      printf '%s' "$((index + 1))" >"$state_dir/index.txt"
                      exit 0
                    fi

                    printf 'unexpected gh invocation: %s\\n' "$*" >&2
                    exit 1
                    """
                ),
            )

            write_executable(
                bin_dir / "sleep",
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    exit 0
                    """
                ),
            )

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env["FAKE_GH_STATE_DIR"] = str(state_dir)

            return subprocess.run(
                ["bash", str(SCRIPT_PATH), "123", "--interval", "0"],
                capture_output=True,
                text=True,
                env=env,
                cwd=REPO_ROOT,
                check=False,
            )

    def test_pending_checks_continue_until_success(self):
        result = self.run_monitor(
            [
                [
                    {
                        "name": "coverage",
                        "bucket": "pending",
                        "state": "IN_PROGRESS",
                        "link": "https://example.invalid/runs/1",
                        "workflow": "ci",
                    }
                ],
                [
                    {
                        "name": "coverage",
                        "bucket": "pass",
                        "state": "SUCCESS",
                        "link": "https://example.invalid/runs/1",
                        "workflow": "ci",
                    }
                ],
            ]
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("all required checks passed", result.stdout)

    def test_first_failure_exits_immediately_with_failed_check_name(self):
        result = self.run_monitor(
            [
                [
                    {
                        "name": "rustfmt",
                        "bucket": "pending",
                        "state": "QUEUED",
                        "link": "https://github.com/example/repo/actions/runs/111/job/222",
                        "workflow": "ci",
                    }
                ],
                [
                    {
                        "name": "rustfmt",
                        "bucket": "fail",
                        "state": "FAILURE",
                        "link": "https://github.com/example/repo/actions/runs/111/job/222",
                        "workflow": "ci",
                    },
                    {
                        "name": "coverage",
                        "bucket": "pending",
                        "state": "IN_PROGRESS",
                        "link": "https://github.com/example/repo/actions/runs/333/job/444",
                        "workflow": "ci",
                    },
                ],
            ]
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed check: rustfmt", result.stdout)
        self.assertIn("gh run view 111 --log-failed", result.stdout)


if __name__ == "__main__":
    unittest.main()
