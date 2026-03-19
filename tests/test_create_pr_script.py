import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = REPO_ROOT / "scripts" / "create-pr.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class CreatePrScriptTests(unittest.TestCase):
    def test_uses_release_llvm_cov_as_single_heavy_local_verification_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "bin").mkdir()
            (root / "state").mkdir()
            shutil.copy2(SOURCE_SCRIPT, root / "scripts" / "create-pr.sh")

            for helper_name in [
                "check-agent-assets.sh",
                "check-repo-settings.sh",
                "monitor-pr-checks.sh",
            ]:
                write_executable(
                    root / "scripts" / helper_name,
                    "#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n",
                )

            (root / "scripts" / "check-docs-site.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "scripts" / "check-coverage.py").write_text("print('ok')\n", encoding="utf-8")

            write_executable(
                root / "bin" / "cargo",
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf '%s\\n' "$*" >>"${FAKE_STATE_DIR:?}/cargo.log"
                    exit 0
                    """
                ),
            )

            write_executable(
                root / "bin" / "python3",
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf '%s\\n' "$*" >>"${FAKE_STATE_DIR:?}/python.log"
                    exit 0
                    """
                ),
            )

            write_executable(
                root / "bin" / "git",
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf '%s\\n' "$*" >>"${FAKE_STATE_DIR:?}/git.log"

                    if [[ "$1" == "status" && "$2" == "--short" ]]; then
                      exit 0
                    fi

                    if [[ "$1" == "branch" && "$2" == "--show-current" ]]; then
                      printf 'feature/test\\n'
                      exit 0
                    fi

                    if [[ "$1" == "log" && "$2" == "-1" ]]; then
                      printf 'subject\\n'
                      exit 0
                    fi

                    if [[ "$1" == "log" ]]; then
                      printf '%s\\n' '- implemented change'
                      exit 0
                    fi

                    if [[ "$1" == "rev-parse" ]]; then
                      exit 1
                    fi

                    if [[ "$1" == "push" ]]; then
                      exit 0
                    fi

                    printf 'unexpected git invocation: %s\\n' "$*" >&2
                    exit 1
                    """
                ),
            )

            write_executable(
                root / "bin" / "gh",
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf '%s\\n' "$*" >>"${FAKE_STATE_DIR:?}/gh.log"

                    if [[ "$1" == "pr" && "$2" == "create" ]]; then
                      body_file=""
                      prev=""
                      for arg in "$@"; do
                        if [[ "$prev" == "--body-file" ]]; then
                          body_file="$arg"
                        fi
                        prev="$arg"
                      done
                      cp "$body_file" "${FAKE_STATE_DIR:?}/pr-body.md"
                      printf 'https://example.invalid/pr/1\\n'
                      exit 0
                    fi

                    if [[ "$1" == "pr" && "$2" == "merge" ]]; then
                      exit 0
                    fi

                    printf 'unexpected gh invocation: %s\\n' "$*" >&2
                    exit 1
                    """
                ),
            )

            env = os.environ.copy()
            env["PATH"] = f"{root / 'bin'}:{env['PATH']}"
            env["FAKE_STATE_DIR"] = str(root / "state")

            result = subprocess.run(
                [
                    "bash",
                    "scripts/create-pr.sh",
                    "--no-auto-merge",
                    "--title",
                    "Explicit title",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            cargo_invocations = (root / "state" / "cargo.log").read_text(encoding="utf-8").splitlines()
            self.assertIn("fmt --all --check", cargo_invocations)
            self.assertNotIn("test --workspace --release", cargo_invocations)
            self.assertIn(
                "llvm-cov --workspace --release --json --output-path coverage.json",
                cargo_invocations,
            )

            body = (root / "state" / "pr-body.md").read_text(encoding="utf-8")
            self.assertIn("`cargo llvm-cov --workspace --release --json --output-path coverage.json`", body)
            self.assertNotIn("`cargo test --workspace --release`", body)


if __name__ == "__main__":
    unittest.main()
