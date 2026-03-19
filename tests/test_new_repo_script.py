import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = REPO_ROOT / "ai" / "new-tensor4all-rust-repo" / "scripts" / "new-repo.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class NewRepoScriptTests(unittest.TestCase):
    def test_uses_release_llvm_cov_as_single_heavy_bootstrap_verification_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scripts_dir = root / "scripts"
            bin_dir = root / "bin"
            state_dir = root / "state"
            dest_path = root / "demo-repo"

            scripts_dir.mkdir()
            bin_dir.mkdir()
            state_dir.mkdir()
            shutil.copy2(SOURCE_SCRIPT, scripts_dir / "new-repo.sh")

            write_executable(
                bin_dir / "cargo",
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
                bin_dir / "python3",
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
                bin_dir / "gh",
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf '%s\\n' "$*" >>"${FAKE_STATE_DIR:?}/gh.log"

                    if [[ "$1" == "auth" && "$2" == "status" ]]; then
                      exit 0
                    fi

                    if [[ "$1" == "repo" && "$2" == "view" ]]; then
                      if [[ "$3" == "tensor4all/template-rs" ]]; then
                        exit 0
                      fi
                      exit 1
                    fi

                    if [[ "$1" == "repo" && "$2" == "create" ]]; then
                      exit 0
                    fi

                    if [[ "$1" == "repo" && "$2" == "clone" ]]; then
                      dest="$4"
                      mkdir -p "$dest/scripts"
                      cat <<'EOF' >"$dest/README.md"
                    # template-rs

                    Template repository for Rust workspace projects in the tensor4all organization.
                    EOF
                      cat <<'EOF' >"$dest/scripts/configure-repo-settings.sh"
                    #!/usr/bin/env bash
                    set -euo pipefail
                    exit 0
                    EOF
                      cat <<'EOF' >"$dest/scripts/sync-agent-assets.sh"
                    #!/usr/bin/env bash
                    set -euo pipefail
                    exit 0
                    EOF
                      cat <<'EOF' >"$dest/scripts/check-coverage.py"
                    print("ok")
                    EOF
                      cat <<'EOF' >"$dest/scripts/check-docs-site.py"
                    print("ok")
                    EOF
                      chmod +x "$dest/scripts/configure-repo-settings.sh" "$dest/scripts/sync-agent-assets.sh"
                      exit 0
                    fi

                    printf 'unexpected gh invocation: %s\\n' "$*" >&2
                    exit 1
                    """
                ),
            )

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env["FAKE_STATE_DIR"] = str(state_dir)

            result = subprocess.run(
                [
                    "bash",
                    "scripts/new-repo.sh",
                    "--repo",
                    "demo-repo",
                    "--description",
                    "Demo repository",
                    "--dest",
                    str(dest_path),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            cargo_invocations = (state_dir / "cargo.log").read_text(encoding="utf-8").splitlines()
            self.assertIn("fmt --all --check", cargo_invocations)
            self.assertNotIn("test --workspace --release", cargo_invocations)
            self.assertIn(
                "llvm-cov --workspace --release --json --output-path coverage.json",
                cargo_invocations,
            )
            self.assertIn("doc --workspace --no-deps", cargo_invocations)


if __name__ == "__main__":
    unittest.main()
