import base64
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
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync-agent-assets.sh"
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check-agent-assets.sh"
DOCS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docs.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class AgentAssetPermissionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

    def make_temp_repo(self) -> tuple[Path, dict[str, str]]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        (root / "ai").mkdir()
        (root / "bin").mkdir()
        (root / "scripts").mkdir()
        (root / "state").mkdir()

        shutil.copy2(SYNC_SCRIPT, root / "scripts" / "sync-agent-assets.sh")
        shutil.copy2(CHECK_SCRIPT, root / "scripts" / "check-agent-assets.sh")

        return root, {
            "state_dir": str(root / "state"),
            "repo": "example/template",
            "revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        }

    def write_fake_gh(self, root: Path, state_dir: Path, repo: str, revision: str) -> None:
        manifest = {
            "managed_assets": [
                {
                    "id": "example-script",
                    "source": "scripts/example.sh",
                    "target": "scripts/example.sh",
                }
            ]
        }
        script = "#!/usr/bin/env bash\nprintf 'synced\\n'\n"

        (state_dir / "upstream-manifest.b64").write_text(
            base64.b64encode(json.dumps(manifest).encode("utf-8")).decode("ascii"),
            encoding="utf-8",
        )
        (state_dir / "example-script.b64").write_text(
            base64.b64encode(script.encode("utf-8")).decode("ascii"),
            encoding="utf-8",
        )

        write_executable(
            root / "bin" / "gh",
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                set -euo pipefail

                if [[ "$1" == "auth" && "$2" == "status" ]]; then
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/{repo}/commits/main" && "$3" == "--jq" && "$4" == ".sha" ]]; then
                  printf '{revision}\\n'
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/{repo}/contents/ai/manifest.json?ref=main" && "$3" == "--jq" && "$4" == ".content" ]]; then
                  cat "{state_dir}/upstream-manifest.b64"
                  exit 0
                fi

                if [[ "$1" == "api" && "$2" == "repos/{repo}/contents/scripts/example.sh?ref=main" && "$3" == "--jq" && "$4" == ".content" ]]; then
                  cat "{state_dir}/example-script.b64"
                  exit 0
                fi

                printf 'unexpected gh invocation: %s\\n' "$*" >&2
                exit 1
                """
            ),
        )

    def test_sync_agent_assets_marks_shell_targets_executable(self) -> None:
        root, state = self.make_temp_repo()
        state_dir = Path(state["state_dir"])
        self.write_fake_gh(root, state_dir, state["repo"], state["revision"])

        env = os.environ.copy()
        env["PATH"] = f"{root / 'bin'}:{env['PATH']}"

        result = subprocess.run(
            [
                "bash",
                "scripts/sync-agent-assets.sh",
                "--source-repo",
                state["repo"],
                "--source-ref",
                "main",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertTrue((root / "scripts" / "example.sh").is_file())
        self.assertTrue(os.access(root / "scripts" / "example.sh", os.X_OK))

    def test_check_agent_assets_flags_shell_targets_that_lose_execute_bit(self) -> None:
        root, state = self.make_temp_repo()
        state_dir = Path(state["state_dir"])
        self.write_fake_gh(root, state_dir, state["repo"], state["revision"])

        upstream_manifest_path = state_dir / "upstream-manifest.json"
        upstream_manifest_path.write_text(
            json.dumps(
                {
                    "bundle_revision": state["revision"],
                    "managed_assets": [
                        {
                            "id": "example-script",
                            "source": "scripts/example.sh",
                            "target": "scripts/example.sh",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["PATH"] = f"{root / 'bin'}:{env['PATH']}"

        sync_result = subprocess.run(
            [
                "bash",
                "scripts/sync-agent-assets.sh",
                "--source-repo",
                state["repo"],
                "--source-ref",
                "main",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        self.assertEqual(
            sync_result.returncode,
            0,
            msg=f"stdout={sync_result.stdout}\nstderr={sync_result.stderr}",
        )

        target = root / "scripts" / "example.sh"
        target.chmod(target.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))

        check_result = subprocess.run(
            [
                "bash",
                "scripts/check-agent-assets.sh",
                "--upstream-manifest-file",
                str(upstream_manifest_path),
                "--upstream-revision",
                state["revision"],
                "--source-repo",
                state["repo"],
                "--source-ref",
                "main",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        self.assertEqual(check_result.returncode, 12, msg=check_result.stderr)
        self.assertIn("managed-file-not-executable: scripts/example.sh", check_result.stdout)

    def test_docs_workflows_invoke_build_docs_site_via_bash(self) -> None:
        docs_workflow = DOCS_WORKFLOW.read_text(encoding="utf-8")
        ci_workflow = CI_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("run: bash scripts/build_docs_site.sh", docs_workflow)
        self.assertIn("run: bash scripts/build_docs_site.sh", ci_workflow)
        self.assertNotIn("run: ./scripts/build_docs_site.sh", docs_workflow)
        self.assertNotIn("run: ./scripts/build_docs_site.sh", ci_workflow)


if __name__ == "__main__":
    unittest.main()
