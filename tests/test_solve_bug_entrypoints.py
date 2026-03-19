import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_CODEX = REPO_ROOT / "ai" / "run-codex-solve-bug.sh"
RUN_CLAUDE = REPO_ROOT / "ai" / "run-claude-solve-bug.sh"
PROMPT = REPO_ROOT / "ai" / "solve_bug_issue.md"
MANIFEST = REPO_ROOT / "ai" / "manifest.json"


class TemplateSolveBugEntrypointsTests(unittest.TestCase):
    def test_prompt_contract(self) -> None:
        self.assertTrue(RUN_CODEX.is_file(), msg=f"missing file: {RUN_CODEX}")
        self.assertTrue(RUN_CLAUDE.is_file(), msg=f"missing file: {RUN_CLAUDE}")
        self.assertTrue(PROMPT.is_file(), msg=f"missing file: {PROMPT}")
        self.assertTrue(MANIFEST.is_file(), msg=f"missing file: {MANIFEST}")

        prompt = PROMPT.read_text(encoding="utf-8")
        manifest = MANIFEST.read_text(encoding="utf-8")

        self.assertIn("bash scripts/create-pr.sh", prompt)
        self.assertIn("bash scripts/monitor-pr-checks.sh", prompt)
        self.assertIn("effectively no open bug or bug-like issues", prompt)
        self.assertIn("run-codex-solve-bug.sh", manifest)
        self.assertIn("run-claude-solve-bug.sh", manifest)
        self.assertIn("solve_bug_issue.md", manifest)

    def test_run_codex_help(self) -> None:
        result = self._run_wrapper_help(RUN_CODEX)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--prompt", result.stdout)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--text", result.stdout)

    def test_run_claude_help(self) -> None:
        result = self._run_wrapper_help(RUN_CLAUDE)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--prompt", result.stdout)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--text", result.stdout)

    def _run_wrapper_help(self, source_script: Path) -> subprocess.CompletedProcess[str]:
        self.assertTrue(source_script.is_file(), msg=f"missing file: {source_script}")
        env = os.environ.copy()
        return subprocess.run(
            ["bash", str(source_script), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
