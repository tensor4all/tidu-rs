import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class NextestRolloutTests(unittest.TestCase):
    def test_ci_uses_nextest_and_explicit_doctests(self) -> None:
        workflow = read_text(".github/workflows/ci.yml")

        self.assertIn("name: nextest (${{ matrix.os }})", workflow)
        self.assertIn("taiki-e/install-action@nextest", workflow)
        self.assertIn(
            "cargo nextest run --workspace --release --no-fail-fast",
            workflow,
        )
        self.assertIn("cargo test --doc --workspace --release", workflow)
        self.assertIn(
            "cargo llvm-cov nextest --workspace --release --json --output-path coverage.json",
            workflow,
        )

    def test_repo_settings_match_nextest_job_names(self) -> None:
        settings = json.loads(read_text("ai/repo-settings.json"))

        self.assertEqual(
            settings["required_status_checks"]["contexts"],
            [
                "rustfmt",
                "nextest (ubuntu-latest)",
                "nextest (macos-latest)",
                "coverage",
                "docs-site",
            ],
        )

    def test_helper_scripts_use_nextest_verification_commands(self) -> None:
        create_pr = read_text("scripts/create-pr.sh")
        bootstrap = read_text("ai/new-tensor4all-rust-repo/scripts/new-repo.sh")

        for content in (create_pr, bootstrap):
            self.assertIn(
                "cargo nextest run --workspace --release --no-fail-fast",
                content,
            )
            self.assertIn("cargo test --doc --workspace --release", content)
            self.assertIn(
                "cargo llvm-cov nextest --workspace --release --json --output-path coverage.json",
                content,
            )

    def test_docs_and_rules_prefer_nextest_with_explicit_doctests(self) -> None:
        for path in (
            "README.md",
            "AGENTS.md",
            "ai/numerical-rust-rules.md",
            "ai/pr-workflow-rules.md",
        ):
            content = read_text(path)
            self.assertIn("cargo nextest", content, msg=f"missing nextest in {path}")
            self.assertIn("cargo test --doc", content, msg=f"missing doctest guidance in {path}")


if __name__ == "__main__":
    unittest.main()
