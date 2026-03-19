import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = REPO_ROOT / "scripts" / "check-coverage.py"


class CheckCoverageScriptTests(unittest.TestCase):
    def test_fails_when_src_file_is_missing_from_coverage_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "src").mkdir()
            (root / "target" / "llvm-cov-target" / "debug" / "deps").mkdir(parents=True)
            shutil.copy2(SOURCE_SCRIPT, root / "scripts" / "check-coverage.py")

            (root / "coverage-thresholds.json").write_text(
                json.dumps({"default": 85, "files": {}}),
                encoding="utf-8",
            )
            (root / "src" / "lib.rs").write_text("pub fn demo() {}\n", encoding="utf-8")
            (
                root / "target" / "llvm-cov-target" / "debug" / "deps" / "demo.d"
            ).write_text(
                f"{root / 'target' / 'llvm-cov-target' / 'debug' / 'deps' / 'demo.d'}: {root / 'src' / 'lib.rs'}\n",
                encoding="utf-8",
            )

            report = {"data": [{"files": []}]}
            report_path = root / "coverage.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = subprocess.run(
                ["python3", "scripts/check-coverage.py", "coverage.json"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(
                result.returncode,
                0,
                msg=f"expected failure, got success:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertIn("src/lib.rs", result.stdout)

    def test_ignores_module_local_test_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "src" / "widget" / "tests").mkdir(parents=True)
            (root / "target" / "llvm-cov-target" / "debug" / "deps").mkdir(parents=True)
            shutil.copy2(SOURCE_SCRIPT, root / "scripts" / "check-coverage.py")

            (root / "coverage-thresholds.json").write_text(
                json.dumps({"default": 85, "files": {}}),
                encoding="utf-8",
            )
            (root / "src" / "lib.rs").write_text("pub mod widget;\n", encoding="utf-8")
            (root / "src" / "widget.rs").write_text("pub fn demo() {}\n", encoding="utf-8")
            (root / "src" / "widget" / "tests" / "mod.rs").write_text(
                "#[test]\nfn smoke() {}\n",
                encoding="utf-8",
            )
            (
                root / "target" / "llvm-cov-target" / "debug" / "deps" / "demo.d"
            ).write_text(
                " ".join(
                    [
                        f"{root / 'target' / 'llvm-cov-target' / 'debug' / 'deps' / 'demo.d'}:",
                        str(root / "src" / "lib.rs"),
                        str(root / "src" / "widget.rs"),
                        str(root / "src" / "widget" / "tests" / "mod.rs"),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = {
                "data": [
                    {
                        "files": [
                            {
                                "filename": str(root / "src" / "lib.rs"),
                                "summary": {"lines": {"percent": 100.0}},
                            },
                            {
                                "filename": str(root / "src" / "widget.rs"),
                                "summary": {"lines": {"percent": 100.0}},
                            },
                        ],
                    }
                ]
            }
            report_path = root / "coverage.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = subprocess.run(
                ["python3", "scripts/check-coverage.py", "coverage.json"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"expected success, got failure:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertNotIn("src/widget/tests/mod.rs", result.stdout)

    def test_ignores_feature_gated_sources_not_present_in_dep_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "src").mkdir()
            (root / "target" / "llvm-cov-target" / "debug" / "deps").mkdir(parents=True)
            shutil.copy2(SOURCE_SCRIPT, root / "scripts" / "check-coverage.py")

            (root / "coverage-thresholds.json").write_text(
                json.dumps({"default": 85, "files": {}}),
                encoding="utf-8",
            )
            (root / "src" / "lib.rs").write_text("pub fn demo() {}\n", encoding="utf-8")
            (root / "src" / "feature_only.rs").write_text("pub fn hidden() {}\n", encoding="utf-8")
            (
                root / "target" / "llvm-cov-target" / "debug" / "deps" / "demo.d"
            ).write_text(
                f"{root / 'target' / 'llvm-cov-target' / 'debug' / 'deps' / 'demo.d'}: {root / 'src' / 'lib.rs'}\n",
                encoding="utf-8",
            )

            report = {
                "data": [
                    {
                        "files": [
                            {
                                "filename": str(root / "src" / "lib.rs"),
                                "summary": {"lines": {"percent": 100.0}},
                            }
                        ],
                    }
                ]
            }
            report_path = root / "coverage.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = subprocess.run(
                ["python3", "scripts/check-coverage.py", "coverage.json"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"expected success, got failure:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertNotIn("src/feature_only.rs", result.stdout)

    def test_ignores_compiled_declaration_only_files_without_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "src").mkdir()
            (root / "target" / "llvm-cov-target" / "debug" / "deps").mkdir(parents=True)
            shutil.copy2(SOURCE_SCRIPT, root / "scripts" / "check-coverage.py")

            (root / "coverage-thresholds.json").write_text(
                json.dumps({"default": 85, "files": {}}),
                encoding="utf-8",
            )
            (root / "src" / "lib.rs").write_text("pub mod api;\n", encoding="utf-8")
            (root / "src" / "api.rs").write_text(
                "pub struct ResultBox<T> { pub value: T }\n\npub trait Demo {\n    fn work(&self);\n}\n",
                encoding="utf-8",
            )
            (
                root / "target" / "llvm-cov-target" / "debug" / "deps" / "demo.d"
            ).write_text(
                " ".join(
                    [
                        f"{root / 'target' / 'llvm-cov-target' / 'debug' / 'deps' / 'demo.d'}:",
                        str(root / "src" / "lib.rs"),
                        str(root / "src" / "api.rs"),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = {
                "data": [
                    {
                        "files": [
                            {
                                "filename": str(root / "src" / "lib.rs"),
                                "summary": {"lines": {"percent": 100.0}},
                            }
                        ],
                    }
                ]
            }
            report_path = root / "coverage.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = subprocess.run(
                ["python3", "scripts/check-coverage.py", "coverage.json"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"expected success, got failure:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertNotIn("src/api.rs", result.stdout)

    def test_resolves_repo_relative_dep_info_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "demo-crate" / "src").mkdir(parents=True)
            (root / "target" / "llvm-cov-target" / "debug" / "deps").mkdir(parents=True)
            shutil.copy2(SOURCE_SCRIPT, root / "scripts" / "check-coverage.py")

            (root / "coverage-thresholds.json").write_text(
                json.dumps({"default": 85, "files": {}}),
                encoding="utf-8",
            )
            (root / "demo-crate" / "src" / "lib.rs").write_text(
                "pub fn demo() {}\n",
                encoding="utf-8",
            )
            (
                root / "target" / "llvm-cov-target" / "debug" / "deps" / "demo.d"
            ).write_text(
                f"{root / 'target' / 'llvm-cov-target' / 'debug' / 'deps' / 'demo.d'}: demo-crate/src/lib.rs\n",
                encoding="utf-8",
            )

            report = {"data": [{"files": []}]}
            report_path = root / "coverage.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = subprocess.run(
                ["python3", "scripts/check-coverage.py", "coverage.json"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(
                result.returncode,
                0,
                msg=f"expected failure, got success:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertIn("demo-crate/src/lib.rs", result.stdout)


if __name__ == "__main__":
    unittest.main()
