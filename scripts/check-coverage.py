#!/usr/bin/env python3
"""Check per-file line coverage against thresholds defined in coverage-thresholds.json."""

import json
import re
import shlex
import sys
from pathlib import Path


def has_runtime_code(path: Path) -> bool:
    pending_signature = ""
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue
        if "//" in stripped:
            stripped = stripped.split("//", 1)[0].strip()
        if not stripped:
            continue

        if pending_signature:
            pending_signature = f"{pending_signature} {stripped}".strip()
            open_brace = pending_signature.find("{")
            semicolon = pending_signature.find(";")
            if open_brace != -1 and (semicolon == -1 or open_brace < semicolon):
                return True
            if semicolon != -1:
                pending_signature = ""
            continue

        if re.search(r"\b(?:const|static)\b.*=\s*{", stripped):
            return True
        if re.search(r"\bfn\b", stripped):
            open_brace = stripped.find("{")
            semicolon = stripped.find(";")
            if open_brace != -1 and (semicolon == -1 or open_brace < semicolon):
                return True
            if semicolon == -1:
                pending_signature = stripped
    return False


def scanned_source_files(root: Path) -> set[str]:
    files = set()
    for path in root.rglob("*.rs"):
        rel = path.relative_to(root)
        if "target" in rel.parts:
            continue
        if "src" not in rel.parts:
            continue
        if "tests" in rel.parts:
            continue
        # Test-only modules moved out of `src` keep the `tests.rs` name. They are
        # compiled only under `cfg(test)`, so the library profile reports them at
        # 0% and would otherwise fail the threshold on every move.
        if path.name == "tests.rs":
            continue
        if not has_runtime_code(path):
            continue
        files.add(str(rel))
    return files


def parse_dep_info_file(root: Path, path: Path) -> set[str]:
    files = set()
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\\\n", " ")
    for line in text.splitlines():
        if ":" not in line:
            continue
        _, deps = line.split(":", 1)
        for token in shlex.split(deps):
            dep_path = Path(token)
            if not dep_path.is_absolute():
                parent_relative = (path.parent / dep_path).resolve()
                root_relative = (root / dep_path).resolve()
                if parent_relative.exists():
                    dep_path = parent_relative
                else:
                    dep_path = root_relative
            try:
                rel = dep_path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            if "src" not in rel.parts:
                continue
            if "tests" in rel.parts:
                continue
            if dep_path.suffix != ".rs":
                continue
            # Same rule as `scanned_source_files`: a `tests.rs` sibling holds
            # tests only and is reported at 0% by the library profile.
            if dep_path.name == "tests.rs":
                continue
            if not has_runtime_code(dep_path):
                continue
            files.add(str(rel))
    return files


def expected_source_files(root: Path) -> set[str]:
    dep_files = set()
    dep_info_dir = root / "target" / "llvm-cov-target"
    for path in dep_info_dir.rglob("*.d"):
        dep_files.update(parse_dep_info_file(root, path))
    if dep_files:
        return dep_files
    return scanned_source_files(root)


def main():
    root = Path(__file__).resolve().parent.parent
    thresholds_path = root / "coverage-thresholds.json"

    with open(thresholds_path) as f:
        config = json.load(f)
    default_threshold = config["default"]
    file_thresholds = config.get("files", {})
    excluded_files = set(config.get("exclude", []))

    report_only = "--report-only" in sys.argv[1:]
    args = [arg for arg in sys.argv[1:] if arg != "--report-only"]

    if args:
        with open(args[0]) as f:
            cov_data = json.load(f)
    else:
        cov_data = json.load(sys.stdin)

    files = cov_data["data"][0]["files"]
    # Resolve both sides before comparing: on macOS the temporary directory and
    # TMPDIR are symlinked (`/var/...` vs `/private/var/...`), and llvm-cov can
    # report either form. A literal prefix comparison then misses every file and
    # reports them all as uncovered.
    resolved_root = root.resolve()
    covered_files = set()

    failures = []
    passed = 0
    skipped = 0

    for entry in files:
        abs_path = entry["filename"]
        try:
            rel_path = str(Path(abs_path).resolve().relative_to(resolved_root))
        except ValueError:
            rel_path = abs_path

        if rel_path in excluded_files:
            skipped += 1
            continue

        # Test-only module files (see `scanned_source_files`): the library
        # profile reports them at 0%, so they are not held to the threshold.
        if Path(rel_path).name == "tests.rs":
            skipped += 1
            continue

        covered_files.add(rel_path)
        lines = entry["summary"]["lines"]
        percent = lines["percent"]
        threshold = file_thresholds.get(rel_path, default_threshold)

        if percent < threshold:
            failures.append((rel_path, percent, threshold))
        else:
            passed += 1

    missing_files = sorted(expected_source_files(root) - covered_files - excluded_files)
    for rel_path in missing_files:
        failures.append((rel_path, 0.0, default_threshold))

    total = passed + len(failures)
    print(f"Coverage check: {passed}/{total} files passed (excluded: {skipped})\n")

    if failures:
        print("FAILED files:")
        for path, actual, required in sorted(failures):
            print(f"  {path}: {actual:.1f}% < {required}%")
        print()
        if report_only:
            print("Report-only mode: not failing.")
            sys.exit(0)
        sys.exit(1)

    print("All files meet their coverage thresholds.")


if __name__ == "__main__":
    main()
