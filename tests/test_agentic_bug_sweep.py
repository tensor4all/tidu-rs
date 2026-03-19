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
PROMPT_PATH = REPO_ROOT / "ai" / "agentic-bug-sweep.md"
SCHEMA_PATH = REPO_ROOT / "ai" / "agentic-bug-sweep.schema.json"
SOURCE_SCRIPT = REPO_ROOT / "scripts" / "agentic-bug-sweep.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def setup_fake_repo(root: Path, *, gh_script: str, codex_script: str) -> None:
    (root / "scripts").mkdir()
    (root / "ai").mkdir()
    (root / "bin").mkdir()
    (root / "state").mkdir()
    (root / "docs" / "test-reports" / "agentic-bug-sweep").mkdir(parents=True)

    shutil.copy2(SOURCE_SCRIPT, root / "scripts" / "agentic-bug-sweep.sh")
    shutil.copy2(PROMPT_PATH, root / "ai" / "agentic-bug-sweep.md")
    shutil.copy2(SCHEMA_PATH, root / "ai" / "agentic-bug-sweep.schema.json")

    write_executable(root / "bin" / "gh", gh_script)
    write_executable(root / "bin" / "codex", codex_script)


def run_bug_sweep(
    root: Path,
    *,
    iterations: str = "1",
    max_consecutive_none: str = "1",
    repo: str = "tensor4all/template-rs",
    workdir: Path | None = None,
    repo_url: str | None = None,
    ref: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = f"{root / 'bin'}:{env['PATH']}"
    env["FAKE_STATE_DIR"] = str(root / "state")
    if extra_env:
        env.update(extra_env)

    args = [
        "bash",
        "scripts/agentic-bug-sweep.sh",
        "--iterations",
        iterations,
        "--max-consecutive-none",
        max_consecutive_none,
        "--repo",
        repo,
    ]
    if workdir is None and repo_url is None:
        workdir = root
    if workdir is not None:
        args.extend(["--workdir", str(workdir)])
    if repo_url is not None:
        args.extend(["--repo-url", repo_url])
    if ref is not None:
        args.extend(["--ref", ref])

    return subprocess.run(
        args,
        cwd=root,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def codex_script_for_payloads(
    payloads: list[dict[str, object]], *, stdout_lines: list[str] | None = None
) -> str:
    responses_text = json.dumps(payloads)
    stdout_prefix = ""
    if stdout_lines:
        stdout_prefix = "".join(f"printf '%s\\n' {line!r}\n" for line in stdout_lines)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{stdout_prefix}"
        "printf 'call\\n' >>\"${FAKE_STATE_DIR:?}/codex.log\"\n"
        "printf '%q ' \"$@\" >>\"${FAKE_STATE_DIR:?}/codex-args.log\"\n"
        "printf '\\n' >>\"${FAKE_STATE_DIR:?}/codex-args.log\"\n"
        "printf '%s\\n' \"${TMPDIR:-}\" >>\"${FAKE_STATE_DIR:?}/codex-tmpdir.log\"\n"
        "counter_file=\"${FAKE_STATE_DIR:?}/codex-counter.txt\"\n"
        "if [[ ! -f \"$counter_file\" ]]; then\n"
        "  printf '0\\n' >\"$counter_file\"\n"
        "fi\n"
        "counter=\"$(cat \"$counter_file\")\"\n"
        "\n"
        "output_path=\"\"\n"
        "prev=\"\"\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$prev\" == \"-o\" || \"$prev\" == \"--output-last-message\" ]]; then\n"
        "    output_path=\"$arg\"\n"
        "  fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "\n"
        "python3 - \"$counter_file\" \"${output_path:?}\" <<'PY'\n"
        "import json\n"
        "import sys\n"
        "\n"
        f"payloads = json.loads({responses_text!r})\n"
        "counter_path, output_path = sys.argv[1], sys.argv[2]\n"
        "with open(counter_path, 'r', encoding='utf-8') as handle:\n"
        "    index = int(handle.read().strip())\n"
        "if index >= len(payloads):\n"
        "    raise SystemExit(f'no payload configured for invocation {index}')\n"
        "with open(output_path, 'w', encoding='utf-8') as handle:\n"
        "    json.dump(payloads[index], handle, indent=2)\n"
        "with open(counter_path, 'w', encoding='utf-8') as handle:\n"
        "    handle.write(str(index + 1))\n"
        "PY\n"
        "exit 0\n"
    )


def codex_script_for_payload(
    payload: dict[str, object], *, stdout_lines: list[str] | None = None
) -> str:
    return codex_script_for_payloads([payload], stdout_lines=stdout_lines)


def gh_script_with_mutations(*, fail_comment: bool = False) -> str:
    comment_failure_branch = ""
    if fail_comment:
        comment_failure_branch = (
            "          printf 'simulated comment failure\\n' >&2\n"
            "          exit 1\n"
        )

    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\\n' "$*" >>"${{FAKE_STATE_DIR:?}}/gh.log"

        if [[ "$1" == "auth" && "$2" == "status" ]]; then
          exit 0
        fi

        if [[ "$1" == "issue" && "$2" == "list" ]]; then
          printf '[{{"number":1,"title":"Tracked bug","body":"details","labels":[{{"name":"bug"}}],"url":"https://example.invalid/issues/1"}}]\\n'
          exit 0
        fi

        if [[ "$1" == "issue" && "$2" == "create" ]]; then
          title=""
          body_file=""
          prev=""
          for arg in "$@"; do
            if [[ "$prev" == "--title" ]]; then
              title="$arg"
            fi
            if [[ "$prev" == "--body-file" || "$prev" == "-F" ]]; then
              body_file="$arg"
            fi
            prev="$arg"
          done
          printf '%s' "$title" >"${{FAKE_STATE_DIR:?}}/create-title.txt"
          cp "${{body_file:?}}" "${{FAKE_STATE_DIR:?}}/create-body.md"
          printf 'https://example.invalid/issues/99\\n'
          exit 0
        fi

        if [[ "$1" == "issue" && "$2" == "comment" ]]; then
          issue_number="$3"
          body_text=""
          body_file=""
          prev=""
          for arg in "$@"; do
            if [[ "$prev" == "--body" || "$prev" == "-b" ]]; then
              body_text="$arg"
            fi
            if [[ "$prev" == "--body-file" || "$prev" == "-F" ]]; then
              body_file="$arg"
            fi
            prev="$arg"
          done
{comment_failure_branch}          if [[ -n "$body_file" ]]; then
            cp "$body_file" "${{FAKE_STATE_DIR:?}}/comment-${{issue_number}}.md"
          else
            printf '%s' "$body_text" >"${{FAKE_STATE_DIR:?}}/comment-${{issue_number}}.md"
          fi
          exit 0
        fi

        if [[ "$1" == "issue" && "$2" == "close" ]]; then
          issue_number="$3"
          printf '%s' "$issue_number" >>"${{FAKE_STATE_DIR:?}}/closed.log"
          printf '\\n' >>"${{FAKE_STATE_DIR:?}}/closed.log"
          exit 0
        fi

        printf 'unexpected gh invocation: %s\\n' "$*" >&2
        exit 1
        """
    )


class AgenticBugSweepTests(unittest.TestCase):
    def test_prompt_and_schema_contract(self) -> None:
        self.assertTrue(PROMPT_PATH.is_file(), msg=f"missing prompt file: {PROMPT_PATH}")
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.assertIn("open bug issues", prompt)
        self.assertIn("prior bug-sweep reports", prompt)
        self.assertIn("test-feature", prompt)
        self.assertIn("related_issue_numbers", prompt)

        self.assertTrue(SCHEMA_PATH.is_file(), msg=f"missing schema file: {SCHEMA_PATH}")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

        action_enum = schema["properties"]["action"]["enum"]
        self.assertEqual(action_enum, ["create", "update", "merge", "none"])
        self.assertIn("related_issue_numbers", schema["properties"])
        self.assertNotIn("allOf", schema)
        self.assertNotIn("if", schema)
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(schema["properties"]["issue"]["type"], ["object", "null"])

    def test_help_path(self) -> None:
        result = subprocess.run(
            ["bash", str(SOURCE_SCRIPT), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("--iterations", result.stdout)
        self.assertIn("--max-consecutive-none", result.stdout)
        self.assertIn("--repo", result.stdout)
        self.assertIn("--repo-url", result.stdout)
        self.assertIn("--ref", result.stdout)
        self.assertIn("--workdir", result.stdout)

    def test_single_iteration_create(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            setup_fake_repo(
                root,
                gh_script=gh_script_with_mutations(),
                codex_script=codex_script_for_payload(
                    {
                        "summary": "Found a new bug",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000001.md",
                        "action": "create",
                        "issue": {
                            "title": "Bug: sample",
                            "body": "body",
                            "labels": ["bug", "prio/p1"],
                        },
                    }
                , stdout_lines=["codex progress chatter"]),
            )
            result = run_bug_sweep(root, workdir=root)

            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
            self.assertTrue((root / "target" / "agentic-bug-sweep" / "context" / "open-issues.json").is_file())
            self.assertTrue((root / "target" / "agentic-bug-sweep" / "output" / "iteration-001.json").is_file())

            codex_invocations = (root / "state" / "codex-args.log").read_text(encoding="utf-8")
            self.assertIn("exec", codex_invocations)
            self.assertIn("--output-schema", codex_invocations)
            self.assertIn("--sandbox", codex_invocations)
            self.assertIn("workspace-write", codex_invocations)
            codex_tmpdir = (root / "state" / "codex-tmpdir.log").read_text(encoding="utf-8").strip()
            self.assertEqual(codex_tmpdir, str(root / "target" / "agentic-bug-sweep" / "tmp"))

    def test_remote_repo_url_clone_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            setup_fake_repo(
                root,
                gh_script=gh_script_with_mutations(),
                codex_script=codex_script_for_payload(
                    {
                        "summary": "No actionable bug found",
                        "report_path": "docs/test-reports/agentic-bug-sweep/bug-sweep-20260308-remote.md",
                        "action": "none",
                    }
                ),
            )

            remote_source = root / "remote-source"
            (remote_source / "docs" / "test-reports" / "agentic-bug-sweep").mkdir(parents=True)
            (remote_source / "README.md").write_text("# remote repo\n", encoding="utf-8")
            (remote_source / "docs" / "test-reports" / "agentic-bug-sweep" / "existing.md").write_text(
                "existing report\n", encoding="utf-8"
            )

            write_executable(
                root / "bin" / "git",
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf '%s\\n' "$*" >>"${FAKE_STATE_DIR:?}/git.log"

                    if [[ "$1" == "clone" ]]; then
                      dest="${@: -1}"
                      mkdir -p "$(dirname "$dest")"
                      cp -R "${REMOTE_SOURCE_DIR:?}" "$dest"
                      exit 0
                    fi

                    printf 'unexpected git invocation: %s\\n' "$*" >&2
                    exit 1
                    """
                ),
            )

            result = run_bug_sweep(
                root,
                repo="tensor4all/demo-repo",
                repo_url="https://github.com/tensor4all/demo-repo.git",
                ref="main",
                extra_env={"REMOTE_SOURCE_DIR": str(remote_source)},
            )

            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            git_log = (root / "state" / "git.log").read_text(encoding="utf-8")
            self.assertIn("clone --depth 1 --branch main https://github.com/tensor4all/demo-repo.git", git_log)

            codex_invocations = (root / "state" / "codex-args.log").read_text(encoding="utf-8")
            self.assertIn(str(root / "target" / "agentic-bug-sweep" / "repos" / "tensor4all-demo-repo"), codex_invocations)
            gh_log = (root / "state" / "gh.log").read_text(encoding="utf-8")
            self.assertIn("issue list --repo tensor4all/demo-repo", gh_log)

            prior_reports = (root / "target" / "agentic-bug-sweep" / "context" / "prior-reports.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("existing.md", prior_reports)

    def test_github_actions(self) -> None:
        cases = [
            {
                "name": "create_with_related",
                "payload": {
                    "summary": "Found a new bug",
                    "report_path": "docs/test-reports/bug-sweep-20260308-000002.md",
                    "action": "create",
                    "issue": {
                        "title": "Bug: create path",
                        "body": "Primary repro",
                        "labels": ["bug", "prio/p1"],
                    },
                    "related_issue_numbers": [12],
                    "related_comment": "Likely same root cause as this new finding.",
                },
            },
            {
                "name": "update",
                "payload": {
                    "summary": "Expanded existing issue",
                    "report_path": "docs/test-reports/bug-sweep-20260308-000003.md",
                    "action": "update",
                    "canonical_issue_number": 21,
                    "issue_comment": "New evidence from automation.",
                },
            },
            {
                "name": "merge",
                "payload": {
                    "summary": "Duplicate of an existing issue",
                    "report_path": "docs/test-reports/bug-sweep-20260308-000004.md",
                    "action": "merge",
                    "canonical_issue_number": 31,
                    "issue_comment": "Canonical issue updated with new repro.",
                    "duplicates_to_close": [32, 33],
                    "duplicate_comment": "Closing in favor of #31.",
                },
            },
            {
                "name": "none",
                "payload": {
                    "summary": "No actionable bug found",
                    "report_path": "docs/test-reports/bug-sweep-20260308-000005.md",
                    "action": "none",
                },
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    setup_fake_repo(
                        root,
                        gh_script=gh_script_with_mutations(),
                        codex_script=codex_script_for_payload(case["payload"]),
                    )
                    result = run_bug_sweep(root)

                    self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
                    gh_log = (root / "state" / "gh.log").read_text(encoding="utf-8")

                    if case["name"] == "create_with_related":
                        self.assertIn("issue create", gh_log)
                        create_body = (root / "state" / "create-body.md").read_text(encoding="utf-8")
                        self.assertIn("Primary repro", create_body)
                        self.assertIn("Related issues", create_body)
                        self.assertIn("#12", create_body)
                    elif case["name"] == "update":
                        self.assertIn("issue comment 21", gh_log)
                        comment = (root / "state" / "comment-21.md").read_text(encoding="utf-8")
                        self.assertIn("New evidence from automation.", comment)
                    elif case["name"] == "merge":
                        log_lines = gh_log.splitlines()
                        canonical_index = next(i for i, line in enumerate(log_lines) if "issue comment 31" in line)
                        duplicate_comment_index = next(i for i, line in enumerate(log_lines) if "issue comment 32" in line)
                        duplicate_close_index = next(i for i, line in enumerate(log_lines) if "issue close 32" in line)
                        self.assertLess(canonical_index, duplicate_comment_index)
                        self.assertLess(duplicate_comment_index, duplicate_close_index)
                    elif case["name"] == "none":
                        self.assertNotIn("issue create", gh_log)
                        self.assertNotIn("issue comment", gh_log)
                        self.assertNotIn("issue close", gh_log)

    def test_stop_conditions(self) -> None:
        cases = [
            {
                "name": "max_iterations",
                "iterations": "2",
                "max_consecutive_none": "5",
                "payloads": [
                    {
                        "summary": "No actionable bug found",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000010.md",
                        "action": "none",
                    },
                    {
                        "summary": "No actionable bug found again",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000011.md",
                        "action": "none",
                    },
                ],
                "expected_invocations": 2,
                "expected_stop_reason": "completed_max_iterations",
            },
            {
                "name": "consecutive_none_threshold",
                "iterations": "5",
                "max_consecutive_none": "2",
                "payloads": [
                    {
                        "summary": "No actionable bug found",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000012.md",
                        "action": "none",
                    },
                    {
                        "summary": "No actionable bug found again",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000013.md",
                        "action": "none",
                    },
                    {
                        "summary": "This payload should not be consumed",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000014.md",
                        "action": "create",
                        "issue": {
                            "title": "Bug: unreachable",
                            "body": "body",
                            "labels": ["bug"],
                        },
                    },
                ],
                "expected_invocations": 2,
                "expected_stop_reason": "completed_consecutive_none_threshold",
            },
            {
                "name": "productive_iteration_resets_none_counter",
                "iterations": "4",
                "max_consecutive_none": "2",
                "payloads": [
                    {
                        "summary": "No actionable bug found",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000015.md",
                        "action": "none",
                    },
                    {
                        "summary": "Found a new bug",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000016.md",
                        "action": "create",
                        "issue": {
                            "title": "Bug: reset counter",
                            "body": "body",
                            "labels": ["bug"],
                        },
                    },
                    {
                        "summary": "No actionable bug found after create",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000017.md",
                        "action": "none",
                    },
                    {
                        "summary": "No actionable bug found again",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000018.md",
                        "action": "none",
                    },
                ],
                "expected_invocations": 4,
                "expected_stop_reason": "completed_consecutive_none_threshold",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    setup_fake_repo(
                        root,
                        gh_script=gh_script_with_mutations(),
                        codex_script=codex_script_for_payloads(case["payloads"]),
                    )
                    result = run_bug_sweep(
                        root,
                        iterations=case["iterations"],
                        max_consecutive_none=case["max_consecutive_none"],
                    )

                    self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

                    codex_invocation_count = int((root / "state" / "codex-counter.txt").read_text(encoding="utf-8"))
                    self.assertEqual(codex_invocation_count, case["expected_invocations"])

                    summary_path = root / "target" / "agentic-bug-sweep" / "output" / "run-summary.json"
                    self.assertTrue(summary_path.is_file())
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    self.assertEqual(summary["iterations_run"], case["expected_invocations"])
                    self.assertEqual(summary["stop_reason"], case["expected_stop_reason"])

    def test_failure_paths(self) -> None:
        cases = [
            {
                "name": "failed_codex_exec",
                "gh_script": gh_script_with_mutations(),
                "codex_script": (
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "printf 'call\\n' >>\"${FAKE_STATE_DIR:?}/codex.log\"\n"
                    "exit 23\n"
                ),
                "expected_stop_reason": "failed_codex_exec",
                "expect_summary": True,
                "expect_iteration_output": False,
            },
            {
                "name": "failed_invalid_json",
                "gh_script": gh_script_with_mutations(),
                "codex_script": (
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "printf 'call\\n' >>\"${FAKE_STATE_DIR:?}/codex.log\"\n"
                    "output_path=\"\"\n"
                    "prev=\"\"\n"
                    "for arg in \"$@\"; do\n"
                    "  if [[ \"$prev\" == \"-o\" || \"$prev\" == \"--output-last-message\" ]]; then\n"
                    "    output_path=\"$arg\"\n"
                    "  fi\n"
                    "  prev=\"$arg\"\n"
                    "done\n"
                    "printf 'not-json\\n' >\"${output_path:?}\"\n"
                    "exit 0\n"
                ),
                "expected_stop_reason": "failed_invalid_json",
                "expect_summary": True,
                "expect_iteration_output": True,
            },
            {
                "name": "failed_github_mutation",
                "gh_script": gh_script_with_mutations(fail_comment=True),
                "codex_script": codex_script_for_payload(
                    {
                        "summary": "Expanded existing issue",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000020.md",
                        "action": "update",
                        "canonical_issue_number": 21,
                        "issue_comment": "New evidence from automation.",
                    }
                ),
                "expected_stop_reason": "failed_github_mutation",
                "expect_summary": True,
                "expect_iteration_output": True,
            },
            {
                "name": "failed_lock_acquisition",
                "gh_script": gh_script_with_mutations(),
                "codex_script": codex_script_for_payload(
                    {
                        "summary": "No actionable bug found",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000021.md",
                        "action": "none",
                    }
                ),
                "expected_message": "failed to acquire lock",
                "expect_summary": False,
                "expect_iteration_output": False,
                "precreate_lock": True,
            },
            {
                "name": "failed_invalid_contract",
                "gh_script": gh_script_with_mutations(),
                "codex_script": codex_script_for_payload(
                    {
                        "summary": "Bad create payload",
                        "report_path": "docs/test-reports/bug-sweep-20260308-000022.md",
                        "action": "create",
                    }
                ),
                "expected_stop_reason": "failed_invalid_contract",
                "expect_summary": True,
                "expect_iteration_output": True,
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    setup_fake_repo(
                        root,
                        gh_script=case["gh_script"],
                        codex_script=case["codex_script"],
                    )
                    if case.get("precreate_lock"):
                        (root / "target" / "agentic-bug-sweep" / "lock").mkdir(parents=True)

                    result = run_bug_sweep(root)

                    self.assertNotEqual(result.returncode, 0)
                    if "expected_message" in case:
                        self.assertIn(case["expected_message"], result.stdout + result.stderr)

                    summary_path = root / "target" / "agentic-bug-sweep" / "output" / "run-summary.json"
                    self.assertEqual(summary_path.is_file(), case["expect_summary"])
                    if case["expect_summary"]:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        self.assertEqual(summary["stop_reason"], case["expected_stop_reason"])

                    iteration_output_path = (
                        root / "target" / "agentic-bug-sweep" / "output" / "iteration-001.json"
                    )
                    self.assertEqual(iteration_output_path.is_file(), case["expect_iteration_output"])


if __name__ == "__main__":
    unittest.main()
