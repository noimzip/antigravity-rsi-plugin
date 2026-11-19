"""Integration tests for agy runner, live CLI hooks, and Selection/Critic boundaries."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PLUGIN_DIR / "scripts"
REPO_ROOT = PLUGIN_DIR.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from common import load_jsonl, load_rsi_config  # noqa: E402
from evolve_outer import (  # noqa: E402
    evaluate_selection_gate,
    run_evolution_cycle,
    validate_proposal_rrsi,
)
from run_bench import run_benchmark  # noqa: E402


FAKE_AGY_SCRIPT = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
cwd = Path.cwd().resolve()

if os.environ.get("FAKE_AGY_MODE") == "malformed":
    print("NOT-JSON-OUTPUT")
    sys.exit(2)

if os.environ.get("FAKE_AGY_MODE") == "empty":
    sys.exit(1)

# Proposer mode inside outer loop worktree
if os.environ.get("RSI_EVOLUTION_MODE") == "1":
    allowed = os.environ.get("RSI_ALLOWED_ROOT", "")
    if not allowed or Path(allowed).resolve() != cwd:
        sys.stderr.write(f"RSI_ALLOWED_ROOT mismatch: {allowed} != {cwd}\\n")
        sys.exit(3)
    rules_path = cwd / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"
    current = rules_path.read_text(encoding="utf-8") if rules_path.is_file() else ""
    addition = (
        "\\n- 文字列パースやログ解析では空行・境界・分割数を検証し、"
        "編集後は必ず unittest を再実行して既存の公開関数・型注釈・定数仕様を維持すること。\\n"
    )
    rules_path.write_text(current.rstrip() + addition, encoding="utf-8")
    print(json.dumps({"status": "ok", "mode": "proposer"}))
    sys.exit(0)

# Benchmark task runner mode
rules_file = cwd / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"
rules_text = rules_file.read_text(encoding="utf-8") if rules_file.is_file() else ""
sol_path = cwd / "solution.py"
sol_text = sol_path.read_text(encoding="utf-8") if sol_path.is_file() else ""

if "safe_ratio_sum" in sol_text:
    sol_path.write_text(
        "def safe_ratio_sum(pairs: list[tuple[float, float]]) -> float:\\n"
        "    if not pairs:\\n"
        "        return 0.0\\n"
        "    total = 0.0\\n"
        "    for num, den in pairs:\\n"
        "        if den == 0:\\n"
        "            raise ValueError('zero')\\n"
        "        total += num / den\\n"
        "    return total\\n",
        encoding="utf-8",
    )
    usage = {"total_tokens": 800}
    conv_id = "agy-task-01"
elif "parse_kv_lines" in sol_text:
    sol_path.write_text(
        "def parse_kv_lines(text: str) -> dict[str, str]:\\n"
        "    out = {}\\n"
        "    for line in text.splitlines():\\n"
        "        s = line.strip()\\n"
        "        if not s or '=' not in s:\\n"
        "            continue\\n"
        "        k, v = s.split('=', 1)\\n"
        "        out[k] = v\\n"
        "    return out\\n",
        encoding="utf-8",
    )
    # Test input_tokens + output_tokens fallback extraction when total_tokens is absent
    usage = {"input_tokens": 700, "output_tokens": 250}
    conv_id = "agy-task-02"
elif "format_summary" in sol_text:
    if "公開" in rules_text or "定数" in rules_text:
        sol_path.write_text(
            "REPORT_VERSION = 'v1'\\n\\n"
            "def format_summary(items: list[str], limit: int = 3) -> str:\\n"
            "    if not items:\\n"
            "        return f'{REPORT_VERSION}: (empty)'\\n"
            "    head = ', '.join(items[:limit])\\n"
            "    extra = len(items) - len(items[:limit])\\n"
            "    if extra > 0:\\n"
            "        return f'{REPORT_VERSION}: {head} (+{extra} more)'\\n"
            "    return f'{REPORT_VERSION}: {head}'\\n",
            encoding="utf-8",
        )
        usage = {"total_tokens": 750}
    else:
        usage = {"total_tokens": 2400}
    conv_id = "agy-task-03"
else:
    usage = {"total_tokens": 500}
    conv_id = "agy-unknown"

trace_dir = cwd / ".rsi" / "traces" / conv_id
trace_dir.mkdir(parents=True, exist_ok=True)
(trace_dir / "summary.json").write_text(
    json.dumps({"conversationId": conv_id, "total_tokens": usage.get("total_tokens", 950)}),
    encoding="utf-8",
)
print(json.dumps({"usage": usage, "result": "done"}))
"""


def _init_temp_git_repo(dest: Path) -> None:
    shutil.copytree(
        REPO_ROOT,
        dest,
        ignore=shutil.ignore_patterns(".git", ".rsi", "__pycache__"),
        dirs_exist_ok=True,
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=dest, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "RSI Test"], cwd=dest, check=True)
    subprocess.run(["git", "config", "user.email", "rsi@example.com"], cwd=dest, check=True)
    subprocess.run(["git", "add", "-A"], cwd=dest, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=dest, check=True, capture_output=True)


class AgyRunnerIntegrationTests(unittest.TestCase):
    def _install_fake_agy(self) -> Path:
        bin_dir = Path(tempfile.mkdtemp(prefix="fake-agy-bin-"))
        self.addCleanup(lambda: shutil.rmtree(bin_dir, ignore_errors=True))
        agy_bin = bin_dir / "agy"
        agy_bin.write_text(FAKE_AGY_SCRIPT, encoding="utf-8")
        agy_bin.chmod(agy_bin.stat().st_mode | stat.S_IEXEC)

        orig_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{orig_path}"
        self.addCleanup(lambda: os.environ.__setitem__("PATH", orig_path))
        return bin_dir

    def test_run_benchmark_with_fake_agy_extracts_tokens_and_traces(self) -> None:
        self._install_fake_agy()
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "exported"
            report = run_benchmark(
                REPO_ROOT,
                REPO_ROOT / "benchmarks" / "tasks.json",
                runner="agy",
                trace_export_dir=export_dir,
            )
            self.assertEqual(report["runner"], "agy")
            self.assertEqual(report["total_tasks"], 3)
            self.assertEqual(report["passed_tasks"], 2)
            # 800 + (700 + 250) + 2400 = 4150
            self.assertEqual(report["total_tokens"], 4150)
            self.assertTrue((export_dir / "agy-task-01" / "summary.json").is_file())
            self.assertTrue((export_dir / "agy-task-02" / "summary.json").is_file())
            self.assertTrue((export_dir / "agy-task-03" / "summary.json").is_file())

    def test_evolve_outer_runner_agy_end_to_end(self) -> None:
        self._install_fake_agy()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            _init_temp_git_repo(repo)
            main_before = subprocess.check_output(
                ["git", "rev-parse", "main"], cwd=repo, text=True
            ).strip()

            proc = subprocess.run(
                [
                    sys.executable,
                    str(repo / ".agents" / "plugins" / "rsi-harness" / "scripts" / "evolve_outer.py"),
                    "--repo",
                    str(repo),
                    "--branch",
                    "rsi/evolved",
                    "--runner",
                    "agy",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            result = json.loads(proc.stdout)
            self.assertTrue(result["accepted"], msg=str(result))
            self.assertEqual(result["baseline"]["total_tokens"], 4150)
            self.assertEqual(result["candidate"]["total_tokens"], 2500)
            self.assertEqual(result["baseline"]["passed_tasks"], 2)
            self.assertEqual(result["candidate"]["passed_tasks"], 3)

            main_after = subprocess.check_output(
                ["git", "rev-parse", "main"], cwd=repo, text=True
            ).strip()
            evolved_sha = subprocess.check_output(
                ["git", "rev-parse", "rsi/evolved"], cwd=repo, text=True
            ).strip()
            self.assertEqual(main_before, main_after)
            self.assertEqual(evolved_sha, result["commit_sha"])

    def test_run_benchmark_handles_malformed_and_empty_agy_output(self) -> None:
        self._install_fake_agy()
        for mode in ("malformed", "empty"):
            os.environ["FAKE_AGY_MODE"] = mode
            try:
                report = run_benchmark(
                    REPO_ROOT,
                    REPO_ROOT / "benchmarks" / "tasks.json",
                    runner="agy",
                )
            finally:
                os.environ.pop("FAKE_AGY_MODE", None)
            self.assertEqual(report["total_tasks"], 3)
            self.assertEqual(report["passed_tasks"], 0)
            self.assertEqual(report["total_tokens"], 0)


class CliHookIntegrationTests(unittest.TestCase):
    def _run_hook(
        self,
        script_name: str,
        payload: dict[str, object],
        *,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, object]:
        cmd = [sys.executable, str(SCRIPTS_DIR / script_name), *(args or [])]
        env = dict(os.environ)
        env.pop("RSI_EVOLUTION_MODE", None)
        env.pop("RSI_ALLOWED_ROOT", None)
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=20,
        )
        return json.loads(proc.stdout.strip())

    def test_guard_harness_cli_allow_ask_deny(self) -> None:
        allow_res = self._run_hook(
            "guard_harness.py",
            {
                "cwd": str(REPO_ROOT),
                "toolCall": {"name": "edit_file", "args": {"Path": "src/tasklib.py"}},
            },
        )
        self.assertEqual(allow_res["decision"], "allow")

        ask_res = self._run_hook(
            "guard_harness.py",
            {
                "cwd": str(REPO_ROOT),
                "toolCall": {
                    "name": "write_file",
                    "args": {"Path": ".agents/plugins/rsi-harness/hooks.json"},
                },
            },
        )
        self.assertEqual(ask_res["decision"], "ask")

        deny_res = self._run_hook(
            "guard_harness.py",
            {
                "cwd": str(REPO_ROOT),
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "echo pwn > benchmarks/tasks.json"},
                },
            },
            env_overrides={
                "RSI_EVOLUTION_MODE": "1",
                "RSI_ALLOWED_ROOT": str(REPO_ROOT),
            },
        )
        self.assertEqual(deny_res["decision"], "deny")

    def test_trace_collector_and_inner_gate_cli_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / "rsi.config.json").write_text(
                json.dumps(
                    {
                        "inner_check_cmd": "python3 -c \"import sys; print('AssertionError: cli fail'); sys.exit(1)\"",
                        "max_inner_retries": 1,
                    }
                ),
                encoding="utf-8",
            )

            post_res = self._run_hook(
                "trace_collector.py",
                {
                    "conversationId": "cli-conv",
                    "cwd": str(cwd),
                    "stepIdx": 2,
                    "toolCall": {"name": "write_file", "args": {"Content": "Z" * 12000}},
                },
                args=["--event", "post_tool"],
            )
            self.assertEqual(post_res, {})
            self.assertTrue((cwd / ".rsi" / "traces" / "cli-conv" / "summary.json").is_file())
            events = load_jsonl(cwd / ".rsi" / "traces" / "cli-conv" / "events.jsonl")
            self.assertEqual(len(events), 1)
            self.assertIn("observationHandle", events[0])

            stop_payload = {
                "conversationId": "cli-conv",
                "cwd": str(cwd),
                "continuationActive": False,
            }
            stop1 = self._run_hook("inner_gate.py", stop_payload)
            self.assertEqual(stop1["decision"], "continue")
            self.assertIn("AssertionError: cli fail", stop1["reason"])

            # Retry cap (max_inner_retries=1) reached -> allows stop
            stop2 = self._run_hook("inner_gate.py", stop_payload)
            self.assertEqual(stop2, {})

            # Passing command -> allows stop immediately
            (cwd / ".rsi" / "traces" / "cli-conv" / "retry_count.txt").write_text("0")
            (cwd / "rsi.config.json").write_text(
                json.dumps({"inner_check_cmd": "python3 -c \"print('ok')\"", "max_inner_retries": 2}),
                encoding="utf-8",
            )
            stop_pass = self._run_hook("inner_gate.py", stop_payload)
            self.assertEqual(stop_pass, {})


class SelectionGateAndCriticBoundaryTests(unittest.TestCase):
    def test_selection_gate_token_cap_and_zero_baseline_boundaries(self) -> None:
        cfg = load_rsi_config(REPO_ROOT)

        # >1.15x token overhead rejected even with full pass-rate jump
        ok_over, reasons_over = evaluate_selection_gate(
            {"pass_rate": 0.0, "total_tokens": 1000},
            {"pass_rate": 1.0, "total_tokens": 1151},
            cfg,
        )
        self.assertFalse(ok_over)
        self.assertTrue(any("token_overhead_too_high" in r for r in reasons_over))

        # Exact 1.15x boundary allowed
        ok_exact, reasons_exact = evaluate_selection_gate(
            {"pass_rate": 0.0, "total_tokens": 1000},
            {"pass_rate": 1.0, "total_tokens": 1150},
            cfg,
        )
        self.assertTrue(ok_exact, msg=str(reasons_exact))

        # Zero baseline tokens handled cleanly without ZeroDivisionError
        ok_zero, _ = evaluate_selection_gate(
            {"pass_rate": 0.5, "total_tokens": 0},
            {"pass_rate": 0.5, "total_tokens": 0},
            cfg,
        )
        self.assertFalse(ok_zero)

    def test_validate_proposal_rrsi_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            _init_temp_git_repo(repo)
            cfg = load_rsi_config(repo)
            base_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            benchmark_ids = ["task_01_edge_case", "task_02_noisy_log"]

            # 1. No changes
            ok_none, reasons_none, _ = validate_proposal_rrsi(repo, base_sha, cfg, [], benchmark_ids)
            self.assertFalse(ok_none)
            self.assertIn("no_changes", reasons_none)

            # 2. Out-of-scope file
            (repo / "src" / "tasklib.py").write_text("# modified\n", encoding="utf-8")
            ok_scope, reasons_scope, _ = validate_proposal_rrsi(repo, base_sha, cfg, [], benchmark_ids)
            self.assertFalse(ok_scope)
            self.assertTrue(any("out_of_scope_file" in r for r in reasons_scope))
            subprocess.run(["git", "checkout", "--", "src/tasklib.py"], cwd=repo, check=True)

            # 3. Too many files (> max_files=2)
            skills_dir = repo / ".agents" / "plugins" / "rsi-harness" / "skills"
            rules_file = repo / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"
            rules_file.write_text(rules_file.read_text(encoding="utf-8") + "\n- extra 1\n", encoding="utf-8")
            (skills_dir / "s1.md").write_text("# s1\n", encoding="utf-8")
            (skills_dir / "s2.md").write_text("# s2\n", encoding="utf-8")
            ok_files, reasons_files, _ = validate_proposal_rrsi(repo, base_sha, cfg, [], benchmark_ids)
            self.assertFalse(ok_files)
            self.assertTrue(any("edit_budget_files_exceeded" in r for r in reasons_files))
            subprocess.run(["git", "reset", "--hard", base_sha], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=repo, check=True, capture_output=True)

            # 4. Duplicate diff_hash
            rules_file.write_text(rules_file.read_text(encoding="utf-8") + "\n- unique rule\n", encoding="utf-8")
            ok_first, _, meta_first = validate_proposal_rrsi(repo, base_sha, cfg, [], benchmark_ids)
            self.assertTrue(ok_first)
            ok_dup, reasons_dup, _ = validate_proposal_rrsi(
                repo,
                base_sha,
                cfg,
                [{"diff_hash": meta_first["diff_hash"]}],
                benchmark_ids,
            )
            self.assertFalse(ok_dup)
            self.assertIn("duplicate_hypothesis", reasons_dup)

            # 5. Rules byte budget exceeded
            small_cfg = json.loads(json.dumps(cfg))
            small_cfg["rrsi"]["max_rules_bytes"] = 20
            ok_bytes, reasons_bytes, _ = validate_proposal_rrsi(repo, base_sha, small_cfg, [], benchmark_ids)
            self.assertFalse(ok_bytes)
            self.assertTrue(any("rules_budget_exceeded" in r for r in reasons_bytes))
            subprocess.run(["git", "reset", "--hard", base_sha], cwd=repo, check=True, capture_output=True)

            # 6. Critic leakage tokens (task IDs and grading filenames)
            for token in ("task_01_edge_case", "test_solution.py", "run_bench.py", "tasks.json"):
                subprocess.run(["git", "reset", "--hard", base_sha], cwd=repo, check=True, capture_output=True)
                rules_file.write_text(
                    rules_file.read_text(encoding="utf-8") + f"\n- hint mentioning {token}\n",
                    encoding="utf-8",
                )
                ok_leak, reasons_leak, _ = validate_proposal_rrsi(repo, base_sha, cfg, [], benchmark_ids)
                self.assertFalse(ok_leak, msg=f"Expected leak detection for {token}")
                self.assertIn(f"critic_benchmark_leakage:{token}", reasons_leak)


if __name__ == "__main__":
    unittest.main()
