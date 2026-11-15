"""Comprehensive unit tests for RSI-Harness inner and outer loops."""

from __future__ import annotations

import json
import os
import shutil
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

from evidence_reducer import reduce_log  # noqa: E402
from evolve_outer import run_evolution_cycle  # noqa: E402
from guard_harness import evaluate_tool_guard  # noqa: E402
from inner_gate import evaluate_stop  # noqa: E402
from trace_collector import record_event  # noqa: E402


class GuardHarnessTests(unittest.TestCase):
    def test_protects_hooks_and_benchmarks(self) -> None:
        os.environ.pop("RSI_EVOLUTION_MODE", None)
        res_hooks = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "toolCall": {
                    "name": "write_file",
                    "args": {"Path": str(PLUGIN_DIR / "hooks.json")},
                },
            }
        )
        self.assertEqual(res_hooks["decision"], "ask")

        res_bench = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "toolCall": {
                    "name": "run_command",
                    "args": {"CommandLine": "rm benchmarks/tasks.json"},
                },
            }
        )
        self.assertEqual(res_bench["decision"], "ask")

    def test_allows_regular_source_and_evolution_mode(self) -> None:
        res_normal = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "toolCall": {
                    "name": "edit_file",
                    "args": {"Path": str(REPO_ROOT / "src" / "tasklib.py")},
                },
            }
        )
        self.assertEqual(res_normal["decision"], "allow")

        os.environ["RSI_EVOLUTION_MODE"] = "1"
        try:
            res_evo = evaluate_tool_guard(
                {
                    "cwd": str(REPO_ROOT),
                    "toolCall": {
                        "name": "write_file",
                        "args": {"Path": str(PLUGIN_DIR / "hooks.json")},
                    },
                }
            )
            self.assertEqual(res_evo["decision"], "allow")
        finally:
            os.environ.pop("RSI_EVOLUTION_MODE", None)


class EvidenceReducerTests(unittest.TestCase):
    def test_short_log_passes_through(self) -> None:
        res = reduce_log("all tests ok", max_chars=500)
        self.assertFalse(res["was_reduced"])
        self.assertTrue(res["verified"])
        self.assertEqual(res["reduced_text"], "all tests ok")

    def test_verbatim_quotes_accepted(self) -> None:
        noise = "\n".join(f"info line {i} padding " + "x" * 60 for i in range(200))
        raw = noise + "\nAssertionError: expected 42 got 0\nDONE\n"
        res = reduce_log(
            raw,
            max_chars=1000,
            summarizer=lambda _: {
                "summary": "Test assertion failed.",
                "quotes": ["AssertionError: expected 42 got 0"],
            },
        )
        self.assertTrue(res["was_reduced"])
        self.assertTrue(res["verified"])
        self.assertFalse(res["fallback_used"])
        self.assertIn("AssertionError: expected 42 got 0", res["reduced_text"])

    def test_hallucinated_quote_triggers_fallback(self) -> None:
        noise = "\n".join(f"info line {i} padding " + "x" * 60 for i in range(200))
        raw = noise + "\nValueError: zero denominator\nFAILED (errors=1)\n"
        res = reduce_log(
            raw,
            max_chars=1000,
            summarizer=lambda _: {
                "summary": "Hallucinated summary.",
                "quotes": ["ValueError: fabricated message not in log"],
            },
        )
        self.assertTrue(res["was_reduced"])
        self.assertFalse(res["verified"])
        self.assertTrue(res["fallback_used"])
        self.assertIn("ValueError: zero denominator", res["reduced_text"])


class TraceAndInnerGateTests(unittest.TestCase):
    def test_trace_collector_and_inner_gate_loop_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / "rsi.config.json").write_text(
                json.dumps(
                    {
                        "inner_check_cmd": "python3 -c \"import sys; print('AssertionError: boom'); sys.exit(1)\"",
                        "max_inner_retries": 2,
                    }
                ),
                encoding="utf-8",
            )
            transcript = cwd / "transcript.jsonl"
            transcript.write_text(
                json.dumps(
                    {
                        "step_index": 0,
                        "status": "DONE",
                        "input_tokens": 100,
                        "cache_read_tokens": 20,
                        "output_tokens": 30,
                        "tool_calls": [{"name": "read_file"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            oversized_payload = {
                "conversationId": "conv-1",
                "cwd": str(cwd),
                "stepIdx": 1,
                "transcriptPath": str(transcript),
                "toolCall": {"name": "write_file", "args": {"Content": "A" * 12000}},
            }
            summary = record_event(oversized_payload, "post_tool")
            self.assertEqual(summary["total_tokens"], 150)
            self.assertTrue((cwd / ".rsi" / "artifacts" / "conv-1" / "step_1.json").is_file())

            stop_payload = {
                "conversationId": "conv-1",
                "cwd": str(cwd),
                "transcriptPath": str(transcript),
                "continuationActive": False,
            }
            r1 = evaluate_stop(stop_payload)
            self.assertEqual(r1["decision"], "continue")
            self.assertIn("AssertionError: boom", r1["reason"])

            r2 = evaluate_stop(stop_payload)
            self.assertEqual(r2["decision"], "continue")

            # Retry cap reached -> must allow stop
            r3 = evaluate_stop(stop_payload)
            self.assertEqual(r3, {})

            # continuationActive=True -> must allow stop immediately
            (cwd / ".rsi" / "traces" / "conv-1" / "retry_count.txt").write_text("0")
            r_cont = evaluate_stop({**stop_payload, "continuationActive": True})
            self.assertEqual(r_cont, {})


class OuterLoopTests(unittest.TestCase):
    def _create_temp_git_repo(self, dest: Path) -> None:
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

    def test_rrsi_rejects_bad_proposals_and_accepts_valid_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            self._create_temp_git_repo(repo)
            main_before = subprocess.check_output(
                ["git", "rev-parse", "main"], cwd=repo, text=True
            ).strip()

            # 1. Leak rejected by Critic
            leak_res = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock", mock_bad_proposal="leak")
            self.assertFalse(leak_res["accepted"])
            self.assertTrue(any("critic_benchmark_leakage" in r for r in leak_res["reasons"]))

            # 2. Edit budget exceeded rejected
            budget_res = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock", mock_bad_proposal="budget")
            self.assertFalse(budget_res["accepted"])
            self.assertTrue(any("edit_budget_lines_exceeded" in r for r in budget_res["reasons"]))

            # 3. Protected file modification rejected
            prot_res = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock", mock_bad_proposal="protected")
            self.assertFalse(prot_res["accepted"])
            self.assertTrue(any("protected_file_modified" in r for r in prot_res["reasons"]))

            # 4. Valid improvement accepted & auto-applied to rsi/evolved while main stays intact
            ok_res = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock")
            self.assertTrue(ok_res["accepted"], msg=str(ok_res))
            self.assertGreater(ok_res["candidate"]["pass_rate"], ok_res["baseline"]["pass_rate"])
            self.assertLess(ok_res["candidate"]["total_tokens"], ok_res["baseline"]["total_tokens"])

            main_after = subprocess.check_output(
                ["git", "rev-parse", "main"], cwd=repo, text=True
            ).strip()
            evolved_sha = subprocess.check_output(
                ["git", "rev-parse", "rsi/evolved"], cwd=repo, text=True
            ).strip()
            self.assertEqual(main_before, main_after)
            self.assertEqual(evolved_sha, ok_res["commit_sha"])


if __name__ == "__main__":
    unittest.main()
