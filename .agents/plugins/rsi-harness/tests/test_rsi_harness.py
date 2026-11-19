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

from common import load_jsonl, load_rsi_config  # noqa: E402
from evidence_reducer import reduce_log  # noqa: E402
from evolve_outer import (  # noqa: E402
    compute_selection_diagnostics,
    evaluate_selection_gate,
    run_evolution_cycle,
)
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
        os.environ["RSI_ALLOWED_ROOT"] = str(REPO_ROOT)
        try:
            # Editable rules file inside allowed worktree root -> allowed
            res_rules = evaluate_tool_guard(
                {
                    "cwd": str(REPO_ROOT),
                    "toolCall": {
                        "name": "write_file",
                        "args": {"Path": str(PLUGIN_DIR / "rules" / "AGENTS.md")},
                    },
                }
            )
            self.assertEqual(res_rules["decision"], "allow")

            # Protected hooks.json even in evolution mode -> denied
            res_hooks_evo = evaluate_tool_guard(
                {
                    "cwd": str(REPO_ROOT),
                    "toolCall": {
                        "name": "write_file",
                        "args": {"Path": str(PLUGIN_DIR / "hooks.json")},
                    },
                }
            )
            self.assertEqual(res_hooks_evo["decision"], "deny")

            # Path outside RSI_ALLOWED_ROOT -> denied
            res_outside = evaluate_tool_guard(
                {
                    "cwd": str(REPO_ROOT),
                    "toolCall": {
                        "name": "edit_file",
                        "args": {"Path": "/tmp/outside-worktree/AGENTS.md"},
                    },
                }
            )
            self.assertEqual(res_outside["decision"], "deny")
        finally:
            os.environ.pop("RSI_EVOLUTION_MODE", None)
            os.environ.pop("RSI_ALLOWED_ROOT", None)

    def test_relative_traversal_and_non_prefix_names(self) -> None:
        os.environ.pop("RSI_EVOLUTION_MODE", None)
        for rel in (
            "src/../benchmarks/tasks.json",
            ".agents/plugins/rsi-harness/rules/../hooks.json",
            ".agents/plugins/rsi-harness/skills/../scripts/common.py",
        ):
            res = evaluate_tool_guard(
                {
                    "cwd": str(REPO_ROOT),
                    "toolCall": {"name": "edit_file", "args": {"Path": rel}},
                }
            )
            self.assertEqual(res["decision"], "ask", msg=f"Expected ask for {rel}: {res}")

        # Similar-prefix paths outside protected directories must not be falsely blocked
        res_sibling = evaluate_tool_guard(
            {
                "cwd": str(REPO_ROOT),
                "toolCall": {"name": "write_file", "args": {"Path": "benchmarks_notes/readme.md"}},
            }
        )
        self.assertEqual(res_sibling["decision"], "allow")

    def test_symlink_escape_blocked_in_file_and_shell_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            outside = root / "outside"
            outside.mkdir(parents=True)
            (outside / "secret.txt").write_text("topsecret", encoding="utf-8")

            rules_dir = repo / ".agents" / "plugins" / "rsi-harness" / "rules"
            bench_dir = repo / "benchmarks"
            rules_dir.mkdir(parents=True)
            bench_dir.mkdir(parents=True)
            (bench_dir / "tasks.json").write_text("{}", encoding="utf-8")
            (repo / ".agents" / "plugins" / "rsi-harness" / "hooks.json").write_text("{}", encoding="utf-8")

            link_to_bench = rules_dir / "bench_link.json"
            link_to_bench.symlink_to(bench_dir / "tasks.json")
            link_to_outside = rules_dir / "outside_link.txt"
            link_to_outside.symlink_to(outside / "secret.txt")

            # Normal mode: symlink into benchmarks -> ask
            os.environ.pop("RSI_EVOLUTION_MODE", None)
            res_norm = evaluate_tool_guard(
                {
                    "cwd": str(repo),
                    "toolCall": {"name": "write_file", "args": {"Path": str(link_to_bench)}},
                }
            )
            self.assertEqual(res_norm["decision"], "ask")

            # Evolution mode: symlink to protected OR symlink outside allowed_root -> deny
            os.environ["RSI_EVOLUTION_MODE"] = "1"
            os.environ["RSI_ALLOWED_ROOT"] = str(repo)
            try:
                res_evo_bench = evaluate_tool_guard(
                    {
                        "cwd": str(repo),
                        "toolCall": {"name": "edit_file", "args": {"Path": str(link_to_bench)}},
                    }
                )
                self.assertEqual(res_evo_bench["decision"], "deny")

                res_evo_out = evaluate_tool_guard(
                    {
                        "cwd": str(repo),
                        "toolCall": {"name": "write_file", "args": {"Path": str(link_to_outside)}},
                    }
                )
                self.assertEqual(res_evo_out["decision"], "deny")

                res_shell_link = evaluate_tool_guard(
                    {
                        "cwd": str(repo),
                        "toolCall": {
                            "name": "run_command",
                            "args": {
                                "CommandLine": "echo hacked > .agents/plugins/rsi-harness/rules/bench_link.json"
                            },
                        },
                    }
                )
                self.assertEqual(res_shell_link["decision"], "deny")
            finally:
                os.environ.pop("RSI_EVOLUTION_MODE", None)
                os.environ.pop("RSI_ALLOWED_ROOT", None)

    def test_shell_redirects_mutating_and_readonly_commands(self) -> None:
        mutating_cmds = [
            "echo x > benchmarks/tasks.json",
            "cat a >> .agents/plugins/rsi-harness/hooks.json",
            "echo x | tee .agents/plugins/rsi-harness/scripts/common.py",
            'sh -c "echo x > benchmarks/tasks.json"',
            "rm benchmarks/tasks.json",
            "mv benchmarks/tasks.json tmp.json",
            "cp src/tasklib.py benchmarks/tasks.json",
            "sed -i 's/a/b/' benchmarks/tasks.json",
            "python3 -c \"open('benchmarks/tasks.json','w').write('{}')\"",
            "echo $(echo x > benchmarks/tasks.json)",
        ]
        os.environ["RSI_EVOLUTION_MODE"] = "1"
        os.environ["RSI_ALLOWED_ROOT"] = str(REPO_ROOT)
        try:
            for cmd in mutating_cmds:
                res = evaluate_tool_guard(
                    {
                        "cwd": str(REPO_ROOT),
                        "toolCall": {"name": "run_command", "args": {"CommandLine": cmd}},
                    }
                )
                self.assertEqual(res["decision"], "deny", msg=f"Expected deny for: {cmd} -> {res}")

            readonly_cmds = [
                "cat .agents/plugins/rsi-harness/scripts/guard_harness.py",
                "grep -n foo .agents/plugins/rsi-harness/scripts/common.py",
                "ls -la benchmarks/",
                "sed -n '1,5p' benchmarks/run_bench.py",
                "python3 -m unittest discover -s .agents/plugins/rsi-harness/tests",
                "git diff benchmarks/run_bench.py",
                "ls -la benchmarks/ 2>/dev/null",
            ]
            for cmd in readonly_cmds:
                res = evaluate_tool_guard(
                    {
                        "cwd": str(REPO_ROOT),
                        "toolCall": {"name": "run_command", "args": {"CommandLine": cmd}},
                    }
                )
                self.assertEqual(res["decision"], "allow", msg=f"False block for: {cmd} -> {res}")
        finally:
            os.environ.pop("RSI_EVOLUTION_MODE", None)
            os.environ.pop("RSI_ALLOWED_ROOT", None)


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
            self.assertIn("traces=3", ok_res["proposer_note"])

            main_after = subprocess.check_output(
                ["git", "rev-parse", "main"], cwd=repo, text=True
            ).strip()
            evolved_sha = subprocess.check_output(
                ["git", "rev-parse", "rsi/evolved"], cwd=repo, text=True
            ).strip()
            self.assertEqual(main_before, main_after)
            self.assertEqual(evolved_sha, ok_res["commit_sha"])

            # History records multi-trial stats and selection diagnostics
            history_records = load_jsonl(repo / ".rsi" / "evolution_history.jsonl")
            last_rec = history_records[-1]
            self.assertEqual(last_rec["n_trials"], 3)
            self.assertEqual(last_rec["baseline"]["n_trials"], 3)
            self.assertIn("stats", last_rec["baseline"])
            self.assertIn("stats", last_rec["candidate"])
            self.assertIn("selection_diagnostics", last_rec)

    def test_materialize_task_writes_task_rsi_config(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
        from run_bench import _materialize_task  # noqa: E402

        tasks = json.loads((REPO_ROOT / "benchmarks" / "tasks.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "t1"
            dest.mkdir()
            _materialize_task(tasks["tasks"][0], dest, REPO_ROOT)
            cfg = json.loads((dest / "rsi.config.json").read_text(encoding="utf-8"))
            self.assertIn("test_*.py", cfg["inner_check_cmd"])

    def test_multi_trial_benchmark_stats_aggregation(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
        from run_bench import run_benchmark, summarize_series  # noqa: E402

        s = summarize_series([100.0, 110.0, 120.0])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], 110.0)
        self.assertAlmostEqual(s["std"], 10.0)
        self.assertAlmostEqual(s["median"], 110.0)
        self.assertGreater(s["iqr"], 0.0)
        self.assertGreater(s["ci95_half_width"], 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "traces"
            report = run_benchmark(
                REPO_ROOT,
                REPO_ROOT / "benchmarks" / "tasks.json",
                runner="mock",
                trace_export_dir=export_dir,
                n_trials=3,
            )
            self.assertEqual(report["n_trials"], 3)
            self.assertEqual(len(report["trials"]), 3)
            for metric in ("pass_rate", "total_tokens", "duration_sec"):
                self.assertIn(metric, report["stats"])
                for key in ("n", "mean", "std", "se", "median", "iqr", "ci95_half_width"):
                    self.assertIn(key, report["stats"][metric])
            self.assertEqual(len(list(export_dir.glob("*/summary.json"))), 3)
            self.assertEqual(len(list(export_dir.glob("*/trial_3.json"))), 3)

    def test_variance_aware_pareto_selection_gate(self) -> None:
        cfg = load_rsi_config(REPO_ROOT)

        # 1. Noisy 10% token drop with 12% std (N=3) at 100% pass rate -> rejected
        noisy_base = {
            "pass_rate": 1.0,
            "total_tokens": 10000,
            "duration_sec": 10.0,
            "n_trials": 3,
            "stats": {
                "pass_rate": {"mean": 1.0, "std": 0.0, "n": 3},
                "total_tokens": {"mean": 10000.0, "std": 1200.0, "n": 3},
                "duration_sec": {"mean": 10.0, "std": 0.2, "n": 3},
            },
        }
        noisy_cand = {
            "pass_rate": 1.0,
            "total_tokens": 9000,
            "duration_sec": 10.0,
            "n_trials": 3,
            "stats": {
                "pass_rate": {"mean": 1.0, "std": 0.0, "n": 3},
                "total_tokens": {"mean": 9000.0, "std": 1200.0, "n": 3},
                "duration_sec": {"mean": 10.0, "std": 0.2, "n": 3},
            },
        }
        ok_noisy, reasons_noisy = evaluate_selection_gate(noisy_base, noisy_cand, cfg)
        self.assertFalse(ok_noisy)
        self.assertTrue(any("noisy_variance_exceeds_gain" in r for r in reasons_noisy))

        # 2. Consistent 12% token drop with 1% std (N=3) at 100% pass rate -> accepted
        stable_base = {
            "pass_rate": 1.0,
            "total_tokens": 10000,
            "duration_sec": 10.0,
            "n_trials": 3,
            "stats": {
                "pass_rate": {"mean": 1.0, "std": 0.0, "n": 3},
                "total_tokens": {"mean": 10000.0, "std": 100.0, "n": 3},
                "duration_sec": {"mean": 10.0, "std": 0.1, "n": 3},
            },
        }
        stable_cand = {
            "pass_rate": 1.0,
            "total_tokens": 8800,
            "duration_sec": 9.8,
            "n_trials": 3,
            "stats": {
                "pass_rate": {"mean": 1.0, "std": 0.0, "n": 3},
                "total_tokens": {"mean": 8800.0, "std": 100.0, "n": 3},
                "duration_sec": {"mean": 9.8, "std": 0.1, "n": 3},
            },
        }
        ok_stable, reasons_stable = evaluate_selection_gate(stable_base, stable_cand, cfg)
        self.assertTrue(ok_stable, msg=str(reasons_stable))
        self.assertTrue(any("token_efficiency_improved" in r for r in reasons_stable))

        # 3. Consistent 20% latency drop with flat tokens at 100% pass rate -> accepted
        lat_cand = {
            "pass_rate": 1.0,
            "total_tokens": 10000,
            "duration_sec": 8.0,
            "n_trials": 3,
            "stats": {
                "pass_rate": {"mean": 1.0, "std": 0.0, "n": 3},
                "total_tokens": {"mean": 10000.0, "std": 50.0, "n": 3},
                "duration_sec": {"mean": 8.0, "std": 0.2, "n": 3},
            },
        }
        ok_lat, reasons_lat = evaluate_selection_gate(stable_base, lat_cand, cfg)
        self.assertTrue(ok_lat, msg=str(reasons_lat))
        self.assertTrue(any("latency_efficiency_improved" in r for r in reasons_lat))

        # 4. Pass-rate regression -> rejected
        reg_cand = {**stable_cand, "pass_rate": 0.66, "stats": {**stable_cand["stats"], "pass_rate": {"mean": 0.66, "std": 0.0, "n": 3}}}
        ok_reg, reasons_reg = evaluate_selection_gate(stable_base, reg_cand, cfg)
        self.assertFalse(ok_reg)
        self.assertTrue(any("pass_rate_regressed" in r for r in reasons_reg))

        # 5. Noisy pass-rate bump below noise_z * se_rate -> rejected
        noisy_rate_base = {
            "pass_rate": 0.70,
            "total_tokens": 5000,
            "n_trials": 3,
            "stats": {"pass_rate": {"mean": 0.70, "std": 0.15, "n": 3}, "total_tokens": {"mean": 5000, "std": 50, "n": 3}},
        }
        noisy_rate_cand = {
            "pass_rate": 0.78,
            "total_tokens": 5000,
            "n_trials": 3,
            "stats": {"pass_rate": {"mean": 0.78, "std": 0.15, "n": 3}, "total_tokens": {"mean": 5000, "std": 50, "n": 3}},
        }
        diag = compute_selection_diagnostics(noisy_rate_base, noisy_rate_cand, cfg)
        self.assertFalse(diag["accepted"])
        self.assertTrue(any("noisy_variance_exceeds_gain" in r for r in diag["reasons"]))


if __name__ == "__main__":
    unittest.main()
