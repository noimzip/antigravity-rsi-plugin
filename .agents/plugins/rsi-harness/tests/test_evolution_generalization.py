"""Tests for M4: declarative mutable surface, weakness mining, annealed budget, pruning, action fusion & compact."""

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
if str(REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from common import append_jsonl, load_rsi_config  # noqa: E402
from evolve_outer import run_evolution_cycle, validate_proposal_rrsi  # noqa: E402
from guard_harness import evaluate_tool_guard  # noqa: E402
from surface import parse_simple_yaml, resolve_mutable_surface  # noqa: E402
from weakness_miner import mine_weaknesses, synthesize_regression_suite  # noqa: E402


class MutableSurfaceTests(unittest.TestCase):
    def test_parse_simple_yaml_handles_nested_maps_and_dict_lists(self) -> None:
        raw = """
# sample harness.yaml
inner_check_cmd: "pytest"
rrsi:
  max_files: 3
  max_rules_bytes: 4000
mutable_surface:
  - path: "AGENTS.md"
    role: "rules"
    max_bytes: 4000
  - path: "skills/*/SKILL.md"
    role: "skills"
protected_surface:
  - "secrets/*.key"
  - "scripts/"
"""
        parsed = parse_simple_yaml(raw)
        self.assertEqual(parsed["inner_check_cmd"], "pytest")
        self.assertEqual(parsed["rrsi"]["max_files"], 3)
        self.assertEqual(parsed["mutable_surface"][0]["path"], "AGENTS.md")
        self.assertEqual(parsed["mutable_surface"][0]["max_bytes"], 4000)
        self.assertEqual(parsed["mutable_surface"][1]["role"], "skills")
        self.assertEqual(parsed["protected_surface"], ["secrets/*.key", "scripts/"])

    def test_standalone_repo_evolves_root_agents_md_without_plugin_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "standalone_repo"
            repo.mkdir()
            (repo / ".gitignore").write_text(".rsi/\n__pycache__/\n", encoding="utf-8")
            (repo / "AGENTS.md").write_text(
                "# Repo Rules\n- Check empty and zero boundary cases carefully.\n",
                encoding="utf-8",
            )
            (repo / "skills" / "triage").mkdir(parents=True)
            (repo / "skills" / "triage" / "SKILL.md").write_text("# Triage\n", encoding="utf-8")
            (repo / "harness.yaml").write_text(
                """
rrsi:
  n_trials: 1
  max_files: 2
  max_rules_bytes: 4000
mutable_surface:
  - path: "AGENTS.md"
    role: "rules"
    max_bytes: 3500
  - path: "skills/*/SKILL.md"
    role: "skills"
protected_surface:
  - "secrets/*.key"
""",
                encoding="utf-8",
            )
            manifest = root / "ext_tasks.json"
            shutil.copy2(REPO_ROOT / "benchmarks" / "tasks.json", manifest)

            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "RSI"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "rsi@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
            main_before = subprocess.check_output(["git", "rev-parse", "main"], cwd=repo, text=True).strip()

            # Protected glob in harness.yaml is enforced by guard_harness
            os.environ["RSI_EVOLUTION_MODE"] = "1"
            os.environ["RSI_ALLOWED_ROOT"] = str(repo)
            try:
                guard_res = evaluate_tool_guard(
                    {"cwd": str(repo), "toolCall": {"name": "write_file", "args": {"Path": "secrets/prod.key"}}}
                )
                self.assertEqual(guard_res["decision"], "deny")
            finally:
                os.environ.pop("RSI_EVOLUTION_MODE", None)
                os.environ.pop("RSI_ALLOWED_ROOT", None)

            # Run outer evolution cycle on standalone repo (no .agents/plugins/rsi-harness/)
            res = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock", manifest_path=manifest)
            self.assertTrue(res["accepted"], msg=str(res))
            self.assertEqual(res["changed_files"], ["AGENTS.md"])
            self.assertGreater(res["candidate"]["pass_rate"], res["baseline"]["pass_rate"])
            self.assertEqual(
                subprocess.check_output(["git", "rev-parse", "main"], cwd=repo, text=True).strip(),
                main_before,
            )

            # Per-entry max_bytes and out-of-scope file validation
            cfg = load_rsi_config(repo)
            surface = resolve_mutable_surface(repo, cfg)
            self.assertIn("AGENTS.md", [p.name for p in surface.list_existing_files(repo, role="rules")])

            (repo / "README.md").write_text("out of scope\n", encoding="utf-8")
            ok_scope, reasons_scope, _ = validate_proposal_rrsi(repo, main_before, cfg, [], [])
            self.assertFalse(ok_scope)
            self.assertTrue(any("out_of_scope_file:README.md" in r for r in reasons_scope))
            (repo / "README.md").unlink()

            (repo / "AGENTS.md").write_text("x" * 3600, encoding="utf-8")
            ok_entry, reasons_entry, _ = validate_proposal_rrsi(repo, main_before, cfg, [], [])
            self.assertFalse(ok_entry)
            self.assertTrue(any("surface_entry_budget_exceeded:AGENTS.md" in r for r in reasons_entry))


class WeaknessMinerTests(unittest.TestCase):
    def test_weakness_mining_and_benchmark_free_evolution_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "no_bench_repo"
            (repo / "src").mkdir(parents=True)
            (repo / "tests").mkdir(parents=True)
            (repo / ".gitignore").write_text(".rsi/\n__pycache__/\n", encoding="utf-8")
            (repo / "src" / "__init__.py").write_text("", encoding="utf-8")
            (repo / "src" / "mathutil.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
            (repo / "tests" / "test_mathutil.py").write_text(
                "import unittest\nfrom src.mathutil import add\n\n"
                "class MathTests(unittest.TestCase):\n"
                "    def test_add(self): self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            (repo / "AGENTS.md").write_text("# Minimal Repo Rules\n- Keep functions typed.\n", encoding="utf-8")
            (repo / "harness.yaml").write_text(
                """
inner_check_cmd: "python3 -m unittest discover -s tests"
rrsi:
  n_trials: 1
  max_files: 2
  max_rules_bytes: 8000
mutable_surface:
  - path: "AGENTS.md"
    role: "rules"
""",
                encoding="utf-8",
            )

            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "RSI"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "rsi@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
            main_before = subprocess.check_output(["git", "rev-parse", "main"], cwd=repo, text=True).strip()

            # Seed .rsi/traces with realistic failures (retry exhaustion, error chain, high tokens, oversized artifacts)
            c1 = repo / ".rsi" / "traces" / "sess-01"
            c1.mkdir(parents=True)
            (c1 / "retry_count.txt").write_text("2", encoding="utf-8")
            (c1 / "summary.json").write_text(
                json.dumps({"conversationId": "sess-01", "total_tokens": 28000, "errors": 2, "lastUserInput": "Fix parser bug"}),
                encoding="utf-8",
            )
            append_jsonl(c1 / "events.jsonl", {"event": "post_tool", "toolName": "edit_file", "error": "SyntaxError at line 42", "artifactRef": "/tmp/a.json"})
            append_jsonl(c1 / "events.jsonl", {"event": "post_tool", "toolName": "edit_file", "error": "SyntaxError at line 99", "artifactRef": "/tmp/b.json"})
            append_jsonl(c1 / "events.jsonl", {"event": "stop", "terminationReason": "error"})

            c2 = repo / ".rsi" / "traces" / "sess-02"
            c2.mkdir(parents=True)
            (c2 / "summary.json").write_text(
                json.dumps({"conversationId": "sess-02", "total_tokens": 4000, "errors": 0, "lastUserInput": "Add helper"}),
                encoding="utf-8",
            )
            append_jsonl(c2 / "events.jsonl", {"event": "post_tool", "toolName": "edit_file", "error": "SyntaxError at line 17"})

            clusters = mine_weaknesses(repo)
            categories = {c.category for c in clusters}
            self.assertIn("retry_exhaustion", categories)
            self.assertIn("tool_error_chain", categories)
            self.assertIn("high_token_usage", categories)
            self.assertIn("oversized_observation", categories)
            self.assertIn("stop_failure", categories)

            # Error chain clustered across sessions despite different line numbers
            err_clusters = [c for c in clusters if c.category == "tool_error_chain"]
            self.assertEqual(err_clusters[0].count, 3)

            # Benchmark-free outer evolution cycle works automatically without benchmarks/tasks.json
            self.assertFalse((repo / "benchmarks" / "tasks.json").exists())
            res = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock")
            self.assertEqual(res["evolution_mode"], "trace-mined")
            self.assertTrue(res["accepted"], msg=str(res))
            self.assertLess(res["candidate"]["total_tokens"], res["baseline"]["total_tokens"])
            self.assertTrue((repo / ".rsi" / "mined_tasks.json").is_file())
            self.assertGreaterEqual(len(res["mined_clusters"]), 1)
            self.assertEqual(
                subprocess.check_output(["git", "rev-parse", "main"], cwd=repo, text=True).strip(),
                main_before,
            )


if __name__ == "__main__":
    unittest.main()
