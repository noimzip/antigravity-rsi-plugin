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

from common import load_rsi_config  # noqa: E402
from evolve_outer import run_evolution_cycle, validate_proposal_rrsi  # noqa: E402
from guard_harness import evaluate_tool_guard  # noqa: E402
from surface import parse_simple_yaml, resolve_mutable_surface  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
