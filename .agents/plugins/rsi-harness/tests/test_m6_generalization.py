"""Tests for Milestone 6 issues (#19-#24)."""

from __future__ import annotations

import json
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pr_reporter import create_review_pr, detect_default_branch, generate_pr_report  # noqa: E402
from surface import load_yaml_file  # noqa: E402
from toolchain import detect_project_toolchain  # noqa: E402


class M6InitAndToolchainTests(unittest.TestCase):
    def _run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "rsi_harness", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )

    def test_quest_qlub_vite_plus_skills_and_oversized_agents_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "quest-qlub"
            repo.mkdir()
            (repo / "package.json").write_text(
                json.dumps(
                    {
                        "name": "quest-qlub",
                        "scripts": {
                            "ready": "vp check && vp test && echo done",
                        },
                        "devDependencies": {"vite-plus": "^0.1.0"},
                    }
                ),
                encoding="utf-8",
            )
            (repo / "pnpm-lock.yaml").write_text("lockfileVersion: 9\n", encoding="utf-8")
            (repo / "tsconfig.json").write_text("{}\n", encoding="utf-8")
            (repo / ".agents" / "skills" / "deploy").mkdir(parents=True)
            (repo / ".agents" / "skills" / "deploy" / "SKILL.md").write_text("# Deploy\n", encoding="utf-8")
            (repo / "dist").mkdir()
            # Oversized AGENTS.md (> 24,000 bytes)
            (repo / "AGENTS.md").write_text("# Rules\n" + ("- Detail rule\n" * 2000), encoding="utf-8")
            # Plugin rules present inside .agents/plugins/rsi-harness/ must be excluded from mutable_surface
            plugin_rules = repo / ".agents" / "plugins" / "rsi-harness" / "rules"
            plugin_rules.mkdir(parents=True)
            (plugin_rules / "AGENTS.md").write_text("# Plugin internal rules\n", encoding="utf-8")

            tc = detect_project_toolchain(repo)
            self.assertEqual(tc.language, "typescript")
            self.assertEqual(tc.tool, "vite-plus")
            self.assertEqual(tc.check_cmd, "pnpm run ready")

            dry = json.loads(self._run_cli("init", "--repo", str(repo), "--dry-run", "--json").stdout)
            self.assertEqual(dry["toolchain"]["tool"], "vite-plus")
            mutable_paths = [e["path"] for e in dry["mutable_surface"]]
            self.assertIn("AGENTS.md", mutable_paths)
            self.assertIn(".agents/skills/*/SKILL.md", mutable_paths)
            self.assertFalse(any("plugins/rsi-harness" in p for p in mutable_paths))
            self.assertIn("dist/", dry["protected_surface"])
            self.assertIn(".agents/plugins/rsi-harness/rules/", dry["protected_surface"])
            self.assertGreater(dry["rules_bytes"], 24000)
            self.assertGreater(dry["recommended_max_rules_bytes"], 24000)
            self.assertTrue(any("24000" in w for w in dry["warnings"]))

            # Live init writes harness.yaml with adjusted budget and without overwriting AGENTS.md
            orig_agents = (repo / "AGENTS.md").read_text(encoding="utf-8")
            self._run_cli("init", "--repo", str(repo), "--json")
            self.assertEqual((repo / "AGENTS.md").read_text(encoding="utf-8"), orig_agents)
            parsed_yaml = load_yaml_file(repo / "harness.yaml")
            self.assertEqual(parsed_yaml["rrsi"]["max_rules_bytes"], dry["recommended_max_rules_bytes"])

    def test_marune_essential_claude_md_skills_and_typecheck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "marune-essential"
            repo.mkdir()
            (repo / "package.json").write_text(
                json.dumps(
                    {
                        "name": "marune-essential",
                        "scripts": {
                            "typecheck": "tsc --noEmit",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (repo / "CLAUDE.md").write_text("# Claude Rules\n- Be concise.\n", encoding="utf-8")
            (repo / ".agents" / "skills" / "linting").mkdir(parents=True)
            (repo / ".agents" / "skills" / "linting" / "SKILL.md").write_text("# Linting\n", encoding="utf-8")

            tc = detect_project_toolchain(repo)
            self.assertEqual(tc.language, "typescript")
            self.assertEqual(tc.check_cmd, "npm run typecheck")

            dry = json.loads(self._run_cli("init", "--repo", str(repo), "--dry-run", "--json").stdout)
            mutable_paths = [e["path"] for e in dry["mutable_surface"]]
            self.assertIn("CLAUDE.md", mutable_paths)
            self.assertNotIn("AGENTS.md", mutable_paths)
            self.assertIn(".agents/skills/*/SKILL.md", mutable_paths)
            self.assertEqual(dry["warnings"], [])

            # Dependency-based vite-plus fallback when no test/check/ready/typecheck scripts exist
            vp_repo = Path(tmp) / "vp-dep-only"
            vp_repo.mkdir()
            (vp_repo / "package.json").write_text(
                json.dumps({"name": "vp-dep", "devDependencies": {"vite-plus": "^0.2.0"}}),
                encoding="utf-8",
            )
            tc_vp = detect_project_toolchain(vp_repo)
            self.assertEqual(tc_vp.tool, "vite-plus")
            self.assertEqual(tc_vp.check_cmd, "npx vp check && npx vp test")

    def test_opengravity_master_branch_auto_detection_and_protected_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "opengravity"
            repo.mkdir()
            (repo / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")
            (repo / "vendor").mkdir()
            (repo / "build").mkdir()
            subprocess.run(["git", "init", "-b", "master"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "RSI"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "rsi@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

            dry = json.loads(self._run_cli("init", "--repo", str(repo), "--dry-run", "--json").stdout)
            self.assertIn("vendor/", dry["protected_surface"])
            self.assertIn("build/", dry["protected_surface"])

            self.assertEqual(detect_default_branch(repo), "master")
            record = {
                "run_id": "m6master1",
                "accepted": True,
                "stage": "selection_gate",
                "branch": "rsi/evolved",
                "node_branch": "rsi/node-m6master1",
                "baseline": {"pass_rate": 0.5, "total_tokens": 1000, "duration_sec": 0.5},
                "candidate": {"pass_rate": 1.0, "total_tokens": 800, "duration_sec": 0.4},
            }
            md = generate_pr_report(record, repo_root=repo)
            self.assertIn("`rsi/node-m6master1` -> `master`", md)
            dry_pr = create_review_pr(repo, record, dry_run=True)
            self.assertEqual(dry_pr["base_branch"], "master")


if __name__ == "__main__":
    unittest.main()
