"""Tests for Milestone 6 issues (#19-#24)."""

from __future__ import annotations

import json
import os
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from budget_and_pruning import analyze_pruning_candidates, apply_pruning_pass  # noqa: E402
from common import load_rsi_config  # noqa: E402
from evolve_outer import (  # noqa: E402
    _build_proposer_prompt,
    evaluate_selection_gate,
    run_evolution_cycle,
    validate_proposal_rrsi,
)
from guard_harness import evaluate_tool_guard  # noqa: E402
from pr_reporter import create_review_pr, detect_default_branch, generate_pr_report  # noqa: E402
from run_bench import run_benchmark  # noqa: E402
from surface import load_yaml_file, resolve_mutable_surface  # noqa: E402
from toolchain import detect_project_toolchain  # noqa: E402
from weakness_miner import WeaknessCluster, synthesize_regression_suite  # noqa: E402


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


class M6ProgressiveDisclosureTests(unittest.TestCase):
    def _init_oversized_repo(self, repo: Path, *, mode: str = "ratchet") -> str:
        repo.mkdir(parents=True, exist_ok=True)
        (repo / ".gitignore").write_text(".rsi/\n__pycache__/\n", encoding="utf-8")
        # > 24,000 bytes initially
        filler = "- Please make sure that you always verify empty input and zero denominator boundary cases before returning.\n"
        (repo / "AGENTS.md").write_text("# Repo Rules\n" + (filler * 260), encoding="utf-8")
        (repo / "harness.yaml").write_text(
            f"""
rrsi:
  n_trials: 1
  max_files: 3
  max_diff_lines: 300
  max_rules_bytes: 24000
  rules_budget_mode: "{mode}"
mutable_surface:
  - path: "AGENTS.md"
    role: "rules"
    max_bytes: 24000
  - path: "skills/*/SKILL.md"
    role: "skills"
""",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "RSI"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "rsi@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

    def test_proposer_prompt_mentions_progressive_disclosure_and_ratchet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            self._init_oversized_repo(repo, mode="progressive_disclosure")
            cfg = load_rsi_config(repo)
            prompt = _build_proposer_prompt(repo, {"summaries": [], "error_samples": []}, [], config=cfg)
            self.assertIn("Pointer Pattern", prompt)
            self.assertIn("input token tax", prompt)
            self.assertIn("RATCHET / PROGRESSIVE DISCLOSURE ACTIVE", prompt)
            self.assertIn("skills/<name>/SKILL.md", prompt)

    def test_ratchet_and_progressive_disclosure_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            base_sha = self._init_oversized_repo(repo, mode="progressive_disclosure")
            cfg = load_rsi_config(repo)
            agents_path = repo / "AGENTS.md"
            orig_text = agents_path.read_text(encoding="utf-8")

            # 1. Net increase rejected by ratchet
            agents_path.write_text(orig_text + "- Extra appended rule.\n", encoding="utf-8")
            ok_inc, reasons_inc, _ = validate_proposal_rrsi(repo, base_sha, cfg, [], [])
            self.assertFalse(ok_inc)
            self.assertTrue(any("rules_ratchet_net_increase" in r for r in reasons_inc), reasons_inc)

            # 2. Same rules bytes when oversized rejected by ratchet_no_reduction
            subprocess.run(["git", "reset", "--hard", base_sha], cwd=repo, check=True, capture_output=True)
            skill_file = repo / "skills" / "debug" / "SKILL.md"
            skill_file.parent.mkdir(parents=True)
            skill_file.write_text("# Debug\n- Steps.\n", encoding="utf-8")
            ok_same, reasons_same, _ = validate_proposal_rrsi(repo, base_sha, cfg, [], [])
            self.assertFalse(ok_same)
            self.assertTrue(any("rules_ratchet_no_reduction" in r for r in reasons_same), reasons_same)

            # 3. Incremental shrink + skill offload passes even if AGENTS.md is still > 24000 bytes or total surface grows
            subprocess.run(["git", "reset", "--hard", base_sha], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=repo, check=True, capture_output=True)
            lines = orig_text.splitlines()
            agents_path.write_text("\n".join(lines[:-10]) + "\n- See skills/boundary/SKILL.md.\n", encoding="utf-8")
            offload_skill = repo / "skills" / "boundary" / "SKILL.md"
            offload_skill.parent.mkdir(parents=True)
            offload_skill.write_text("# Boundary Skill\n" + ("- Check empty inputs.\n" * 60), encoding="utf-8")
            ok_pd, reasons_pd, meta_pd = validate_proposal_rrsi(repo, base_sha, cfg, [], [])
            self.assertTrue(ok_pd, reasons_pd)
            self.assertLess(meta_pd["rules_bytes_delta"], 0)
            self.assertGreater(meta_pd["skills_bytes_delta"], 0)

            # 4. Explicit max_skills_bytes still enforced
            capped_cfg = json.loads(json.dumps(cfg))
            capped_cfg["rrsi"]["max_skills_bytes"] = 50
            ok_cap, reasons_cap, _ = validate_proposal_rrsi(repo, base_sha, capped_cfg, [], [])
            self.assertFalse(ok_cap)
            self.assertTrue(any("skills_budget_exceeded" in r for r in reasons_cap), reasons_cap)

    def test_verbose_and_pointer_pruning_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "prune_verbose"
            repo.mkdir()
            long_rule = "- " + ("detailed verification instruction " * 10)
            (repo / "AGENTS.md").write_text(
                "# Rules\n"
                "- Please make sure that you always verify empty input before parsing.\n"
                "- 必ず忘れないように毎回編集後はテストを実行することを徹底して心がけること。\n"
                f"{long_rule}\n",
                encoding="utf-8",
            )
            (repo / "harness.yaml").write_text(
                """
mutable_surface:
  - path: "AGENTS.md"
    role: "rules"
""",
                encoding="utf-8",
            )
            cfg = load_rsi_config(repo)
            analysis = analyze_pruning_candidates(repo, cfg, [])
            self.assertEqual(len(analysis["verbose_rules"]), 2)
            self.assertEqual(len(analysis["pointer_candidates"]), 1)
            self.assertGreater(analysis["verbose_bytes_saved"], 0)

            before = (repo / "AGENTS.md").stat().st_size
            modified = apply_pruning_pass(repo, analysis, trim_verbose=True)
            self.assertEqual(modified, ["AGENTS.md"])
            after = (repo / "AGENTS.md").stat().st_size
            self.assertLess(after, before)
            new_text = (repo / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("- Always verify empty input before parsing.", new_text)
            self.assertIn("- 必ず編集後はテストを実行すること。", new_text)

    def test_end_to_end_compression_evolution_and_pr_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            self._init_oversized_repo(repo, mode="progressive_disclosure")
            manifest = root / "tasks.json"
            # Use tasks.json copy so mock runner evaluates rules text size & boundary keywords
            import shutil

            shutil.copy2(REPO_ROOT / "benchmarks" / "tasks.json", manifest)

            class _CompressingProposer:
                def propose(self, context: object) -> str:
                    wt = context.worktree_dir  # type: ignore[attr-defined]
                    (wt / "AGENTS.md").write_text(
                        "# Concise Rules\n"
                        "- Check empty and zero boundary cases safely.\n"
                        "- Verify split() and logs; re-run unittest keeping public signature and constants.\n"
                        "- Pointer: see skills/details/SKILL.md.\n",
                        encoding="utf-8",
                    )
                    sk = wt / "skills" / "details" / "SKILL.md"
                    sk.parent.mkdir(parents=True, exist_ok=True)
                    sk.write_text("# Details\n- Full checklist.\n", encoding="utf-8")
                    return "Compressed oversized AGENTS.md into Pointer Pattern + SKILL.md"

            res = run_evolution_cycle(
                repo,
                branch="rsi/evolved",
                runner="mock",
                manifest_path=manifest,
                proposer=_CompressingProposer(),
                pr_mode="dry-run",
            )
            self.assertTrue(res["accepted"], res)
            self.assertLess(res["rules_bytes_delta"], 0)
            self.assertLess(res["candidate"]["total_tokens"], res["baseline"]["total_tokens"])
            report_md = Path(res["pr_report_path"]).read_text(encoding="utf-8")
            self.assertIn("Progressive Disclosure Footprint", report_md)


class M6HarnessLockAndBehaviorTests(unittest.TestCase):
    def test_eval_harness_tampering_detected_and_rejected_by_selection_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "AGENTS.md").write_text("# Original Rules\n- Follow invariants.\n", encoding="utf-8")
            skill = repo / "skills" / "triage" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("# Triage Skill\n", encoding="utf-8")
            (repo / "harness.yaml").write_text(
                """
mutable_surface:
  - path: "AGENTS.md"
    role: "rules"
  - path: "skills/*/SKILL.md"
    role: "skills"
""",
                encoding="utf-8",
            )
            manifest = root / "bench.json"
            manifest.write_text(
                json.dumps(
                    {
                        "suite_name": "lock-suite",
                        "tasks": [
                            {
                                "id": "t_tamper_agents",
                                "split": "train",
                                "prompt": "Solve task",
                                "files": {
                                    "solution.py": "def ok(): return True\n",
                                    "test_solution.py": "import unittest\nfrom solution import ok\nclass T(unittest.TestCase):\n    def test_ok(self): self.assertTrue(ok())\n",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_agy = bin_dir / "agy"
            fake_agy.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "assert os.environ.get('RSI_EVAL_MODE') == '1'\n"
                "p = Path('AGENTS.md')\n"
                "os.chmod(p, 0o644)\n"
                "p.write_text('# Tampered during eval\\n', encoding='utf-8')\n"
                "print(json.dumps({'usage': {'total_tokens': 120}}))\n",
                encoding="utf-8",
            )
            fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)

            orig_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{orig_path}"
            try:
                report = run_benchmark(repo, manifest, runner="agy")
            finally:
                os.environ["PATH"] = orig_path

            self.assertFalse(report["anti_goodhart_passed"])
            self.assertEqual(report["passed_tasks"], 0)
            self.assertEqual(report["tampered_tasks"], ["t_tamper_agents"])
            self.assertTrue(report["tasks"][0]["tampered"])
            self.assertTrue(any("harness_modified:AGENTS.md" in v for v in report["tasks"][0]["tamper_violations"]))

            ok_gate, reasons = evaluate_selection_gate(
                {"pass_rate": 0.0, "total_tokens": 500},
                report,
                load_rsi_config(repo),
            )
            self.assertFalse(ok_gate)
            self.assertIn("anti_goodhart_verifier_tampered", reasons)

    def test_guard_harness_denies_harness_edits_under_rsi_eval_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")
            (repo / "harness.yaml").write_text(
                'mutable_surface:\n  - path: "CUSTOM_RULES.md"\n    role: "rules"\n',
                encoding="utf-8",
            )
            os.environ["RSI_EVAL_MODE"] = "1"
            os.environ.pop("RSI_EVOLUTION_MODE", None)
            try:
                for path in ("AGENTS.md", "skills/new/SKILL.md", "harness.yaml", "CUSTOM_RULES.md"):
                    res = evaluate_tool_guard(
                        {"cwd": str(repo), "toolCall": {"name": "write_file", "args": {"Path": path}}}
                    )
                    self.assertEqual(res["decision"], "deny", msg=f"Expected deny for {path}: {res}")

                shell_res = evaluate_tool_guard(
                    {
                        "cwd": str(repo),
                        "toolCall": {"name": "run_command", "args": {"CommandLine": "echo x >> AGENTS.md"}},
                    }
                )
                self.assertEqual(shell_res["decision"], "deny")

                src_res = evaluate_tool_guard(
                    {
                        "cwd": str(repo),
                        "toolCall": {"name": "edit_file", "args": {"Path": "src/app.py"}},
                    }
                )
                self.assertEqual(src_res["decision"], "allow")
            finally:
                os.environ.pop("RSI_EVAL_MODE", None)

    def test_behavior_assertions_fail_on_trace_violations_and_synthesized_in_miner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")
            manifest = root / "behavior_tasks.json"
            manifest.write_text(
                json.dumps(
                    {
                        "suite_name": "behavior-suite",
                        "tasks": [
                            {
                                "id": "t_beh_fail",
                                "split": "train",
                                "prompt": "Solve cleanly",
                                "files": {
                                    "solution.py": "def val(): return 1\n",
                                    "test_solution.py": "import unittest\nfrom solution import val\nclass T(unittest.TestCase):\n    def test_v(self): self.assertEqual(val(), 1)\n",
                                },
                                "verifier": {
                                    "behavior_assertions": {
                                        "max_consecutive_errors": 1,
                                        "max_tool_calls": 5,
                                        "forbidden_commands": ["rm -rf"],
                                        "forbidden_read_paths": ["secret.key"],
                                    }
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_agy = bin_dir / "agy"
            fake_agy.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "from pathlib import Path\n"
                "tr = Path('.rsi/traces/conv-b')\n"
                "tr.mkdir(parents=True, exist_ok=True)\n"
                "(tr / 'summary.json').write_text(json.dumps({'conversationId': 'conv-b', 'total_tokens': 250}), encoding='utf-8')\n"
                "events = [\n"
                "    {'event': 'post_tool', 'toolName': 'read_file', 'toolArgs': {'Path': 'secret.key'}, 'error': 'err1'},\n"
                "    {'event': 'post_tool', 'toolName': 'run_command', 'toolArgs': {'CommandLine': 'rm -rf tmp'}, 'error': 'err2'},\n"
                "]\n"
                "(tr / 'events.jsonl').write_text('\\n'.join(json.dumps(e) for e in events) + '\\n', encoding='utf-8')\n"
                "print(json.dumps({'usage': {'total_tokens': 250}}))\n",
                encoding="utf-8",
            )
            fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)

            orig_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{orig_path}"
            try:
                rep = run_benchmark(repo, manifest, runner="agy")
            finally:
                os.environ["PATH"] = orig_path

            self.assertFalse(rep["behavior_passed"])
            self.assertEqual(rep["behavior_failed_tasks"], ["t_beh_fail"])
            self.assertEqual(rep["passed_tasks"], 0)
            task_res = rep["tasks"][0]
            self.assertFalse(task_res["behavior_passed"])
            self.assertTrue(any("max_consecutive_errors_exceeded" in v for v in task_res["behavior_violations"]))
            self.assertTrue(any("forbidden_command:rm -rf" in v for v in task_res["behavior_violations"]))
            self.assertTrue(any("forbidden_read_path:secret.key" in v for v in task_res["behavior_violations"]))

            # Weakness miner attaches default behavior assertions
            suite = synthesize_regression_suite(
                repo,
                [
                    WeaknessCluster(
                        cluster_id="c1",
                        category="tool_error_chain",
                        signature="edit_file:SyntaxError",
                        count=3,
                        severity_score=5.0,
                    )
                ],
                behavior_assertions={"max_tool_calls": 10},
            )
            assertions = suite["tasks"][0]["verifier"]["behavior_assertions"]
            self.assertEqual(assertions["max_consecutive_errors"], 2)
            self.assertEqual(assertions["max_tool_calls"], 10)


if __name__ == "__main__":
    unittest.main()
