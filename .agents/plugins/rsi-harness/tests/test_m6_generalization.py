"""Tests for Milestone 6 issues (#19-#24)."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.runtime import AgyAdapter  # noqa: E402
from budget_and_pruning import analyze_pruning_candidates, apply_pruning_pass  # noqa: E402
from common import compute_effective_token_cost, load_jsonl, load_rsi_config  # noqa: E402
from evolve_outer import (  # noqa: E402
    _build_proposer_prompt,
    build_leakage_fingerprint,
    compute_selection_diagnostics,
    evaluate_selection_gate,
    merge_benchmark_reports,
    run_evolution_cycle,
    should_expand_adaptive_trials,
    validate_proposal_rrsi,
)
from git_bench_miner import is_impl_file, is_test_file, mine_git_regression_suite  # noqa: E402
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


class M6AdaptiveTrialsAndCostTests(unittest.TestCase):
    def _init_repo(self, repo: Path, adaptive_cfg: dict[str, object]) -> None:
        repo.mkdir(parents=True, exist_ok=True)
        (repo / ".gitignore").write_text(".rsi/\n__pycache__/\n", encoding="utf-8")
        (repo / "AGENTS.md").write_text("# Rules\n- Be careful.\n", encoding="utf-8")
        (repo / "rsi.config.json").write_text(
            json.dumps(
                {
                    "rrsi": {
                        "n_trials": 3,
                        "max_files": 2,
                        "min_token_reduction_ratio": 0.1,
                        "use_effective_token_cost": True,
                        "cost_weights": {
                            "input_token_weight": 1.0,
                            "cache_read_token_weight": 0.1,
                            "output_token_weight": 4.0,
                            "tool_call_weight": 50.0,
                        },
                        "adaptive_trials": adaptive_cfg,
                    },
                    "mutable_surface": [{"path": "AGENTS.md", "role": "rules"}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "RSI"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "rsi@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    def test_runtime_result_enriched_from_traces_and_effective_cost_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = root / "wt"
            cwd.mkdir()
            tr = cwd / ".rsi" / "traces" / "c1"
            tr.mkdir(parents=True)
            (tr / "summary.json").write_text(
                json.dumps(
                    {
                        "conversationId": "c1",
                        "input_tokens": 1000,
                        "cache_read_tokens": 2000,
                        "output_tokens": 300,
                        "total_tokens": 3300,
                    }
                ),
                encoding="utf-8",
            )
            (tr / "events.jsonl").write_text(
                json.dumps({"event": "post_tool", "toolName": "read_file"}) + "\n"
                + json.dumps({"event": "post_tool", "toolName": "edit_file"}) + "\n"
                + json.dumps({"event": "stop"}) + "\n",
                encoding="utf-8",
            )

            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_agy = bin_dir / "agy"
            fake_agy.write_text(
                "#!/usr/bin/env python3\nimport json\nprint(json.dumps({'usage': {'total_tokens': 3300}}))\n",
                encoding="utf-8",
            )
            fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)
            orig_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{orig_path}"
            try:
                res = AgyAdapter().run("hi", cwd, timeout_sec=10)
            finally:
                os.environ["PATH"] = orig_path

            self.assertEqual(res.input_tokens, 1000)
            self.assertEqual(res.cache_read_tokens, 2000)
            self.assertEqual(res.output_tokens, 300)
            self.assertEqual(res.tool_calls, 2)
            self.assertEqual(res.steps, 3)

            cfg = {
                "rrsi": {
                    "use_effective_token_cost": True,
                    "min_token_reduction_ratio": 0.1,
                    "cost_weights": {
                        "input_token_weight": 1.0,
                        "cache_read_token_weight": 0.1,
                        "output_token_weight": 4.0,
                        "tool_call_weight": 50.0,
                    },
                }
            }
            # Candidate has nearly identical raw total_tokens (4000 vs 4100 = -2.4%),
            # but far lower output_tokens (100 vs 600) & tool_calls (2 vs 8) + higher cache_read.
            base = {
                "pass_rate": 1.0,
                "total_tokens": 4100,
                "input_tokens": 2500,
                "cache_read_tokens": 1000,
                "output_tokens": 600,
                "tool_calls": 8,
                "steps": 10,
                "duration_sec": 1.0,
            }
            cand = {
                "pass_rate": 1.0,
                "total_tokens": 4000,
                "input_tokens": 900,
                "cache_read_tokens": 3000,
                "output_tokens": 100,
                "tool_calls": 2,
                "steps": 4,
                "duration_sec": 0.9,
            }
            self.assertGreater(compute_effective_token_cost(base, cfg), compute_effective_token_cost(cand, cfg))
            diag = compute_selection_diagnostics(base, cand, cfg)
            self.assertTrue(diag["accepted"], diag)
            self.assertEqual(diag["token_breakdown"]["delta"]["tool_calls"], -6)
            self.assertEqual(diag["token_breakdown"]["delta"]["output_tokens"], -500)

    def test_adaptive_trials_early_reject_expand_accept_and_noisy_reject(self) -> None:
        class _StubProposer:
            def propose(self, context: object) -> str:
                wt = context.worktree_dir  # type: ignore[attr-defined]
                p = wt / "AGENTS.md"
                p.write_text(p.read_text(encoding="utf-8") + "- Adapted rule.\n", encoding="utf-8")
                return "stub_proposal"

        class _SequenceEvaluator:
            def __init__(self, base_series: list[dict[str, float]], cand_series: list[dict[str, float]]) -> None:
                self.base_series = list(base_series)
                self.cand_series = list(cand_series)
                self.base_calls = 0
                self.cand_calls = 0

            def _make_trial(self, row: dict[str, float], idx: int) -> dict[str, object]:
                return {
                    "trial": idx,
                    "total_tasks": 2,
                    "passed_tasks": int(round(row["pass_rate"] * 2)),
                    "pass_rate": row["pass_rate"],
                    "total_tokens": int(row["total_tokens"]),
                    "effective_token_cost": float(row.get("effective_token_cost", row["total_tokens"])),
                    "input_tokens": int(row.get("input_tokens", row["total_tokens"] * 0.5)),
                    "cache_read_tokens": int(row.get("cache_read_tokens", row["total_tokens"] * 0.3)),
                    "output_tokens": int(row.get("output_tokens", row["total_tokens"] * 0.2)),
                    "tool_calls": int(row.get("tool_calls", 4)),
                    "steps": int(row.get("steps", 6)),
                    "duration_sec": float(row.get("duration_sec", 0.5)),
                    "anti_goodhart_passed": True,
                    "tampered_tasks": [],
                    "behavior_passed": True,
                    "behavior_failed_tasks": [],
                    "tasks": [],
                }

            def evaluate(self, worktree_dir: Path, **kwargs: object) -> dict[str, object]:
                n_trials = int(kwargs.get("n_trials", 1))
                is_cand = "Adapted rule." in (worktree_dir / "AGENTS.md").read_text(encoding="utf-8")
                trials = []
                for _ in range(n_trials):
                    if is_cand:
                        row = self.cand_series[self.cand_calls]
                        self.cand_calls += 1
                    else:
                        row = self.base_series[self.base_calls]
                        self.base_calls += 1
                    trials.append(self._make_trial(row, len(trials) + 1))
                return merge_benchmark_reports(
                    [
                        {
                            "suite": "stub",
                            "runner": "mock",
                            "split": "all",
                            "trials": trials,
                        }
                    ]
                )

        adaptive_cfg = {
            "enabled": True,
            "initial_trials": 1,
            "max_trials": 3,
            "early_reject_pass_rate_drop": 0.05,
            "early_reject_token_overhead_ratio": 0.15,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "tasks.json"
            shutil.copy2(REPO_ROOT / "benchmarks" / "tasks.json", manifest)

            # 1. Immediate reject on 1st trial (pass_rate regressed -> no extra trials run)
            repo_reject = root / "repo_reject"
            self._init_repo(repo_reject, adaptive_cfg)
            ev_reject = _SequenceEvaluator(
                [{"pass_rate": 1.0, "total_tokens": 2000}] * 3,
                [{"pass_rate": 0.5, "total_tokens": 1800}] * 3,
            )
            res_reject = run_evolution_cycle(
                repo_reject,
                manifest_path=manifest,
                proposer=_StubProposer(),
                evaluator=ev_reject,
            )
            self.assertFalse(res_reject["accepted"])
            self.assertFalse(res_reject["adaptive_trials"]["expanded"])
            self.assertEqual(ev_reject.base_calls, 1)
            self.assertEqual(ev_reject.cand_calls, 1)

            # 2. Borderline on trial 1 (-12% cost) -> expanded to 3 trials -> consistent -> accepted + PR table
            repo_acc = root / "repo_acc"
            self._init_repo(repo_acc, adaptive_cfg)
            ev_acc = _SequenceEvaluator(
                [
                    {"pass_rate": 1.0, "total_tokens": 2000, "effective_token_cost": 2000.0, "tool_calls": 6},
                    {"pass_rate": 1.0, "total_tokens": 2010, "effective_token_cost": 2010.0, "tool_calls": 6},
                    {"pass_rate": 1.0, "total_tokens": 1990, "effective_token_cost": 1990.0, "tool_calls": 6},
                ],
                [
                    {"pass_rate": 1.0, "total_tokens": 1700, "effective_token_cost": 1600.0, "tool_calls": 3},
                    {"pass_rate": 1.0, "total_tokens": 1710, "effective_token_cost": 1610.0, "tool_calls": 3},
                    {"pass_rate": 1.0, "total_tokens": 1690, "effective_token_cost": 1590.0, "tool_calls": 3},
                ],
            )
            res_acc = run_evolution_cycle(
                repo_acc,
                manifest_path=manifest,
                proposer=_StubProposer(),
                evaluator=ev_acc,
                pr_mode="dry-run",
            )
            self.assertTrue(res_acc["accepted"], res_acc)
            self.assertTrue(res_acc["adaptive_trials"]["expanded"])
            self.assertEqual(res_acc["n_trials"], 3)
            self.assertEqual(ev_acc.base_calls, 3)
            self.assertEqual(ev_acc.cand_calls, 3)
            report_md = Path(res_acc["pr_report_path"]).read_text(encoding="utf-8")
            self.assertIn("effective_token_cost", report_md)
            self.assertIn("cache_read_tokens", report_md)
            self.assertIn("tool_calls", report_md)
            hist_rec = load_jsonl(repo_acc / ".rsi" / "evolution_history.jsonl")[-1]
            self.assertIn("token_breakdown", hist_rec["selection_diagnostics"])

            # 3. Promising on trial 1 (-15% cost) -> expanded to 3 trials -> high variance -> noisy reject
            repo_noisy = root / "repo_noisy"
            self._init_repo(repo_noisy, adaptive_cfg)
            ev_noisy = _SequenceEvaluator(
                [
                    {"pass_rate": 1.0, "total_tokens": 2000, "effective_token_cost": 2000.0},
                    {"pass_rate": 1.0, "total_tokens": 2400, "effective_token_cost": 2400.0},
                    {"pass_rate": 1.0, "total_tokens": 1600, "effective_token_cost": 1600.0},
                ],
                [
                    {"pass_rate": 1.0, "total_tokens": 1700, "effective_token_cost": 1700.0},
                    {"pass_rate": 1.0, "total_tokens": 2200, "effective_token_cost": 2200.0},
                    {"pass_rate": 1.0, "total_tokens": 1400, "effective_token_cost": 1400.0},
                ],
            )
            res_noisy = run_evolution_cycle(
                repo_noisy,
                manifest_path=manifest,
                proposer=_StubProposer(),
                evaluator=ev_noisy,
            )
            self.assertFalse(res_noisy["accepted"])
            self.assertTrue(res_noisy["adaptive_trials"]["expanded"])
            self.assertTrue(any("noisy_variance_exceeds_gain" in r for r in res_noisy["reasons"]), res_noisy)

            # Direct helper sanity check
            ok_exp, _ = should_expand_adaptive_trials(
                {"pass_rate": 1.0, "total_tokens": 1000, "duration_sec": 1.0},
                {"pass_rate": 1.0, "total_tokens": 950, "duration_sec": 1.0},
                {"rrsi": {"adaptive_trials": adaptive_cfg}},
            )
            self.assertTrue(ok_exp)


class M6GitBenchMinerTests(unittest.TestCase):
    def test_polyglot_test_and_impl_classifiers(self) -> None:
        self.assertTrue(is_test_file("tests/test_app.py"))
        self.assertTrue(is_test_file("src/utils.test.ts"))
        self.assertTrue(is_test_file("pkg/calc_test.go"))
        self.assertTrue(is_impl_file("src/utils.ts"))
        self.assertFalse(is_impl_file("README.md"))
        self.assertFalse(is_impl_file("tests/test_app.py"))

    def test_mine_from_git_fail_to_pass_benchmark_and_anti_goodhart_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "app_repo"
            (repo / "src").mkdir(parents=True)
            (repo / "tests").mkdir(parents=True)
            (repo / ".gitignore").write_text(".rsi/\n__pycache__/\n", encoding="utf-8")
            (repo / "AGENTS.md").write_text("# App Rules\n- Handle edge inputs.\n", encoding="utf-8")
            (repo / "pyproject.toml").write_text('[project]\nname="app"\n', encoding="utf-8")
            (repo / "src" / "__init__.py").write_text("", encoding="utf-8")
            (repo / "src" / "calc.py").write_text(
                "def safe_div(a: float, b: float) -> float:\n    return a / b\n",
                encoding="utf-8",
            )
            (repo / "tests" / "test_calc.py").write_text(
                "import unittest\nfrom src.calc import safe_div\n\n"
                "class CalcTests(unittest.TestCase):\n"
                "    def test_basic(self):\n"
                "        self.assertEqual(safe_div(6, 2), 3.0)\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "RSI"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "rsi@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init buggy calc"], cwd=repo, check=True, capture_output=True)

            # Commit 2: unrelated doc commit (should be ignored)
            (repo / "README.md").write_text("# Calc App\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "docs: add readme"], cwd=repo, check=True, capture_output=True)
            pre_fix_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

            # Commit 3: actual Fail-to-Pass bugfix commit with regression test
            (repo / "src" / "calc.py").write_text(
                "def safe_div(a: float, b: float) -> float:\n"
                "    if b == 0:\n"
                "        return 0.0\n"
                "    return a / b\n",
                encoding="utf-8",
            )
            (repo / "tests" / "test_calc.py").write_text(
                "import unittest\nfrom src.calc import safe_div\n\n"
                "class CalcTests(unittest.TestCase):\n"
                "    def test_basic(self):\n"
                "        self.assertEqual(safe_div(6, 2), 3.0)\n\n"
                "    def test_zero_division_guard(self):\n"
                "        self.assertEqual(safe_div(10, 0), 0.0)\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "fix: return 0.0 on zero divisor"], cwd=repo, check=True, capture_output=True)
            fix_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

            # Commit 4: refactor+test commit that already passed at pre-commit (not Fail-to-Pass, must be filtered out!)
            (repo / "src" / "calc.py").write_text(
                "def safe_div(a: float, b: float) -> float:\n"
                "    return 0.0 if b == 0 else a / b\n",
                encoding="utf-8",
            )
            (repo / "tests" / "test_calc.py").write_text(
                "import unittest\nfrom src.calc import safe_div\n\n"
                "class CalcTests(unittest.TestCase):\n"
                "    def test_basic(self):\n"
                "        self.assertEqual(safe_div(6, 2), 3.0)\n\n"
                "    def test_zero_division_guard(self):\n"
                "        self.assertEqual(safe_div(10, 0), 0.0)\n\n"
                "    def test_negative(self):\n"
                "        self.assertEqual(safe_div(-6, 2), -3.0)\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "test: add negative test and ternary"], cwd=repo, check=True, capture_output=True)

            manifest_out = root / "git_holdout.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rsi_harness",
                    "mine",
                    "--repo",
                    str(repo),
                    "--from-git",
                    "--split",
                    "holdout",
                    "--output",
                    str(manifest_out),
                    "--json",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            mined = json.loads(proc.stdout)
            self.assertEqual(mined["verified_count"], 1, mined)
            self.assertEqual(len(mined["tasks"]), 1)
            task = mined["tasks"][0]
            self.assertEqual(task["split"], "holdout")
            self.assertEqual(task["environment"]["git_ref"], pre_fix_sha)
            self.assertEqual(task["git_bug_pair"]["fix_commit"], fix_sha)
            self.assertIn("test_zero_division_guard", task["verifier"]["test_patch"])

            # Without fix (mock) -> pre_commit + transplanted test FAILS
            fail_rep = run_benchmark(repo, manifest_out, runner="mock")
            self.assertEqual(fail_rep["passed_tasks"], 0)

            # With fake agy fixing src/calc.py (verifying hidden test & .git are NOT exposed during eval)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_agy = bin_dir / "agy"
            fake_agy.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "from pathlib import Path\n"
                "assert not Path('.git').exists(), '.git history must not leak into eval workspace'\n"
                "test_txt = Path('tests/test_calc.py').read_text(encoding='utf-8')\n"
                "assert 'test_zero_division_guard' not in test_txt, 'transplanted test leaked before verifier!'\n"
                "Path('src/calc.py').write_text('def safe_div(a, b):\\n    return 0.0 if b == 0 else a / b\\n', encoding='utf-8')\n"
                "print(json.dumps({'usage': {'total_tokens': 300}}))\n",
                encoding="utf-8",
            )
            fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)
            orig_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{orig_path}"
            try:
                pass_rep = run_benchmark(repo, manifest_out, runner="agy")
            finally:
                os.environ["PATH"] = orig_path
            self.assertEqual(pass_rep["passed_tasks"], 1)

            # Leakage Critic blocks referencing hidden holdout test identifier or assertion from test_patch
            fp = build_leakage_fingerprint(mined["tasks"], manifest_out)
            self.assertIn("test_zero_division_guard", fp["hidden_identifiers"])
            self.assertIn("test_calc.py", fp["verifier_filenames"])

            head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            cfg = load_rsi_config(repo)
            (repo / "AGENTS.md").write_text(
                "# App Rules\n- Check test_zero_division_guard case.\n",
                encoding="utf-8",
            )
            ok_leak, reasons_leak, _ = validate_proposal_rrsi(
                repo,
                head_sha,
                cfg,
                [],
                fp["task_ids"],
                leakage_profile=fp,
            )
            self.assertFalse(ok_leak)
            self.assertTrue(
                any("test_zero_division_guard" in r for r in reasons_leak),
                reasons_leak,
            )


if __name__ == "__main__":
    unittest.main()
