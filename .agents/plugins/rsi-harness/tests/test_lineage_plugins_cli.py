"""Tests for M5: DGM lineage archive, pluggable interfaces, PR reporter & standalone CLI."""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
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

from evolve_outer import compute_selection_diagnostics, run_evolution_cycle  # noqa: E402
from interfaces import (  # noqa: E402
    Critic,
    Evaluator,
    MetaRewardEvaluator,
    ProposalContext,
    Proposer,
    load_plugin,
)
from lineage import LineageArchive, main as lineage_main  # noqa: E402
from pr_reporter import create_review_pr, generate_pr_report  # noqa: E402
from toolchain import detect_project_toolchain  # noqa: E402


def _init_git_repo(repo: Path, *, with_harness_yaml: bool = True) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / ".gitignore").write_text(".rsi/\n__pycache__/\n", encoding="utf-8")
    (repo / "AGENTS.md").write_text("# Repo Rules\n- Be careful.\n", encoding="utf-8")
    if with_harness_yaml:
        (repo / "harness.yaml").write_text(
            """
rrsi:
  n_trials: 1
  max_files: 3
  max_rules_bytes: 6000
mutable_surface:
  - path: "AGENTS.md"
    role: "rules"
    max_bytes: 5000
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


class LineageDagTests(unittest.TestCase):
    def test_multi_branch_dag_pareto_novelty_and_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            archive = LineageArchive.load(repo)
            self.assertIsNone(archive.sample_parent("dgm", seed=1))
            self.assertIn("empty", archive.render_summary().lower())

            archive.ensure_root(
                commit_sha="sha0",
                branch="main",
                pass_rate=0.5,
                total_tokens=2000,
                latency_ms=500.0,
            )
            # n1: better pass_rate, same tokens/latency -> dominates root
            n1 = archive.add_node(
                node_id="n1",
                parent_id="root",
                commit_sha="sha1",
                branch="rsi/node-1",
                pass_rate=0.8,
                total_tokens=1500,
                latency_ms=400.0,
                diff_hash="h1",
                changed_files=["rules/a.md"],
            )
            # n2: lower pass_rate than n1, but lower tokens -> non-dominated trade-off!
            n2 = archive.add_node(
                node_id="n2",
                parent_id="root",
                commit_sha="sha2",
                branch="rsi/node-2",
                pass_rate=0.7,
                total_tokens=900,
                latency_ms=300.0,
                diff_hash="h2",
                changed_files=["skills/b/SKILL.md"],
            )
            # n3: strictly dominated by n1 on every objective, and duplicate edit files/hash with n1
            n3 = archive.add_node(
                node_id="n3",
                parent_id="n1",
                commit_sha="sha3",
                branch="rsi/node-3",
                pass_rate=0.75,
                total_tokens=1600,
                latency_ms=450.0,
                diff_hash="h1",
                changed_files=["rules/a.md"],
            )

            frontier_ids = [n.node_id for n in archive.pareto_frontier()]
            self.assertEqual(frontier_ids, ["n1", "n2"])
            self.assertNotIn("root", frontier_ids)
            self.assertNotIn("n3", frontier_ids)

            archive.recompute_novelty()
            self.assertGreater(n2.novelty_score, n3.novelty_score)

            best = archive.sample_parent("best")
            latest = archive.sample_parent("latest")
            self.assertIsNotNone(best)
            self.assertEqual(best.node_id, "n1")
            self.assertIsNotNone(latest)
            self.assertEqual(latest.node_id, "n3")

            pick_a = archive.sample_parent("dgm", seed=42)
            pick_b = archive.sample_parent("dgm", seed=42)
            self.assertIsNotNone(pick_a)
            self.assertEqual(pick_a.node_id, pick_b.node_id)

            seen_nodes = {archive.sample_parent("dgm", seed=s).node_id for s in range(40)}
            self.assertGreaterEqual(len(seen_nodes), 2)

            with self.assertRaises(ValueError):
                archive.sample_parent("unknown")

            saved_path = archive.save()
            reloaded = LineageArchive.load(saved_path)
            self.assertEqual(len(reloaded.nodes), 4)
            self.assertEqual(reloaded.get_node("root").children_ids, ["n1", "n2"])

            summary = reloaded.render_summary()
            self.assertIn("Lineage DAG:", summary)
            self.assertIn("Pareto Frontier", summary)
            self.assertIn("n1", summary)
            self.assertIn("n2", summary)

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = lineage_main(["--repo", str(repo), "--pareto", "--json"])
            self.assertEqual(rc, 0)
            parsed = json.loads(buf.getvalue())
            self.assertEqual([x["node_id"] for x in parsed], ["n1", "n2"])

    def test_evolve_outer_branches_and_records_lineage_dag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _init_git_repo(repo)
            manifest = root / "tasks.json"
            shutil.copy2(REPO_ROOT / "benchmarks" / "tasks.json", manifest)

            res1 = run_evolution_cycle(
                repo,
                branch="rsi/evolved",
                runner="mock",
                manifest_path=manifest,
                parent_strategy="dgm",
                seed=7,
            )
            self.assertTrue(res1["accepted"], res1)
            self.assertEqual(res1["parent_node_id"], "root")
            self.assertTrue(res1["lineage_node_id"].startswith("node-"))
            self.assertEqual(res1["node_branch"], f"rsi/node-{res1['run_id']}")

            # Verify branch exists in git
            branch_probe = subprocess.run(
                ["git", "rev-parse", "--verify", f"refs/heads/{res1['node_branch']}"],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            self.assertEqual(branch_probe.returncode, 0)

            # Branch explicitly from root again to form a multi-child DAG
            res2 = run_evolution_cycle(
                repo,
                branch="rsi/evolved",
                runner="mock",
                manifest_path=manifest,
                parent_node_id="root",
            )
            self.assertTrue(res2["accepted"], res2)
            self.assertEqual(res2["parent_node_id"], "root")
            self.assertNotEqual(res1["lineage_node_id"], res2["lineage_node_id"])

            archive = LineageArchive.load(repo)
            self.assertIn("root", archive.nodes)
            self.assertEqual(len(archive.nodes["root"].children_ids), 2)
            self.assertIn(res1["lineage_node_id"], archive.nodes)
            self.assertIn(res2["lineage_node_id"], archive.nodes)

            with self.assertRaises(ValueError):
                run_evolution_cycle(
                    repo,
                    branch="rsi/evolved",
                    runner="mock",
                    manifest_path=manifest,
                    parent_node_id="does-not-exist",
                )


class PluggableInterfacesTests(unittest.TestCase):
    def test_external_plugins_loaded_from_rsi_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _init_git_repo(repo)
            manifest = root / "tasks.json"
            shutil.copy2(REPO_ROOT / "benchmarks" / "tasks.json", manifest)

            plugin_code = '''from pathlib import Path
from typing import Any

class CustomProposer:
    def propose(self, context: Any) -> str:
        target = context.worktree_dir / "AGENTS.md"
        old = target.read_text(encoding="utf-8")
        target.write_text(old.rstrip() + "\\n- Custom plugin rule applied.\\n", encoding="utf-8")
        return "custom_plugin_propose_note"

class CustomEvaluator:
    def __init__(self, bonus: float = 0.25) -> None:
        self.bonus = bonus

    def evaluate(
        self,
        worktree_dir: Path,
        *,
        runner: str,
        config: dict[str, Any],
        manifest_path: Path | None = None,
        split: str | None = None,
        trace_export_dir: Path | None = None,
        n_trials: int = 1,
        sandbox: Any = None,
    ) -> dict[str, Any]:
        rules = (worktree_dir / "AGENTS.md").read_text(encoding="utf-8")
        improved = "Custom plugin rule applied." in rules
        rate = min(1.0, 0.5 + (self.bonus if improved else 0.0))
        tokens = 1100 if improved else 2000
        return {
            "pass_rate": rate,
            "total_tokens": tokens,
            "duration_sec": 0.2 if improved else 0.4,
            "anti_goodhart_passed": True,
            "stats": {
                "pass_rate": {"n": 1, "mean": rate, "std": 0.0, "se": 0.0},
                "total_tokens": {"n": 1, "mean": float(tokens), "std": 0.0, "se": 0.0},
                "duration_sec": {"n": 1, "mean": 0.2 if improved else 0.4, "std": 0.0, "se": 0.0},
            },
        }

class CustomCritic:
    def check(
        self,
        worktree_dir: Path,
        *,
        base_sha: str,
        config: dict[str, Any],
        history: list[dict[str, Any]],
        benchmark_ids: list[str],
        leakage_profile: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str], dict[str, Any]]:
        return True, [], {"changed_files": ["AGENTS.md"], "diff_lines": 1, "custom_critic": True}
'''
            (repo / "custom_rsi_plugins.py").write_text(plugin_code, encoding="utf-8")
            (repo / "rsi.config.json").write_text(
                json.dumps(
                    {
                        "plugins": {
                            "proposer": "custom_rsi_plugins:CustomProposer",
                            "evaluator": {
                                "spec": "custom_rsi_plugins:CustomEvaluator",
                                "kwargs": {"bonus": 0.35},
                            },
                            "critic": "custom_rsi_plugins:CustomCritic",
                        }
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "add plugins"], cwd=repo, check=True, capture_output=True)

            res = run_evolution_cycle(repo, branch="rsi/evolved", runner="mock", manifest_path=manifest)
            self.assertTrue(res["accepted"], res)
            self.assertEqual(res["proposer_note"], "custom_plugin_propose_note")
            self.assertTrue(res["custom_critic"])
            self.assertIn("CustomProposer", res["plugins_used"]["proposer"])
            self.assertIn("CustomEvaluator", res["plugins_used"]["evaluator"])
            self.assertIn("CustomCritic", res["plugins_used"]["critic"])
            self.assertAlmostEqual(res["candidate"]["pass_rate"], 0.85, places=4)

    def test_meta_reward_evaluator_blending_cli_and_anti_goodhart(self) -> None:
        class StubBaseEvaluator:
            def __init__(self, anti_goodhart_passed: bool = True, pass_rate: float = 0.6) -> None:
                self._ag = anti_goodhart_passed
                self._rate = pass_rate

            def evaluate(self, worktree_dir: Path, **kwargs: object) -> dict[str, object]:
                return {
                    "pass_rate": self._rate,
                    "total_tokens": 1000,
                    "duration_sec": 0.25,
                    "anti_goodhart_passed": self._ag,
                    "stats": {
                        "pass_rate": {"n": 1, "mean": self._rate, "std": 0.0, "se": 0.0},
                    },
                }

        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            # 1. Callable judge blends score when anti_goodhart_passed=True
            meta_eval = MetaRewardEvaluator(
                base=StubBaseEvaluator(anti_goodhart_passed=True, pass_rate=0.6),
                judge=lambda _wt, _rep: {"score": 1.0, "notes": "clean"},
                judge_weight=0.25,
            )
            rep = meta_eval.evaluate(wt, runner="mock", config={})
            self.assertTrue(rep["anti_goodhart_passed"])
            self.assertAlmostEqual(rep["pass_rate"], 0.7, places=4)
            self.assertAlmostEqual(rep["stats"]["pass_rate"]["mean"], 0.7, places=4)
            self.assertAlmostEqual(rep["meta_reward"]["base_pass_rate"], 0.6, places=4)

            # 2. CLI judge script works
            judge_script = wt / "judge.py"
            judge_script.write_text('import json; print(json.dumps({"score": 0.8, "notes": "cli"}))\n')
            cli_eval = MetaRewardEvaluator(
                base=StubBaseEvaluator(anti_goodhart_passed=True, pass_rate=0.4),
                judge=f"{sys.executable} {judge_script}",
                judge_weight=0.5,
            )
            cli_rep = cli_eval.evaluate(wt, runner="mock", config={})
            self.assertAlmostEqual(cli_rep["pass_rate"], 0.6, places=4)
            self.assertEqual(cli_rep["meta_reward"]["notes"], "cli")

            # 3. Anti-Goodhart failure cannot be rescued by 1.0 judge score
            tampered_eval = MetaRewardEvaluator(
                base=StubBaseEvaluator(anti_goodhart_passed=False, pass_rate=0.0),
                judge=lambda _wt, _rep: {"score": 1.0, "notes": "looks great"},
                judge_weight=0.5,
            )
            tampered_rep = tampered_eval.evaluate(wt, runner="mock", config={})
            self.assertFalse(tampered_rep["anti_goodhart_passed"])
            self.assertEqual(tampered_rep["pass_rate"], 0.0)
            self.assertFalse(tampered_rep["meta_reward"]["anti_goodhart_preserved"])

            sel = compute_selection_diagnostics(
                {"pass_rate": 0.0, "total_tokens": 1000, "duration_sec": 0.25},
                tampered_rep,
                {"rrsi": {"enforce_anti_goodhart": True}},
            )
            self.assertFalse(sel["accepted"])
            self.assertIn("anti_goodhart_verifier_tampered", sel["reasons"])

    def test_load_plugin_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            load_plugin("", Proposer, lambda: None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            load_plugin({"kwargs": {}}, Evaluator, lambda: None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            load_plugin("missing_entry_point", Critic, lambda: None, entry_point_group="rsi_harness.critics")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            load_plugin(12345, Proposer, lambda: None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            load_plugin("builtins:dict", Proposer, lambda: None)  # type: ignore[arg-type]


class PrReporterAndCliTests(unittest.TestCase):
    def test_pr_reporter_safety_dry_run_and_mocked_gh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _init_git_repo(repo)
            record = {
                "run_id": "abcd1234",
                "accepted": True,
                "stage": "selection_gate",
                "branch": "rsi/evolved",
                "node_branch": "rsi/node-abcd1234",
                "parent_node_id": "root",
                "lineage_node_id": "node-abcd1234",
                "parent_strategy": "dgm",
                "commit_sha": "",
                "evolution_mode": "benchmark",
                "split_mode": "single",
                "n_trials": 3,
                "reasons": ["pass_rate_improved:+0.333"],
                "proposer_note": "Added generalized boundary check",
                "plugins_used": {"proposer": "DefaultProposer"},
                "pareto_frontier_ids": ["node-abcd1234"],
                "changed_files": ["AGENTS.md"],
                "diff_lines": 2,
                "rules_bytes": 120,
                "diff_hash": "0123456789abcdef",
                "baseline": {"pass_rate": 0.3333, "total_tokens": 5700, "duration_sec": 0.3, "anti_goodhart_passed": True},
                "candidate": {"pass_rate": 0.6667, "total_tokens": 4100, "duration_sec": 0.25, "anti_goodhart_passed": True},
            }

            report_md = generate_pr_report(record, repo_root=repo, diff_text="+ - Verify empty inputs\n")
            self.assertIn("RSI-Harness Evolution Report", report_md)
            self.assertIn("0.6667", report_md)
            self.assertIn("+ - Verify empty inputs", report_md)

            # Safety refusals
            with self.assertRaises(ValueError):
                create_review_pr(repo, record, base_branch="main", head_branch="main", dry_run=True)
            with self.assertRaises(ValueError):
                create_review_pr(repo, record, base_branch="develop", head_branch="master", dry_run=True)
            with self.assertRaises(ValueError):
                create_review_pr(repo, record, base_branch="feature", head_branch="feature", dry_run=True)
            with self.assertRaises(ValueError):
                create_review_pr(repo, {**record, "accepted": False}, base_branch="main", dry_run=False, push=False)

            # Dry-run writes report file cleanly without calling gh
            dry_res = create_review_pr(
                repo,
                record,
                base_branch="main",
                dry_run=True,
                diff_text="+ - Boundary check\n",
            )
            self.assertTrue(dry_res["dry_run"])
            self.assertFalse(dry_res["created"])
            report_path = Path(dry_res["report_path"])
            self.assertTrue(report_path.is_file())
            self.assertIn("AGENTS.md", report_path.read_text(encoding="utf-8"))

            # Mocked gh executable on PATH
            bin_dir = root / "bin"
            bin_dir.mkdir()
            gh_log = root / "gh_args.json"
            gh_script = bin_dir / "gh"
            gh_script.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                f"open({str(gh_log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                "print('https://github.com/example/repo/pull/99')\n",
                encoding="utf-8",
            )
            gh_script.chmod(gh_script.stat().st_mode | stat.S_IEXEC)

            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
            try:
                live_res = create_review_pr(repo, record, base_branch="main", dry_run=False, push=False)
            finally:
                os.environ["PATH"] = old_path

            self.assertTrue(live_res["created"])
            self.assertEqual(live_res["pr_url"], "https://github.com/example/repo/pull/99")
            captured = json.loads(gh_log.read_text(encoding="utf-8"))
            self.assertEqual(captured[:4], ["pr", "create", "--base", "main"])
            self.assertIn("rsi/node-abcd1234", captured)
            self.assertIn(str(report_path), captured)

    def test_pyproject_pure_stdlib_and_repo_toolchain_still_unittest(self) -> None:
        pyproject_path = REPO_ROOT / "pyproject.toml"
        self.assertTrue(pyproject_path.is_file())
        parsed = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["project"]["name"], "rsi-harness")
        self.assertEqual(parsed["project"]["dependencies"], [])
        self.assertEqual(parsed["project"]["scripts"]["rsi-harness"], "rsi_harness.cli:main")
        self.assertNotIn("pytest", pyproject_path.read_text(encoding="utf-8").lower())

        tc = detect_project_toolchain(REPO_ROOT)
        self.assertEqual(tc.language, "python")
        self.assertEqual(tc.tool, "unittest")

    def test_cli_init_polyglot_non_clobber_and_commands(self) -> None:
        def _run_cli(*args: str) -> dict[str, object] | subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, "-m", "rsi_harness", *args],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                ("py_repo", {"pyproject.toml": "[project]\nname='x'\n"}, "python", "unittest"),
                ("node_repo", {"package.json": json.dumps({"scripts": {"test": "node --test"}})}, "javascript", "npm"),
                ("rust_repo", {"Cargo.toml": "[package]\nname='x'\nversion='0.1.0'\n"}, "rust", "cargo"),
                ("go_repo", {"go.mod": "module example.com/x\n"}, "go", "go"),
            ]
            for name, files, exp_lang, exp_tool in cases:
                rdir = root / name
                rdir.mkdir()
                for fname, content in files.items():
                    (rdir / fname).write_text(content, encoding="utf-8")
                if name == "py_repo":
                    (rdir / "tests").mkdir()

                proc = _run_cli("init", "--repo", str(rdir), "--dry-run", "--json")
                data = json.loads(proc.stdout)
                self.assertTrue(data["dry_run"])
                self.assertEqual(data["toolchain"]["language"], exp_lang)
                self.assertEqual(data["toolchain"]["tool"], exp_tool)
                self.assertFalse((rdir / "harness.yaml").exists())
                self.assertFalse((rdir / "rsi.config.json").exists())

            # Live init + non-clobbering re-init
            work = root / "py_repo"
            (work / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
            proc_live = _run_cli("init", "--repo", str(work), "--runtime", "claude", "--json")
            live_data = json.loads(proc_live.stdout)
            self.assertEqual(live_data["runtime"], "claude")
            self.assertTrue((work / "harness.yaml").is_file())
            self.assertTrue((work / "rsi.config.json").is_file())
            self.assertTrue((work / "AGENTS.md").is_file())
            gi_text = (work / ".gitignore").read_text(encoding="utf-8")
            self.assertIn("*.pyc", gi_text)
            self.assertIn(".rsi/", gi_text)

            # User edits AGENTS.md & .gitignore; second init without --force preserves them
            (work / "AGENTS.md").write_text("# Custom user rules\n", encoding="utf-8")
            proc_again = _run_cli("init", "--repo", str(work), "--json")
            again_data = json.loads(proc_again.stdout)
            self.assertTrue(all(a["action"] == "skip" for a in again_data["actions"]))
            self.assertEqual((work / "AGENTS.md").read_text(encoding="utf-8"), "# Custom user rules\n")

            # Seed git + run status, mine, evolve --pr-dry-run, and lineage
            subprocess.run(["git", "init", "-b", "main"], cwd=work, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "RSI"], cwd=work, check=True)
            subprocess.run(["git", "config", "user.email", "rsi@example.com"], cwd=work, check=True)
            subprocess.run(["git", "add", "-A"], cwd=work, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=work, check=True, capture_output=True)

            st_pre = json.loads(_run_cli("status", "--repo", str(work), "--json").stdout)
            self.assertEqual(st_pre["evolution"]["total_cycles"], 0)
            self.assertEqual(st_pre["runtime"], "claude")

            manifest = root / "tasks.json"
            shutil.copy2(REPO_ROOT / "benchmarks" / "tasks.json", manifest)
            ev_proc = _run_cli(
                "evolve",
                "--repo",
                str(work),
                "--runner",
                "mock",
                "--manifest",
                str(manifest),
                "--pr-dry-run",
            )
            ev_data = json.loads(ev_proc.stdout)
            self.assertTrue(ev_data["accepted"], ev_data)
            self.assertTrue(Path(ev_data["pr_report_path"]).is_file())

            st_post = json.loads(_run_cli("status", "--repo", str(work), "--json").stdout)
            self.assertEqual(st_post["evolution"]["accepted_cycles"], 1)
            self.assertEqual(len(st_post["pr_reports"]), 1)

            lin_data = json.loads(_run_cli("lineage", "--repo", str(work), "--pareto", "--json").stdout)
            self.assertGreaterEqual(len(lin_data), 1)

            mine_data = json.loads(_run_cli("mine", "--repo", str(work), "--json").stdout)
            self.assertIn("cluster_count", mine_data)

    def test_copied_plugin_without_local_benchmarks_and_readonly_vendor_evolves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "target_app"
            _init_git_repo(repo)

            # Copy plugin bundle into target repo WITHOUT copying benchmarks/
            dst_plugin = repo / ".agents" / "plugins" / "rsi-harness"
            dst_plugin.mkdir(parents=True, exist_ok=True)
            for fname in ("hooks.json", "plugin.json"):
                shutil.copy2(PLUGIN_DIR / fname, dst_plugin / fname)
            shutil.copytree(
                PLUGIN_DIR / "scripts",
                dst_plugin / "scripts",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

            # Simulate heavy node_modules & 0555 read-only vendor dir
            nm = repo / "node_modules" / "pkg"
            nm.mkdir(parents=True)
            (nm / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")
            vendor = repo / "vendor"
            vendor.mkdir()
            vbin = vendor / "tool"
            vbin.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
            vbin.chmod(0o555)
            vendor.chmod(0o555)

            # Seed trace so trace-mined mode synthesizes source_dir task referencing target_app
            tr = repo / ".rsi" / "traces" / "s1"
            tr.mkdir(parents=True)
            (tr / "summary.json").write_text(
                json.dumps({"conversationId": "s1", "total_tokens": 25000, "errors": 1}),
                encoding="utf-8",
            )
            (tr / "events.jsonl").write_text(
                json.dumps({"toolName": "run_command", "error": "AssertionError: failed"}) + "\n",
                encoding="utf-8",
            )

            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "rsi_harness",
                        "evolve",
                        "--repo",
                        str(repo),
                        "--mode",
                        "trace-mined",
                        "--runner",
                        "mock",
                        "--pr-dry-run",
                        "--pr-base",
                        "main",
                    ],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                data = json.loads(proc.stdout)
                self.assertEqual(data["evolution_mode"], "trace-mined")
                self.assertIn("baseline", data)
                self.assertTrue(Path(data["pr_report_path"]).is_file())
            finally:
                vendor.chmod(0o755)
                vbin.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
