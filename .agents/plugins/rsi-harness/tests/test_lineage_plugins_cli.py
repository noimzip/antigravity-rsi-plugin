"""Tests for M5: DGM lineage archive, pluggable interfaces, PR reporter & standalone CLI."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
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

from evolve_outer import run_evolution_cycle  # noqa: E402
from lineage import LineageArchive, main as lineage_main  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
