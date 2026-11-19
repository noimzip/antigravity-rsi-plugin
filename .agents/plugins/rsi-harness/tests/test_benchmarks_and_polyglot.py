"""Tests for M3: polyglot toolchains, Harbor contracts, SandboxAdapters, and Anti-Goodhart evaluation."""

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

from adapters.sandbox import (  # noqa: E402
    DockerSandbox,
    GitWorktreeSandbox,
    RemoteSandboxAdapter,
    SandboxConfig,
    SandboxExecResult,
    get_sandbox_adapter,
)
from common import load_rsi_config  # noqa: E402
from evidence_reducer import deterministic_error_slice, reduce_log  # noqa: E402
from evolve_outer import (  # noqa: E402
    build_leakage_fingerprint,
    evaluate_selection_gate,
    run_evolution_cycle,
    validate_proposal_rrsi,
)
from inner_gate import evaluate_stop  # noqa: E402
from run_bench import load_benchmark_suite, run_benchmark  # noqa: E402
from toolchain import detect_project_toolchain, resolve_inner_check  # noqa: E402


class ToolchainAndPolyglotTests(unittest.TestCase):
    def test_detects_python_node_rust_go_java_cpp_and_makefile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1. Python pyproject with pytest
            py_pytest = root / "py_pytest"
            py_pytest.mkdir()
            (py_pytest / "pyproject.toml").write_text(
                '[project]\nname="app"\ndependencies=["pytest>=8"]\n', encoding="utf-8"
            )
            info = detect_project_toolchain(py_pytest)
            self.assertEqual(info.language, "python")
            self.assertEqual(info.tool, "pytest")
            self.assertEqual(info.check_cmd, "pytest")
            self.assertEqual(info.default_timeout_sec, 30)

            # 2. Python pyproject without pytest + tests dir -> unittest
            py_unit = root / "py_unit"
            (py_unit / "tests").mkdir(parents=True)
            (py_unit / "pyproject.toml").write_text('[project]\nname="u"\n', encoding="utf-8")
            info_u = detect_project_toolchain(py_unit)
            self.assertEqual(info_u.tool, "unittest")
            self.assertEqual(info_u.check_cmd, "python3 -m unittest discover -s tests")

            # 3. Node pnpm + npm + tsconfig fallback
            node_pnpm = root / "node_pnpm"
            node_pnpm.mkdir()
            (node_pnpm / "package.json").write_text(
                json.dumps({"name": "web", "scripts": {"test": "vitest run"}}),
                encoding="utf-8",
            )
            (node_pnpm / "pnpm-lock.yaml").write_text("lockfileVersion: 9\n", encoding="utf-8")
            (node_pnpm / "tsconfig.json").write_text("{}\n", encoding="utf-8")
            info_np = detect_project_toolchain(node_pnpm)
            self.assertEqual(info_np.language, "typescript")
            self.assertEqual(info_np.tool, "pnpm")
            self.assertEqual(info_np.check_cmd, "pnpm test")

            node_yarn = root / "node_yarn"
            node_yarn.mkdir()
            (node_yarn / "package.json").write_text(
                json.dumps({"name": "y", "scripts": {"test": "jest"}}), encoding="utf-8"
            )
            (node_yarn / "yarn.lock").write_text("", encoding="utf-8")
            self.assertEqual(detect_project_toolchain(node_yarn).check_cmd, "yarn test")

            ts_only = root / "ts_only"
            ts_only.mkdir()
            (ts_only / "package.json").write_text(
                json.dumps({"name": "t", "scripts": {"test": 'echo "Error: no test specified" && exit 1'}}),
                encoding="utf-8",
            )
            (ts_only / "tsconfig.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(detect_project_toolchain(ts_only).check_cmd, "npx tsc --noEmit")

            # 4. Rust
            rust_dir = root / "rust_app"
            rust_dir.mkdir()
            (rust_dir / "Cargo.toml").write_text('[package]\nname="r"\n', encoding="utf-8")
            info_r = detect_project_toolchain(rust_dir)
            self.assertEqual(info_r.language, "rust")
            self.assertEqual(info_r.check_cmd, "cargo test")
            self.assertEqual(info_r.default_timeout_sec, 120)

            # 5. Go
            go_dir = root / "go_app"
            go_dir.mkdir()
            (go_dir / "go.mod").write_text("module example.com/app\n", encoding="utf-8")
            info_g = detect_project_toolchain(go_dir)
            self.assertEqual(info_g.language, "go")
            self.assertEqual(info_g.check_cmd, "go test ./...")
            self.assertEqual(info_g.default_timeout_sec, 60)

            # 6. Makefile fallback
            mk_dir = root / "mk_app"
            mk_dir.mkdir()
            (mk_dir / "Makefile").write_text("build:\n\techo hi\ntest:\n\t./run_tests\n", encoding="utf-8")
            self.assertEqual(detect_project_toolchain(mk_dir).check_cmd, "make test")

            # 7. Explicit config overrides auto detection
            (rust_dir / "rsi.config.json").write_text(
                json.dumps({"inner_check_cmd": "cargo test --lib", "inner_check_timeout_sec": 75}),
                encoding="utf-8",
            )
            cfg = load_rsi_config(rust_dir)
            cmd, timeout_sec, _ = resolve_inner_check(rust_dir, cfg)
            self.assertEqual(cmd, "cargo test --lib")
            self.assertEqual(timeout_sec, 75.0)

    def test_inner_gate_adaptive_timeout_and_graceful_timeout_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / "rsi.config.json").write_text(
                json.dumps(
                    {
                        "inner_check_cmd": "python3 -c \"import time; print('working'); time.sleep(2)\"",
                        "inner_check_timeout_sec": 0.2,
                        "max_inner_retries": 2,
                    }
                ),
                encoding="utf-8",
            )
            res = evaluate_stop(
                {"conversationId": "conv-to", "cwd": str(cwd), "continuationActive": False}
            )
            self.assertEqual(res["decision"], "continue")
            self.assertIn("timed out", res["reason"].lower())

    def test_polyglot_error_extraction_in_noisy_logs(self) -> None:
        noise = "\n".join(f"[build] compiling step {i:03d} " + "x" * 50 for i in range(120))

        cases = [
            ("ts", "src/app.ts(12,4): TS2322: Type 'string' is not assignable to type 'number'."),
            ("jest", "FAIL src/utils.test.ts\nExpected: 42\nReceived: 0"),
            ("rust", "error[E0308]: mismatched types\nthread 'tests::it_works' panicked at src/lib.rs:14:5"),
            ("go", "--- FAIL: TestSum (0.00s)\n    sum_test.go:42: expected 5, got 4\nFAIL\texample.com/pkg"),
            ("java", "Exception in thread \"main\" java.lang.NullPointerException: missing key"),
            ("cpp", "main.cpp:18:5: fatal error: missing.h: No such file or directory\nundefined reference to `foo()'"),
        ]
        for label, err_snippet in cases:
            raw = f"{noise}\n{err_snippet}\nfinished exit 1\n"
            excerpt = deterministic_error_slice(raw, max_chars=2500)
            first_line = err_snippet.splitlines()[0]
            self.assertIn(first_line, excerpt, msg=f"Missed {label} error line in slice")
            reduced = reduce_log(raw, max_chars=500)
            self.assertTrue(reduced["was_reduced"])
            self.assertIn(first_line, reduced["reduced_text"])


class HarborContractTests(unittest.TestCase):
    def test_sample_harbor_directory_and_legacy_manifest(self) -> None:
        sample_dir = REPO_ROOT / "benchmarks" / "harbor_examples" / "sample_task"
        suite = load_benchmark_suite(sample_dir)
        self.assertEqual(len(suite["tasks"]), 1)
        self.assertIn("clamped_add", suite["tasks"][0]["prompt"])

        # Mock runner fixes harbor_sample_calc using AGENTS.md boundary rule -> passes independent test.sh
        report = run_benchmark(REPO_ROOT, sample_dir, runner="mock")
        self.assertEqual(report["total_tasks"], 1)
        self.assertEqual(report["passed_tasks"], 1)
        self.assertEqual(report["tasks"][0]["id"], "harbor_sample_calc")

        # Legacy tasks.json remains 100% compatible
        legacy_report = run_benchmark(REPO_ROOT, REPO_ROOT / "benchmarks" / "tasks.json", runner="mock")
        self.assertEqual(legacy_report["total_tasks"], 3)
        self.assertEqual(legacy_report["passed_tasks"], 2)

    def test_harbor_git_repo_checkout_and_independent_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ext_repo = root / "external_repo"
            ext_repo.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=ext_repo, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "T"], cwd=ext_repo, check=True)
            subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=ext_repo, check=True)
            (ext_repo / "pkg.py").write_text("def mul(a, b):\n    return a + b  # buggy\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=ext_repo, check=True)
            subprocess.run(["git", "commit", "-m", "buggy v1"], cwd=ext_repo, check=True, capture_output=True)
            v1_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ext_repo, text=True).strip()

            # Advance main further so git_ref checkout is actually exercised
            (ext_repo / "README.md").write_text("later commit\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=ext_repo, check=True)
            subprocess.run(["git", "commit", "-m", "later"], cwd=ext_repo, check=True, capture_output=True)

            harbor_task = root / "harbor_suite" / "git_task"
            (harbor_task / "verifier").mkdir(parents=True)
            (harbor_task / "instruction.md").write_text("Fix mul(a, b) in pkg.py to multiply.\n", encoding="utf-8")
            verifier_sh = harbor_task / "verifier" / "test.sh"
            verifier_sh.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "test ! -d verifier\n"
                "python3 -c 'from pkg import mul; assert mul(3, 4) == 12'\n",
                encoding="utf-8",
            )
            verifier_sh.chmod(verifier_sh.stat().st_mode | stat.S_IEXEC)
            (harbor_task / "task.json").write_text(
                json.dumps(
                    {
                        "id": "harbor_git_mul",
                        "format": "harbor",
                        "instruction_file": "instruction.md",
                        "environment": {
                            "type": "git",
                            "git_repo": str(ext_repo),
                            "git_ref": v1_sha,
                        },
                        "verifier": {"script": "verifier/test.sh", "timeout_sec": 20},
                    }
                ),
                encoding="utf-8",
            )

            # Without fix (mock unknown) -> independent verifier fails
            fail_rep = run_benchmark(REPO_ROOT, harbor_task, runner="mock")
            self.assertEqual(fail_rep["passed_tasks"], 0)

            # With fake `agy` fixing pkg.py -> independent verifier passes
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_agy = bin_dir / "agy"
            fake_agy.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "from pathlib import Path\n"
                "Path('pkg.py').write_text('def mul(a, b):\\n    return a * b\\n', encoding='utf-8')\n"
                "print(json.dumps({'usage': {'total_tokens': 321}}))\n",
                encoding="utf-8",
            )
            fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)
            orig_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{orig_path}"
            try:
                pass_rep = run_benchmark(REPO_ROOT, harbor_task, runner="agy")
            finally:
                os.environ["PATH"] = orig_path
            self.assertEqual(pass_rep["passed_tasks"], 1)
            self.assertEqual(pass_rep["total_tokens"], 321)


class SandboxAdapterTests(unittest.TestCase):
    def _init_repo(self, dest: Path) -> str:
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()

    def test_git_worktree_sandbox_lifecycle_and_readonly_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            sha = self._init_repo(repo)
            src_file = root / "secret_ro.txt"
            src_file.write_text("immutable", encoding="utf-8")

            sb = get_sandbox_adapter({"sandbox": {"backend": "worktree"}})
            self.assertIsInstance(sb, GitWorktreeSandbox)
            wt = sb.prepare(repo, sha, "run01")
            self.assertTrue(wt.is_dir())

            synced = sb.sync_files(src_file, "synced/secret_ro.txt", read_only=True)
            self.assertTrue(synced.is_file())
            self.assertEqual(synced.stat().st_mode & 0o222, 0)

            res = sb.exec(["python3", "-c", "print('inside-wt')"])
            self.assertEqual(res.returncode, 0)
            self.assertIn("inside-wt", res.stdout)

            sb.cleanup()
            self.assertFalse(wt.exists())
            branches = subprocess.check_output(["git", "branch"], cwd=repo, text=True)
            self.assertNotIn("rsi/cand-run01", branches)

    def test_docker_sandbox_hardened_command_and_fake_docker_exec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            sha = self._init_repo(repo)

            fake_docker = root / "fake_docker"
            fake_docker.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "print(json.dumps({'argv': sys.argv[1:]}))\n",
                encoding="utf-8",
            )
            fake_docker.chmod(fake_docker.stat().st_mode | stat.S_IEXEC)

            ro_host = root / "readonly_verifier"
            ro_host.mkdir()
            sb = get_sandbox_adapter(
                {
                    "sandbox": {
                        "backend": "docker",
                        "docker_bin": str(fake_docker),
                        "docker_image": "python:3.12-slim",
                        "network_disabled": True,
                        "cpus": 2.0,
                        "memory_mb": 512,
                        "pids_limit": 64,
                        "read_only_root": True,
                    }
                }
            )
            self.assertIsInstance(sb, DockerSandbox)
            wt = sb.prepare(repo, sha, "dock01")
            try:
                cmd = sb.build_docker_command(
                    ["python3", "-m", "unittest"],
                    env={"RSI_TEST": "1"},
                    readonly_mounts=[(ro_host, "/verifier")],
                )
                self.assertIn("--network", cmd)
                self.assertEqual(cmd[cmd.index("--network") + 1], "none")
                self.assertEqual(cmd[cmd.index("--cpus") + 1], "2.0")
                self.assertEqual(cmd[cmd.index("--memory") + 1], "512m")
                self.assertEqual(cmd[cmd.index("--pids-limit") + 1], "64")
                self.assertIn("--cap-drop=ALL", cmd)
                self.assertIn("--security-opt=no-new-privileges", cmd)
                self.assertIn("--read-only", cmd)
                self.assertIn(f"{ro_host.resolve()}:/verifier:ro", cmd)

                res = sb.exec(["pytest", "-q"], readonly_mounts=[(ro_host, "/verifier")])
                self.assertEqual(res.returncode, 0)
                captured = json.loads(res.stdout)
                self.assertIn("python:3.12-slim", captured["argv"])
                self.assertEqual(captured["argv"][-2:], ["pytest", "-q"])
            finally:
                sb.cleanup()

    def test_remote_sandbox_hook_and_unknown_backend(self) -> None:
        remote = get_sandbox_adapter({"sandbox": {"backend": "remote"}})
        self.assertIsInstance(remote, RemoteSandboxAdapter)
        with self.assertRaises(NotImplementedError):
            remote.prepare(REPO_ROOT, "HEAD", "rem01")

        class _MockDriver:
            def prepare(self, repo, base_sha, run_id):
                return repo

            def sync_files(self, src, rel_dest, *, read_only=False):
                return src

            def exec(self, cmd, *, cwd=None, env=None, timeout=120, readonly_mounts=None):
                return SandboxExecResult(returncode=0, stdout="remote-ok")

            def cleanup(self):
                return None

        wired = RemoteSandboxAdapter(SandboxConfig(backend="remote"), driver=_MockDriver())
        wired.prepare(REPO_ROOT, "HEAD", "rem02")
        self.assertEqual(wired.exec("echo hi").stdout, "remote-ok")
        wired.cleanup()

        with self.assertRaises(ValueError):
            get_sandbox_adapter({"sandbox": {"backend": "invalid-backend"}})


class AntiGoodhartAndSplitTests(unittest.TestCase):
    def _init_repo(self, dest: Path) -> str:
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()

    def test_anti_goodhart_detects_test_tampering_even_after_chmod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_agy = bin_dir / "agy"
            fake_agy.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os\n"
                "from pathlib import Path\n"
                "t = Path('test_solution.py')\n"
                "os.chmod(t, 0o644)\n"
                "t.write_text('import unittest\\nclass Cheat(unittest.TestCase):\\n    def test_ok(self): pass\\n', encoding='utf-8')\n"
                "print(json.dumps({'usage': {'total_tokens': 100}}))\n",
                encoding="utf-8",
            )
            fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)

            orig_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{orig_path}"
            try:
                report = run_benchmark(
                    REPO_ROOT,
                    REPO_ROOT / "benchmarks" / "tasks.json",
                    runner="agy",
                )
            finally:
                os.environ["PATH"] = orig_path

            self.assertFalse(report["anti_goodhart_passed"])
            self.assertEqual(report["passed_tasks"], 0)
            self.assertEqual(len(report["tampered_tasks"]), 3)
            self.assertTrue(all(t["tampered"] for t in report["tasks"]))

            cfg = load_rsi_config(REPO_ROOT)
            ok_gate, gate_reasons = evaluate_selection_gate(
                {"pass_rate": 0.5, "total_tokens": 1000},
                report,
                cfg,
            )
            self.assertFalse(ok_gate)
            self.assertIn("anti_goodhart_verifier_tampered", gate_reasons)

    def test_train_holdout_ood_isolation_and_ood_regression_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            self._init_repo(repo)
            proposer_log = root / "proposer_seen.txt"

            manifest_path = repo / "benchmarks" / "split_tasks.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "suite_name": "split-suite",
                        "tasks": [
                            {
                                "id": "train_task_alpha",
                                "split": "train",
                                "prompt": "TRAIN PROMPT",
                                "files": {
                                    "solution.py": "def f(): return 0\n",
                                    "test_solution.py": "import unittest\nfrom solution import f\nclass T(unittest.TestCase):\n    def test_t(self): self.assertEqual(f(), 1)\n",
                                },
                            },
                            {
                                "id": "holdout_task_beta",
                                "split": "holdout",
                                "prompt": "HOLDOUT PROMPT",
                                "files": {
                                    "solution.py": "def g(): return 0\n",
                                    "test_solution.py": "import unittest\nfrom solution import g\nclass H(unittest.TestCase):\n    def test_h(self): self.assertEqual(g(), 2)\n",
                                },
                            },
                            {
                                "id": "ood_task_gamma",
                                "split": "ood",
                                "prompt": "OOD PROMPT",
                                "files": {
                                    "solution.py": "def h(): return 1\n",
                                    "test_solution.py": "import unittest\nfrom solution import h\nclass O(unittest.TestCase):\n    def test_o(self): self.assertEqual(h(), 1)\n",
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_agy = bin_dir / "agy"
            fake_agy.write_text(
                f"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

cwd = Path.cwd()
if os.environ.get("RSI_EVOLUTION_MODE") == "1":
    trace_dirs = sorted(p.name for p in (cwd / ".rsi" / "traces").iterdir()) if (cwd / ".rsi" / "traces").is_dir() else []
    Path({str(proposer_log)!r}).write_text(json.dumps({{"traces": trace_dirs, "prompt": sys.argv}}), encoding="utf-8")
    rules = cwd / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"
    rules.write_text(rules.read_text(encoding="utf-8") + "\\n- Generalize across inputs and keep OOD invariants intact.\\n", encoding="utf-8")
    print(json.dumps({{"ok": True}}))
    sys.exit(0)

rules_txt = (cwd / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md").read_text(encoding="utf-8")
evolved = "Generalize across inputs" in rules_txt
if (cwd / "solution.py").is_file():
    sol = (cwd / "solution.py").read_text(encoding="utf-8")
    if "def g()" in sol and evolved:
        (cwd / "solution.py").write_text("def g(): return 2\\n", encoding="utf-8")
    elif "def h()" in sol and evolved and os.environ.get("FORCE_OOD_REGRESS") == "1":
        (cwd / "solution.py").write_text("def h(): return 999\\n", encoding="utf-8")
    elif "def f()" in sol:
        (cwd / "solution.py").write_text("def f(): return 1\\n", encoding="utf-8")
print(json.dumps({{"usage": {{"total_tokens": 500}}}}))
""",
                encoding="utf-8",
            )
            fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IEXEC)

            orig_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{orig_path}"
            os.environ["FORCE_OOD_REGRESS"] = "1"
            try:
                res_reg = run_evolution_cycle(
                    repo,
                    branch="rsi/evolved",
                    runner="agy",
                    manifest_path=manifest_path,
                )
            finally:
                os.environ["PATH"] = orig_path
                os.environ.pop("FORCE_OOD_REGRESS", None)

            # Proposer never saw holdout/ood traces
            seen = json.loads(proposer_log.read_text(encoding="utf-8"))
            joined_seen = json.dumps(seen)
            self.assertIn("train_task_alpha", joined_seen)
            self.assertNotIn("holdout_task_beta", joined_seen)
            self.assertNotIn("ood_task_gamma", joined_seen)

            # Holdout improved (0 -> 1), but OOD regressed (1 -> 0) -> rejected!
            self.assertEqual(res_reg["split_mode"], "train_holdout_ood")
            self.assertFalse(res_reg["accepted"])
            self.assertTrue(any("ood_generalization_regressed" in r for r in res_reg["reasons"]))

    def test_generalized_leakage_critic_blocks_verifier_ids_and_assertions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            base_sha = self._init_repo(repo)
            cfg = load_rsi_config(repo)
            tasks = [
                {
                    "id": "holdout_alpha",
                    "split": "holdout",
                    "files": {
                        "verify_secret.py": "class SecretHoldoutSuite:\n    def test_secret_holdout_case(self):\n        self.assertEqual(compute_x(7), 49)\n"
                    },
                    "verifier": {"script": "verify_secret.sh", "readonly_paths": ["verify_secret.py"]},
                }
            ]
            fp = build_leakage_fingerprint(tasks)
            rules = repo / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"
            orig_rules = rules.read_text(encoding="utf-8")

            cases = [
                ("verify_secret.sh", "critic_benchmark_leakage:verify_secret.sh"),
                ("test_secret_holdout_case", "critic_benchmark_leakage:hidden_id:test_secret_holdout_case"),
                ("self.assertEqual(compute_x(7), 49)", "critic_benchmark_leakage:expected_hash:"),
            ]
            for leak_snippet, expected_prefix in cases:
                subprocess.run(["git", "reset", "--hard", base_sha], cwd=repo, check=True, capture_output=True)
                rules.write_text(orig_rules + f"\n- {leak_snippet}\n", encoding="utf-8")
                ok, reasons, _ = validate_proposal_rrsi(
                    repo,
                    base_sha,
                    cfg,
                    [],
                    fp["task_ids"],
                    leakage_profile=fp,
                )
                self.assertFalse(ok, msg=f"Expected rejection for {leak_snippet}")
                self.assertTrue(any(expected_prefix in r for r in reasons), msg=f"Got {reasons}")


if __name__ == "__main__":
    unittest.main()
