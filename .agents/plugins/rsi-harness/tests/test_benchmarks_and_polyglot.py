"""Tests for M3: polyglot toolchains, Harbor contracts, SandboxAdapters, and Anti-Goodhart evaluation."""

from __future__ import annotations

import json
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
from evidence_reducer import deterministic_error_slice, reduce_log  # noqa: E402
from inner_gate import evaluate_stop  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
