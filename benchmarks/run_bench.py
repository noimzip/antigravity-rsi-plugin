#!/usr/bin/env python3
"""Mini-benchmark runner for evaluating baseline vs evolved RSI harnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".agents" / "plugins" / "rsi-harness" / "scripts"
if _SCRIPTS_DIR.is_dir() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from adapters.runtime import AgyAdapter, get_runtime_adapter, list_runtime_adapters  # noqa: E402
from adapters.sandbox import get_sandbox_adapter  # noqa: E402
from common import load_rsi_config  # noqa: E402


def _load_rules_text(repo_root: Path) -> str:
    rules_file = repo_root / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"
    if not rules_file.is_file():
        return ""
    return rules_file.read_text(encoding="utf-8")


def _hydrate_harbor_task(task: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    hydrated = dict(task)
    hydrated["_base_dir"] = str(base_dir.resolve())
    hydrated.setdefault("split", "train")
    if hydrated.get("instruction_file") and not hydrated.get("prompt"):
        inst_path = Path(str(hydrated["instruction_file"]))
        if not inst_path.is_absolute():
            inst_path = (base_dir / inst_path).resolve()
        if inst_path.is_file():
            hydrated["prompt"] = inst_path.read_text(encoding="utf-8").strip()
    return hydrated


def load_benchmark_suite(manifest_or_dir: Path) -> dict[str, Any]:
    """Load inline ``tasks.json`` manifests, Harbor task directories, or mixed suites."""
    target = manifest_or_dir.resolve()
    if target.is_dir():
        single_task_json = target / "task.json"
        if single_task_json.is_file():
            task_data = json.loads(single_task_json.read_text(encoding="utf-8"))
            return {
                "suite_name": target.name,
                "tasks": [_hydrate_harbor_task(task_data, target)],
            }
        tasks: list[dict[str, Any]] = []
        for child in sorted(target.iterdir()):
            t_json = child / "task.json"
            if child.is_dir() and t_json.is_file():
                t_data = json.loads(t_json.read_text(encoding="utf-8"))
                tasks.append(_hydrate_harbor_task(t_data, child))
        return {"suite_name": target.name, "tasks": tasks}

    manifest = json.loads(target.read_text(encoding="utf-8"))
    if "tasks" not in manifest and "id" in manifest:
        return {
            "suite_name": str(manifest.get("id") or target.parent.name),
            "tasks": [_hydrate_harbor_task(manifest, target.parent)],
        }

    hydrated_tasks: list[dict[str, Any]] = []
    for entry in manifest.get("tasks", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("task_dir"):
            task_dir = (target.parent / str(entry["task_dir"])).resolve()
            task_json = task_dir / "task.json"
            loaded = json.loads(task_json.read_text(encoding="utf-8")) if task_json.is_file() else {}
            merged = {**loaded, **entry}
            hydrated_tasks.append(_hydrate_harbor_task(merged, task_dir))
        else:
            hydrated_tasks.append(_hydrate_harbor_task(entry, target.parent))
    return {
        "suite_name": manifest.get("suite_name", target.stem),
        "tasks": hydrated_tasks,
    }


def _materialize_task(task: dict[str, Any], dest: Path, repo_root: Path) -> dict[str, Any]:
    """Materialize inline or Harbor task workspace files into ``dest``."""
    base_dir = Path(str(task.get("_base_dir") or repo_root))

    if isinstance(task.get("files"), dict):
        for rel_path, content in task["files"].items():
            target = dest / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    env_cfg = task.get("environment")
    if isinstance(env_cfg, dict):
        if env_cfg.get("source_dir"):
            src_dir = Path(str(env_cfg["source_dir"]))
            if not src_dir.is_absolute():
                src_dir = (base_dir / src_dir).resolve()
            if src_dir.is_dir():
                shutil.copytree(src_dir, dest, dirs_exist_ok=True)

        if env_cfg.get("git_repo"):
            raw_repo = str(env_cfg["git_repo"])
            local_candidate = (base_dir / raw_repo).resolve()
            repo_url = str(local_candidate) if local_candidate.exists() else raw_repo
            with tempfile.TemporaryDirectory(prefix="rsi-harbor-clone-") as clone_tmp:
                clone_dir = Path(clone_tmp) / "repo"
                subprocess.run(
                    ["git", "clone", repo_url, str(clone_dir)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=int(env_cfg.get("timeout_sec", 60)),
                )
                if env_cfg.get("git_ref"):
                    subprocess.run(
                        ["git", "checkout", str(env_cfg["git_ref"])],
                        cwd=str(clone_dir),
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=30,
                    )
                shutil.copytree(clone_dir, dest, dirs_exist_ok=True)

        if env_cfg.get("setup_cmd"):
            setup_env = dict(os.environ)
            if isinstance(env_cfg.get("env"), dict):
                setup_env.update({str(k): str(v) for k, v in env_cfg["env"].items()})
            subprocess.run(
                str(env_cfg["setup_cmd"]),
                cwd=str(dest),
                shell=True,
                env=setup_env,
                capture_output=True,
                text=True,
                check=True,
                timeout=int(env_cfg.get("setup_timeout_sec", 60)),
            )

    plugin_src = repo_root / ".agents" / "plugins" / "rsi-harness"
    if plugin_src.is_dir():
        plugin_dst = dest / ".agents" / "plugins" / "rsi-harness"
        shutil.copytree(plugin_src, plugin_dst, dirs_exist_ok=True)

    cfg_path = dest / "rsi.config.json"
    if not cfg_path.is_file():
        default_cmd = (
            str(task["inner_check_cmd"])
            if task.get("inner_check_cmd") is not None
            else (
                'python3 -m unittest discover -s . -p "test_*.py"'
                if isinstance(task.get("files"), dict)
                else "auto"
            )
        )
        task_rsi_config = {
            "inner_check_cmd": default_cmd,
            "max_inner_retries": 2,
            "evidence_max_chars": 10000,
        }
        cfg_path.write_text(
            json.dumps(task_rsi_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {"base_dir": str(base_dir)}


def _apply_mock_solution(task_id: str, task_dir: Path, rules_text: str) -> tuple[int, str]:
    """Deterministically simulate agent behavior conditioned on general rules."""
    lower = rules_text.lower()
    has_boundary_rule = any(k in lower for k in ("境界", "empty", "zero", "boundary"))
    has_log_rule = any(
        k in lower for k in ("split", "抜粋", "error", "traceback", "ログ", "1回だけ", "noisy")
    )
    has_contract_rule = any(
        k in lower
        for k in ("公開", "signature", "定数", "再実行", "unittest", "contract", "型注釈")
    )

    extra_tokens = max(0, (len(rules_text) - 500) // 2)
    sol_path = task_dir / "solution.py"

    if task_id == "task_01_edge_case":
        if has_boundary_rule:
            sol_path.write_text(
                "def safe_ratio_sum(pairs: list[tuple[float, float]]) -> float:\n"
                "    if not pairs:\n"
                "        return 0.0\n"
                "    total = 0.0\n"
                "    for num, den in pairs:\n"
                "        if den == 0:\n"
                "            raise ValueError('zero denominator')\n"
                "        total += num / den\n"
                "    return total\n",
                encoding="utf-8",
            )
        return 1200 + extra_tokens, "mock_applied_task_01"

    if task_id == "task_02_noisy_log":
        if has_log_rule and ("split(" in lower or "空行" in lower or "検証" in lower or "再実行" in lower):
            sol_path.write_text(
                "def parse_kv_lines(text: str) -> dict[str, str]:\n"
                "    result: dict[str, str] = {}\n"
                "    for line in text.splitlines():\n"
                "        stripped = line.strip()\n"
                "        if not stripped or '=' not in stripped:\n"
                "            continue\n"
                "        k, v = stripped.split('=', 1)\n"
                "        result[k] = v\n"
                "    return result\n",
                encoding="utf-8",
            )
            return 1600 + extra_tokens, "mock_applied_task_02"
        return 3400 + extra_tokens, "mock_missed_task_02"

    if task_id == "task_03_multi_file_rule":
        if has_contract_rule:
            sol_path.write_text(
                "REPORT_VERSION = 'v1'\n\n"
                "def format_summary(items: list[str], limit: int = 3) -> str:\n"
                "    if not items:\n"
                "        return f'{REPORT_VERSION}: (empty)'\n"
                "    head = ', '.join(items[:limit])\n"
                "    extra = len(items) - len(items[:limit])\n"
                "    if extra > 0:\n"
                "        return f'{REPORT_VERSION}: {head} (+{extra} more)'\n"
                "    return f'{REPORT_VERSION}: {head}'\n",
                encoding="utf-8",
            )
            return 1300 + extra_tokens, "mock_applied_task_03"
        return 2900 + extra_tokens, "mock_missed_task_03"

    if task_id == "harbor_sample_calc":
        calc_path = task_dir / "calc.py"
        if has_boundary_rule:
            calc_path.write_text(
                '"""Sample Harbor workspace module."""\n\n\n'
                "def clamped_add(a: int, b: int, max_val: int) -> int:\n"
                '    """Add ``a`` and ``b`` clamped at ``max_val``."""\n'
                "    return min(a + b, max_val)\n",
                encoding="utf-8",
            )
            return 1100 + extra_tokens, "mock_applied_harbor_sample_calc"
        return 2200 + extra_tokens, "mock_missed_harbor_sample_calc"

    return 1500 + extra_tokens, "mock_unknown"


def _run_task_verifier(
    task: dict[str, Any],
    task_dir: Path,
    verifier_stage_dir: Path,
) -> subprocess.CompletedProcess[str]:
    """Execute independent Harbor verifier (outside agent workspace) or default unittest."""
    verifier = task.get("verifier")
    if isinstance(verifier, dict):
        timeout_sec = int(verifier.get("timeout_sec", 30))
        base_dir = Path(str(task.get("_base_dir") or task_dir))
        if verifier.get("script"):
            script_src = Path(str(verifier["script"]))
            if not script_src.is_absolute():
                script_src = (base_dir / script_src).resolve()
            verifier_stage_dir.mkdir(parents=True, exist_ok=True)
            staged = verifier_stage_dir / script_src.name
            shutil.copy2(script_src, staged)
            staged.chmod(staged.stat().st_mode | stat.S_IEXEC)
            env = dict(os.environ)
            env["RSI_TASK_DIR"] = str(task_dir)
            env["RSI_VERIFIER_DIR"] = str(verifier_stage_dir)
            return subprocess.run(
                ["bash", str(staged)],
                cwd=str(task_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        if verifier.get("command"):
            return subprocess.run(
                str(verifier["command"]),
                cwd=str(task_dir),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )

    return subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
        cwd=str(task_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_agy_task(
    prompt: str,
    task_dir: Path,
    timeout_sec: int = 120,
    runtime_config: dict[str, Any] | None = None,
) -> tuple[int, str]:
    res = AgyAdapter(runtime_config).run(prompt, task_dir, timeout_sec=timeout_sec)
    return res.total_tokens, res.raw_output[:500]


def _run_adapter_task(
    runner: str,
    prompt: str,
    task_dir: Path,
    *,
    runtime_config: dict[str, Any] | None = None,
    timeout_sec: int = 120,
) -> tuple[int, str]:
    eff_timeout = int((runtime_config or {}).get("timeout_sec", timeout_sec))
    if runner == "agy" and not runtime_config:
        return _run_agy_task(prompt, task_dir, timeout_sec=eff_timeout)
    adapter = get_runtime_adapter(runner, runtime_config)
    res = adapter.run(prompt, task_dir, timeout_sec=eff_timeout)
    return res.total_tokens, (res.raw_output or res.stderr)[:500]


def summarize_series(values: list[float]) -> dict[str, float]:
    """Compute mean, sample std, SE, median, IQR, min/max, and 95% CI half-width."""
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "se": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95_half_width": 0.0,
        }
    floats = [float(v) for v in values]
    mean = statistics.mean(floats)
    std = statistics.stdev(floats) if n >= 2 else 0.0
    se = std / math.sqrt(n)
    median = statistics.median(floats)
    if n >= 2:
        q25, _, q75 = statistics.quantiles(floats, n=4, method="inclusive")
        iqr = q75 - q25
    else:
        iqr = 0.0
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "se": round(se, 6),
        "median": round(median, 6),
        "iqr": round(iqr, 6),
        "min": round(min(floats), 6),
        "max": round(max(floats), 6),
        "ci95_half_width": round(1.96 * se, 6),
    }


def aggregate_trial_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate summary statistics across multiple benchmark trials."""
    return {
        "pass_rate": summarize_series([float(t.get("pass_rate", 0.0)) for t in trials]),
        "total_tokens": summarize_series([float(t.get("total_tokens", 0)) for t in trials]),
        "duration_sec": summarize_series([float(t.get("duration_sec", 0.0)) for t in trials]),
    }


def _export_task_traces(
    task_dir: Path,
    task_id: str,
    runner: str,
    tokens: int,
    passed: bool,
    export_dir: Path | None,
    *,
    trial_idx: int = 1,
    n_trials: int = 1,
) -> None:
    """Copy task `.rsi/traces/*` (or emit mock trace) into ``export_dir``."""
    if export_dir is None:
        return
    export_dir.mkdir(parents=True, exist_ok=True)
    src_traces = task_dir / ".rsi" / "traces"
    copied = False
    if src_traces.is_dir():
        for conv_dir in src_traces.iterdir():
            if conv_dir.is_dir():
                dst_conv = export_dir / conv_dir.name
                shutil.copytree(conv_dir, dst_conv, dirs_exist_ok=True)
                if n_trials > 1:
                    (dst_conv / f"trial_{trial_idx}.json").write_text(
                        json.dumps(
                            {
                                "trial": trial_idx,
                                "task_id": task_id,
                                "tokens": tokens,
                                "passed": passed,
                            },
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                copied = True
    if not copied:
        fallback_dir = export_dir / f"{runner}-{task_id}"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        (fallback_dir / "summary.json").write_text(
            json.dumps(
                {
                    "conversationId": f"{runner}-{task_id}",
                    "total_tokens": tokens,
                    "errors": 0 if passed else 1,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if n_trials > 1:
            (fallback_dir / f"trial_{trial_idx}.json").write_text(
                json.dumps(
                    {
                        "trial": trial_idx,
                        "task_id": task_id,
                        "tokens": tokens,
                        "passed": passed,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


def _identify_verifier_files(task: dict[str, Any], task_dir: Path) -> list[Path]:
    """Identify test and verifier files inside ``task_dir`` to lock and SHA-256 verify."""
    protected: list[Path] = []
    seen: set[Path] = set()
    for pattern in ("test_*.py", "*_test.py"):
        for candidate in sorted(task_dir.rglob(pattern)):
            rel_parts = candidate.relative_to(task_dir).parts
            if rel_parts and rel_parts[0] in {".agents", ".rsi", ".git", "__pycache__"}:
                continue
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                protected.append(candidate)

    verifier = task.get("verifier")
    if isinstance(verifier, dict):
        for raw_rel in verifier.get("readonly_paths") or []:
            p = (task_dir / str(raw_rel)).resolve(strict=False)
            if p.is_file() and p.is_relative_to(task_dir.resolve()) and p not in seen:
                seen.add(p)
                protected.append(p)
    return protected


def _hash_files(root: Path, files: list[Path]) -> dict[str, str]:
    """Return ``{relative_posix_path: sha256_hex}`` for ``files``."""
    hashes: dict[str, str] = {}
    root_res = root.resolve()
    for p in files:
        if p.is_file() and not p.is_symlink():
            rel = p.resolve().relative_to(root_res).as_posix()
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def _set_readonly(files: list[Path], readonly: bool) -> None:
    """Toggle read-only permissions (``0o444`` vs ``0o644``) on verifier files."""
    for p in files:
        if not p.exists() or p.is_symlink():
            continue
        try:
            mode = p.stat().st_mode
            if readonly:
                os.chmod(p, mode & ~0o222)
            else:
                os.chmod(p, mode | stat.S_IWUSR)
        except OSError:
            pass


def _verify_integrity(root: Path, expected_hashes: dict[str, str]) -> list[str]:
    """Verify SHA-256 checksums of verifier/test files after agent execution."""
    violations: list[str] = []
    root_res = root.resolve()
    for rel, expected in expected_hashes.items():
        target = root_res / rel
        if target.is_symlink():
            violations.append(f"symlink_swapped:{rel}")
        elif not target.is_file():
            violations.append(f"deleted:{rel}")
        else:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                violations.append(f"modified:{rel}")
    return violations


def _filter_tasks_by_split(
    tasks: list[dict[str, Any]],
    split: str | list[str] | None,
) -> tuple[list[dict[str, Any]], str]:
    if split is None:
        return tasks, "all"
    if isinstance(split, str):
        parts = [s.strip().lower() for s in split.split(",") if s.strip()]
    else:
        parts = [str(s).strip().lower() for s in split if str(s).strip()]
    if not parts or "all" in parts:
        return tasks, "all"
    allowed = set(parts)
    filtered = [t for t in tasks if str(t.get("split", "train")).strip().lower() in allowed]
    return filtered, ",".join(sorted(allowed))


def _run_single_trial(
    repo_root: Path,
    tasks: list[dict[str, Any]],
    rules_text: str,
    runner: str,
    trace_export_dir: Path | None,
    trial_idx: int,
    n_trials: int,
    runtime_config: dict[str, Any] | None = None,
    enforce_anti_goodhart: bool = True,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    passed_count = 0
    total_tokens = 0
    tampered_tasks: list[str] = []
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="rsi-bench-") as tmp_root:
        for task in tasks:
            task_id = task["id"]
            task_dir = Path(tmp_root) / task_id
            verifier_stage_dir = Path(tmp_root) / "_verifiers" / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            _materialize_task(task, task_dir, repo_root)

            verifier_files = _identify_verifier_files(task, task_dir)
            pre_hashes = _hash_files(task_dir, verifier_files)
            ro_enabled = enforce_anti_goodhart and bool(
                (task.get("verifier") or {}).get("readonly", True)
            )
            tamper_violations: list[str] = []
            try:
                if ro_enabled and verifier_files:
                    _set_readonly(verifier_files, True)

                if runner == "mock":
                    tokens, note = _apply_mock_solution(task_id, task_dir, rules_text)
                else:
                    tokens, note = _run_adapter_task(
                        runner,
                        task.get("prompt", ""),
                        task_dir,
                        runtime_config=runtime_config,
                    )

                if enforce_anti_goodhart and pre_hashes:
                    tamper_violations = _verify_integrity(task_dir, pre_hashes)

                if tamper_violations:
                    passed = False
                    tampered_tasks.append(task_id)
                    grader_tail = f"ANTI_GOODHART_TAMPER_DETECTED: {tamper_violations}"
                else:
                    grade = _run_task_verifier(task, task_dir, verifier_stage_dir)
                    passed = grade.returncode == 0
                    grader_tail = (grade.stderr or grade.stdout)[-300:]
            finally:
                if verifier_files:
                    _set_readonly(verifier_files, False)

            if passed:
                passed_count += 1
            total_tokens += tokens

            _export_task_traces(
                task_dir,
                task_id,
                runner,
                tokens,
                passed,
                trace_export_dir,
                trial_idx=trial_idx,
                n_trials=n_trials,
            )

            results.append(
                {
                    "id": task_id,
                    "split": str(task.get("split", "train")),
                    "passed": passed,
                    "tampered": bool(tamper_violations),
                    "tamper_violations": tamper_violations,
                    "tokens": tokens,
                    "note": note,
                    "grader_tail": grader_tail,
                }
            )

    total = len(tasks)
    return {
        "trial": trial_idx,
        "total_tasks": total,
        "passed_tasks": passed_count,
        "pass_rate": round(passed_count / total, 4) if total else 0.0,
        "total_tokens": total_tokens,
        "duration_sec": round(time.monotonic() - started, 3),
        "anti_goodhart_passed": len(tampered_tasks) == 0,
        "tampered_tasks": sorted(set(tampered_tasks)),
        "tasks": results,
    }


def run_benchmark(
    repo_root: Path,
    manifest_path: Path,
    runner: str = "mock",
    trace_export_dir: Path | None = None,
    n_trials: int = 1,
    sandbox_backend: str | None = None,
    split: str | list[str] | None = None,
) -> dict[str, Any]:
    """Execute benchmark suite over ``n_trials`` isolated runs and aggregate metrics."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    manifest = load_benchmark_suite(manifest_path)
    rules_text = _load_rules_text(repo_root)
    tasks, split_label = _filter_tasks_by_split(manifest.get("tasks", []), split)
    cfg = load_rsi_config(repo_root)
    runtime_config = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else None
    enforce_ag = bool((cfg.get("rrsi") or {}).get("enforce_anti_goodhart", True))
    if sandbox_backend:
        # Validate requested sandbox adapter early via registry factory
        _ = get_sandbox_adapter({"sandbox": {**(cfg.get("sandbox") or {}), "backend": sandbox_backend}})

    trials: list[dict[str, Any]] = []
    for trial_idx in range(1, n_trials + 1):
        trials.append(
            _run_single_trial(
                repo_root,
                tasks,
                rules_text,
                runner,
                trace_export_dir,
                trial_idx=trial_idx,
                n_trials=n_trials,
                runtime_config=runtime_config,
                enforce_anti_goodhart=enforce_ag,
            )
        )

    stats = aggregate_trial_metrics(trials)
    mean_passed = statistics.mean(float(t["passed_tasks"]) for t in trials) if trials else 0.0
    all_tampered = sorted({tid for t in trials for tid in t.get("tampered_tasks", [])})
    total = len(tasks)
    return {
        "suite": manifest.get("suite_name", "mini-bench"),
        "runner": runner,
        "split": split_label,
        "n_trials": n_trials,
        "total_tasks": total,
        "passed_tasks": int(round(mean_passed)),
        "pass_rate": round(stats["pass_rate"]["mean"], 4),
        "total_tokens": int(round(stats["total_tokens"]["mean"])),
        "duration_sec": round(stats["duration_sec"]["mean"], 3),
        "anti_goodhart_passed": len(all_tampered) == 0,
        "tampered_tasks": all_tampered,
        "stats": stats,
        "trials": trials,
        "tasks": trials[-1]["tasks"] if trials else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RSI-Harness mini benchmark.")
    parser.add_argument("--repo", default=".", help="Repository root containing candidate harness")
    parser.add_argument("--manifest", default="benchmarks/tasks.json", help="Benchmark tasks JSON")
    parser.add_argument(
        "--runner",
        default="mock",
        choices=["mock", *list_runtime_adapters()],
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter: all, train, holdout, ood, or comma-separated",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of repeated benchmark trials for variance aggregation (>= 1)",
    )
    parser.add_argument(
        "--trace-export-dir",
        default=None,
        help="Optional directory to preserve per-task .rsi/traces before temp cleanup",
    )
    parser.add_argument(
        "--sandbox-backend",
        default=None,
        choices=["worktree", "docker", "remote"],
        help="Optional sandbox backend validation override",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be >= 1")

    repo_root = Path(args.repo).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    trace_export_dir = Path(args.trace_export_dir).resolve() if args.trace_export_dir else None

    report = run_benchmark(
        repo_root,
        manifest_path,
        runner=args.runner,
        trace_export_dir=trace_export_dir,
        n_trials=args.trials,
        sandbox_backend=args.sandbox_backend,
        split=args.split,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
