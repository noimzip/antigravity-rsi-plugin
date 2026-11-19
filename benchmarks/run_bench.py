#!/usr/bin/env python3
"""Mini-benchmark runner for evaluating baseline vs evolved RSI harnesses."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _load_rules_text(repo_root: Path) -> str:
    rules_file = repo_root / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"
    if not rules_file.is_file():
        return ""
    return rules_file.read_text(encoding="utf-8")


def _materialize_task(task: dict[str, Any], dest: Path, repo_root: Path) -> None:
    for rel_path, content in task["files"].items():
        target = dest / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    plugin_src = repo_root / ".agents" / "plugins" / "rsi-harness"
    if plugin_src.is_dir():
        plugin_dst = dest / ".agents" / "plugins" / "rsi-harness"
        shutil.copytree(plugin_src, plugin_dst, dirs_exist_ok=True)

    task_rsi_config = {
        "inner_check_cmd": 'python3 -m unittest discover -s . -p "test_*.py"',
        "max_inner_retries": 2,
        "evidence_max_chars": 10000,
    }
    (dest / "rsi.config.json").write_text(
        json.dumps(task_rsi_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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

    return 1500 + extra_tokens, "mock_unknown"


def _run_agy_task(prompt: str, task_dir: Path, timeout_sec: int = 120) -> tuple[int, str]:
    cmd = [
        "agy",
        "-p",
        prompt,
        "--dangerously-skip-permissions",
        "--output-format",
        "json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(task_dir),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    tokens = 0
    raw = (proc.stdout or "").strip()
    if raw:
        try:
            data = json.loads(raw)
            usage = data.get("usage") or {}
            tokens = int(
                usage.get("total_tokens")
                or (
                    int(usage.get("input_tokens", 0))
                    + int(usage.get("output_tokens", 0))
                )
            )
        except (ValueError, json.JSONDecodeError, TypeError):
            pass
    return tokens, raw[:500]


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
    if not copied and runner == "mock":
        mock_dir = export_dir / f"mock-{task_id}"
        mock_dir.mkdir(parents=True, exist_ok=True)
        (mock_dir / "summary.json").write_text(
            json.dumps(
                {
                    "conversationId": f"mock-{task_id}",
                    "total_tokens": tokens,
                    "errors": 0 if passed else 1,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if n_trials > 1:
            (mock_dir / f"trial_{trial_idx}.json").write_text(
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


def _run_single_trial(
    repo_root: Path,
    tasks: list[dict[str, Any]],
    rules_text: str,
    runner: str,
    trace_export_dir: Path | None,
    trial_idx: int,
    n_trials: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    passed_count = 0
    total_tokens = 0
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="rsi-bench-") as tmp_root:
        for task in tasks:
            task_id = task["id"]
            task_dir = Path(tmp_root) / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            _materialize_task(task, task_dir, repo_root)

            if runner == "mock":
                tokens, note = _apply_mock_solution(task_id, task_dir, rules_text)
            else:
                tokens, note = _run_agy_task(task["prompt"], task_dir)

            grade = subprocess.run(
                ["python3", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
                cwd=str(task_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            passed = grade.returncode == 0
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
                    "passed": passed,
                    "tokens": tokens,
                    "note": note,
                    "grader_tail": (grade.stderr or grade.stdout)[-300:],
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
        "tasks": results,
    }


def run_benchmark(
    repo_root: Path,
    manifest_path: Path,
    runner: str = "mock",
    trace_export_dir: Path | None = None,
    n_trials: int = 1,
) -> dict[str, Any]:
    """Execute benchmark suite over ``n_trials`` isolated runs and aggregate metrics."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rules_text = _load_rules_text(repo_root)
    tasks = manifest.get("tasks", [])

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
            )
        )

    stats = aggregate_trial_metrics(trials)
    mean_passed = statistics.mean(float(t["passed_tasks"]) for t in trials) if trials else 0.0
    total = len(tasks)
    return {
        "suite": manifest.get("suite_name", "mini-bench"),
        "runner": runner,
        "n_trials": n_trials,
        "total_tasks": total,
        "passed_tasks": int(round(mean_passed)),
        "pass_rate": round(stats["pass_rate"]["mean"], 4),
        "total_tokens": int(round(stats["total_tokens"]["mean"])),
        "duration_sec": round(stats["duration_sec"]["mean"], 3),
        "stats": stats,
        "trials": trials,
        "tasks": trials[-1]["tasks"] if trials else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RSI-Harness mini benchmark.")
    parser.add_argument("--repo", default=".", help="Repository root containing candidate harness")
    parser.add_argument("--manifest", default="benchmarks/tasks.json", help="Benchmark tasks JSON")
    parser.add_argument("--runner", default="mock", choices=["mock", "agy"])
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
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
