"""Shared helpers for RSI-Harness hooks and Outer Loop scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from surface import load_yaml_file

DEFAULT_CONFIG: dict[str, Any] = {
    "inner_check_cmd": "auto",
    "inner_check_timeout_sec": None,
    "max_inner_retries": 2,
    "evidence_max_chars": 10000,
    "mutable_surface": [],
    "protected_surface": [],
    "reducer": {
        "provider": "none",
        "command": None,
        "endpoint": None,
        "model": "",
        "api_key_env": "",
        "timeout_sec": 15,
    },
    "runtime": {
        "default": "agy",
        "model": None,
        "timeout_sec": 120,
        "extra_args": [],
        "pty": {"command": None},
    },
    "sandbox": {
        "backend": "worktree",
        "docker_image": "python:3.12-slim",
        "docker_bin": "docker",
        "network_disabled": True,
        "cpus": 1.0,
        "memory_mb": 1024,
        "pids_limit": 128,
        "read_only_root": False,
        "extra_flags": [],
        "remote_driver": None,
        "remote_endpoint": None,
    },
    "rrsi": {
        "max_files": 2,
        "max_diff_lines": 80,
        "n_trials": 1,
        "noise_z": 1.96,
        "min_pass_rate_gain": 0.05,
        "min_token_reduction_ratio": 0.1,
        "min_latency_reduction_ratio": 0.1,
        "max_token_overhead_ratio": 0.15,
        "max_latency_overhead_ratio": 0.25,
        "pareto_tie_epsilon": 0.005,
        "max_ood_regression": 0.05,
        "enforce_anti_goodhart": True,
        "max_rules_bytes": 24000,
        "rules_budget_mode": "standard",
        "max_skills_bytes": None,
        "annealing": {
            "enabled": True,
            "decay_rate": 0.85,
            "saturation_decay": 0.9,
            "min_files": 1,
            "min_diff_lines": 15,
        },
        "pruning": {
            "jaccard_threshold": 0.75,
            "warn_ratio": 0.75,
        },
        "editable_prefixes": [
            ".agents/plugins/rsi-harness/rules/",
            ".agents/plugins/rsi-harness/skills/",
        ],
        "protected_prefixes": [
            ".agents/plugins/rsi-harness/hooks.json",
            ".agents/plugins/rsi-harness/scripts/",
            "benchmarks/",
        ],
    },
    "plugins": {
        "proposer": None,
        "evaluator": None,
        "critic": None,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a merged dictionary without mutating inputs."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_rsi_config(cwd: str | Path) -> dict[str, Any]:
    """Load config from ``harness.yaml``/``harness.yml`` and ``rsi.config.json`` merged with defaults."""
    cwd_path = Path(cwd)
    merged = deep_merge(DEFAULT_CONFIG, {})

    for yaml_name in ("harness.yaml", "harness.yml"):
        yaml_path = cwd_path / yaml_name
        if yaml_path.is_file():
            try:
                loaded_yaml = load_yaml_file(yaml_path)
                if isinstance(loaded_yaml, dict):
                    merged = deep_merge(merged, loaded_yaml)
            except (OSError, ValueError):
                pass
            break

    config_path = cwd_path / "rsi.config.json"
    if config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                merged = deep_merge(merged, loaded)
        except (OSError, json.JSONDecodeError):
            pass

    return merged


def read_hook_input() -> dict[str, Any]:
    """Read camelCase hook payload from stdin."""
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def write_hook_output(payload: dict[str, Any]) -> None:
    """Write hook response JSON to stdout."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a JSON object as one line, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from ``path``, ignoring malformed or empty lines."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            continue
    return records


def summarize_transcript(transcript_path: str | Path | None) -> dict[str, Any]:
    """Aggregate token usage, steps, tool calls, and errors from ``transcript.jsonl`` or ``AgentTrace``."""
    summary: dict[str, Any] = {
        "transcript_path": str(transcript_path or ""),
        "steps": 0,
        "tool_calls": 0,
        "errors": 0,
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0.0,
    }
    if not transcript_path:
        return summary
    path = Path(transcript_path)
    if not path.is_file():
        return summary

    explicit_total = 0
    for entry in load_jsonl(path):
        if (
            "schema_version" in entry
            and isinstance(entry.get("token_usage"), dict)
            and isinstance(entry.get("execution_steps"), list)
        ):
            exec_steps = entry["execution_steps"]
            summary["steps"] += len(exec_steps)
            for step in exec_steps:
                if not isinstance(step, dict):
                    continue
                if step.get("status") == "ERROR":
                    summary["errors"] += 1
                if step.get("kind") in ("tool", "mcp") or step.get("tool_name"):
                    summary["tool_calls"] += 1
            usage = entry["token_usage"]
            for token_key in ("input_tokens", "cache_read_tokens", "output_tokens"):
                val = usage.get(token_key)
                if isinstance(val, int):
                    summary[token_key] += val
            tot_val = usage.get("total_tokens")
            if isinstance(tot_val, int):
                explicit_total += tot_val
            lat_val = entry.get("latency_ms")
            if isinstance(lat_val, (int, float)):
                summary["latency_ms"] += float(lat_val)
            continue

        summary["steps"] += 1
        if entry.get("status") == "ERROR":
            summary["errors"] += 1
        tool_calls = entry.get("tool_calls")
        if isinstance(tool_calls, list):
            summary["tool_calls"] += len(tool_calls)
        for token_key in ("input_tokens", "cache_read_tokens", "output_tokens"):
            val = entry.get(token_key)
            if isinstance(val, int):
                summary[token_key] += val
        lat_val = entry.get("latency_ms")
        if isinstance(lat_val, (int, float)):
            summary["latency_ms"] += float(lat_val)

    computed = summary["input_tokens"] + summary["cache_read_tokens"] + summary["output_tokens"]
    summary["total_tokens"] = max(computed, explicit_total)
    summary["latency_ms"] = round(summary["latency_ms"], 3)
    return summary
