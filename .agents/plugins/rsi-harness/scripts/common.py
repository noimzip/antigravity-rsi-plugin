"""Shared helpers for RSI-Harness hooks and Outer Loop scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "inner_check_cmd": "python3 -m unittest discover -s tests",
    "max_inner_retries": 2,
    "evidence_max_chars": 10000,
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
        "max_rules_bytes": 24000,
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
    """Load ``rsi.config.json`` from ``cwd`` if present, merged with defaults."""
    config_path = Path(cwd) / "rsi.config.json"
    if not config_path.is_file():
        return deep_merge(DEFAULT_CONFIG, {})
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return deep_merge(DEFAULT_CONFIG, loaded)
    except (OSError, json.JSONDecodeError):
        pass
    return deep_merge(DEFAULT_CONFIG, {})


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
    """Aggregate token usage, steps, tool calls, and errors from ``transcript.jsonl``."""
    summary: dict[str, Any] = {
        "transcript_path": str(transcript_path or ""),
        "steps": 0,
        "tool_calls": 0,
        "errors": 0,
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    if not transcript_path:
        return summary
    path = Path(transcript_path)
    if not path.is_file():
        return summary

    for entry in load_jsonl(path):
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
    summary["total_tokens"] = (
        summary["input_tokens"] + summary["cache_read_tokens"] + summary["output_tokens"]
    )
    return summary
