#!/usr/bin/env python3
"""PreToolUse hook protecting harness code and evaluation benchmarks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from common import load_rsi_config, read_hook_input, write_hook_output

ALWAYS_PROTECTED = (
    ".agents/plugins/rsi-harness/hooks.json",
    ".agents/plugins/rsi-harness/scripts/",
    "benchmarks/",
)


def _extract_candidate_targets(args: dict[str, Any]) -> list[str]:
    """Extract path or command strings from tool call arguments."""
    targets: list[str] = []
    for key in ("Path", "TargetFile", "path", "CommandLine", "command"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            targets.append(val)
    return targets


def _normalize_target(raw: str, cwd: str) -> str:
    """Normalize a path-like string relative to ``cwd`` when possible."""
    try:
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate.resolve().relative_to(Path(cwd).resolve()).as_posix()
    except (ValueError, OSError):
        pass
    return raw.replace("\\", "/").lstrip("./")


def evaluate_tool_guard(payload: dict[str, Any]) -> dict[str, Any]:
    """Return PreToolUse decision dict for ``payload``."""
    if os.environ.get("RSI_EVOLUTION_MODE") == "1":
        return {"decision": "allow"}

    cwd = str(payload.get("cwd") or ".")
    config = load_rsi_config(cwd)
    protected = tuple(config.get("rrsi", {}).get("protected_prefixes", [])) + ALWAYS_PROTECTED

    tool_call = payload.get("toolCall") or {}
    args = tool_call.get("args") or {}
    targets = _extract_candidate_targets(args)

    for raw_target in targets:
        normalized = _normalize_target(raw_target, cwd)
        for prefix in protected:
            clean_prefix = prefix.lstrip("./")
            if clean_prefix in normalized:
                return {
                    "decision": "ask",
                    "reason": (
                        f"Protected RSI harness/benchmark target matched ({clean_prefix}). "
                        "Approve explicitly or use Outer Loop evolution mode."
                    ),
                }
    return {"decision": "allow"}


def main() -> None:
    payload = read_hook_input()
    write_hook_output(evaluate_tool_guard(payload))


if __name__ == "__main__":
    main()
