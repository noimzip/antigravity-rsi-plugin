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


def _is_within_root(raw_path: str, allowed_root: Path, cwd: str) -> bool:
    """Return True if ``raw_path`` resolves inside ``allowed_root``."""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    try:
        candidate.resolve().relative_to(allowed_root.resolve())
        return True
    except (ValueError, OSError):
        return False


def evaluate_tool_guard(payload: dict[str, Any]) -> dict[str, Any]:
    """Return PreToolUse decision dict for ``payload``."""
    cwd = str(payload.get("cwd") or ".")
    config = load_rsi_config(cwd)
    protected = tuple(config.get("rrsi", {}).get("protected_prefixes", [])) + ALWAYS_PROTECTED

    tool_call = payload.get("toolCall") or {}
    args = tool_call.get("args") or {}
    targets = _extract_candidate_targets(args)
    evo_mode = os.environ.get("RSI_EVOLUTION_MODE") == "1"
    allowed_root_raw = os.environ.get("RSI_ALLOWED_ROOT", "").strip()

    if evo_mode and allowed_root_raw:
        allowed_root = Path(allowed_root_raw)
        for key in ("Path", "TargetFile", "path"):
            val = args.get(key)
            if isinstance(val, str) and val.strip() and not _is_within_root(val, allowed_root, cwd):
                return {
                    "decision": "deny",
                    "reason": (
                        f"Evolution mode restricts edits strictly inside {allowed_root_raw}. "
                        f"Refusing path outside worktree: {val}"
                    ),
                }

    for raw_target in targets:
        normalized = _normalize_target(raw_target, cwd)
        for prefix in protected:
            clean_prefix = prefix.lstrip("./")
            if clean_prefix in normalized:
                return {
                    "decision": "deny" if evo_mode else "ask",
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
