#!/usr/bin/env python3
"""Stop hook implementing bounded inner-loop verification + trace persistence."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from common import load_rsi_config, read_hook_input, write_hook_output
from evidence_reducer import build_summarizer_from_config, reduce_log
from trace_collector import record_event


def _read_retry_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return 0


def evaluate_stop(payload: dict[str, Any]) -> dict[str, Any]:
    """Run trace persistence and bounded inner-loop verification."""
    if not payload:
        return {}

    record_event(payload, "stop")

    if payload.get("continuationActive"):
        return {}
    if os.environ.get("RSI_INNER_GATE_RUNNING") == "1":
        return {}

    cwd = Path(str(payload.get("cwd") or "."))
    conversation_id = str(payload.get("conversationId") or "unknown")
    config = load_rsi_config(cwd)

    check_cmd = str(config.get("inner_check_cmd") or "").strip()
    max_retries = int(config.get("max_inner_retries", 2))
    max_chars = int(config.get("evidence_max_chars", 10000))

    if not check_cmd:
        return {}

    retry_file = cwd / ".rsi" / "traces" / conversation_id / "retry_count.txt"
    retries = _read_retry_count(retry_file)
    if retries >= max_retries:
        return {}

    env = dict(os.environ)
    env["RSI_INNER_GATE_RUNNING"] = "1"
    result = subprocess.run(
        check_cmd,
        cwd=str(cwd),
        shell=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    if result.returncode == 0:
        return {}

    retry_file.parent.mkdir(parents=True, exist_ok=True)
    retry_file.write_text(str(retries + 1), encoding="utf-8")

    combined_log = (result.stdout or "") + "\n" + (result.stderr or "")
    summarizer = build_summarizer_from_config(config, cwd=cwd)
    reduced = reduce_log(combined_log, max_chars=max_chars, summarizer=summarizer)
    return {
        "decision": "continue",
        "reason": (
            f"Inner check failed (retry {retries + 1}/{max_retries}). "
            f"Fix the issue based on verified output:\n{reduced['reduced_text']}"
        ),
    }


def main() -> None:
    payload = read_hook_input()
    write_hook_output(evaluate_stop(payload))


if __name__ == "__main__":
    main()
