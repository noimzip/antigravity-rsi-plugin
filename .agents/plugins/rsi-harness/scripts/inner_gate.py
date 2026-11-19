#!/usr/bin/env python3
"""Stop hook implementing bounded inner-loop verification + trace persistence."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from adapters.events import normalize_stop_event, serialize_hook_response
from common import load_rsi_config, read_hook_input, write_hook_output
from evidence_reducer import build_summarizer_from_config, reduce_log
from toolchain import resolve_inner_check
from trace_collector import record_normalized_event


def _read_retry_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return 0


def evaluate_stop(payload: dict[str, Any]) -> dict[str, Any]:
    """Run trace persistence and bounded inner-loop verification via normalized events."""
    if not payload:
        return {}

    event = normalize_stop_event(payload)
    record_normalized_event(event, "stop")

    if event.continuation_active:
        return {}
    if os.environ.get("RSI_INNER_GATE_RUNNING") == "1":
        return {}

    cwd = Path(event.cwd or ".")
    conversation_id = event.conversation_id or "unknown"
    config = load_rsi_config(cwd)

    check_cmd, timeout_sec, _ = resolve_inner_check(cwd, config)
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
    timed_out = False
    try:
        result = subprocess.run(
            check_cmd,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_sec,
        )
        if result.returncode == 0:
            return {}
        combined_log = (result.stdout or "") + "\n" + (result.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        out_part = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err_part = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        combined_log = f"{out_part}\n{err_part}\nERROR: Inner check command timed out after {timeout_sec}s".strip()

    retry_file.parent.mkdir(parents=True, exist_ok=True)
    retry_file.write_text(str(retries + 1), encoding="utf-8")

    summarizer = build_summarizer_from_config(config, cwd=cwd)
    reduced = reduce_log(combined_log, max_chars=max_chars, summarizer=summarizer)
    prefix = (
        f"Inner check timed out after {timeout_sec}s (retry {retries + 1}/{max_retries})."
        if timed_out
        else f"Inner check failed (retry {retries + 1}/{max_retries})."
    )
    return serialize_hook_response(
        event.runtime,
        "stop",
        {
            "decision": "continue",
            "reason": f"{prefix} Fix the issue based on verified output:\n{reduced['reduced_text']}",
        },
    )


def main() -> None:
    payload = read_hook_input()
    write_hook_output(evaluate_stop(payload))


if __name__ == "__main__":
    main()
