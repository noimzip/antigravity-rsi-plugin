#!/usr/bin/env python3
"""Meta-Harness & SoL-Pi trace telemetry collector for PostToolUse and Stop."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import append_jsonl, read_hook_input, summarize_transcript, write_hook_output

OVERSIZED_THRESHOLD_BYTES = 10_000


def _archive_if_oversized(
    cwd: Path, conversation_id: str, step_idx: Any, payload: dict[str, Any]
) -> str | None:
    """Persist oversized tool arguments/errors as an ObservationPack artifact."""
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized.encode("utf-8")) <= OVERSIZED_THRESHOLD_BYTES:
        return None
    artifact_path = cwd / ".rsi" / "artifacts" / conversation_id / f"step_{step_idx}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(serialized, encoding="utf-8")
    return str(artifact_path)


def record_event(payload: dict[str, Any], event_name: str) -> dict[str, Any]:
    """Record a hook event and update conversation trace summary."""
    cwd = Path(str(payload.get("cwd") or "."))
    conversation_id = str(payload.get("conversationId") or "unknown")
    trace_dir = cwd / ".rsi" / "traces" / conversation_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    tool_call = payload.get("toolCall") or {}
    step_idx = payload.get("stepIdx", "na")
    artifact_ref = _archive_if_oversized(cwd, conversation_id, step_idx, tool_call)

    event_record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
        "hookEventName": payload.get("hookEventName", event_name),
        "stepIdx": step_idx,
        "toolName": tool_call.get("name"),
        "error": payload.get("error"),
        "terminationReason": payload.get("terminationReason"),
        "artifactRef": artifact_ref,
    }
    if artifact_ref is None and tool_call:
        event_record["toolArgs"] = tool_call.get("args")

    append_jsonl(trace_dir / "events.jsonl", event_record)

    transcript_summary = summarize_transcript(payload.get("transcriptPath"))
    summary_payload = {
        "conversationId": conversation_id,
        "cwd": str(cwd),
        "lastUserInput": payload.get("lastUserInput", ""),
        "updatedAt": event_record["timestamp"],
        **transcript_summary,
    }
    (trace_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RSI trace telemetry.")
    parser.add_argument("--event", default="post_tool", choices=["post_tool", "stop"])
    args = parser.parse_args()

    payload = read_hook_input()
    if payload:
        record_event(payload, args.event)
    write_hook_output({})


if __name__ == "__main__":
    main()
