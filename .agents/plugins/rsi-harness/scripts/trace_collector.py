#!/usr/bin/env python3
"""Meta-Harness & SoL-Pi trace telemetry collector for PostToolUse and Stop."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapters.events import (
    NormalizedStopEvent,
    NormalizedToolPostEvent,
    normalize_post_tool_event,
    normalize_stop_event,
)
from common import append_jsonl, read_hook_input, summarize_transcript, write_hook_output
from evidence_reducer import create_observation_handle

OVERSIZED_THRESHOLD_BYTES = 10_000


def _archive_oversized_with_handle(
    cwd: Path, conversation_id: str, step_idx: Any, payload: dict[str, Any]
) -> tuple[str | None, dict[str, Any] | None]:
    """Persist oversized payload and create an ``ObservationPack`` paging handle."""
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized.encode("utf-8")) <= OVERSIZED_THRESHOLD_BYTES:
        return None, None
    artifact_path = cwd / ".rsi" / "artifacts" / conversation_id / f"step_{step_idx}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(serialized, encoding="utf-8")
    handle = create_observation_handle(
        artifact_path,
        step_idx=step_idx,
        tool_name=payload.get("name") if isinstance(payload, dict) else None,
    )
    return str(artifact_path), handle


def _archive_if_oversized(
    cwd: Path, conversation_id: str, step_idx: Any, payload: dict[str, Any]
) -> str | None:
    """Persist oversized tool arguments/errors as an ObservationPack artifact."""
    artifact_ref, _ = _archive_oversized_with_handle(cwd, conversation_id, step_idx, payload)
    return artifact_ref


def record_normalized_event(
    event: NormalizedToolPostEvent | NormalizedStopEvent,
    event_name: str,
) -> dict[str, Any]:
    """Record a normalized hook event and update conversation trace summary."""
    cwd = Path(event.cwd or ".")
    conversation_id = event.conversation_id or "unknown"
    trace_dir = cwd / ".rsi" / "traces" / conversation_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    tool_payload: dict[str, Any] = {}
    if event.tool_name is not None or event.tool_args:
        tool_payload = {"name": event.tool_name, "args": event.tool_args}

    artifact_ref, observation_handle = _archive_oversized_with_handle(
        cwd, conversation_id, event.step_idx, tool_payload
    )

    event_record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime": event.runtime,
        "event": event_name,
        "hookEventName": event.hook_event_name or event_name,
        "stepIdx": event.step_idx,
        "toolName": event.tool_name,
        "error": event.error,
        "terminationReason": event.termination_reason,
        "artifactRef": artifact_ref,
    }
    if observation_handle is not None:
        event_record["observationHandle"] = observation_handle
    if artifact_ref is None and tool_payload:
        event_record["toolArgs"] = event.tool_args

    append_jsonl(trace_dir / "events.jsonl", event_record)

    transcript_summary = summarize_transcript(event.transcript_path)
    summary_payload = {
        "conversationId": conversation_id,
        "runtime": event.runtime,
        "cwd": str(cwd),
        "lastUserInput": event.last_user_input,
        "updatedAt": event_record["timestamp"],
        **transcript_summary,
    }
    (trace_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_payload


def record_event(payload: dict[str, Any], event_name: str) -> dict[str, Any]:
    """Normalize ``payload`` and record the hook event and trace summary."""
    if event_name == "stop":
        event: NormalizedToolPostEvent | NormalizedStopEvent = normalize_stop_event(payload)
    else:
        event = normalize_post_tool_event(payload, default_event_name=event_name)
    return record_normalized_event(event, event_name)


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
