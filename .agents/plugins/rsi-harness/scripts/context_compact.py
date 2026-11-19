#!/usr/bin/env python3
"""SoL-Pi Online Context Compact helper and subtask-boundary hook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adapters.events import normalize_post_tool_event  # noqa: E402
from common import load_jsonl, read_hook_input, write_hook_output  # noqa: E402


def _extract_event_paths(ev: dict[str, Any]) -> list[str]:
    args = ev.get("toolArgs")
    if not isinstance(args, dict):
        return []
    found: list[str] = []
    for k in ("Path", "TargetFile", "path", "file_path", "target_file"):
        v = args.get(k)
        if isinstance(v, str) and v.strip() and v not in found:
            found.append(v.strip())
    return found


def compact_subtask_context(
    events: list[dict[str, Any]],
    *,
    max_keep_recent: int = 5,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """Compact older subtask events while preserving errors, files, and ObservationPack refs."""
    original_json = json.dumps(events, ensure_ascii=False)
    original_chars = len(original_json)

    if max_keep_recent < 0:
        raise ValueError("max_keep_recent must be >= 0")

    if max_keep_recent == 0:
        older = list(events)
        recent: list[dict[str, Any]] = []
    elif len(events) > max_keep_recent:
        older = events[:-max_keep_recent]
        recent = events[-max_keep_recent:]
    else:
        older = []
        recent = list(events)

    digest_lines: list[str] = []
    preserved_files: list[str] = []
    preserved_errors: list[str] = []
    preserved_artifacts: list[str] = []

    for ev in older:
        paths = _extract_event_paths(ev)
        for p in paths:
            if p not in preserved_files:
                preserved_files.append(p)
        err = ev.get("error")
        if err:
            err_str = str(err).splitlines()[0][:160]
            if err_str not in preserved_errors:
                preserved_errors.append(err_str)
        obs = ev.get("observationHandle")
        art_id = (
            obs.get("artifact_id")
            if isinstance(obs, dict) and obs.get("artifact_id")
            else ev.get("artifactRef")
        )
        if art_id and str(art_id) not in preserved_artifacts:
            preserved_artifacts.append(str(art_id))

        parts = [f"step={ev.get('stepIdx', 'na')}", f"tool={ev.get('toolName') or ev.get('event')}"]
        if paths:
            parts.append(f"files={','.join(paths)}")
        if err:
            parts.append(f"err={str(err).splitlines()[0][:80]}")
        if art_id:
            parts.append(f"pack={art_id}")
        digest_lines.append("- " + " | ".join(parts))

    sections: list[str] = []
    if digest_lines:
        sections.append(f"[Compacted Subtask History ({len(older)} events)]\n" + "\n".join(digest_lines))
    if preserved_errors:
        sections.append("[Preserved Errors]\n" + "\n".join(f"- {e}" for e in preserved_errors))
    if preserved_artifacts:
        sections.append("[ObservationPack Handles]\n" + "\n".join(f"- {a}" for a in preserved_artifacts))
    if recent:
        recent_lines = [
            json.dumps(
                {
                    "stepIdx": ev.get("stepIdx"),
                    "event": ev.get("event"),
                    "toolName": ev.get("toolName"),
                    "error": ev.get("error"),
                    "artifactRef": ev.get("artifactRef"),
                    "files": _extract_event_paths(ev),
                },
                ensure_ascii=False,
            )
            for ev in recent
        ]
        sections.append(f"[Recent Events ({len(recent)})]\n" + "\n".join(recent_lines))

    compacted_text = "\n\n".join(sections)
    if len(compacted_text) > max_chars:
        compacted_text = compacted_text[:max_chars]

    compacted_chars = len(compacted_text)
    reduction_ratio = (
        round((original_chars - compacted_chars) / original_chars, 4)
        if original_chars > 0
        else 0.0
    )
    tokens_saved = max(0, (original_chars - compacted_chars) // 4)

    return {
        "total_events": len(events),
        "compacted_events": len(older),
        "recent_events": len(recent),
        "original_chars": original_chars,
        "compacted_chars": compacted_chars,
        "reduction_ratio": reduction_ratio,
        "estimated_tokens_saved": tokens_saved,
        "preserved_files": preserved_files,
        "preserved_errors": preserved_errors,
        "preserved_artifacts": preserved_artifacts,
        "compacted_text": compacted_text,
    }


def evaluate_compact_hook(
    payload: dict[str, Any],
    *,
    max_keep_recent: int = 5,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """Run online context compaction on the conversation's ``events.jsonl`` trace."""
    event = normalize_post_tool_event(payload, default_event_name="compact")
    cwd = Path(event.cwd or ".")
    conv_id = event.conversation_id or "unknown"
    trace_dir = cwd / ".rsi" / "traces" / conv_id
    events = load_jsonl(trace_dir / "events.jsonl")
    result = compact_subtask_context(
        events,
        max_keep_recent=max_keep_recent,
        max_chars=max_chars,
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "compacted_context.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SoL-Pi Online Context Compact.")
    parser.add_argument("--max-keep-recent", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=4000)
    args = parser.parse_args()

    payload = read_hook_input()
    if payload:
        out = evaluate_compact_hook(
            payload,
            max_keep_recent=args.max_keep_recent,
            max_chars=args.max_chars,
        )
        write_hook_output(out)
    else:
        write_hook_output({})


if __name__ == "__main__":
    main()
