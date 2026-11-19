"""Runtime-agnostic hook event schemas, normalizers, and response serializers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_PATH_KEYS = ("Path", "TargetFile", "path", "file_path", "target_file", "destination")
_CMD_KEYS = ("CommandLine", "command", "cmd")

_AGY_KEYS = {
    "toolCall",
    "conversationId",
    "transcriptPath",
    "continuationActive",
    "stepIdx",
    "lastUserInput",
    "terminationReason",
    "hookEventName",
}
_CLAUDE_KEYS = {
    "tool_name",
    "tool_input",
    "session_id",
    "stop_hook_active",
    "hook_event_name",
    "transcript_path",
    "tool_response",
}

_RUNTIME_ALIASES = {
    "agy": "antigravity",
    "antigravity": "antigravity",
    "claude": "claude_code",
    "claude-code": "claude_code",
    "claude_code": "claude_code",
    "codex": "codex",
    "codex-cli": "codex",
    "gemini": "gemini",
    "gemini-cli": "gemini",
    "pty": "pty",
    "generic": "generic",
}


@dataclass
class NormalizedToolPreEvent:
    """Normalized PreToolUse event independent of agent runtime schema."""

    runtime: str
    conversation_id: str
    cwd: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    file_paths: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    step_idx: Any = "na"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedToolPostEvent:
    """Normalized PostToolUse event independent of agent runtime schema."""

    runtime: str
    conversation_id: str
    cwd: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    error: Any = None
    step_idx: Any = "na"
    transcript_path: str | None = None
    last_user_input: str = ""
    hook_event_name: str = "post_tool"
    termination_reason: Any = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedStopEvent:
    """Normalized Stop event independent of agent runtime schema."""

    runtime: str
    conversation_id: str
    cwd: str
    continuation_active: bool = False
    transcript_path: str | None = None
    termination_reason: Any = None
    last_user_input: str = ""
    step_idx: Any = "na"
    hook_event_name: str = "stop"
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    error: Any = None
    raw: dict[str, Any] = field(default_factory=dict)


def detect_hook_runtime(payload: dict[str, Any]) -> str:
    """Infer originating agent runtime from hook payload shape."""
    explicit = payload.get("runtime")
    if isinstance(explicit, str) and explicit.strip():
        return _RUNTIME_ALIASES.get(explicit.strip().lower(), explicit.strip().lower())

    keys = set(payload.keys())
    if keys & _AGY_KEYS:
        return "antigravity"
    if keys & _CLAUDE_KEYS:
        return "claude_code"
    if "call_id" in keys or "codex_session_id" in keys or (
        "arguments" in keys and ("name" in keys or "tool" in keys)
    ):
        return "codex"
    if "functionCall" in keys or "gemini_session_id" in keys:
        return "gemini"
    return "antigravity"


def _extract_conversation_id(payload: dict[str, Any]) -> str:
    for key in (
        "conversationId",
        "session_id",
        "sessionId",
        "codex_session_id",
        "gemini_session_id",
    ):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "unknown"


def _coerce_args_dict(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _extract_tool_name_and_args(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if isinstance(payload.get("toolCall"), dict):
        tc = payload["toolCall"]
        name = tc.get("name")
        return (str(name) if name is not None else None), _coerce_args_dict(tc.get("args"))

    if "tool_name" in payload or "tool_input" in payload:
        name = payload.get("tool_name")
        return (str(name) if name is not None else None), _coerce_args_dict(payload.get("tool_input"))

    if isinstance(payload.get("functionCall"), dict):
        fc = payload["functionCall"]
        name = fc.get("name")
        return (str(name) if name is not None else None), _coerce_args_dict(fc.get("args"))

    name = payload.get("name") or payload.get("tool")
    raw_args = (
        payload.get("arguments")
        if "arguments" in payload
        else (payload.get("args") if "args" in payload else payload.get("input"))
    )
    return (str(name) if name is not None else None), _coerce_args_dict(raw_args)


def _extract_string_values(args: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for key in keys:
        val = args.get(key)
        if isinstance(val, str) and val.strip() and val not in found:
            found.append(val)
    return found


def normalize_pre_tool_event(payload: dict[str, Any]) -> NormalizedToolPreEvent:
    """Convert runtime-specific PreToolUse JSON into ``NormalizedToolPreEvent``."""
    runtime = detect_hook_runtime(payload)
    tool_name, tool_args = _extract_tool_name_and_args(payload)
    step_idx = payload["stepIdx"] if "stepIdx" in payload else payload.get("step_idx", "na")
    return NormalizedToolPreEvent(
        runtime=runtime,
        conversation_id=_extract_conversation_id(payload),
        cwd=str(payload.get("cwd") or "."),
        tool_name=tool_name,
        tool_args=tool_args,
        file_paths=_extract_string_values(tool_args, _PATH_KEYS),
        commands=_extract_string_values(tool_args, _CMD_KEYS),
        step_idx=step_idx,
        raw=dict(payload),
    )


def normalize_post_tool_event(
    payload: dict[str, Any],
    default_event_name: str = "post_tool",
) -> NormalizedToolPostEvent:
    """Convert runtime-specific PostToolUse JSON into ``NormalizedToolPostEvent``."""
    runtime = detect_hook_runtime(payload)
    tool_name, tool_args = _extract_tool_name_and_args(payload)
    step_idx = payload["stepIdx"] if "stepIdx" in payload else payload.get("step_idx", "na")
    transcript_raw = payload.get("transcriptPath") or payload.get("transcript_path")
    transcript_path = str(transcript_raw) if transcript_raw else None
    tool_output = (
        payload["toolResult"]
        if "toolResult" in payload
        else (payload["tool_response"] if "tool_response" in payload else payload.get("output"))
    )
    error = payload["error"] if payload.get("error") is not None else payload.get("is_error")
    last_user_input = str(
        payload.get("lastUserInput") or payload.get("prompt") or payload.get("last_user_input") or ""
    )
    hook_event_name = str(
        payload.get("hookEventName") or payload.get("hook_event_name") or default_event_name
    )
    termination_reason = (
        payload.get("terminationReason") or payload.get("stop_reason") or payload.get("reason")
    )
    return NormalizedToolPostEvent(
        runtime=runtime,
        conversation_id=_extract_conversation_id(payload),
        cwd=str(payload.get("cwd") or "."),
        tool_name=tool_name,
        tool_args=tool_args,
        tool_output=tool_output,
        error=error,
        step_idx=step_idx,
        transcript_path=transcript_path,
        last_user_input=last_user_input,
        hook_event_name=hook_event_name,
        termination_reason=termination_reason,
        raw=dict(payload),
    )


def normalize_stop_event(payload: dict[str, Any]) -> NormalizedStopEvent:
    """Convert runtime-specific Stop JSON into ``NormalizedStopEvent``."""
    runtime = detect_hook_runtime(payload)
    if "continuationActive" in payload:
        cont = bool(payload.get("continuationActive"))
    else:
        cont = bool(payload.get("stop_hook_active", False))
    transcript_raw = payload.get("transcriptPath") or payload.get("transcript_path")
    transcript_path = str(transcript_raw) if transcript_raw else None
    step_idx = payload["stepIdx"] if "stepIdx" in payload else payload.get("step_idx", "na")
    tool_name, tool_args = _extract_tool_name_and_args(payload)
    return NormalizedStopEvent(
        runtime=runtime,
        conversation_id=_extract_conversation_id(payload),
        cwd=str(payload.get("cwd") or "."),
        continuation_active=cont,
        transcript_path=transcript_path,
        termination_reason=(
            payload.get("terminationReason") or payload.get("stop_reason") or payload.get("reason")
        ),
        last_user_input=str(
            payload.get("lastUserInput") or payload.get("prompt") or payload.get("last_user_input") or ""
        ),
        step_idx=step_idx,
        hook_event_name=str(
            payload.get("hookEventName") or payload.get("hook_event_name") or "stop"
        ),
        tool_name=tool_name,
        tool_args=tool_args,
        error=payload["error"] if payload.get("error") is not None else payload.get("is_error"),
        raw=dict(payload),
    )


def serialize_hook_response(
    runtime: str,
    event_type: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Serialize internal hook decision into the originating runtime's expected JSON."""
    norm_runtime = _RUNTIME_ALIASES.get(runtime.strip().lower(), runtime.strip().lower())
    if event_type == "post_tool":
        return {}

    if event_type == "pre_tool":
        decision = str(result.get("decision") or "allow")
        reason = result.get("reason")
        if norm_runtime == "claude_code":
            hook_out: dict[str, Any] = {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
            }
            if reason:
                hook_out["permissionDecisionReason"] = str(reason)
            return {"hookSpecificOutput": hook_out}
        out: dict[str, Any] = {"decision": decision}
        if reason:
            out["reason"] = str(reason)
        return out

    if event_type == "stop":
        if not result:
            return {}
        reason = str(result.get("reason") or "")
        if norm_runtime == "claude_code":
            return {"decision": "block", "reason": reason}
        return {
            "decision": str(result.get("decision") or "continue"),
            "reason": reason,
        }

    return dict(result)
