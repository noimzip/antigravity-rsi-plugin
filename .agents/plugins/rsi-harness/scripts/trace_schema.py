"""Canonical OpenTelemetry GenAI (``gen_ai.*``) AgentTrace schema and exporters."""

from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import append_jsonl, load_jsonl

_VALID_STEP_KINDS = {"llm", "tool", "mcp", "hook"}
_VALID_STEP_STATUSES = {"OK", "ERROR"}

_RUNTIME_TO_GEN_AI_SYSTEM = {
    "antigravity": "antigravity",
    "agy": "antigravity",
    "claude_code": "anthropic",
    "claude": "anthropic",
    "codex": "openai",
    "gemini": "google.gemini",
    "pty": "generic.pty",
}


def _safe_nonneg_int(val: Any) -> int:
    try:
        parsed = int(val)
        return max(0, parsed)
    except (TypeError, ValueError):
        return 0


def _compute_span_id(seed: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_OID, seed).hex[:16]


@dataclass
class TokenUsage:
    """Canonical token usage breakdown aligned with OTel ``gen_ai.usage.*``."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def to_otel_attributes(self) -> dict[str, int]:
        attrs = {
            "gen_ai.usage.input_tokens": int(self.input_tokens),
            "gen_ai.usage.output_tokens": int(self.output_tokens),
            "gen_ai.usage.cache_read_input_tokens": int(self.cache_read_tokens),
            "gen_ai.usage.total_tokens": int(self.total_tokens),
        }
        if self.cache_creation_tokens:
            attrs["gen_ai.usage.cache_creation_input_tokens"] = int(self.cache_creation_tokens)
        return attrs

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsage":
        inp = _safe_nonneg_int(data.get("input_tokens"))
        out = _safe_nonneg_int(data.get("output_tokens"))
        cache_r = _safe_nonneg_int(data.get("cache_read_tokens"))
        cache_c = _safe_nonneg_int(data.get("cache_creation_tokens"))
        tot = _safe_nonneg_int(data.get("total_tokens")) or (inp + out + cache_r + cache_c)
        return cls(
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cache_r,
            cache_creation_tokens=cache_c,
            total_tokens=tot,
        )


@dataclass
class ExecutionStep:
    """Single LLM, tool, MCP, or hook execution span inside an ``AgentTrace``."""

    step_index: int
    span_id: str
    parent_span_id: str | None = None
    name: str = "step"
    kind: str = "llm"
    status: str = "OK"
    timestamp: str = ""
    latency_ms: float = 0.0
    tool_name: str | None = None
    error: str | None = None
    artifact_ref: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionStep":
        return cls(
            step_index=int(data.get("step_index", 0)),
            span_id=str(data.get("span_id") or ""),
            parent_span_id=data.get("parent_span_id"),
            name=str(data.get("name") or "step"),
            kind=str(data.get("kind") or "llm"),
            status=str(data.get("status") or "OK"),
            timestamp=str(data.get("timestamp") or ""),
            latency_ms=float(data.get("latency_ms", 0.0)),
            tool_name=data.get("tool_name"),
            error=data.get("error"),
            artifact_ref=data.get("artifact_ref"),
            attributes=dict(data.get("attributes") or {}),
        )


@dataclass
class AgentTrace:
    """Canonical runtime-agnostic AgentTrace aligned with OTel GenAI Semantic Conventions."""

    trace_id: str
    runtime: str
    schema_version: str = "1.0.0"
    parent_harness_version: str = "unknown"
    candidate_harness_version: str = "unknown"
    environment: dict[str, Any] = field(default_factory=dict)
    execution_steps: list[ExecutionStep] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    evaluation_result: dict[str, Any] | None = None
    gate_decision: dict[str, Any] | None = None
    otel_attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "runtime": self.runtime,
            "parent_harness_version": self.parent_harness_version,
            "candidate_harness_version": self.candidate_harness_version,
            "environment": dict(self.environment),
            "execution_steps": [step.to_dict() for step in self.execution_steps],
            "token_usage": self.token_usage.to_dict(),
            "latency_ms": float(self.latency_ms),
            "evaluation_result": self.evaluation_result,
            "gate_decision": self.gate_decision,
            "otel_attributes": dict(self.otel_attributes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentTrace":
        steps = [
            ExecutionStep.from_dict(s)
            for s in (data.get("execution_steps") or [])
            if isinstance(s, dict)
        ]
        usage = TokenUsage.from_dict(data.get("token_usage") or {})
        return cls(
            schema_version=str(data.get("schema_version") or "1.0.0"),
            trace_id=str(data.get("trace_id") or ""),
            runtime=str(data.get("runtime") or "antigravity"),
            parent_harness_version=str(data.get("parent_harness_version") or "unknown"),
            candidate_harness_version=str(data.get("candidate_harness_version") or "unknown"),
            environment=dict(data.get("environment") or {}),
            execution_steps=steps,
            token_usage=usage,
            latency_ms=float(data.get("latency_ms", 0.0)),
            evaluation_result=data.get("evaluation_result"),
            gate_decision=data.get("gate_decision"),
            otel_attributes=dict(data.get("otel_attributes") or {}),
        )


def validate_agent_trace(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate serialized ``AgentTrace`` structure and required OTel ``gen_ai.*`` attributes."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["trace payload must be a dict"]

    required_fields = (
        "schema_version",
        "trace_id",
        "runtime",
        "parent_harness_version",
        "candidate_harness_version",
        "environment",
        "execution_steps",
        "token_usage",
        "latency_ms",
        "otel_attributes",
    )
    for field_name in required_fields:
        if field_name not in data:
            errors.append(f"missing_field:{field_name}")

    if not isinstance(data.get("trace_id"), str) or not str(data.get("trace_id")).strip():
        errors.append("invalid_trace_id")
    if not isinstance(data.get("runtime"), str) or not str(data.get("runtime")).strip():
        errors.append("invalid_runtime")
    if not isinstance(data.get("environment"), dict):
        errors.append("invalid_environment")

    latency = data.get("latency_ms")
    if not isinstance(latency, (int, float)) or latency < 0:
        errors.append("invalid_latency_ms")

    usage = data.get("token_usage")
    if not isinstance(usage, dict):
        errors.append("invalid_token_usage")
    else:
        for tk in ("input_tokens", "output_tokens", "cache_read_tokens", "total_tokens"):
            val = usage.get(tk)
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                errors.append(f"invalid_token_field:{tk}")

    steps = data.get("execution_steps")
    if not isinstance(steps, list):
        errors.append("invalid_execution_steps")
    else:
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"invalid_step_type:{idx}")
                continue
            if not isinstance(step.get("step_index"), int) or isinstance(step.get("step_index"), bool):
                errors.append(f"invalid_step_index:{idx}")
            if not isinstance(step.get("span_id"), str) or not step.get("span_id"):
                errors.append(f"missing_span_id:{idx}")
            if step.get("kind") not in _VALID_STEP_KINDS:
                errors.append(f"invalid_step_kind:{idx}:{step.get('kind')}")
            if step.get("status") not in _VALID_STEP_STATUSES:
                errors.append(f"invalid_step_status:{idx}:{step.get('status')}")
            step_lat = step.get("latency_ms", 0.0)
            if not isinstance(step_lat, (int, float)) or step_lat < 0:
                errors.append(f"invalid_step_latency:{idx}")

    otel_attrs = data.get("otel_attributes")
    if not isinstance(otel_attrs, dict):
        errors.append("invalid_otel_attributes")
    else:
        for req_attr in ("gen_ai.system", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"):
            if req_attr not in otel_attrs:
                errors.append(f"missing_otel_attr:{req_attr}")

    return len(errors) == 0, errors


def _default_environment(cwd: Path | str | None = None) -> dict[str, Any]:
    return {
        "cwd": str(cwd or os.getcwd()),
        "python_version": platform.python_version(),
        "os": platform.system().lower(),
    }


def _build_otel_attributes(
    runtime: str,
    model: str,
    usage: TokenUsage,
    *,
    include_content: bool = False,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "service.name": "rsi-harness",
        "gen_ai.system": _RUNTIME_TO_GEN_AI_SYSTEM.get(runtime.lower(), runtime.lower()),
        "gen_ai.operation.name": "agent.session",
        "rsi.content_captured": bool(include_content),
    }
    if model:
        attrs["gen_ai.request.model"] = model
    attrs.update(usage.to_otel_attributes())
    return attrs


def _is_mcp_tool(tool_name: str, tool_call: dict[str, Any]) -> bool:
    return tool_name.startswith("mcp_") or bool(tool_call.get("server")) or tool_call.get("type") == "mcp"


def transcript_to_agent_trace(
    transcript_path: str | Path | None,
    *,
    events_path: str | Path | None = None,
    trace_id: str | None = None,
    runtime: str = "antigravity",
    model: str = "",
    parent_harness_version: str = "unknown",
    candidate_harness_version: str = "unknown",
    cwd: str | Path | None = None,
    include_content: bool = False,
    evaluation_result: dict[str, Any] | None = None,
    gate_decision: dict[str, Any] | None = None,
) -> AgentTrace:
    """Convert ``transcript.jsonl`` + optional ``events.jsonl`` into canonical ``AgentTrace``."""
    resolved_trace_id = trace_id or uuid.uuid4().hex
    steps: list[ExecutionStep] = []
    inp = 0
    out = 0
    cache_read = 0
    total_latency_ms = 0.0
    step_counter = 0

    if transcript_path and Path(transcript_path).is_file():
        for entry in load_jsonl(Path(transcript_path)):
            idx = entry.get("step_index")
            s_idx = int(idx) if isinstance(idx, int) else step_counter
            step_counter = max(step_counter + 1, s_idx + 1)
            status = "ERROR" if entry.get("status") == "ERROR" else "OK"
            step_lat = float(entry.get("latency_ms", 0.0) or 0.0)
            total_latency_ms += max(0.0, step_lat)

            e_inp = _safe_nonneg_int(entry.get("input_tokens"))
            e_cache = _safe_nonneg_int(entry.get("cache_read_tokens"))
            e_out = _safe_nonneg_int(entry.get("output_tokens"))
            inp += e_inp
            cache_read += e_cache
            out += e_out

            span_id = _compute_span_id(f"{resolved_trace_id}:llm:{s_idx}:{len(steps)}")
            step_attrs: dict[str, Any] = {
                "gen_ai.operation.name": "chat",
                "gen_ai.system": _RUNTIME_TO_GEN_AI_SYSTEM.get(runtime.lower(), runtime.lower()),
                "gen_ai.usage.input_tokens": e_inp,
                "gen_ai.usage.output_tokens": e_out,
            }
            if include_content and entry.get("content") is not None:
                step_attrs["gen_ai.completion" if entry.get("type") == "PLANNER_RESPONSE" else "gen_ai.prompt"] = str(
                    entry.get("content")
                )

            steps.append(
                ExecutionStep(
                    step_index=s_idx,
                    span_id=span_id,
                    name=str(entry.get("type") or f"step_{s_idx}"),
                    kind="llm",
                    status=status,
                    timestamp=str(entry.get("created_at") or entry.get("timestamp") or ""),
                    latency_ms=max(0.0, step_lat),
                    error=str(entry.get("error")) if entry.get("error") else None,
                    attributes=step_attrs,
                )
            )

            tool_calls = entry.get("tool_calls")
            if isinstance(tool_calls, list):
                for t_idx, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    tool_name = str(tc.get("name") or tc.get("tool") or "unknown_tool")
                    is_mcp = _is_mcp_tool(tool_name, tc)
                    tool_attrs: dict[str, Any] = {
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": tool_name,
                        "gen_ai.tool.type": "mcp" if is_mcp else "function",
                    }
                    if is_mcp:
                        tool_attrs["mcp.method.name"] = tool_name
                    if include_content and tc.get("args") is not None:
                        tool_attrs["gen_ai.tool.call.arguments"] = tc.get("args")
                    steps.append(
                        ExecutionStep(
                            step_index=s_idx,
                            span_id=_compute_span_id(f"{resolved_trace_id}:tool:{s_idx}:{t_idx}:{tool_name}"),
                            parent_span_id=span_id,
                            name=tool_name,
                            kind="mcp" if is_mcp else "tool",
                            status="ERROR" if tc.get("error") else "OK",
                            timestamp=str(entry.get("created_at") or ""),
                            tool_name=tool_name,
                            error=str(tc.get("error")) if tc.get("error") else None,
                            attributes=tool_attrs,
                        )
                    )

    if events_path and Path(events_path).is_file():
        for ev_idx, ev in enumerate(load_jsonl(Path(events_path))):
            tool_name = ev.get("toolName")
            is_mcp = isinstance(tool_name, str) and _is_mcp_tool(tool_name, {})
            kind = "mcp" if is_mcp else ("tool" if tool_name else "hook")
            ev_attrs: dict[str, Any] = {
                "gen_ai.operation.name": "execute_tool" if tool_name else "hook.event",
                "hook.event": ev.get("event"),
            }
            if tool_name:
                ev_attrs["gen_ai.tool.name"] = tool_name
            if include_content and ev.get("toolArgs") is not None:
                ev_attrs["gen_ai.tool.call.arguments"] = ev.get("toolArgs")

            raw_step = ev.get("stepIdx")
            s_idx = int(raw_step) if isinstance(raw_step, int) else step_counter + ev_idx
            steps.append(
                ExecutionStep(
                    step_index=s_idx,
                    span_id=_compute_span_id(f"{resolved_trace_id}:hook:{ev_idx}:{ev.get('event')}"),
                    name=str(tool_name or ev.get("event") or "hook"),
                    kind=kind,
                    status="ERROR" if ev.get("error") else "OK",
                    timestamp=str(ev.get("timestamp") or ""),
                    tool_name=str(tool_name) if tool_name else None,
                    error=str(ev.get("error")) if ev.get("error") else None,
                    artifact_ref=ev.get("artifactRef"),
                    attributes=ev_attrs,
                )
            )

    usage = TokenUsage(
        input_tokens=inp,
        output_tokens=out,
        cache_read_tokens=cache_read,
        total_tokens=inp + cache_read + out,
    )
    return AgentTrace(
        trace_id=resolved_trace_id,
        runtime=runtime,
        parent_harness_version=parent_harness_version,
        candidate_harness_version=candidate_harness_version,
        environment=_default_environment(cwd),
        execution_steps=steps,
        token_usage=usage,
        latency_ms=round(total_latency_ms, 3),
        evaluation_result=evaluation_result,
        gate_decision=gate_decision,
        otel_attributes=_build_otel_attributes(runtime, model, usage, include_content=include_content),
    )


def runtime_result_to_agent_trace(
    result: Any,
    *,
    trace_id: str | None = None,
    model: str = "",
    parent_harness_version: str = "unknown",
    candidate_harness_version: str = "unknown",
    cwd: str | Path | None = None,
    include_content: bool = False,
    evaluation_result: dict[str, Any] | None = None,
    gate_decision: dict[str, Any] | None = None,
) -> AgentTrace:
    """Convert a ``RuntimeRunResult`` (or dict) into canonical ``AgentTrace``."""
    data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    runtime = str(data.get("runtime_name") or "unknown")
    resolved_trace_id = trace_id or uuid.uuid4().hex
    inp = _safe_nonneg_int(data.get("input_tokens"))
    out = _safe_nonneg_int(data.get("output_tokens"))
    cache = _safe_nonneg_int(data.get("cache_read_tokens"))
    tot = _safe_nonneg_int(data.get("total_tokens")) or (inp + out + cache)
    latency_ms = round(max(0.0, float(data.get("duration_sec", 0.0))) * 1000.0, 3)

    usage = TokenUsage(
        input_tokens=inp,
        output_tokens=out,
        cache_read_tokens=cache,
        total_tokens=tot,
    )
    step_attrs: dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.system": _RUNTIME_TO_GEN_AI_SYSTEM.get(runtime.lower(), runtime.lower()),
        "process.exit_code": int(data.get("exit_code", 0)),
    }
    if include_content and data.get("raw_output"):
        step_attrs["gen_ai.completion"] = str(data.get("raw_output"))

    step = ExecutionStep(
        step_index=0,
        span_id=_compute_span_id(f"{resolved_trace_id}:runtime:0"),
        name=f"{runtime}.run",
        kind="llm",
        status="OK" if int(data.get("exit_code", 0)) == 0 else "ERROR",
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_ms=latency_ms,
        error=str(data.get("stderr")) if int(data.get("exit_code", 0)) != 0 and data.get("stderr") else None,
        attributes=step_attrs,
    )
    return AgentTrace(
        trace_id=resolved_trace_id,
        runtime=runtime,
        parent_harness_version=parent_harness_version,
        candidate_harness_version=candidate_harness_version,
        environment=_default_environment(cwd),
        execution_steps=[step],
        token_usage=usage,
        latency_ms=latency_ms,
        evaluation_result=evaluation_result,
        gate_decision=gate_decision,
        otel_attributes=_build_otel_attributes(runtime, model, usage, include_content=include_content),
    )


def export_agent_trace_jsonl(
    trace: AgentTrace,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and export ``trace`` as a JSONL record to ``output_path``."""
    payload = trace.to_dict()
    valid, errs = validate_agent_trace(payload)
    if not valid:
        raise ValueError(f"Invalid AgentTrace: {', '.join(errs)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        output_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        append_jsonl(output_path, payload)
    return output_path
