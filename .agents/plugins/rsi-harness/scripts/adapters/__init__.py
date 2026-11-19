"""Runtime and hook event adapters for RSI-Harness."""

from __future__ import annotations

from .events import (
    NormalizedStopEvent,
    NormalizedToolPostEvent,
    NormalizedToolPreEvent,
    detect_hook_runtime,
    normalize_post_tool_event,
    normalize_pre_tool_event,
    normalize_stop_event,
    serialize_hook_response,
)
from .runtime import (
    RUNTIME_REGISTRY,
    AgentRuntimeAdapter,
    AgyAdapter,
    ClaudeCodeAdapter,
    CodexCliAdapter,
    GeminiCliAdapter,
    PtyWrapperAdapter,
    RuntimeRunResult,
    get_runtime_adapter,
    list_runtime_adapters,
    register_runtime_adapter,
)
from .sandbox import (
    DockerSandbox,
    GitWorktreeSandbox,
    RemoteSandboxAdapter,
    SandboxAdapter,
    SandboxConfig,
    SandboxExecResult,
    get_sandbox_adapter,
)

__all__ = [
    "DockerSandbox",
    "GitWorktreeSandbox",
    "NormalizedStopEvent",
    "NormalizedToolPostEvent",
    "NormalizedToolPreEvent",
    "RUNTIME_REGISTRY",
    "RemoteSandboxAdapter",
    "AgentRuntimeAdapter",
    "AgyAdapter",
    "ClaudeCodeAdapter",
    "CodexCliAdapter",
    "GeminiCliAdapter",
    "PtyWrapperAdapter",
    "RuntimeRunResult",
    "SandboxAdapter",
    "SandboxConfig",
    "SandboxExecResult",
    "detect_hook_runtime",
    "get_runtime_adapter",
    "get_sandbox_adapter",
    "list_runtime_adapters",
    "normalize_post_tool_event",
    "normalize_pre_tool_event",
    "normalize_stop_event",
    "register_runtime_adapter",
    "serialize_hook_response",
]
