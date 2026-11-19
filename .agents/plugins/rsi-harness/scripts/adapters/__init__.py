"""Runtime and hook event adapters for RSI-Harness."""

from __future__ import annotations

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

__all__ = [
    "RUNTIME_REGISTRY",
    "AgentRuntimeAdapter",
    "AgyAdapter",
    "ClaudeCodeAdapter",
    "CodexCliAdapter",
    "GeminiCliAdapter",
    "PtyWrapperAdapter",
    "RuntimeRunResult",
    "get_runtime_adapter",
    "list_runtime_adapters",
    "register_runtime_adapter",
]
