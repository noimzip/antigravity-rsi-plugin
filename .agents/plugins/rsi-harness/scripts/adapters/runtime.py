"""Pluggable CLI runtime adapters (Harbor-inspired) for coding agent execution."""

from __future__ import annotations

import json
import os
import pty
import re
import select
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_TOKEN_FALLBACK_RE = re.compile(r"(?:total_)?tokens?\s*[=:]\s*(\d+)", re.IGNORECASE)


def _to_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _parse_last_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from ``raw_text``, falling back to the last JSON line."""
    stripped = raw_text.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def _parse_json_lines(raw_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            continue
    return records


@dataclass
class RuntimeRunResult:
    """Normalized execution output and token/trace metadata from an agent CLI."""

    runtime_name: str
    command: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    total_tokens: int = 0
    duration_sec: float = 0.0
    trace_paths: list[str] = field(default_factory=list)
    raw_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRuntimeAdapter(ABC):
    """Abstract Harbor-style interface for invoking coding agent CLIs."""

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def build_command(
        self,
        prompt: str,
        cwd: Path,
        *,
        model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build non-interactive CLI argument vector."""

    def build_env(
        self,
        base_env: dict[str, str] | None = None,
        *,
        evolution_mode: bool = False,
        allowed_root: Path | None = None,
    ) -> dict[str, str]:
        """Construct subprocess environment with optional RSI evolution boundary vars."""
        env = dict(base_env if base_env is not None else os.environ)
        if evolution_mode:
            env["RSI_EVOLUTION_MODE"] = "1"
            if allowed_root is not None:
                env["RSI_ALLOWED_ROOT"] = str(allowed_root.resolve())
        return env

    @abstractmethod
    def parse_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_sec: float,
        command: list[str],
    ) -> RuntimeRunResult:
        """Extract token usage, metadata, and normalized output from CLI execution."""

    def resolve_trace_paths(self, cwd: Path, result: RuntimeRunResult) -> list[Path]:
        """Discover trace/transcript artifacts produced inside ``cwd/.rsi/traces``."""
        traces_dir = cwd / ".rsi" / "traces"
        if not traces_dir.is_dir():
            return []
        found: list[Path] = []
        for pattern in ("*/summary.json", "*/events.jsonl", "*/agent_trace.jsonl", "*/pty_transcript.log"):
            found.extend(sorted(traces_dir.glob(pattern)))
        return found

    def run(
        self,
        prompt: str,
        cwd: Path,
        *,
        timeout_sec: int = 120,
        evolution_mode: bool = False,
        allowed_root: Path | None = None,
        model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> RuntimeRunResult:
        """Execute agent CLI non-interactively and return normalized ``RuntimeRunResult``."""
        eff_model = model if model is not None else self.config.get("model")
        eff_extra = extra_args if extra_args is not None else self.config.get("extra_args")
        eff_extra_list = [str(a) for a in eff_extra] if isinstance(eff_extra, list) else None
        cmd = self.build_command(
            prompt,
            cwd,
            model=str(eff_model) if eff_model else None,
            extra_args=eff_extra_list,
        )
        env = self.build_env(evolution_mode=evolution_mode, allowed_root=allowed_root)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr = f"{stderr}\nTimeoutExpired after {timeout_sec}s".strip()
            exit_code = 124
        except FileNotFoundError as exc:
            stdout = ""
            stderr = str(exc)
            exit_code = 127

        duration_sec = round(time.monotonic() - started, 4)
        result = self.parse_output(stdout, stderr, exit_code, duration_sec, cmd)
        discovered = [str(p) for p in self.resolve_trace_paths(cwd, result)]
        for p in discovered:
            if p not in result.trace_paths:
                result.trace_paths.append(p)
        return result


class AgyAdapter(AgentRuntimeAdapter):
    """Adapter for Antigravity CLI (``agy -p ... --output-format json``)."""

    name = "agy"

    def build_command(
        self,
        prompt: str,
        cwd: Path,
        *,
        model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        cmd = [
            "agy",
            "-p",
            prompt,
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
        ]
        if model:
            cmd.extend(["--model", model])
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def parse_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_sec: float,
        command: list[str],
    ) -> RuntimeRunResult:
        data = _parse_last_json_object(stdout)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        inp = _to_int(usage.get("input_tokens"))
        out = _to_int(usage.get("output_tokens"))
        cache = _to_int(usage.get("cache_read_tokens") or usage.get("cache_read_input_tokens"))
        total = _to_int(usage.get("total_tokens")) or (inp + out + cache)
        return RuntimeRunResult(
            runtime_name=self.name,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cache,
            total_tokens=total,
            duration_sec=duration_sec,
            raw_output=stdout.strip() or stderr.strip(),
            metadata={"parsed": data} if data else {},
        )


class ClaudeCodeAdapter(AgentRuntimeAdapter):
    """Adapter for Claude Code CLI (``claude -p ... --output-format json``)."""

    name = "claude"

    def build_command(
        self,
        prompt: str,
        cwd: Path,
        *,
        model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
        ]
        if model:
            cmd.extend(["--model", model])
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def parse_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_sec: float,
        command: list[str],
    ) -> RuntimeRunResult:
        data = _parse_last_json_object(stdout)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        inp = _to_int(usage.get("input_tokens"))
        out = _to_int(usage.get("output_tokens"))
        cache_read = _to_int(
            usage.get("cache_read_input_tokens") or usage.get("cache_read_tokens")
        )
        cache_create = _to_int(usage.get("cache_creation_input_tokens"))
        total = _to_int(usage.get("total_tokens")) or (inp + out + cache_read + cache_create)
        meta: dict[str, Any] = {}
        for key in ("session_id", "total_cost_usd", "subtype", "is_error"):
            if key in data:
                meta[key] = data[key]
        if cache_create:
            meta["cache_creation_input_tokens"] = cache_create
        return RuntimeRunResult(
            runtime_name=self.name,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cache_read,
            total_tokens=total,
            duration_sec=duration_sec,
            raw_output=str(data.get("result") or stdout.strip() or stderr.strip()),
            metadata=meta,
        )


class CodexCliAdapter(AgentRuntimeAdapter):
    """Adapter for OpenAI Codex CLI (``codex exec --json --full-auto``)."""

    name = "codex"

    def build_command(
        self,
        prompt: str,
        cwd: Path,
        *,
        model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        cmd = ["codex", "exec", "--json", "--full-auto"]
        if model:
            cmd.extend(["--model", model])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(prompt)
        return cmd

    def parse_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_sec: float,
        command: list[str],
    ) -> RuntimeRunResult:
        inp = 0
        out = 0
        cache = 0
        total = 0
        events = _parse_json_lines(stdout)
        if not events:
            single = _parse_last_json_object(stdout)
            if single:
                events = [single]

        for ev in events:
            usage = ev.get("usage") if isinstance(ev.get("usage"), dict) else None
            if usage is None and isinstance(ev.get("turn"), dict):
                usage = ev["turn"].get("usage")
            if isinstance(usage, dict):
                inp += _to_int(usage.get("input_tokens"))
                out += _to_int(usage.get("output_tokens"))
                cache += _to_int(
                    usage.get("cached_input_tokens") or usage.get("cache_read_tokens")
                )
                total += _to_int(usage.get("total_tokens"))

        if total == 0:
            total = inp + out
        return RuntimeRunResult(
            runtime_name=self.name,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cache,
            total_tokens=total,
            duration_sec=duration_sec,
            raw_output=stdout.strip() or stderr.strip(),
            metadata={"event_count": len(events)},
        )


class GeminiCliAdapter(AgentRuntimeAdapter):
    """Adapter for Gemini CLI (``gemini -p ... --output-format json --yolo``)."""

    name = "gemini"

    def build_command(
        self,
        prompt: str,
        cwd: Path,
        *,
        model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        cmd = ["gemini", "-p", prompt, "--output-format", "json", "--yolo"]
        if model:
            cmd.extend(["-m", model])
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def parse_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_sec: float,
        command: list[str],
    ) -> RuntimeRunResult:
        data = _parse_last_json_object(stdout)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        inp = _to_int(usage.get("input_tokens"))
        out = _to_int(usage.get("output_tokens"))
        cache = _to_int(usage.get("cache_read_tokens") or usage.get("cached"))
        total = _to_int(usage.get("total_tokens"))

        models_stats = ((data.get("stats") or {}).get("models") or {}) if isinstance(data.get("stats"), dict) else {}
        if isinstance(models_stats, dict) and models_stats and total == 0:
            for model_entry in models_stats.values():
                if not isinstance(model_entry, dict):
                    continue
                toks = model_entry.get("tokens")
                if not isinstance(toks, dict):
                    continue
                inp += _to_int(toks.get("prompt") or toks.get("input"))
                out += _to_int(toks.get("candidates") or toks.get("output"))
                cache += _to_int(toks.get("cached"))
                total += _to_int(toks.get("total"))

        if total == 0:
            total = inp + out
        return RuntimeRunResult(
            runtime_name=self.name,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cache,
            total_tokens=total,
            duration_sec=duration_sec,
            raw_output=str(data.get("response") or stdout.strip() or stderr.strip()),
            metadata={"models": list(models_stats.keys())} if isinstance(models_stats, dict) and models_stats else {},
        )


class PtyWrapperAdapter(AgentRuntimeAdapter):
    """PTY/CLI fallback adapter for runtimes lacking native JSON hooks."""

    name = "pty"

    def build_command(
        self,
        prompt: str,
        cwd: Path,
        *,
        model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        pty_cfg = self.config.get("pty") if isinstance(self.config.get("pty"), dict) else {}
        raw_template = pty_cfg.get("command") or self.config.get("command") or ["sh", "-c", "{prompt}"]
        if isinstance(raw_template, str):
            template = [raw_template]
        else:
            template = [str(part) for part in raw_template]

        replaced = False
        cmd: list[str] = []
        for part in template:
            if "{prompt}" in part:
                cmd.append(part.replace("{prompt}", prompt))
                replaced = True
            else:
                cmd.append(part)
        if model and "--model" not in cmd:
            cmd.extend(["--model", model])
        if extra_args:
            cmd.extend(extra_args)
        if not replaced:
            cmd.append(prompt)
        return cmd

    def parse_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_sec: float,
        command: list[str],
    ) -> RuntimeRunResult:
        clean = _ANSI_ESCAPE_RE.sub("", stdout).replace("\r\n", "\n")
        data = _parse_last_json_object(clean)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        inp = _to_int(usage.get("input_tokens"))
        out = _to_int(usage.get("output_tokens"))
        cache = _to_int(usage.get("cache_read_tokens"))
        total = _to_int(usage.get("total_tokens")) or (inp + out + cache)

        if total == 0:
            match = _TOKEN_FALLBACK_RE.search(clean)
            if match:
                total = int(match.group(1))

        return RuntimeRunResult(
            runtime_name=self.name,
            command=command,
            exit_code=exit_code,
            stdout=clean,
            stderr=stderr,
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cache,
            total_tokens=total,
            duration_sec=duration_sec,
            raw_output=clean.strip() or stderr.strip(),
            metadata={"pty": True},
        )

    def run(
        self,
        prompt: str,
        cwd: Path,
        *,
        timeout_sec: int = 120,
        evolution_mode: bool = False,
        allowed_root: Path | None = None,
        model: str | None = None,
        extra_args: list[str] | None = None,
    ) -> RuntimeRunResult:
        eff_model = model if model is not None else self.config.get("model")
        eff_extra = extra_args if extra_args is not None else self.config.get("extra_args")
        eff_extra_list = [str(a) for a in eff_extra] if isinstance(eff_extra, list) else None
        cmd = self.build_command(
            prompt,
            cwd,
            model=str(eff_model) if eff_model else None,
            extra_args=eff_extra_list,
        )
        env = self.build_env(evolution_mode=evolution_mode, allowed_root=allowed_root)
        started = time.monotonic()

        chunks: list[bytes] = []
        stderr = ""
        exit_code = 0
        try:
            master_fd, slave_fd = pty.openpty()
        except OSError:
            return super().run(
                prompt,
                cwd,
                timeout_sec=timeout_sec,
                evolution_mode=evolution_mode,
                allowed_root=allowed_root,
                model=model,
                extra_args=extra_args,
            )

        proc: subprocess.Popen[bytes] | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=str(cwd),
                env=env,
                close_fds=True,
            )
            os.close(slave_fd)
            slave_fd = -1
            os.set_blocking(master_fd, False)
            deadline = time.monotonic() + timeout_sec

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.kill()
                    exit_code = 124
                    stderr = f"PTY execution timed out after {timeout_sec}s"
                    break
                ready, _, _ = select.select([master_fd], [], [], min(0.05, remaining))
                if ready:
                    try:
                        data = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    chunks.append(data)
                    continue
                if proc.poll() is not None:
                    # Drain any remaining bytes buffered in PTY master.
                    while True:
                        r2, _, _ = select.select([master_fd], [], [], 0.02)
                        if not r2:
                            break
                        try:
                            tail = os.read(master_fd, 4096)
                        except OSError:
                            tail = b""
                        if not tail:
                            break
                        chunks.append(tail)
                    break

            if exit_code == 0:
                exit_code = proc.wait(timeout=5)
        except FileNotFoundError as exc:
            exit_code = 127
            stderr = str(exc)
        finally:
            if slave_fd >= 0:
                try:
                    os.close(slave_fd)
                except OSError:
                    pass
            try:
                os.close(master_fd)
            except OSError:
                pass
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

        duration_sec = round(time.monotonic() - started, 4)
        stdout = b"".join(chunks).decode("utf-8", errors="replace")
        result = self.parse_output(stdout, stderr, exit_code, duration_sec, cmd)

        conv_id = f"pty-{uuid.uuid4().hex[:8]}"
        trace_dir = cwd / ".rsi" / "traces" / conv_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        transcript_log = trace_dir / "pty_transcript.log"
        transcript_log.write_text(result.stdout, encoding="utf-8")
        summary_json = trace_dir / "summary.json"
        summary_json.write_text(
            json.dumps(
                {
                    "conversationId": conv_id,
                    "runtime": self.name,
                    "total_tokens": result.total_tokens,
                    "errors": 0 if result.exit_code == 0 else 1,
                    "pty_transcript": str(transcript_log),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        result.trace_paths = [str(transcript_log), str(summary_json)]
        return result


RUNTIME_REGISTRY: dict[str, type[AgentRuntimeAdapter]] = {
    "agy": AgyAdapter,
    "claude": ClaudeCodeAdapter,
    "claude-code": ClaudeCodeAdapter,
    "codex": CodexCliAdapter,
    "codex-cli": CodexCliAdapter,
    "gemini": GeminiCliAdapter,
    "gemini-cli": GeminiCliAdapter,
    "pty": PtyWrapperAdapter,
}


def register_runtime_adapter(name: str, cls: type[AgentRuntimeAdapter]) -> None:
    """Register a custom ``AgentRuntimeAdapter`` subclass under ``name``."""
    RUNTIME_REGISTRY[name.strip().lower()] = cls


def list_runtime_adapters() -> list[str]:
    """Return sorted registered runtime adapter names (including aliases)."""
    return sorted(RUNTIME_REGISTRY.keys())


def get_runtime_adapter(
    name: str, config: dict[str, Any] | None = None
) -> AgentRuntimeAdapter:
    """Instantiate a registered ``AgentRuntimeAdapter`` for ``name``."""
    key = name.strip().lower()
    cls = RUNTIME_REGISTRY.get(key)
    if cls is None:
        supported = ", ".join(list_runtime_adapters())
        raise ValueError(f"Unknown runtime adapter {name!r}. Supported: {supported}")
    return cls(config=config)
