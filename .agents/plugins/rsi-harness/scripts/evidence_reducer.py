#!/usr/bin/env python3
"""SoL-Pi inspired Evidence-Preserving Reducer & ObservationPack paging handle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ERROR_LINE_RE = re.compile(
    r"("
    r"ERROR|FAIL|FAILED|Exception|Traceback|AssertionError|KeyError|ValueError|TypeError|"
    r"ImportError|ModuleNotFoundError|SyntaxError|ReferenceError|RangeError|"
    r"TS\d{4}:|FAIL\s+|Expected:|Received:|npm ERR!|✕|×|"
    r"error\[E\d{4}\]:|^error:|thread '.*' panicked at|test .* \.\.\. FAILED|assertion .* failed|"
    r"--- FAIL:|FAIL\t|panic:|\.go:\d+(?::\d+)?:|"
    r"Exception in thread|Caused by:|java\.[\w.]+Exception|\[ERROR\]|Tests run:.*Failures: [1-9]|"
    r":\d+:\d+: (?:fatal )?error:|undefined reference to|Segmentation fault|Sanitizer"
    r")",
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def verify_quotes_verbatim(raw_log: str, quotes: list[str]) -> bool:
    """Verify every non-empty quote line matches ``raw_log`` verbatim."""
    non_empty = [line for quote in quotes for line in quote.splitlines() if line.strip()]
    if not non_empty:
        return False
    raw_lines = set(raw_log.splitlines())
    return all(line in raw_lines or line in raw_log for line in non_empty)


def deterministic_error_slice(raw_log: str, max_chars: int = 3000) -> str:
    """Extract error-bearing windows and tail from ``raw_log`` within ``max_chars``."""
    lines = raw_log.splitlines()
    if not lines:
        return ""
    matched_indices = [idx for idx, line in enumerate(lines) if ERROR_LINE_RE.search(line)]
    selected_indices: set[int] = set()
    for idx in matched_indices:
        for offset in range(max(0, idx - 2), min(len(lines), idx + 4)):
            selected_indices.add(offset)

    # Always include the final 15 lines for exit summary context.
    for idx in range(max(0, len(lines) - 15), len(lines)):
        selected_indices.add(idx)

    excerpt = "\n".join(lines[idx] for idx in sorted(selected_indices))
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[-max_chars:]


def parse_summarizer_response(raw_text: str) -> dict[str, Any]:
    """Parse ``{"summary": str, "quotes": list[str]}`` from CLI or LLM endpoint output."""
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Empty summarizer response")
    cleaned = _CODE_FENCE_RE.sub("", cleaned).strip()

    parsed = json.loads(cleaned)
    if isinstance(parsed, dict) and "summary" in parsed:
        return parsed

    # Unwrap OpenAI / Anthropic style JSON envelopes if returned by an HTTP proxy/endpoint.
    if isinstance(parsed, dict):
        if isinstance(parsed.get("choices"), list) and parsed["choices"]:
            content = (parsed["choices"][0].get("message") or {}).get("content", "")
            if isinstance(content, str) and content.strip():
                return parse_summarizer_response(content)
        if isinstance(parsed.get("content"), list) and parsed["content"]:
            first = parsed["content"][0]
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                return parse_summarizer_response(first["text"])
        if isinstance(parsed.get("output"), str) and parsed["output"].strip():
            return parse_summarizer_response(parsed["output"])

    raise ValueError("Summarizer response missing 'summary' field")


def make_cli_summarizer(
    command: list[str] | str,
    timeout_sec: float = 15.0,
    cwd: str | Path | None = None,
) -> Callable[[str], dict[str, Any]]:
    """Build a summarizer callable backed by a local CLI command reading stdin."""
    cmd_parts = shlex.split(command) if isinstance(command, str) else [str(c) for c in command]
    if not cmd_parts:
        raise ValueError("CLI summarizer command must not be empty")

    def _summarize(raw_log: str) -> dict[str, Any]:
        proc = subprocess.run(
            cmd_parts,
            input=raw_log,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"CLI summarizer exited with {proc.returncode}: {(proc.stderr or '')[:200]}"
            )
        return parse_summarizer_response(proc.stdout)

    return _summarize


def make_http_summarizer(
    endpoint: str,
    *,
    model: str = "",
    api_key_env: str = "",
    timeout_sec: float = 15.0,
) -> Callable[[str], dict[str, Any]]:
    """Build a summarizer callable backed by an HTTP JSON endpoint."""
    parsed_url = urllib.parse.urlparse(endpoint)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"Unsupported HTTP summarizer endpoint: {endpoint}")

    def _summarize(raw_log: str) -> dict[str, Any]:
        payload = {
            "model": model,
            "raw_log": raw_log,
            "instruction": (
                "Summarize the failure concisely and extract exact verbatim error quotes "
                'from raw_log as JSON {"summary": str, "quotes": list[str]}.'
            ),
        }
        headers = {"Content-Type": "application/json"}
        if api_key_env:
            token = os.environ.get(api_key_env, "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        return parse_summarizer_response(body)

    return _summarize


def build_summarizer_from_config(
    config: dict[str, Any],
    *,
    cwd: str | Path | None = None,
) -> Callable[[str], dict[str, Any]] | None:
    """Instantiate configured reducer summarizer (``cli`` or ``http``), or ``None``."""
    reducer_cfg = config.get("reducer")
    if not isinstance(reducer_cfg, dict):
        return None
    provider = str(reducer_cfg.get("provider") or "none").strip().lower()
    if provider in {"", "none", "disabled", "off"}:
        return None
    timeout_sec = float(reducer_cfg.get("timeout_sec", 15))
    if provider == "cli":
        cmd = reducer_cfg.get("command")
        if not cmd:
            return None
        return make_cli_summarizer(cmd, timeout_sec=timeout_sec, cwd=cwd)
    if provider == "http":
        endpoint = str(reducer_cfg.get("endpoint") or "").strip()
        if not endpoint:
            return None
        return make_http_summarizer(
            endpoint,
            model=str(reducer_cfg.get("model") or ""),
            api_key_env=str(reducer_cfg.get("api_key_env") or ""),
            timeout_sec=timeout_sec,
        )
    return None


def create_observation_handle(
    artifact_path: Path,
    *,
    step_idx: Any = None,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """Create sidecar metadata, update artifact ``index.json``, and return handle dict."""
    resolved = artifact_path.resolve()
    raw_bytes = resolved.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    line_count = len(raw_text.splitlines())
    byte_size = len(raw_bytes)
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    artifact_id = f"{resolved.parent.name}/{resolved.stem}"
    meta_path = resolved.with_name(f"{resolved.stem}.meta.json")
    index_path = resolved.parent / "index.json"

    handle: dict[str, Any] = {
        "artifact_id": artifact_id,
        "path": str(artifact_path),
        "meta_path": str(meta_path),
        "index_path": str(index_path),
        "byte_size": byte_size,
        "line_count": line_count,
        "sha256": sha256,
        "step_idx": step_idx,
        "tool_name": tool_name,
        "preview_head": raw_text[:200],
        "preview_tail": raw_text[-200:],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(
        json.dumps(handle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    index_data: dict[str, Any] = {"artifacts": {}}
    if index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("artifacts"), dict):
                index_data = loaded
        except (OSError, json.JSONDecodeError):
            pass
    index_data["artifacts"][artifact_id] = {
        "artifact_id": artifact_id,
        "path": str(artifact_path),
        "meta_path": str(meta_path),
        "byte_size": byte_size,
        "line_count": line_count,
        "sha256": sha256,
        "step_idx": step_idx,
        "tool_name": tool_name,
    }
    index_path.write_text(
        json.dumps(index_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return handle


class ObservationPack:
    """Paging handle for reading oversized archived artifacts by lines, bytes, or regex."""

    def __init__(self, path_or_handle: str | Path | dict[str, Any]) -> None:
        if isinstance(path_or_handle, dict):
            raw_path = path_or_handle.get("path") or path_or_handle.get("meta_path")
            if not raw_path:
                raise ValueError("ObservationPack handle dict must include 'path'")
            candidate = Path(str(raw_path))
        else:
            candidate = Path(path_or_handle)

        if candidate.name.endswith(".meta.json") and candidate.is_file():
            meta = json.loads(candidate.read_text(encoding="utf-8"))
            self.meta_path: Path | None = candidate
            self.path = Path(str(meta["path"]))
        else:
            self.path = candidate
            sidecar = candidate.with_name(f"{candidate.stem}.meta.json")
            self.meta_path = sidecar if sidecar.is_file() else None

        if not self.path.is_file():
            raise FileNotFoundError(f"ObservationPack artifact not found: {self.path}")

    def metadata(self) -> dict[str, Any]:
        """Return sidecar metadata if available, otherwise compute metadata dynamically."""
        if self.meta_path is not None and self.meta_path.is_file():
            try:
                loaded = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded
            except (OSError, json.JSONDecodeError):
                pass
        raw_bytes = self.path.read_bytes()
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        return {
            "artifact_id": f"{self.path.parent.name}/{self.path.stem}",
            "path": str(self.path),
            "byte_size": len(raw_bytes),
            "line_count": len(raw_text.splitlines()),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "preview_head": raw_text[:200],
            "preview_tail": raw_text[-200:],
        }

    def read_lines(
        self,
        start_line: int = 1,
        end_line: int | None = None,
        max_bytes: int = 8192,
    ) -> dict[str, Any]:
        """Read 1-indexed inclusive lines within ``max_bytes`` and return paging cursor."""
        if start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {start_line}")
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        if end_line is not None and end_line < start_line:
            raise ValueError(f"end_line ({end_line}) must be >= start_line ({start_line})")

        text = self.path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        if total_lines == 0 or start_line > total_lines:
            return {
                "content": "",
                "start_line": start_line,
                "end_line": start_line - 1,
                "total_lines": total_lines,
                "has_more": False,
                "next_line": None,
            }

        stop = min(total_lines, end_line) if end_line is not None else total_lines
        selected: list[str] = []
        used_bytes = 0
        for idx in range(start_line - 1, stop):
            line_bytes = len(lines[idx].encode("utf-8"))
            if selected and used_bytes + line_bytes > max_bytes:
                break
            selected.append(lines[idx])
            used_bytes += line_bytes

        actual_end = start_line + len(selected) - 1
        has_more = actual_end < total_lines
        return {
            "content": "".join(selected),
            "start_line": start_line,
            "end_line": actual_end,
            "total_lines": total_lines,
            "has_more": has_more,
            "next_line": (actual_end + 1) if has_more else None,
        }

    def read_bytes(self, offset: int = 0, length: int = 4096) -> dict[str, Any]:
        """Read a UTF-8 safe byte page starting at ``offset``."""
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")

        raw = self.path.read_bytes()
        total_bytes = len(raw)
        if offset >= total_bytes:
            return {
                "content": "",
                "offset": offset,
                "length": 0,
                "total_bytes": total_bytes,
                "has_more": False,
                "next_offset": None,
            }

        chunk = raw[offset : offset + length]
        if offset + len(chunk) < total_bytes:
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError as exc:
                if exc.start > 0 and exc.reason == "unexpected end of data":
                    chunk = chunk[: exc.start]
                    text = chunk.decode("utf-8")
                else:
                    text = chunk.decode("utf-8", errors="replace")
        else:
            text = chunk.decode("utf-8", errors="replace")

        actual_len = len(chunk)
        next_off = offset + actual_len
        has_more = next_off < total_bytes
        return {
            "content": text,
            "offset": offset,
            "length": actual_len,
            "total_bytes": total_bytes,
            "has_more": has_more,
            "next_offset": next_off if has_more else None,
        }

    def search(self, pattern: str, max_matches: int = 20) -> list[dict[str, Any]]:
        """Search artifact lines matching regex ``pattern``."""
        if max_matches < 1:
            raise ValueError(f"max_matches must be >= 1, got {max_matches}")
        compiled = re.compile(pattern)
        matches: list[dict[str, Any]] = []
        for idx, line in enumerate(
            self.path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if compiled.search(line):
                matches.append({"line_number": idx, "line": line})
                if len(matches) >= max_matches:
                    break
        return matches


def reduce_log(
    raw_log: str,
    max_chars: int = 10000,
    summarizer: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reduce oversized logs while enforcing verbatim evidence verification.

    When ``summarizer`` returns ``{"summary": str, "quotes": list[str]}``, every quote
    line must match ``raw_log`` verbatim. Any mismatch triggers an automatic fallback
    to ``deterministic_error_slice(raw_log)``.
    """
    raw_bytes = len(raw_log.encode("utf-8"))
    if len(raw_log) <= max_chars:
        return {
            "reduced_text": raw_log,
            "was_reduced": False,
            "verified": True,
            "quotes": [],
            "fallback_used": False,
            "raw_bytes": raw_bytes,
        }

    if summarizer is not None:
        try:
            proposal = summarizer(raw_log)
            summary = str(proposal.get("summary", "")).strip()
            quotes = [str(q) for q in proposal.get("quotes", []) if str(q).strip()]
            if summary and verify_quotes_verbatim(raw_log, quotes):
                evidence_block = "\n".join(f"> {q}" for q in quotes)
                return {
                    "reduced_text": f"{summary}\n\n[Verified Evidence]\n{evidence_block}",
                    "was_reduced": True,
                    "verified": True,
                    "quotes": quotes,
                    "fallback_used": False,
                    "raw_bytes": raw_bytes,
                }
        except Exception:
            pass
        return {
            "reduced_text": deterministic_error_slice(raw_log, max_chars=min(3000, max_chars)),
            "was_reduced": True,
            "verified": False,
            "quotes": [],
            "fallback_used": True,
            "raw_bytes": raw_bytes,
        }

    return {
        "reduced_text": deterministic_error_slice(raw_log, max_chars=min(3000, max_chars)),
        "was_reduced": True,
        "verified": True,
        "quotes": [],
        "fallback_used": False,
        "raw_bytes": raw_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ObservationPack artifact pages.")
    parser.add_argument("--pack", required=True, help="Path to artifact or .meta.json file")
    parser.add_argument("--metadata", action="store_true", help="Print artifact metadata")
    parser.add_argument("--start-line", type=int, default=None, help="1-indexed start line")
    parser.add_argument("--end-line", type=int, default=None, help="1-indexed end line")
    parser.add_argument("--max-bytes", type=int, default=8192, help="Max bytes for line paging")
    parser.add_argument("--offset", type=int, default=None, help="Byte offset for byte paging")
    parser.add_argument("--length", type=int, default=4096, help="Byte length for byte paging")
    parser.add_argument("--search", default=None, help="Regex pattern to search lines")
    parser.add_argument("--max-matches", type=int, default=20, help="Max search matches")
    args = parser.parse_args()

    pack = ObservationPack(args.pack)
    if args.metadata:
        result: Any = pack.metadata()
    elif args.search is not None:
        result = {"matches": pack.search(args.search, max_matches=args.max_matches)}
    elif args.offset is not None:
        result = pack.read_bytes(offset=args.offset, length=args.length)
    else:
        result = pack.read_lines(
            start_line=args.start_line or 1,
            end_line=args.end_line,
            max_bytes=args.max_bytes,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
