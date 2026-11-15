#!/usr/bin/env python3
"""SoL-Pi inspired Evidence-Preserving Reducer with verbatim quote verification."""

from __future__ import annotations

import re
from typing import Any, Callable

ERROR_LINE_RE = re.compile(
    r"(ERROR|FAIL|FAILED|Exception|Traceback|AssertionError|KeyError|ValueError|SyntaxError)",
    re.IGNORECASE,
)


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
