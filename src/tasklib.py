"""Sample Python module for RSI-Harness testing."""

from __future__ import annotations


def moving_average(values: list[float], window: int) -> list[float]:
    """Return simple moving averages over non-empty slices of size ``window``."""
    if window <= 0:
        raise ValueError("window must be positive")
    if not values or window > len(values):
        return []
    result: list[float] = []
    running = sum(values[:window])
    result.append(running / window)
    for idx in range(window, len(values)):
        running += values[idx] - values[idx - window]
        result.append(running / window)
    return result


def normalize_tags(tags: list[str]) -> list[str]:
    """Trim, lowercase, and deduplicate tags while preserving first-seen order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in tags:
        cleaned = raw.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return normalized
