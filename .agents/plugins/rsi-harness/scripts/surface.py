"""Declarative mutable/protected surface resolution and zero-dependency YAML loader."""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_IGNORED_DIRS = {".git", ".rsi", "__pycache__", ".pytest_cache", ".venv", "node_modules"}


def _strip_inline_comment(line: str) -> str:
    in_sq = False
    in_dq = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == "#" and not in_sq and not in_dq:
            if idx == 0 or line[idx - 1].isspace():
                return line[:idx]
    return line


def _parse_scalar(raw: str) -> Any:
    val = raw.strip()
    if val == "" or val.lower() in {"null", "~"}:
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if val.startswith("[") or val.startswith("{"):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            pass
    try:
        if val.startswith(("0x", "0o")):
            return int(val, 0)
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _split_kv(text: str) -> tuple[str, str]:
    in_sq = False
    in_dq = False
    for idx, ch in enumerate(text):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == ":" and not in_sq and not in_dq:
            if idx + 1 == len(text) or text[idx + 1].isspace():
                return text[:idx].strip().strip("'\""), text[idx + 1 :].strip()
    raise ValueError(f"Invalid YAML mapping line: {text!r}")


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a practical YAML subset (dicts, lists, scalars) using pure stdlib."""
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        loaded = json.loads(stripped)
        return loaded if isinstance(loaded, dict) else {}

    tokens: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        clean = _strip_inline_comment(raw_line).rstrip()
        if not clean.strip():
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        tokens.append((indent, clean.strip()))

    if not tokens:
        return {}

    def _parse_block(idx: int, min_indent: int) -> tuple[Any, int]:
        if idx >= len(tokens):
            return {}, idx
        indent, content = tokens[idx]
        if content.startswith("- "):
            return _parse_list(idx, indent)
        return _parse_dict(idx, indent)

    def _parse_list(idx: int, list_indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        n = len(tokens)
        while idx < n:
            indent, content = tokens[idx]
            if indent < list_indent or not content.startswith("-"):
                break
            if indent > list_indent:
                break
            after = content[1:].strip()
            if not after:
                val, idx = _parse_block(idx + 1, indent + 1)
                items.append(val)
                continue
            if ":" in after and not (after.startswith(("\"", "'", "{", "["))):
                try:
                    k, v_str = _split_kv(after)
                    entry: dict[str, Any] = {}
                    if v_str == "":
                        if idx + 1 < n and tokens[idx + 1][0] > indent:
                            entry[k], idx = _parse_block(idx + 1, tokens[idx + 1][0])
                        else:
                            entry[k] = None
                            idx += 1
                    else:
                        entry[k] = _parse_scalar(v_str)
                        idx += 1
                    # Parse subsequent sibling keys belonging to this list-item dict
                    if idx < n and tokens[idx][0] > indent and not tokens[idx][1].startswith("-"):
                        rest, idx = _parse_dict(idx, tokens[idx][0])
                        if isinstance(rest, dict):
                            entry.update(rest)
                    items.append(entry)
                    continue
                except ValueError:
                    pass
            items.append(_parse_scalar(after))
            idx += 1
        return items, idx

    def _parse_dict(idx: int, dict_indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        n = len(tokens)
        while idx < n:
            indent, content = tokens[idx]
            if indent < dict_indent or content.startswith("-"):
                break
            if indent > dict_indent:
                break
            k, v_str = _split_kv(content)
            if v_str == "":
                if idx + 1 < n and tokens[idx + 1][0] > indent:
                    val, idx = _parse_block(idx + 1, tokens[idx + 1][0])
                    result[k] = val
                elif idx + 1 < n and tokens[idx + 1][0] == indent and tokens[idx + 1][1].startswith("-"):
                    val, idx = _parse_list(idx + 1, indent)
                    result[k] = val
                else:
                    result[k] = None
                    idx += 1
            else:
                result[k] = _parse_scalar(v_str)
                idx += 1
        return result, idx

    parsed, _ = _parse_block(0, tokens[0][0])
    return parsed if isinstance(parsed, dict) else {}


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load YAML file using PyYAML if available, otherwise stdlib ``parse_simple_yaml``."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        return parse_simple_yaml(text)


def _norm_pattern(pat: str) -> str:
    cleaned = pat.replace("\\", "/").strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def _matches_pattern(rel_posix: str, pattern: str) -> bool:
    rel = _norm_pattern(rel_posix)
    pat = _norm_pattern(pattern)
    if not pat:
        return False
    if pat.endswith("/"):
        return rel.startswith(pat) or rel == pat.rstrip("/")
    if any(ch in pat for ch in ("*", "?", "[")):
        return fnmatch.fnmatch(rel, pat)
    return rel == pat or rel.startswith(pat + "/")


@dataclass
class SurfaceEntry:
    """One mutable boundary target declaration."""

    path: str
    role: str = "rules"
    max_bytes: int | None = None


@dataclass
class MutableSurface:
    """Resolved mutable & protected harness boundaries and size budgets."""

    entries: list[SurfaceEntry] = field(default_factory=list)
    protected_patterns: list[str] = field(default_factory=list)
    total_max_bytes: int = 24000

    def matches_protected(self, rel_posix: str) -> str | None:
        for pat in self.protected_patterns:
            if _matches_pattern(rel_posix, pat):
                return pat
        return None

    def matches_mutable(self, rel_posix: str) -> SurfaceEntry | None:
        for entry in self.entries:
            if _matches_pattern(rel_posix, entry.path):
                return entry
        return None

    def _walk_repo_rel_files(self, repo_root: Path) -> list[tuple[str, Path]]:
        root = repo_root.resolve()
        found: list[tuple[str, Path]] = []
        if not root.is_dir():
            return found
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIRS)
            for fname in sorted(filenames):
                full = Path(dirpath) / fname
                rel = full.relative_to(root).as_posix()
                found.append((rel, full))
        return found

    def list_existing_files(
        self,
        repo_root: Path,
        *,
        role: str | None = None,
    ) -> list[Path]:
        matched: list[Path] = []
        seen: set[Path] = set()
        for rel, full in self._walk_repo_rel_files(repo_root):
            if self.matches_protected(rel):
                continue
            for entry in self.entries:
                if role is not None and entry.role != role:
                    continue
                if _matches_pattern(rel, entry.path):
                    if full not in seen:
                        seen.add(full)
                        matched.append(full)
                    break
        return matched

    def primary_rules_path(self, repo_root: Path) -> Path:
        existing_rules = self.list_existing_files(repo_root, role="rules")
        if existing_rules:
            return existing_rules[0]
        for entry in self.entries:
            if entry.role == "rules":
                pat = _norm_pattern(entry.path)
                if not any(ch in pat for ch in ("*", "?", "[")):
                    if pat.endswith("/"):
                        return repo_root / pat / "AGENTS.md"
                    return repo_root / pat
        return repo_root / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"

    def compute_surface_bytes(
        self,
        repo_root: Path,
    ) -> tuple[int, dict[str, int], list[str]]:
        root = repo_root.resolve()
        files = self._walk_repo_rel_files(root)
        unique_bytes: dict[str, int] = {}
        entry_totals: dict[str, int] = {}
        violations: list[str] = []

        for entry in self.entries:
            e_bytes = 0
            for rel, full in files:
                if self.matches_protected(rel):
                    continue
                if _matches_pattern(rel, entry.path):
                    sz = len(full.read_bytes()) if full.is_file() else 0
                    unique_bytes[rel] = sz
                    e_bytes += sz
            entry_totals[entry.path] = e_bytes
            if entry.max_bytes is not None and e_bytes > entry.max_bytes:
                violations.append(
                    f"surface_entry_budget_exceeded:{entry.path}:{e_bytes}>{entry.max_bytes}"
                )

        total_bytes = sum(unique_bytes.values())
        if total_bytes > self.total_max_bytes:
            violations.append(f"rules_budget_exceeded:{total_bytes}>{self.total_max_bytes}")
        return total_bytes, entry_totals, violations


def resolve_mutable_surface(repo_root: Path, config: dict[str, Any]) -> MutableSurface:
    """Resolve declarative ``mutable_surface`` / ``protected_surface`` or legacy RRSI prefixes."""
    rrsi = config.get("rrsi") if isinstance(config.get("rrsi"), dict) else {}
    total_max_bytes = int(rrsi.get("max_rules_bytes", 24000))

    entries: list[SurfaceEntry] = []
    raw_surface = config.get("mutable_surface")
    if isinstance(raw_surface, list) and raw_surface:
        for item in raw_surface:
            if isinstance(item, str) and item.strip():
                entries.append(SurfaceEntry(path=_norm_pattern(item), role="rules"))
            elif isinstance(item, dict) and item.get("path"):
                mb = item.get("max_bytes")
                entries.append(
                    SurfaceEntry(
                        path=_norm_pattern(str(item["path"])),
                        role=str(item.get("role") or "rules"),
                        max_bytes=int(mb) if mb is not None else None,
                    )
                )
    else:
        for pref in rrsi.get("editable_prefixes", []):
            norm = _norm_pattern(str(pref))
            if not norm:
                continue
            role = "skills" if "skill" in norm.lower() else "rules"
            entries.append(SurfaceEntry(path=norm, role=role))

    protected: list[str] = []
    for raw_p in list(config.get("protected_surface") or []) + list(
        rrsi.get("protected_prefixes") or []
    ):
        np = _norm_pattern(str(raw_p))
        if np and np not in protected:
            protected.append(np)

    return MutableSurface(
        entries=entries,
        protected_patterns=protected,
        total_max_bytes=total_max_bytes,
    )
