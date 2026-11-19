"""RRSI annealed edit budget scheduler and structural pruning analyzer."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import load_jsonl  # noqa: E402
from surface import resolve_mutable_surface  # noqa: E402

_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_NON_ASCII_RUN_RE = re.compile(r"[^\x00-\x7f]+")


def _semantic_tokens(text: str) -> set[str]:
    lower = text.lower()
    tokens = set(_ASCII_WORD_RE.findall(lower))
    for run in _NON_ASCII_RUN_RE.findall(lower):
        cleaned = re.sub(r"[\s、。・「」（）()\[\]:：,，.]+", "", run)
        if len(cleaned) < 2:
            if cleaned:
                tokens.add(cleaned)
            continue
        for idx in range(len(cleaned) - 1):
            tokens.add(cleaned[idx : idx + 2])
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


def compute_annealed_budget(
    config: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute generation- and saturation-annealed ``max_files`` & ``max_diff_lines``."""
    rrsi = config.get("rrsi") if isinstance(config.get("rrsi"), dict) else {}
    base_files = max(1, int(rrsi.get("max_files", 2)))
    base_lines = max(1, int(rrsi.get("max_diff_lines", 80)))

    anneal = rrsi.get("annealing") if isinstance(rrsi.get("annealing"), dict) else {}
    enabled = bool(anneal.get("enabled", True))
    decay_rate = float(anneal.get("decay_rate", 0.85))
    saturation_decay = float(anneal.get("saturation_decay", 0.9))
    min_files = max(1, int(anneal.get("min_files", 1)))
    min_lines = max(1, int(anneal.get("min_diff_lines", 15)))

    accepted_gens = sum(1 for h in history if h.get("accepted"))
    saturation_streak = 0
    for h in reversed(history):
        if h.get("accepted"):
            break
        saturation_streak += 1

    if not enabled or (accepted_gens == 0 and saturation_streak == 0):
        return {
            "enabled": enabled,
            "max_files": base_files,
            "max_diff_lines": base_lines,
            "base_max_files": base_files,
            "base_max_diff_lines": base_lines,
            "accepted_generations": accepted_gens,
            "saturation_streak": saturation_streak,
            "decay_factor": 1.0,
        }

    file_factor = decay_rate ** accepted_gens
    line_factor = file_factor * (saturation_decay ** min(saturation_streak, 3))
    eff_files = max(min_files, min(base_files, round(base_files * file_factor)))
    eff_lines = max(min_lines, min(base_lines, round(base_lines * line_factor)))
    return {
        "enabled": enabled,
        "max_files": eff_files,
        "max_diff_lines": eff_lines,
        "base_max_files": base_files,
        "base_max_diff_lines": base_lines,
        "accepted_generations": accepted_gens,
        "saturation_streak": saturation_streak,
        "decay_factor": round(line_factor, 4),
    }


def analyze_pruning_candidates(
    repo_root: Path,
    config: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Identify redundant/duplicate rules and unreferenced dead skills for pruning."""
    repo = repo_root.resolve()
    rrsi = config.get("rrsi") if isinstance(config.get("rrsi"), dict) else {}
    pruning_cfg = rrsi.get("pruning") if isinstance(rrsi.get("pruning"), dict) else {}
    jaccard_threshold = float(pruning_cfg.get("jaccard_threshold", 0.75))
    warn_ratio = float(pruning_cfg.get("warn_ratio", 0.75))

    surface = resolve_mutable_surface(repo, config)
    total_bytes, _, _ = surface.compute_surface_bytes(repo)

    seen_bullets: list[tuple[str, int, str, set[str]]] = []
    redundant_rules: list[dict[str, Any]] = []

    for rule_file in surface.list_existing_files(repo, role="rules"):
        rel = rule_file.resolve().relative_to(repo).as_posix()
        try:
            lines = rule_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, raw_line in enumerate(lines, start=1):
            m = _BULLET_RE.match(raw_line)
            if not m:
                continue
            body = m.group(1)
            toks = _semantic_tokens(body)
            if not toks:
                continue
            matched_prev: tuple[str, int, str] | None = None
            for prev_rel, prev_ln, prev_text, prev_toks in seen_bullets:
                if _jaccard(toks, prev_toks) >= jaccard_threshold:
                    matched_prev = (prev_rel, prev_ln, prev_text)
                    break
            if matched_prev is not None:
                redundant_rules.append(
                    {
                        "file": rel,
                        "line_number": line_no,
                        "text": raw_line,
                        "duplicate_of": f"{matched_prev[0]}:{matched_prev[1]}",
                        "byte_len": len((raw_line + "\n").encode("utf-8")),
                    }
                )
            else:
                seen_bullets.append((rel, line_no, body, toks))

    # Dead skill detection against recent traces & history references.
    trace_corpus_parts: list[str] = [json_part for json_part in (
        str(h.get("changed_files", "")) + " " + str(h.get("proposer_note", ""))
        for h in history
    )]
    traces_dir = repo / ".rsi" / "traces"
    has_trace_evidence = False
    if traces_dir.is_dir():
        for conv_dir in sorted(traces_dir.iterdir()):
            if not conv_dir.is_dir():
                continue
            for fn in ("events.jsonl", "summary.json"):
                fp = conv_dir / fn
                if fp.is_file():
                    has_trace_evidence = True
                    trace_corpus_parts.append(fp.read_text(encoding="utf-8", errors="replace"))
    trace_corpus = "\n".join(trace_corpus_parts)

    dead_skills: list[dict[str, Any]] = []
    if has_trace_evidence or history:
        for skill_file in surface.list_existing_files(repo, role="skills"):
            rel = skill_file.resolve().relative_to(repo).as_posix()
            skill_slug = skill_file.parent.name if skill_file.name == "SKILL.md" else skill_file.stem
            if skill_slug not in trace_corpus and rel not in trace_corpus:
                dead_skills.append(
                    {
                        "file": rel,
                        "skill": skill_slug,
                        "byte_len": len(skill_file.read_bytes()),
                    }
                )

    estimated_saved = sum(int(r["byte_len"]) for r in redundant_rules)
    should_prune = bool(redundant_rules) or total_bytes > int(surface.total_max_bytes * warn_ratio)
    return {
        "total_surface_bytes": total_bytes,
        "max_surface_bytes": surface.total_max_bytes,
        "redundant_rules": redundant_rules,
        "dead_skills": dead_skills,
        "estimated_bytes_saved": estimated_saved,
        "should_prune": should_prune,
    }


def apply_pruning_pass(worktree: Path, analysis: dict[str, Any]) -> list[str]:
    """Remove redundant duplicate rule lines identified by ``analyze_pruning_candidates``."""
    redundant = analysis.get("redundant_rules") or []
    by_file: dict[str, set[int]] = {}
    for item in redundant:
        rel = str(item.get("file") or "")
        ln = item.get("line_number")
        if rel and isinstance(ln, int):
            by_file.setdefault(rel, set()).add(ln)

    modified: list[str] = []
    wt = worktree.resolve()
    for rel, drop_lines in sorted(by_file.items()):
        target = (wt / rel).resolve(strict=False)
        if not target.is_file() or not target.is_relative_to(wt):
            continue
        orig_lines = target.read_text(encoding="utf-8").splitlines()
        kept = [line for idx, line in enumerate(orig_lines, start=1) if idx not in drop_lines]
        if len(kept) < len(orig_lines):
            target.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
            modified.append(rel)
    return modified
