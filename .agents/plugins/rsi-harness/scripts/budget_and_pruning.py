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

_BULLET_RE = re.compile(r"^(\s*[-*]\s+)(.+?)\s*$")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_NON_ASCII_RUN_RE = re.compile(r"[^\x00-\x7f]+")

_VERBOSE_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bPlease make sure that you always\s+", re.IGNORECASE), "Always "),
    (re.compile(r"\bPlease ensure that you always\s+", re.IGNORECASE), "Always "),
    (re.compile(r"\bMake sure that you always\s+", re.IGNORECASE), "Always "),
    (re.compile(r"\bMake sure to always\s+", re.IGNORECASE), "Always "),
    (re.compile(r"\bBe sure to always\s+", re.IGNORECASE), "Always "),
    (re.compile(r"\bIt is very important to note that\s+", re.IGNORECASE), ""),
    (re.compile(r"\bIt is important to note that\s+", re.IGNORECASE), ""),
    (re.compile(r"\bWhenever possible,\s*please\s+", re.IGNORECASE), ""),
    (re.compile(r"\bIn order to\s+", re.IGNORECASE), "To "),
    (re.compile(r"\bPlease remember to\s+", re.IGNORECASE), ""),
    (re.compile(r"必ず忘れないように毎回"), "必ず"),
    (re.compile(r"することを徹底して心がけること"), "すること"),
    (re.compile(r"することを忘れないように注意してください"), "すること"),
    (re.compile(r"するように心がけてください"), "すること"),
    (re.compile(r"することが非常に重要です"), "すること"),
    (re.compile(r"あらかじめ事前に"), "事前に"),
)


def _simplify_verbose_line(raw_line: str) -> str:
    """Trim boilerplate filler phrases inside a markdown bullet without losing core meaning."""
    m = _BULLET_RE.match(raw_line)
    if not m:
        return raw_line
    prefix, body = m.group(1), m.group(2)
    trimmed = body
    for pat, repl in _VERBOSE_REWRITES:
        trimmed = pat.sub(repl, trimmed)
    if trimmed == body:
        return raw_line
    trimmed = re.sub(r"[ \t]{2,}", " ", trimmed).strip()
    if not trimmed:
        return raw_line
    if trimmed[0].isascii() and trimmed[0].islower() and body[0].isupper():
        trimmed = trimmed[0].upper() + trimmed[1:]
    candidate = f"{prefix}{trimmed}"
    if len(candidate.encode("utf-8")) < len(raw_line.encode("utf-8")):
        return candidate
    return raw_line


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
    pointer_line_chars = int(pruning_cfg.get("pointer_line_chars", 220))

    surface = resolve_mutable_surface(repo, config)
    total_bytes, _, _ = surface.compute_surface_bytes(repo)

    seen_bullets: list[tuple[str, int, str, set[str]]] = []
    redundant_rules: list[dict[str, Any]] = []
    verbose_rules: list[dict[str, Any]] = []
    pointer_candidates: list[dict[str, Any]] = []

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
            body = m.group(2)
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
                simplified = _simplify_verbose_line(raw_line)
                if simplified != raw_line:
                    saved = len(raw_line.encode("utf-8")) - len(simplified.encode("utf-8"))
                    if saved > 0:
                        verbose_rules.append(
                            {
                                "file": rel,
                                "line_number": line_no,
                                "original": raw_line,
                                "trimmed": simplified,
                                "byte_saved": saved,
                            }
                        )
                if len(body) > pointer_line_chars:
                    pointer_candidates.append(
                        {
                            "file": rel,
                            "line_number": line_no,
                            "char_len": len(body),
                            "preview": body[:100],
                        }
                    )

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

    redundant_saved = sum(int(r["byte_len"]) for r in redundant_rules)
    verbose_saved = sum(int(v["byte_saved"]) for v in verbose_rules)
    estimated_saved = redundant_saved + verbose_saved
    should_prune = (
        bool(redundant_rules)
        or bool(verbose_rules)
        or total_bytes > int(surface.total_max_bytes * warn_ratio)
    )
    return {
        "total_surface_bytes": total_bytes,
        "max_surface_bytes": surface.total_max_bytes,
        "redundant_rules": redundant_rules,
        "verbose_rules": verbose_rules,
        "pointer_candidates": pointer_candidates,
        "dead_skills": dead_skills,
        "verbose_bytes_saved": verbose_saved,
        "estimated_bytes_saved": estimated_saved,
        "should_prune": should_prune,
    }


def apply_pruning_pass(
    worktree: Path,
    analysis: dict[str, Any],
    *,
    trim_verbose: bool = True,
) -> list[str]:
    """Remove redundant duplicate rules and optionally simplify verbose phrasing."""
    redundant = analysis.get("redundant_rules") or []
    verbose = (analysis.get("verbose_rules") or []) if trim_verbose else []

    by_file_drop: dict[str, set[int]] = {}
    for item in redundant:
        rel = str(item.get("file") or "")
        ln = item.get("line_number")
        if rel and isinstance(ln, int):
            by_file_drop.setdefault(rel, set()).add(ln)

    by_file_rewrite: dict[str, dict[int, str]] = {}
    for item in verbose:
        rel = str(item.get("file") or "")
        ln = item.get("line_number")
        trimmed = item.get("trimmed")
        if rel and isinstance(ln, int) and isinstance(trimmed, str):
            by_file_rewrite.setdefault(rel, {})[ln] = trimmed

    modified: list[str] = []
    wt = worktree.resolve()
    all_files = sorted(set(by_file_drop) | set(by_file_rewrite))
    for rel in all_files:
        target = (wt / rel).resolve(strict=False)
        if not target.is_file() or not target.is_relative_to(wt):
            continue
        orig_text = target.read_text(encoding="utf-8")
        orig_lines = orig_text.splitlines()
        drop_lines = by_file_drop.get(rel, set())
        rewrites = by_file_rewrite.get(rel, {})

        kept: list[str] = []
        blank_run = 0
        for idx, line in enumerate(orig_lines, start=1):
            if idx in drop_lines:
                continue
            new_line = rewrites.get(idx, line)
            if not new_line.strip():
                blank_run += 1
                if blank_run > 1:
                    continue
            else:
                blank_run = 0
            kept.append(new_line)

        new_text = "\n".join(kept).rstrip() + ("\n" if kept else "")
        if new_text != orig_text:
            target.write_text(new_text, encoding="utf-8")
            modified.append(rel)
    return modified
