#!/usr/bin/env python3
"""Meta-Harness + RRSI + SoL-Pi Outer Loop Evolver for RSI-Harness.

Analyzes raw traces under `.rsi/traces/`, proposes bounded harness edits in an
isolated git worktree, screens proposals with RRSI regularization & Critic checks,
evaluates quality and token efficiency via `benchmarks/run_bench.py`, and
automatically commits winning candidates to the target evolution branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent


def _resolve_benchmarks_dir() -> Path:
    env_dir = os.environ.get("RSI_BENCHMARKS_DIR", "").strip()
    if env_dir:
        cand = Path(env_dir).resolve()
        if (cand / "run_bench.py").is_file():
            return cand
    local = SCRIPTS_DIR.parents[3] / "benchmarks"
    if (local / "run_bench.py").is_file():
        return local
    try:
        import importlib.util

        spec = importlib.util.find_spec("rsi_harness")
        if spec and spec.origin:
            pkg_bench = Path(spec.origin).resolve().parents[1] / "benchmarks"
            if (pkg_bench / "run_bench.py").is_file():
                return pkg_bench
    except Exception:
        pass
    return local


BENCHMARKS_DIR = _resolve_benchmarks_dir()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if BENCHMARKS_DIR.is_dir() and str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from adapters.runtime import get_runtime_adapter, list_runtime_adapters  # noqa: E402
from adapters.sandbox import SandboxAdapter, get_sandbox_adapter  # noqa: E402
from budget_and_pruning import (  # noqa: E402
    analyze_pruning_candidates,
    apply_pruning_pass,
    compute_annealed_budget,
)
from common import append_jsonl, compute_effective_token_cost, load_jsonl, load_rsi_config  # noqa: E402
from interfaces import (  # noqa: E402
    BenchmarkEvaluator,
    Critic,
    DefaultProposer,
    Evaluator,
    ProposalContext,
    Proposer,
    RrsiCritic,
    load_plugin,
)
from lineage import LineageArchive  # noqa: E402
from pr_reporter import create_review_pr, detect_default_branch  # noqa: E402
try:
    from run_bench import aggregate_trial_metrics, load_benchmark_suite  # noqa: E402
except ImportError:
    def load_benchmark_suite(manifest_or_dir: Path) -> dict[str, Any]:
        return json.loads(manifest_or_dir.read_text(encoding="utf-8"))

    def aggregate_trial_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
        import statistics as _stats

        def _ser(vals: list[float]) -> dict[str, float]:
            n = len(vals)
            if n == 0:
                return {"n": 0, "mean": 0.0, "std": 0.0, "se": 0.0, "median": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0, "ci95_half_width": 0.0}
            mean = _stats.mean(vals)
            std = _stats.stdev(vals) if n >= 2 else 0.0
            se = std / math.sqrt(n)
            return {
                "n": n,
                "mean": round(mean, 6),
                "std": round(std, 6),
                "se": round(se, 6),
                "median": round(_stats.median(vals), 6),
                "iqr": 0.0,
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
                "ci95_half_width": round(1.96 * se, 6),
            }

        return {
            "pass_rate": _ser([float(t.get("pass_rate", 0.0)) for t in trials]),
            "total_tokens": _ser([float(t.get("total_tokens", 0)) for t in trials]),
            "effective_token_cost": _ser([float(t.get("effective_token_cost", t.get("total_tokens", 0))) for t in trials]),
            "input_tokens": _ser([float(t.get("input_tokens", 0)) for t in trials]),
            "cache_read_tokens": _ser([float(t.get("cache_read_tokens", 0)) for t in trials]),
            "output_tokens": _ser([float(t.get("output_tokens", 0)) for t in trials]),
            "tool_calls": _ser([float(t.get("tool_calls", 0)) for t in trials]),
            "steps": _ser([float(t.get("steps", 0)) for t in trials]),
            "duration_sec": _ser([float(t.get("duration_sec", 0.0)) for t in trials]),
        }
from surface import resolve_mutable_surface  # noqa: E402
from trace_store import TraceStore  # noqa: E402
from weakness_miner import mine_weaknesses, synthesize_regression_suite  # noqa: E402


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=check,
    )


def _ensure_branch(repo: Path, branch: str) -> str:
    """Return commit SHA for ``branch``, creating it from HEAD if missing."""
    probe = _git(repo, "rev-parse", "--verify", f"refs/heads/{branch}", check=False)
    if probe.returncode == 0:
        return probe.stdout.strip()
    head_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "branch", branch, head_sha)
    return head_sha


def _collect_trace_context(repo: Path) -> dict[str, Any]:
    """Read raw trace summaries and recent error events (Meta-Harness style)."""
    traces_root = repo / ".rsi" / "traces"
    summaries: list[dict[str, Any]] = []
    error_samples: list[dict[str, Any]] = []
    if traces_root.is_dir():
        for summary_file in sorted(traces_root.glob("*/summary.json"))[-10:]:
            try:
                summaries.append(json.loads(summary_file.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        for events_file in sorted(traces_root.glob("*/events.jsonl"))[-10:]:
            for ev in load_jsonl(events_file):
                if ev.get("error"):
                    error_samples.append(ev)
    store_entries = load_jsonl(repo / ".rsi" / "store" / "index.jsonl")
    return {
        "summaries": summaries,
        "error_samples": error_samples[-20:],
        "history_count": len(load_jsonl(repo / ".rsi" / "evolution_history.jsonl")),
        "store_generations": len(store_entries),
        "recent_store_entries": store_entries[-5:],
    }


def _mock_propose(
    worktree: Path,
    trace_ctx: dict[str, Any],
    history: list[dict[str, Any]],
    override_mode: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Deterministic Proposer for offline tests and dry-runs."""
    cfg = config or load_rsi_config(worktree)
    surface = resolve_mutable_surface(worktree, cfg)
    rules_path = surface.primary_rules_path(worktree)
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    current = rules_path.read_text(encoding="utf-8") if rules_path.is_file() else ""

    if override_mode == "leak":
        rules_path.write_text(
            current + "\n- task_02_noisy_log では parse_kv_lines を修正すること。\n",
            encoding="utf-8",
        )
        return "Injected intentional benchmark leak for testing Critic"

    if override_mode == "budget":
        extra = "\n".join(f"- padding rule line {i}" for i in range(95))
        rules_path.write_text(current + "\n" + extra + "\n", encoding="utf-8")
        return "Injected oversized edit for testing RRSI budget"

    if override_mode == "protected":
        hooks_path = worktree / ".agents" / "plugins" / "rsi-harness" / "hooks.json"
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text("{}\n", encoding="utf-8")
        return "Injected protected file edit for testing guard"

    seen_hashes = {h.get("diff_hash") for h in history}
    additions = [
        (
            "\n- 文字列パースやログ解析では空行・境界・分割数を検証し、"
            "編集後は必ず unittest を再実行して既存の公開関数・型注釈・定数仕様を維持すること。\n"
        ),
        (
            "\n- ログ調査時は失敗行周辺と境界条件を検証し、"
            "公開インターフェースを変えずに最小差分で修正後に再実行すること。\n"
        ),
    ]
    for addition in additions:
        candidate_text = current.rstrip() + addition
        rules_path.write_text(candidate_text, encoding="utf-8")
        diff_probe = _git(worktree, "diff", check=False).stdout
        candidate_hash = hashlib.sha256(diff_probe.encode("utf-8")).hexdigest()[:16]
        if candidate_hash not in seen_hashes:
            break

    return (
        f"Propose generalized post-edit verification & parsing rule "
        f"(traces={len(trace_ctx['summaries'])}, errors={len(trace_ctx['error_samples'])})"
    )


def _seed_worktree_traces(repo: Path, worktree: Path) -> None:
    """Mount historical store, trace archive, and history into ``worktree`` via symlinks."""
    TraceStore(repo).mount_readonly_into_worktree(worktree)


def _build_proposer_prompt(
    resolved_wt: Path,
    trace_ctx: dict[str, Any],
    history: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> str:
    cfg = config or load_rsi_config(resolved_wt)
    rrsi = cfg.get("rrsi") if isinstance(cfg.get("rrsi"), dict) else {}
    surface = resolve_mutable_surface(resolved_wt, cfg)
    max_files = int(rrsi.get("max_files", 2))
    max_diff_lines = int(rrsi.get("max_diff_lines", 80))
    rules_budget_mode = str(rrsi.get("rules_budget_mode") or "standard").strip().lower()
    role_bytes = surface.compute_role_bytes(resolved_wt)
    rules_only_bytes = int(role_bytes.get("rules", 0))
    skills_bytes = int(role_bytes.get("skills", 0))

    target_desc = (
        ", ".join(f"`{resolved_wt}/{e.path}` ({e.role})" for e in surface.entries)
        or f"`{resolved_wt}/.agents/plugins/rsi-harness/rules/AGENTS.md`"
    )
    protected_desc = ", ".join(f"`{p}`" for p in surface.protected_patterns) or "`hooks.json`, `scripts/`, `benchmarks/`"
    history_tail = json.dumps(history[-5:], ensure_ascii=False)
    trace_summary = json.dumps(trace_ctx, ensure_ascii=False)

    ratchet_note = ""
    if rules_budget_mode in {"ratchet", "progressive_disclosure"} or rules_only_bytes > surface.total_max_bytes:
        ratchet_note = (
            f"6. RATCHET / PROGRESSIVE DISCLOSURE ACTIVE (mode={rules_budget_mode}, "
            f"current_rules_bytes={rules_only_bytes}, skills_bytes={skills_bytes}, "
            f"surface_limit={surface.total_max_bytes}): "
            "Do NOT increase total `role=rules` bytes (`AGENTS.md` / `CLAUDE.md`); "
            "shrink or refactor rules into single-level `skills/<name>/SKILL.md` (or `.agents/skills/<name>/SKILL.md`) "
            "using concise pointer entries.\n"
        )

    return (
        f"You are a Meta-Harness Proposer working ONLY inside worktree `{resolved_wt}`.\n"
        f"Target mutable surface you may edit: {target_desc}.\n"
        "Constraints (RRSI) & Token-Efficiency Principles:\n"
        f"1. NEVER read/edit any file outside `{resolved_wt}` (do NOT touch parent directories).\n"
        f"2. Edit at most {max_files} files, <= {max_diff_lines} diff lines, and <= {surface.total_max_bytes} total surface bytes "
        "ONLY within the declared mutable surface.\n"
        f"3. Never touch protected paths: {protected_desc}.\n"
        "4. Do NOT hardcode task names or test filenames; propose generalizable engineering heuristics.\n"
        "5. Progressive Disclosure & Token Tax Awareness:\n"
        "   - Every byte in `AGENTS.md` / `CLAUDE.md` is injected into EVERY turn as an input token tax and increases Attention Dilution.\n"
        "   - Avoid overly broad mandatory verification rules that force extra `run_command` retries/tokens on unrelated tasks.\n"
        "   - Keep `AGENTS.md` concise (~<= 60 lines of universal invariants + index pointers) and offload situational playbooks into single-level `skills/<name>/SKILL.md` files (Pointer Pattern).\n"
        "   - Prefer compressing, deduplicating, or extracting redundant prose over append-only additions.\n"
        f"{ratchet_note}"
        f"Current surface footprint: rules_bytes={rules_only_bytes}, skills_bytes={skills_bytes}.\n"
        f"Recent trace context: {trace_summary}\n"
        f"Recent evolution history (avoid failed hypotheses): {history_tail}\n"
        f"Inspect `{resolved_wt}/.rsi/traces/` and uncompressed history in `{resolved_wt}/.rsi/store/` "
        f"(or run `python3 .agents/plugins/rsi-harness/scripts/trace_store.py --repo {resolved_wt} query/grep`) "
        f"if present, then make one concise improvement inside `{resolved_wt}`."
    )


def _runtime_propose(
    runner: str,
    worktree: Path,
    trace_ctx: dict[str, Any],
    history: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> str:
    """Invoke configured runtime adapter as Meta-Harness Proposer inside ``worktree``."""
    resolved_wt = worktree.resolve()
    prompt = _build_proposer_prompt(resolved_wt, trace_ctx, history, config=config)
    runtime_cfg = (config or {}).get("runtime") if isinstance((config or {}).get("runtime"), dict) else {}
    adapter = get_runtime_adapter(runner, runtime_cfg)
    timeout_sec = int(runtime_cfg.get("proposer_timeout_sec", 180))
    res = adapter.run(
        prompt,
        resolved_wt,
        timeout_sec=timeout_sec,
        evolution_mode=True,
        allowed_root=resolved_wt,
    )
    return (res.raw_output or res.stderr or f"{runner} proposer finished")[:400]


def _agy_propose(
    worktree: Path,
    trace_ctx: dict[str, Any],
    history: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> str:
    """Invoke Antigravity CLI as Meta-Harness Proposer inside ``worktree``."""
    return _runtime_propose("agy", worktree, trace_ctx, history, config=config)


_TEST_DEF_RE = re.compile(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
_GO_TEST_DEF_RE = re.compile(r"^\s*func\s+(Test[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
_JS_TEST_DEF_RE = re.compile(r"""(?:^|\b)(?:it|test)\(\s*['"]([A-Za-z0-9_ .:-]{8,})['"]""", re.MULTILINE)
_CLASS_DEF_RE = re.compile(r"^\s*class\s+([A-Za-z0-9_]+)\s*(?:\(|:)", re.MULTILINE)
_ASSERT_LINE_RE = re.compile(
    r"(?:assert|assertEqual|assertAlmostEqual|assertRaises|expected|expect\(|assert\.|assert_eq!|t\.Error|t\.Fatal)"
)


def build_leakage_fingerprint(
    tasks: list[dict[str, Any]],
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Extract task IDs, verifier filenames, hidden identifiers, and assertion hashes."""
    task_ids: list[str] = []
    verifier_filenames: set[str] = {"test_solution.py", "run_bench.py", "tasks.json"}
    if manifest_path is not None:
        verifier_filenames.add(manifest_path.name)

    hidden_identifiers: set[str] = set()
    expected_line_hashes: dict[str, str] = {}
    raw_assertions: list[str] = []

    for task in tasks:
        tid = task.get("id")
        if isinstance(tid, str) and tid and tid not in task_ids:
            task_ids.append(tid)

        split = str(task.get("split", "train")).strip().lower()
        base_dir = Path(str(task.get("_base_dir") or (manifest_path.parent if manifest_path else ".")))
        sources_to_scan: list[str] = []

        verifier = task.get("verifier")
        ro_paths: set[str] = set()
        if isinstance(verifier, dict):
            for ro_rel in list(verifier.get("readonly_paths") or []) + list(verifier.get("test_files") or []):
                ro_str = str(ro_rel)
                ro_paths.add(ro_str)
                ro_paths.add(Path(ro_str).name)
                verifier_filenames.add(Path(ro_str).name)
            script_rel = verifier.get("script")
            if isinstance(script_rel, str) and script_rel.strip():
                verifier_filenames.add(Path(script_rel).name)
                if split in {"holdout", "ood"}:
                    sp = Path(script_rel)
                    if not sp.is_absolute():
                        sp = (base_dir / sp).resolve()
                    if sp.is_file():
                        sources_to_scan.append(sp.read_text(encoding="utf-8", errors="replace"))
            if split in {"holdout", "ood"}:
                patch_text = ""
                if isinstance(verifier.get("test_patch"), str) and verifier["test_patch"].strip():
                    patch_text = verifier["test_patch"]
                elif verifier.get("test_patch_file"):
                    pf = Path(str(verifier["test_patch_file"]))
                    if not pf.is_absolute():
                        pf = (base_dir / pf).resolve()
                    if pf.is_file():
                        patch_text = pf.read_text(encoding="utf-8", errors="replace")
                if patch_text:
                    added_patch_lines = [
                        ln[1:]
                        for ln in patch_text.splitlines()
                        if ln.startswith("+") and not ln.startswith("+++")
                    ]
                    if added_patch_lines:
                        sources_to_scan.append("\n".join(added_patch_lines))

        files = task.get("files")
        if isinstance(files, dict):
            for rel_name, content in files.items():
                base_name = Path(str(rel_name)).name
                is_test_file = base_name.startswith("test_") or base_name.endswith("_test.py")
                is_ro = str(rel_name) in ro_paths or base_name in ro_paths
                if is_test_file:
                    verifier_filenames.add(base_name)
                if (is_test_file or is_ro) and split in {"holdout", "ood"} and isinstance(content, str):
                    sources_to_scan.append(content)

        for src_text in sources_to_scan:
            for ident in _TEST_DEF_RE.findall(src_text):
                hidden_identifiers.add(ident)
            for ident in _GO_TEST_DEF_RE.findall(src_text):
                hidden_identifiers.add(ident)
            for ident in _JS_TEST_DEF_RE.findall(src_text):
                hidden_identifiers.add(ident)
            for cls_name in _CLASS_DEF_RE.findall(src_text):
                hidden_identifiers.add(cls_name)
            for line in src_text.splitlines():
                stripped = line.strip()
                if len(stripped) >= 16 and _ASSERT_LINE_RE.search(stripped):
                    h = hashlib.sha256(stripped.encode("utf-8")).hexdigest()[:16]
                    expected_line_hashes[h] = stripped
                    if stripped not in raw_assertions:
                        raw_assertions.append(stripped)

    return {
        "task_ids": task_ids,
        "verifier_filenames": sorted(verifier_filenames),
        "hidden_identifiers": sorted(hidden_identifiers),
        "expected_line_hashes": expected_line_hashes,
        "raw_assertions": raw_assertions,
    }


def validate_proposal_rrsi(
    worktree: Path,
    base_sha: str,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    benchmark_ids: list[str],
    leakage_profile: dict[str, Any] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Enforce RRSI Proposal-Side Regularization & Critic Leakage/Budget Screening."""
    annealed = compute_annealed_budget(config, history)
    max_files = int(annealed["max_files"])
    max_diff_lines = int(annealed["max_diff_lines"])
    surface = resolve_mutable_surface(worktree, config)

    status_out = _git(worktree, "status", "--porcelain", "--untracked-files=all").stdout.splitlines()
    changed_files: list[str] = []
    for line in status_out:
        if len(line) >= 4:
            changed_files.append(line[3:].strip().strip('"'))

    if not changed_files:
        return False, ["no_changes"], {"changed_files": [], "diff_lines": 0, "diff_hash": ""}

    # Include untracked files in diff inspection via git add -N (safe inside temp worktree).
    _git(worktree, "add", "-N", ".")
    numstat = _git(worktree, "diff", "--numstat", base_sha).stdout.splitlines()
    diff_lines = 0
    for row in numstat:
        parts = row.split("\t")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            diff_lines += int(parts[0]) + int(parts[1])

    diff_text = _git(worktree, "diff", base_sha).stdout
    diff_hash = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()[:16]

    reasons: list[str] = []
    for rel in changed_files:
        if surface.matches_protected(rel) is not None:
            reasons.append(f"protected_file_modified:{rel}")
        elif surface.entries and surface.matches_mutable(rel) is None:
            reasons.append(f"out_of_scope_file:{rel}")

    if len(changed_files) > max_files:
        reasons.append(f"edit_budget_files_exceeded:{len(changed_files)}>{max_files}")
    if diff_lines > max_diff_lines:
        reasons.append(f"edit_budget_lines_exceeded:{diff_lines}>{max_diff_lines}")

    seen_accepted_or_rejected = {h.get("diff_hash") for h in history if h.get("diff_hash")}
    if diff_hash in seen_accepted_or_rejected:
        reasons.append("duplicate_hypothesis")

    # Generalized Critic leakage screen: block benchmark IDs, verifier names,
    # hidden holdout/OOD identifiers, and verbatim expected assertion hashes.
    profile = leakage_profile or {}
    leak_tokens = list(
        dict.fromkeys(
            list(benchmark_ids)
            + list(profile.get("task_ids", []))
            + list(profile.get("verifier_filenames", ["test_solution.py", "run_bench.py", "tasks.json"]))
        )
    )
    added_lines = [
        ln[1:] for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++")
    ]
    added_blob = "\n".join(added_lines)
    for token in leak_tokens:
        if token in added_blob:
            reasons.append(f"critic_benchmark_leakage:{token}")

    for ident in profile.get("hidden_identifiers", []):
        if re.search(rf"\b{re.escape(str(ident))}\b", added_blob):
            reasons.append(f"critic_benchmark_leakage:hidden_id:{ident}")

    expected_hashes: dict[str, str] = dict(profile.get("expected_line_hashes") or {})
    raw_assertions: list[str] = list(profile.get("raw_assertions") or [])
    flagged_hashes: set[str] = set()
    for ln in added_lines:
        clean_ln = ln.strip().lstrip("-* ").strip()
        if not clean_ln:
            continue
        ln_hash = hashlib.sha256(clean_ln.encode("utf-8")).hexdigest()[:16]
        if ln_hash in expected_hashes and ln_hash not in flagged_hashes:
            flagged_hashes.add(ln_hash)
            reasons.append(f"critic_benchmark_leakage:expected_hash:{ln_hash}")
            continue
        for raw_assert in raw_assertions:
            if raw_assert in clean_ln:
                ah = hashlib.sha256(raw_assert.encode("utf-8")).hexdigest()[:16]
                if ah not in flagged_hashes:
                    flagged_hashes.add(ah)
                    reasons.append(f"critic_benchmark_leakage:expected_hash:{ah}")

    rules_bytes, surface_bytes, budget_violations = surface.compute_surface_bytes(worktree)
    rrsi_cfg = config.get("rrsi") if isinstance(config.get("rrsi"), dict) else {}
    rules_budget_mode = str(rrsi_cfg.get("rules_budget_mode") or "standard").strip().lower()
    raw_max_skills = rrsi_cfg.get("max_skills_bytes")
    max_skills_bytes = int(raw_max_skills) if raw_max_skills is not None else None

    base_role_bytes = surface.compute_git_ref_role_bytes(worktree, base_sha)
    cand_role_bytes = surface.compute_role_bytes(worktree)
    base_rules_bytes = int(base_role_bytes.get("rules", 0))
    cand_rules_bytes = int(cand_role_bytes.get("rules", 0))
    rules_bytes_delta = cand_rules_bytes - base_rules_bytes
    base_skills_bytes = int(base_role_bytes.get("skills", 0))
    cand_skills_bytes = int(cand_role_bytes.get("skills", 0))
    skills_bytes_delta = cand_skills_bytes - base_skills_bytes

    rules_entry_paths = {e.path for e in surface.entries if e.role == "rules"}

    if rules_budget_mode in {"ratchet", "progressive_disclosure"}:
        if cand_rules_bytes > base_rules_bytes:
            reasons.append(f"rules_ratchet_net_increase:{cand_rules_bytes}>{base_rules_bytes}")
        elif base_rules_bytes > surface.total_max_bytes and cand_rules_bytes >= base_rules_bytes:
            reasons.append(f"rules_ratchet_no_reduction:{cand_rules_bytes}>={base_rules_bytes}")

        exempt_total = (
            (base_rules_bytes > surface.total_max_bytes and cand_rules_bytes < base_rules_bytes)
            or (rules_budget_mode == "progressive_disclosure" and cand_rules_bytes < base_rules_bytes and skills_bytes_delta > 0)
        )
        exempt_oversized_rules_entry = base_rules_bytes > surface.total_max_bytes and cand_rules_bytes < base_rules_bytes
        for viol in budget_violations:
            if viol.startswith("rules_budget_exceeded:") and exempt_total:
                continue
            if viol.startswith("surface_entry_budget_exceeded:") and exempt_oversized_rules_entry:
                entry_path = viol.split(":", 2)[1]
                if entry_path in rules_entry_paths:
                    continue
            reasons.append(viol)

        if rules_budget_mode == "progressive_disclosure" and max_skills_bytes is not None and cand_skills_bytes > max_skills_bytes:
            reasons.append(f"skills_budget_exceeded:{cand_skills_bytes}>{max_skills_bytes}")
    else:
        reasons.extend(budget_violations)

    meta = {
        "changed_files": changed_files,
        "diff_lines": diff_lines,
        "diff_hash": diff_hash,
        "rules_bytes": rules_bytes,
        "base_rules_bytes": base_rules_bytes,
        "rules_only_bytes": cand_rules_bytes,
        "rules_bytes_delta": rules_bytes_delta,
        "skills_bytes": cand_skills_bytes,
        "skills_bytes_delta": skills_bytes_delta,
        "rules_budget_mode": rules_budget_mode,
        "surface_bytes": surface_bytes,
        "effective_budget": annealed,
        "pruning_recommended": cand_rules_bytes > int(surface.total_max_bytes * 0.75)
        or rules_bytes > int(surface.total_max_bytes * 0.75),
    }
    return len(reasons) == 0, reasons, meta


def _extract_metric(report: dict[str, Any], key: str) -> tuple[float, float, float, int]:
    """Return ``(mean, std, se, n)`` from multi-trial ``stats`` or scalar report fields."""
    n_default = max(1, int(report.get("n_trials", 1)))
    stats = report.get("stats")
    if isinstance(stats, dict) and isinstance(stats.get(key), dict):
        entry = stats[key]
        mean = float(entry.get("mean", report.get(key, 0.0)))
        std = float(entry.get("std", 0.0))
        n = max(1, int(entry.get("n", n_default)))
        se = float(entry["se"]) if "se" in entry else (std / math.sqrt(n))
        return mean, std, se, n
    return float(report.get(key, 0.0)), 0.0, 0.0, n_default


def _extract_effective_cost_metric(
    report: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, float, float, int]:
    """Extract ``(mean, std, se, n)`` for ``effective_token_cost`` with fallback computation."""
    stats = report.get("stats")
    if isinstance(stats, dict) and isinstance(stats.get("effective_token_cost"), dict):
        return _extract_metric(report, "effective_token_cost")
    if "effective_token_cost" in report and report["effective_token_cost"] is not None:
        return _extract_metric(report, "effective_token_cost")
    n_default = max(1, int(report.get("n_trials", 1)))
    computed = compute_effective_token_cost(report, config)
    return computed, 0.0, 0.0, n_default


def compute_selection_diagnostics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate candidate vs baseline using variance-aware Pareto dominance & noise floors."""
    rrsi = config.get("rrsi", {})
    min_gain = float(rrsi.get("min_pass_rate_gain", 0.05))
    min_tok_red = float(rrsi.get("min_token_reduction_ratio", 0.1))
    min_lat_red = float(rrsi.get("min_latency_reduction_ratio", 0.1))
    max_tok_overhead = float(rrsi.get("max_token_overhead_ratio", 0.15))
    max_lat_overhead = float(rrsi.get("max_latency_overhead_ratio", 0.25))
    noise_z = float(rrsi.get("noise_z", 1.96))
    eps = float(rrsi.get("pareto_tie_epsilon", 0.005))
    use_eff_cost = bool(rrsi.get("use_effective_token_cost", False))

    b_rate, b_rate_std, b_rate_se, b_n = _extract_metric(baseline, "pass_rate")
    c_rate, c_rate_std, c_rate_se, c_n = _extract_metric(candidate, "pass_rate")
    b_raw_tok, b_raw_tok_std, b_raw_tok_se, _ = _extract_metric(baseline, "total_tokens")
    c_raw_tok, c_raw_tok_std, c_raw_tok_se, _ = _extract_metric(candidate, "total_tokens")
    b_eff_cost, b_eff_std, b_eff_se, _ = _extract_effective_cost_metric(baseline, config)
    c_eff_cost, c_eff_std, c_eff_se, _ = _extract_effective_cost_metric(candidate, config)
    b_lat, b_lat_std, b_lat_se, _ = _extract_metric(baseline, "duration_sec")
    c_lat, c_lat_std, c_lat_se, _ = _extract_metric(candidate, "duration_sec")

    if use_eff_cost:
        b_tok, b_tok_std, b_tok_se = b_eff_cost, b_eff_std, b_eff_se
        c_tok, c_tok_std, c_tok_se = c_eff_cost, c_eff_std, c_eff_se
    else:
        b_tok, b_tok_std, b_tok_se = b_raw_tok, b_raw_tok_std, b_raw_tok_se
        c_tok, c_tok_std, c_tok_se = c_raw_tok, c_raw_tok_std, c_raw_tok_se

    rate_gain = c_rate - b_rate
    tok_reduction = ((b_tok - c_tok) / b_tok) if b_tok > 0 else 0.0
    eff_cost_reduction = ((b_eff_cost - c_eff_cost) / b_eff_cost) if b_eff_cost > 0 else 0.0
    lat_reduction = ((b_lat - c_lat) / b_lat) if b_lat > 0 else 0.0

    se_rate = math.hypot(b_rate_se, c_rate_se)
    se_tok = math.hypot(b_tok_se, c_tok_se)
    se_lat = math.hypot(b_lat_se, c_lat_se)

    rate_noise_floor = noise_z * se_rate
    eff_min_rate_gain = max(min_gain, rate_noise_floor)
    tok_noise_floor = noise_z * se_tok
    eff_min_tok_delta = max(b_tok * min_tok_red, tok_noise_floor)
    lat_noise_floor = noise_z * se_lat
    eff_min_lat_delta = max(b_lat * min_lat_red, lat_noise_floor)

    tok_cap = int(b_tok * (1.0 + max_tok_overhead)) if b_tok > 0 else 0
    if rrsi.get("enforce_anti_goodhart", True) and candidate.get("anti_goodhart_passed") is False:
        accepted = False
        reasons = ["anti_goodhart_verifier_tampered"]
    elif c_rate < b_rate - eps:
        accepted = False
        reasons = [f"pass_rate_regressed:{c_rate:.3f}<{b_rate:.3f}"]
    elif b_tok > 0 and c_tok > tok_cap:
        accepted = False
        reasons = [f"token_overhead_too_high:{int(round(c_tok))}>{tok_cap}"]
    elif rate_gain >= eff_min_rate_gain:
        accepted = True
        reasons = [f"pass_rate_improved:+{rate_gain:.3f}"]
    elif (
        c_rate >= b_rate - eps
        and b_tok > 0
        and (b_tok - c_tok) >= eff_min_tok_delta
        and (b_lat <= 0 or c_lat <= b_lat * (1.0 + max_lat_overhead))
    ):
        accepted = True
        reasons = [f"token_efficiency_improved:-{tok_reduction * 100:.1f}%"]
    elif (
        c_rate >= b_rate - eps
        and b_lat > 0
        and (b_lat - c_lat) >= eff_min_lat_delta
        and (b_tok <= 0 or c_tok <= b_tok + tok_noise_floor)
    ):
        accepted = True
        reasons = [f"latency_efficiency_improved:-{lat_reduction * 100:.1f}%"]
    else:
        accepted = False
        raw_met_fixed = (
            (rate_gain >= min_gain)
            or (c_rate >= b_rate - eps and b_tok > 0 and tok_reduction >= min_tok_red)
            or (c_rate >= b_rate - eps and b_lat > 0 and lat_reduction >= min_lat_red)
        )
        if raw_met_fixed:
            reasons = [
                f"noisy_variance_exceeds_gain:rate_delta={rate_gain:.3f},"
                f"token_reduction={tok_reduction:.3f},latency_reduction={lat_reduction:.3f}"
            ]
        else:
            reasons = [
                f"insufficient_gain:rate_delta={rate_gain:.3f},token_reduction={tok_reduction:.3f}"
            ]

    breakdown_keys = ("input_tokens", "cache_read_tokens", "output_tokens", "tool_calls", "steps")
    base_breakdown = {k: int(round(float(baseline.get(k) or 0))) for k in breakdown_keys}
    cand_breakdown = {k: int(round(float(candidate.get(k) or 0))) for k in breakdown_keys}
    delta_breakdown = {k: cand_breakdown[k] - base_breakdown[k] for k in breakdown_keys}

    return {
        "accepted": accepted,
        "reasons": reasons,
        "baseline_n": b_n,
        "candidate_n": c_n,
        "rate_gain": round(rate_gain, 6),
        "tok_reduction": round(tok_reduction, 6),
        "lat_reduction": round(lat_reduction, 6),
        "use_effective_token_cost": use_eff_cost,
        "effective_token_cost": {
            "baseline": round(b_eff_cost, 4),
            "candidate": round(c_eff_cost, 4),
            "delta": round(c_eff_cost - b_eff_cost, 4),
            "reduction": round(eff_cost_reduction, 6),
        },
        "token_breakdown": {
            "baseline": base_breakdown,
            "candidate": cand_breakdown,
            "delta": delta_breakdown,
        },
        "baseline_std": {"pass_rate": round(b_rate_std, 6), "total_tokens": round(b_tok_std, 6), "duration_sec": round(b_lat_std, 6)},
        "candidate_std": {"pass_rate": round(c_rate_std, 6), "total_tokens": round(c_tok_std, 6), "duration_sec": round(c_lat_std, 6)},
        "se_rate": round(se_rate, 6),
        "se_tok": round(se_tok, 6),
        "se_lat": round(se_lat, 6),
        "rate_noise_floor": round(rate_noise_floor, 6),
        "tok_noise_floor": round(tok_noise_floor, 6),
        "lat_noise_floor": round(lat_noise_floor, 6),
        "eff_min_rate_gain": round(eff_min_rate_gain, 6),
        "eff_min_tok_delta": round(eff_min_tok_delta, 6),
        "eff_min_lat_delta": round(eff_min_lat_delta, 6),
    }


def should_expand_adaptive_trials(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, str]:
    """Decide whether initial-trial candidate merits expansion to ``max_trials``."""
    rrsi = config.get("rrsi") if isinstance(config.get("rrsi"), dict) else {}
    adaptive = rrsi.get("adaptive_trials") if isinstance(rrsi.get("adaptive_trials"), dict) else {}
    early_drop = float(adaptive.get("early_reject_pass_rate_drop", rrsi.get("pareto_tie_epsilon", 0.005)))
    early_overhead = float(
        adaptive.get("early_reject_token_overhead_ratio", rrsi.get("max_token_overhead_ratio", 0.15))
    )
    eps = float(rrsi.get("pareto_tie_epsilon", 0.005))
    use_eff = bool(rrsi.get("use_effective_token_cost", False))

    if rrsi.get("enforce_anti_goodhart", True) and candidate.get("anti_goodhart_passed") is False:
        return False, "early_reject_anti_goodhart"

    b_rate = float(baseline.get("pass_rate", 0.0))
    c_rate = float(candidate.get("pass_rate", 0.0))
    if c_rate < b_rate - early_drop:
        return False, f"early_reject_pass_rate_regressed:{c_rate:.3f}<{b_rate:.3f}"

    b_cost = (
        compute_effective_token_cost(baseline, config)
        if use_eff
        else float(baseline.get("total_tokens", 0.0))
    )
    c_cost = (
        compute_effective_token_cost(candidate, config)
        if use_eff
        else float(candidate.get("total_tokens", 0.0))
    )
    if b_cost > 0 and c_cost > b_cost * (1.0 + early_overhead):
        return False, f"early_reject_token_overhead_too_high:{int(round(c_cost))}>{int(b_cost * (1.0 + early_overhead))}"

    b_lat = float(baseline.get("duration_sec", 0.0))
    c_lat = float(candidate.get("duration_sec", 0.0))
    if (
        c_rate <= b_rate + eps
        and c_rate <= b_rate
        and (b_cost <= 0 or c_cost >= b_cost)
        and (b_lat <= 0 or c_lat >= b_lat)
    ):
        return False, "early_reject_no_improvement_on_initial_trial"

    return True, "expand_for_statistical_confirmation"


def merge_benchmark_reports(
    reports: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge multiple benchmark reports (each with 1+ trials) into one aggregated report."""
    if not reports:
        raise ValueError("Cannot merge empty report list")

    all_trials: list[dict[str, Any]] = []
    for rep in reports:
        rep_trials = rep.get("trials")
        if isinstance(rep_trials, list) and rep_trials:
            for t in rep_trials:
                trial_copy = dict(t)
                trial_copy["trial"] = len(all_trials) + 1
                all_trials.append(trial_copy)
        else:
            single = {
                "trial": len(all_trials) + 1,
                "total_tasks": int(rep.get("total_tasks", 0)),
                "passed_tasks": int(rep.get("passed_tasks", 0)),
                "pass_rate": float(rep.get("pass_rate", 0.0)),
                "total_tokens": float(rep.get("total_tokens", 0)),
                "effective_token_cost": float(
                    rep.get("effective_token_cost", compute_effective_token_cost(rep, config))
                ),
                "input_tokens": float(rep.get("input_tokens", 0)),
                "cache_read_tokens": float(rep.get("cache_read_tokens", 0)),
                "output_tokens": float(rep.get("output_tokens", 0)),
                "tool_calls": float(rep.get("tool_calls", 0)),
                "steps": float(rep.get("steps", 0)),
                "duration_sec": float(rep.get("duration_sec", 0.0)),
                "anti_goodhart_passed": rep.get("anti_goodhart_passed", True),
                "tampered_tasks": list(rep.get("tampered_tasks") or []),
                "behavior_passed": rep.get("behavior_passed", True),
                "behavior_failed_tasks": list(rep.get("behavior_failed_tasks") or []),
                "tasks": list(rep.get("tasks") or []),
            }
            all_trials.append(single)

    stats = aggregate_trial_metrics(all_trials)
    first = reports[0]
    last = reports[-1]
    all_tampered = sorted({tid for t in all_trials for tid in (t.get("tampered_tasks") or [])})
    all_beh_failed = sorted({tid for t in all_trials for tid in (t.get("behavior_failed_tasks") or [])})
    ag_passed = len(all_tampered) == 0 and all(t.get("anti_goodhart_passed", True) is not False for t in all_trials)
    beh_passed = len(all_beh_failed) == 0 and all(t.get("behavior_passed", True) is not False for t in all_trials)
    mean_passed = sum(float(t.get("passed_tasks", 0)) for t in all_trials) / len(all_trials)

    merged = dict(last)
    merged.update(
        {
            "suite": first.get("suite", last.get("suite", "mini-bench")),
            "runner": first.get("runner", last.get("runner", "mock")),
            "split": first.get("split", last.get("split", "all")),
            "n_trials": len(all_trials),
            "total_tasks": int(last.get("total_tasks", first.get("total_tasks", 0))),
            "passed_tasks": int(round(mean_passed)),
            "pass_rate": round(stats["pass_rate"]["mean"], 4),
            "total_tokens": int(round(stats["total_tokens"]["mean"])),
            "effective_token_cost": round(stats["effective_token_cost"]["mean"], 4),
            "input_tokens": int(round(stats["input_tokens"]["mean"])),
            "cache_read_tokens": int(round(stats["cache_read_tokens"]["mean"])),
            "output_tokens": int(round(stats["output_tokens"]["mean"])),
            "tool_calls": int(round(stats["tool_calls"]["mean"])),
            "steps": int(round(stats["steps"]["mean"])),
            "duration_sec": round(stats["duration_sec"]["mean"], 3),
            "anti_goodhart_passed": ag_passed,
            "tampered_tasks": all_tampered,
            "behavior_passed": beh_passed,
            "behavior_failed_tasks": all_beh_failed,
            "stats": stats,
            "trials": all_trials,
            "tasks": all_trials[-1].get("tasks", last.get("tasks", [])),
        }
    )
    return merged


def evaluate_selection_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply RRSI Noise-Adjusted Floor + SoL-Pi Pareto Selection Gate."""
    diag = compute_selection_diagnostics(baseline, candidate, config)
    return bool(diag["accepted"]), list(diag["reasons"])


def _run_bench_in_repo(
    repo_path: Path,
    runner: str,
    trace_export_dir: Path | None = None,
    n_trials: int = 1,
    sandbox: SandboxAdapter | None = None,
    manifest_path: Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    local_bench = repo_path / "benchmarks" / "run_bench.py"
    bench_script = "benchmarks/run_bench.py" if local_bench.is_file() else str(BENCHMARKS_DIR / "run_bench.py")
    cmd = [
        "python3",
        bench_script,
        "--repo",
        ".",
        "--runner",
        runner,
        "--trials",
        str(max(1, n_trials)),
    ]
    if manifest_path is not None:
        try:
            manifest_arg = manifest_path.resolve().relative_to(repo_path.resolve()).as_posix()
        except ValueError:
            manifest_arg = str(manifest_path.resolve())
        cmd.extend(["--manifest", manifest_arg])
    if split is not None:
        cmd.extend(["--split", split])
    if trace_export_dir is not None:
        try:
            export_arg = trace_export_dir.resolve().relative_to(repo_path.resolve()).as_posix()
        except ValueError:
            export_arg = str(trace_export_dir)
        cmd.extend(["--trace-export-dir", export_arg])
    if sandbox is not None:
        res = sandbox.exec(cmd, cwd=repo_path, timeout=300)
        if res.returncode != 0:
            raise subprocess.CalledProcessError(res.returncode, res.command, res.stdout, res.stderr)
        return json.loads(res.stdout)
    proc = subprocess.run(
        cmd,
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return json.loads(proc.stdout)


def _maybe_handle_pr(
    repo: Path,
    record: dict[str, Any],
    *,
    pr_mode: str,
    pr_base: str | None,
    diff_text: str | None = None,
) -> None:
    if pr_mode == "none":
        return
    effective_base = (pr_base or "").strip() or detect_default_branch(repo)
    if pr_mode == "dry-run":
        pr_res = create_review_pr(repo, record, base_branch=effective_base, dry_run=True, diff_text=diff_text)
        record["pr_result"] = pr_res
        record["pr_report_path"] = pr_res["report_path"]
    elif pr_mode == "create":
        if not record.get("accepted"):
            record["pr_result"] = {
                "dry_run": False,
                "created": False,
                "skipped": True,
                "reason": "candidate_not_accepted",
            }
            return
        pr_res = create_review_pr(repo, record, base_branch=effective_base, dry_run=False, diff_text=diff_text)
        record["pr_result"] = pr_res
        record["pr_report_path"] = pr_res["report_path"]
    else:
        raise ValueError(f"Unsupported pr_mode: {pr_mode!r}")


def run_evolution_cycle(
    repo_root: Path,
    branch: str = "rsi/evolved",
    runner: str = "mock",
    mock_bad_proposal: str | None = None,
    manifest_path: Path | None = None,
    mode: str = "auto",
    prune: bool = False,
    parent_strategy: str = "head",
    parent_node_id: str | None = None,
    seed: int | None = None,
    proposer: Proposer | str | dict[str, Any] | None = None,
    evaluator: Evaluator | str | dict[str, Any] | None = None,
    critic: Critic | str | dict[str, Any] | None = None,
    pr_mode: str = "none",
    pr_base: str | None = None,
    adaptive_trials: bool | None = None,
) -> dict[str, Any]:
    """Run one full Meta-Harness + RRSI + SoL-Pi evolution cycle."""
    repo = repo_root.resolve()
    config = load_rsi_config(repo)
    plugin_cfg = config.get("plugins") if isinstance(config.get("plugins"), dict) else {}
    active_proposer = load_plugin(
        proposer if proposer is not None else plugin_cfg.get("proposer"),
        Proposer,
        DefaultProposer,
        entry_point_group="rsi_harness.proposers",
        search_paths=[repo],
    )
    active_evaluator = load_plugin(
        evaluator if evaluator is not None else plugin_cfg.get("evaluator"),
        Evaluator,
        BenchmarkEvaluator,
        entry_point_group="rsi_harness.evaluators",
        search_paths=[repo],
    )
    active_critic = load_plugin(
        critic if critic is not None else plugin_cfg.get("critic"),
        Critic,
        RrsiCritic,
        entry_point_group="rsi_harness.critics",
        search_paths=[repo],
    )
    plugins_used = {
        "proposer": f"{type(active_proposer).__module__}.{type(active_proposer).__qualname__}",
        "evaluator": f"{type(active_evaluator).__module__}.{type(active_evaluator).__qualname__}",
        "critic": f"{type(active_critic).__module__}.{type(active_critic).__qualname__}",
    }
    rrsi_cfg = config.get("rrsi", {})
    n_trials = max(1, int(rrsi_cfg.get("n_trials", 1)))
    adaptive_cfg = rrsi_cfg.get("adaptive_trials") if isinstance(rrsi_cfg.get("adaptive_trials"), dict) else {}
    adaptive_enabled = (
        bool(adaptive_cfg.get("enabled", False))
        if adaptive_trials is None
        else bool(adaptive_trials)
    )
    if adaptive_enabled:
        initial_trials = max(1, int(adaptive_cfg.get("initial_trials", 1)))
        max_trials = max(initial_trials, int(adaptive_cfg.get("max_trials", max(n_trials, 3))))
        eval_n_trials = initial_trials
    else:
        initial_trials = n_trials
        max_trials = n_trials
        eval_n_trials = n_trials
    max_ood_reg = float(rrsi_cfg.get("max_ood_regression", 0.05))
    history_path = repo / ".rsi" / "evolution_history.jsonl"
    history = load_jsonl(history_path)
    trace_ctx = _collect_trace_context(repo)

    default_manifest = repo / "benchmarks" / "tasks.json"
    use_trace_mined = mode == "trace-mined" or (
        mode == "auto" and manifest_path is None and not default_manifest.is_file()
    )
    evolution_mode = "trace-mined" if use_trace_mined else "benchmark"
    mined_clusters: list[dict[str, Any]] = []

    if use_trace_mined:
        clusters = mine_weaknesses(repo)
        if not clusters:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": uuid.uuid4().hex[:8],
                "branch": branch,
                "evolution_mode": evolution_mode,
                "accepted": False,
                "stage": "weakness_miner",
                "reasons": ["no_weakness_clusters_found"],
                "mined_clusters": [],
            }
            _maybe_handle_pr(repo, record, pr_mode=pr_mode, pr_base=pr_base, diff_text=None)
            append_jsonl(history_path, record)
            return record
        mined_clusters = [asdict(c) for c in clusters[:5]]
        effective_manifest = (repo / ".rsi" / "mined_tasks.json").resolve()
        synthesize_regression_suite(repo, clusters, output_path=effective_manifest)
    else:
        effective_manifest = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else default_manifest.resolve()
        )
    suite = load_benchmark_suite(effective_manifest)
    all_tasks = suite.get("tasks", [])
    leakage_profile = build_leakage_fingerprint(all_tasks, effective_manifest)
    benchmark_ids = list(leakage_profile["task_ids"])

    has_holdout = any(str(t.get("split", "train")).strip().lower() == "holdout" for t in all_tasks)
    has_ood = any(str(t.get("split", "train")).strip().lower() == "ood" for t in all_tasks)
    split_mode = (
        "train_holdout_ood"
        if (has_holdout and has_ood)
        else ("train_holdout" if has_holdout else "single")
    )

    head_sha = _ensure_branch(repo, branch)
    archive = LineageArchive.load(repo)
    repo_head_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    repo_head_branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip() or "main"
    root_node = archive.ensure_root(commit_sha=repo_head_sha, branch=repo_head_branch)

    if parent_node_id is not None:
        selected_parent = archive.get_node(parent_node_id)
        if selected_parent is None:
            raise ValueError(f"Unknown parent_node_id: {parent_node_id}")
        parent_node = selected_parent
        base_sha = parent_node.commit_sha or head_sha
    elif parent_strategy != "head":
        parent_node = archive.sample_parent(strategy=parent_strategy, seed=seed) or root_node
        base_sha = parent_node.commit_sha or head_sha
    else:
        matching = [n for n in archive.nodes.values() if n.commit_sha == head_sha]
        parent_node = matching[-1] if matching else root_node
        base_sha = head_sha

    run_id = uuid.uuid4().hex[:8]
    sandbox = get_sandbox_adapter(config)
    worktree_dir = sandbox.prepare(repo, base_sha, run_id)
    try:
        _seed_worktree_traces(repo, worktree_dir)
        wt_traces_dir = worktree_dir / ".rsi" / "traces"
        train_baseline: dict[str, Any] | None = None
        ood_baseline: dict[str, Any] | None = None
        ood_candidate: dict[str, Any] | None = None

        if has_holdout:
            # Proposer sees ONLY train traces; holdout & OOD stay completely isolated.
            train_baseline = active_evaluator.evaluate(
                worktree_dir,
                runner=runner,
                config=config,
                trace_export_dir=wt_traces_dir,
                n_trials=eval_n_trials,
                sandbox=sandbox,
                manifest_path=effective_manifest,
                split="train",
            )
            if wt_traces_dir.is_dir():
                shutil.copytree(wt_traces_dir, repo / ".rsi" / "traces", dirs_exist_ok=True)
            baseline_report = active_evaluator.evaluate(
                worktree_dir,
                runner=runner,
                config=config,
                trace_export_dir=None,
                n_trials=eval_n_trials,
                sandbox=sandbox,
                manifest_path=effective_manifest,
                split="holdout",
            )
            if has_ood:
                ood_baseline = active_evaluator.evaluate(
                    worktree_dir,
                    runner=runner,
                    config=config,
                    trace_export_dir=None,
                    n_trials=eval_n_trials,
                    sandbox=sandbox,
                    manifest_path=effective_manifest,
                    split="ood",
                )
        else:
            baseline_report = active_evaluator.evaluate(
                worktree_dir,
                runner=runner,
                config=config,
                trace_export_dir=wt_traces_dir,
                n_trials=eval_n_trials,
                sandbox=sandbox,
                manifest_path=effective_manifest,
            )
            if wt_traces_dir.is_dir():
                shutil.copytree(wt_traces_dir, repo / ".rsi" / "traces", dirs_exist_ok=True)

        if root_node.metadata.get("metrics_initialized") is not True and parent_node.commit_sha == root_node.commit_sha:
            root_node.pass_rate = float(baseline_report.get("pass_rate", 0.0))
            root_node.total_tokens = float(baseline_report.get("total_tokens", 0.0))
            root_node.latency_ms = float(
                baseline_report.get("latency_ms", float(baseline_report.get("duration_sec", 0.0)) * 1000.0)
            )
            root_node.metadata["metrics_initialized"] = True
            archive.recompute_novelty()

        trace_ctx = _collect_trace_context(worktree_dir)

        proposer_note = active_proposer.propose(
            ProposalContext(
                repo_root=repo,
                worktree_dir=worktree_dir,
                runner=runner,
                trace_context=trace_ctx,
                history=history,
                config=config,
                mock_bad_proposal=mock_bad_proposal,
                parent_node_id=parent_node.node_id,
            )
        )

        pruning_analysis = analyze_pruning_candidates(worktree_dir, config, history)
        pruned_files = apply_pruning_pass(worktree_dir, pruning_analysis) if prune else []

        prop_ok, prop_reasons, prop_meta = active_critic.check(
            worktree_dir,
            base_sha=base_sha,
            config=config,
            history=history,
            benchmark_ids=benchmark_ids,
            leakage_profile=leakage_profile,
        )
        prop_meta["pruning_analysis"] = pruning_analysis
        prop_meta["pruned_files"] = pruned_files
        store = TraceStore(repo)
        diff_text = _git(worktree_dir, "diff", base_sha).stdout

        if not prop_ok:
            archive.save()
            pareto_ids = [n.node_id for n in archive.pareto_frontier()]
            gen_rec = store.record_generation(
                run_id=run_id,
                branch=branch,
                parent_sha=base_sha,
                commit_sha=None,
                accepted=False,
                stage="proposal_critic",
                reasons=prop_reasons,
                diff_text=diff_text,
                changed_files=list(prop_meta.get("changed_files", [])),
                config=config,
                baseline_report=baseline_report,
                candidate_report=None,
                traces_src_dir=wt_traces_dir,
                worktree=worktree_dir,
                proposer_note=proposer_note,
            )
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "branch": branch,
                "evolution_mode": evolution_mode,
                "mined_clusters": mined_clusters,
                "split_mode": split_mode,
                "n_trials": n_trials,
                "parent_strategy": parent_strategy,
                "parent_node_id": parent_node.node_id,
                "lineage_node_id": None,
                "node_branch": None,
                "pareto_frontier_ids": pareto_ids,
                "plugins_used": plugins_used,
                "accepted": False,
                "stage": "proposal_critic",
                "reasons": prop_reasons,
                "proposer_note": proposer_note,
                "baseline": baseline_report,
                "train_baseline": train_baseline,
                "ood_baseline": ood_baseline,
                "candidate": None,
                "ood_candidate": None,
                "store_generation": gen_rec["generation"],
                "store_dir": gen_rec["snapshot_dir"],
                **prop_meta,
            }
            _maybe_handle_pr(repo, record, pr_mode=pr_mode, pr_base=pr_base, diff_text=diff_text)
            append_jsonl(history_path, record)
            return record

        cand_traces_dir = worktree_dir / ".rsi" / "cand_traces"
        candidate_report = active_evaluator.evaluate(
            worktree_dir,
            runner=runner,
            config=config,
            trace_export_dir=cand_traces_dir,
            n_trials=eval_n_trials,
            sandbox=sandbox,
            manifest_path=effective_manifest,
            split=("holdout" if has_holdout else None),
        )
        adaptive_expanded = False
        adaptive_reason = "fixed_trials"
        if adaptive_enabled and max_trials > eval_n_trials:
            should_expand, adaptive_reason = should_expand_adaptive_trials(
                baseline_report,
                candidate_report,
                config,
            )
            if should_expand:
                adaptive_expanded = True
                extra_n = max_trials - eval_n_trials
                extra_cand = active_evaluator.evaluate(
                    worktree_dir,
                    runner=runner,
                    config=config,
                    trace_export_dir=cand_traces_dir,
                    n_trials=extra_n,
                    sandbox=sandbox,
                    manifest_path=effective_manifest,
                    split=("holdout" if has_holdout else None),
                )
                candidate_report = merge_benchmark_reports(
                    [candidate_report, extra_cand],
                    config=config,
                )
                base_exp_sb = get_sandbox_adapter(config)
                base_exp_wt = base_exp_sb.prepare(repo, base_sha, f"{run_id}-bexp")
                try:
                    _seed_worktree_traces(repo, base_exp_wt)
                    extra_base = active_evaluator.evaluate(
                        base_exp_wt,
                        runner=runner,
                        config=config,
                        trace_export_dir=None,
                        n_trials=extra_n,
                        sandbox=base_exp_sb,
                        manifest_path=effective_manifest,
                        split=("holdout" if has_holdout else None),
                    )
                    baseline_report = merge_benchmark_reports(
                        [baseline_report, extra_base],
                        config=config,
                    )
                finally:
                    base_exp_sb.cleanup()

        effective_n_trials = int(candidate_report.get("n_trials", eval_n_trials))
        sel_diag = compute_selection_diagnostics(baseline_report, candidate_report, config)
        adaptive_info = {
            "enabled": adaptive_enabled,
            "initial_trials": initial_trials,
            "max_trials": max_trials,
            "expanded": adaptive_expanded,
            "reason": adaptive_reason,
            "baseline_trials": int(baseline_report.get("n_trials", eval_n_trials)),
            "candidate_trials": effective_n_trials,
        }
        sel_diag["adaptive_trials"] = adaptive_info
        sel_ok = bool(sel_diag["accepted"])
        sel_reasons = list(sel_diag["reasons"])

        if sel_ok and has_ood and ood_baseline is not None:
            ood_candidate = active_evaluator.evaluate(
                worktree_dir,
                runner=runner,
                config=config,
                trace_export_dir=None,
                n_trials=effective_n_trials,
                sandbox=sandbox,
                manifest_path=effective_manifest,
                split="ood",
            )
            if rrsi_cfg.get("enforce_anti_goodhart", True) and ood_candidate.get("anti_goodhart_passed") is False:
                sel_ok = False
                sel_reasons = ["anti_goodhart_verifier_tampered"]
                sel_diag["accepted"] = False
                sel_diag["reasons"] = list(sel_reasons)
            elif float(ood_candidate.get("pass_rate", 0.0)) < float(ood_baseline.get("pass_rate", 0.0)) - max_ood_reg:
                sel_ok = False
                sel_reasons = [
                    f"ood_generalization_regressed:{float(ood_candidate['pass_rate']):.3f}"
                    f"<{float(ood_baseline['pass_rate']):.3f}"
                ]
                sel_diag["accepted"] = False
                sel_diag["reasons"] = list(sel_reasons)

        commit_sha = None
        lineage_node_id: str | None = None
        node_branch: str | None = None
        if sel_ok:
            _git(worktree_dir, "add", "-A")
            staged_diff = _git(worktree_dir, "diff", "--cached", base_sha).stdout
            if staged_diff:
                diff_text = staged_diff
            msg = (
                f"rsi(evolve): pass_rate {baseline_report['pass_rate']}->"
                f"{candidate_report['pass_rate']}, tokens {baseline_report['total_tokens']}->"
                f"{candidate_report['total_tokens']}"
            )
            _git(worktree_dir, "commit", "-m", msg)
            commit_sha = _git(worktree_dir, "rev-parse", "HEAD").stdout.strip()
            node_branch = f"rsi/node-{run_id}"
            _git(repo, "branch", "-f", node_branch, commit_sha)
            lineage_node_id = f"node-{run_id}"
            changed_list = sorted(str(p) for p in prop_meta.get("changed_files", []))
            diff_hash = hashlib.sha256(
                (diff_text + "\n" + "\n".join(changed_list)).encode("utf-8")
            ).hexdigest()[:16]
            cand_latency_ms = float(
                candidate_report.get("latency_ms", float(candidate_report.get("duration_sec", 0.0)) * 1000.0)
            )
            archive.add_node(
                node_id=lineage_node_id,
                parent_id=parent_node.node_id,
                commit_sha=commit_sha,
                branch=node_branch,
                pass_rate=float(candidate_report.get("pass_rate", 0.0)),
                total_tokens=float(candidate_report.get("total_tokens", 0.0)),
                latency_ms=cand_latency_ms,
                diff_hash=diff_hash,
                changed_files=changed_list,
                metadata={
                    "run_id": run_id,
                    "parent_strategy": parent_strategy,
                    "proposer_note": proposer_note,
                },
            )
            pareto_ids = [n.node_id for n in archive.pareto_frontier()]
            if parent_strategy == "head" or lineage_node_id in pareto_ids:
                _git(repo, "branch", "-f", branch, commit_sha)
        else:
            pareto_ids = [n.node_id for n in archive.pareto_frontier()]

        archive.save()

        gen_rec = store.record_generation(
            run_id=run_id,
            branch=node_branch or branch,
            parent_sha=base_sha,
            commit_sha=commit_sha,
            accepted=sel_ok,
            stage="selection_gate",
            reasons=sel_reasons,
            diff_text=diff_text,
            changed_files=list(prop_meta.get("changed_files", [])),
            config=config,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            traces_src_dir=wt_traces_dir,
            candidate_traces_dir=cand_traces_dir,
            worktree=worktree_dir,
            proposer_note=proposer_note,
        )

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "branch": branch,
            "evolution_mode": evolution_mode,
            "mined_clusters": mined_clusters,
            "split_mode": split_mode,
            "n_trials": effective_n_trials,
            "adaptive_trials": adaptive_info,
            "parent_strategy": parent_strategy,
            "parent_node_id": parent_node.node_id,
            "lineage_node_id": lineage_node_id,
            "node_branch": node_branch,
            "pareto_frontier_ids": pareto_ids,
            "plugins_used": plugins_used,
            "accepted": sel_ok,
            "commit_sha": commit_sha,
            "stage": "selection_gate",
            "reasons": sel_reasons,
            "selection_diagnostics": sel_diag,
            "proposer_note": proposer_note,
            "baseline": baseline_report,
            "train_baseline": train_baseline,
            "ood_baseline": ood_baseline,
            "candidate": candidate_report,
            "ood_candidate": ood_candidate,
            "store_generation": gen_rec["generation"],
            "store_dir": gen_rec["snapshot_dir"],
            **prop_meta,
        }
        _maybe_handle_pr(repo, record, pr_mode=pr_mode, pr_base=pr_base, diff_text=diff_text)
        append_jsonl(history_path, record)
        return record
    finally:
        sandbox.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one RSI-Harness outer evolution cycle.")
    parser.add_argument("--repo", default=".", help="Target repository path")
    parser.add_argument("--branch", default="rsi/evolved", help="Branch to auto-apply winning edits")
    parser.add_argument("--runner", default="mock", choices=["mock", *list_runtime_adapters()])
    parser.add_argument("--manifest", default=None, help="Optional benchmark manifest path or Harbor directory")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "benchmark", "trace-mined"],
        help="Evolution mode: auto (falls back to trace-mined if tasks.json absent), benchmark, or trace-mined",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Run structural pruning pass to remove redundant rules inside candidate worktree",
    )
    parser.add_argument(
        "--parent-strategy",
        default="head",
        choices=["head", "dgm", "latest", "best"],
        help="Parent selection strategy from lineage archive (head preserves branch-head evolution)",
    )
    parser.add_argument(
        "--parent-node-id",
        default=None,
        help="Explicit parent lineage node ID to branch from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional PRNG seed for deterministic DGM parent sampling",
    )
    parser.add_argument(
        "--mock-bad-proposal",
        default=None,
        choices=["leak", "budget", "protected"],
        help="Optional negative test mode for mock proposer",
    )
    parser.add_argument(
        "--pr-dry-run",
        action="store_true",
        help="Generate evidence-backed PR report under .rsi/pr_reports/ without pushing",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Push accepted evolution branch and open a PR via gh CLI",
    )
    parser.add_argument(
        "--pr-base",
        default=None,
        help="Base branch for PR report/creation (auto-detected if omitted)",
    )
    parser.add_argument(
        "--adaptive-trials",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable adaptive multi-trial evaluation expansion",
    )
    args = parser.parse_args()

    pr_mode = "create" if args.create_pr else ("dry-run" if args.pr_dry_run else "none")
    result = run_evolution_cycle(
        Path(args.repo),
        branch=args.branch,
        runner=args.runner,
        mock_bad_proposal=args.mock_bad_proposal,
        manifest_path=Path(args.manifest) if args.manifest else None,
        mode=args.mode,
        prune=args.prune,
        parent_strategy=args.parent_strategy,
        parent_node_id=args.parent_node_id,
        seed=args.seed,
        pr_mode=pr_mode,
        pr_base=args.pr_base,
        adaptive_trials=args.adaptive_trials,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
