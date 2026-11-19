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
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import append_jsonl, load_jsonl, load_rsi_config  # noqa: E402


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
    return {
        "summaries": summaries,
        "error_samples": error_samples[-20:],
        "history_count": len(load_jsonl(repo / ".rsi" / "evolution_history.jsonl")),
    }


def _mock_propose(
    worktree: Path,
    trace_ctx: dict[str, Any],
    history: list[dict[str, Any]],
    override_mode: str | None = None,
) -> str:
    """Deterministic Proposer for offline tests and dry-runs."""
    rules_path = worktree / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"
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
        hooks_path.write_text("{}\n", encoding="utf-8")
        return "Injected protected file edit for testing guard"

    seen_hashes = {h.get("diff_hash") for h in history}
    addition = (
        "\n- 文字列パースやログ解析では空行・境界・分割数を検証し、"
        "編集後は必ず unittest を再実行して既存の公開関数・型注釈・定数仕様を維持すること。\n"
    )
    candidate_text = current.rstrip() + addition
    candidate_hash = hashlib.sha256(addition.encode("utf-8")).hexdigest()[:12]
    if candidate_hash in seen_hashes:
        addition = (
            "\n- エラー再現時は失敗行周辺のみを確認し、公開インターフェースを変えずに最小差分で修正後にテストを実行すること。\n"
        )
        candidate_text = current.rstrip() + addition

    rules_path.write_text(candidate_text, encoding="utf-8")
    return (
        f"Propose generalized post-edit verification & parsing rule "
        f"(traces={len(trace_ctx['summaries'])}, errors={len(trace_ctx['error_samples'])})"
    )


def _seed_worktree_traces(repo: Path, worktree: Path) -> None:
    """Copy ignored `.rsi/traces` and history into candidate ``worktree`` readably."""
    src_traces = repo / ".rsi" / "traces"
    if src_traces.is_dir():
        dst_traces = worktree / ".rsi" / "traces"
        shutil.copytree(src_traces, dst_traces, dirs_exist_ok=True)
    src_history = repo / ".rsi" / "evolution_history.jsonl"
    if src_history.is_file():
        dst_history = worktree / ".rsi" / "evolution_history.jsonl"
        dst_history.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_history, dst_history)


def _agy_propose(worktree: Path, trace_ctx: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """Invoke Antigravity CLI as Meta-Harness Proposer inside ``worktree``."""
    resolved_wt = worktree.resolve()
    history_tail = json.dumps(history[-5:], ensure_ascii=False)
    trace_summary = json.dumps(trace_ctx, ensure_ascii=False)
    prompt = (
        f"You are a Meta-Harness Proposer working ONLY inside worktree `{resolved_wt}`.\n"
        f"Target files you may edit: `{resolved_wt}/.agents/plugins/rsi-harness/rules/AGENTS.md` "
        f"or `{resolved_wt}/.agents/plugins/rsi-harness/skills/*/SKILL.md`.\n"
        "Constraints (RRSI):\n"
        f"1. NEVER read/edit any file outside `{resolved_wt}` (do NOT touch parent directories).\n"
        "2. Edit at most 2 files and <= 80 diff lines ONLY under `.agents/plugins/rsi-harness/rules/` "
        "or `.agents/plugins/rsi-harness/skills/`.\n"
        "3. Never touch `hooks.json`, `scripts/`, or `benchmarks/`.\n"
        "4. Do NOT hardcode task names or test filenames; propose generalizable engineering heuristics.\n"
        f"Recent trace context: {trace_summary}\n"
        f"Recent evolution history (avoid failed hypotheses): {history_tail}\n"
        f"Inspect `{resolved_wt}/.rsi/traces/` if present, then make one concise improvement inside `{resolved_wt}`."
    )
    env = dict(os.environ)
    env["RSI_EVOLUTION_MODE"] = "1"
    env["RSI_ALLOWED_ROOT"] = str(resolved_wt)
    proc = subprocess.run(
        ["agy", "-p", prompt, "--dangerously-skip-permissions", "--output-format", "json"],
        cwd=str(resolved_wt),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return (proc.stdout or proc.stderr or "agy proposer finished")[:400]


def validate_proposal_rrsi(
    worktree: Path,
    base_sha: str,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    benchmark_ids: list[str],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Enforce RRSI Proposal-Side Regularization & Critic Leakage/Budget Screening."""
    rrsi = config.get("rrsi", {})
    max_files = int(rrsi.get("max_files", 2))
    max_diff_lines = int(rrsi.get("max_diff_lines", 80))
    max_rules_bytes = int(rrsi.get("max_rules_bytes", 24000))
    editable_prefixes = tuple(rrsi.get("editable_prefixes", []))
    protected_prefixes = tuple(rrsi.get("protected_prefixes", []))

    status_out = _git(worktree, "status", "--porcelain").stdout.splitlines()
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
        if any(rel.startswith(p) for p in protected_prefixes):
            reasons.append(f"protected_file_modified:{rel}")
        elif editable_prefixes and not any(rel.startswith(p) for p in editable_prefixes):
            reasons.append(f"out_of_scope_file:{rel}")

    if len(changed_files) > max_files:
        reasons.append(f"edit_budget_files_exceeded:{len(changed_files)}>{max_files}")
    if diff_lines > max_diff_lines:
        reasons.append(f"edit_budget_lines_exceeded:{diff_lines}>{max_diff_lines}")

    seen_accepted_or_rejected = {h.get("diff_hash") for h in history if h.get("diff_hash")}
    if diff_hash in seen_accepted_or_rejected:
        reasons.append("duplicate_hypothesis")

    # Critic leakage screen: block benchmark IDs or grading file references.
    leak_tokens = list(benchmark_ids) + ["test_solution.py", "run_bench.py", "tasks.json"]
    added_lines = [
        ln[1:] for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++")
    ]
    added_blob = "\n".join(added_lines)
    for token in leak_tokens:
        if token in added_blob:
            reasons.append(f"critic_benchmark_leakage:{token}")

    rules_file = worktree / ".agents" / "plugins" / "rsi-harness" / "rules" / "AGENTS.md"
    rules_bytes = len(rules_file.read_bytes()) if rules_file.is_file() else 0
    if rules_bytes > max_rules_bytes:
        reasons.append(f"rules_budget_exceeded:{rules_bytes}>{max_rules_bytes}")

    meta = {
        "changed_files": changed_files,
        "diff_lines": diff_lines,
        "diff_hash": diff_hash,
        "rules_bytes": rules_bytes,
        "pruning_recommended": rules_bytes > int(max_rules_bytes * 0.75),
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

    b_rate, b_rate_std, b_rate_se, b_n = _extract_metric(baseline, "pass_rate")
    c_rate, c_rate_std, c_rate_se, c_n = _extract_metric(candidate, "pass_rate")
    b_tok, b_tok_std, b_tok_se, _ = _extract_metric(baseline, "total_tokens")
    c_tok, c_tok_std, c_tok_se, _ = _extract_metric(candidate, "total_tokens")
    b_lat, b_lat_std, b_lat_se, _ = _extract_metric(baseline, "duration_sec")
    c_lat, c_lat_std, c_lat_se, _ = _extract_metric(candidate, "duration_sec")

    rate_gain = c_rate - b_rate
    tok_reduction = ((b_tok - c_tok) / b_tok) if b_tok > 0 else 0.0
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
    if c_rate < b_rate - eps:
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

    return {
        "accepted": accepted,
        "reasons": reasons,
        "baseline_n": b_n,
        "candidate_n": c_n,
        "rate_gain": round(rate_gain, 6),
        "tok_reduction": round(tok_reduction, 6),
        "lat_reduction": round(lat_reduction, 6),
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
) -> dict[str, Any]:
    cmd = [
        "python3",
        "benchmarks/run_bench.py",
        "--repo",
        ".",
        "--runner",
        runner,
        "--trials",
        str(max(1, n_trials)),
    ]
    if trace_export_dir is not None:
        cmd.extend(["--trace-export-dir", str(trace_export_dir)])
    proc = subprocess.run(
        cmd,
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return json.loads(proc.stdout)


def run_evolution_cycle(
    repo_root: Path,
    branch: str = "rsi/evolved",
    runner: str = "mock",
    mock_bad_proposal: str | None = None,
) -> dict[str, Any]:
    """Run one full Meta-Harness + RRSI + SoL-Pi evolution cycle."""
    repo = repo_root.resolve()
    config = load_rsi_config(repo)
    n_trials = max(1, int(config.get("rrsi", {}).get("n_trials", 1)))
    history_path = repo / ".rsi" / "evolution_history.jsonl"
    history = load_jsonl(history_path)
    trace_ctx = _collect_trace_context(repo)

    tasks_manifest = json.loads((repo / "benchmarks" / "tasks.json").read_text(encoding="utf-8"))
    benchmark_ids = [t["id"] for t in tasks_manifest.get("tasks", [])]

    base_sha = _ensure_branch(repo, branch)

    run_id = uuid.uuid4().hex[:8]
    cand_branch = f"rsi/cand-{run_id}"
    tmp_wt_root = Path(tempfile.mkdtemp(prefix="rsi-wt-"))
    worktree_dir = tmp_wt_root / f"cand-{run_id}"

    _git(repo, "worktree", "add", "-b", cand_branch, str(worktree_dir), base_sha)
    try:
        _seed_worktree_traces(repo, worktree_dir)
        wt_traces_dir = worktree_dir / ".rsi" / "traces"
        baseline_report = _run_bench_in_repo(
            worktree_dir,
            runner=runner,
            trace_export_dir=wt_traces_dir,
            n_trials=n_trials,
        )
        # Persist exported baseline traces back into main repo's ignored .rsi/traces/
        # and refresh trace_ctx so Proposer sees the real baseline execution traces.
        if wt_traces_dir.is_dir():
            shutil.copytree(wt_traces_dir, repo / ".rsi" / "traces", dirs_exist_ok=True)
        trace_ctx = _collect_trace_context(worktree_dir)

        if runner == "mock":
            proposer_note = _mock_propose(
                worktree_dir, trace_ctx, history, override_mode=mock_bad_proposal
            )
        else:
            proposer_note = _agy_propose(worktree_dir, trace_ctx, history)

        prop_ok, prop_reasons, prop_meta = validate_proposal_rrsi(
            worktree_dir, base_sha, config, history, benchmark_ids
        )

        if not prop_ok:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "branch": branch,
                "n_trials": n_trials,
                "accepted": False,
                "stage": "proposal_critic",
                "reasons": prop_reasons,
                "proposer_note": proposer_note,
                "baseline": baseline_report,
                "candidate": None,
                **prop_meta,
            }
            append_jsonl(history_path, record)
            return record

        candidate_report = _run_bench_in_repo(worktree_dir, runner=runner, n_trials=n_trials)
        sel_diag = compute_selection_diagnostics(baseline_report, candidate_report, config)
        sel_ok = bool(sel_diag["accepted"])
        sel_reasons = list(sel_diag["reasons"])

        commit_sha = None
        if sel_ok:
            _git(worktree_dir, "add", "-A")
            msg = (
                f"rsi(evolve): pass_rate {baseline_report['pass_rate']}->"
                f"{candidate_report['pass_rate']}, tokens {baseline_report['total_tokens']}->"
                f"{candidate_report['total_tokens']}"
            )
            _git(worktree_dir, "commit", "-m", msg)
            commit_sha = _git(worktree_dir, "rev-parse", "HEAD").stdout.strip()
            _git(repo, "branch", "-f", branch, commit_sha)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "branch": branch,
            "n_trials": n_trials,
            "accepted": sel_ok,
            "commit_sha": commit_sha,
            "stage": "selection_gate",
            "reasons": sel_reasons,
            "selection_diagnostics": sel_diag,
            "proposer_note": proposer_note,
            "baseline": baseline_report,
            "candidate": candidate_report,
            **prop_meta,
        }
        append_jsonl(history_path, record)
        return record
    finally:
        _git(repo, "worktree", "remove", "--force", str(worktree_dir), check=False)
        _git(repo, "branch", "-D", cand_branch, check=False)
        shutil.rmtree(tmp_wt_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one RSI-Harness outer evolution cycle.")
    parser.add_argument("--repo", default=".", help="Target repository path")
    parser.add_argument("--branch", default="rsi/evolved", help="Branch to auto-apply winning edits")
    parser.add_argument("--runner", default="mock", choices=["mock", "agy"])
    parser.add_argument(
        "--mock-bad-proposal",
        default=None,
        choices=["leak", "budget", "protected"],
        help="Optional negative test mode for mock proposer",
    )
    args = parser.parse_args()

    result = run_evolution_cycle(
        Path(args.repo),
        branch=args.branch,
        runner=args.runner,
        mock_bad_proposal=args.mock_bad_proposal,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
