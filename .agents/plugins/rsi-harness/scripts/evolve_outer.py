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
import os
import subprocess
import sys
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


def _agy_propose(worktree: Path, trace_ctx: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """Invoke Antigravity CLI as Meta-Harness Proposer inside ``worktree``."""
    history_tail = json.dumps(history[-5:], ensure_ascii=False)
    trace_summary = json.dumps(trace_ctx, ensure_ascii=False)
    prompt = (
        "You are a Meta-Harness Proposer optimizing `.agents/plugins/rsi-harness/rules/AGENTS.md` "
        "or `.agents/plugins/rsi-harness/skills/*/SKILL.md`.\n"
        "Constraints (RRSI):\n"
        "1. Edit at most 2 files and <= 80 diff lines ONLY under `.agents/plugins/rsi-harness/rules/` "
        "or `.agents/plugins/rsi-harness/skills/`.\n"
        "2. Never touch `hooks.json`, `scripts/`, or `benchmarks/`.\n"
        "3. Do NOT hardcode task names or test filenames; propose generalizable engineering heuristics.\n"
        f"Recent trace context: {trace_summary}\n"
        f"Recent evolution history (avoid failed hypotheses): {history_tail}\n"
        "Inspect `.rsi/traces/` if present, then make one concise improvement."
    )
    env = dict(os.environ)
    env["RSI_EVOLUTION_MODE"] = "1"
    proc = subprocess.run(
        ["agy", "-p", prompt, "--dangerously-skip-permissions", "--output-format", "json"],
        cwd=str(worktree),
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


def evaluate_selection_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply RRSI Noise-Adjusted Floor + SoL-Pi Token Cost Gate."""
    rrsi = config.get("rrsi", {})
    min_gain = float(rrsi.get("min_pass_rate_gain", 0.05))
    min_tok_red = float(rrsi.get("min_token_reduction_ratio", 0.1))

    base_rate = float(baseline.get("pass_rate", 0.0))
    cand_rate = float(candidate.get("pass_rate", 0.0))
    base_tok = int(baseline.get("total_tokens", 0))
    cand_tok = int(candidate.get("total_tokens", 0))

    rate_gain = cand_rate - base_rate
    tok_reduction = ((base_tok - cand_tok) / base_tok) if base_tok > 0 else 0.0

    if rate_gain >= min_gain:
        if base_tok > 0 and cand_tok > int(base_tok * 1.15):
            return False, [f"token_overhead_too_high:{cand_tok}>{int(base_tok * 1.15)}"]
        return True, [f"pass_rate_improved:+{rate_gain:.3f}"]

    if cand_rate >= base_rate and tok_reduction >= min_tok_red:
        return True, [f"token_efficiency_improved:-{tok_reduction * 100:.1f}%"]

    return False, [
        f"insufficient_gain:rate_delta={rate_gain:.3f},token_reduction={tok_reduction:.3f}"
    ]


def _run_bench_in_repo(repo_path: Path, runner: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["python3", "benchmarks/run_bench.py", "--repo", ".", "--runner", runner],
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
    history_path = repo / ".rsi" / "evolution_history.jsonl"
    history = load_jsonl(history_path)
    trace_ctx = _collect_trace_context(repo)

    tasks_manifest = json.loads((repo / "benchmarks" / "tasks.json").read_text(encoding="utf-8"))
    benchmark_ids = [t["id"] for t in tasks_manifest.get("tasks", [])]

    base_sha = _ensure_branch(repo, branch)

    run_id = uuid.uuid4().hex[:8]
    cand_branch = f"rsi/cand-{run_id}"
    worktree_dir = repo / ".rsi" / "worktrees" / f"cand-{run_id}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    _git(repo, "worktree", "add", "-b", cand_branch, str(worktree_dir), base_sha)
    try:
        baseline_report = _run_bench_in_repo(worktree_dir, runner=runner)

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

        candidate_report = _run_bench_in_repo(worktree_dir, runner=runner)
        sel_ok, sel_reasons = evaluate_selection_gate(baseline_report, candidate_report, config)

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
            "accepted": sel_ok,
            "commit_sha": commit_sha,
            "stage": "selection_gate",
            "reasons": sel_reasons,
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
