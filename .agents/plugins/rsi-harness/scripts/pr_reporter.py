#!/usr/bin/env python3
"""Evidence-backed Markdown PR report generator and safe gh PR creator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


_PROTECTED_BASE_BRANCHES = {"main", "master"}


def _format_metric_row(name: str, baseline: dict[str, Any] | None, candidate: dict[str, Any] | None, key: str) -> str:
    b_val = (baseline or {}).get(key)
    c_val = (candidate or {}).get(key)
    if b_val is None and c_val is None:
        return f"| {name} | - | - | - |"
    try:
        b_num = float(b_val) if b_val is not None else None
        c_num = float(c_val) if c_val is not None else None
    except (TypeError, ValueError):
        return f"| {name} | {b_val} | {c_val} | - |"
    b_str = f"{b_num:.4f}" if key == "pass_rate" and b_num is not None else (f"{b_num:.3f}" if key == "duration_sec" and b_num is not None else (str(int(b_num)) if b_num is not None else "-"))
    c_str = f"{c_num:.4f}" if key == "pass_rate" and c_num is not None else (f"{c_num:.3f}" if key == "duration_sec" and c_num is not None else (str(int(c_num)) if c_num is not None else "-"))
    if b_num is not None and c_num is not None:
        delta = c_num - b_num
        d_str = f"{delta:+.4f}" if key == "pass_rate" else (f"{delta:+.3f}" if key == "duration_sec" else f"{delta:+.0f}")
    else:
        d_str = "-"
    return f"| {name} | {b_str} | {c_str} | {d_str} |"


def generate_pr_report(
    record: dict[str, Any],
    repo_root: Path | None = None,
    *,
    base_branch: str = "main",
    diff_text: str | None = None,
    max_diff_chars: int = 4000,
) -> str:
    """Render an evidence-backed Markdown summary for a single evolution cycle."""
    run_id = str(record.get("run_id") or "unknown")
    accepted = bool(record.get("accepted"))
    stage = str(record.get("stage") or "unknown")
    head_branch = str(record.get("node_branch") or record.get("branch") or "rsi/evolved")
    baseline = record.get("baseline") if isinstance(record.get("baseline"), dict) else {}
    candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else None
    ood_base = record.get("ood_baseline") if isinstance(record.get("ood_baseline"), dict) else None
    ood_cand = record.get("ood_candidate") if isinstance(record.get("ood_candidate"), dict) else None

    if diff_text is None and repo_root is not None and record.get("commit_sha"):
        probe = subprocess.run(
            ["git", "show", "--patch", "--stat", str(record["commit_sha"])],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            diff_text = probe.stdout

    if not diff_text and isinstance(record.get("store_dir"), str):
        patch_file = Path(record["store_dir"]) / "proposal.diff"
        if patch_file.is_file():
            diff_text = patch_file.read_text(encoding="utf-8", errors="replace")

    lines: list[str] = [
        f"# RSI-Harness Evolution Report (`{run_id}`)",
        "",
        "## Summary",
        f"- **Status**: {'ACCEPTED' if accepted else 'REJECTED'} (stage: `{stage}`)",
        f"- **Target Branch**: `{head_branch}` -> `{base_branch}`",
        f"- **Commit SHA**: `{record.get('commit_sha') or 'uncommitted'}`",
        f"- **Mode / Split**: `{record.get('evolution_mode', 'benchmark')}` / `{record.get('split_mode', 'single')}` (trials={record.get('n_trials', 1)})",
        f"- **Lineage**: parent=`{record.get('parent_node_id') or 'root'}`, node=`{record.get('lineage_node_id') or '-'}`, strategy=`{record.get('parent_strategy', 'head')}`",
        f"- **Reasons**: {', '.join(f'`{r}`' for r in (record.get('reasons') or [])) or 'none'}",
        "",
        "## Benchmark & Selection Evidence",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        _format_metric_row("pass_rate", baseline, candidate, "pass_rate"),
        _format_metric_row("total_tokens", baseline, candidate, "total_tokens"),
        _format_metric_row("duration_sec", baseline, candidate, "duration_sec"),
        "",
        f"- **Anti-Goodhart Verifier Integrity**: baseline=`{baseline.get('anti_goodhart_passed', True)}`, candidate=`{(candidate or {}).get('anti_goodhart_passed', 'n/a')}`",
    ]

    if ood_base is not None or ood_cand is not None:
        lines.extend(
            [
                "",
                "### Out-of-Distribution (OOD) Generalization",
                "| Metric | OOD Baseline | OOD Candidate | Delta |",
                "|---|---:|---:|---:|",
                _format_metric_row("ood_pass_rate", ood_base, ood_cand, "pass_rate"),
                _format_metric_row("ood_total_tokens", ood_base, ood_cand, "total_tokens"),
            ]
        )

    if isinstance((candidate or {}).get("meta_reward"), dict):
        lines.extend(["", f"- **Meta-Reward Breakdown**: `{json.dumps(candidate['meta_reward'], ensure_ascii=False)}`"])

    lines.extend(
        [
            "",
            "## Proposer & Plugins",
            f"- **Note**: {record.get('proposer_note') or '-'}",
            f"- **Plugins**: `{json.dumps(record.get('plugins_used') or {}, ensure_ascii=False)}`",
            f"- **Pareto Frontier**: `{', '.join(record.get('pareto_frontier_ids') or []) or '-'}`",
            "",
            "## RRSI Bounded Surface & Diff",
            f"- **Changed Files**: {', '.join(f'`{p}`' for p in (record.get('changed_files') or [])) or 'none'}",
            f"- **Diff Lines**: `{record.get('diff_lines', 0)}` | **Rules Bytes**: `{record.get('rules_bytes', 0)}` | **Diff Hash**: `{record.get('diff_hash') or '-'}`",
            f"- **TraceStore Snapshot**: `{record.get('store_dir') or '-'}` (gen={record.get('store_generation', '-')})",
        ]
    )

    if diff_text:
        clipped = diff_text.strip()
        truncated = len(clipped) > max_diff_chars
        if truncated:
            clipped = clipped[:max_diff_chars].rstrip() + "\n... [truncated]"
        lines.extend(["", "```diff", clipped, "```"])

    return "\n".join(lines) + "\n"


def create_review_pr(
    repo_root: Path,
    record: dict[str, Any],
    *,
    base_branch: str = "main",
    head_branch: str | None = None,
    dry_run: bool = False,
    push: bool = True,
    title: str | None = None,
    diff_text: str | None = None,
) -> dict[str, Any]:
    """Write PR report and either dry-run or open a GitHub PR via ``gh``."""
    repo = repo_root.resolve()
    target_head = (head_branch or record.get("node_branch") or record.get("branch") or "").strip()

    if not target_head:
        raise ValueError("Refusing to create PR without a non-empty head branch")
    if target_head in _PROTECTED_BASE_BRANCHES:
        raise ValueError(f"Refusing to create PR from protected branch {target_head!r}")
    if target_head == base_branch.strip():
        raise ValueError(f"Refusing to create PR where head == base ({target_head!r})")
    if not dry_run and not record.get("accepted"):
        raise ValueError("Refusing to open a live PR for an unaccepted evolution cycle")

    run_id = str(record.get("run_id") or "unknown")
    reports_dir = repo / ".rsi" / "pr_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"pr_{run_id}.md"
    markdown = generate_pr_report(record, repo_root=repo, base_branch=base_branch, diff_text=diff_text)
    report_path.write_text(markdown, encoding="utf-8")

    b_rate = float((record.get("baseline") or {}).get("pass_rate", 0.0))
    c_rate = float((record.get("candidate") or {}).get("pass_rate", b_rate))
    node_label = record.get("lineage_node_id") or run_id
    pr_title = title or f"rsi(evolve): {node_label} (pass_rate {b_rate:.1%} -> {c_rate:.1%})"

    push_cmd = ["git", "push", "-u", "origin", target_head]
    pr_cmd = [
        "gh",
        "pr",
        "create",
        "--base",
        base_branch,
        "--head",
        target_head,
        "--title",
        pr_title,
        "--body-file",
        str(report_path),
    ]

    if dry_run:
        return {
            "dry_run": True,
            "created": False,
            "report_path": str(report_path),
            "head_branch": target_head,
            "base_branch": base_branch,
            "title": pr_title,
            "commands": [push_cmd, pr_cmd] if push else [pr_cmd],
        }

    if push:
        push_proc = subprocess.run(push_cmd, cwd=str(repo), capture_output=True, text=True)
        if push_proc.returncode != 0:
            raise RuntimeError(f"git push failed: {push_proc.stderr.strip() or push_proc.stdout.strip()}")

    gh_bin = shutil.which("gh")
    if not gh_bin:
        raise FileNotFoundError("GitHub CLI ('gh') executable not found on PATH")
    pr_cmd[0] = gh_bin
    pr_proc = subprocess.run(pr_cmd, cwd=str(repo), capture_output=True, text=True)
    if pr_proc.returncode != 0:
        raise RuntimeError(f"gh pr create failed: {pr_proc.stderr.strip() or pr_proc.stdout.strip()}")

    return {
        "dry_run": False,
        "created": True,
        "pr_url": pr_proc.stdout.strip(),
        "report_path": str(report_path),
        "head_branch": target_head,
        "base_branch": base_branch,
        "title": pr_title,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or submit an RSI-Harness review PR")
    parser.add_argument("--repo", type=Path, default=Path("."), help="Repository path")
    parser.add_argument("--base", default="main", help="Base branch for the PR")
    parser.add_argument("--head", default=None, help="Override head branch")
    parser.add_argument("--dry-run", action="store_true", help="Write report without pushing or opening PR")
    parser.add_argument("--no-push", action="store_true", help="Skip git push before gh pr create")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    history_file = repo / ".rsi" / "evolution_history.jsonl"
    if not history_file.is_file():
        raise FileNotFoundError(f"No evolution history at {history_file}")
    lines = [ln for ln in history_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Evolution history is empty")
    record = json.loads(lines[-1])
    res = create_review_pr(
        repo,
        record,
        base_branch=args.base,
        head_branch=args.head,
        dry_run=args.dry_run,
        push=not args.no_push,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
