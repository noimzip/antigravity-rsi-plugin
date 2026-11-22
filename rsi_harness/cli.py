#!/usr/bin/env python3
"""Standalone `rsi-harness` CLI entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


def _ensure_scripts_on_path(repo_root: Path | None = None) -> Path:
    pkg_root = Path(__file__).resolve().parents[1]
    pkg_bench = pkg_root / "benchmarks"
    if (pkg_bench / "run_bench.py").is_file():
        os.environ.setdefault("RSI_BENCHMARKS_DIR", str(pkg_bench))
        if str(pkg_bench) not in sys.path:
            sys.path.insert(0, str(pkg_bench))

    candidates: list[Path] = [
        pkg_root / ".agents" / "plugins" / "rsi-harness" / "scripts",
    ]
    if repo_root is not None:
        candidates.append(repo_root.resolve() / ".agents" / "plugins" / "rsi-harness" / "scripts")
    for cand in candidates:
        if cand.is_dir():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            bench_dir = cand.parents[2] / "benchmarks"
            if (bench_dir / "run_bench.py").is_file() and str(bench_dir) not in sys.path:
                os.environ.setdefault("RSI_BENCHMARKS_DIR", str(bench_dir))
                sys.path.insert(0, str(bench_dir))
            return cand
    raise FileNotFoundError(
        "Could not locate '.agents/plugins/rsi-harness/scripts' directory"
    )


def _detect_runtime(repo: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if (repo / ".claude").exists() or (repo / "CLAUDE.md").exists():
        return "claude"
    if (repo / ".codex").exists():
        return "codex"
    if (repo / ".agents").exists():
        return "agy"
    if (repo / ".gemini").exists():
        return "gemini"
    for rt in ("agy", "claude", "codex", "gemini"):
        if shutil.which(rt):
            return rt
    return "agy"


def _detect_init_surfaces(repo: Path) -> tuple[list[dict[str, Any]], list[str], int, int, list[str], bool]:
    """Scan repo for mutable/protected surfaces excluding `.agents/plugins/rsi-harness/`."""
    mutable: list[dict[str, Any]] = []
    create_default_agents = False

    has_agents = (repo / "AGENTS.md").is_file()
    has_claude = (repo / "CLAUDE.md").is_file()
    if has_agents:
        mutable.append({"path": "AGENTS.md", "role": "rules"})
    if has_claude:
        mutable.append({"path": "CLAUDE.md", "role": "rules"})
    if not has_agents and not has_claude:
        mutable.append({"path": "AGENTS.md", "role": "rules"})
        create_default_agents = True

    has_dot_agents_skills = (repo / ".agents" / "skills").is_dir() or any(
        (repo / ".agents" / "skills").glob("*/SKILL.md")
    )
    has_skills_dir = (repo / "skills").is_dir() or any((repo / "skills").glob("*/SKILL.md"))
    if has_dot_agents_skills:
        mutable.append({"path": ".agents/skills/*/SKILL.md", "role": "skills"})
    if has_skills_dir:
        mutable.append({"path": "skills/*/SKILL.md", "role": "skills"})
    if not has_dot_agents_skills and not has_skills_dir:
        mutable.append({"path": "skills/*/SKILL.md", "role": "skills"})

    protected: list[str] = [
        ".agents/plugins/rsi-harness/hooks.json",
        ".agents/plugins/rsi-harness/scripts/",
        ".agents/plugins/rsi-harness/rules/",
        "benchmarks/",
    ]
    for dep_dir in ("vendor", "dist", "build"):
        if (repo / dep_dir).exists():
            protected.append(f"{dep_dir}/")

    actual_rules_bytes = 0
    for entry in mutable:
        if entry["role"] == "rules":
            rp = repo / str(entry["path"])
            if rp.is_file():
                actual_rules_bytes += len(rp.read_bytes())

    default_max = 24000
    warnings: list[str] = []
    if actual_rules_bytes > default_max:
        recommended_max = ((actual_rules_bytes + 999) // 1000) * 1000
        warnings.append(
            f"Existing rules surface is {actual_rules_bytes} bytes (> default {default_max} bytes); "
            f"adjusting max_rules_bytes to {recommended_max}. Consider Progressive Disclosure or "
            "rrsi.rules_budget_mode='ratchet'."
        )
    else:
        recommended_max = default_max

    for entry in mutable:
        if entry["role"] == "rules":
            entry["max_bytes"] = recommended_max

    return mutable, protected, actual_rules_bytes, recommended_max, warnings, create_default_agents


def _render_harness_yaml(
    runtime: str,
    check_cmd: str,
    timeout_sec: int,
    mutable_surface: list[dict[str, Any]],
    protected_surface: list[str],
    max_rules_bytes: int = 24000,
) -> str:
    escaped_cmd = check_cmd.replace('"', '\\"')
    lines = [
        f'runtime: "{runtime}"',
        f'inner_check_cmd: "{escaped_cmd}"',
        f"inner_timeout_sec: {timeout_sec}",
        "rrsi:",
        "  n_trials: 1",
        "  max_files: 3",
        "  max_diff_lines: 80",
        f"  max_rules_bytes: {max_rules_bytes}",
        "mutable_surface:",
    ]
    for entry in mutable_surface:
        lines.append(f'  - path: "{entry["path"]}"')
        lines.append(f'    role: "{entry["role"]}"')
        if entry.get("max_bytes") is not None:
            lines.append(f'    max_bytes: {int(entry["max_bytes"])}')
    lines.append("protected_surface:")
    for pat in protected_surface:
        lines.append(f'  - "{pat}"')
    return "\n".join(lines) + "\n"


def _cmd_init(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    _ensure_scripts_on_path(repo)
    from toolchain import detect_project_toolchain

    tc = detect_project_toolchain(repo)
    runtime = _detect_runtime(repo, args.runtime)
    (
        mutable_surface,
        protected_surface,
        actual_rules_bytes,
        recommended_max_rules_bytes,
        warnings,
        create_default_agents,
    ) = _detect_init_surfaces(repo)

    harness_path = repo / "harness.yaml"
    config_path = repo / "rsi.config.json"
    agents_path = repo / "AGENTS.md"
    gitignore_path = repo / ".gitignore"

    harness_content = _render_harness_yaml(
        runtime,
        tc.check_cmd,
        tc.default_timeout_sec,
        mutable_surface,
        protected_surface,
        max_rules_bytes=recommended_max_rules_bytes,
    )
    config_payload = {
        "runtime": runtime,
        "inner_gate": {
            "check_cmd": tc.check_cmd or None,
            "timeout_sec": tc.default_timeout_sec,
        },
        "rrsi": {
            "n_trials": 1,
            "max_rules_bytes": recommended_max_rules_bytes,
        },
        "plugins": {
            "proposer": None,
            "evaluator": None,
            "critic": None,
        },
    }
    agents_content = (
        "# Agent Operational Rules\n\n"
        f"- Project language: `{tc.language}` (tool: `{tc.tool}`).\n"
        + (
            f"- Run `{tc.check_cmd}` after code edits and inspect failure tracebacks before retrying.\n"
            if tc.check_cmd
            else "- Verify edits with repository tests before finishing.\n"
        )
        + "- Handle empty, zero, and boundary inputs safely.\n"
    )

    actions: list[dict[str, str]] = []

    def _handle_file(target: Path, content: str) -> None:
        rel = target.relative_to(repo).as_posix()
        existed = target.exists()
        if existed and not args.force:
            actions.append({"path": rel, "action": "skip", "reason": "already_exists"})
            return
        act = "overwrite" if existed else "create"
        actions.append({"path": rel, "action": act, "reason": "forced" if existed else "missing"})
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    _handle_file(harness_path, harness_content)
    _handle_file(config_path, json.dumps(config_payload, indent=2) + "\n")
    if create_default_agents or (repo / "AGENTS.md").is_file():
        _handle_file(agents_path, agents_content)

    # .gitignore append-only handling so existing entries are never lost
    if gitignore_path.exists():
        gi_text = gitignore_path.read_text(encoding="utf-8", errors="replace")
        has_rsi_ignore = any(line.strip() in {".rsi", ".rsi/", "/.rsi", "/.rsi/"} for line in gi_text.splitlines())
        if has_rsi_ignore:
            actions.append({"path": ".gitignore", "action": "skip", "reason": "already_ignores_.rsi"})
        else:
            actions.append({"path": ".gitignore", "action": "append", "reason": "add_.rsi/"})
            if not args.dry_run:
                sep = "" if (not gi_text or gi_text.endswith("\n")) else "\n"
                gitignore_path.write_text(gi_text + f"{sep}.rsi/\n", encoding="utf-8")
    else:
        actions.append({"path": ".gitignore", "action": "create", "reason": "missing"})
        if not args.dry_run:
            gitignore_path.write_text(".rsi/\n", encoding="utf-8")

    summary = {
        "repo": str(repo),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "runtime": runtime,
        "toolchain": asdict(tc),
        "mutable_surface": mutable_surface,
        "protected_surface": protected_surface,
        "rules_bytes": actual_rules_bytes,
        "recommended_max_rules_bytes": recommended_max_rules_bytes,
        "warnings": warnings,
        "actions": actions,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        mode_tag = "[DRY-RUN] " if args.dry_run else ""
        print(f"{mode_tag}RSI-Harness init in {repo} (runtime={runtime}, toolchain={tc.language}/{tc.tool}, cmd={tc.check_cmd!r})")
        for act in actions:
            print(f"  - {act['action']}: {act['path']} ({act['reason']})")
        for warn in warnings:
            print(f"  ! WARNING: {warn}")
    return 0


def _cmd_evolve(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    _ensure_scripts_on_path(repo)
    from evolve_outer import run_evolution_cycle

    pr_mode = "create" if args.create_pr else ("dry-run" if args.pr_dry_run else "none")
    result = run_evolution_cycle(
        repo,
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
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    _ensure_scripts_on_path(repo)
    from common import load_jsonl, load_rsi_config
    from lineage import LineageArchive
    from surface import resolve_mutable_surface
    from toolchain import detect_project_toolchain

    cfg = load_rsi_config(repo)
    tc = detect_project_toolchain(repo)
    surface = resolve_mutable_surface(repo, cfg)
    history = load_jsonl(repo / ".rsi" / "evolution_history.jsonl")
    archive = LineageArchive.load(repo)
    store_index = load_jsonl(repo / ".rsi" / "store" / "index.jsonl")
    traces_root = repo / ".rsi" / "traces"
    trace_sessions = len(list(traces_root.glob("*/summary.json"))) if traces_root.is_dir() else 0
    pr_dir = repo / ".rsi" / "pr_reports"
    pr_reports = sorted(p.name for p in pr_dir.glob("*.md")) if pr_dir.is_dir() else []

    status_data = {
        "repo": str(repo),
        "has_harness_yaml": (repo / "harness.yaml").is_file(),
        "has_rsi_config": (repo / "rsi.config.json").is_file(),
        "has_rsi_dir": (repo / ".rsi").is_dir(),
        "runtime": cfg.get("runtime", "agy"),
        "toolchain": asdict(tc),
        "mutable_surface_files": [
            p.relative_to(repo).as_posix() for p in surface.list_existing_files(repo)
        ],
        "evolution": {
            "total_cycles": len(history),
            "accepted_cycles": sum(1 for h in history if h.get("accepted")),
            "latest_run_id": history[-1].get("run_id") if history else None,
            "latest_accepted": history[-1].get("accepted") if history else None,
        },
        "lineage": {
            "node_count": len(archive.nodes),
            "root_id": archive.root_id,
            "pareto_frontier_ids": [n.node_id for n in archive.pareto_frontier()],
        },
        "trace_store": {
            "generations": len(store_index),
            "trace_sessions": trace_sessions,
        },
        "pr_reports": pr_reports,
    }
    if args.json:
        print(json.dumps(status_data, ensure_ascii=False, indent=2))
    else:
        print(f"Repository: {status_data['repo']}")
        print(f"Runtime: {status_data['runtime']} | Toolchain: {tc.language}/{tc.tool} ({tc.check_cmd or 'none'})")
        print(
            f"Evolution: {status_data['evolution']['accepted_cycles']}/{status_data['evolution']['total_cycles']} accepted | "
            f"Lineage nodes: {status_data['lineage']['node_count']} | TraceStore gens: {status_data['trace_store']['generations']}"
        )
        print(f"Pareto Frontier: {', '.join(status_data['lineage']['pareto_frontier_ids']) or '-'}")
    return 0


def _cmd_mine(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    _ensure_scripts_on_path(repo)
    from weakness_miner import mine_weaknesses, synthesize_regression_suite

    clusters = mine_weaknesses(repo)
    output_file: str | None = None
    if args.output and clusters:
        out_path = Path(args.output).resolve()
        synthesize_regression_suite(repo, clusters, output_path=out_path)
        output_file = str(out_path)
    payload = {
        "repo": str(repo),
        "cluster_count": len(clusters),
        "clusters": [asdict(c) for c in clusters],
        "synthesized_manifest": output_file,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Mined {len(clusters)} weakness cluster(s) from {repo}:")
        for c in clusters:
            print(f"  - [{c.category}] {c.title} (count={c.count}, avg_tokens={c.avg_tokens})")
        if output_file:
            print(f"Synthesized regression suite: {output_file}")
    return 0


def _cmd_lineage(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    _ensure_scripts_on_path(repo)
    from lineage import main as lineage_main

    sub_argv = ["--repo", str(repo)]
    if args.pareto:
        sub_argv.append("--pareto")
    if args.sample:
        sub_argv.extend(["--sample", args.sample])
    if args.seed is not None:
        sub_argv.extend(["--seed", str(args.seed)])
    if args.json:
        sub_argv.append("--json")
    return int(lineage_main(sub_argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsi-harness",
        description="Standalone CLI for RSI-Harness initialization, evolution, weakness mining, and lineage inspection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Auto-detect project toolchain and scaffold RSI config cleanly")
    p_init.add_argument("--repo", default=".", help="Target repository directory")
    p_init.add_argument(
        "--runtime",
        default="auto",
        choices=["auto", "agy", "claude", "codex", "gemini", "pty"],
        help="Target agent runtime",
    )
    p_init.add_argument("--dry-run", action="store_true", help="Report planned actions without writing files")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing harness.yaml / rsi.config.json / AGENTS.md")
    p_init.add_argument("--json", action="store_true", help="Emit JSON summary")
    p_init.set_defaults(func=_cmd_init)

    p_evolve = sub.add_parser("evolve", help="Run one bounded outer evolution cycle")
    p_evolve.add_argument("--repo", default=".", help="Target repository path")
    p_evolve.add_argument("--branch", default="rsi/evolved", help="Branch to apply accepted winning edits")
    p_evolve.add_argument("--runner", default="mock", help="Evaluation/proposer runner (mock, agy, claude, codex, gemini, pty)")
    p_evolve.add_argument("--manifest", default=None, help="Optional benchmark manifest or Harbor directory")
    p_evolve.add_argument("--mode", default="auto", choices=["auto", "benchmark", "trace-mined"])
    p_evolve.add_argument("--prune", action="store_true", help="Prune redundant rules / dead skills in candidate worktree")
    p_evolve.add_argument("--parent-strategy", default="head", choices=["head", "dgm", "latest", "best"])
    p_evolve.add_argument("--parent-node-id", default=None)
    p_evolve.add_argument("--seed", type=int, default=None)
    p_evolve.add_argument("--pr-dry-run", action="store_true")
    p_evolve.add_argument("--create-pr", action="store_true")
    p_evolve.add_argument("--pr-base", default=None, help="Base branch for PR (auto-detected if omitted)")
    p_evolve.add_argument(
        "--adaptive-trials",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable adaptive multi-trial evaluation expansion",
    )
    p_evolve.add_argument("--mock-bad-proposal", default=None, choices=["leak", "budget", "protected"])
    p_evolve.set_defaults(func=_cmd_evolve)

    p_status = sub.add_parser("status", help="Inspect RSI configuration, evolution history, and lineage state")
    p_status.add_argument("--repo", default=".", help="Target repository path")
    p_status.add_argument("--json", action="store_true", help="Emit JSON output")
    p_status.set_defaults(func=_cmd_status)

    p_mine = sub.add_parser("mine", help="Mine failure/inefficiency clusters from trace logs")
    p_mine.add_argument("--repo", default=".", help="Target repository path")
    p_mine.add_argument("--output", default=None, help="Optional path to write synthesized regression manifest JSON")
    p_mine.add_argument("--json", action="store_true", help="Emit JSON output")
    p_mine.set_defaults(func=_cmd_mine)

    p_lineage = sub.add_parser("lineage", help="Inspect DGM evolutionary DAG and Pareto frontier")
    p_lineage.add_argument("--repo", default=".", help="Target repository path or lineage.json")
    p_lineage.add_argument("--pareto", action="store_true", help="Show Pareto frontier nodes only")
    p_lineage.add_argument("--sample", default=None, choices=["dgm", "latest", "best"])
    p_lineage.add_argument("--seed", type=int, default=None)
    p_lineage.add_argument("--json", action="store_true", help="Emit JSON output")
    p_lineage.set_defaults(func=_cmd_lineage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
