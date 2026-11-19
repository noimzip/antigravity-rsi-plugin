#!/usr/bin/env python3
"""Pluggable Proposer, Evaluator, Critic protocols and dynamic loader."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable


@dataclass
class ProposalContext:
    """Context passed to a Proposer plugin inside an isolated worktree."""

    repo_root: Path
    worktree_dir: Path
    runner: str
    trace_context: dict[str, Any]
    history: list[dict[str, Any]]
    config: dict[str, Any]
    mock_bad_proposal: str | None = None
    parent_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Proposer(Protocol):
    """Protocol for pluggable candidate edit proposers."""

    def propose(self, context: ProposalContext) -> str: ...


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for pluggable benchmark or meta-reward evaluators."""

    def evaluate(
        self,
        worktree_dir: Path,
        *,
        runner: str,
        config: dict[str, Any],
        manifest_path: Path | None = None,
        split: str | None = None,
        trace_export_dir: Path | None = None,
        n_trials: int = 1,
        sandbox: Any = None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class Critic(Protocol):
    """Protocol for pluggable proposal-side regularization and safety critics."""

    def check(
        self,
        worktree_dir: Path,
        *,
        base_sha: str,
        config: dict[str, Any],
        history: list[dict[str, Any]],
        benchmark_ids: list[str],
        leakage_profile: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str], dict[str, Any]]: ...


class DefaultProposer:
    """Built-in proposer supporting mock, agy, and registered runtime adapters."""

    def propose(self, context: ProposalContext) -> str:
        import evolve_outer

        if context.runner == "mock":
            return evolve_outer._mock_propose(
                context.worktree_dir,
                context.trace_context,
                context.history,
                override_mode=context.mock_bad_proposal,
                config=context.config,
            )
        if context.runner == "agy":
            return evolve_outer._agy_propose(
                context.worktree_dir,
                context.trace_context,
                context.history,
                config=context.config,
            )
        return evolve_outer._runtime_propose(
            context.runner,
            context.worktree_dir,
            context.trace_context,
            context.history,
            config=context.config,
        )


class BenchmarkEvaluator:
    """Built-in benchmark evaluator invoking benchmarks/run_bench.py."""

    def evaluate(
        self,
        worktree_dir: Path,
        *,
        runner: str,
        config: dict[str, Any],
        manifest_path: Path | None = None,
        split: str | None = None,
        trace_export_dir: Path | None = None,
        n_trials: int = 1,
        sandbox: Any = None,
    ) -> dict[str, Any]:
        import evolve_outer

        return evolve_outer._run_bench_in_repo(
            worktree_dir,
            runner=runner,
            trace_export_dir=trace_export_dir,
            n_trials=n_trials,
            sandbox=sandbox,
            manifest_path=manifest_path,
            split=split,
        )


_MODULE_ATTR_SPEC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")


class MetaRewardEvaluator:
    """Evaluator wrapper combining hard verifier results with an LLM/script judge.

    Anti-Goodhart invariant: when ``anti_goodhart_passed`` is ``False`` on the base
    report, this evaluator NEVER flips it to ``True`` or inflates ``pass_rate``.
    """

    def __init__(
        self,
        base: Evaluator | str | dict[str, Any] | None = None,
        judge: str | Callable[[Path, dict[str, Any]], dict[str, Any] | float] | None = None,
        judge_weight: float = 0.25,
        judge_timeout: int = 60,
        search_paths: list[Path] | None = None,
    ) -> None:
        if isinstance(base, (str, dict)):
            self.base = load_plugin(
                base,
                Evaluator,
                BenchmarkEvaluator,
                entry_point_group="rsi_harness.evaluators",
                search_paths=search_paths,
            )
        elif base is None:
            self.base = BenchmarkEvaluator()
        else:
            self.base = base
        self.judge = judge
        self.judge_weight = max(0.0, min(1.0, float(judge_weight)))
        self.judge_timeout = max(1, int(judge_timeout))
        self.search_paths = list(search_paths or [])

    def _run_judge(self, worktree_dir: Path, report: dict[str, Any]) -> tuple[float | None, str]:
        if self.judge is None:
            return None, "no_judge_configured"

        judge_fn: Callable[[Path, dict[str, Any]], Any] | None = None
        if callable(self.judge):
            judge_fn = self.judge
        elif isinstance(self.judge, str) and _MODULE_ATTR_SPEC_RE.match(self.judge.strip()):
            loaded = _resolve_module_attr(self.judge.strip(), search_paths=self.search_paths)
            if callable(loaded):
                judge_fn = loaded

        if judge_fn is not None:
            res = judge_fn(worktree_dir, report)
            if isinstance(res, (int, float)):
                return max(0.0, min(1.0, float(res))), "callable_judge"
            if isinstance(res, dict):
                raw_score = res.get("score", res.get("pass_rate"))
                score = max(0.0, min(1.0, float(raw_score))) if raw_score is not None else None
                return score, str(res.get("notes") or "callable_judge")
            raise TypeError(f"Judge callable returned unsupported type: {type(res).__name__}")

        assert isinstance(self.judge, str)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(report, tmp)
            report_file = Path(tmp.name)
        try:
            cmd_str = self.judge.format(worktree=str(worktree_dir), report_file=str(report_file))
            env = dict(os.environ)
            env["RSI_WORKTREE"] = str(worktree_dir)
            env["RSI_BENCH_REPORT_FILE"] = str(report_file)
            proc = subprocess.run(
                shlex.split(cmd_str),
                cwd=str(worktree_dir),
                input=json.dumps(report),
                capture_output=True,
                text=True,
                timeout=self.judge_timeout,
                env=env,
            )
            if proc.returncode != 0:
                return None, f"judge_exit_{proc.returncode}: {proc.stderr.strip()[:160]}"
            out = proc.stdout.strip()
            if not out:
                return None, "judge_empty_output"
            try:
                parsed = json.loads(out)
                if isinstance(parsed, (int, float)):
                    return max(0.0, min(1.0, float(parsed))), "cli_judge_scalar"
                if isinstance(parsed, dict):
                    raw_score = parsed.get("score", parsed.get("pass_rate"))
                    score = max(0.0, min(1.0, float(raw_score))) if raw_score is not None else None
                    return score, str(parsed.get("notes") or "cli_judge")
            except json.JSONDecodeError:
                last_line = out.splitlines()[-1].strip()
                return max(0.0, min(1.0, float(last_line))), "cli_judge_line"
            return None, "judge_unparsed"
        finally:
            report_file.unlink(missing_ok=True)

    def evaluate(
        self,
        worktree_dir: Path,
        *,
        runner: str,
        config: dict[str, Any],
        manifest_path: Path | None = None,
        split: str | None = None,
        trace_export_dir: Path | None = None,
        n_trials: int = 1,
        sandbox: Any = None,
    ) -> dict[str, Any]:
        report = dict(
            self.base.evaluate(
                worktree_dir,
                runner=runner,
                config=config,
                manifest_path=manifest_path,
                split=split,
                trace_export_dir=trace_export_dir,
                n_trials=n_trials,
                sandbox=sandbox,
            )
        )
        base_pass_rate = float(report.get("pass_rate", 0.0))
        judge_score, judge_notes = self._run_judge(worktree_dir, report)
        anti_goodhart_passed = report.get("anti_goodhart_passed")

        if anti_goodhart_passed is False:
            report["anti_goodhart_passed"] = False
            report["meta_reward"] = {
                "base_pass_rate": base_pass_rate,
                "judge_score": judge_score,
                "judge_weight": self.judge_weight,
                "combined_pass_rate": base_pass_rate,
                "anti_goodhart_preserved": False,
                "notes": judge_notes,
            }
            return report

        combined = base_pass_rate
        if judge_score is not None:
            combined = round((1.0 - self.judge_weight) * base_pass_rate + self.judge_weight * judge_score, 4)
            report["pass_rate"] = combined
            if isinstance(report.get("stats"), dict) and isinstance(report["stats"].get("pass_rate"), dict):
                stats_copy = dict(report["stats"])
                pr_stats = dict(stats_copy["pass_rate"])
                pr_stats["mean"] = combined
                stats_copy["pass_rate"] = pr_stats
                report["stats"] = stats_copy

        report["meta_reward"] = {
            "base_pass_rate": base_pass_rate,
            "judge_score": judge_score,
            "judge_weight": self.judge_weight,
            "combined_pass_rate": combined,
            "anti_goodhart_preserved": True,
            "notes": judge_notes,
        }
        return report


class RrsiCritic:
    """Built-in Critic enforcing RRSI budgets, protected paths, and leakage checks."""

    def check(
        self,
        worktree_dir: Path,
        *,
        base_sha: str,
        config: dict[str, Any],
        history: list[dict[str, Any]],
        benchmark_ids: list[str],
        leakage_profile: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str], dict[str, Any]]:
        import evolve_outer

        return evolve_outer.validate_proposal_rrsi(
            worktree_dir,
            base_sha,
            config,
            history,
            benchmark_ids,
            leakage_profile=leakage_profile,
        )


T = TypeVar("T")


def _import_module_with_search_paths(mod_name: str, search_paths: list[Path] | None = None) -> Any:
    paths = [Path(p).resolve() for p in (search_paths or []) if Path(p).is_dir()]
    rel_parts = mod_name.split(".")
    for sp in paths:
        candidate_py = sp.joinpath(*rel_parts[:-1], f"{rel_parts[-1]}.py")
        candidate_pkg = sp.joinpath(*rel_parts, "__init__.py")
        candidate = candidate_py if candidate_py.is_file() else (candidate_pkg if candidate_pkg.is_file() else None)
        if candidate is not None:
            existing = sys.modules.get(mod_name)
            existing_file = Path(getattr(existing, "__file__", "") or "").resolve() if existing else None
            if existing is None or existing_file != candidate.resolve():
                spec = importlib.util.spec_from_file_location(mod_name, str(candidate))
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = mod
                    spec.loader.exec_module(mod)
                    return mod

    inserted: list[str] = []
    for sp in reversed(paths):
        s = str(sp)
        if s not in sys.path:
            sys.path.insert(0, s)
            inserted.append(s)
    importlib.invalidate_caches()
    try:
        return importlib.import_module(mod_name)
    finally:
        for s in inserted:
            if s in sys.path:
                sys.path.remove(s)


def _resolve_module_attr(raw_spec: str, search_paths: list[Path] | None = None) -> Any:
    if ":" in raw_spec:
        mod_name, attr_path = raw_spec.split(":", 1)
    elif "." in raw_spec:
        mod_name, attr_path = raw_spec.rsplit(".", 1)
    else:
        raise ValueError(f"Plugin spec {raw_spec!r} must be 'module:attr' or a registered entry point")
    mod_name = mod_name.strip()
    attr_path = attr_path.strip()
    if not mod_name or not attr_path:
        raise ValueError(f"Invalid module/attribute plugin spec: {raw_spec!r}")
    mod = _import_module_with_search_paths(mod_name, search_paths=search_paths)
    obj: Any = mod
    for part in attr_path.split("."):
        if not hasattr(obj, part):
            raise ValueError(f"Attribute {attr_path!r} not found in module {mod_name!r}")
        obj = getattr(obj, part)
    return obj


def load_plugin(
    spec: Any,
    expected_protocol: type[T],
    default_factory: Callable[[], T],
    *,
    entry_point_group: str | None = None,
    search_paths: list[Path] | None = None,
) -> T:
    """Resolve, instantiate, and protocol-check a Proposer/Evaluator/Critic plugin."""
    if spec is None:
        return default_factory()

    if isinstance(spec, expected_protocol):
        return spec

    kwargs: dict[str, Any] = {}
    if isinstance(spec, dict):
        raw_spec = spec.get("spec") or spec.get("target")
        if not isinstance(raw_spec, str) or not raw_spec.strip():
            raise ValueError(f"Plugin dict config must include non-empty 'spec': {spec!r}")
        raw_kwargs = spec.get("kwargs", {})
        if raw_kwargs is None:
            raw_kwargs = {}
        if not isinstance(raw_kwargs, dict):
            raise ValueError(f"Plugin 'kwargs' must be a dict: {spec!r}")
        kwargs = dict(raw_kwargs)
        raw_spec = raw_spec.strip()
    elif isinstance(spec, str):
        raw_spec = spec.strip()
        if not raw_spec:
            raise ValueError("Plugin spec string cannot be empty")
    else:
        raise TypeError(
            f"Unsupported plugin spec {spec!r} for protocol {expected_protocol.__name__}"
        )

    target: Any = None
    if ":" not in raw_spec and "." not in raw_spec and entry_point_group:
        eps = importlib.metadata.entry_points(group=entry_point_group)
        matches = [ep for ep in eps if ep.name == raw_spec]
        if not matches:
            raise ValueError(
                f"Entry point {raw_spec!r} not found in group {entry_point_group!r}"
            )
        target = matches[0].load()
    else:
        target = _resolve_module_attr(raw_spec, search_paths=search_paths)

    if isinstance(target, expected_protocol) and not isinstance(target, type) and not kwargs:
        return target

    if callable(target):
        instance = target(**kwargs)
    else:
        if kwargs:
            raise ValueError(f"Cannot pass kwargs to non-callable plugin target {raw_spec!r}")
        instance = target

    if not isinstance(instance, expected_protocol):
        raise TypeError(
            f"Plugin {raw_spec!r} resolved to {type(instance).__name__}, "
            f"which does not implement {expected_protocol.__name__}"
        )
    return instance
