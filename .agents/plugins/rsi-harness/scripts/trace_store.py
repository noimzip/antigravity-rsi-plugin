#!/usr/bin/env python3
"""Meta-Harness uncompressed diagnostic trace & generation lineage store."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import append_jsonl, load_jsonl  # noqa: E402

_ERROR_EXTRACT_RE = re.compile(
    r"\b([A-Za-z_]*Error|Exception|FAILED|critic_[a-z_]+|edit_budget_[a-z_]+|"
    r"protected_file_modified|out_of_scope_file|duplicate_hypothesis|"
    r"rules_budget_exceeded|token_overhead_too_high|pass_rate_regressed|"
    r"noisy_variance_exceeds_gain|insufficient_gain|no_changes)\b"
)
_SEARCHABLE_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".diff", ".md", ".py"}


def _safe_rel_copy(src_root: Path, rel_path: str, dst_root: Path) -> None:
    rel = Path(rel_path)
    if rel.is_absolute() or ".." in rel.parts:
        return
    src = (src_root / rel).resolve(strict=False)
    if not src.is_file() or not src.is_relative_to(src_root.resolve()):
        return
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _extract_task_and_error_signals(
    reasons: list[str],
    baseline_report: dict[str, Any] | None,
    candidate_report: dict[str, Any] | None,
    traces_dir: Path | None,
) -> tuple[list[str], list[str], list[str]]:
    task_ids: set[str] = set()
    failed_task_ids: set[str] = set()
    error_types: set[str] = set()

    for reason in reasons:
        head = str(reason).split(":", 1)[0].strip()
        if head:
            error_types.add(head)
        for m in _ERROR_EXTRACT_RE.findall(str(reason)):
            error_types.add(m)

    for report in (baseline_report, candidate_report):
        if not isinstance(report, dict):
            continue
        for task in report.get("tasks", []):
            if not isinstance(task, dict):
                continue
            tid = task.get("id")
            if isinstance(tid, str) and tid:
                task_ids.add(tid)
                if not task.get("passed"):
                    failed_task_ids.add(tid)
            tail = str(task.get("grader_tail") or "")
            for m in _ERROR_EXTRACT_RE.findall(tail):
                error_types.add(m)

    if traces_dir is not None and traces_dir.is_dir():
        for ev_file in traces_dir.glob("*/events.jsonl"):
            for ev in load_jsonl(ev_file):
                err = ev.get("error")
                if err:
                    for m in _ERROR_EXTRACT_RE.findall(str(err)):
                        error_types.add(m)

    return sorted(task_ids), sorted(failed_task_ids), sorted(error_types)


def _iter_generation_text_files(gen_dir: Path) -> list[Path]:
    files: list[Path] = []
    for name in ("metadata.json", "proposal.diff", "config_snapshot.json"):
        p = gen_dir / name
        if p.is_file():
            files.append(p)
    for sub in ("traces", "candidate_traces", "harness_files"):
        sub_dir = gen_dir / sub
        if sub_dir.is_dir():
            for p in sorted(sub_dir.rglob("*")):
                if p.is_file() and (p.suffix.lower() in _SEARCHABLE_SUFFIXES or p.suffix == ""):
                    files.append(p)
    return files


class TraceStore:
    """Local filesystem store for uncompressed candidate snapshots, diffs, and traces."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.rsi_dir = self.repo_root / ".rsi"
        self.store_root = self.rsi_dir / "store"
        self.index_path = self.store_root / "index.jsonl"
        self.generations_dir = self.store_root / "generations"

    def next_generation_number(self) -> int:
        return len(load_jsonl(self.index_path)) + 1

    def record_generation(
        self,
        *,
        run_id: str,
        branch: str,
        parent_sha: str,
        commit_sha: str | None,
        accepted: bool,
        stage: str,
        reasons: list[str],
        diff_text: str,
        changed_files: list[str],
        config: dict[str, Any],
        baseline_report: dict[str, Any] | None,
        candidate_report: dict[str, Any] | None,
        traces_src_dir: Path | None = None,
        candidate_traces_dir: Path | None = None,
        worktree: Path | None = None,
        proposer_note: str = "",
    ) -> dict[str, Any]:
        """Persist uncompressed snapshot, diff, config, and raw traces for one generation."""
        gen_num = self.next_generation_number()
        gen_dir = self.generations_dir / f"gen_{gen_num:04d}_{run_id}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        (gen_dir / "proposal.diff").write_text(diff_text, encoding="utf-8")
        (gen_dir / "config_snapshot.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if worktree is not None and changed_files:
            harness_dst = gen_dir / "harness_files"
            for rel_path in changed_files:
                _safe_rel_copy(worktree, rel_path, harness_dst)

        if traces_src_dir is not None and traces_src_dir.is_dir():
            shutil.copytree(traces_src_dir, gen_dir / "traces", dirs_exist_ok=True, symlinks=False)

        if candidate_traces_dir is not None and candidate_traces_dir.is_dir():
            shutil.copytree(
                candidate_traces_dir,
                gen_dir / "candidate_traces",
                dirs_exist_ok=True,
                symlinks=False,
            )

        task_ids, failed_task_ids, error_types = _extract_task_and_error_signals(
            reasons, baseline_report, candidate_report, traces_src_dir
        )

        record: dict[str, Any] = {
            "generation": gen_num,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "branch": branch,
            "parent_sha": parent_sha,
            "commit_sha": commit_sha,
            "accepted": bool(accepted),
            "stage": stage,
            "reasons": list(reasons),
            "changed_files": list(changed_files),
            "task_ids": task_ids,
            "failed_task_ids": failed_task_ids,
            "error_types": error_types,
            "baseline_pass_rate": (
                baseline_report.get("pass_rate") if isinstance(baseline_report, dict) else None
            ),
            "baseline_total_tokens": (
                baseline_report.get("total_tokens") if isinstance(baseline_report, dict) else None
            ),
            "candidate_pass_rate": (
                candidate_report.get("pass_rate") if isinstance(candidate_report, dict) else None
            ),
            "candidate_total_tokens": (
                candidate_report.get("total_tokens") if isinstance(candidate_report, dict) else None
            ),
            "proposer_note": proposer_note,
            "snapshot_dir": str(gen_dir),
        }
        (gen_dir / "metadata.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        append_jsonl(self.index_path, record)
        return record

    def list_generations(
        self,
        *,
        accepted: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        entries = load_jsonl(self.index_path)
        if accepted is not None:
            entries = [e for e in entries if bool(e.get("accepted")) is accepted]
        if limit is not None and limit > 0:
            entries = entries[-limit:]
        return entries

    def query(
        self,
        *,
        generation: int | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        error_type: str | None = None,
        text: str | None = None,
        accepted: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Query stored generations by generation, run_id, task, error type, status, or text."""
        results: list[dict[str, Any]] = []
        err_lower = error_type.lower() if error_type else None
        txt_lower = text.lower() if text else None

        for rec in load_jsonl(self.index_path):
            if generation is not None and rec.get("generation") != generation:
                continue
            if run_id is not None and rec.get("run_id") != run_id:
                continue
            if accepted is not None and bool(rec.get("accepted")) is not accepted:
                continue
            if task_id is not None:
                tasks_here = set(rec.get("task_ids", [])) | set(rec.get("failed_task_ids", []))
                if task_id not in tasks_here:
                    continue
            if err_lower is not None:
                hay = [str(x).lower() for x in (rec.get("error_types", []) + rec.get("reasons", []))]
                if not any(err_lower in item for item in hay):
                    gen_dir = Path(str(rec.get("snapshot_dir", "")))
                    if not self._snapshot_contains(gen_dir, err_lower):
                        continue
            if txt_lower is not None:
                gen_dir = Path(str(rec.get("snapshot_dir", "")))
                if not self._snapshot_contains(gen_dir, txt_lower):
                    continue
            results.append(rec)
        return results

    def _snapshot_contains(self, gen_dir: Path, needle_lower: str) -> bool:
        if not gen_dir.is_dir():
            return False
        for file_path in _iter_generation_text_files(gen_dir):
            try:
                if needle_lower in file_path.read_text(encoding="utf-8", errors="replace").lower():
                    return True
            except OSError:
                continue
        return False

    def grep_traces(
        self,
        pattern: str,
        *,
        generation: int | None = None,
        max_hits: int = 50,
    ) -> list[dict[str, Any]]:
        """Regex search across uncompressed trace files, diffs, and harness snapshots."""
        if max_hits < 1:
            raise ValueError(f"max_hits must be >= 1, got {max_hits}")
        compiled = re.compile(pattern)
        hits: list[dict[str, Any]] = []
        for rec in load_jsonl(self.index_path):
            if generation is not None and rec.get("generation") != generation:
                continue
            gen_dir = Path(str(rec.get("snapshot_dir", "")))
            if not gen_dir.is_dir():
                continue
            for file_path in _iter_generation_text_files(gen_dir):
                try:
                    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                rel = file_path.relative_to(gen_dir).as_posix()
                for idx, line in enumerate(lines, start=1):
                    if compiled.search(line):
                        hits.append(
                            {
                                "generation": rec.get("generation"),
                                "run_id": rec.get("run_id"),
                                "file": rel,
                                "line_number": idx,
                                "line": line,
                            }
                        )
                        if len(hits) >= max_hits:
                            return hits
        return hits

    def get_generation_bundle(self, run_id_or_gen: str | int) -> dict[str, Any] | None:
        """Load full diagnostic bundle (diff, config, metadata, file lists) for one generation."""
        target_rec: dict[str, Any] | None = None
        for rec in load_jsonl(self.index_path):
            if isinstance(run_id_or_gen, int) and rec.get("generation") == run_id_or_gen:
                target_rec = rec
                break
            if str(run_id_or_gen) in {str(rec.get("run_id")), str(rec.get("generation"))}:
                target_rec = rec
                break
        if target_rec is None:
            return None

        gen_dir = Path(str(target_rec.get("snapshot_dir", "")))
        diff_path = gen_dir / "proposal.diff"
        cfg_path = gen_dir / "config_snapshot.json"
        harness_dir = gen_dir / "harness_files"
        traces_dir = gen_dir / "traces"
        return {
            "metadata": target_rec,
            "proposal_diff": diff_path.read_text(encoding="utf-8") if diff_path.is_file() else "",
            "config_snapshot": (
                json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
            ),
            "harness_files": (
                [p.relative_to(harness_dir).as_posix() for p in sorted(harness_dir.rglob("*")) if p.is_file()]
                if harness_dir.is_dir()
                else []
            ),
            "trace_files": (
                [p.relative_to(traces_dir).as_posix() for p in sorted(traces_dir.rglob("*")) if p.is_file()]
                if traces_dir.is_dir()
                else []
            ),
        }

    def mount_readonly_into_worktree(self, worktree: Path) -> dict[str, str]:
        """Expose historical store/traces/history to ``worktree`` via symlinks without copying."""
        mounted: dict[str, str] = {}
        wt_rsi = worktree / ".rsi"
        wt_rsi.mkdir(parents=True, exist_ok=True)
        # Keep worktree/.rsi/traces as a fresh directory for current baseline exports.
        (wt_rsi / "traces").mkdir(parents=True, exist_ok=True)

        links = [
            (self.store_root, wt_rsi / "store", "store"),
            (self.rsi_dir / "traces", wt_rsi / "traces_archive", "traces_archive"),
            (self.rsi_dir / "evolution_history.jsonl", wt_rsi / "evolution_history.jsonl", "history"),
        ]
        for src, dst, label in links:
            if not src.exists():
                continue
            if dst.is_symlink() or dst.exists():
                if dst.is_symlink():
                    dst.unlink()
                else:
                    continue
            try:
                os.symlink(str(src), str(dst), target_is_directory=src.is_dir())
                mounted[label] = str(dst)
            except OSError:
                if src.is_file():
                    shutil.copy2(src, dst)
                    mounted[label] = str(dst)
        return mounted


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Meta-Harness uncompressed TraceStore.")
    parser.add_argument("--repo", default=".", help="Repository root containing .rsi/store")
    sub = parser.add_subparsers(dest="command", required=True)

    q_cmd = sub.add_parser("query", help="Query stored evolution generations")
    q_cmd.add_argument("--generation", type=int, default=None)
    q_cmd.add_argument("--run-id", default=None)
    q_cmd.add_argument("--task", default=None)
    q_cmd.add_argument("--error", default=None)
    q_cmd.add_argument("--text", default=None)
    q_cmd.add_argument("--accepted", choices=["true", "false"], default=None)

    g_cmd = sub.add_parser("grep", help="Regex search across uncompressed traces and diffs")
    g_cmd.add_argument("--pattern", required=True)
    g_cmd.add_argument("--generation", type=int, default=None)
    g_cmd.add_argument("--max-hits", type=int, default=50)

    s_cmd = sub.add_parser("show", help="Show full bundle for a generation or run_id")
    s_cmd.add_argument("--id", required=True)

    args = parser.parse_args()
    store = TraceStore(Path(args.repo))

    if args.command == "query":
        acc = None if args.accepted is None else (args.accepted == "true")
        out: Any = store.query(
            generation=args.generation,
            run_id=args.run_id,
            task_id=args.task,
            error_type=args.error,
            text=args.text,
            accepted=acc,
        )
    elif args.command == "grep":
        out = store.grep_traces(
            args.pattern, generation=args.generation, max_hits=args.max_hits
        )
    else:
        target: str | int = int(args.id) if args.id.isdigit() else args.id
        out = store.get_generation_bundle(target)

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
