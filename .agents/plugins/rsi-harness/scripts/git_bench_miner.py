#!/usr/bin/env python3
"""Git-history Fail-to-Pass bugfix commit mining and test transplantation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import load_rsi_config  # noqa: E402
from toolchain import resolve_inner_check  # noqa: E402

_DOC_OR_LOCK_NAMES = {
    "license",
    "readme",
    ".gitignore",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lock",
    "bun.lockb",
    "cargo.lock",
    "poetry.lock",
}
_DOC_EXTS = {".md", ".rst", ".txt"}


def is_test_file(rel_path: str) -> bool:
    """Return True if ``rel_path`` looks like an automated test file across languages."""
    norm = rel_path.replace("\\", "/").strip().lstrip("./")
    if not norm:
        return False
    lower = norm.lower()
    if lower.startswith((".agents/plugins/rsi-harness/", "benchmarks/")):
        return False
    parts = lower.split("/")
    name = parts[-1]
    if any(seg in {"tests", "test", "__tests__"} for seg in parts[:-1]):
        return True
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py") or name.endswith("_test.go") or name.endswith("_test.rs"):
        return True
    if name.endswith(("_test.cpp", "_test.cc")) or (name.startswith("test_") and name.endswith((".cpp", ".cc", ".c"))):
        return True
    if ".test." in name or ".spec." in name:
        return True
    if name.endswith(("test.java", "tests.java")) and name != "test.java":
        return True
    return False


def is_impl_file(rel_path: str) -> bool:
    """Return True if ``rel_path`` is a non-test implementation source file."""
    norm = rel_path.replace("\\", "/").strip().lstrip("./")
    if not norm or is_test_file(norm):
        return False
    lower = norm.lower()
    if lower.startswith((".agents/plugins/rsi-harness/", "benchmarks/", ".git/", ".rsi/")):
        return False
    name = Path(lower).name
    if name in _DOC_OR_LOCK_NAMES:
        return False
    if Path(lower).suffix in _DOC_EXTS:
        return False
    return True


@dataclass
class GitBugCandidate:
    """Candidate bug-fix commit pair containing both implementation and test changes."""

    fix_commit: str
    pre_commit: str
    subject: str
    body: str
    test_files: list[str]
    impl_files: list[str]
    test_patch: str
    check_cmd: str = ""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=check,
    )


def find_bugfix_commit_pairs(
    repo_root: Path,
    *,
    max_commits: int = 50,
) -> list[GitBugCandidate]:
    """Scan recent non-merge commits for pairs modifying both implementation and tests."""
    repo = repo_root.resolve()
    log_proc = _git(
        repo,
        "log",
        f"-n{max(1, int(max_commits))}",
        "--no-merges",
        "--format=%H%x1f%P%x1f%s%x1f%b%x1e",
        check=False,
    )
    if log_proc.returncode != 0:
        return []

    candidates: list[GitBugCandidate] = []
    for raw_rec in log_proc.stdout.split("\x1e"):
        rec = raw_rec.strip("\r\n ")
        if not rec:
            continue
        fields = rec.split("\x1f")
        if len(fields) < 4:
            continue
        fix_sha, parents_str, subject, body = (f.strip() for f in fields[:4])
        parents = [p for p in parents_str.split() if p]
        if not fix_sha or not parents:
            continue
        pre_sha = parents[0]

        changed_proc = _git(repo, "diff", "--name-only", pre_sha, fix_sha, check=False)
        if changed_proc.returncode != 0:
            continue
        changed_paths = [ln.strip() for ln in changed_proc.stdout.splitlines() if ln.strip()]
        test_files = [p for p in changed_paths if is_test_file(p)]
        impl_files = [p for p in changed_paths if is_impl_file(p)]
        if not test_files or not impl_files:
            continue

        patch_proc = _git(repo, "diff", pre_sha, fix_sha, "--", *test_files, check=False)
        if patch_proc.returncode != 0 or not patch_proc.stdout.strip():
            continue

        candidates.append(
            GitBugCandidate(
                fix_commit=fix_sha,
                pre_commit=pre_sha,
                subject=subject,
                body=body,
                test_files=test_files,
                impl_files=impl_files,
                test_patch=patch_proc.stdout,
            )
        )
    return candidates


def _symlink_heavy_deps(repo_root: Path, wt_dir: Path) -> None:
    for dep_dir in ("node_modules", "vendor", ".venv", "venv", "target"):
        src = repo_root / dep_dir
        dst = wt_dir / dep_dir
        if src.exists() and not dst.exists():
            try:
                os.symlink(src.resolve(), dst, target_is_directory=True)
            except OSError:
                pass


def verify_fail_to_pass(
    repo_root: Path,
    candidate: GitBugCandidate,
    *,
    check_cmd: str,
    timeout_sec: int = 30,
) -> tuple[bool, dict[str, Any]]:
    """Verify without any agent that ``pre_commit + test_patch`` fails and ``fix_commit`` passes."""
    repo = repo_root.resolve()
    if not check_cmd.strip():
        return False, {"reason": "empty_check_cmd"}

    with tempfile.TemporaryDirectory(prefix="rsi-git-mine-") as tmp:
        tmp_root = Path(tmp)
        pre_wt = tmp_root / "pre"
        fix_wt = tmp_root / "fix"
        try:
            add_pre = _git(repo, "worktree", "add", "--detach", str(pre_wt), candidate.pre_commit, check=False)
            if add_pre.returncode != 0:
                return False, {"reason": "worktree_pre_failed", "stderr": add_pre.stderr.strip()[:240]}
            _symlink_heavy_deps(repo, pre_wt)

            apply_proc = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=str(pre_wt),
                input=candidate.test_patch,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if apply_proc.returncode != 0:
                return False, {"reason": "test_patch_apply_failed", "stderr": apply_proc.stderr.strip()[:240]}

            try:
                pre_run = subprocess.run(
                    check_cmd,
                    cwd=str(pre_wt),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False, {"reason": "pre_commit_timeout"}
            if pre_run.returncode == 0:
                return False, {"reason": "pre_commit_did_not_fail"}

            add_fix = _git(repo, "worktree", "add", "--detach", str(fix_wt), candidate.fix_commit, check=False)
            if add_fix.returncode != 0:
                return False, {"reason": "worktree_fix_failed", "stderr": add_fix.stderr.strip()[:240]}
            _symlink_heavy_deps(repo, fix_wt)

            try:
                fix_run = subprocess.run(
                    check_cmd,
                    cwd=str(fix_wt),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False, {"reason": "fix_commit_timeout"}
            if fix_run.returncode != 0:
                return False, {
                    "reason": "fix_commit_did_not_pass",
                    "stderr": (fix_run.stderr or fix_run.stdout).strip()[:240],
                }

            return True, {
                "pre_returncode": pre_run.returncode,
                "fix_returncode": fix_run.returncode,
            }
        finally:
            for wt in (pre_wt, fix_wt):
                if wt.exists():
                    _git(repo, "worktree", "remove", "--force", str(wt), check=False)
                    shutil.rmtree(wt, ignore_errors=True)
            _git(repo, "worktree", "prune", check=False)


def _assign_splits(count: int, split_mode: str) -> list[str]:
    norm = (split_mode or "train").strip().lower()
    if norm in {"train", "holdout", "ood"}:
        return [norm] * count
    if norm == "auto":
        splits: list[str] = []
        for idx in range(count):
            if count >= 3 and idx == count - 1:
                splits.append("ood")
            elif idx >= max(1, (count * 3 + 4) // 5):
                splits.append("holdout")
            else:
                splits.append("train")
        return splits
    raise ValueError(f"Unsupported split mode: {split_mode!r}")


def mine_git_regression_suite(
    repo_root: Path,
    *,
    max_commits: int = 50,
    max_tasks: int = 5,
    split: str = "train",
    check_cmd: str | None = None,
    verify: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Synthesize a Harbor-compatible Fail-to-Pass benchmark suite from git commit pairs."""
    repo = repo_root.resolve()
    cfg = load_rsi_config(repo)
    detected_cmd, detected_timeout, _ = resolve_inner_check(repo, cfg)
    eff_cmd = (check_cmd if check_cmd is not None else detected_cmd).strip()
    eff_timeout = max(15, int(detected_timeout))

    candidates = find_bugfix_commit_pairs(repo, max_commits=max_commits)
    selected: list[tuple[GitBugCandidate, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []

    if not eff_cmd and verify:
        suite = {
            "suite_name": f"{repo.name}-git-regression",
            "source": "git_bench_miner",
            "split_mode": split,
            "candidates_scanned": len(candidates),
            "verified_count": 0,
            "rejected": [{"reason": "no_check_cmd_detected"}],
            "tasks": [],
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return suite

    for cand in candidates:
        if len(selected) >= max_tasks:
            break
        if verify:
            ok, diag = verify_fail_to_pass(repo, cand, check_cmd=eff_cmd, timeout_sec=eff_timeout)
            if not ok:
                rejected.append({"fix_commit": cand.fix_commit[:12], **diag})
                continue
        else:
            diag = {"verified": False}
        selected.append((cand, diag))

    splits = _assign_splits(len(selected), split)
    tasks: list[dict[str, Any]] = []
    for idx, ((cand, diag), task_split) in enumerate(zip(selected, splits), start=1):
        prompt_body = f" Context: {cand.body[:280]}" if cand.body else ""
        tasks.append(
            {
                "id": f"git_{idx:02d}_{cand.fix_commit[:8]}",
                "format": "harbor",
                "split": task_split,
                "title": cand.subject or f"Fix bug from commit {cand.fix_commit[:8]}",
                "prompt": (
                    f"Fix the bug described by commit summary: '{cand.subject}'.{prompt_body} "
                    f"Ensure `{eff_cmd or 'repository tests'}` passes cleanly without modifying protected harness or verifier files."
                ),
                "environment": {
                    "type": "git",
                    "git_repo": str(repo),
                    "git_ref": cand.pre_commit,
                },
                "verifier": {
                    "command": eff_cmd,
                    "test_patch": cand.test_patch,
                    "test_files": list(cand.test_files),
                    "timeout_sec": eff_timeout,
                    "readonly": True,
                },
                "git_bug_pair": {
                    "pre_commit": cand.pre_commit,
                    "fix_commit": cand.fix_commit,
                    "impl_files": list(cand.impl_files),
                    "test_files": list(cand.test_files),
                    "verification": diag,
                },
            }
        )

    suite = {
        "suite_name": f"{repo.name}-git-regression",
        "source": "git_bench_miner",
        "split_mode": split,
        "candidates_scanned": len(candidates),
        "verified_count": len(tasks),
        "rejected": rejected,
        "tasks": tasks,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine Fail-to-Pass regression benchmarks from git history.")
    parser.add_argument("--repo", default=".", help="Target git repository path")
    parser.add_argument("--max-commits", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--split", default="train", choices=["train", "holdout", "ood", "auto"])
    parser.add_argument("--check-cmd", default=None)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    suite = mine_git_regression_suite(
        Path(args.repo),
        max_commits=args.max_commits,
        max_tasks=args.max_tasks,
        split=args.split,
        check_cmd=args.check_cmd,
        verify=not args.no_verify,
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps(suite, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
