#!/usr/bin/env python3
"""SoL-Pi Action Fusion wrapper: atomic multi-file edit + lint/test in one round-trip."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import load_rsi_config  # noqa: E402
from evidence_reducer import reduce_log  # noqa: E402
from toolchain import resolve_inner_check  # noqa: E402


def _rollback_backups(backups: dict[Path, str | None]) -> None:
    for path, original in backups.items():
        if original is None:
            if path.exists():
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(original, encoding="utf-8")


def run_fused_edit_and_check(
    repo_root: Path,
    edits: list[dict[str, Any]],
    *,
    check_cmd: str | None = None,
    lint_cmd: str | None = None,
    max_log_chars: int = 3000,
    rollback_on_syntax_error: bool = True,
) -> dict[str, Any]:
    """Apply ``edits`` atomically, syntax-check Python files, and run lint/test in 1 round-trip."""
    repo = repo_root.resolve()
    backups: dict[Path, str | None] = {}
    modified_rel: list[str] = []

    try:
        for edit in edits:
            raw_rel = str(edit.get("path") or "").strip()
            if not raw_rel:
                raise ValueError("Each edit must include a non-empty 'path'")
            target = (repo / raw_rel).resolve(strict=False)
            if not target.is_relative_to(repo):
                raise ValueError(f"Edit path escapes repository root: {raw_rel}")

            if target not in backups:
                backups[target] = target.read_text(encoding="utf-8") if target.is_file() else None

            if "content" in edit:
                new_text = str(edit["content"])
            elif "target" in edit and "replacement" in edit:
                existing = target.read_text(encoding="utf-8") if target.is_file() else ""
                tgt = str(edit["target"])
                if tgt not in existing:
                    raise ValueError(f"Target snippet not found in {raw_rel}")
                new_text = existing.replace(tgt, str(edit["replacement"]))
            else:
                raise ValueError(f"Edit for {raw_rel} requires 'content' or 'target'+'replacement'")

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_text, encoding="utf-8")
            rel_posix = target.relative_to(repo).as_posix()
            if rel_posix not in modified_rel:
                modified_rel.append(rel_posix)
    except Exception:
        _rollback_backups(backups)
        raise

    syntax_errors: list[dict[str, Any]] = []
    for path in backups:
        if path.suffix == ".py" and path.is_file():
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=path.relative_to(repo).as_posix())
            except SyntaxError as exc:
                syntax_errors.append(
                    {
                        "file": path.relative_to(repo).as_posix(),
                        "line": exc.lineno,
                        "message": exc.msg,
                    }
                )

    if syntax_errors and rollback_on_syntax_error:
        _rollback_backups(backups)
        return {
            "ok": False,
            "rolled_back": True,
            "modified_files": modified_rel,
            "syntax_errors": syntax_errors,
            "lint_returncode": None,
            "check_returncode": None,
            "separate_roundtrips": len(edits) + int(bool(lint_cmd)) + 1,
            "fused_roundtrips": 1,
            "roundtrips_saved": 0,
            "estimated_tokens_saved": 0,
            "reduced_output": "",
        }

    cfg = load_rsi_config(repo)
    if check_cmd is None or check_cmd == "auto":
        eff_check_cmd, timeout_sec, _ = resolve_inner_check(repo, cfg)
    else:
        eff_check_cmd = str(check_cmd).strip()
        timeout_sec = float(cfg.get("inner_check_timeout_sec") or 30.0)

    logs: list[str] = []
    lint_rc: int | None = None
    if lint_cmd and str(lint_cmd).strip():
        l_proc = subprocess.run(
            str(lint_cmd),
            cwd=str(repo),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        lint_rc = l_proc.returncode
        logs.append(f"[lint rc={lint_rc}]\n{l_proc.stdout}\n{l_proc.stderr}".strip())

    check_rc: int | None = None
    if eff_check_cmd:
        c_proc = subprocess.run(
            eff_check_cmd,
            cwd=str(repo),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        check_rc = c_proc.returncode
        logs.append(f"[check rc={check_rc}]\n{c_proc.stdout}\n{c_proc.stderr}".strip())

    raw_output = "\n\n".join(part for part in logs if part)
    reduced = reduce_log(raw_output, max_chars=max_log_chars)

    separate_rt = len(edits) + int(lint_rc is not None) + int(check_rc is not None)
    fused_rt = 1
    rt_saved = max(0, separate_rt - fused_rt)
    log_saved_tokens = max(0, (len(raw_output) - len(reduced["reduced_text"])) // 4)
    tokens_saved = rt_saved * 600 + log_saved_tokens

    ok = (not syntax_errors) and (lint_rc in (None, 0)) and (check_rc in (None, 0))
    return {
        "ok": ok,
        "rolled_back": False,
        "modified_files": modified_rel,
        "syntax_errors": syntax_errors,
        "lint_returncode": lint_rc,
        "check_returncode": check_rc,
        "check_cmd": eff_check_cmd,
        "separate_roundtrips": separate_rt,
        "fused_roundtrips": fused_rt,
        "roundtrips_saved": rt_saved,
        "estimated_tokens_saved": tokens_saved,
        "reduced_output": reduced["reduced_text"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute SoL-Pi fused edit + lint/check.")
    parser.add_argument("--repo", default=".", help="Target repository path")
    parser.add_argument("--edits-json", required=True, help="JSON list of edit objects")
    parser.add_argument("--lint-cmd", default=None, help="Optional lint command")
    parser.add_argument("--check-cmd", default="auto", help="Test/check command (or 'auto')")
    parser.add_argument("--max-log-chars", type=int, default=3000)
    args = parser.parse_args()

    edits = json.loads(args.edits_json)
    result = run_fused_edit_and_check(
        Path(args.repo),
        edits,
        check_cmd=args.check_cmd,
        lint_cmd=args.lint_cmd,
        max_log_chars=args.max_log_chars,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
