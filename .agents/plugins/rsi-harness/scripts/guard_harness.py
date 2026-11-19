#!/usr/bin/env python3
"""PreToolUse hook protecting harness code and evaluation benchmarks."""

from __future__ import annotations

import ast
import os
import re
import shlex
from pathlib import Path
from typing import Any

from adapters.events import (
    NormalizedToolPreEvent,
    normalize_pre_tool_event,
    serialize_hook_response,
)
from common import load_rsi_config, read_hook_input, write_hook_output

ALWAYS_PROTECTED = (
    ".agents/plugins/rsi-harness/hooks.json",
    ".agents/plugins/rsi-harness/scripts/",
    "benchmarks/",
)

_PATH_ARG_KEYS = ("Path", "TargetFile", "path", "file_path")
_CMD_ARG_KEYS = ("CommandLine", "command")

_OUTPUT_REDIRECT_OPS = {">", ">>", ">|", "&>", "&>>", "<>", ">&"}
_INPUT_REDIRECT_OPS = {"<", "<<", "<<<", "<<-", "<&"}
_SAFE_DEVICE_PATHS = {"/dev/null", "/dev/stdout", "/dev/stderr"}

_READ_ONLY_CMDS = {
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "grep",
    "rg",
    "egrep",
    "fgrep",
    "ls",
    "ll",
    "dir",
    "stat",
    "file",
    "wc",
    "diff",
    "cmp",
    "pwd",
    "echo",
    "printf",
    "which",
    "whereis",
    "type",
    "readlink",
    "realpath",
    "sha256sum",
    "sha1sum",
    "md5sum",
    "cksum",
    "sort",
    "uniq",
    "cut",
    "tr",
    "jq",
    "yq",
    "test",
    "[",
    "true",
    "false",
    "basename",
    "dirname",
    "date",
    "uname",
    "id",
    "whoami",
    "printenv",
    "tree",
    "du",
    "df",
    "ag",
    "fd",
    "pytest",
    "unittest",
}

_MUTATING_ALL_ARGS_CMDS = {"rm", "unlink", "rmdir", "shred", "mv"}
_READ_ONLY_GIT_SUBCMDS = {
    "status",
    "diff",
    "log",
    "show",
    "rev-parse",
    "ls-files",
    "ls-tree",
    "grep",
    "describe",
    "remote",
    "blame",
    "shortlog",
    "rev-list",
    "cat-file",
    "check-ignore",
}
_MUTATING_GIT_SUBCMDS = {"checkout", "restore", "reset", "clean", "rm", "mv", "apply", "am"}
_INLINE_WRITE_MARKERS = (
    "write_text",
    "write_bytes",
    "unlink",
    "remove(",
    "rmdir(",
    "rename(",
    "replace(",
    "rmtree(",
    "copy(",
    "copy2(",
    "copytree(",
    "move(",
    "symlink_to",
    "hardlink_to",
    "writelines",
    "fs.write",
    "fs.append",
    "fs.rm",
    "fs.unlink",
    "fs.rename",
    "fs.copy",
    "fs.mkdir",
    "os.system",
    "subprocess.",
)
_QUOTED_STR_RE = re.compile(r"""(['"])((?:\\.|(?!\1).)+)\1""")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _extract_candidate_targets(args: dict[str, Any]) -> list[str]:
    """Extract path or command strings from tool call arguments."""
    targets: list[str] = []
    for key in _PATH_ARG_KEYS + _CMD_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            targets.append(val)
    return targets


def _build_path_views(raw_path: str, cwd: Path) -> tuple[Path, Path]:
    """Return ``(lexical_norm_path, resolved_path)`` for ``raw_path`` relative to ``cwd``."""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        lex_base = candidate
        res_base = candidate
    else:
        lex_base = Path(os.path.abspath(str(cwd))) / candidate
        res_base = cwd.resolve(strict=False) / candidate
    lexical = Path(os.path.normpath(str(lex_base)))
    resolved = res_base.resolve(strict=False)
    return lexical, resolved


def _strip_dot_slash(raw: str) -> str:
    """Strip leading ``./`` segments without stripping leading dots from dotfiles."""
    cleaned = raw.replace("\\", "/").strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


def _normalize_target(raw: str, cwd: str) -> str:
    """Normalize a path-like string relative to ``cwd`` when possible."""
    cwd_path = Path(cwd)
    _, resolved = _build_path_views(raw, cwd_path)
    cwd_resolved = cwd_path.resolve(strict=False)
    if resolved.is_relative_to(cwd_resolved):
        return resolved.relative_to(cwd_resolved).as_posix()
    return _strip_dot_slash(raw)


def _is_within_root(raw_path: str, allowed_root: Path, cwd: str) -> bool:
    """Return True if ``raw_path`` resolves strictly inside ``allowed_root``."""
    try:
        lexical, resolved = _build_path_views(raw_path, Path(cwd))
        allowed_res = allowed_root.resolve(strict=False)
        allowed_lex = Path(os.path.normpath(os.path.abspath(str(allowed_root))))
    except (ValueError, OSError):
        return False
    if not resolved.is_relative_to(allowed_res):
        return False
    return lexical.is_relative_to(allowed_lex) or lexical.is_relative_to(allowed_res)


def _build_protected_paths(
    cwd: Path, prefixes: tuple[str, ...]
) -> list[tuple[str, Path, Path]]:
    """Build ``(clean_prefix, lexical_path, resolved_path)`` tuples for protected prefixes."""
    cwd_abs = Path(os.path.abspath(str(cwd)))
    cwd_res = cwd.resolve(strict=False)
    result: list[tuple[str, Path, Path]] = []
    for prefix in prefixes:
        clean = _strip_dot_slash(prefix).rstrip("/")
        if not clean or clean == ".":
            continue
        lex = Path(os.path.normpath(str(cwd_abs / clean)))
        res = (cwd_res / clean).resolve(strict=False)
        result.append((clean, lex, res))
    return result


def _check_path_against_boundaries(
    raw_path: str,
    cwd: Path,
    protected_paths: list[tuple[str, Path, Path]],
    allowed_root: Path | None,
    evo_mode: bool,
    *,
    allow_safe_devices: bool = False,
) -> dict[str, Any] | None:
    """Return a guard decision dict if ``raw_path`` violates boundaries, else ``None``."""
    stripped = raw_path.strip()
    if not stripped:
        return None
    if allow_safe_devices and stripped in _SAFE_DEVICE_PATHS:
        return None

    try:
        lexical, resolved = _build_path_views(stripped, cwd)
    except (ValueError, OSError):
        return {
            "decision": "deny" if evo_mode else "ask",
            "reason": f"Failed to safely resolve path boundary for: {raw_path}",
        }

    if evo_mode and allowed_root is not None:
        if not _is_within_root(stripped, allowed_root, str(cwd)):
            return {
                "decision": "deny",
                "reason": (
                    f"Evolution mode restricts edits strictly inside {allowed_root}. "
                    f"Refusing path outside worktree: {raw_path}"
                ),
            }

    for clean_prefix, prot_lex, prot_res in protected_paths:
        if resolved.is_relative_to(prot_res) or lexical.is_relative_to(prot_lex):
            return {
                "decision": "deny" if evo_mode else "ask",
                "reason": (
                    f"Protected RSI harness/benchmark target matched ({clean_prefix}). "
                    "Approve explicitly or use Outer Loop evolution mode."
                ),
            }
    return None


def _extract_command_substitutions(command: str) -> list[str]:
    """Extract inner snippets from unescaped ``$(...)`` and backtick substitutions."""
    subs: list[str] = []
    n = len(command)
    i = 0
    in_single = False
    in_double = False
    while i < n:
        ch = command[i]
        if ch == "\\" and not in_single and i + 1 < n:
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if not in_single:
            if ch == "$" and i + 1 < n and command[i + 1] == "(":
                depth = 1
                j = i + 2
                inner_sq = False
                inner_dq = False
                while j < n and depth > 0:
                    c2 = command[j]
                    if c2 == "\\" and not inner_sq and j + 1 < n:
                        j += 2
                        continue
                    if c2 == "'" and not inner_dq:
                        inner_sq = not inner_sq
                    elif c2 == '"' and not inner_sq:
                        inner_dq = not inner_dq
                    elif not inner_sq and not inner_dq:
                        if c2 == "(":
                            depth += 1
                        elif c2 == ")":
                            depth -= 1
                    j += 1
                if depth == 0:
                    subs.append(command[i + 2 : j - 1])
                    i = j
                    continue
            if ch == "`":
                j = i + 1
                while j < n:
                    if command[j] == "\\" and j + 1 < n:
                        j += 2
                        continue
                    if command[j] == "`":
                        break
                    j += 1
                if j < n:
                    subs.append(command[i + 1 : j])
                    i = j + 1
                    continue
        i += 1
    return subs


def _analyze_inline_code(code: str) -> tuple[list[str], bool]:
    """Inspect inline Python/JS/Perl code for filesystem write indicators and literal paths."""
    has_open_write = "open(" in code and (
        any(m in code for m in ("'w", '"w', "'a", '"a', "'x", '"x', "+'", '+"', ".write", "O_WRONLY", "O_RDWR", "O_CREAT"))
        or bool(re.search(r"""mode\s*=\s*['"][^'"]*[wax+]""", code))
    )
    has_write = has_open_write or any(marker in code for marker in _INLINE_WRITE_MARKERS)
    if not has_write:
        return [], False

    candidates: list[str] = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidates.append(node.value)
    except SyntaxError:
        for _, body in _QUOTED_STR_RE.findall(code):
            candidates.append(body)

    path_targets = [
        item
        for item in candidates
        if ("/" in item or "\\" in item or item.endswith((".py", ".json", ".md", ".sh", ".yaml", ".yml")))
    ]
    return path_targets, True


def _strip_cmd_wrappers(seg: list[str]) -> list[str]:
    """Strip leading env assignments and wrapper binaries from a command segment."""
    idx = 0
    while idx < len(seg):
        tok = seg[idx]
        if _ENV_ASSIGN_RE.match(tok):
            idx += 1
            continue
        cmd_name = Path(tok).name
        if cmd_name in {"command", "builtin", "nohup"}:
            idx += 1
            continue
        if cmd_name == "env":
            idx += 1
            while idx < len(seg):
                t = seg[idx]
                if t in {"-u", "-C", "--chdir", "--unset"} and idx + 1 < len(seg):
                    idx += 2
                elif t.startswith("-") or _ENV_ASSIGN_RE.match(t):
                    idx += 1
                else:
                    break
            continue
        if cmd_name in {"sudo", "nice", "timeout"}:
            skip_next_flags = {"-u", "-g", "-n", "-k", "-s"}
            idx += 1
            if cmd_name == "timeout":
                while idx < len(seg) and seg[idx].startswith("-"):
                    if seg[idx] in skip_next_flags and idx + 1 < len(seg):
                        idx += 2
                    else:
                        idx += 1
                if idx < len(seg):
                    idx += 1  # duration argument
                continue
            while idx < len(seg) and seg[idx].startswith("-"):
                if seg[idx] in skip_next_flags and idx + 1 < len(seg):
                    idx += 2
                else:
                    idx += 1
            continue
        break
    return seg[idx:]


def _positional_args(
    args: list[str], value_flags: set[str] | None = None
) -> tuple[list[str], list[str]]:
    """Return ``(positional_args, flagged_value_targets)`` respecting ``--``."""
    value_flags = value_flags or set()
    positional: list[str] = []
    flagged: list[str] = []
    after_dashdash = False
    i = 0
    while i < len(args):
        tok = args[i]
        if after_dashdash:
            positional.append(tok)
            i += 1
            continue
        if tok == "--":
            after_dashdash = True
            i += 1
            continue
        if tok.startswith("--target-directory="):
            flagged.append(tok.split("=", 1)[1])
            i += 1
            continue
        if tok in value_flags and i + 1 < len(args):
            if tok in {"-t", "--target-directory"}:
                flagged.append(args[i + 1])
            i += 2
            continue
        if tok.startswith("-") and tok != "-":
            i += 1
            continue
        positional.append(tok)
        i += 1
    return positional, flagged


def _analyze_sed_args(args: list[str]) -> list[str]:
    """Return write targets only when ``sed`` is invoked with ``-i`` / ``--in-place``."""
    in_place = False
    has_expr_flag = False
    positional: list[str] = []
    after_dashdash = False
    i = 0
    while i < len(args):
        tok = args[i]
        if after_dashdash:
            positional.append(tok)
            i += 1
            continue
        if tok == "--":
            after_dashdash = True
            i += 1
            continue
        if tok == "--in-place" or tok.startswith("--in-place="):
            in_place = True
            i += 1
            continue
        if tok.startswith("-") and not tok.startswith("--"):
            flags = tok[1:]
            if "i" in flags:
                in_place = True
            if tok in {"-e", "-f"} and i + 1 < len(args):
                has_expr_flag = True
                i += 2
                continue
            i += 1
            continue
        if tok in {"--expression", "--file"} and i + 1 < len(args):
            has_expr_flag = True
            i += 2
            continue
        if tok.startswith(("--expression=", "--file=")):
            has_expr_flag = True
            i += 1
            continue
        if tok.startswith("--"):
            i += 1
            continue
        positional.append(tok)
        i += 1

    if not in_place:
        return []
    return positional if has_expr_flag else positional[1:]


def _analyze_git_args(args: list[str]) -> list[str]:
    """Extract write targets from mutating ``git`` subcommands."""
    base_dir = ""
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in {"-C", "-c", "--git-dir", "--work-tree"} and i + 1 < len(args):
            if tok == "-C":
                base_dir = args[i + 1]
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        break

    if i >= len(args):
        return []
    subcmd = args[i]
    rest = args[i + 1 :]
    if subcmd in _READ_ONLY_GIT_SUBCMDS:
        return []
    if subcmd == "branch" and not any(f in rest for f in ("-D", "-d", "-m", "-M", "-f")):
        return []
    if subcmd not in _MUTATING_GIT_SUBCMDS:
        return []

    if "--" in rest:
        idx = rest.index("--")
        paths = rest[idx + 1 :]
    else:
        paths, _ = _positional_args(rest)
        if subcmd in {"checkout", "restore", "reset"} and len(paths) > 1:
            paths = paths[1:]

    if base_dir:
        return [str(Path(base_dir) / p) for p in paths]
    return paths


def _analyze_command_segment(seg: list[str], depth: int) -> tuple[list[str], bool]:
    """Return ``(write_targets, suspicious)`` for a single command segment."""
    clean = _strip_cmd_wrappers(seg)
    if not clean:
        return [], False

    cmd = Path(clean[0]).name
    args = clean[1:]
    targets: list[str] = []
    suspicious = False

    if cmd in _READ_ONLY_CMDS:
        return [], False

    if cmd in {"sh", "bash", "zsh", "dash", "ksh"}:
        for i, tok in enumerate(args):
            if tok.startswith("-") and not tok.startswith("--") and "c" in tok[1:] and i + 1 < len(args):
                sub_targets, sub_susp = _extract_shell_write_targets(args[i + 1], _depth=depth + 1)
                targets.extend(sub_targets)
                suspicious = suspicious or sub_susp
        return targets, suspicious

    if cmd in {"eval", "source", "."}:
        if cmd == "eval" and args:
            sub_targets, sub_susp = _extract_shell_write_targets(" ".join(args), _depth=depth + 1)
            return sub_targets, sub_susp or True
        return [], True

    if cmd == "tee":
        pos, _ = _positional_args(args)
        return pos, False

    if cmd in _MUTATING_ALL_ARGS_CMDS:
        pos, flagged = _positional_args(args, value_flags={"-S", "--suffix", "-t", "--target-directory"})
        return pos + flagged, False

    if cmd in {"touch", "truncate", "mkdir", "chmod", "chown", "chgrp"}:
        pos, _ = _positional_args(args, value_flags={"-d", "-r", "-t", "-s", "-m"})
        if cmd in {"chmod", "chown", "chgrp"} and len(pos) > 1:
            return pos[1:], False
        return pos, False

    if cmd in {"cp", "install", "ln"}:
        pos, flagged = _positional_args(
            args, value_flags={"-t", "--target-directory", "-S", "--suffix", "-m", "-o", "-g"}
        )
        if flagged:
            return flagged, False
        return ([pos[-1]] if pos else []), False

    if cmd == "sed":
        return _analyze_sed_args(args), False

    if cmd in {"awk", "gawk", "mawk"}:
        if any(a == "--inplace" or a.startswith("-i") for a in args):
            pos, _ = _positional_args(args, value_flags={"-F", "-v", "-f"})
            return (pos[1:] if len(pos) > 1 else pos), False
        return [], False

    if cmd == "dd":
        return [a[3:] for a in args if a.startswith("of=")], False

    if cmd == "find":
        mutating_flags = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fls", "-fprint", "-fprint0", "-fprintf"}
        if any(a in mutating_flags for a in args):
            for idx, a in enumerate(args):
                if a in {"-fls", "-fprint", "-fprint0", "-fprintf"} and idx + 1 < len(args):
                    targets.append(args[idx + 1])
                elif not a.startswith("-") and a not in {"(", ")", "!", ";", "+"}:
                    targets.append(a)
                elif a in {"-exec", "-execdir"}:
                    sub_seg: list[str] = []
                    j = idx + 1
                    while j < len(args) and args[j] not in {";", "+"}:
                        if args[j] != "{}":
                            sub_seg.append(args[j])
                        j += 1
                    sub_t, sub_s = _analyze_command_segment(sub_seg, depth + 1)
                    targets.extend(sub_t)
                    suspicious = suspicious or sub_s
                    break
            return targets, suspicious
        return [], False

    if cmd == "git":
        return _analyze_git_args(args), False

    if cmd.startswith("python") or cmd in {"node", "nodejs", "perl", "ruby"}:
        inline_codes: list[str] = []
        in_place = False
        for i, tok in enumerate(args):
            if tok in {"-c", "-e", "--eval"} and i + 1 < len(args):
                inline_codes.append(args[i + 1])
            elif tok.startswith("-i") or tok == "--in-place":
                in_place = True
        if in_place:
            pos, _ = _positional_args(args, value_flags={"-c", "-e", "--eval"})
            targets.extend(pos)
        for code in inline_codes:
            code_targets, code_susp = _analyze_inline_code(code)
            targets.extend(code_targets)
            suspicious = suspicious or code_susp
        return targets, suspicious

    return targets, suspicious


def _extract_shell_write_targets(command: str, _depth: int = 0) -> tuple[list[str], bool]:
    """Parse ``command`` with ``shlex`` and return ``(write_targets, suspicious)``."""
    if _depth > 5 or not command.strip():
        return [], False

    write_targets: list[str] = []
    suspicious = False

    for sub in _extract_command_substitutions(command):
        sub_targets, sub_susp = _extract_shell_write_targets(sub, _depth=_depth + 1)
        write_targets.extend(sub_targets)
        suspicious = suspicious or sub_susp

    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return write_targets, True

    segments: list[list[str]] = []
    current_seg: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _OUTPUT_REDIRECT_OPS:
            if current_seg and current_seg[-1].isdigit():
                current_seg.pop()
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if not (tok == ">&" and (nxt.isdigit() or nxt == "-")):
                    write_targets.append(nxt)
                i += 2
                continue
            i += 1
            continue
        if tok in _INPUT_REDIRECT_OPS:
            if current_seg and current_seg[-1].isdigit():
                current_seg.pop()
            i += 2 if i + 1 < len(tokens) else 1
            continue
        if all(ch in ";&|\n" for ch in tok):
            if current_seg:
                segments.append(current_seg)
                current_seg = []
            i += 1
            continue
        current_seg.append(tok)
        i += 1

    if current_seg:
        segments.append(current_seg)

    for seg in segments:
        seg_targets, seg_susp = _analyze_command_segment(seg, _depth)
        write_targets.extend(seg_targets)
        suspicious = suspicious or seg_susp

    return write_targets, suspicious


def evaluate_normalized_tool_guard(event: NormalizedToolPreEvent) -> dict[str, Any]:
    """Evaluate PreToolUse boundary rules from a runtime-agnostic ``NormalizedToolPreEvent``."""
    cwd_path = Path(event.cwd or ".")
    config = load_rsi_config(cwd_path)
    protected_prefixes = (
        tuple(config.get("rrsi", {}).get("protected_prefixes", [])) + ALWAYS_PROTECTED
    )
    protected_paths = _build_protected_paths(cwd_path, protected_prefixes)

    evo_mode = os.environ.get("RSI_EVOLUTION_MODE") == "1"
    allowed_root_raw = os.environ.get("RSI_ALLOWED_ROOT", "").strip()
    allowed_root = Path(allowed_root_raw) if (evo_mode and allowed_root_raw) else None

    for val in event.file_paths:
        decision = _check_path_against_boundaries(
            val, cwd_path, protected_paths, allowed_root, evo_mode
        )
        if decision is not None:
            return decision

    for cmd_val in event.commands:
        write_targets, suspicious = _extract_shell_write_targets(cmd_val)
        for target in write_targets:
            decision = _check_path_against_boundaries(
                target,
                cwd_path,
                protected_paths,
                allowed_root,
                evo_mode,
                allow_safe_devices=True,
            )
            if decision is not None:
                return decision
        if suspicious:
            normalized_cmd = cmd_val.replace("\\", "/")
            for clean_prefix, _, _ in protected_paths:
                if clean_prefix in normalized_cmd:
                    return {
                        "decision": "deny" if evo_mode else "ask",
                        "reason": (
                            f"Suspicious shell command referencing protected target ({clean_prefix}). "
                            "Approve explicitly or use Outer Loop evolution mode."
                        ),
                    }

    return {"decision": "allow"}


def evaluate_tool_guard(payload: dict[str, Any]) -> dict[str, Any]:
    """Return runtime-formatted PreToolUse decision dict for ``payload``."""
    event = normalize_pre_tool_event(payload)
    raw_decision = evaluate_normalized_tool_guard(event)
    return serialize_hook_response(event.runtime, "pre_tool", raw_decision)


def main() -> None:
    payload = read_hook_input()
    write_hook_output(evaluate_tool_guard(payload))


if __name__ == "__main__":
    main()
