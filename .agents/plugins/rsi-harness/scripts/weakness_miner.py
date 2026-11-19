#!/usr/bin/env python3
"""Real-session trace clustering (Weakness Miner) and regression suite synthesizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import load_jsonl, load_rsi_config  # noqa: E402
from toolchain import resolve_inner_check  # noqa: E402

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-f]{8,}\b", re.IGNORECASE)
_TMP_PATH_RE = re.compile(r"/tmp/[^\s:'\"]+")
_LINE_COL_RE = re.compile(r"(?:line\s+\d+|:\d+(?::\d+)?)", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+\b")

_CATEGORY_RECOMMENDATIONS = {
    "retry_exhaustion": "Diagnose root cause before retrying edits; verify failing boundary cases once with focused test runs.",
    "stop_failure": "Run targeted verification and fix remaining test or build errors before terminating a subtask.",
    "tool_error_chain": "Inspect tool invocation parameters and preconditions after the first failure to avoid repeated error chains.",
    "high_token_usage": "Use targeted searches, bounded line reads, and concise evidence slices to eliminate redundant token consumption.",
    "oversized_observation": "Offload large outputs into ObservationPack artifacts and page only relevant sections instead of re-reading full logs.",
}


@dataclass
class WeaknessCluster:
    """Ranked cluster of recurring failure or inefficiency modes mined from traces."""

    cluster_id: str
    category: str
    signature: str
    count: int
    severity_score: float
    sample_conversations: list[str] = field(default_factory=list)
    sample_errors: list[str] = field(default_factory=list)
    sample_prompts: list[str] = field(default_factory=list)
    recommended_focus: str = ""


def normalize_error_signature(raw: str) -> str:
    """Normalize volatile paths, line numbers, UUIDs, and numbers into a stable cluster key."""
    text = str(raw or "").strip()
    if not text:
        return "unknown"
    first_line = text.splitlines()[0].strip()
    norm = _UUID_RE.sub("<UUID>", first_line)
    norm = _TMP_PATH_RE.sub("<TMP_PATH>", norm)
    norm = _LINE_COL_RE.sub(":<LINE>", norm)
    norm = _HEX_RE.sub("<HEX>", norm)
    norm = _NUM_RE.sub("<N>", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm[:140]


def _read_int_file(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def mine_weaknesses(
    repo_root: Path,
    *,
    max_inner_retries: int | None = None,
    token_threshold: int = 20000,
    min_cluster_size: int = 1,
) -> list[WeaknessCluster]:
    """Scan ``.rsi/traces/*`` and cluster failure modes and waste patterns."""
    repo = repo_root.resolve()
    cfg = load_rsi_config(repo)
    retry_limit = (
        int(max_inner_retries)
        if max_inner_retries is not None
        else max(1, int(cfg.get("max_inner_retries", 2)))
    )

    traces_root = repo / ".rsi" / "traces"
    if not traces_root.is_dir():
        return []

    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "weight": 0.0,
            "convs": [],
            "errors": [],
            "prompts": [],
        }
    )

    def _add_signal(
        category: str,
        signature: str,
        conv_id: str,
        *,
        weight: float = 1.0,
        error_text: str = "",
        prompt_text: str = "",
    ) -> None:
        b = buckets[(category, signature)]
        b["count"] += 1
        b["weight"] += weight
        if conv_id and conv_id not in b["convs"]:
            b["convs"].append(conv_id)
        if error_text and error_text not in b["errors"] and len(b["errors"]) < 5:
            b["errors"].append(error_text[:240])
        if prompt_text and prompt_text not in b["prompts"] and len(b["prompts"]) < 5:
            b["prompts"].append(prompt_text[:240])

    session_records: list[tuple[str, int, str]] = []

    for conv_dir in sorted(traces_root.iterdir()):
        if not conv_dir.is_dir():
            continue
        conv_id = conv_dir.name
        summary: dict[str, Any] = {}
        sum_file = conv_dir / "summary.json"
        if sum_file.is_file():
            try:
                loaded = json.loads(sum_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    summary = loaded
            except (OSError, json.JSONDecodeError):
                pass

        last_prompt = str(summary.get("lastUserInput") or "").strip()
        total_tokens = int(summary.get("total_tokens") or 0)
        if total_tokens > 0:
            session_records.append((conv_id, total_tokens, last_prompt))

        retries = _read_int_file(conv_dir / "retry_count.txt")
        if retries >= retry_limit:
            _add_signal(
                "retry_exhaustion",
                f"retries>={retry_limit}",
                conv_id,
                weight=3.0 + retries,
                error_text=f"Inner check retries exhausted ({retries}/{retry_limit})",
                prompt_text=last_prompt,
            )

        events = load_jsonl(conv_dir / "events.jsonl")
        consecutive_tool_errs = 0
        oversized_count = 0
        saw_stop_failure = False

        for ev in events:
            ev_type = str(ev.get("event") or "")
            tool_name = str(ev.get("toolName") or "unknown")
            err = ev.get("error")
            term = str(ev.get("terminationReason") or "")

            if ev.get("artifactRef"):
                oversized_count += 1

            if err:
                consecutive_tool_errs += 1
                sig = f"{tool_name}:{normalize_error_signature(str(err))}"
                chain_bonus = 2.0 if consecutive_tool_errs >= 2 else 0.0
                _add_signal(
                    "tool_error_chain",
                    sig,
                    conv_id,
                    weight=2.0 + chain_bonus,
                    error_text=str(err),
                    prompt_text=last_prompt,
                )
            else:
                consecutive_tool_errs = 0

            if ev_type == "stop" and (err or term.lower() in {"error", "failed", "timeout"}):
                saw_stop_failure = True
                sig = normalize_error_signature(str(err or term))
                _add_signal(
                    "stop_failure",
                    sig,
                    conv_id,
                    weight=2.5,
                    error_text=str(err or term),
                    prompt_text=last_prompt,
                )

        if not saw_stop_failure and int(summary.get("errors") or 0) > 0:
            _add_signal(
                "stop_failure",
                "summary_errors_nonzero",
                conv_id,
                weight=1.5,
                error_text=f"Session recorded {summary.get('errors')} error(s)",
                prompt_text=last_prompt,
            )

        if oversized_count >= 2:
            _add_signal(
                "oversized_observation",
                "repeated_oversized_artifacts",
                conv_id,
                weight=1.5 + 0.5 * oversized_count,
                error_text=f"{oversized_count} oversized tool artifacts archived in session",
                prompt_text=last_prompt,
            )

    if session_records:
        all_tokens = [tok for _, tok, _ in session_records]
        med_tokens = statistics.median(all_tokens)
        for conv_id, tokens, last_prompt in session_records:
            is_outlier = len(all_tokens) >= 3 and tokens > max(5000, 2.0 * med_tokens)
            if tokens >= token_threshold or is_outlier:
                sig = "outlier_vs_median" if is_outlier and tokens < token_threshold else f"tokens>={token_threshold}"
                _add_signal(
                    "high_token_usage",
                    sig,
                    conv_id,
                    weight=2.0 + min(3.0, tokens / max(1, token_threshold)),
                    error_text=f"High token consumption: {tokens} tokens (median={int(med_tokens)})",
                    prompt_text=last_prompt,
                )

    clusters: list[WeaknessCluster] = []
    for (category, signature), data in buckets.items():
        if data["count"] < min_cluster_size:
            continue
        cid = hashlib.sha256(f"{category}:{signature}".encode("utf-8")).hexdigest()[:10]
        clusters.append(
            WeaknessCluster(
                cluster_id=cid,
                category=category,
                signature=signature,
                count=int(data["count"]),
                severity_score=round(float(data["weight"]), 3),
                sample_conversations=list(data["convs"]),
                sample_errors=list(data["errors"]),
                sample_prompts=list(data["prompts"]),
                recommended_focus=_CATEGORY_RECOMMENDATIONS.get(
                    category, "Prevent recurring session failure mode."
                ),
            )
        )

    clusters.sort(key=lambda c: (-c.severity_score, -c.count, c.category, c.signature))
    return clusters


def synthesize_regression_suite(
    repo_root: Path,
    clusters: list[WeaknessCluster],
    *,
    max_tasks: int = 5,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Synthesize a Harbor-compatible regression suite from mined weakness clusters."""
    repo = repo_root.resolve()
    cfg = load_rsi_config(repo)
    check_cmd, timeout_sec, _ = resolve_inner_check(repo, cfg)
    verifier_cmd = check_cmd or 'python3 -c "print(\'ok\')"'
    verifier_timeout = max(15, int(timeout_sec))

    tasks: list[dict[str, Any]] = []
    for idx, cluster in enumerate(clusters[:max_tasks], start=1):
        sample_prompt = (
            f" Representative prompt: {cluster.sample_prompts[0]}"
            if cluster.sample_prompts
            else ""
        )
        sample_err = (
            f" Observed symptom: {cluster.sample_errors[0]}"
            if cluster.sample_errors
            else ""
        )
        tasks.append(
            {
                "id": f"mined_{idx:02d}_{cluster.category}",
                "format": "harbor",
                "split": "train",
                "title": f"Mitigate mined {cluster.category} ({cluster.signature})",
                "prompt": (
                    f"Address recurring session weakness '{cluster.category}' "
                    f"[{cluster.signature}]. {cluster.recommended_focus}"
                    f"{sample_prompt}{sample_err} Ensure `{verifier_cmd}` passes cleanly."
                ),
                "environment": {
                    "type": "directory",
                    "source_dir": str(repo),
                },
                "verifier": {
                    "command": verifier_cmd,
                    "timeout_sec": verifier_timeout,
                },
                "weakness_cluster": asdict(cluster),
            }
        )

    suite = {
        "suite_name": f"{repo.name}-mined-regression",
        "source": "weakness_miner",
        "tasks": tasks,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster .rsi/traces weaknesses and synthesize regression tasks.")
    parser.add_argument("--repo", default=".", help="Target repository path")
    parser.add_argument("--token-threshold", type=int, default=20000)
    parser.add_argument("--min-cluster-size", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--synthesize-out", default=None, help="Optional JSON output path for mined suite")
    args = parser.parse_args()

    repo = Path(args.repo)
    clusters = mine_weaknesses(
        repo,
        token_threshold=args.token_threshold,
        min_cluster_size=args.min_cluster_size,
    )
    out: dict[str, Any] = {"clusters": [asdict(c) for c in clusters]}
    if args.synthesize_out:
        out["suite"] = synthesize_regression_suite(
            repo,
            clusters,
            max_tasks=args.max_tasks,
            output_path=Path(args.synthesize_out),
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
