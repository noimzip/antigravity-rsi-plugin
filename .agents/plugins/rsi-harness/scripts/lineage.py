#!/usr/bin/env python3
"""DGM-style open-ended lineage archive, Pareto frontier, and diversity sampling."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LineageNode:
    """Single evolutionary candidate node in the lineage DAG."""

    node_id: str
    parent_id: str | None
    commit_sha: str
    branch: str
    generation: int
    pass_rate: float = 0.0
    total_tokens: float = 0.0
    latency_ms: float = 0.0
    diff_hash: str = ""
    changed_files: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)
    novelty_score: float = 1.0
    created_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageNode:
        return cls(
            node_id=str(data["node_id"]),
            parent_id=data.get("parent_id"),
            commit_sha=str(data.get("commit_sha", "")),
            branch=str(data.get("branch", "")),
            generation=int(data.get("generation", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            total_tokens=float(data.get("total_tokens", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            diff_hash=str(data.get("diff_hash", "")),
            changed_files=[str(x) for x in data.get("changed_files", [])],
            children_ids=[str(x) for x in data.get("children_ids", [])],
            novelty_score=float(data.get("novelty_score", 1.0)),
            created_at=str(data.get("created_at") or _utc_now_iso()),
            metadata=dict(data.get("metadata") or {}),
        )


def _resolve_archive_path(path_or_repo: Path) -> Path:
    path = Path(path_or_repo)
    if path.suffix == ".json" or path.name == "lineage.json":
        return path
    return path / ".rsi" / "lineage.json"


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


class LineageArchive:
    """Persistent evolutionary DAG with Pareto frontier and DGM parent sampling."""

    def __init__(
        self,
        archive_path: Path | None = None,
        nodes: dict[str, LineageNode] | None = None,
        root_id: str | None = None,
    ) -> None:
        self.archive_path = archive_path or Path(".rsi/lineage.json")
        self.nodes: dict[str, LineageNode] = dict(nodes or {})
        self.root_id = root_id

    @classmethod
    def load(cls, path_or_repo: Path) -> LineageArchive:
        archive_path = _resolve_archive_path(path_or_repo)
        if not archive_path.is_file():
            return cls(archive_path=archive_path)
        raw = json.loads(archive_path.read_text(encoding="utf-8"))
        nodes: dict[str, LineageNode] = {}
        raw_nodes = raw.get("nodes", {})
        if isinstance(raw_nodes, dict):
            for key, item in raw_nodes.items():
                if isinstance(item, dict):
                    node = LineageNode.from_dict(item)
                    nodes[node.node_id or str(key)] = node
        elif isinstance(raw_nodes, list):
            for item in raw_nodes:
                if isinstance(item, dict):
                    node = LineageNode.from_dict(item)
                    nodes[node.node_id] = node
        root_id = raw.get("root_id")
        if root_id is None and "root" in nodes:
            root_id = "root"
        archive = cls(archive_path=archive_path, nodes=nodes, root_id=root_id)
        if archive.nodes:
            archive.recompute_novelty()
        return archive

    def save(self, path: Path | None = None) -> Path:
        target = _resolve_archive_path(path) if path is not None else self.archive_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "root_id": self.root_id,
            "updated_at": _utc_now_iso(),
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        self.archive_path = target
        return target

    def get_node(self, node_id: str) -> LineageNode | None:
        return self.nodes.get(node_id)

    def ensure_root(
        self,
        commit_sha: str,
        branch: str = "main",
        pass_rate: float = 0.0,
        total_tokens: float = 0.0,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> LineageNode:
        if self.root_id and self.root_id in self.nodes:
            root = self.nodes[self.root_id]
            if not root.commit_sha and commit_sha:
                root.commit_sha = commit_sha
            if root.metadata.get("metrics_initialized") is not True and (
                pass_rate > 0 or total_tokens > 0 or latency_ms > 0
            ):
                root.pass_rate = float(pass_rate)
                root.total_tokens = float(total_tokens)
                root.latency_ms = float(latency_ms)
                root.metadata["metrics_initialized"] = True
            return root

        root_id = self.root_id or "root"
        root_meta = dict(metadata or {})
        if pass_rate > 0 or total_tokens > 0 or latency_ms > 0:
            root_meta.setdefault("metrics_initialized", True)
        root = LineageNode(
            node_id=root_id,
            parent_id=None,
            commit_sha=commit_sha,
            branch=branch,
            generation=0,
            pass_rate=float(pass_rate),
            total_tokens=float(total_tokens),
            latency_ms=float(latency_ms),
            diff_hash="",
            changed_files=[],
            children_ids=[],
            novelty_score=1.0,
            metadata=root_meta,
        )
        self.nodes[root_id] = root
        self.root_id = root_id
        self.recompute_novelty()
        return root

    def add_node(
        self,
        node: LineageNode | None = None,
        *,
        node_id: str | None = None,
        parent_id: str | None = None,
        commit_sha: str = "",
        branch: str = "",
        generation: int | None = None,
        pass_rate: float = 0.0,
        total_tokens: float = 0.0,
        latency_ms: float = 0.0,
        diff_hash: str = "",
        changed_files: list[str] | None = None,
        novelty_score: float = 1.0,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageNode:
        if node is None:
            resolved_id = node_id or f"node-{len(self.nodes) + 1:04d}"
            parent = self.nodes.get(parent_id) if parent_id is not None else None
            resolved_gen = generation if generation is not None else (parent.generation + 1 if parent else 0)
            node = LineageNode(
                node_id=resolved_id,
                parent_id=parent_id,
                commit_sha=commit_sha,
                branch=branch,
                generation=resolved_gen,
                pass_rate=float(pass_rate),
                total_tokens=float(total_tokens),
                latency_ms=float(latency_ms),
                diff_hash=diff_hash,
                changed_files=list(changed_files or []),
                children_ids=[],
                novelty_score=float(novelty_score),
                created_at=created_at or _utc_now_iso(),
                metadata=dict(metadata or {}),
            )

        if node.parent_id and node.parent_id in self.nodes:
            parent_node = self.nodes[node.parent_id]
            if node.node_id not in parent_node.children_ids:
                parent_node.children_ids.append(node.node_id)
            if generation is None and node.generation == 0:
                node.generation = parent_node.generation + 1

        if self.root_id is None and node.parent_id is None:
            self.root_id = node.node_id

        self.nodes[node.node_id] = node
        self.recompute_novelty()
        return node

    @staticmethod
    def dominates(a: LineageNode, b: LineageNode) -> bool:
        """Return True when node `a` Pareto-dominates node `b`."""
        no_worse = (
            a.pass_rate >= b.pass_rate
            and a.total_tokens <= b.total_tokens
            and a.latency_ms <= b.latency_ms
        )
        strictly_better = (
            a.pass_rate > b.pass_rate
            or a.total_tokens < b.total_tokens
            or a.latency_ms < b.latency_ms
        )
        return no_worse and strictly_better

    def pareto_frontier(self) -> list[LineageNode]:
        """Return all non-dominated nodes across (pass_rate, -tokens, -latency)."""
        all_nodes = list(self.nodes.values())
        frontier: list[LineageNode] = []
        for candidate in all_nodes:
            if any(
                other.node_id != candidate.node_id and self.dominates(other, candidate)
                for other in all_nodes
            ):
                continue
            frontier.append(candidate)
        frontier.sort(
            key=lambda n: (-n.pass_rate, n.total_tokens, n.latency_ms, n.generation, n.node_id)
        )
        return frontier

    def recompute_novelty(self) -> None:
        """Recompute diversity/novelty scores across the archive."""
        nodes = list(self.nodes.values())
        if not nodes:
            return
        if len(nodes) == 1:
            nodes[0].novelty_score = 1.0
            return

        diff_counts: dict[str, int] = {}
        for n in nodes:
            if n.diff_hash:
                diff_counts[n.diff_hash] = diff_counts.get(n.diff_hash, 0) + 1

        max_tokens = max((n.total_tokens for n in nodes), default=1.0) or 1.0
        max_latency = max((n.latency_ms for n in nodes), default=1.0) or 1.0

        for n in nodes:
            files_n = set(n.changed_files)
            nearest_dist = 1.0
            total_dist = 0.0
            peers = 0
            for other in nodes:
                if other.node_id == n.node_id:
                    continue
                peers += 1
                files_o = set(other.changed_files)
                if files_n or files_o:
                    file_dist = _jaccard_distance(files_n, files_o)
                elif n.diff_hash and other.diff_hash and n.diff_hash == other.diff_hash:
                    file_dist = 0.0
                else:
                    file_dist = 0.5
                metric_dist = (
                    abs(n.pass_rate - other.pass_rate)
                    + abs(n.total_tokens - other.total_tokens) / max_tokens
                    + abs(n.latency_ms - other.latency_ms) / max_latency
                ) / 3.0
                pair_dist = 0.7 * file_dist + 0.3 * metric_dist
                nearest_dist = min(nearest_dist, pair_dist)
                total_dist += pair_dist

            avg_dist = total_dist / max(peers, 1)
            hash_uniqueness = 1.0 / diff_counts.get(n.diff_hash, 1) if n.diff_hash else 0.5
            score = 0.5 * nearest_dist + 0.3 * avg_dist + 0.2 * hash_uniqueness
            n.novelty_score = round(max(0.0, min(1.0, score)), 4)

    def sample_parent(self, strategy: str = "dgm", seed: int | None = None) -> LineageNode | None:
        """Select a parent node using `dgm`, `best`, or `latest` strategy."""
        if not self.nodes:
            return None

        ordered = list(self.nodes.values())
        norm_strategy = (strategy or "dgm").strip().lower()

        if norm_strategy == "latest":
            return max(
                enumerate(ordered),
                key=lambda item: (item[1].generation, item[1].created_at, item[0]),
            )[1]

        if norm_strategy == "best":
            return min(
                ordered,
                key=lambda n: (-n.pass_rate, n.total_tokens, n.latency_ms, -n.generation, n.node_id),
            )

        if norm_strategy != "dgm":
            raise ValueError(f"Unsupported parent selection strategy: {strategy!r}")

        frontier_ids = {n.node_id for n in self.pareto_frontier()}
        self.recompute_novelty()
        rng = random.Random(seed)
        weights: list[float] = []
        for n in ordered:
            perf_term = max(0.05, n.pass_rate + 0.05)
            novelty_term = 0.35 + max(0.0, n.novelty_score)
            frontier_bonus = 1.4 if n.node_id in frontier_ids else 1.0
            child_penalty = 1.0 / (1.0 + len(n.children_ids))
            weight = perf_term * novelty_term * frontier_bonus * child_penalty
            weights.append(max(weight, 1e-4))
        return rng.choices(ordered, weights=weights, k=1)[0]

    def render_summary(self) -> str:
        """Render ASCII DAG tree and Pareto frontier table."""
        if not self.nodes:
            return "Lineage archive is empty."

        frontier_ids = {n.node_id for n in self.pareto_frontier()}
        lines: list[str] = ["Lineage DAG:"]
        visited: set[str] = set()

        def _walk(node_id: str, prefix: str, is_last: bool, is_root: bool) -> None:
            if node_id in visited or node_id not in self.nodes:
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            marker = "*" if node.node_id in frontier_ids else " "
            connector = "" if is_root else ("└─ " if is_last else "├─ ")
            lines.append(
                f"{prefix}{connector}[{marker}] {node.node_id} "
                f"(gen={node.generation}, branch={node.branch or '-'}, "
                f"pass={node.pass_rate:.3f}, tok={int(node.total_tokens)}, "
                f"lat={node.latency_ms:.1f}ms, nov={node.novelty_score:.3f})"
            )
            child_prefix = prefix + ("" if is_root else ("   " if is_last else "│  "))
            valid_children = [cid for cid in node.children_ids if cid in self.nodes]
            for idx, cid in enumerate(valid_children):
                _walk(cid, child_prefix, idx == len(valid_children) - 1, False)

        roots = [n.node_id for n in self.nodes.values() if not n.parent_id or n.parent_id not in self.nodes]
        if self.root_id and self.root_id in roots:
            roots = [self.root_id] + [r for r in roots if r != self.root_id]
        for rid in roots:
            _walk(rid, "", True, True)
        for nid in self.nodes:
            if nid not in visited:
                _walk(nid, "", True, True)

        lines.append("")
        lines.append("Pareto Frontier (* = non-dominated):")
        lines.append("  node_id | gen | pass_rate | tokens | latency_ms | novelty | branch")
        for node in self.pareto_frontier():
            lines.append(
                f"  {node.node_id} | {node.generation} | {node.pass_rate:.4f} | "
                f"{int(node.total_tokens)} | {node.latency_ms:.1f} | "
                f"{node.novelty_score:.3f} | {node.branch or '-'}"
            )
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect DGM-style RSI lineage archive")
    parser.add_argument("--repo", type=Path, default=Path("."), help="Repository root or lineage.json path")
    parser.add_argument("--pareto", action="store_true", help="Output only Pareto frontier nodes")
    parser.add_argument(
        "--sample",
        choices=["dgm", "latest", "best"],
        default=None,
        help="Sample a parent node using the given strategy",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for dgm sampling")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    archive = LineageArchive.load(args.repo.resolve())
    if args.sample:
        chosen = archive.sample_parent(strategy=args.sample, seed=args.seed)
        if args.json:
            print(json.dumps(chosen.to_dict() if chosen else None, indent=2))
        else:
            print(chosen.node_id if chosen else "No nodes available")
        return 0

    if args.pareto:
        frontier = [n.to_dict() for n in archive.pareto_frontier()]
        if args.json:
            print(json.dumps(frontier, indent=2))
        else:
            print(archive.render_summary())
        return 0

    if args.json:
        print(
            json.dumps(
                {
                    "archive_path": str(archive.archive_path),
                    "root_id": archive.root_id,
                    "node_count": len(archive.nodes),
                    "pareto_frontier_ids": [n.node_id for n in archive.pareto_frontier()],
                    "nodes": {nid: n.to_dict() for nid, n in archive.nodes.items()},
                },
                indent=2,
            )
        )
    else:
        print(archive.render_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
