# antigravity-rsi-plugin (`rsi-harness`)

An open-ended **Recursive Self-Improvement (RSI) & Agent Harness Evolution** framework built as both a standalone zero-dependency Python package/CLI (`rsi-harness`) and an Antigravity IDE plugin (`.agents/plugins/rsi-harness/`).

It implements and unifies:

- **Meta-Harness** — Raw execution traces stored uncompressed in `.rsi/store/` & OTel `gen_ai.*` `agent_trace.jsonl` so the Outer Loop proposer inspects full evidence and tool history via symlinked worktrees.
- **RRSI (Regularized Recursive Self-Improvement)** — Proposal-side budget annealing, structural pruning (`--prune`), generalized holdout/OOD leakage Critic, Anti-Goodhart SHA-256 read-only verifier protection, and variance-aware Pareto multi-trial selection gates.
- **SoL-Pi** — Inner-loop execution/trace gates (`guard_harness.py`, `inner_gate.py`, `evidence_reducer.py` + `ObservationPack` paging, polyglot `toolchain.py`, atomic multi-edit `action_fusion.py`, and trace compaction `context_compact.py`).
- **DGM (Darwin Gödel Machine)** — Open-ended `LineageArchive` DAG (`.rsi/lineage.json`) supporting multi-branch evolution, Pareto frontier tracking, novelty scoring, and diversity-preserving parent sampling.

---

## Repository Structure

- `rsi_harness/` & `pyproject.toml`: Zero-dependency Python 3.11+ package and `rsi-harness` CLI (`init`, `evolve`, `status`, `mine`, `lineage`).
- `.agents/plugins/rsi-harness/`: Antigravity plugin bundle (`plugin.json`, `hooks.json`, `rules/`, `skills/`, `scripts/`, `tests/`):
  - `scripts/adapters/`: Multi-runtime adapters (`agy`, `claude`, `codex`, `gemini`, `pty`), normalized hook events (`events.py`), and sandbox backends (`GitWorktreeSandbox`, hardened `DockerSandbox`, `RemoteSandboxAdapter`).
  - `scripts/trace_schema.py`, `scripts/trace_collector.py`, `scripts/trace_store.py`: OTel `gen_ai.*` trace normalization and uncompressed generational trace store.
  - `scripts/toolchain.py`, `scripts/surface.py`, `scripts/weakness_miner.py`, `scripts/budget_and_pruning.py`: Polyglot test detection, declarative `harness.yaml` mutable surface, real-session weakness mining, and annealed pruning.
  - `scripts/lineage.py`, `scripts/interfaces.py`, `scripts/pr_reporter.py`, `scripts/evolve_outer.py`: DGM DAG archive, pluggable `Proposer`/`Evaluator`/`Critic` + `MetaRewardEvaluator`, evidence-backed PR reporting (`gh pr create`), and outer loop orchestration.
- `benchmarks/`: Multi-trial Harbor & JSON benchmark runner (`benchmarks/run_bench.py`, `benchmarks/tasks.json`, `benchmarks/harbor_examples/sample_task/`).
- `src/` & `tests/`: Sample baseline target library and regression tests.

---

## Quick Start (`rsi-harness` CLI)

```bash
# Auto-detect project toolchain & scaffold harness.yaml / rsi.config.json safely
python3 -m rsi_harness init --repo . --dry-run

# Inspect configuration, trace store, and lineage status
python3 -m rsi_harness status --repo . --json

# Mine failure/inefficiency clusters from trace logs
python3 -m rsi_harness mine --repo .

# Run one outer evolution cycle and generate a PR report (dry-run)
python3 -m rsi_harness evolve --repo . --runner mock --parent-strategy dgm --pr-dry-run

# Inspect the DGM lineage DAG and Pareto frontier
python3 -m rsi_harness lineage --repo . --pareto
```

---

## Verification Commands

```bash
# Validate Antigravity plugin schema strictly
agy plugin validate ./.agents/plugins/rsi-harness --strict --json

# Run baseline library unit tests
python3 -m unittest discover -s tests -v

# Run full RSI-Harness plugin, adapter, benchmark, and CLI test suite
python3 -m unittest discover -s .agents/plugins/rsi-harness/tests -v

# Run 3-trial mock benchmark suite
python3 benchmarks/run_bench.py --runner mock --trials 3

# Inspect CLI status output
python3 -m rsi_harness status --repo . --json
```

---

## License & References

- Licensed under the **MIT License** (see `LICENSE`).
- Research inspirations:
  - **Meta-Harness** (`arXiv:2603.24693`) — End-to-End Optimization of Model Harnesses
  - **Darwin Gödel Machine** (`arXiv:2505.22954`) — Open-Ended Evolution of Self-Improving Agents
  - **RRSI** (`arXiv:2604.06232`) — Regularizing Recursive Self-Improvement Against Goodhart's Law
  - **SoL-Pi** (`arXiv:2604.14228`) — Joint Prompt & Code Co-Evolution with Action Fusion & Context Compact
