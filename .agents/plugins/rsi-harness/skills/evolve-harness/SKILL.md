---
name: evolve-harness
description: >-
  Use this skill when the user asks to analyze execution traces and evolve or improve the agent harness (rules/skills) via RRSI, Meta-Harness, and SoL-Pi gates.
---

# Evolve Harness (Outer Loop)

過去の会話トレース（`.rsi/traces/` および `transcript.jsonl`）を根拠に、ハーネス（`rules/AGENTS.md` および `skills/*/SKILL.md`）を安全かつ正則化された手順で改善し、`rsi/evolved` ブランチへ自動適用する手順です。

## Steps

1. 直近のトレースと進化履歴を確認する:
   - `.rsi/traces/` 内のサマリおよび生の `transcript.jsonl` を `grep` / `read_file` で調査し、失敗モードや無駄なターン消費を特定する。
   - `.rsi/evolution_history.jsonl` を確認し、過去に棄却された仮説を繰り返さないようにする。
2. Outer Loop オーケストレータを実行する:
   - モック検証:
     `python3 .agents/plugins/rsi-harness/scripts/evolve_outer.py --runner mock --branch rsi/evolved`
   - 実機 Antigravity (`agy -p`) 評価:
     `python3 .agents/plugins/rsi-harness/scripts/evolve_outer.py --runner agy --branch rsi/evolved`
3. 結果を確認する:
   - 出力 JSON の `accepted`、`baseline`、`candidate`、`critic_reasons`、`branch` を確認する。
   - `accepted: true` の場合、`git log -n 3 rsi/evolved` で自動適用された差分を確認する。
