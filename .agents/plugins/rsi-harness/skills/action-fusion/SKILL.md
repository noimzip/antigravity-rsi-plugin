---
name: action-fusion
description: >-
  SoL-Pi Action Fusion & Online Context Compact skill for combining edits with lint/test verification in a single round-trip and compacting subtask trace context.
---

# Action Fusion & Online Context Compact

定型シーケンス（複数ファイル編集＋構文検査＋Lint/Test 実行）を 1 ラウンドトリップへ統合し、サブタスク境界でコンテキストをオンライン圧縮するための SoL-Pi ガイドとラッパーです。

## Action Fusion

`action_fusion.py` を使用すると、複数ファイルのパッチ適用・Python `ast.parse` 構文検証（失敗時は自動ロールバック）・テスト/Lint 実行・エビデンス縮約を 1 回のコマンド呼び出しで完結できます。

```bash
python3 .agents/plugins/rsi-harness/scripts/action_fusion.py \
  --repo . \
  --edits-json '[{"path": "src/app.py", "target": "old", "replacement": "new"}]' \
  --check-cmd auto
```

出力には `roundtrips_saved` と `estimated_tokens_saved` が記録されます。

## Online Context Compact

サブタスクの区切りで過去イベントを圧縮し、未解決エラー・編集済みファイル・`ObservationPack` ハンドルのみを保持します。

```bash
echo '{"cwd": ".", "conversationId": "<id>"}' | \
  python3 .agents/plugins/rsi-harness/scripts/context_compact.py --max-keep-recent 5
```

圧縮結果と削減率 (`reduction_ratio`, `estimated_tokens_saved`) は `.rsi/traces/<id>/compacted_context.json` に保存されます。
