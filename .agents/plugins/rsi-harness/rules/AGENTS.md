# RSI-Harness Operational Rules

- 変更を加える前に、対象ファイルのテストと境界条件（空入力・負数・エラー終了）を確認すること。
- 巨大なログ出力は `.rsi/artifacts/` の参照情報または検証済み抜粋を優先し、無関係な全量コピーを避けること。
- `.agents/plugins/rsi-harness/hooks.json`、`scripts/`、`benchmarks/` は保護対象であり、進化ワークフロー以外で直接改変しないこと。
