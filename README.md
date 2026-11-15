# RSI-Harness 試験用リポジトリ

**Meta-Harness**・**RRSI**・**SoL-Pi** の設計思想を取り入れた Antigravity 向け自己再帰的改善ハーネス（`.agents/plugins/rsi-harness/`）の検証用リポジトリです。

## 構成

- `src/tasklib.py` / `tests/test_tasklib.py`: サンプル Python モジュールと単体テスト
- `rsi.config.json`: Inner Loop の検査コマンドおよび RRSI の編集予算・ゲート閾値設定
- `.agents/plugins/rsi-harness/`: Antigravity プラグイン本体
  - `hooks.json`: `PreToolUse` 改変ガード / `PostToolUse` トレース収集 / `Stop` 品質ゲート
  - `rules/AGENTS.md`: 常時適用ルール（Outer Loop による改善対象）
  - `skills/evolve-harness/SKILL.md`: Outer Loop 実行用スキル
  - `scripts/`: 各種フックスクリプトと `evolve_outer.py`
- `benchmarks/`: 検証用ミニベンチマーク（3課題）とランナー

## 検証コマンド

```bash
# 1. プラグインマニフェスト・構造検証
agy plugin validate ./.agents/plugins/rsi-harness --strict --json

# 2. サンプルプロジェクトのテスト
python3 -m unittest discover -s tests -v

# 3. プラグイン & Outer Loop の単体テスト
python3 -m unittest discover -s .agents/plugins/rsi-harness/tests -v

# 4. ミニベンチマーク実行（モックランナー）
python3 benchmarks/run_bench.py --runner mock

# 5. Outer Loop 進化サイクルのテスト実行（rsi/evolved ブランチへ自動適用）
python3 .agents/plugins/rsi-harness/scripts/evolve_outer.py --runner mock --branch rsi/evolved
```
