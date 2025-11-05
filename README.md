# Kabuu1 - ML Prediction Pipeline

高度な機械学習と強化学習を組み合わせた予測パイプラインシステム

## 🎯 概要

- **予測モデル**: LSTM + 強化学習による動的最適化
- **リアルタイム処理**: スケジューラーによる定期実行
- **自動改善**: モデル性能に基づく自動再学習
- **可視化**: ダッシュボード + メトリクス追跡

## 📁 プロジェクト構造

```
kabuu1/
├── src/                          # メインコード
│   ├── reinforcement_learning.py # 強化学習モジュール
│   ├── prediction_pipeline.py    # 予測パイプライン
│   └── monitor.py                # モニタリング
├── tests/                        # テストコード
│   ├── test_rl.py               # RL テスト
│   └── test_pipeline.py         # パイプラインテスト
├── config/                       # 設定
│   └── config.yaml              # YAML設定
├── data/                         # データ出力 (Git無視)
│   ├── rl_results/
│   ├── validation_results/
│   └── prediction_results/
├── logs/                         # ログ出力 (Git無視)
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml       # GitHub Actions
├── requirements.txt              # 依存関係
├── Dockerfile                    # Docker設定
├── .gitignore                    # Git無視ルール
└── README.md                     # このファイル
```

## 🚀 クイックスタート

### ローカル開発

```bash
# リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/kabuu1.git
cd kabuu1

# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt

# テスト実行
pytest tests/ -v

# 予測パイプライン実行
python -m src.prediction_pipeline
```

### Docker での実行

```bash
# イメージビルド
docker build -t kabuu1:latest .

# コンテナ実行
docker run -v $(pwd)/data:/app/data kabuu1:latest

# Docker Compose 使用
docker-compose up -d
```

## 📊 機能一覧

### 1. 強化学習ベース最適化
- 予測精度に基づく動的な学習率調整
- ボラティリティを考慮した報酬計算
- 自動モデル改善判定

### 2. 予測パイプライン
- 複数銘柄の並列処理
- スケジューラーによる定期実行
- 実績値との照合と自動更新

### 3. リアルタイム監視
- Flaskベースのダッシュボード
- メトリクスの自動保存
- 性能トレンド分析

### 4. 自動改善
- 精度低下の自動検出
- 緊急改善モード
- パラメータ自動チューニング

## ⚙️ 設定

`config/config.yaml` を編集：

```yaml
model:
  learning_rate: 0.001
  hidden_units: 128
  batch_size: 32
  epochs: 10
  lookback_days: 365

schedule:
  prediction_time: "09:00"
  model_review_day: "sunday"
  model_review_time: "22:00"

logging:
  level: "INFO"
  file: "logs/prediction_pipeline.log"
```

## 🧪 テスト

```bash
# すべてのテスト実行
pytest tests/ -v

# カバレッジ付きテスト
pytest tests/ --cov=src --cov-report=html

# 特定のテストファイル
pytest tests/test_rl.py -v
```

## 📈 パフォーマンスメトリクス

システムは以下のメトリクスを自動追跡：

- **予測誤差率**: 実際値との乖離度
- **報酬スコア**: 強化学習の評価値
- **シャープレシオ**: リスク調整済みリターン
- **改善トレンド**: 精度改善傾向

## 🔄 GitHub Actions ワークフロー

自動実行トリガー：

- **Push**: main/develop ブランチへの push
- **Pull Request**: main ブランチへの PR
- **スケジュール**: 毎日 09:00 UTC、毎週日曜 22:00 UTC

## 📝 ログ出力

ログは `logs/` ディレクトリに自動保存：

```
logs/
├── prediction_pipeline.log      # 主要ログ
├── reinforcement_learning.log   # RL ログ
└── monitor.log                  # モニタリングログ
```

## 🐛 トラブルシューティング

### 依存関係エラー
```bash
# 依存関係の再インストール
pip install --upgrade -r requirements.txt
```

### TA-Lib のインストール失敗
```bash
# システムライブラリをインストール
sudo apt-get install build-essential
pip install TA-Lib==0.4.28
```

### ポート競合
```bash
# 別のポートでダッシュボード実行
python monitor.py --port 5001
```

## 🤝 貢献

1. Feature ブランチを作成
2. 変更をコミット
3. Pull Request を作成
4. レビュー後に マージ

## 📄 ライセンス

MIT License

## 📞 サポート

問題が発生した場合は GitHub Issues で報告してください。

## 🎓 参考資料

- [TensorFlow ドキュメント](https://www.tensorflow.org/)
- [scikit-learn ドキュメント](https://scikit-learn.org/)
- [強化学習入門](https://www.rl-book.com/)

---

**最終更新**: 2025-11-05
**バージョン**: 1.0.0
