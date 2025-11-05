import sys
import time
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from prediction_pipeline.prediction_pipeline import PredictionPipeline, Config

def test_pipeline():
    """パイプラインのテスト実行"""
    # 設定を読み込み
    config = Config()
    
    # パイプラインを初期化
    pipeline = PredictionPipeline(config)
    
    # テスト用の予測を実行
    print("=== テスト予測を実行します ===\n")
    
    test_cases = [
        {"ticker": "9984.T", "predicted": 1000, "actual": 980},
        {"ticker": "6758.T", "predicted": 5000, "actual": 5100},
        {"ticker": "7203.T", "predicted": 2500, "actual": 2450},
    ]
    
    # 予測を記録
    for case in test_cases:
        ticker = case["ticker"]
        predicted = case["predicted"]
        
        # 予測を実行
        features = pipeline.prepare_features(ticker)
        result = pipeline.predict(ticker, features)
        print(f"✓ {ticker} の予測を記録しました: {result}")
        
        # 実際の価格を更新（少し遅延させて実行）
        time.sleep(1)
        pipeline.update_with_actual_price(ticker, case["actual"])
        print(f"✓ {ticker} の実際の価格を更新しました: {case['actual']}\n")
    
    print("=== テスト完了 ===")
    print("\nメトリクスファイルを確認してください:")
    print("- performance_metrics.json")
    print("\nダッシュボードをリロードしてデータを確認してください:")
    print("- http://localhost:5000")

if __name__ == "__main__":
    test_pipeline()
