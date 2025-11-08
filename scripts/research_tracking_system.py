#!/usr/bin/env python3
"""
research_tracking_system.py
予測精度を追跡し、研究用データセットを自動生成
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

class PredictionResearchTracker:
    """予測と実績を追跡・分析する研究用ツール"""
    
    def __init__(self, data_dir: str = "data/research"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.data_dir / "predictions_history.jsonl"
        self.results_file = self.data_dir / "actual_results.jsonl"
        self.analysis_file = self.data_dir / "accuracy_analysis.json"
    
    def save_prediction(self, prediction_json: Dict):
        """予測結果を記録"""
        record = {
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "prediction_date": prediction_json.get("next_trading_day"),
            "data": prediction_json
        }
        
        with open(self.predictions_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        print(f"✅ Prediction saved for {prediction_json.get('next_trading_day')}")
    
    def record_actual_prices(self, symbol: str, actual_price: float, prediction_date: str):
        """実際の終値を記録"""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "actual_price": actual_price,
            "prediction_date": prediction_date,
            "recorded_date": datetime.now().date().isoformat()
        }
        
        with open(self.results_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def calculate_accuracy(self) -> Dict:
        """精度を計算"""
        # ファイルが存在しない場合は初期化
        if not self.predictions_file.exists():
            return {"status": "insufficient_data", "message": "No predictions recorded yet"}
        
        if not self.results_file.exists():
            return {"status": "insufficient_data", "message": "No actual results recorded yet"}
        
        predictions = []
        with open(self.predictions_file, "r") as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        
        results = {}
        with open(self.results_file, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    key = f"{record['symbol']}_{record['prediction_date']}"
                    results[key] = record
        
        if not predictions or not results:
            return {"status": "insufficient_data", "message": "Not enough data for accuracy calculation"}
        
        # 精度指標を計算
        matches = []
        for pred in predictions:
            pred_date = pred.get("prediction_date")
            for forecast in pred.get("data", {}).get("forecasts", []):
                symbol = forecast.get("symbol")
                predicted_price = forecast.get("forecast")
                current_price = forecast.get("current_price", predicted_price)
                key = f"{symbol}_{pred_date}"
                
                if key in results:
                    actual_price = results[key]["actual_price"]
                    error_pct = abs((predicted_price - actual_price) / actual_price) * 100
                    
                    matches.append({
                        "symbol": symbol,
                        "prediction_date": pred_date,
                        "predicted_price": predicted_price,
                        "actual_price": actual_price,
                        "error_pct": error_pct,
                        "correct_direction": (predicted_price - current_price) * (actual_price - current_price) > 0
                    })
        
        if not matches:
            return {"status": "insufficient_data", "message": "Not enough matched predictions and results"}
        
        df = pd.DataFrame(matches)
        
        return {
            "total_predictions": len(matches),
            "mean_absolute_error_pct": float(df["error_pct"].mean()),
            "median_absolute_error_pct": float(df["error_pct"].median()),
            "std_error_pct": float(df["error_pct"].std()),
            "direction_accuracy": float(df["correct_direction"].sum() / len(df) * 100),
            "by_symbol": df.groupby("symbol").agg({
                "error_pct": ["mean", "median", "count"],
                "correct_direction": "sum"
            }).to_dict(),
            "predictions": df.to_dict(orient="records")
        }
    
    def generate_research_report(self) -> str:
        """研究用レポートを生成"""
        accuracy = self.calculate_accuracy()
        
        if accuracy.get("status") == "insufficient_data":
            return "## 研究データ - 準備中\n\nデータが集まるまでお待ちください。"
        
        report = f"""# 📊 Stock Prediction Research Report

## データ収集状況
- **総予測数**: {accuracy['total_predictions']}
- **収集日**: {datetime.now().isoformat()}

## 精度指標

### 全体
- **平均絶対誤差 (MAE)**: {accuracy['mean_absolute_error_pct']:.2f}%
- **中央絶対誤差 (Median)**: {accuracy['median_absolute_error_pct']:.2f}%
- **標準偏差**: {accuracy['std_error_pct']:.2f}%
- **方向性正確度**: {accuracy['direction_accuracy']:.1f}%

### 銘柄別パフォーマンス
"""
        
        for symbol, stats in accuracy.get("by_symbol", {}).items():
            if isinstance(stats, dict) and "error_pct" in stats:
                report += f"\n#### {symbol}\n"
                report += f"- 平均誤差: {stats['error_pct']['mean']:.2f}%\n"
                report += f"- 中央誤差: {stats['error_pct']['median']:.2f}%\n"
                report += f"- 予測数: {int(stats['error_pct']['count'])}\n"
                if "correct_direction" in stats:
                    correct = stats['correct_direction']['sum']
                    total = stats['error_pct']['count']
                    report += f"- 方向性正確度: {correct/total*100:.1f}%\n"
        
        report += "\n## 研究用途\n\n"
        report += "このデータセットは以下の研究に使用できます：\n\n"
        report += "1. **機械学習モデルの比較研究**\n"
        report += "   - LSTM vs XGBoost の予測精度比較\n"
        report += "   - テクニカル指標の有効性検証\n\n"
        report += "2. **LLM の金融分析能力評価**\n"
        report += "   - Claude vs GPT-4 vs Gemini の精度比較\n"
        report += "   - LLM バイアス分析\n\n"
        report += "3. **市場効率性の実証研究**\n"
        report += "   - 短期予測可能性の検証\n"
        report += "   - 技術的分析の信頼性\n\n"
        report += "4. **時系列予測方法論**\n"
        report += "   - アンサンブル学習の有効性\n"
        report += "   - 予測期間別精度比較\n"
        
        return report
    
    def export_dataset(self, format: str = "csv") -> Path:
        """研究用データセットをエクスポート"""
        accuracy = self.calculate_accuracy()
        
        if accuracy.get("status") == "insufficient_data":
            print(f"⚠️ {accuracy.get('message', 'Not enough data for export')}")
            return None
        
        df = pd.DataFrame(accuracy.get("predictions", []))
        
        if df.empty:
            print("⚠️ No data to export")
            return None
        
        if format == "csv":
            output_path = self.data_dir / "research_dataset.csv"
            df.to_csv(output_path, index=False)
        elif format == "json":
            output_path = self.data_dir / "research_dataset.json"
            df.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Dataset exported: {output_path}")
        return output_path


def main():
    """メイン処理"""
    tracker = PredictionResearchTracker()
    
    # レポート生成
    report = tracker.generate_research_report()
    print(report)
    
    # データセットエクスポート
    tracker.export_dataset("csv")
    tracker.export_dataset("json")


if __name__ == "__main__":
    main()
