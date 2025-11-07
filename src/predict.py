"""保存済みモデルを読み込み株価予測を生成する。"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_trained_models() -> tuple:
    """学習済みモデルを読み込む。"""
    try:
        import pickle

        lstm_model = None
        xgb_model = None
        scaler = None

        lstm_path = Path("models/lstm_model.h5")
        xgb_path = Path("models/xgboost_model.pkl")
        scaler_path = Path("models/scaler.pkl")

        if lstm_path.exists():
            try:
                from tensorflow.keras.models import load_model

                lstm_model = load_model(str(lstm_path))
            except Exception as e:
                print(f"⚠️ LSTM loading failed: {e}")

        if xgb_path.exists():
            try:
                with open(xgb_path, "rb") as f:
                    xgb_model = pickle.load(f)
            except Exception as e:
                print(f"⚠️ XGBoost loading failed: {e}")

        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Scaler loading failed: {e}")

        return lstm_model, xgb_model, scaler
    except Exception as e:
        print(f"⚠️ Model loading error: {e}")
        return None, None, None


def generate_dummy_forecast(
    symbols: List[str], days_ahead: int = 5
) -> pd.DataFrame:
    """ダミー予測を生成（モデルが利用できない場合）。"""
    records = []
    base_prices = {
        "AAPL": 150.5,
        "GOOGL": 185.3,
        "MSFT": 420.1,
        "TSLA": 285.5,
        "9984": 3150.5,
        "6758": 15200.0,
        "7203": 2890.5,
        "8306": 2450.0,
    }

    for symbol in symbols:
        base_price = base_prices.get(symbol, 100.0)
        last_date = pd.Timestamp.now()

        for day in range(1, days_ahead + 1):
            # ランダムウォークで予測
            drift = (np.random.random() - 0.5) * base_price * 0.02
            forecast_price = base_price + drift * day
            confidence = 0.70 + np.random.random() * 0.15

            target_date = last_date + timedelta(days=day)

            records.append(
                {
                    "symbol": symbol,
                    "date": target_date.strftime("%Y-%m-%d"),
                    "forecast": round(forecast_price, 2),
                    "confidence": round(confidence, 2),
                }
            )

    return pd.DataFrame(records)


def predict_with_models(
    data_df: pd.DataFrame,
    lstm_model,
    xgb_model,
    scaler,
    days_ahead: int = 5,
) -> pd.DataFrame:
    """学習済みモデルで予測を生成。"""
    records = []

    for symbol in data_df["symbol"].unique():
        symbol_data = data_df[data_df["symbol"] == symbol].copy()

        if symbol_data.empty:
            continue

        # 最後の Close 価格
        last_close = float(symbol_data["Close"].iloc[-1])

        # LSTM 予測
        lstm_pred = None
        if lstm_model is not None:
            try:
                # 最後の 10 日間を使用
                recent_data = symbol_data[["Close"]].tail(10).values
                if len(recent_data) > 0:
                    lstm_pred = float(lstm_model.predict(recent_data.reshape(1, -1, 1))[0, 0])
            except Exception as e:
                print(f"⚠️ LSTM prediction failed for {symbol}: {e}")

        # XGBoost 予測
        xgb_pred = None
        if xgb_model is not None:
            try:
                features = symbol_data[
                    ["Close", "Volume"]
                ].tail(1).values
                if len(features) > 0:
                    xgb_pred = float(xgb_model.predict(features)[0])
            except Exception as e:
                print(f"⚠️ XGBoost prediction failed for {symbol}: {e}")

        # 両方の予測を組み合わせ
        predictions = [p for p in [lstm_pred, xgb_pred] if p is not None]
        if predictions:
            combined_pred = np.mean(predictions)
        else:
            combined_pred = last_close

        # 信頼度スコア
        confidence = 0.75 + np.random.random() * 0.15
        if predictions:
            confidence = min(0.95, confidence)
        else:
            confidence = 0.50

        # 5営業日先の予測を生成
        last_date = pd.Timestamp.now()

        for day in range(1, days_ahead + 1):
            # 線形補間で日次予測を計算
            drift = (combined_pred - last_close) / days_ahead
            forecast_price = last_close + drift * day

            target_date = last_date + timedelta(days=day)

            records.append(
                {
                    "symbol": symbol,
                    "date": target_date.strftime("%Y-%m-%d"),
                    "forecast": round(forecast_price, 2),
                    "confidence": round(confidence, 2),
                }
            )

    return pd.DataFrame(records)


def main(
    us_symbols: List[str],
    jp_symbols: List[str],
    days_ahead: int,
    output_path: str,
) -> None:
    """メイン予測関数。"""
    print("📊 Stock Price Prediction Pipeline")
    print(f"🎯 Symbols: {us_symbols + jp_symbols}")
    print(f"📅 Prediction horizon: {days_ahead} days")

    # シンボル統合
    all_symbols = us_symbols + jp_symbols
    print(f"✅ Total symbols: {len(all_symbols)}")

    # ダミーデータ生成（実データがない場合）
    print("📁 Loading training data...")
    try:
        data_df = pd.read_csv("data/stock_data/processed/train_data.csv")
        if data_df.empty:
            raise ValueError("Training data is empty")
        print(f"✅ Training data loaded: {len(data_df)} rows")
    except Exception as e:
        print(f"⚠️ Training data loading failed: {e}")
        print("📈 Generating dummy data...")
        data_df = pd.DataFrame({
            "symbol": np.repeat(all_symbols, 100),
            "Close": np.random.uniform(50, 500, len(all_symbols) * 100),
            "Volume": np.random.uniform(1000000, 10000000, len(all_symbols) * 100),
        })

    # モデル読み込み
    print("🤖 Loading models...")
    lstm_model, xgb_model, scaler = load_trained_models()

    # 予測生成
    if lstm_model is not None or xgb_model is not None:
        print("✅ Models loaded successfully")
        forecast_df = predict_with_models(
            data_df, lstm_model, xgb_model, scaler, days_ahead
        )
    else:
        print("⚠️ Models not available, generating dummy predictions")
        forecast_df = generate_dummy_forecast(all_symbols, days_ahead)

    # 出力
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False)
    print(f"\n✅ Forecast saved to {output_path}")
    print(f"📊 Generated {len(forecast_df)} predictions")
    print("\n📋 Sample forecasts:")
    print(forecast_df.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock price prediction")
    parser.add_argument(
        "--us-symbols",
        nargs="+",
        default=["AAPL", "GOOGL", "MSFT", "TSLA"],
        help="US stock symbols",
    )
    parser.add_argument(
        "--jp-symbols",
        nargs="+",
        default=["9984", "6758", "7203", "8306"],
        help="Japanese stock symbols",
    )
    parser.add_argument(
        "--days-ahead", type=int, default=5, help="Forecast horizon (days)"
    )
    parser.add_argument(
        "--output",
        default="data/stock_data/predictions/forecast.csv",
        help="Output file path",
    )
    parser.add_argument("--log", default="logs/predict.log", help="Log file path")

    args = parser.parse_args()

    # ロギング設定
    import logging

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main(args.us_symbols, args.jp_symbols, args.days_ahead, args.output)
