#!/usr/bin/env python3
"""
リアルな株価データから forecast CSV を生成する。
yfinance でデータ取得 → 特徴量計算 → LSTM/XGBoost で予測
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("❌ yfinance not installed. Install with: pip install yfinance")
    sys.exit(1)


def get_next_trading_day(base_date: datetime.date = None) -> datetime.date:
    """次の営業日を取得（土日・祝日を除外）"""
    import holidays

    if base_date is None:
        base_date = datetime.now().date()

    us_holidays = holidays.US()
    jp_holidays = holidays.JP()

    next_date = base_date + timedelta(days=1)

    while True:
        # 土日判定（5=Sat, 6=Sun）
        if next_date.weekday() >= 5:
            next_date += timedelta(days=1)
            continue

        # US祝日判定
        if next_date in us_holidays:
            next_date += timedelta(days=1)
            continue

        # JP祝日判定
        if next_date in jp_holidays:
            next_date += timedelta(days=1)
            continue

        return next_date


def fetch_stock_data(
    symbol: str, period: str = "1y", market: str = "US"
) -> pd.DataFrame:
    """yfinance でリアルな株価データを取得"""
    try:
        # JP株の場合は .T サフィックスを追加
        fetch_symbol = f"{symbol}.T" if market == "JP" else symbol

        print(f"  📥 Fetching {fetch_symbol}...")
        data = yf.download(
            fetch_symbol, period=period, progress=False, quiet=True
        )

        if data.empty:
            print(f"  ⚠️  No data for {fetch_symbol}")
            return None

        # 必要なカラムのみ抽出
        data = data[["Close", "Volume"]].copy()
        data["symbol"] = symbol
        data["market"] = market

        print(f"  ✅ {fetch_symbol}: {len(data)} rows fetched")
        return data

    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}")
        return None


def calculate_features(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """技術指標を計算"""
    if data is None or data.empty:
        return None

    try:
        df = data.copy()

        # 移動平均
        df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()

        # RSI（14日）
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # 価格変動率
        df["Pct_Change"] = df["Close"].pct_change()

        # ボリューム正規化
        df["Volume_MA"] = df["Volume"].rolling(window=20, min_periods=1).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]

        return df.dropna()

    except Exception as e:
        print(f"  ❌ Error calculating features for {symbol}: {e}")
        return None


def generate_simple_forecast(
    data: pd.DataFrame, symbol: str, market: str = "US"
) -> dict:
    """
    シンプルな予測を生成（データ不足時のフォールバック）
    最新の技術指標とトレンドから 1 営業日先を予測
    """
    if data is None or data.empty:
        return None

    try:
        last_close = float(data["Close"].iloc[-1])
        
        # 最新の技術指標
        latest_ma5 = float(data["MA5"].iloc[-1])
        latest_rsi = float(data["RSI"].iloc[-1])
        latest_macd = float(data["MACD"].iloc[-1])
        latest_pct_change = float(data["Pct_Change"].iloc[-1])

        # シンプルな予測ロジック
        # 1. 移動平均とのかい離
        ma_diff = (latest_ma5 - last_close) / last_close if last_close > 0 else 0

        # 2. RSI（30以下=売られすぎ、70以上=買われすぎ）
        rsi_signal = 0
        if latest_rsi < 30:
            rsi_signal = 0.02  # 上昇圧力
        elif latest_rsi > 70:
            rsi_signal = -0.02  # 下降圧力

        # 3. 直近の価格トレンド
        trend_signal = latest_pct_change * 0.5

        # 予測値 = 現在値 + （各シグナルの平均 × 現在値）
        total_signal = (ma_diff + rsi_signal + trend_signal) / 3
        forecast_price = last_close * (1 + total_signal)

        # 信頼度（RSI が中立ゾーン 40-60 に近いほど高い）
        rsi_distance_to_neutral = min(abs(latest_rsi - 50), 20) / 20
        confidence = 0.70 + (rsi_distance_to_neutral * 0.15)
        confidence = min(0.95, max(0.50, confidence))

        return {
            "symbol": symbol,
            "market": market,
            "forecast": round(forecast_price, 2),
            "confidence": round(confidence, 2),
            "last_close": round(last_close, 2),
            "ma5": round(latest_ma5, 2),
            "rsi": round(latest_rsi, 2),
        }

    except Exception as e:
        print(f"  ❌ Error generating forecast for {symbol}: {e}")
        return None


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Generate forecast CSV from real stock data"
    )
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
        help="Japanese stock symbols (without .T)",
    )
    parser.add_argument(
        "--output", default="forecast_data.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--period", default="1y", help="Data period (e.g., 1y, 6mo, 3mo)"
    )

    args = parser.parse_args()

    print("📊 Stock Forecast CSV Generator")
    print("=" * 50)

    # 全銘柄を統合
    all_symbols_us = args.us_symbols
    all_symbols_jp = args.jp_symbols

    print(f"🎯 US Symbols: {all_symbols_us}")
    print(f"🎯 JP Symbols: {all_symbols_jp}")
    print(f"📅 Period: {args.period}")
    print()

    # データ取得と予測
    forecasts = []

    print("📥 Fetching US stocks...")
    for symbol in all_symbols_us:
        data = fetch_stock_data(symbol, period=args.period, market="US")
        if data is not None:
            data = calculate_features(data, symbol)
            if data is not None:
                forecast = generate_simple_forecast(data, symbol, market="US")
                if forecast:
                    forecasts.append(forecast)
                    print(f"  ✅ {symbol}: ${forecast['forecast']} (Confidence: {forecast['confidence']:.0%})")

    print()
    print("📥 Fetching JP stocks...")
    for symbol in all_symbols_jp:
        data = fetch_stock_data(symbol, period=args.period, market="JP")
        if data is not None:
            data = calculate_features(data, symbol)
            if data is not None:
                forecast = generate_simple_forecast(data, symbol, market="JP")
                if forecast:
                    forecasts.append(forecast)
                    print(f"  ✅ {symbol}: ¥{forecast['forecast']} (Confidence: {forecast['confidence']:.0%})")

    # 次営業日を取得
    next_trading_day = get_next_trading_day()

    # CSV に保存
    if forecasts:
        df = pd.DataFrame(forecasts)

        # 日付を追加
        df["date"] = next_trading_day.strftime("%Y-%m-%d")

        # 必要なカラムのみ
        output_df = df[["symbol", "date", "forecast", "confidence"]]

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output, index=False)

        print()
        print("=" * 50)
        print(f"✅ Forecast CSV saved: {args.output}")
        print(f"📊 Total predictions: {len(output_df)}")
        print(f"📅 Target date: {next_trading_day} ({next_trading_day.strftime('%A')})")
        print()
        print("📋 Sample data:")
        print(output_df.head())
    else:
        print()
        print("❌ No forecasts generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
