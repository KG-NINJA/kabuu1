#!/usr/bin/env python3
"""
generate_forecast_csv.py
株価データを取得し、単純移動平均ベースで翌営業日の予測値を出力するスクリプト
"""

import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import holidays
import os


def get_next_trading_day(market: str = "US") -> str:
    """次の営業日を返す"""
    today = datetime.utcnow().date()
    offset = 1
    while True:
        next_day = today + timedelta(days=offset)
        if market == "JP":
            jp_holidays = holidays.Japan()
            if next_day.weekday() < 5 and next_day not in jp_holidays:
                return next_day.isoformat()
        else:
            us_holidays = holidays.US()
            if next_day.weekday() < 5 and next_day not in us_holidays:
                return next_day.isoformat()
        offset += 1


def forecast_stock(symbol: str, market: str) -> dict:
    """単純な移動平均による予測値を生成"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")
        if data.empty:
            raise ValueError(f"No data for {symbol}")
        latest_close = data["Close"].iloc[-1]
        ma_5 = data["Close"].tail(5).mean()
        ma_20 = data["Close"].tail(20).mean()
        forecast = (latest_close * 0.7) + (ma_5 * 0.2) + (ma_20 * 0.1)
        return {
            "symbol": symbol,
            "market": market,
            "current_price": round(latest_close, 2),
            "forecast": round(forecast, 2),
            "next_trading_day": get_next_trading_day(market),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "market": market,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Generate forecast CSV")
    parser.add_argument("--us-symbols", nargs="*", default=[], help="US stock symbols")
    parser.add_argument("--jp-symbols", nargs="*", default=[], help="JP stock symbols")
    parser.add_argument("--output", type=str, default="forecast_data.csv")
    args = parser.parse_args()

    records = []
    for sym in args.us_symbols:
        records.append(forecast_stock(sym, "US"))
    for sym in args.jp_symbols:
        records.append(forecast_stock(sym, "JP"))

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"✅ Forecast CSV saved: {args.output}")
    print(df.head())


if __name__ == "__main__":
    main()
