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
    
    if market == "JP":
        jp_holidays = holidays.Japan()
        while True:
            next_day = today + timedelta(days=offset)
            if next_day.weekday() < 5 and next_day not in jp_holidays:
                return next_day.isoformat()
            offset += 1
    else:
        us_holidays = holidays.US()
        while True:
            next_day = today + timedelta(days=offset)
            if next_day.weekday() < 5 and next_day not in us_holidays:
                return next_day.isoformat()
            offset += 1


def resolve_symbol(symbol: str, market: str) -> str:
    """yfinance 用のシンボルを生成"""
    if market == "JP":
        # JP株は .T サフィックスが必須
        return f"{symbol}.T"
    return symbol


def forecast_stock(symbol: str, market: str) -> dict:
    """単純な移動平均による予測値を生成"""
    try:
        # yfinance用シンボルに変換
        yf_symbol = resolve_symbol(symbol, market)
        
        print(f"  📥 Fetching {yf_symbol}...")
        stock = yf.Ticker(yf_symbol)
        data = stock.history(period="1y")
        
        if data.empty:
            raise ValueError(f"No data for {yf_symbol}")
        
        latest_close = data["Close"].iloc[-1]
        ma_5 = data["Close"].tail(5).mean()
        ma_20 = data["Close"].tail(20).mean()
        
        # 加重平均で予測
        forecast = (latest_close * 0.7) + (ma_5 * 0.2) + (ma_20 * 0.1)
        
        # 信頼度を簡易計算
        volatility = data["Close"].pct_change().std()
        confidence = max(0.5, min(0.95, 1.0 - volatility * 2))
        
        print(f"  ✅ {symbol}: ${latest_close:.2f} → ${forecast:.2f} (Confidence: {confidence:.0%})")
        
        return {
            "symbol": symbol,
            "date": get_next_trading_day(market),
            "forecast": round(forecast, 2),
            "confidence": round(confidence, 2),
        }
    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate forecast CSV")
    parser.add_argument(
        "--us-symbols",
        nargs="*",
        default=["AAPL", "GOOGL", "MSFT", "TSLA"],
        help="US stock symbols"
    )
    parser.add_argument(
        "--jp-symbols",
        nargs="*",
        default=["9984", "6758", "7203", "8306"],
        help="JP stock symbols (without .T)"
    )
    parser.add_argument("--output", type=str, default="forecast_data.csv")
    
    args = parser.parse_args()
    
    print("📊 Stock Forecast Generator")
    print("=" * 50)
    print(f"🎯 US Symbols: {args.us_symbols}")
    print(f"🎯 JP Symbols: {args.jp_symbols}")
    print()
    
    records = []
    
    print("📥 Fetching US stocks...")
    for sym in args.us_symbols:
        result = forecast_stock(sym, "US")
        if result:
            records.append(result)
    
    print()
    print("📥 Fetching JP stocks...")
    for sym in args.jp_symbols:
        result = forecast_stock(sym, "JP")
        if result:
            records.append(result)
    
    if not records:
        print("❌ No data fetched")
        return
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print()
    print("=" * 50)
    print(f"✅ Forecast CSV saved: {args.output}")
    print(f"📊 Total records: {len(df)}")
    print()
    print("📋 Preview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
