#!/usr/bin/env python3
"""Generate a lightweight forecast CSV using the shared data fetcher.

This script retrieves historical price data through :mod:`src.data_fetcher`
so that local runs and the GitHub Actions workflows share the exact same
logic, including the deterministic sample fallback when the network is not
available.  The resulting CSV now contains the current price column expected
by ``src.predict`` which allows the validation pipeline to consume the live
forecasts without falling back to mock data.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.data_fetcher import fetch_stock_data
from src.validation_helpers import get_next_trading_day

FORECAST_COLUMNS = [
    "symbol",
    "market",
    "date",
    "forecast",
    "confidence",
    "current_price",
]


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _calculate_confidence(close: pd.Series) -> float:
    volatility = close.pct_change().std()
    if pd.isna(volatility):
        volatility = 0.0
    return max(0.5, min(0.95, 1.0 - float(volatility) * 2))


def build_forecast_table(history: pd.DataFrame, days_ahead: int = 1) -> pd.DataFrame:
    """Create a forecast table from a historical price DataFrame."""

    if history is None or history.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)

    working = history.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["close"] = pd.to_numeric(working["close"], errors="coerce")
    working.dropna(subset=["date", "close", "symbol"], inplace=True)

    if working.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)

    records = []
    grouped = working.sort_values("date").groupby(["symbol", "market"], sort=True)

    for (symbol, market), group in grouped:
        if group.empty:
            continue

        close_series = group["close"].astype(float)
        latest_close = float(close_series.iloc[-1])
        ma_5 = close_series.tail(5).mean()
        ma_20 = close_series.tail(20).mean()
        forecast_price = (latest_close * 0.7) + (ma_5 * 0.2) + (ma_20 * 0.1)

        confidence = _calculate_confidence(close_series)

        last_trading_day = group["date"].iloc[-1].date()
        forecast_date = last_trading_day
        for _ in range(max(days_ahead, 1)):
            forecast_date = get_next_trading_day(forecast_date, str(market))

        records.append(
            {
                "symbol": str(symbol),
                "market": str(market),
                "date": forecast_date.isoformat(),
                "forecast": round(float(forecast_price), 2),
                "confidence": round(float(confidence), 2),
                "current_price": round(latest_close, 2),
            }
        )

    return pd.DataFrame(records, columns=FORECAST_COLUMNS)


def collect_forecasts(
    us_symbols: Sequence[str],
    jp_symbols: Sequence[str],
    *,
    lookback_days: int = 365,
    days_ahead: int = 1,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch history for the requested symbols and build the forecast table."""

    history = fetch_stock_data(
        us_symbols=us_symbols,
        jp_symbols=jp_symbols,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
    )
    return build_forecast_table(history, days_ahead=days_ahead)


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate forecast CSV")
    parser.add_argument(
        "--us-symbols",
        nargs="*",
        default=["AAPL", "GOOGL", "MSFT", "TSLA"],
        help="US stock symbols",
    )
    parser.add_argument(
        "--jp-symbols",
        nargs="*",
        default=["9984", "6758", "7203", "8306"],
        help="JP stock symbols (without .T)",
    )
    parser.add_argument("--output", type=str, default="forecast_data.csv")
    parser.add_argument("--lookback-days", type=int, default=365, help="History window length")
    parser.add_argument("--days-ahead", type=int, default=1, help="How many trading days ahead to forecast")
    parser.add_argument("--start-date", type=str, help="Optional history start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Optional history end date (YYYY-MM-DD)")
    return parser


def main() -> None:
    parser = _create_argument_parser()
    args = parser.parse_args()

    us_symbols = [symbol for symbol in args.us_symbols if symbol]
    jp_symbols = [symbol for symbol in args.jp_symbols if symbol]

    print("📊 Stock Forecast Generator")
    print("=" * 50)
    print(f"🎯 US Symbols: {us_symbols or 'None'}")
    print(f"🎯 JP Symbols: {jp_symbols or 'None'}")
    print(f"📈 Lookback days: {args.lookback_days}")
    print(f"📅 Days ahead: {args.days_ahead}")
    print()

    forecast_table = collect_forecasts(
        us_symbols,
        jp_symbols,
        lookback_days=args.lookback_days,
        days_ahead=args.days_ahead,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
    )

    if forecast_table.empty:
        print("❌ No data fetched")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_table.to_csv(output_path, index=False)

    print()
    print("=" * 50)
    print(f"✅ Forecast CSV saved: {output_path}")
    print(f"📊 Total records: {len(forecast_table)}")
    print()
    print("📋 Preview:")
    print(forecast_table.to_string(index=False))


if __name__ == "__main__":
    main()
