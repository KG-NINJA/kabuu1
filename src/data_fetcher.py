"""Stock market data fetching utilities.

This module downloads historical price data for US and Japanese equities
using the `yfinance` package.  It provides a small command line interface
that mirrors the arguments used inside the GitHub Actions workflow so the
automation can persist a CSV file with the raw prices.  When network access
is unavailable (for example inside CI during tests) the module falls back to
deterministic sample data so downstream stages still receive a well-formed
dataset.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf


LOGGER = logging.getLogger(__name__)

__all__ = [
    "fetch_stock_data",
    "save_stock_data",
    "main",
]


@dataclass
class FetchConfig:
    """Configuration for a fetch run."""

    us_symbols: List[str]
    jp_symbols: List[str]
    start_date: date
    end_date: date
    output: Optional[Path] = None


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _resolve_symbol(symbol: str, market: str) -> str:
    """Return a ticker symbol formatted for Yahoo Finance."""

    if market == "JP":
        if symbol.endswith(".T"):
            return symbol
        if symbol.isdigit():
            return f"{symbol}.T"
        return f"{symbol}.T"
    return symbol


def _configure_logging(log_path: Optional[str]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def _fetch_symbol_history(symbol: str, market: str, start: date, end: date) -> pd.DataFrame:
    """Download price history for a single symbol.

    When the download fails an empty DataFrame is returned so the caller can
    gracefully fall back to mock data.
    """

    formatted_symbol = _resolve_symbol(symbol, market)
    LOGGER.info(
        "Fetching %s data for %s between %s and %s",
        market,
        formatted_symbol,
        start.isoformat(),
        end.isoformat(),
    )
    try:
        data = yf.download(
            formatted_symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        LOGGER.warning("Failed to download %s: %s", formatted_symbol, exc)
        return pd.DataFrame()

    if data.empty:
        LOGGER.warning("No data returned for %s", formatted_symbol)
        return pd.DataFrame()

    data = data.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    data["symbol"] = symbol
    data["market"] = market
    return data


def _create_sample_history(symbols: Iterable[str], market: str, start: date, end: date) -> pd.DataFrame:
    """Create deterministic sample data when live fetching is unavailable."""

    days = max((end - start).days, 5)
    base_dates = [end - timedelta(days=offset) for offset in range(days)][::-1]
    rows = []
    for index, symbol in enumerate(symbols):
        base_price = 100 + (index * 25)
        for i, target_date in enumerate(base_dates[-30:]):
            close_price = base_price * (1 + 0.002 * i)
            rows.append(
                {
                    "date": target_date,
                    "open": close_price * 0.99,
                    "high": close_price * 1.01,
                    "low": close_price * 0.98,
                    "close": close_price,
                    "adj_close": close_price,
                    "volume": 1_000_000 + 10_000 * i,
                    "symbol": symbol,
                    "market": market,
                }
            )
    return pd.DataFrame(rows)


def fetch_stock_data(
    us_symbols: Iterable[str],
    jp_symbols: Iterable[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Fetch stock history for the requested symbols."""

    if end_date is None:
        end_date = datetime.utcnow().date()
    if start_date is None:
        start_date = end_date - timedelta(days=lookback_days)

    frames: List[pd.DataFrame] = []

    for symbol in us_symbols:
        frame = _fetch_symbol_history(symbol, "US", start_date, end_date)
        if not frame.empty:
            frames.append(frame)
    for symbol in jp_symbols:
        frame = _fetch_symbol_history(symbol, "JP", start_date, end_date)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        LOGGER.warning("Falling back to sample data because live fetch returned nothing")
        frames = []
        if us_symbols:
            frames.append(_create_sample_history(us_symbols, "US", start_date, end_date))
        if jp_symbols:
            frames.append(_create_sample_history(jp_symbols, "JP", start_date, end_date))
        if not frames:
            frames.append(_create_sample_history(["SAMPLE"], "US", start_date, end_date))

    data = pd.concat(frames, ignore_index=True)
    data.sort_values(["symbol", "date"], inplace=True)
    return data


def save_stock_data(data: pd.DataFrame, output_path: Path) -> Path:
    """Persist the fetched history to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = data.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"]).dt.strftime("%Y-%m-%d")
    cleaned.to_csv(output_path, index=False)
    LOGGER.info("Saved %s rows to %s", len(cleaned), output_path)
    return output_path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch historical stock data")
    parser.add_argument("--us-symbols", nargs="*", default=[], help="US ticker symbols")
    parser.add_argument("--jp-symbols", nargs="*", default=[], help="JP ticker symbols")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD", default=None)
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD", default=None)
    parser.add_argument("--lookback-days", type=int, default=365, help="Lookback period when start date omitted")
    parser.add_argument("--output", type=str, help="Output CSV path", default="data/stock_data/raw_data.csv")
    parser.add_argument("--log", type=str, help="Optional log file path")
    return parser


def main(argv: Optional[List[str]] = None) -> Path:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log)

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    data = fetch_stock_data(
        us_symbols=args.us_symbols,
        jp_symbols=args.jp_symbols,
        start_date=start_date,
        end_date=end_date,
        lookback_days=args.lookback_days,
    )

    output_path = save_stock_data(data, Path(args.output))
    LOGGER.info("Historical data collection complete")
    return output_path


if __name__ == "__main__":
    main()
