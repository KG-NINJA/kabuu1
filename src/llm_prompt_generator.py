"""Generate LLM-ready prompt JSON from forecast CSV or live data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data_fetcher import fetch_stock_data
from src.predict import (
    generate_llm_prompts,
    prepare_forecast_dataframe,
    save_json_output,
    save_forecast_csv,
    _parse_optional_date,
    _setup_logging,
)


def _default_symbols() -> List[str]:
    return ["AAPL", "GOOGL", "MSFT", "TSLA"]


def _default_jp_symbols() -> List[str]:
    return ["9984", "6758", "7203", "8306"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM prompt generator")
    parser.add_argument("--forecast-csv", type=str, help="予測CSVのパス", default="data/stock_data/predictions/forecast.csv")
    parser.add_argument("--json-output", type=str, default="darwin_analysis/llm_prompts/forecast.json", help="LLM JSONの出力先")
    parser.add_argument("--us-symbols", nargs="*", default=_default_symbols(), help="米国株シンボル")
    parser.add_argument("--jp-symbols", nargs="*", default=_default_jp_symbols(), help="日本株シンボル")
    parser.add_argument("--days-ahead", type=int, default=1, help="予測営業日数")
    parser.add_argument("--lookback-days", type=int, default=365, help="履歴取得日数")
    parser.add_argument("--start-date", type=str, help="履歴開始日")
    parser.add_argument("--end-date", type=str, help="履歴終了日")
    parser.add_argument("--log", type=str, help="ログファイルのパス")
    return parser


def load_or_create_forecast(
    forecast_csv: Path,
    us_symbols: List[str],
    jp_symbols: List[str],
    days_ahead: int,
    lookback_days: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    if forecast_csv.exists():
        logging.info("Using existing forecast CSV: %s", forecast_csv)
        return pd.read_csv(forecast_csv)

    logging.info("Forecast CSV not found, fetching fresh data")
    history_df = fetch_stock_data(
        us_symbols=us_symbols,
        jp_symbols=jp_symbols,
        start_date=_parse_optional_date(start_date),
        end_date=_parse_optional_date(end_date),
        lookback_days=lookback_days,
    )

    forecast_df = prepare_forecast_dataframe(history_df, days_ahead=days_ahead)
    save_forecast_csv(forecast_df, str(forecast_csv))
    return forecast_df


def main(argv: Optional[List[str]] = None) -> Path:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.log)

    forecast_path = Path(args.forecast_csv)
    forecast_path.parent.mkdir(parents=True, exist_ok=True)

    forecast_df = load_or_create_forecast(
        forecast_path,
        us_symbols=list(args.us_symbols),
        jp_symbols=list(args.jp_symbols),
        days_ahead=args.days_ahead,
        lookback_days=args.lookback_days,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    result = generate_llm_prompts(forecast_df=forecast_df)
    output_path = save_json_output(result, args.json_output)
    logging.info("Generated LLM prompt JSON: %s", output_path)
    return Path(output_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
