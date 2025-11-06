"""株価データの取得とダミーデータ生成を担当するモジュール。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - ランタイム環境に依存
    import yfinance as yf
except Exception:  # pragma: no cover - オプション依存
    yf = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

DEFAULT_TICKERS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "TSLA",
    "9984.T",
    "6758.T",
    "7203.T",
    "8306.T",
]


@dataclass
class FetchResult:
    """取得したデータフレームと保存先パスを保持する。"""

    ticker: str
    dataframe: pd.DataFrame
    path: Path


@dataclass
class DataFetcher:
    """yfinance を利用した株価ダウンローダー。"""

    tickers: Iterable[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    data_dir: Path = Path("data/raw")
    period: str = "1y"
    interval: str = "1d"

    def __post_init__(self) -> None:
        """保存先ディレクトリを作成する。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all(self) -> List[FetchResult]:
        """全銘柄のデータを取得し CSV に保存する。"""
        results: List[FetchResult] = []
        for ticker in self.tickers:
            dataframe = self._download_ticker(ticker)
            dataframe["ticker"] = ticker
            path = self.data_dir / f"{ticker.replace('.', '_')}.csv"
            dataframe.to_csv(path, index=False)
            results.append(FetchResult(ticker=ticker, dataframe=dataframe, path=path))
        if results:
            combined = pd.concat([res.dataframe for res in results], ignore_index=True)
            combined_path = self.data_dir / "combined.csv"
            combined.to_csv(combined_path, index=False)
            LOGGER.info("株価データを %s に保存しました", combined_path)
        return results

    def _download_ticker(self, ticker: str) -> pd.DataFrame:
        """単一銘柄のデータを取得する。失敗時はダミーデータを生成する。"""
        dataframe: Optional[pd.DataFrame] = None
        if yf is not None:
            try:
                dataframe = yf.download(
                    ticker,
                    period=self.period,
                    interval=self.interval,
                    progress=False,
                )
            except Exception as error:  # pragma: no cover - ネットワーク依存
                LOGGER.warning("%s の取得に失敗しました: %s", ticker, error)
        if dataframe is None or dataframe.empty:
            LOGGER.info("%s のダミーデータを生成します", ticker)
            dataframe = self._generate_dummy_data(ticker)
        dataframe = dataframe.reset_index(drop=False)
        dataframe = dataframe.rename(columns={"Date": "date"})
        for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if column not in dataframe:
                dataframe[column] = dataframe.get("Close", pd.Series(dtype=float))
        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
        dataframe[numeric_columns] = (
            dataframe[numeric_columns].fillna(method="ffill").bfill()
        )
        dataframe["date"] = pd.to_datetime(dataframe["date"]).dt.strftime("%Y-%m-%d")
        return dataframe

    def _generate_dummy_data(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """トレード日数分のダミー株価を生成する。"""
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        dates = pd.date_range(start=start, end=end, freq="B")
        rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
        base_price = 100 + rng.normal(scale=5.0)
        changes = rng.normal(loc=0.0, scale=1.5, size=len(dates))
        prices = base_price + np.cumsum(changes)
        highs = prices + rng.normal(loc=1.0, scale=0.5, size=len(dates))
        lows = prices - rng.normal(loc=1.0, scale=0.5, size=len(dates))
        volumes = rng.integers(low=500_000, high=5_000_000, size=len(dates))
        dataframe = pd.DataFrame(
            {
                "date": dates,
                "Open": prices,
                "High": highs,
                "Low": lows,
                "Close": prices,
                "Adj Close": prices,
                "Volume": volumes,
            }
        )
        return dataframe


def fetch_and_save() -> Dict[str, Path]:
    """DataFetcher を利用してデータを取得し、保存パスを返す。"""
    fetcher = DataFetcher()
    results = fetcher.fetch_all()
    return {result.ticker: result.path for result in results}


__all__ = ["DataFetcher", "fetch_and_save", "FetchResult"]
