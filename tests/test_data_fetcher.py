"""Unit tests for the :mod:`src.data_fetcher` module."""

from datetime import date
from unittest.mock import patch

import pandas as pd

from src import data_fetcher


def _mock_history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [1.0, 1.1, 1.2],
            "High": [1.2, 1.3, 1.4],
            "Low": [0.9, 1.0, 1.1],
            "Close": [1.05, 1.15, 1.25],
            "Adj Close": [1.05, 1.15, 1.25],
            "Volume": [100, 110, 120],
        }
    )


def test_resolve_symbol_adds_suffix_for_jp():
    """JP市場では自動的に ``.T`` サフィックスが付与される。"""

    assert data_fetcher._resolve_symbol("7203", "JP") == "7203.T"  # type: ignore[attr-defined]
    assert data_fetcher._resolve_symbol("7203.T", "JP") == "7203.T"  # type: ignore[attr-defined]
    assert data_fetcher._resolve_symbol("NVDA", "US") == "NVDA"  # type: ignore[attr-defined]



@patch("src.data_fetcher.yf.download")
def test_fetch_stock_data_uses_live_data(mock_download):
    """The fetcher normalises the downloaded frame into a tidy format."""

    mock_download.return_value = _mock_history().set_index("Date")

    frame = data_fetcher.fetch_stock_data(
        us_symbols=["NVDA"],

        jp_symbols=[],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    assert list(frame.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "symbol",
        "market",
    ]
    assert frame.iloc[0]["symbol"] == "NVDA"

    assert frame.iloc[0]["market"] == "US"


@patch("src.data_fetcher.yf.download", side_effect=Exception("network error"))
def test_fetch_stock_data_falls_back_to_sample(mock_download):  # noqa: ARG001
    """When yfinance fails the module should fall back to deterministic sample data."""

    frame = data_fetcher.fetch_stock_data(
        us_symbols=["NVDA"],

        jp_symbols=["7203"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )

    assert not frame.empty
    assert set(frame["market"].unique()) == {"US", "JP"}
