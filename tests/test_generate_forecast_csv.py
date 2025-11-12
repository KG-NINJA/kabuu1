from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import generate_forecast_csv as generator


def _history_frame() -> pd.DataFrame:
    dates = pd.to_datetime([
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ])
    return pd.DataFrame(
        {
            "date": dates,
            "close": [100.0, 101.0, 102.0],
            "symbol": [generator.TARGET_SYMBOL] * 3,
            "market": ["US", "US", "US"],
        }
    )


@patch("scripts.generate_forecast_csv.get_next_trading_day")
def test_build_forecast_table_uses_trading_calendar(mock_next_day):
    mock_next_day.return_value = date(2024, 1, 5)
    history = _history_frame()

    table = generator.build_forecast_table(history, days_ahead=1)

    assert list(table.columns) == generator.FORECAST_COLUMNS
    assert table.iloc[0]["symbol"] == generator.TARGET_SYMBOL
    assert table.iloc[0]["market"] == "US"
    assert table.iloc[0]["date"] == "2024-01-05"
    assert table.iloc[0]["current_price"] == pytest.approx(102.0)
    assert table.iloc[0]["forecast"] == pytest.approx(101.7, rel=0.01)
    assert 0.5 <= table.iloc[0]["confidence"] <= 0.95
    mock_next_day.assert_called_once_with(date(2024, 1, 4), "US")


def test_build_forecast_table_empty_history_returns_empty_frame():
    empty = pd.DataFrame(columns=["date", "close", "symbol", "market"])

    result = generator.build_forecast_table(empty)

    assert result.empty
    assert list(result.columns) == generator.FORECAST_COLUMNS


@patch("scripts.generate_forecast_csv.build_forecast_table")
@patch("scripts.generate_forecast_csv.fetch_stock_data")
def test_collect_forecasts_fetches_history(mock_fetch, mock_build):
    mock_history = _history_frame()
    mock_fetch.return_value = mock_history
    expected = pd.DataFrame(
        [
            {
                "symbol": generator.TARGET_SYMBOL,
                "market": "US",
                "date": "2024-01-05",
                "forecast": 101.7,
                "confidence": 0.82,
                "current_price": 102.0,
            }
        ]
    )
    mock_build.return_value = expected

    frame = generator.collect_forecasts(
        [generator.TARGET_SYMBOL],
        ["7203"],
        lookback_days=90,
        days_ahead=2,
    )

    mock_fetch.assert_called_once_with(
        us_symbols=[generator.TARGET_SYMBOL],
        jp_symbols=["7203"],
        start_date=None,
        end_date=None,
        lookback_days=90,
    )
    mock_build.assert_called_once_with(mock_history, days_ahead=2)
    pd.testing.assert_frame_equal(frame, expected)
