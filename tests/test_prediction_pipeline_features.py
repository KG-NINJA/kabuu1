from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.prediction_pipeline import Config, PredictionModel, PredictionPipeline


def _build_history(rows: int = 120) -> pd.DataFrame:
    today = datetime.now(UTC).date()
    dates = pd.bdate_range(end=today, periods=rows).to_pydatetime()
    records = []
    base_price = 400.0
    for index, ts in enumerate(dates):
        close_price = base_price + index * 0.5
        records.append(
            {
                "date": ts,
                "open": close_price - 1.0,
                "high": close_price + 1.5,
                "low": close_price - 2.0,
                "close": close_price,
                "adj_close": close_price,
                "volume": 1_000_000 + index * 5_000,
                "symbol": "NVDA",
                "market": "US",
            }
        )
    return pd.DataFrame(records)


@pytest.fixture()
def sample_history() -> pd.DataFrame:
    return _build_history()


@pytest.fixture()
def configured_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path, sample_history):
    from src import prediction_pipeline

    def fake_fetch_stock_data(*args, **kwargs):
        return sample_history.copy()

    monkeypatch.setattr(prediction_pipeline, "fetch_stock_data", fake_fetch_stock_data)

    config = Config()
    config._config.setdefault("nvda", {})
    config._config["nvda"].update(
        {
            "symbol": "NVDA",
            "base_dir": str(tmp_path / "nvda_learning"),
            "reward_threshold": 0.5,
        }
    )
    config._config.setdefault("features", {})["lookback_days"] = 60

    pipeline = PredictionPipeline(config)
    pipeline.metrics_path = tmp_path / "performance_metrics.json"
    pipeline.pending_predictions = []
    return pipeline


def test_prepare_features_uses_recent_history(configured_pipeline, sample_history):
    features = configured_pipeline.prepare_features("NVDA")

    assert set(PredictionModel.feature_columns).issubset(features)
    last_close = sample_history["close"].iloc[-1]
    expected_ma5 = sample_history["close"].iloc[-5:].mean()
    expected_ma20 = sample_history["close"].iloc[-20:].mean()

    assert features["close"] == pytest.approx(last_close, rel=1e-5)
    assert features["ma_5"] == pytest.approx(expected_ma5, rel=1e-5)
    assert features["ma_20"] == pytest.approx(expected_ma20, rel=1e-5)
    assert "as_of" in features


def test_collect_actual_price_confirms_pending_prediction(configured_pipeline, sample_history, tmp_path):
    configured_pipeline.metrics_path = tmp_path / "metrics.json"
    configured_pipeline.pending_predictions = [
        {
            "ticker": "NVDA",
            "predicted_price": 450.0,
            "features": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "pending",
        }
    ]

    configured_pipeline.collect_actual_price()

    assert configured_pipeline.pending_predictions == []
