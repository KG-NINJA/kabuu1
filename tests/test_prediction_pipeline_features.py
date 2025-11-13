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



def test_run_exits_after_duration(monkeypatch: pytest.MonkeyPatch, configured_pipeline):
    from src import prediction_pipeline as pipeline_module

    call_count = {"value": 0}

    def fake_run_pending():
        call_count["value"] += 1

    readings = iter([0.0, 30.0, 61.0, 61.0])

    def fake_monotonic():
        try:
            return next(readings)
        except StopIteration:
            return 61.0

    monkeypatch.setattr(pipeline_module.schedule, "run_pending", fake_run_pending)
    monkeypatch.setattr(pipeline_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(pipeline_module.time, "sleep", lambda _: None)

    configured_pipeline.run(duration_minutes=1, sleep_seconds=5)

    assert call_count["value"] >= 1


def test_run_cycle_respects_flags(monkeypatch: pytest.MonkeyPatch, configured_pipeline):
    calls: list[str] = []

    monkeypatch.setattr(
        configured_pipeline,
        "predict_all_tickers",
        lambda: calls.append("predict"),
    )
    monkeypatch.setattr(
        configured_pipeline,
        "collect_actual_price",
        lambda: calls.append("actual"),
    )
    monkeypatch.setattr(
        configured_pipeline,
        "weekly_model_review",
        lambda: calls.append("review"),
    )

    configured_pipeline.run_cycle(run_prediction=True, run_actuals=False, run_review=True)

    assert calls == ["predict", "review"]

