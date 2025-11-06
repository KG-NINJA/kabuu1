"""保存済みモデルを読み込み株価予測を生成する。"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_engineering import FeatureEngineer
from preprocessor import Preprocessor
from train_lstm import LSTMTrainer, tf
from train_xgboost import XGBoostTrainer, xgb


@dataclass
class Predictor:
    """LSTM と XGBoost の予測を組み合わせて 5 営業日先を推定する。"""

    horizon: int = 5
    output_path: Path = Path("data/forecast.csv")

    def __post_init__(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_scaler(self) -> Optional[StandardScaler]:
        """LSTM で保存したスケーラーを読み込む。"""
        scaler_path = LSTMTrainer().scaler_path
        if scaler_path.exists():
            with scaler_path.open("rb") as handle:
                return pickle.load(handle)
        return None

    def _inverse_close(self, scaler: Optional[StandardScaler], value: float) -> float:
        """標準化された Close を元スケールへ戻す。"""
        if scaler is None or not hasattr(scaler, "mean_"):
            return float(value)
        names = getattr(scaler, "feature_names_in_", None)
        if names is None or "Close" not in names:
            return float(value)
        index = list(names).index("Close")
        mean = float(scaler.mean_[index])
        scale = float(scaler.scale_[index])
        return float(value * scale + mean)

    @dataclass
    class TickerDataset:
        raw: pd.DataFrame
        scaled: pd.DataFrame
        scaler: Optional[StandardScaler]

    def _prepare_datasets(self) -> Dict[str, "Predictor.TickerDataset"]:
        """最新の特徴量とスケーリング済みデータを整形する。"""
        features = FeatureEngineer().engineer()
        train_df, test_df, scaler = Preprocessor().preprocess()
        scaler_obj: Optional[StandardScaler] = scaler
        scaled_df = pd.concat([train_df, test_df], ignore_index=True)
        scaler_from_file = self._load_scaler()
        if scaler_from_file is not None:
            scaler_obj = scaler_from_file
        grouped: Dict[str, Predictor.TickerDataset] = {}
        for ticker, group in features.groupby("ticker", sort=False):
            scaled_group = scaled_df[scaled_df["ticker"] == ticker]
            if scaled_group.empty:
                continue
            grouped[ticker] = Predictor.TickerDataset(
                raw=group.copy(),
                scaled=scaled_group.copy(),
                scaler=scaler_obj,
            )
        return grouped

    def _predict_with_lstm(
        self,
        scaled: pd.DataFrame,
        scaler: Optional[StandardScaler],
    ) -> float | None:
        """LSTM モデルが利用可能な場合に予測を実行する。"""
        model_path = LSTMTrainer().model_path
        if tf is None or not model_path.exists():
            return None
        trainer = LSTMTrainer()
        sequences, _, _ = trainer._create_sequences(scaled)
        if len(sequences) == 0:
            return None
        model = tf.keras.models.load_model(model_path)
        prediction = float(model.predict(sequences[-1:], verbose=0).flatten()[0])
        return self._inverse_close(scaler, prediction)

    def _predict_with_xgb(
        self, scaled: pd.DataFrame, scaler: Optional[StandardScaler]
    ) -> float | None:
        """XGBoost モデルの予測値を返す。"""
        model_path = XGBoostTrainer().model_path
        if xgb is None or not model_path.exists():
            return None
        trainer = XGBoostTrainer()
        features, _ = trainer._split_features(scaled)
        if len(features) == 0:
            return None
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        prediction = float(model.predict(features[-1:])[0])
        return self._inverse_close(scaler, prediction)

    def _generate_forecast(
        self,
        ticker: str,
        raw: pd.DataFrame,
        scaled: pd.DataFrame,
        scaler: Optional[StandardScaler],
    ) -> List[Dict[str, object]]:
        """単一銘柄の 5 営業日予測を生成する。"""
        last_close = float(raw["Close"].iloc[-1])
        lstm_value = self._predict_with_lstm(scaled, scaler)
        xgb_value = self._predict_with_xgb(scaled, scaler)
        candidates = [value for value in [lstm_value, xgb_value] if value is not None]
        predicted_base = float(np.mean(candidates)) if candidates else last_close
        drift = (predicted_base - last_close) / max(self.horizon, 1)
        last_date = pd.to_datetime(raw["date"].iloc[-1])
        future_dates = pd.bdate_range(
            last_date + timedelta(days=1), periods=self.horizon
        )
        records: List[Dict[str, object]] = []
        for index, target_date in enumerate(future_dates, start=1):
            price = last_close + drift * index
            records.append(
                {
                    "ticker": ticker,
                    "days_ahead": index,
                    "target_date": target_date.date().isoformat(),
                    "predicted_close": round(float(price), 4),
                }
            )
        return records

    def predict(self) -> pd.DataFrame:
        """全銘柄の予測を計算し CSV に保存する。"""
        datasets = self._prepare_datasets()
        if not datasets:
            raise RuntimeError("予測対象データが見つかりません")
        forecasts: List[Dict[str, object]] = []
        for ticker, payload in datasets.items():
            forecasts.extend(
                self._generate_forecast(
                    ticker,
                    payload.raw,
                    payload.scaled,
                    payload.scaler,
                )
            )
        forecast_df = pd.DataFrame(forecasts)
        forecast_df.to_csv(self.output_path, index=False)
        return forecast_df


def run_prediction() -> pd.DataFrame:
    """Predictor を実行して予測結果を返す。"""
    predictor = Predictor()
    return predictor.predict()


__all__ = ["Predictor", "run_prediction"]
