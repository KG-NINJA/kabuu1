"""XGBoost リグレッサーを学習しモデルファイルを保存する。"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - XGBoost はオプション依存
    import xgboost as xgb
except Exception:  # pragma: no cover - インポート失敗時
    xgb = None  # type: ignore[assignment]

from preprocessor import Preprocessor


@dataclass
class XGBoostTrainer:
    """前処理済みデータで XGBoost モデルを訓練する。"""

    train_path: Path = Path("data/processed/train_data.csv")
    test_path: Path = Path("data/processed/test_data.csv")
    model_path: Path = Path("models/xgboost_model.pkl")

    def __post_init__(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """CSV が存在しない場合は前処理を実行する。"""
        preprocessor = Preprocessor(
            train_path=self.train_path,
            test_path=self.test_path,
        )
        if not self.train_path.exists() or not self.test_path.exists():
            preprocessor.preprocess()
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        return train_df, test_df

    def _split_features(self, dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """数値列を特徴量とターゲットに分割する。"""
        if "Close" not in dataframe.columns:
            raise ValueError("Close 列が見つかりません")
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col != "Close"]
        if not feature_columns:
            raise ValueError("特徴量が不足しています")
        features = dataframe[feature_columns].to_numpy(dtype=np.float32)
        target = dataframe["Close"].to_numpy(dtype=np.float32)
        return features, target

    def train(self) -> float:
        """XGBoost モデルを学習して保存する。"""
        if xgb is None:  # pragma: no cover - ライブラリ未導入時
            raise ImportError("XGBoost がインストールされていません")
        train_df, test_df = self._load_datasets()
        x_train, y_train = self._split_features(train_df)
        x_test, y_test = self._split_features(test_df)
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(x_train, y_train, verbose=False)
        predictions = model.predict(x_test)
        rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
        with self.model_path.open("wb") as handle:
            pickle.dump(model, handle)
        return rmse


def train_xgboost_model() -> float:
    """XGBoostTrainer を用いて学習を実行する。"""
    trainer = XGBoostTrainer()
    return trainer.train()


__all__ = ["XGBoostTrainer", "train_xgboost_model"]
