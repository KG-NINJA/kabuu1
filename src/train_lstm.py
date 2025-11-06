"""LSTM モデルを構築して学習済みファイルを出力するモジュール。"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - TensorFlow は重いため
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
except Exception:  # pragma: no cover - オプション依存
    tf = None  # type: ignore[assignment]

from preprocessor import Preprocessor


@dataclass
class LSTMTrainer:
    """前処理済みデータから LSTM を学習する。"""

    train_path: Path = Path("data/processed/train_data.csv")
    test_path: Path = Path("data/processed/test_data.csv")
    model_path: Path = Path("models/lstm_model.h5")
    scaler_path: Path = Path("models/lstm_scaler.pkl")
    lookback: int = 10

    def __post_init__(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_or_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Preprocessor]:
        """CSV が無ければ前処理を実行して読み込む。"""
        preprocessor = Preprocessor(
            train_path=self.train_path,
            test_path=self.test_path,
        )
        if not self.train_path.exists() or not self.test_path.exists():
            preprocessor.preprocess()
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        return train_df, test_df, preprocessor

    def _create_sequences(
        self, dataframe: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """LSTM 入力用に時系列シーケンスへ変換する。"""
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if "Close" not in dataframe.columns:
            raise ValueError("Close 列が見つかりません")
        target = dataframe["Close"].to_numpy(dtype=np.float32)
        features = dataframe[numeric_columns].to_numpy(dtype=np.float32)
        if len(features) <= self.lookback:
            padded = np.tile(features, (self.lookback + 1, 1))
            features = padded
            target = np.tile(target, self.lookback + 1)
        sequences = []
        labels = []
        for index in range(self.lookback, len(features)):
            start = index - self.lookback
            sequences.append(features[start:index])
            labels.append(target[index])
        return np.array(sequences), np.array(labels), numeric_columns

    def train(self, epochs: int = 2, batch_size: int = 32) -> float:
        """LSTM を学習しモデルとスケーラーを保存する。"""
        if tf is None:  # pragma: no cover - TensorFlow 未導入時
            raise ImportError("TensorFlow がインストールされていません")
        train_df, test_df, preprocessor = self._load_or_prepare_data()
        _, _, scaler = preprocessor.preprocess()
        x_train, y_train, feature_names = self._create_sequences(train_df)
        x_test, y_test, _ = self._create_sequences(test_df)
        model = Sequential(
            [
                LSTM(
                    64,
                    activation="tanh",
                    return_sequences=True,
                    input_shape=(self.lookback, len(feature_names)),
                ),
                LSTM(32, activation="tanh", return_sequences=True),
                LSTM(16, activation="tanh"),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        tf.random.set_seed(42)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        loss = float(model.evaluate(x_test, y_test, verbose=0))
        model.save(self.model_path)
        with self.scaler_path.open("wb") as handle:
            pickle.dump(scaler, handle)
        return loss


def train_lstm_model() -> float:
    """LSTMTrainer を利用してモデルを学習する。"""
    trainer = LSTMTrainer()
    return trainer.train()


__all__ = ["LSTMTrainer", "train_lstm_model"]
