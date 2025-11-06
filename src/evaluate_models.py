"""学習済みモデルの性能を評価しレポートを出力する。"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessor import Preprocessor
from train_lstm import LSTMTrainer, tf
from train_xgboost import XGBoostTrainer, xgb


@dataclass
class EvaluationResult:
    """単一モデルの評価結果を保持する。"""

    model_name: str
    mae: float
    rmse: float
    r2: float

    def format(self) -> str:
        """テキストファイル用のフォーマットを返す。"""
        return (
            f"Model: {self.model_name}\n"
            f"  MAE : {self.mae:.4f}\n"
            f"  RMSE: {self.rmse:.4f}\n"
            f"  R2  : {self.r2:.4f}\n"
        )


@dataclass
class ModelEvaluator:
    """LSTM と XGBoost の性能を比較する。"""

    test_path: Path = Path("data/processed/test_data.csv")
    report_path: Path = Path("models/evaluation_report.txt")

    def __post_init__(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_test_data(self) -> pd.DataFrame:
        """テストデータをロードする。必要に応じて前処理を実行する。"""
        preprocessor = Preprocessor(test_path=self.test_path)
        if not self.test_path.exists():
            preprocessor.preprocess()
        dataframe = pd.read_csv(self.test_path)
        if dataframe.empty:
            raise ValueError("テストデータが空です")
        return dataframe

    def evaluate(self) -> List[EvaluationResult]:
        """全モデルの評価を実施しレポートを保存する。"""
        dataframe = self._load_test_data()
        results: List[EvaluationResult] = []
        lstm_result = self._evaluate_lstm(dataframe)
        if lstm_result is not None:
            results.append(lstm_result)
        xgb_result = self._evaluate_xgboost(dataframe)
        if xgb_result is not None:
            results.append(xgb_result)
        if not results:
            raise RuntimeError("評価可能なモデルが存在しません")
        self._write_report(results)
        return results

    def _evaluate_lstm(self, dataframe: pd.DataFrame) -> Optional[EvaluationResult]:
        """LSTM モデルを評価する。ファイルが無い場合は None を返す。"""
        model_path = LSTMTrainer().model_path
        scaler_path = LSTMTrainer().scaler_path
        if tf is None or not model_path.exists() or not scaler_path.exists():
            return None
        trainer = LSTMTrainer()
        sequences, labels, _ = trainer._create_sequences(dataframe)
        model = tf.keras.models.load_model(model_path)
        predictions = model.predict(sequences, verbose=0).flatten()
        mae = float(mean_absolute_error(labels, predictions))
        rmse = float(np.sqrt(mean_squared_error(labels, predictions)))
        r2 = float(r2_score(labels, predictions))
        return EvaluationResult("LSTM", mae, rmse, r2)

    def _evaluate_xgboost(self, dataframe: pd.DataFrame) -> Optional[EvaluationResult]:
        """XGBoost モデルを評価する。"""
        model_path = XGBoostTrainer().model_path
        if xgb is None or not model_path.exists():
            return None
        trainer = XGBoostTrainer()
        features, target = trainer._split_features(dataframe)
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        predictions = model.predict(features)
        mae = float(mean_absolute_error(target, predictions))
        rmse = float(np.sqrt(mean_squared_error(target, predictions)))
        r2 = float(r2_score(target, predictions))
        return EvaluationResult("XGBoost", mae, rmse, r2)

    def _write_report(self, results: List[EvaluationResult]) -> None:
        """評価結果をテキストレポートとして保存する。"""
        lines = ["Model Evaluation Report", "======================"]
        for result in results:
            lines.append("")
            lines.append(result.format())
        self.report_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_models() -> List[EvaluationResult]:
    """ModelEvaluator を実行して評価結果を返す。"""
    evaluator = ModelEvaluator()
    return evaluator.evaluate()


__all__ = ["ModelEvaluator", "EvaluationResult", "evaluate_models"]
