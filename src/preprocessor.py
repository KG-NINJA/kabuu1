"""特徴量データを正規化し学習・評価データに分割する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_engineering import FeatureEngineer


@dataclass
class Preprocessor:
    """正規化とデータ分割を提供する。"""

    features_path: Path = Path("data/processed/features.csv")
    train_path: Path = Path("data/processed/train_data.csv")
    test_path: Path = Path("data/processed/test_data.csv")

    def load_features(self) -> pd.DataFrame:
        """特徴量 CSV を読み込む。無ければ生成する。"""
        if not self.features_path.exists():
            FeatureEngineer().engineer()
        dataframe = pd.read_csv(self.features_path)
        if "date" in dataframe:
            dataframe["date"] = pd.to_datetime(dataframe["date"])
        return dataframe

    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """前処理を行い、訓練・テストデータとスケーラーを返す。"""
        dataframe = self.load_features()
        sort_columns = ["ticker"]
        if "date" in dataframe.columns:
            sort_columns.append("date")
        dataframe = dataframe.sort_values(sort_columns).reset_index(drop=True)
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        dataframe[numeric_columns] = dataframe[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        dataframe[numeric_columns] = dataframe[numeric_columns].ffill().bfill()
        dataframe[numeric_columns] = dataframe[numeric_columns].fillna(
            dataframe[numeric_columns].mean()
        )
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(dataframe[numeric_columns])
        scaled_df = dataframe.copy()
        scaled_df[numeric_columns] = scaled_values
        split_index = max(int(len(scaled_df) * 0.8), 1)
        train_df = scaled_df.iloc[:split_index].reset_index(drop=True)
        test_df = scaled_df.iloc[split_index:].reset_index(drop=True)
        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)
        return train_df, test_df, scaler


def run_preprocessing() -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """前処理を実行するヘルパー関数。"""
    preprocessor = Preprocessor()
    return preprocessor.preprocess()


__all__ = ["Preprocessor", "run_preprocessing"]
