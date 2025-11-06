"""テクニカル指標を計算して特徴量 CSV を生成するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from data_fetcher import DataFetcher


@dataclass
class FeatureEngineer:
    """移動平均や RSI などの指標を計算する。"""

    input_path: Path = Path("data/raw/combined.csv")
    output_path: Path = Path("data/processed/features.csv")

    def __post_init__(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_source(self) -> pd.DataFrame:
        """入力データが無い場合は DataFetcher で生成する。"""
        if not self.input_path.exists():
            fetcher = DataFetcher()
            fetcher.fetch_all()
        dataframe = pd.read_csv(self.input_path)
        if "date" in dataframe:
            dataframe["date"] = pd.to_datetime(dataframe["date"])
        return dataframe

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """標準的な RSI を計算する。"""
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(method="bfill").fillna(50)

    def _calculate_macd(self, close: pd.Series) -> pd.DataFrame:
        """MACD とシグナル、ヒストグラムを計算する。"""
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return pd.DataFrame({"MACD": macd, "MACD_signal": signal, "MACD_hist": hist})

    def _calculate_bbands(self, close: pd.Series, window: int = 20) -> pd.DataFrame:
        """ボリンジャーバンド (±2σ) を算出する。"""
        rolling_mean = close.rolling(window=window, min_periods=window).mean()
        rolling_std = close.rolling(window=window, min_periods=window).std()
        upper = rolling_mean + 2 * rolling_std
        lower = rolling_mean - 2 * rolling_std
        return pd.DataFrame(
            {
                "BB_upper": upper,
                "BB_middle": rolling_mean,
                "BB_lower": lower,
            }
        )

    def engineer(self) -> pd.DataFrame:
        """テクニカル指標を計算して保存する。"""
        dataframe = self.load_source()
        if "ticker" not in dataframe.columns:
            dataframe["ticker"] = "UNKNOWN"
        sort_columns = ["ticker"]
        if "date" in dataframe.columns:
            sort_columns.append("date")
        dataframe = dataframe.sort_values(sort_columns).reset_index(drop=True)
        dataframe["MA_5"] = dataframe.groupby("ticker")["Close"].transform(
            lambda series: series.rolling(window=5, min_periods=5).mean()
        )
        dataframe["MA_20"] = dataframe.groupby("ticker")["Close"].transform(
            lambda series: series.rolling(window=20, min_periods=20).mean()
        )
        dataframe["MA_50"] = dataframe.groupby("ticker")["Close"].transform(
            lambda series: series.rolling(window=50, min_periods=50).mean()
        )
        dataframe["RSI_14"] = dataframe.groupby("ticker")["Close"].transform(
            self._calculate_rsi
        )
        dataframe[["MACD", "MACD_signal", "MACD_hist"]] = 0.0
        dataframe[["BB_upper", "BB_middle", "BB_lower"]] = 0.0
        for ticker, group in dataframe.groupby("ticker", sort=False):
            macd = self._calculate_macd(group["Close"])
            bbands = self._calculate_bbands(group["Close"])
            dataframe.loc[group.index, "MACD"] = macd["MACD"].values
            dataframe.loc[group.index, "MACD_signal"] = macd["MACD_signal"].values
            dataframe.loc[group.index, "MACD_hist"] = macd["MACD_hist"].values
            dataframe.loc[group.index, "BB_upper"] = bbands["BB_upper"].values
            dataframe.loc[group.index, "BB_middle"] = bbands["BB_middle"].values
            dataframe.loc[group.index, "BB_lower"] = bbands["BB_lower"].values
        dataframe.fillna(method="bfill", inplace=True)
        dataframe.fillna(method="ffill", inplace=True)
        dataframe.to_csv(self.output_path, index=False)
        return dataframe


def build_features() -> pd.DataFrame:
    """特徴量生成を実行しデータフレームを返す。"""
    engineer = FeatureEngineer()
    return engineer.engineer()


__all__ = ["FeatureEngineer", "build_features"]
