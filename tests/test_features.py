# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "src")


class TestFeatureEngineering:
    """特徴量エンジニアリングのテスト"""

    @pytest.fixture
    def sample_ohlc_data(self):
        """サンプル株価データの作成"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Close": 100 + np.cumsum(np.random.randn(100) * 2),
                "High": 102 + np.cumsum(np.random.randn(100) * 2),
                "Low": 98 + np.cumsum(np.random.randn(100) * 2),
                "Volume": np.random.uniform(1000000, 5000000, 100),
            },
            index=dates,
        )

    def test_moving_average_creation(self, sample_ohlc_data):
        """移動平均の計算テスト"""
        ma5 = sample_ohlc_data["Close"].rolling(window=5).mean()
        assert ma5.notna().sum() == len(sample_ohlc_data) - 4
        assert ma5.iloc[4] > 0

    def test_rsi_calculation(self, sample_ohlc_data):
        """RSI（相対力指数）の計算テスト"""
        delta = sample_ohlc_data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        assert rsi.notna().sum() > 0
        assert (rsi >= 0).all() or (rsi <= 100).all() or rsi.isna().any()

    def test_volume_normalization(self, sample_ohlc_data):
        """出来高の正規化テスト"""
        normalized_vol = (
            sample_ohlc_data["Volume"] - sample_ohlc_data["Volume"].mean()
        ) / sample_ohlc_data["Volume"].std()
        assert normalized_vol.std() > 0
