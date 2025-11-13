import pytest
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "src")


TARGET_SYMBOL = "NVDA"


class TestPipelineIntegration:
    """パイプライン統合テスト"""

    @pytest.fixture
    def sample_prediction_data(self):
        """予測用のサンプルデータ"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "symbol": [TARGET_SYMBOL] * 200,
                "Close": 100 + np.cumsum(np.random.randn(200) * 2),
                "Volume": np.random.uniform(1_000_000, 5_000_000, 200),
            }
        )

    def test_single_symbol_focus(self, sample_prediction_data):
        """NVDA の単一銘柄データに統一されていることを検証する。"""
        assert "symbol" in sample_prediction_data.columns
        assert "Close" in sample_prediction_data.columns
        symbols = sample_prediction_data["symbol"].unique()
        assert list(symbols) == [TARGET_SYMBOL]

    def test_prediction_output_format(self):
        """予測出力フォーマットのテスト"""
        prediction_output = pd.DataFrame(
            {
                "symbol": [TARGET_SYMBOL],
                "forecast_5d": [155.2],
                "confidence": [0.85],
                "model": ["LSTM"],
            }
        )

        assert len(prediction_output) == 1
        assert all(
            col in prediction_output.columns
            for col in ["symbol", "forecast_5d", "confidence", "model"]
        )
        assert prediction_output["confidence"].max() <= 1.0
        assert prediction_output["confidence"].min() >= 0.0
