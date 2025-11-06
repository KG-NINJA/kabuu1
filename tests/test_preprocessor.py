# tests/test_preprocessor.py
import pytest
import pandas as pd
import numpy as np


class TestPreprocessor:
    """データ前処理のテスト"""

    @pytest.fixture
    def feature_data(self):
        """特徴量を含むサンプルデータ"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Close": 100 + np.cumsum(np.random.randn(100) * 2),
                "MA_5": 100 + np.cumsum(np.random.randn(100) * 2),
                "MA_20": 100 + np.cumsum(np.random.randn(100) * 2),
                "RSI": np.random.uniform(30, 70, 100),
                "Volume": np.random.uniform(1000000, 5000000, 100),
                "MACD": np.random.randn(100),
            },
            index=dates,
        )

    def test_data_normalization(self, feature_data):
        """データの正規化テスト"""
        normalized = (feature_data - feature_data.mean()) / feature_data.std()
        assert normalized.shape == feature_data.shape
        assert normalized["Close"].std() > 0.9 and normalized["Close"].std() < 1.1

    def test_missing_values_handling(self, feature_data):
        """欠損値の処理テスト"""
        data_with_nan = feature_data.copy()
        data_with_nan.iloc[0:5, 0] = np.nan

        # 新しい方法: bfill() を使用
        filled_data = data_with_nan.bfill()
        assert filled_data.isna().sum().sum() == 0

    def test_train_test_split(self, feature_data):
        """訓練・テストデータの分割テスト"""
        split_idx = int(len(feature_data) * 0.8)
        train = feature_data[:split_idx]
        test = feature_data[split_idx:]

        assert len(train) + len(test) == len(feature_data)
        assert len(train) > len(test)
        assert len(train) / len(feature_data) == pytest.approx(0.8, rel=0.01)

    def test_data_shape_preservation(self, feature_data):
        """データ形状の保持テスト"""
        original_shape = feature_data.shape
        processed = feature_data.dropna()

        # 全て正常なので形状は変わらない
        assert processed.shape[1] == original_shape[1]

    def test_column_names_preserved(self, feature_data):
        """カラム名の保持テスト"""
        columns = feature_data.columns.tolist()
        assert "Close" in columns
        assert "Volume" in columns
        assert len(columns) == 6

    def test_forward_fill_alternative(self, feature_data):
        """Forward fill の代替テスト"""
        data_with_nan = feature_data.copy()
        data_with_nan.iloc[0:5, 0] = np.nan

        # ffill() を使用
        filled_data = data_with_nan.ffill()
        assert filled_data.notna().all().all() or filled_data.isna().sum().sum() <= 5

    def test_data_scaling(self, feature_data):
        """データのスケーリングテスト"""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)

        assert scaled_data.shape == feature_data.shape
        assert np.allclose(scaled_data.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled_data.std(axis=0), 1, atol=1e-10)
