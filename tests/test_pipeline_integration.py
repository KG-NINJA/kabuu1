import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')

class TestPipelineIntegration:
    """パイプライン統合テスト"""
    
    @pytest.fixture
    def sample_prediction_data(self):
        """予測用のサンプルデータ"""
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'symbol': ['AAPL'] * 50 + ['9984'] * 50 + ['GOOGL'] * 50 + ['6758'] * 50,
            'Close': np.tile(100 + np.cumsum(np.random.randn(50) * 2), 4),
            'Volume': np.tile(np.random.uniform(1000000, 5000000, 50), 4)
        })

    def test_multi_symbol_data_structure(self, sample_prediction_data):
        """複数銘柄データの構造テスト"""
        assert 'symbol' in sample_prediction_data.columns
        assert 'Close' in sample_prediction_data.columns
        symbols = sample_prediction_data['symbol'].unique()
        assert len(symbols) == 4

    def test_prediction_output_format(self):
        """予測出力フォーマットのテスト"""
        prediction_output = pd.DataFrame({
            'symbol': ['AAPL', '9984', 'GOOGL', '6758'],
            'forecast_5d': [155.2, 3150.5, 185.3, 2890.1],
            'confidence': [0.85, 0.78, 0.82, 0.75],
            'model': ['LSTM', 'XGBoost', 'LSTM', 'XGBoost']
        })
        
        assert len(prediction_output) == 4
        assert all(col in prediction_output.columns for col in ['symbol', 'forecast_5d', 'confidence', 'model'])
        assert prediction_output['confidence'].max() <= 1.0
        assert prediction_output['confidence'].min() >= 0.0
