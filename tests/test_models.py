import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

def test_lstm_model_structure():
    """LSTM モデルの構造をテスト"""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(10, 5)),
            Dense(1)
        ])
        assert model is not None
        assert len(model.layers) == 2
    except ImportError:
        pytest.skip("TensorFlow not installed")

def test_xgboost_initialization():
    """XGBoost モデルの初期化をテスト"""
    try:
        import xgboost as xgb
        
        model = xgb.XGBRegressor(n_estimators=10, max_depth=5)
        assert model is not None
        assert model.n_estimators == 10
    except ImportError:
        pytest.skip("XGBoost not installed")
