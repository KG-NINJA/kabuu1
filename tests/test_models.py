# tests/test_models.py
import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

class TestModelFrameworks:
    """モデルフレームワークのテスト"""
    
    def test_tensorflow_available(self):
        """TensorFlow が利用可能か確認"""
        try:
            import tensorflow as tf
            assert tf.__version__ is not None
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_xgboost_available(self):
        """XGBoost が利用可能か確認"""
        try:
            import xgboost as xgb
            assert xgb.__version__ is not None
        except ImportError:
            pytest.skip("XGBoost not installed")

    def test_scikit_learn_available(self):
        """scikit-learn が利用可能か確認"""
        try:
            from sklearn import __version__
            assert __version__ is not None
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_lstm_model_creation(self):
        """LSTM モデルの作成テスト"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            
            model = Sequential([
                LSTM(32, activation='relu', input_shape=(10, 5)),
                LSTM(16, activation='relu'),
                Dense(1)
            ])
            
            assert model is not None
            assert len(model.layers) == 3
            assert model.layers[0].units == 32
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_xgboost_regressor_creation(self):
        """XGBoost リグレッサーの作成テスト"""
        try:
            import xgboost as xgb
            
            model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            assert model is not None
            assert model.n_estimators == 50
            assert model.max_depth == 5
        except ImportError:
            pytest.skip("XGBoost not installed")

    def test_dummy_prediction(self):
        """ダミー予測テスト"""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X[:10])
        
        assert len(predictions) == 10
        assert predictions.dtype in [np.float64, np.float32]
