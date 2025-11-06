import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')

from feature_engineering import FeatureEngineer

@pytest.fixture
def engineer():
    return FeatureEngineer()

@pytest.fixture
def sample_data():
    dates = pd.date_range('2024-01-01', periods=100)
    return pd.DataFrame({
        'Close': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Volume': np.random.uniform(1000000, 5000000, 100)
    }, index=dates)

def test_moving_average(engineer, sample_data):
    result = engineer.add_moving_average(sample_data, windows=[5, 20])
    assert 'MA_5' in result.columns
    assert 'MA_20' in result.columns

def test_rsi(engineer, sample_data):
    result = engineer.add_rsi(sample_data, period=14)
    assert 'RSI' in result.columns
    assert result['RSI'].notna().sum() > 0

def test_macd(engineer, sample_data):
    result = engineer.add_macd(sample_data)
    assert 'MACD' in result.columns
    assert 'Signal' in result.columns
