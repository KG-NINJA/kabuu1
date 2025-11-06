import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')

from preprocessor import DataPreprocessor

@pytest.fixture
def preprocessor():
    return DataPreprocessor()

@pytest.fixture
def sample_data():
    dates = pd.date_range('2024-01-01', periods=100)
    return pd.DataFrame({
        'Close': np.random.uniform(100, 200, 100),
        'MA_5': np.random.uniform(100, 200, 100),
        'RSI': np.random.uniform(30, 70, 100),
        'Volume': np.random.uniform(1000000, 5000000, 100)
    }, index=dates)

def test_normalize_data(preprocessor, sample_data):
    normalized = preprocessor.normalize(sample_data)
    assert normalized.shape == sample_data.shape
    assert normalized.min().min() >= -2
    assert normalized.max().max() <= 2

def test_train_test_split(preprocessor, sample_data):
    train, test = preprocessor.split_data(sample_data, test_size=0.2)
    assert len(train) + len(test) == len(sample_data)
    assert len(train) > len(test)
