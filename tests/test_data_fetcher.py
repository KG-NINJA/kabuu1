import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, 'src')

from data_fetcher import StockDataFetcher

@pytest.fixture
def fetcher():
    return StockDataFetcher()

def test_fetcher_initialization(fetcher):
    assert fetcher is not None

@patch('yfinance.download')
def test_fetch_us_stocks(mock_download, fetcher):
    mock_data = pd.DataFrame({
        'Close': [150.0, 151.0, 152.0],
        'Volume': [1000000, 1100000, 1200000]
    })
    mock_download.return_value = mock_data
    
    result = fetcher.fetch_us_stocks(['AAPL'], '2024-01-01', '2024-12-31')
    assert result is not None

@patch('yfinance.download')
def test_fetch_jp_stocks(mock_download, fetcher):
    mock_data = pd.DataFrame({
        'Close': [3000.0, 3100.0, 3200.0],
        'Volume': [500000, 600000, 700000]
    })
    mock_download.return_value = mock_data
    
    result = fetcher.fetch_jp_stocks(['9984.T'], '2024-01-01', '2024-12-31')
    assert result is not None
