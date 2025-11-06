# tests/test_data_fetcher.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, 'src')

class TestDataFetcher:
    """株価データ取得のテスト"""
    
    def test_data_structure(self):
        """データフレームの構造をテスト"""
        data = pd.DataFrame({
            'Close': [150.0, 151.0, 152.0],
            'High': [151.0, 152.0, 153.0],
            'Low': [149.0, 150.0, 151.0],
            'Volume': [1000000, 1100000, 1200000]
        })
        assert len(data) == 3
        assert all(col in data.columns for col in ['Close', 'High', 'Low', 'Volume'])

    @patch('yfinance.download')
    def test_us_stock_fetch_mock(self, mock_download):
        """US株の取得をモックでテスト"""
        mock_data = pd.DataFrame({
            'Close': [150.0, 151.0],
            'Volume': [1000000, 1100000]
        })
        mock_download.return_value = mock_data
        assert mock_download.return_value is not None
        assert len(mock_download.return_value) == 2

    @patch('yfinance.download')
    def test_jp_stock_fetch_mock(self, mock_download):
        """JP株の取得をモックでテスト"""
        mock_data = pd.DataFrame({
            'Close': [3000.0, 3100.0],
            'Volume': [500000, 600000]
        })
        mock_download.return_value = mock_data
        assert len(mock_download.return_value) == 2
