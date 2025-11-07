"""
実際の株価データから予測CSVを生成
GitHub Actionsで実行される
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import sys
import os

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 対象銘柄リスト
TICKERS = {
    'US': {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.',
        'MSFT': 'Microsoft Corporation',
        'TSLA': 'Tesla Inc.'
    },
    'JP': {
        # 自動車・製造業
        '7203.T': 'Toyota Motor Corporation (トヨタ自動車)',
        '7267.T': 'Honda Motor Co., Ltd. (本田技研工業)',
        '6501.T': 'Hitachi, Ltd. (日立製作所)',
        '6752.T': 'Panasonic Corporation (パナソニック)',
        
        # エレクトロニクス・半導体
        '6758.T': 'Sony Group Corporation (ソニーグループ)',
        '8035.T': 'Tokyo Electron Limited (東京エレクトロン)',
        '6861.T': 'Keyence Corporation (キーエンス)',
        '7974.T': 'Nintendo Co., Ltd. (任天堂)',
        
        # 金融
        '8306.T': 'Mitsubishi UFJ Financial Group (三菱UFJフィナンシャルグループ)',
        '9984.T': 'SoftBank Group Corp. (ソフトバンクグループ)',
        
        # 小売・サービス
        '9983.T': 'Fast Retailing Co., Ltd. (ファーストリテイリング)',
        
        # 商社
        '8058.T': 'Mitsubishi Corporation (三菱商事)',
        
        # 通信
        '9432.T': 'Nippon Telegraph and Telephone Corporation (日本電信電話)',
        
        # 製薬
        '4502.T': 'Takeda Pharmaceutical Company Limited (武田薬品工業)',
        
        # その他
        '6702.T': 'Sumitomo Electric Industries, Ltd. (住友電気工業)'
    }
}


def get_current_price(ticker: str) -> float:
    """
    現在の株価を取得
    
    Args:
        ticker: 銘柄コード
    
    Returns:
        float: 現在の株価
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='5d')
        
        if len(data) == 0:
            return None
        
        return float(data['Close'].iloc[-1])
    except Exception as e:
        print(f"警告: {ticker} の価格取得に失敗: {e}")
        return None


def simple_forecast(ticker: str, current_price: float) -> tuple:
    """
    シンプルな予測を生成（移動平均ベース）
    
    Args:
        ticker: 銘柄コード
        current_price: 現在の株価
    
    Returns:
        tuple: (予測価格, 信頼度)
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='30d')
        
        if len(data) < 10:
            # データが少ない場合は現在価格をそのまま使用
            return current_price, 0.5
        
        # 移動平均を計算
        sma_5 = data['Close'].tail(5).mean()
        sma_10 = data['Close'].tail(10).mean()
        sma_20 = data['Close'].tail(20).mean() if len(data) >= 20 else sma_10
        
        # トレンドを計算
        recent_trend = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
        
        # ボラティリティを計算
        volatility = data['Close'].pct_change().tail(10).std()
        
        # 予測価格（移動平均とトレンドを考慮）
        forecast_price = current_price * (1 + recent_trend * 0.5)
        
        # 信頼度計算（ボラティリティが低いほど信頼度が高い）
        confidence = max(0.5, min(0.9, 1.0 - volatility * 10))
        
        return float(forecast_price), float(confidence)
        
    except Exception as e:
        print(f"警告: {ticker} の予測生成に失敗: {e}")
        # フォールバック: 現在価格をそのまま使用
        return current_price, 0.5


def generate_forecast_csv(output_path: str = 'forecast_data.csv') -> str:
    """
    実際の株価データから予測CSVを生成
    
    Args:
        output_path: 出力CSVファイルパス
    
    Returns:
        str: 生成されたCSVファイルパス
    """
    forecasts = []
    today = date.today()
    
    print("📊 実際の株価データから予測CSVを生成中...")
    print(f"実行日時: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # 全銘柄を処理
    for market, stocks in TICKERS.items():
        print(f"\n{market}市場:")
        for ticker, name in stocks.items():
            print(f"  処理中: {name} ({ticker})...", end=" ")
            
            # 現在価格を取得
            current_price = get_current_price(ticker)
            
            if current_price is None:
                print("❌ 価格取得失敗")
                continue
            
            # 予測を生成
            forecast_price, confidence = simple_forecast(ticker, current_price)
            
            # シンボルを正規化（.Tを削除）
            symbol = ticker.replace('.T', '')
            
            forecasts.append({
                'symbol': symbol,
                'forecast': round(forecast_price, 2),
                'current_price': round(current_price, 2),
                'confidence': round(confidence, 3),
                'date': today.strftime('%Y-%m-%d')
            })
            
            change_pct = ((forecast_price - current_price) / current_price) * 100
            print(f"✅ 現在: ${current_price:.2f}, 予測: ${forecast_price:.2f} ({change_pct:+.2f}%)")
    
    # DataFrameに変換
    if len(forecasts) == 0:
        print("\n❌ エラー: 予測データが生成されませんでした")
        return None
    
    df = pd.DataFrame(forecasts)
    
    # CSVファイルを保存
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✅ CSV生成完了: {output_file}")
    print(f"   銘柄数: {len(forecasts)}")
    print(f"   平均信頼度: {df['confidence'].mean():.3f}")
    
    return str(output_file)


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='実際の株価データから予測CSVを生成')
    parser.add_argument('--output', type=str, default='forecast_data.csv',
                       help='出力CSVファイルパス')
    args = parser.parse_args()
    
    output_path = generate_forecast_csv(args.output)
    
    if output_path:
        print(f"\n📄 生成されたCSV: {output_path}")
        return 0
    else:
        print("\n❌ CSV生成に失敗しました")
        return 1


if __name__ == "__main__":
    exit(main())

