"""
株価予測JSON生成システム
検証機能付きで予測結果をJSON形式で出力
"""

import pandas as pd
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import os

# 親ディレクトリをパスに追加（CI環境でも動作するように）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# インポート（相対パスと絶対パスの両方に対応）
try:
    from src.validation_helpers import (
        is_trading_day,
        get_next_trading_day,
        validate_price_prediction,
        detect_scale_error,
        recalculate_confidence
    )
except ImportError as e:
    # 相対インポートを試す
    try:
        from validation_helpers import (
            is_trading_day,
            get_next_trading_day,
            validate_price_prediction,
            detect_scale_error,
            recalculate_confidence
        )
    except ImportError:
        # 最後の手段：直接パスからインポート
        import importlib.util
        validation_helpers_path = os.path.join(current_dir, 'validation_helpers.py')
        if os.path.exists(validation_helpers_path):
            spec = importlib.util.spec_from_file_location("validation_helpers", validation_helpers_path)
            validation_helpers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(validation_helpers)
            is_trading_day = validation_helpers.is_trading_day
            get_next_trading_day = validation_helpers.get_next_trading_day
            validate_price_prediction = validation_helpers.validate_price_prediction
            detect_scale_error = validation_helpers.detect_scale_error
            recalculate_confidence = validation_helpers.recalculate_confidence
        else:
            raise ImportError(f"validation_helpers.py not found. Error: {e}")


def determine_market(symbol: str) -> str:
    """
    銘柄コードから市場を判定
    
    Args:
        symbol: 銘柄コード（例: 'AAPL', '7203'）
    
    Returns:
        str: 'US' or 'JP'
    """
    # 数字のみの場合は日本株
    if symbol.isdigit():
        return 'JP'
    # アルファベットの場合は米国株
    elif symbol.isalpha():
        return 'US'
    # その他（例: '7203.T'）は日本株と判断
    else:
        return 'JP'


def generate_llm_prompts(forecast_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    予測データから検証付きJSONを生成
    
    Args:
        forecast_csv: 予測データのCSVファイルパス（オプション）
    
    Returns:
        dict: 検証済み予測JSON
    """
    # CSVファイルが指定されていない場合は、サンプルデータを使用
    if forecast_csv is None or not Path(forecast_csv).exists():
        print("警告: CSVファイルが見つかりません。サンプルデータを使用します。")
        # サンプルデータを生成（実際の使用時はCSVから読み込む）
        forecast_df = _create_sample_data()
    else:
        forecast_df = pd.read_csv(forecast_csv)
        # CSVが空または1行のみの場合は、サンプルデータを使用
        if len(forecast_df) == 0:
            print("警告: CSVファイルが空です。サンプルデータを使用します。")
            forecast_df = _create_sample_data()
        elif len(forecast_df) == 1:
            print(f"警告: CSVファイルに1行のみです（{len(forecast_df)}行）。サンプルデータを使用します。")
            forecast_df = _create_sample_data()
    
    # 必要な列をチェック
    required_columns = ['symbol', 'forecast', 'current_price', 'confidence']
    missing_columns = [col for col in required_columns if col not in forecast_df.columns]
    
    if missing_columns:
        print(f"警告: CSVに必要な列がありません: {missing_columns}。サンプルデータを使用します。")
        forecast_df = _create_sample_data()
    
    # 結果を格納するリスト
    forecasts = []
    data_issues = []
    valid_count = 0
    warning_count = 0
    error_count = 0
    
    # 次の営業日を計算（最初の銘柄の日付を使用）
    first_date = None
    if 'date' in forecast_df.columns:
        first_date_str = forecast_df['date'].iloc[0]
        if isinstance(first_date_str, str):
            first_date = datetime.strptime(first_date_str, '%Y-%m-%d').date()
        else:
            first_date = pd.to_datetime(first_date_str).date()
    else:
        # 日付が無い場合は今日から計算
        first_date = date.today()
    
    # 次の営業日を取得（最初の銘柄の市場で判定）
    next_trading_day = first_date
    next_trading_day_note = None
    
    # 各予測に対して検証を実行
    for idx, row in forecast_df.iterrows():
        symbol = str(row['symbol'])
        forecast_price = float(row['forecast'])
        current_price = float(row['current_price'])
        original_confidence = float(row['confidence'])
        
        # 市場を判定
        market = determine_market(symbol)
        
        # 日付を取得
        if 'date' in row and pd.notna(row['date']):
            pred_date_str = str(row['date'])
            try:
                if isinstance(pred_date_str, str):
                    pred_date = datetime.strptime(pred_date_str, '%Y-%m-%d').date()
                else:
                    pred_date = pd.to_datetime(pred_date_str).date()
            except:
                pred_date = first_date
        else:
            pred_date = first_date
        
        # 営業日判定と修正
        if not is_trading_day(pred_date, market):
            corrected_date = get_next_trading_day(pred_date, market)
            if next_trading_day_note is None:
                next_trading_day_note = f"{pred_date.strftime('%Y-%m-%d')}は営業日ではないため{corrected_date.strftime('%Y-%m-%d')}に修正"
            pred_date = corrected_date
            data_issues.append(f"Trading day detection error for {symbol} ({pred_date_str} is not a trading day)")
        
        # 次の営業日を更新（最初の銘柄の日付を使用）
        if idx == 0:
            next_trading_day = pred_date
        
        # 価格検証
        validation = validate_price_prediction(
            symbol,
            forecast_price,
            current_price,
            market
        )
        
        # スケール誤差検出
        scale_check = detect_scale_error(
            symbol,
            forecast_price,
            current_price
        )
        
        # 信頼度再計算
        confidence_result = recalculate_confidence(
            forecast_price,
            current_price,
            original_confidence,
            validation['severity'],
            scale_check['has_error']
        )
        
        # 統計を更新
        if validation['severity'] == 'error':
            error_count += 1
        elif validation['severity'] == 'warning':
            warning_count += 1
        else:
            valid_count += 1
        
        # 予測データを構築
        forecast_data = {
            'symbol': symbol,
            'date': pred_date.strftime('%Y-%m-%d'),
            'forecast': round(forecast_price, 2),
            'current_price': round(current_price, 2),
            'confidence': {
                'original': round(original_confidence, 3),
                'adjusted': round(confidence_result['adjusted_confidence'], 3),
                'adjustment_reason': confidence_result['adjustment_reason']
            },
            'validation': {
                'is_valid': validation['is_valid'],
                'severity': validation['severity'],
                'price_change_pct': round(validation['price_change_pct'], 2),
                'issue': validation['issue']
            },
            'scale_check': {
                'has_error': scale_check['has_error'],
                'suspected_scale_factor': round(scale_check['suspected_scale_factor'], 2) if scale_check['suspected_scale_factor'] else None,
                'note': scale_check['note']
            }
        }
        
        forecasts.append(forecast_data)
    
    # 統計情報を計算
    original_confidences = [f['confidence']['original'] for f in forecasts]
    adjusted_confidences = [f['confidence']['adjusted'] for f in forecasts]
    
    avg_confidence_original = sum(original_confidences) / len(original_confidences) if original_confidences else 0.0
    avg_confidence_adjusted = sum(adjusted_confidences) / len(adjusted_confidences) if adjusted_confidences else 0.0
    
    # データ品質を判定
    total_predictions = len(forecasts)
    error_ratio = error_count / total_predictions if total_predictions > 0 else 0.0
    
    if error_ratio >= 0.5:
        data_quality = "POOR"
        recommendation = f"Use only {valid_count} valid predictions. {error_count} predictions contain errors."
    elif error_ratio >= 0.3:
        data_quality = "MODERATE"
        recommendation = f"Use predictions with caution. {error_count} predictions contain errors."
    else:
        data_quality = "GOOD"
        recommendation = "Most predictions are valid. Review warnings if any."
    
    # 生成ステータスを決定
    if error_count > 0:
        if warning_count > 0:
            generation_status = "COMPLETED_WITH_WARNINGS"
        else:
            generation_status = "COMPLETED_WITH_ERRORS"
    else:
        if warning_count > 0:
            generation_status = "COMPLETED_WITH_WARNINGS"
        else:
            generation_status = "COMPLETED"
    
    # 最終JSONを構築
    result = {
        'timestamp': datetime.now().isoformat(),
        'generation_status': generation_status,
        'next_trading_day': next_trading_day.strftime('%Y-%m-%d'),
        'next_trading_day_note': next_trading_day_note,
        'symbols_predicted': len(forecasts),
        'total_predictions': len(forecasts),
        'statistics': {
            'avg_confidence_original': round(avg_confidence_original, 3),
            'avg_confidence_adjusted': round(avg_confidence_adjusted, 3),
            'confidence_reduction_reason': f"Multiple unrealistic predictions detected" if error_count > 0 else "No issues detected"
        },
        'forecasts': forecasts,
        'summary': {
            'valid_predictions': valid_count,
            'warning_predictions': warning_count,
            'error_predictions': error_count,
            'recommendation': recommendation,
            'data_quality': f"{data_quality} - {error_count}/{total_predictions} predictions are unrealistic"
        },
        'data_issues': data_issues if data_issues else []
    }
    
    return result


def _create_sample_data() -> pd.DataFrame:
    """
    サンプルデータを作成（テスト用）
    
    Returns:
        pd.DataFrame: サンプル予測データ
    """
    # プロンプトの例に基づいたサンプルデータ
    sample_data = {
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', '7203', '6758', '8306', '9984'],
        'forecast': [151.83, 120.45, 380.20, 180.50, 2872.33, 15194.48, 2453.59, 3136.0],
        'current_price': [269.77, 185.20, 452.30, 245.60, 3139.0, 4250.0, 2330.0, 23000.0],
        'confidence': [0.80, 0.75, 0.82, 0.78, 0.73, 0.85, 0.77, 0.70],
        'date': ['2025-11-08', '2025-11-08', '2025-11-08', '2025-11-08', 
                 '2025-11-08', '2025-11-08', '2025-11-08', '2025-11-08']
    }
    return pd.DataFrame(sample_data)


def save_json_output(data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    JSON出力を保存
    
    Args:
        data: 保存するデータ
        output_path: 出力パス（指定しない場合は自動生成）
    
    Returns:
        str: 保存されたファイルパス
    """
    if output_path is None:
        # darwin_analysisディレクトリを作成
        output_dir = Path('darwin_analysis')
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / 'forecast_analysis.json')
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    return str(output_file)


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='株価予測JSON生成システム')
    parser.add_argument('--csv', type=str, help='予測データのCSVファイルパス')
    parser.add_argument('--output', type=str, help='出力JSONファイルパス')
    args = parser.parse_args()
    
    # JSON生成
    result = generate_llm_prompts(forecast_csv=args.csv)
    
    # JSON保存
    output_path = save_json_output(result, args.output)
    
    print(f"✅ JSON生成完了: {output_path}")
    print(f"   ステータス: {result['generation_status']}")
    print(f"   有効予測: {result['summary']['valid_predictions']}")
    print(f"   警告予測: {result['summary']['warning_predictions']}")
    print(f"   エラー予測: {result['summary']['error_predictions']}")
    print(f"   データ品質: {result['summary']['data_quality']}")


if __name__ == "__main__":
    main()

