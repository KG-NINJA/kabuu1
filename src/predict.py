"""
株価予測JSON生成システム
検証機能付きで予測結果をJSON形式で出力
"""

import json
import logging
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# 親ディレクトリをパスに追加（CI環境でも動作するように）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# インポート（相対パスと絶対パスの両方に対応）
try:
    from src.data_fetcher import fetch_stock_data
except ImportError:
    try:
        from data_fetcher import fetch_stock_data  # type: ignore
    except ImportError:
        fetch_stock_data = None  # type: ignore

try:
        from src.validation_helpers import (
            is_trading_day,
            get_next_trading_day,
            validate_price_prediction,
            detect_scale_error,
            enforce_price_ratio_limits,
            validate_history_bounds,
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
            enforce_price_ratio_limits,
            validate_history_bounds,
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
            enforce_price_ratio_limits = validation_helpers.enforce_price_ratio_limits
            validate_history_bounds = validation_helpers.validate_history_bounds
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


def _parse_optional_date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    return datetime.strptime(value, '%Y-%m-%d').date()


def _setup_logging(log_path: Optional[str]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def _calculate_forecast_date(last_trading_day: date, market: str, days_ahead: int) -> date:
    """次の営業日を days_ahead 分進めた日付を求める。"""

    target_date = last_trading_day
    for _ in range(max(days_ahead, 1)):
        target_date = get_next_trading_day(target_date, market)
    return target_date


def _summarize_history(rows: pd.DataFrame, limit: int = 60) -> List[Dict[str, Any]]:
    """LLM に渡しやすいように直近データを整形する。"""

    history: List[Dict[str, Any]] = []
    for entry in rows.tail(limit).itertuples():
        volume_value = getattr(entry, 'volume', None)
        if volume_value is None:
            volume = None
        else:
            try:
                volume = int(float(volume_value))
            except (TypeError, ValueError):
                volume = None

        history.append(
            {
                'date': pd.to_datetime(entry.date).strftime('%Y-%m-%d'),
                'open': round(float(getattr(entry, 'open')), 4) if hasattr(entry, 'open') else None,
                'high': round(float(getattr(entry, 'high')), 4) if hasattr(entry, 'high') else None,
                'low': round(float(getattr(entry, 'low')), 4) if hasattr(entry, 'low') else None,
                'close': round(float(getattr(entry, 'close')), 4) if hasattr(entry, 'close') else None,
                'volume': volume,
            }
        )
    return history


def _compute_confidence(volatility: float) -> float:
    """ボラティリティから信頼度を算出。"""

    if math.isnan(volatility) or volatility <= 0:
        return 0.65
    confidence = 0.9 - min(volatility * 12, 0.6)
    return max(0.3, min(0.95, confidence))


def prepare_forecast_dataframe(
    history_df: pd.DataFrame,
    days_ahead: int = 1,
) -> pd.DataFrame:
    """履歴データから予測対象の DataFrame を構築する。"""

    if history_df is None or history_df.empty:
        return _create_sample_data()

    records: List[Dict[str, Any]] = []
    grouped = history_df.copy()
    grouped['date'] = pd.to_datetime(grouped['date'])

    for symbol, group in grouped.groupby('symbol'):
        group = group.sort_values('date')
        market = group['market'].iloc[-1] if 'market' in group.columns else determine_market(str(symbol))
        last_close = float(group['close'].iloc[-1]) if 'close' in group.columns else float(group.iloc[-1].get('close', 0))

        if last_close <= 0:
            continue

        returns = group['close'].pct_change().dropna()
        recent_return = returns.tail(max(5, days_ahead * 3)).mean() if not returns.empty else 0.0
        forecast_price = last_close * (1 + recent_return * max(days_ahead, 1))
        if not math.isfinite(forecast_price) or forecast_price <= 0:
            forecast_price = last_close

        volatility = returns.tail(20).std() if not returns.empty else float('nan')
        confidence = _compute_confidence(volatility)

        last_trading_day = group['date'].iloc[-1].date()
        forecast_date = _calculate_forecast_date(last_trading_day, market, days_ahead)

        record = {
            'symbol': str(symbol),
            'market': market,
            'date': forecast_date.strftime('%Y-%m-%d'),
            'forecast': round(float(forecast_price), 4),
            'current_price': round(last_close, 4),
            'confidence': round(confidence, 4),
            'history': _summarize_history(group),
        }

        records.append(record)

    if not records:
        return _create_sample_data()

    return pd.DataFrame(records)


def save_forecast_csv(forecast_df: pd.DataFrame, output_path: str) -> Path:
    """予測結果をCSVに保存する。"""

    csv_df = forecast_df.copy()
    if 'history' in csv_df.columns:
        csv_df = csv_df.drop(columns=['history'])

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(path, index=False)
    logging.info("Saved forecast CSV: %s", path)
    return path


def generate_llm_prompts(
    forecast_csv: Optional[str] = None,
    forecast_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    予測データから検証付きJSONを生成
    
    Args:
        forecast_csv: 予測データのCSVファイルパス（オプション）
    
    Returns:
        dict: 検証済み予測JSON
    """
    # CSVファイルが指定されていない場合は、サンプルデータを使用
    if forecast_df is None:
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
        market = row['market'] if 'market' in row and pd.notna(row['market']) else determine_market(symbol)

        # 日付を取得
        pred_date_str = None
        if 'date' in row and pd.notna(row['date']):
            pred_date_str = str(row['date'])
            try:
                if isinstance(pred_date_str, str):
                    pred_date = datetime.strptime(pred_date_str, '%Y-%m-%d').date()
                else:
                    pred_date = pd.to_datetime(pred_date_str).date()
            except Exception:
                pred_date = first_date
        else:
            pred_date = first_date

        symbol_issues: List[Dict[str, Any]] = []
        zero_reasons: List[str] = []

        # 営業日判定と修正
        if not is_trading_day(pred_date, market):
            corrected_date = get_next_trading_day(pred_date, market)
            if next_trading_day_note is None:
                next_trading_day_note = (
                    f"{pred_date.strftime('%Y-%m-%d')}は営業日ではないため"
                    f"{corrected_date.strftime('%Y-%m-%d')}に修正"
                )
            data_issue = {
                'symbol': symbol,
                'issue_type': 'trading_day_adjustment',
                'details': {
                    'market': market,
                    'original_date': pred_date.strftime('%Y-%m-%d'),
                    'corrected_date': corrected_date.strftime('%Y-%m-%d'),
                },
            }
            data_issues.append(data_issue)
            symbol_issues.append(data_issue)
            pred_date = corrected_date

        # 次の営業日を更新（最初の銘柄の日付を使用）
        if idx == 0:
            next_trading_day = pred_date

        # 履歴データを正規化
        history_records: List[Dict[str, Any]] = []
        history_value: Optional[Any] = None
        if 'history' in row:
            raw_history = row['history']
            if isinstance(raw_history, list):
                history_records = [entry for entry in raw_history if isinstance(entry, dict)]
            elif isinstance(raw_history, dict):
                history_records = [raw_history]
            elif isinstance(raw_history, str):
                try:
                    parsed_history = json.loads(raw_history)
                except json.JSONDecodeError:
                    parsed_history = None
                if isinstance(parsed_history, dict):
                    history_records = [parsed_history]
                elif isinstance(parsed_history, list):
                    history_records = [entry for entry in parsed_history if isinstance(entry, dict)]
            elif raw_history is None or (isinstance(raw_history, float) and math.isnan(raw_history)):
                history_records = []
            history_value = history_records if history_records else None

        # 価格検証
        validation = validate_price_prediction(
            symbol,
            forecast_price,
            current_price,
            market
        )

        # 倍率チェック
        ratio_check = enforce_price_ratio_limits(
            symbol,
            forecast_price,
            current_price,
            market,
        )

        # 履歴範囲チェック
        history_check = validate_history_bounds(
            symbol,
            market,
            history_records,
            forecast_price,
            current_price,
        )

        # スケール誤差検出
        scale_check = detect_scale_error(
            symbol,
            forecast_price,
            current_price
        )

        # 重大な問題を収集
        if validation.get('hard_limit_triggered'):
            zero_reasons.append(validation['issue'] or 'Daily move exceeded hard limit')
            data_issue = {
                'symbol': symbol,
                'issue_type': 'daily_change_limit',
                'details': {
                    'market': market,
                    'forecast_price': forecast_price,
                    'current_price': current_price,
                    'price_change_pct': validation['price_change_pct'],
                    'limit_threshold_pct': validation['limit_threshold_pct'],
                },
            }
            data_issues.append(data_issue)
            symbol_issues.append(data_issue)

        if not ratio_check['is_valid']:
            zero_reasons.append(ratio_check['note'])
            data_issue = {
                'symbol': symbol,
                'issue_type': 'price_ratio_violation',
                'details': {
                    'market': market,
                    'forecast_price': forecast_price,
                    'current_price': current_price,
                    'ratio': ratio_check['ratio'],
                    'bounds': ratio_check['bounds'],
                },
            }
            data_issues.append(data_issue)
            symbol_issues.append(data_issue)
        elif ratio_check['severity'] == 'warning':
            symbol_issues.append(
                {
                    'symbol': symbol,
                    'issue_type': 'price_ratio_near_limit',
                    'details': {
                        'market': market,
                        'ratio': ratio_check['ratio'],
                        'bounds': ratio_check['bounds'],
                    },
                }
            )

        if history_check['severity'] == 'error':
            zero_reasons.append(history_check['note'])
            data_issue = {
                'symbol': symbol,
                'issue_type': 'history_outlier',
                'details': {
                    'market': market,
                    'forecast_price': forecast_price,
                    'current_price': current_price,
                    'stats': history_check.get('stats', {}),
                },
            }
            data_issues.append(data_issue)
            symbol_issues.append(data_issue)
        elif history_check['severity'] == 'warning':
            symbol_issues.append(
                {
                    'symbol': symbol,
                    'issue_type': 'history_near_boundary',
                    'details': {
                        'market': market,
                        'forecast_price': forecast_price,
                        'current_price': current_price,
                        'stats': history_check.get('stats', {}),
                    },
                }
            )

        if scale_check['has_error']:
            zero_reasons.append(scale_check['note'])
            data_issue = {
                'symbol': symbol,
                'issue_type': 'scale_error',
                'details': {
                    'market': market,
                    'forecast_price': forecast_price,
                    'current_price': current_price,
                    'suspected_scale_factor': scale_check['suspected_scale_factor'],
                },
            }
            data_issues.append(data_issue)
            symbol_issues.append(data_issue)

        combined_zero_reason = '; '.join(dict.fromkeys(zero_reasons)) if zero_reasons else None

        # 信頼度再計算
        confidence_result = recalculate_confidence(
            forecast_price,
            current_price,
            original_confidence,
            validation['severity'],
            scale_check['has_error'],
            force_zero_reason=combined_zero_reason,
        )

        # 全体的な深刻度を判定
        overall_severity = validation['severity']
        if (
            validation.get('hard_limit_triggered')
            or not ratio_check['is_valid']
            or history_check['severity'] == 'error'
            or scale_check['has_error']
        ):
            overall_severity = 'error'
        elif overall_severity != 'warning' and (
            ratio_check['severity'] == 'warning' or history_check['severity'] == 'warning'
        ):
            overall_severity = 'warning'

        # 統計を更新
        if overall_severity == 'error':
            error_count += 1
        elif overall_severity == 'warning':
            warning_count += 1
        else:
            valid_count += 1

        ratio_value = ratio_check.get('ratio')
        ratio_serializable: Optional[float] = None
        if isinstance(ratio_value, (int, float)) and math.isfinite(ratio_value):
            ratio_serializable = round(float(ratio_value), 4)

        limit_threshold = validation.get('limit_threshold_pct')
        if isinstance(limit_threshold, (int, float)) and math.isfinite(limit_threshold):
            limit_threshold = round(float(limit_threshold), 2)
        else:
            limit_threshold = None

        # 予測データを構築
        forecast_data = {
            'symbol': symbol,
            'date': pred_date.strftime('%Y-%m-%d'),
            'forecast': round(forecast_price, 2),
            'current_price': round(current_price, 2),
            'confidence': {
                'original': round(original_confidence, 3),
                'adjusted': round(confidence_result['adjusted_confidence'], 3),
                'adjustment_reason': confidence_result['adjustment_reason'],
            },
            'validation': {
                'is_valid': validation['is_valid'],
                'severity': validation['severity'],
                'price_change_pct': round(validation['price_change_pct'], 2),
                'issue': validation['issue'],
                'hard_limit_triggered': validation.get('hard_limit_triggered', False),
                'limit_threshold_pct': limit_threshold,
            },
            'scale_check': {
                'has_error': scale_check['has_error'],
                'suspected_scale_factor': round(scale_check['suspected_scale_factor'], 2)
                if scale_check['suspected_scale_factor']
                else None,
                'note': scale_check['note'],
            },
            'ratio_check': {
                'is_valid': ratio_check['is_valid'],
                'severity': ratio_check['severity'],
                'ratio': ratio_serializable,
                'bounds': ratio_check.get('bounds'),
                'note': ratio_check['note'],
            },
            'history_check': {
                'severity': history_check['severity'],
                'note': history_check['note'],
                'stats': history_check.get('stats'),
            },
            'overall_severity': overall_severity,
        }

        if 'market' in row:
            forecast_data['market'] = row['market']
        else:
            forecast_data['market'] = market

        if history_value is not None:
            forecast_data['historical_context'] = history_value

        if symbol_issues:
            forecast_data['data_quality_flags'] = [issue['issue_type'] for issue in symbol_issues]

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
            'confidence_reduction_reason': (
                'Guardrail violations triggered confidence resets'
                if error_count > 0
                else 'No guardrail violations detected'
            )
        },
        'forecasts': forecasts,
        'summary': {
            'valid_predictions': valid_count,
            'warning_predictions': warning_count,
            'error_predictions': error_count,
            'recommendation': recommendation,
            'data_quality': f"{data_quality} - {error_count}/{total_predictions} predictions flagged by guardrails"
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
    today = date.today()
    us_date = get_next_trading_day(today, 'US')
    jp_date = get_next_trading_day(today, 'JP')

    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', '7203', '6758', '8306', '9984']
    markets = ['US', 'US', 'US', 'US', 'JP', 'JP', 'JP', 'JP']
    current_prices = [192.15, 135.42, 415.33, 248.27, 2250.5, 15500.0, 1780.0, 6600.0]
    forecast_prices = [price * 1.015 for price in current_prices]
    confidences = [0.8, 0.76, 0.82, 0.78, 0.74, 0.85, 0.77, 0.72]
    history_placeholder: List[List[Dict[str, Any]]] = [[] for _ in symbols]
    dates = [us_date.strftime('%Y-%m-%d')] * 4 + [jp_date.strftime('%Y-%m-%d')] * 4

    sample_data = {
        'symbol': symbols,
        'market': markets,
        'forecast': [round(value, 2) for value in forecast_prices],
        'current_price': current_prices,
        'confidence': confidences,
        'history': history_placeholder,
        'date': dates,
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
    parser.add_argument('--us-symbols', nargs='*', default=[], help='米国株のシンボル')
    parser.add_argument('--jp-symbols', nargs='*', default=[], help='日本株のシンボル')
    parser.add_argument('--days-ahead', type=int, default=1, help='予測する営業日数')
    parser.add_argument('--lookback-days', type=int, default=365, help='履歴取得日数')
    parser.add_argument('--start-date', type=str, help='履歴開始日 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='履歴終了日 (YYYY-MM-DD)')
    parser.add_argument('--csv', type=str, help='既存の予測CSVファイル')
    parser.add_argument('--output', type=str, default='data/stock_data/predictions/forecast.csv', help='予測CSVの出力先')
    parser.add_argument('--json-output', type=str, default='darwin_analysis/llm_prompts/forecast.json', help='LLM用JSONの出力先')
    parser.add_argument('--raw-output', type=str, help='取得した履歴データの保存先')
    parser.add_argument('--log', type=str, help='ログファイルのパス')
    args = parser.parse_args()

    _setup_logging(args.log)

    forecast_df: Optional[pd.DataFrame] = None

    if args.csv and Path(args.csv).exists():
        logging.info("Loading forecast from existing CSV: %s", args.csv)
        forecast_df = pd.read_csv(args.csv)
    else:
        logging.info("Fetching historical data for prediction")
        if fetch_stock_data is None:
            logging.warning("fetch_stock_data is unavailable, using sample data")
            history_df = pd.DataFrame()
        else:
            history_df = fetch_stock_data(
                us_symbols=args.us_symbols,
                jp_symbols=args.jp_symbols,
                start_date=_parse_optional_date(args.start_date),
                end_date=_parse_optional_date(args.end_date),
                lookback_days=args.lookback_days,
            )

        if args.raw_output and 'history_df' in locals() and not history_df.empty:
            raw_path = Path(args.raw_output)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_df = history_df.copy()
            tmp_df['date'] = pd.to_datetime(tmp_df['date']).dt.strftime('%Y-%m-%d')
            tmp_df.to_csv(raw_path, index=False)
            logging.info("Saved raw history CSV: %s", raw_path)

        forecast_df = prepare_forecast_dataframe(history_df, days_ahead=args.days_ahead)
        save_forecast_csv(forecast_df, args.output)

    result = generate_llm_prompts(forecast_df=forecast_df)

    json_output = save_json_output(result, args.json_output)

    print(f"✅ JSON生成完了: {json_output}")
    print(f"   ステータス: {result['generation_status']}")
    print(f"   有効予測: {result['summary']['valid_predictions']}")
    print(f"   警告予測: {result['summary']['warning_predictions']}")
    print(f"   エラー予測: {result['summary']['error_predictions']}")
    print(f"   データ品質: {result['summary']['data_quality']}")


if __name__ == "__main__":
    main()

