"""
株価予測検証ヘルパー関数
営業日判定、価格検証、スケール誤差検出、信頼度調整を提供
"""

from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
import holidays


def is_trading_day(check_date: date, market: str = 'US') -> bool:
    """
    営業日かどうかを判定
    
    Args:
        check_date: 判定する日付 (datetime.date)
        market: 'US' or 'JP'
    
    Returns:
        bool: True if trading day, False otherwise
    """
    # 土日を除外 (weekday: 5=土曜, 6=日曜)
    if check_date.weekday() >= 5:
        return False
    
    # 市場別の祝日をチェック
    if market == 'US':
        us_holidays = holidays.UnitedStates()
        if check_date in us_holidays:
            return False
    elif market == 'JP':
        jp_holidays = holidays.Japan()
        if check_date in jp_holidays:
            return False
    
    return True


def get_next_trading_day(start_date: date, market: str = 'US') -> date:
    """
    次の営業日を取得
    
    Args:
        start_date: 開始日
        market: 'US' or 'JP'
    
    Returns:
        date: 次の営業日
    """
    next_date = start_date + timedelta(days=1)
    while not is_trading_day(next_date, market):
        next_date += timedelta(days=1)
    return next_date


def validate_price_prediction(
    symbol: str,
    forecast_price: float,
    current_price: float,
    market: str = 'US'
) -> Dict[str, Any]:
    """
    予測価格が現実的かチェック
    
    Args:
        symbol: 銘柄コード（AAPL等）
        forecast_price: 予測価格
        current_price: 現在価格
        market: 'US' or 'JP'
    
    Returns:
        dict: {
            'is_valid': bool,
            'issue': str or None,
            'price_change_pct': float,
            'severity': 'error' | 'warning' | 'ok'
        }
    """
    if current_price <= 0:
        return {
            'is_valid': False,
            'issue': 'Current price is zero or negative',
            'price_change_pct': 0.0,
            'severity': 'error'
        }
    
    if forecast_price <= 0:
        return {
            'is_valid': False,
            'issue': 'Forecast price is zero or negative',
            'price_change_pct': 0.0,
            'severity': 'error'
        }
    
    # 価格変動率を計算
    change_pct = ((forecast_price - current_price) / current_price) * 100
    
    # 検証ルール
    # 1営業日の変動は ±5% を超えない（金融危機を除く）
    # ±10% を超えたら警告
    # ±20% を超えたら エラー
    
    if abs(change_pct) > 20:
        return {
            'is_valid': False,
            'issue': f'{change_pct:.1f}% change is unrealistic for single trading day',
            'price_change_pct': change_pct,
            'severity': 'error'
        }
    elif abs(change_pct) > 10:
        return {
            'is_valid': True,
            'issue': f'Large change: {change_pct:.1f}% - possible but unusual',
            'price_change_pct': change_pct,
            'severity': 'warning'
        }
    elif abs(change_pct) > 5:
        return {
            'is_valid': True,
            'issue': f'Moderate change: {change_pct:.1f}%',
            'price_change_pct': change_pct,
            'severity': 'ok'
        }
    else:
        return {
            'is_valid': True,
            'issue': None,
            'price_change_pct': change_pct,
            'severity': 'ok'
        }


def detect_scale_error(
    symbol: str,
    forecast_price: float,
    current_price: float
) -> Dict[str, Any]:
    """
    スケール誤差の検出
    例）ソフトバンク予測 3,136円（実際 23,000円）→ 明らかに桁違い
    
    Args:
        symbol: 銘柄コード
        forecast_price: 予測価格
        current_price: 現在価格
    
    Returns:
        dict: {
            'has_error': bool,
            'suspected_scale_factor': float,
            'corrected_price': float or None,
            'note': str
        }
    """
    if forecast_price <= 0 or current_price <= 0:
        return {
            'has_error': False,
            'suspected_scale_factor': 1.0,
            'corrected_price': None,
            'note': 'Invalid price values'
        }
    
    # 価格比率を計算
    ratio = max(forecast_price, current_price) / min(forecast_price, current_price)
    
    # 価格変動率も考慮
    change_pct = abs(((forecast_price - current_price) / current_price) * 100)
    
    # 10倍以上の乖離 → スケール誤差疑い
    # または、価格変動率が100%以上で比率が3倍以上 → スケール誤差疑い
    # または、価格変動率が80%以上で比率が5倍以上 → スケール誤差疑い（ソフトバンク等）
    if ratio > 10:
        return {
            'has_error': True,
            'suspected_scale_factor': ratio,
            'corrected_price': None,  # 補正不可（要調査）
            'note': f'{ratio:.2f}x deviation - scale error suspected'
        }
    elif ratio > 3 and change_pct > 100:
        # 価格変動率が100%以上で比率が3倍以上の場合もスケール誤差疑い
        return {
            'has_error': True,
            'suspected_scale_factor': ratio,
            'corrected_price': None,
            'note': f'{ratio:.2f}x deviation with {change_pct:.1f}% change - scale error suspected'
        }
    elif ratio > 5 and change_pct > 80:
        # 価格変動率が80%以上で比率が5倍以上の場合もスケール誤差疑い
        return {
            'has_error': True,
            'suspected_scale_factor': ratio,
            'corrected_price': None,
            'note': f'{ratio:.2f}x deviation with {change_pct:.1f}% change - scale error suspected'
        }
    elif ratio > 5:
        return {
            'has_error': False,
            'suspected_scale_factor': ratio,
            'corrected_price': None,
            'note': f'{ratio:.2f}x deviation - unusual but possible'
        }
    else:
        return {
            'has_error': False,
            'suspected_scale_factor': ratio,
            'corrected_price': None,
            'note': 'Price scale appears correct'
        }


def recalculate_confidence(
    forecast_price: float,
    current_price: float,
    original_confidence: float,
    validation_severity: str,
    scale_error: bool = False
) -> Dict[str, Any]:
    """
    信頼度を検証結果に基づいて調整
    
    Args:
        forecast_price: 予測価格
        current_price: 現在価格
        original_confidence: 元の信頼度
        validation_severity: 'error' | 'warning' | 'ok'
        scale_error: スケール誤差があるかどうか
    
    Returns:
        dict: {
            'original_confidence': float,
            'adjusted_confidence': float,
            'adjustment_reason': str
        }
    """
    adjusted_confidence = original_confidence
    reason = ""
    
    # スケール誤差がある場合は信頼度を0に
    if scale_error:
        adjusted_confidence = 0.0
        reason = "Scale error detected - data corruption suspected"
    # 検証結果に基づいて調整
    elif validation_severity == 'error':
        adjusted_confidence = 0.0
        reason = "Unrealistic price change - likely model error"
    elif validation_severity == 'warning':
        adjusted_confidence *= 0.5
        reason = "Large but possible price change - reduced confidence"
    else:
        reason = "Within normal range - confidence maintained"
    
    return {
        'original_confidence': original_confidence,
        'adjusted_confidence': adjusted_confidence,
        'adjustment_reason': reason
    }

