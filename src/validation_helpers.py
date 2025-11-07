"""株価予測検証ヘルパー関数.

営業日判定、価格検証、スケール誤差検出、信頼度調整を提供
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import holidays
import pandas as pd

try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal
except Exception:  # pragma: no cover - graceful fallback
    mcal = None


MARKET_CALENDAR_CODES: Dict[str, str] = {
    'US': 'XNYS',
    'JP': 'XTKS',
}

_CALENDAR_CACHE: Dict[str, Any] = {}

DAILY_CHANGE_LIMITS: Dict[str, Any] = {
    'default': 0.20,
    'markets': {
        'JP': 0.18,
    },
    'symbols': {
        'TSLA': 0.25,
    },
}

PRICE_RATIO_LIMITS: Dict[str, Any] = {
    'default': (0.5, 2.0),
    'markets': {
        'JP': (0.45, 2.1),
    },
    'symbols': {
        'TSLA': (0.4, 2.5),
        '9984': (0.45, 1.9),
        '6758': (0.5, 1.8),
    },
}

HISTORY_TOLERANCE: Dict[str, Any] = {
    'default': 0.15,
    'markets': {
        'JP': 0.10,
    },
}

HISTORY_STD_MULTIPLIER: Dict[str, Any] = {
    'default': 3.5,
    'markets': {
        'JP': 3.0,
    },
}


def _resolve_config_value(config: Dict[str, Any], symbol: str, market: str) -> Any:
    """設定辞書から銘柄・市場に応じた値を取得する。"""

    if 'symbols' in config and symbol in config['symbols']:
        return config['symbols'][symbol]
    if 'markets' in config and market in config['markets']:
        return config['markets'][market]
    return config.get('default')


def _get_market_calendar(market: str):
    """pandas-market-calendars を利用して市場カレンダーを取得。"""

    if mcal is None:
        return None
    if market not in _CALENDAR_CACHE:
        calendar_code = MARKET_CALENDAR_CODES.get(market, MARKET_CALENDAR_CODES['US'])
        try:
            _CALENDAR_CACHE[market] = mcal.get_calendar(calendar_code)
        except Exception:  # pragma: no cover - ライブラリ未対応市場
            _CALENDAR_CACHE[market] = None
    return _CALENDAR_CACHE[market]


def is_trading_day(check_date: date, market: str = 'US') -> bool:
    """
    営業日かどうかを判定
    
    Args:
        check_date: 判定する日付 (datetime.date)
        market: 'US' or 'JP'
    
    Returns:
        bool: True if trading day, False otherwise
    """
    calendar = _get_market_calendar(market)
    if calendar is not None:
        schedule = calendar.schedule(start_date=check_date, end_date=check_date)
        return not schedule.empty

    # pandas-market-calendars が利用できない場合は祝日ライブラリで判定
    if check_date.weekday() >= 5:
        return False

    if market == 'US':
        holiday_calendar = holidays.UnitedStates()
    elif market == 'JP':
        holiday_calendar = holidays.Japan()
    else:
        holiday_calendar = holidays.UnitedStates()

    return check_date not in holiday_calendar


def get_next_trading_day(start_date: date, market: str = 'US') -> date:
    """
    次の営業日を取得
    
    Args:
        start_date: 開始日
        market: 'US' or 'JP'
    
    Returns:
        date: 次の営業日
    """
    calendar = _get_market_calendar(market)
    if calendar is not None:
        schedule = calendar.schedule(start_date=start_date, end_date=start_date + timedelta(days=14))
        future_days = schedule.index.date
        for future_day in future_days:
            if future_day > start_date:
                return future_day

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
            'severity': 'error',
            'hard_limit_triggered': True,
            'limit_threshold_pct': 0.0,
        }

    if forecast_price <= 0:
        return {
            'is_valid': False,
            'issue': 'Forecast price is zero or negative',
            'price_change_pct': 0.0,
            'severity': 'error',
            'hard_limit_triggered': True,
            'limit_threshold_pct': 0.0,
        }

    # 価格変動率を計算
    change_pct = ((forecast_price - current_price) / current_price) * 100

    hard_limit = _resolve_config_value(DAILY_CHANGE_LIMITS, symbol, market) or DAILY_CHANGE_LIMITS['default']
    hard_limit_pct = hard_limit * 100
    warning_threshold = min(hard_limit_pct * 0.75, hard_limit_pct - 2.0)
    elevated_threshold = min(hard_limit_pct * 0.5, max(5.0, hard_limit_pct * 0.5))

    if abs(change_pct) > hard_limit_pct:
        return {
            'is_valid': False,
            'issue': f'{change_pct:.1f}% change exceeds {hard_limit_pct:.1f}% daily limit',
            'price_change_pct': change_pct,
            'severity': 'error',
            'hard_limit_triggered': True,
            'limit_threshold_pct': hard_limit_pct,
        }
    elif abs(change_pct) > warning_threshold:
        return {
            'is_valid': True,
            'issue': f'Large change: {change_pct:.1f}% - possible but unusual',
            'price_change_pct': change_pct,
            'severity': 'warning',
            'hard_limit_triggered': False,
            'limit_threshold_pct': hard_limit_pct,
        }
    elif abs(change_pct) > elevated_threshold:
        return {
            'is_valid': True,
            'issue': f'Moderate change: {change_pct:.1f}%',
            'price_change_pct': change_pct,
            'severity': 'ok',
            'hard_limit_triggered': False,
            'limit_threshold_pct': hard_limit_pct,
        }
    else:
        return {
            'is_valid': True,
            'issue': None,
            'price_change_pct': change_pct,
            'severity': 'ok',
            'hard_limit_triggered': False,
            'limit_threshold_pct': hard_limit_pct,
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


def enforce_price_ratio_limits(
    symbol: str,
    forecast_price: float,
    current_price: float,
    market: str = 'US'
) -> Dict[str, Any]:
    """現在値との倍率チェックを厳密化。"""

    lower, upper = _resolve_config_value(PRICE_RATIO_LIMITS, symbol, market) or PRICE_RATIO_LIMITS['default']
    ratio = forecast_price / current_price if current_price else float('inf')

    if current_price <= 0 or forecast_price <= 0:
        return {
            'is_valid': False,
            'severity': 'error',
            'note': 'Invalid price values',
            'ratio': ratio,
            'bounds': {'lower': lower, 'upper': upper},
        }

    if ratio < lower or ratio > upper:
        return {
            'is_valid': False,
            'severity': 'error',
            'note': f'Forecast ratio {ratio:.2f}x outside allowable range [{lower:.2f}, {upper:.2f}]',
            'ratio': ratio,
            'bounds': {'lower': lower, 'upper': upper},
        }

    buffer = 0.1
    if ratio < lower + buffer or ratio > upper - buffer:
        return {
            'is_valid': True,
            'severity': 'warning',
            'note': f'Forecast ratio {ratio:.2f}x near boundary',
            'ratio': ratio,
            'bounds': {'lower': lower, 'upper': upper},
        }

    return {
        'is_valid': True,
        'severity': 'ok',
        'note': 'Ratio within acceptable range',
        'ratio': ratio,
        'bounds': {'lower': lower, 'upper': upper},
    }


def _normalize_history(history: Any) -> List[Dict[str, Any]]:
    if history is None:
        return []
    if isinstance(history, str):
        try:
            parsed = json.loads(history)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [entry for entry in parsed if isinstance(entry, dict)]
        return []
    if isinstance(history, pd.DataFrame):
        return history.to_dict(orient='records')
    if isinstance(history, Iterable):
        normalized: List[Dict[str, Any]] = []
        for entry in history:
            if isinstance(entry, dict):
                normalized.append(entry)
        return normalized
    return []


def validate_history_bounds(
    symbol: str,
    market: str,
    history: Any,
    forecast_price: float,
    current_price: float,
    lookback_days: int = 60,
) -> Dict[str, Any]:
    """過去価格帯と標準偏差を利用してスケール誤差を検証。"""

    records = _normalize_history(history)
    if not records:
        return {
            'is_valid': True,
            'severity': 'unknown',
            'note': 'Insufficient history for validation',
        }

    df = pd.DataFrame(records)
    if 'close' not in df.columns:
        return {
            'is_valid': True,
            'severity': 'unknown',
            'note': 'History lacks close prices',
        }

    df['date'] = pd.to_datetime(df.get('date', datetime.utcnow()))
    df = df.sort_values('date').tail(lookback_days)
    closes = df['close'].astype(float)
    if closes.empty:
        return {
            'is_valid': True,
            'severity': 'unknown',
            'note': 'No close data after filtering',
        }

    history_min = closes.min()
    history_max = closes.max()
    avg_close = closes.mean()
    std_close = closes.std(ddof=0)

    tolerance = _resolve_config_value(HISTORY_TOLERANCE, symbol, market) or HISTORY_TOLERANCE['default']
    std_multiplier = _resolve_config_value(HISTORY_STD_MULTIPLIER, symbol, market) or HISTORY_STD_MULTIPLIER['default']

    lower_bound = min(history_min, current_price) * (1 - tolerance)
    upper_bound = max(history_max, current_price) * (1 + tolerance)

    if std_close > 0:
        std_lower = avg_close - std_multiplier * std_close
        std_upper = avg_close + std_multiplier * std_close
        lower_bound = min(lower_bound, std_lower)
        upper_bound = max(upper_bound, std_upper)

    if forecast_price < lower_bound or forecast_price > upper_bound:
        return {
            'is_valid': False,
            'severity': 'error',
            'note': (
                f'Forecast {forecast_price:.2f} outside 60-day band '
                f'[{lower_bound:.2f}, {upper_bound:.2f}] (min={history_min:.2f}, max={history_max:.2f})'
            ),
            'stats': {
                'history_min': float(history_min),
                'history_max': float(history_max),
                'history_mean': float(avg_close),
                'history_std': float(std_close),
                'tolerance': tolerance,
                'std_multiplier': std_multiplier,
            },
        }

    if forecast_price < history_min or forecast_price > history_max:
        return {
            'is_valid': True,
            'severity': 'warning',
            'note': (
                f'Forecast {forecast_price:.2f} outside raw min/max but within '
                f'adjusted band [{lower_bound:.2f}, {upper_bound:.2f}]'
            ),
            'stats': {
                'history_min': float(history_min),
                'history_max': float(history_max),
                'history_mean': float(avg_close),
                'history_std': float(std_close),
            },
        }

    return {
        'is_valid': True,
        'severity': 'ok',
        'note': 'Forecast within historical envelope',
        'stats': {
            'history_min': float(history_min),
            'history_max': float(history_max),
            'history_mean': float(avg_close),
            'history_std': float(std_close),
        },
    }


def recalculate_confidence(
    forecast_price: float,
    current_price: float,
    original_confidence: float,
    validation_severity: str,
    scale_error: bool = False,
    force_zero_reason: Optional[str] = None
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

    if force_zero_reason:
        adjusted_confidence = 0.0
        reason = force_zero_reason

    return {
        'original_confidence': original_confidence,
        'adjusted_confidence': adjusted_confidence,
        'adjustment_reason': reason
    }

