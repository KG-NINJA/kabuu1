#!/usr/bin/env python3
"""Validate JSON structure of forecast analysis."""

import json
import sys

try:
    with open('darwin_analysis/forecast_analysis.json') as f:
        data = json.load(f)
    
    required_fields = [
        'timestamp',
        'next_trading_day',
        'symbols_predicted',
        'total_predictions',
        'forecasts',
        'statistics',
        'summary',
        'data_issues'
    ]
    
    missing = [f for f in required_fields if f not in data]
    
    if missing:
        print('ERROR: Missing fields: ' + str(missing))
        sys.exit(1)
    
    if not isinstance(data['forecasts'], list):
        print('ERROR: forecasts is not a list')
        sys.exit(1)
    
    for forecast in data['forecasts']:
        required_pred_fields = [
            'symbol', 'date', 'forecast', 'confidence',
            'validation', 'scale_check'
        ]
        missing_pred = [f for f in required_pred_fields if f not in forecast]
        if missing_pred:
            print('ERROR: Missing prediction fields in ' + forecast.get('symbol', 'unknown') + ': ' + str(missing_pred))
            sys.exit(1)
    
    print('OK: JSON structure is valid')
    print('OK: ' + str(len(data['forecasts'])) + ' forecasts validated')
    
except json.JSONDecodeError as e:
    print('ERROR: JSON decode error: ' + str(e))
    sys.exit(1)
except Exception as e:
    print('ERROR: ' + str(e))
    sys.exit(1)
