#!/usr/bin/env python3
"""Generate quality report from forecast analysis JSON."""

import json
from pathlib import Path

def generate_quality_report():
    """Read forecast_analysis.json and generate markdown report."""
    
    with open('darwin_analysis/forecast_analysis.json') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    gen_status = data.get('generation_status', 'UNKNOWN')
    timestamp = data.get('timestamp', 'N/A')
    next_trading = data.get('next_trading_day', 'N/A')
    symbols = data.get('symbols_predicted', 0)
    total_preds = data.get('total_predictions', 0)
    valid_preds = summary.get('valid_predictions', 0)
    warning_preds = summary.get('warning_predictions', 0)
    error_preds = summary.get('error_predictions', 0)
    
    stats = data.get('statistics', {})
    avg_conf_orig = stats.get('avg_confidence_original', 0)
    avg_conf_adj = stats.get('avg_confidence_adjusted', 0)
    conf_reduction = (avg_conf_orig - avg_conf_adj) * 100
    
    data_quality = summary.get('data_quality', 'UNKNOWN')
    recommendation = summary.get('recommendation', 'N/A')
    
    issues = data.get('data_issues', [])
    issues_text = '\n'.join(['- ' + issue for issue in issues])
    if not issues_text:
        issues_text = '- No issues detected'
    
    reliable_preds = []
    for forecast in data.get('forecasts', []):
        if forecast.get('validation', {}).get('severity') == 'ok':
            symbol = forecast['symbol']
            pred = forecast.get('forecast', 'N/A')
            conf = forecast.get('confidence', {}).get('adjusted', 0)
            reliable_preds.append('- **' + symbol + '**: ' + str(pred) + ' (Confidence: ' + f'{conf:.0%}' + ')')
    
    reliable_text = '\n'.join(reliable_preds)
    if not reliable_text:
        reliable_text = '- No reliable predictions'
    
    report = '''# Stock Prediction Quality Report

## Generation Status
- **Status**: ''' + gen_status + '''
- **Timestamp**: ''' + timestamp + '''
- **Next Trading Day**: ''' + next_trading + '''

## Prediction Statistics
- **Total Symbols**: ''' + str(symbols) + '''
- **Total Predictions**: ''' + str(total_preds) + '''
- **Valid Predictions**: ''' + str(valid_preds) + '''
- **Warning Predictions**: ''' + str(warning_preds) + '''
- **Error Predictions**: ''' + str(error_preds) + '''

## Confidence Scores
- **Original Average**: ''' + f'{avg_conf_orig:.2%}' + '''
- **Adjusted Average**: ''' + f'{avg_conf_adj:.2%}' + '''
- **Reduction**: ''' + f'{conf_reduction:.1f}%' + '''

## Data Quality
- **Quality**: ''' + data_quality + '''
- **Recommendation**: ''' + recommendation + '''

## Issues Detected
''' + issues_text + '''

## Reliable Predictions
Only use these for LLM analysis:
''' + reliable_text
    
    output_path = Path('darwin_analysis/QUALITY_REPORT.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    
    print(report)

if __name__ == '__main__':
    generate_quality_report()
