name: Prediction Pipeline CI with LLM Output

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    # JP市場終値後：毎営業日 UTC 06:00（日本時間 15:00）
    - cron: '0 6 * * 1-5'
    # US市場終値後：毎営業日 UTC 21:00（日本時間 翌日 06:00）
    - cron: '0 21 * * 1-5'
  workflow_dispatch:

jobs:
  predict_and_validate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install core dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -r requirements.txt
          pip install holidays
          if [ -f prediction_pipeline/requirements.txt ]; then
            pip install -r prediction_pipeline/requirements.txt
          fi

      - name: Verify validation helpers
        run: |
          python -c "
          import sys
          sys.path.insert(0, '.')
          from src.validation_helpers import (
              is_trading_day,
              validate_price_prediction,
              detect_scale_error,
              recalculate_confidence
          )
          print('✅ All validation helpers imported successfully')
          "

      - name: Generate forecast CSV from real stock data
        run: |
          echo "📊 Generating forecast from real stock data..."
          PYTHONPATH=. python scripts/generate_forecast_csv.py \
            --output forecast_data.csv \
            --us-symbols AAPL GOOGL MSFT TSLA \
            --jp-symbols 9984 6758 7203 8306
          
          echo "✅ Forecast CSV generated"
          echo "📋 File content:"
          head -20 forecast_data.csv
          echo "📊 Total rows: $(wc -l < forecast_data.csv)"

      - name: Run prediction with validation
        run: |
          echo "🔍 Running prediction with validation..."
          PYTHONPATH=. python src/predict.py \
            --csv forecast_data.csv \
            --output darwin_analysis/forecast_analysis.json \
            --validate \
            --verbose
          
          echo "✅ Prediction completed"
          python -c "
          import json
          with open('darwin_analysis/forecast_analysis.json') as f:
              data = json.load(f)
          
          print(f'Symbols: {data.get(\"symbols_predicted\", 0)}')
          print(f'Total predictions: {data.get(\"total_predictions\", 0)}')
          print(f'Status: {data.get(\"generation_status\", \"unknown\")}')
          print(f'Valid predictions: {data.get(\"summary\", {}).get(\"valid_predictions\", 0)}')
          print(f'Error predictions: {data.get(\"summary\", {}).get(\"error_predictions\", 0)}')
          print(f'Data quality: {data.get(\"summary\", {}).get(\"data_quality\", \"unknown\")}')
          "

      - name: Validate JSON structure
        run: |
          echo "🔍 Validating JSON structure..."
          python -c "
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
                  print(f'❌ Missing fields: {missing}')
                  sys.exit(1)
              
              if not isinstance(data['forecasts'], list):
                  print('❌ forecasts is not a list')
                  sys.exit(1)
              
              for forecast in data['forecasts']:
                  required_pred_fields = [
                      'symbol', 'date', 'forecast', 'confidence',
                      'validation', 'scale_check'
                  ]
                  missing_pred = [f for f in required_pred_fields if f not in forecast]
                  if missing_pred:
                      print(f'❌ Missing prediction fields in {forecast.get(\"symbol\")}: {missing_pred}')
                      sys.exit(1)
              
              print('✅ JSON structure is valid')
              print(f'✅ {len(data[\"forecasts\"])} forecasts validated')
              
          except json.JSONDecodeError as e:
              print(f'❌ JSON decode error: {e}')
              sys.exit(1)
          except Exception as e:
              print(f'❌ Error: {e}')
              sys.exit(1)
          "

      - name: Generate LLM Prompts
        run: |
          echo "🧠 Generating LLM prompts..."
          mkdir -p darwin_analysis/llm_prompts
          
          PYTHONPATH=. python src/llm_prompt_generator.py \
            --json darwin_analysis/forecast_analysis.json \
            --output darwin_analysis/llm_prompts
          
          echo "✅ LLM prompts generated"
          echo "📋 Generated files:"
          ls -la darwin_analysis/llm_prompts/

      - name: Generate quality report
        run: |
          python << 'PYTHON_SCRIPT'
          import json
          from pathlib import Path
          
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
          issues_text = '\n'.join([f'- {issue}' for issue in issues])
          
          reliable_preds = []
          for forecast in data.get('forecasts', []):
              if forecast.get('validation', {}).get('severity') == 'ok':
                  symbol = forecast['symbol']
                  pred = forecast.get('forecast', 'N/A')
                  conf = forecast.get('confidence', {}).get('adjusted', 0)
                  reliable_preds.append(f'- **{symbol}**: {pred} (Confidence: {conf:.0%})')
          
          reliable_text = '\n'.join(reliable_preds)
          
          report = f'''# Stock Prediction Quality Report
          
## Generation Status
- **Status**: {gen_status}
- **Timestamp**: {timestamp}
- **Next Trading Day**: {next_trading}

## Prediction Statistics
- **Total Symbols**: {symbols}
- **Total Predictions**: {total_preds}
- **Valid Predictions**: {valid_preds}
- **Warning Predictions**: {warning_preds}
- **Error Predictions**: {error_preds}

## Confidence Scores
- **Original Average**: {avg_conf_orig:.2%}
- **Adjusted Average**: {avg_conf_adj:.2%}
- **Reduction**: {conf_reduction:.1f}%

## Data Quality
- **Quality**: {data_quality}
- **Recommendation**: {recommendation}

## Issues Detected
{issues_text if issues_text else '- No issues detected'}

## Reliable Predictions
Only use these for LLM analysis:
{reliable_text if reliable_text else '- No reliable predictions'}
'''
          
          Path('darwin_analysis/QUALITY_REPORT.md').write_text(report)
          print(report)
          PYTHON_SCRIPT

      - name: Upload artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: prediction-analysis-${{ github.run_number }}
          path: darwin_analysis/
          retention-days: 30

      - name: Summary
        run: |
          echo "## ✅ Pipeline Completed"
          echo ""
          echo "### 📁 Generated Files"
          echo "- **JSON Analysis**: darwin_analysis/forecast_analysis.json"
          echo "- **LLM Prompts**: darwin_analysis/llm_prompts/"
          echo "- **Quality Report**: darwin_analysis/QUALITY_REPORT.md"
          echo ""
          echo "### 🧠 Ready for LLM Analysis"
          echo "Copy any prompt from darwin_analysis/llm_prompts/ to:"
          echo "- Claude: https://claude.ai"
          echo "- GPT-4: https://openai.com/chat"
          echo "- Gemini: https://gemini.google.com"
          echo "- Perplexity: https://perplexity.ai"
          echo ""
          echo "Each LLM will provide its own analysis based on the same data."
