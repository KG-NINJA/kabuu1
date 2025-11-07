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

permissions:
  contents: write

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

      - name: Generate forecast directly (no raw_data.csv needed)
        run: |
          echo "📊 Generating forecast from stock data..."
          mkdir -p darwin_analysis
          
          PYTHONPATH=. python scripts/generate_forecast_csv.py \
            --us-symbols AAPL GOOGL MSFT TSLA \
            --jp-symbols 9984 6758 7203 8306 \
            --output forecast_data.csv \
          data = stock.history(period='1y')

          
          echo ""
          echo "✅ Forecast generated"
          echo "📋 File content:"
          head -20 forecast_data.csv
          echo ""
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
          echo "📊 JSON Statistics:"
          python -c 
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
          echo "📊 Generating quality report..."
          python -c "
          import json
          from pathlib import Path
          
          with open('darwin_analysis/forecast_analysis.json') as f:
              data = json.load(f)
          
          summary = data.get('summary', {})
          
          report = f'''# 📊 Stock Prediction Quality Report
          
## Generation Status
- **Status**: {data.get('generation_status', 'UNKNOWN')}
- **Timestamp**: {data.get('timestamp', 'N/A')}
- **Next Trading Day**: {data.get('next_trading_day', 'N/A')}

## Prediction Statistics
- **Total Symbols**: {data.get('symbols_predicted', 0)}
- **Total Predictions**: {data.get('total_predictions', 0)}
- **Valid Predictions**: {summary.get('valid_predictions', 0)}
- **Warning Predictions**: {summary.get('warning_predictions', 0)}
- **Error Predictions**: {summary.get('error_predictions', 0)}

## Confidence Scores
- **Original Average**: {data.get('statistics', {}).get('avg_confidence_original', 0):.2%}
- **Adjusted Average**: {data.get('statistics', {}).get('avg_confidence_adjusted', 0):.2%}
- **Reduction**: {(data.get('statistics', {}).get('avg_confidence_original', 0) - data.get('statistics', {}).get('avg_confidence_adjusted', 0)) * 100:.1f}%

## Data Quality
- **Quality**: {summary.get('data_quality', 'UNKNOWN')}
- **Recommendation**: {summary.get('recommendation', 'N/A')}

## Issues Detected
{chr(10).join([f\"- {issue}\" for issue in data.get('data_issues', [])])}

## Reliable Predictions
Only use these for LLM analysis:
'''
          
          for forecast in data.get('forecasts', []):
              if forecast.get('validation', {}).get('severity') == 'ok':
                  report += f\"\\n- **{forecast['symbol']}**: {forecast.get('forecast', 'N/A')} (Confidence: {forecast.get('confidence', {}).get('adjusted', 0):.0%})\"
          
          Path('darwin_analysis/QUALITY_REPORT.md').write_text(report)
          print(report)
          "

      - name: Commit results to repository
        run: |
          git config user.email "action@github.com"
          git config user.name "GitHub Action"
          
          git add darwin_analysis/ || true
          
          # ファイルが変更されたかチェック
          if ! git diff --staged --quiet; then
              git commit -m "🧬 Stock prediction analysis - $(date +'%Y-%m-%d %H:%M:%S UTC')" || true
              git push origin main || echo "Push skipped"
          else
              echo "ℹ️ No changes to commit"
          fi
        continue-on-error: true
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

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
