"""
予測パイプラインのメトリクス可視化用ミニダッシュボード
Flask + Plotlyで軽量に閲覧可能
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List

from flask import Flask, jsonify, render_template

# Flaskアプリを初期化
app = Flask(__name__, template_folder="templates")


def load_performance_data() -> List[dict]:
    """メトリクスファイルを読み込み（存在しない場合は空配列）"""
    metrics_path = Path("performance_metrics.json")
    if not metrics_path.exists():
        return []

    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            data: Any = json.load(handle)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


@app.route("/")
def dashboard() -> Any:
    """ダッシュボードを描画"""
    metrics = load_performance_data()
    threshold = datetime.now() - timedelta(days=30)
    recent_metrics = [
        m
        for m in metrics
        if datetime.fromisoformat(m.get("timestamp", "1970-01-01")) > threshold
    ]
    return render_template("dashboard.html", metrics=recent_metrics)


@app.route("/api/metrics")
def get_metrics() -> Any:
    """メトリクスをJSONで返すAPI"""
    return jsonify(load_performance_data())


def ensure_template() -> None:
    """テンプレートが存在しない場合はサンプルを自動生成"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    dashboard_path = templates_dir / "dashboard.html"

    if dashboard_path.exists():
        return

    # シンプルなテンプレートを生成（日本語コメント入り）
    dashboard_html = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8" />
    <title>予測パフォーマンスダッシュボード</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { margin-bottom: 30px; }
        .metric-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <h1>予測パフォーマンスダッシュボード</h1>

    <div class="chart" id="errorChart"></div>
    <div class="chart" id="rewardChart"></div>

    <h2>最近のメトリクス</h2>
    <div id="recentMetrics"></div>

    <script>
        fetch('/api/metrics')
            .then(response => response.json())
            .then(metrics => {
                metrics.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

                const dates = metrics.map(m => m.timestamp);
                const errors = metrics.map(m => (m.prediction_error || 0) * 100);

                Plotly.newPlot('errorChart', [{
                    x: dates,
                    y: errors,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '予測誤差 (%)'
                }], {
                    title: '予測誤差の推移',
                    xaxis: { title: '日時' },
                    yaxis: { title: '誤差 (%)' }
                });

                const rewards = metrics.map(m => m.reward || 0);
                Plotly.newPlot('rewardChart', [{
                    x: dates,
                    y: rewards,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '報酬'
                }], {
                    title: '報酬の推移',
                    xaxis: { title: '日時' },
                    yaxis: { title: '報酬' }
                });

                const recentMetrics = metrics.slice(-5).reverse();
                const metricsHtml = recentMetrics.map(m => `
                    <div class="metric-card">
                        <h3>${new Date(m.timestamp).toLocaleString('ja-JP')}</h3>
                        <p>銘柄: ${m.ticker || '不明'}</p>
                        <p>予測価格: ${m.predicted_price ? m.predicted_price.toFixed(2) : 'N/A'}</p>
                        <p>実際の価格: ${m.actual_price ? m.actual_price.toFixed(2) : '未反映'}</p>
                        <p>誤差: ${m.prediction_error ? (m.prediction_error * 100).toFixed(2) : 'N/A'}%</p>
                        <p>報酬: ${m.reward ? m.reward.toFixed(2) : 'N/A'}</p>
                    </div>
                `).join('');

                document.getElementById('recentMetrics').innerHTML = metricsHtml;
            })
            .catch(error => {
                console.error('メトリクス取得に失敗しました:', error);
                document.getElementById('recentMetrics').innerHTML =
                    '<p>メトリクス読み込みエラーが発生しました。</p>';
            });
    </script>
</body>
</html>
"""
    dashboard_path.write_text(dashboard_html, encoding="utf-8")


if __name__ == "__main__":
    ensure_template()
    host = os.environ.get("MONITOR_HOST", "0.0.0.0")
    port = int(os.environ.get("MONITOR_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
