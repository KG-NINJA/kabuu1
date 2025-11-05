# マルチステージビルド - ビルドステージ
FROM python:3.11-slim as builder

WORKDIR /build

# ビルド依存関係をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 依存関係をインストール
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ========== 本番ステージ ==========
FROM python:3.11-slim

# メタデータ
LABEL maintainer="kabuu1"
LABEL description="Kabuu1 ML Prediction Pipeline"
LABEL version="1.0.0"

# 環境変数設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TZ=UTC

# 作業ディレクトリ設定
WORKDIR /app

# システムライブラリのインストール（実行時に必要なもののみ）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ビルドステージからPythonパッケージをコピー
COPY --from=builder /root/.local /root/.local

# PATHにユーザーのPythonパッケージ追加
ENV PATH=/root/.local/bin:$PATH

# アプリケーションコードをコピー
COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/
COPY requirements.txt .
COPY .gitignore .

# 必要なディレクトリを作成
RUN mkdir -p data/rl_results \
    && mkdir -p data/validation_results \
    && mkdir -p data/prediction_results \
    && mkdir -p logs

# 非rootユーザーを作成（セキュリティ）
RUN groupadd -r appuser && useradd -r -g appuser appuser

# アプリケーションの所有権変更
RUN chown -R appuser:appuser /app

# 非rootユーザーに切り替え
USER appuser

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; print('OK')" || exit 1

# デフォルトコマンド
CMD ["python", "-m", "src.prediction_pipeline"]
