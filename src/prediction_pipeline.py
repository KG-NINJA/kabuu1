"""
予測パイプラインのエントリーポイント
強化学習との連携やスケジューリングを統合
"""

import json
import logging
import logging.handlers
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import schedule
import yaml  # type: ignore[import-untyped]

# プロジェクトルートをパスに追加（強化学習モジュールを参照するため）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ログ対象ライブラリの静音化を後段で実施


def setup_logging(config: dict) -> None:
    """ロギング設定を初期化（日本語コメント必須ルールに対応）"""
    log_config = config.get("logging", {})
    log_file = log_config.get("file", "prediction_pipeline.log")

    handlers: List[logging.Handler] = [
        RotatingFileHandler(
            log_file,
            maxBytes=log_config.get("max_bytes", 5 * 1024 * 1024),
            backupCount=log_config.get("backup_count", 5),
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ]

    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # ノイズの多い外部ライブラリのログレベルを抑制
    for lib in ["matplotlib", "tensorflow", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


class Config:
    """YAMLベースの設定管理クラス"""

    def __init__(self, config_path: Optional[str] = None) -> None:
        # デフォルトの設定ファイルパスをスクリプト配置ディレクトリ基準に設定
        self.config_path: Path
        if config_path is None:
            # スクリプトの配置ディレクトリを基準に config.yaml を探す
            self.config_path = Path(__file__).resolve().parent / "config.yaml"
        else:
            self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """設定ファイルを読み込む（存在しない場合は空設定）"""
        try:
            with self.config_path.open("r", encoding="utf-8") as handle:
                self._config = yaml.safe_load(handle) or {}
            logging.info(f"設定ファイルを読み込みました: {self.config_path}")
        except FileNotFoundError:
            logging.warning(
                f"設定ファイルが見つかりませんでした: {self.config_path}。デフォルト値を使用します。"
            )
            self._config = {}
        except Exception as exc:
            logging.error(
                f"設定ファイル読み込み中にエラー: {exc}。デフォルト値を使用します。"
            )
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """ドット区切りキーで設定を取得"""
        value: Any = self._config
        for fragment in key.split("."):
            if not isinstance(value, dict) or fragment not in value:
                return default
            value = value[fragment]
        return value


class PredictionModel:
    """予測モデルのダミー実装（本番では差し替え予定）"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = self._build_model()

    def _build_model(self) -> Dict[str, Any]:
        """モデルの構築（ここでは仮実装）"""
        logging.info("ダミー予測モデルを構築しました")
        return {"model": "sample_model"}

    def predict(self, features: Dict[str, Any]) -> float:
        """特徴量から予測値を生成（デモとして固定値を返す）"""
        logging.debug(f"入力特徴量: {features}")
        return 100.0

    def get_params(self) -> Dict[str, Any]:
        """現在のハイパーパラメータを返却"""
        return {
            "learning_rate": self.config.get("model.learning_rate", 0.001),
            "hidden_units": self.config.get("model.hidden_units", 128),
            "batch_size": self.config.get("model.batch_size", 32),
            "epochs": self.config.get("model.epochs", 10),
        }

    def adjust_learning_rate(self, new_lr: float) -> None:
        """学習率を更新（本実装ではログのみ）"""
        logging.info(f"学習率を {new_lr} に変更します")

    def retrain(self, lookback_days: int = 365) -> None:
        """再学習処理の呼び出し（データ取得等は今後実装）"""
        logging.info(f"モデルを再学習します（直近 {lookback_days} 日のデータを想定）")
        logging.info("再学習が完了しました")


class PredictionPipeline:
    """予測処理と強化学習を束ねるパイプライン"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = PredictionModel(config)

        # NVDA専用の強化学習ハブを初期化
        from nvda_reinforcement import NvdaReinforcementHub, TARGET_SYMBOL  # type: ignore[import-not-found]

        self.target_symbol = str(config.get("nvda.symbol", TARGET_SYMBOL) or TARGET_SYMBOL)
        base_dir = Path(config.get("nvda.base_dir", "nvda_learning"))
        reward_threshold = float(config.get("nvda.reward_threshold", 0.0))
        self.rl_hub = NvdaReinforcementHub(
            base_dir=base_dir,
            reward_threshold=reward_threshold,
        )

        # 出力ディレクトリは NVDA 専用ハブと共有
        self.validation_dir = self.rl_hub.validation_dir
        self.prediction_dir = self.rl_hub.prediction_dir
        self.metrics_path = Path("performance_metrics.json")
        self.pending_path = self.prediction_dir / "pending_predictions.json"

        for directory in [self.validation_dir, self.prediction_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.pending_predictions: List[Dict[str, Any]] = (
            self._load_pending_predictions()
        )

        self.setup_scheduler()
        logging.info("予測パイプラインを初期化しました")

    # -------------------- 内部ユーティリティ --------------------
    def _load_pending_predictions(self) -> List[Dict[str, Any]]:
        """待機中の予測レコードをファイルから復元"""
        if not self.pending_path.exists():
            return []
        try:
            with self.pending_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            logging.warning("待機中予測ファイルの読み込みに失敗したため初期化します")
            return []

    def _persist_pending_predictions(self) -> None:
        """待機リストを永続化"""
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        with self.pending_path.open("w", encoding="utf-8") as handle:
            json.dump(self.pending_predictions, handle, ensure_ascii=False, indent=2)

    def _append_metrics(self, metrics: Dict[str, Any]) -> None:
        """メトリクスを `performance_metrics.json` へ追記"""
        existing: List[Dict[str, Any]] = []
        if self.metrics_path.exists():
            try:
                with self.metrics_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
            except json.JSONDecodeError:
                logging.warning("メトリクスファイルが破損していたため再生成します")
                existing = []

        existing.append(metrics)
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2)

    # -------------------- 主要ロジック --------------------
    def predict(self, ticker: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """単一銘柄の予測を実行し、結果を待機リストへ登録"""
        try:
            logging.info(f"{ticker} の予測を開始します")
            predicted_price = self.model.predict(features)

            record = {
                "ticker": ticker,
                "predicted_price": predicted_price,
                "features": features,
                "timestamp": datetime.now().isoformat(),
                "status": "pending",
            }

            self.pending_predictions.append(record)
            self._persist_pending_predictions()
            self._append_metrics(
                {
                    "ticker": ticker,
                    "timestamp": record["timestamp"],
                    "predicted_price": predicted_price,
                    "status": "pending",
                }
            )

            logging.info(f"{ticker} の予測値: {predicted_price:.2f}")
            return record
        except Exception as exc:
            error_msg = f"{ticker} の予測中にエラーが発生しました: {exc}"
            logging.error(error_msg, exc_info=True)
            failure_record = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(exc),
            }
            self._append_metrics(failure_record)
            return failure_record

    def update_with_actual_price(self, ticker: str, actual_price: float) -> None:
        """実際の価格で保留中の予測を確定し、強化学習へ渡す"""
        try:
            for record in self.pending_predictions:
                if record.get("ticker") == ticker and record.get("status") == "pending":
                    record["actual_price"] = actual_price
                    record["status"] = "completed"
                    record["actual_timestamp"] = datetime.now().isoformat()

                    predicted_price = record["predicted_price"]
                    reward = self.rl_hub.record_outcome(
                        ticker=ticker,
                        predicted_price=predicted_price,
                        actual_price=actual_price,
                        model_params=self.model.get_params(),
                    )

                    record["reward"] = reward
                    record["prediction_error"] = (
                        predicted_price - actual_price
                    ) / actual_price
                    self._append_metrics(record)
                    logging.info(
                        f"{ticker} の実際値 {actual_price:.2f} を反映（報酬: {reward:.3f}）"
                    )
                    break
            else:
                logging.warning(f"{ticker} の保留予測が見つかりませんでした")

            # 完了済みレコードは保持しない（履歴は metrics に保存済み）
            self.pending_predictions = [
                rec
                for rec in self.pending_predictions
                if rec.get("status") != "completed"
            ]
            self._persist_pending_predictions()

            self.check_model_improvement()
        except Exception as exc:
            logging.error(f"実際価格の更新中にエラー: {exc}", exc_info=True)

    def check_model_improvement(self) -> None:
        """強化学習の指標に基づき改善が必要か判定"""
        try:
            improvement_needed, strategy = self.rl_hub.should_improve()
            if improvement_needed:
                logging.info(f"モデル改善が必要と判断: {strategy}")
                self.retrain_model(strategy)
        except Exception as exc:
            logging.error(f"改善判定中にエラー: {exc}", exc_info=True)

    def retrain_model(self, strategy: Dict[str, Any]) -> None:
        """強化学習から提示された戦略で再学習"""
        try:
            if "learning_rate" in strategy:
                self.model.adjust_learning_rate(strategy["learning_rate"])
            lookback_days = strategy.get(
                "lookback_period", self.config.get("model.lookback_days", 365)
            )
            self.model.retrain(lookback_days=lookback_days)
        except Exception as exc:
            logging.error(f"再学習中にエラー: {exc}", exc_info=True)

    # -------------------- スケジューリング関連 --------------------
    def setup_scheduler(self) -> None:
        """日次予測と週次レビューのスケジュールを登録"""
        prediction_time = self.config.get("schedule.prediction_time", "09:00")
        schedule.every().day.at(prediction_time).do(self.predict_all_tickers)

        model_review_day = self.config.get(
            "schedule.model_review_day", "sunday"
        ).lower()
        review_time = self.config.get("schedule.model_review_time", "22:00")
        getattr(schedule.every(), model_review_day).at(review_time).do(
            self.weekly_model_review
        )

        logging.info("スケジューラーを設定しました")

    def predict_all_tickers(self) -> None:
        """設定された銘柄リストに対して順次予測を実行"""
        try:
            tickers = self.get_tracking_tickers()
            logging.info(f"{len(tickers)} 銘柄の予測を開始します")
            for ticker in tickers:
                features = self.prepare_features(ticker)
                self.predict(ticker, features)
        except Exception as exc:
            logging.error(f"一括予測中にエラー: {exc}", exc_info=True)

    def weekly_model_review(self) -> None:
        """週次レビューで学習状況を確認"""
        try:
            insights = self.rl_hub.get_learning_insights()
            logging.info("週次レビューレポート")
            for key, value in insights.items():
                logging.info(f"- {key}: {value}")
            # レビュー結果もメトリクスとして記録
            review_record = {
                "timestamp": datetime.now().isoformat(),
                "type": "weekly_review",
                "insights": insights,
            }
            self._append_metrics(review_record)
            self.check_model_improvement()
        except Exception as exc:
            logging.error(f"週次レビュー中にエラー: {exc}", exc_info=True)

    # -------------------- プレースホルダーメソッド --------------------
    def get_tracking_tickers(self) -> List[str]:
        """監視対象の銘柄一覧（暫定実装）"""
        return [str(self.target_symbol)]

    def prepare_features(self, ticker: str) -> Dict[str, Any]:
        """特徴量を生成（データ取得ロジックは別途実装予定）"""
        logging.debug(f"{ticker} の特徴量を生成します")
        return {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1_000_000,
        }

    # -------------------- エントリーポイント --------------------
    def run(self) -> None:
        """常駐プロセスとしてスケジューラを実行"""
        logging.info("予測パイプラインを起動しました")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logging.info("ユーザー操作によりパイプラインを終了します")
        except Exception as exc:
            logging.error(f"パイプライン実行中にエラー: {exc}", exc_info=True)
            raise


def main() -> None:
    """スクリプト起動時のメイン処理"""
    config = Config()
    setup_logging(config._config)
    pipeline = PredictionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
