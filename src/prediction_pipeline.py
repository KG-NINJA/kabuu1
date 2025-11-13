"""
予測パイプラインのエントリーポイント
強化学習との連携やスケジューリングを統合
"""

import argparse
import json
import logging
import logging.handlers
import sys
import time
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import schedule
import yaml  # type: ignore[import-untyped]

import pandas as pd
from sklearn.linear_model import LinearRegression

# プロジェクトルートをパスに追加（強化学習モジュールを参照するため）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ログ対象ライブラリの静音化を後段で実施


from .data_fetcher import fetch_stock_data


def setup_logging(config: dict) -> None:
    """ロギング設定を初期化（日本語コメント必須ルールに対応）"""
    log_config = config.get("logging", {})
    log_file = log_config.get("file", "prediction_pipeline.log")

    log_path: Optional[Path]
    if log_file:
        candidate = Path(log_file)
        log_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    else:
        log_path = None

    warnings: List[str] = []
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - unexpected but defensive
            warnings.append(
                f"ログディレクトリの作成に失敗したため標準出力のみを使用します: {exc}"
            )
            log_path = None

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        handlers.insert(
            0,
            RotatingFileHandler(
                log_path,
                maxBytes=int(log_config.get("max_bytes", 5 * 1024 * 1024)),
                backupCount=int(log_config.get("backup_count", 5)),
                encoding="utf-8",
            ),
        )

    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    for message in warnings:
        logging.getLogger(__name__).warning(message)

    # ノイズの多い外部ライブラリのログレベルを抑制
    for lib in ["matplotlib", "tensorflow", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


class Config:
    """YAMLベースの設定管理クラス"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        # デフォルトの設定ファイルパスをリポジトリルート基準で解決
        self.config_path = self._resolve_config_path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    @staticmethod
    def _resolve_config_path(
        config_path: Optional[Union[str, Path]]
    ) -> Path:
        """設定ファイルの場所を解決する"""

        if config_path is None:
            return DEFAULT_CONFIG_PATH

        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate

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
    """NVDA 専用の線形回帰モデルを管理するクラス"""

    feature_columns = ["close", "ma_5", "ma_20", "return_1d", "volatility_5d"]

    def __init__(self, config: Config, target_symbol: str) -> None:
        self.config = config
        self.target_symbol = target_symbol
        self.lookback_days = int(self.config.get("model.lookback_days", 365))
        self.regressor = LinearRegression()
        self._is_trained = False
        self.last_trained_at: Optional[str] = None
        self.retrain(lookback_days=self.lookback_days)

    def _prepare_training_frame(self, history: pd.DataFrame) -> pd.DataFrame:
        augmented = self._augment_with_features(history)
        augmented["target"] = augmented["close"].shift(-1)
        augmented = augmented.dropna(subset=self.feature_columns + ["target"])
        return augmented

    @staticmethod
    def _augment_with_features(history: pd.DataFrame) -> pd.DataFrame:
        frame = history.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame.sort_values("date", inplace=True)


        # GitHub Actions 上で `close` 列が DataFrame や配列になり TypeError が
        # 発生したため、あらゆるケースに対応して 1 次元の Series として
        # 正規化してから数値変換を行う。これにより学習データが欠損しても
        # 無視され、スタブではなく実データでも安定する。
        raw_close = frame.get("close")
        if isinstance(raw_close, pd.DataFrame):
            raw_close = raw_close.iloc[:, 0]
        elif not isinstance(raw_close, pd.Series):
            raw_close = pd.Series(raw_close, index=frame.index)

        frame.loc[:, "close"] = pd.to_numeric(raw_close, errors="coerce")
        frame["ma_5"] = frame["close"].rolling(window=5, min_periods=5).mean()
        frame["ma_20"] = frame["close"].rolling(window=20, min_periods=20).mean()
        frame["return_1d"] = frame["close"].pct_change()
        frame["volatility_5d"] = (
            frame["return_1d"].rolling(window=5, min_periods=5).std().fillna(0.0)
        )
        return frame

    def predict(self, features: Dict[str, Any]) -> float:
        """学習済みモデルから終値を予測する"""

        logging.debug(f"入力特徴量: {features}")
        if not self._is_trained:
            logging.warning("モデルが未学習のため終値を直接返します")
            return float(features.get("close", 0.0))

        vector = [[float(features[column]) for column in self.feature_columns]]
        prediction = float(self.regressor.predict(vector)[0])
        logging.debug(f"線形回帰モデルによる予測値: {prediction}")
        return prediction

    def get_params(self) -> Dict[str, Any]:
        """現在のハイパーパラメータを返却"""

        return {
            "learning_rate": self.config.get("model.learning_rate", 0.001),
            "hidden_units": self.config.get("model.hidden_units", 128),
            "batch_size": self.config.get("model.batch_size", 32),
            "epochs": self.config.get("model.epochs", 10),
            "trained_at": self.last_trained_at,
        }

    def adjust_learning_rate(self, new_lr: float) -> None:
        """学習率を更新（ロギングのみ）"""

        logging.info(f"学習率を {new_lr} に変更します")

    def retrain(self, lookback_days: int = 365) -> None:
        """直近データで線形回帰モデルを再学習"""

        logging.info(f"NVDA の過去 {lookback_days} 日データで再学習を実行します")
        history = fetch_stock_data(
            us_symbols=[self.target_symbol],
            jp_symbols=[],
            lookback_days=lookback_days,
        )
        history = history[history["symbol"] == self.target_symbol]

        if history.empty:
            logging.error("学習用データが取得できなかったため再学習をスキップします")
            self._is_trained = False
            return

        training_frame = self._prepare_training_frame(history)
        if training_frame.empty:
            logging.error("特徴量が不足しているためモデルの学習に失敗しました")
            self._is_trained = False
            return

        self.regressor.fit(training_frame[self.feature_columns], training_frame["target"])
        self._is_trained = True
        self.last_trained_at = datetime.now(UTC).isoformat()
        logging.info("再学習が完了しました")


class PredictionPipeline:
    """予測処理と強化学習を束ねるパイプライン"""

    def __init__(self, config: Config) -> None:
        self.config = config

        # NVDA専用の強化学習ハブを初期化
        from .nvda_reinforcement import NvdaReinforcementHub, TARGET_SYMBOL  # type: ignore[import-not-found]

        self.target_symbol = str(config.get("nvda.symbol", TARGET_SYMBOL) or TARGET_SYMBOL)
        self.model = PredictionModel(config, self.target_symbol)
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

        actual_price_time = self.config.get("schedule.actual_price_time", "16:30")
        schedule.every().day.at(actual_price_time).do(self.collect_actual_price)

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

    def collect_actual_price(self) -> None:
        """NVDA の直近実績終値を取得し pending 予測を確定する"""

        try:
            latest = self._fetch_latest_close(self.target_symbol)
            if latest is None:
                logging.warning("実績株価が取得できなかったため更新をスキップします")
                return

            actual_price, actual_date = latest
            logging.info(
                f"{self.target_symbol} の実績終値 {actual_price:.2f} ({actual_date}) を反映します"
            )
            self.update_with_actual_price(self.target_symbol, actual_price)
        except Exception as exc:
            logging.error(f"実績株価の収集中にエラー: {exc}", exc_info=True)

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
        """NVDA の株価履歴から特徴量を生成"""

        lookback_days = int(self.config.get("features.lookback_days", 90))
        logging.debug(f"{ticker} の特徴量を生成します (lookback={lookback_days})")
        history = fetch_stock_data(
            us_symbols=[ticker],
            jp_symbols=[],
            lookback_days=max(lookback_days, 30),
        )
        history = history[history["symbol"] == ticker]

        if history.empty:
            raise ValueError(f"{ticker} の履歴データが取得できませんでした")

        augmented = self.model._augment_with_features(history)
        prepared = augmented.dropna(subset=self.model.feature_columns)
        if prepared.empty:
            raise ValueError("十分な履歴データがなく特徴量を生成できませんでした")
        latest = prepared.iloc[-1]

        feature_dict = {column: float(latest[column]) for column in self.model.feature_columns}
        feature_dict.update(
            {
                "as_of": str(latest["date"]).split("T")[0],
                "volume": float(latest.get("volume", 0.0)),
            }
        )
        logging.debug(f"{ticker} の特徴量: {feature_dict}")
        return feature_dict

    def _fetch_latest_close(self, ticker: str) -> Optional[tuple[float, str]]:
        """直近の終値と日付を取得する"""

        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=10)
        history = fetch_stock_data(
            us_symbols=[ticker],
            jp_symbols=[],
            start_date=start_date,
            end_date=end_date,
            lookback_days=(end_date - start_date).days,
        )
        history = history[history["symbol"] == ticker]
        if history.empty:
            return None

        history = history.sort_values("date")
        latest_row = history.iloc[-1]
        return float(latest_row["close"]), str(latest_row["date"]).split("T")[0]

    # -------------------- エントリーポイント --------------------
    def run(
        self,
        duration_minutes: Optional[float] = None,
        sleep_seconds: float = 60.0,
    ) -> None:
        """スケジューラを実行し、任意で稼働時間を制限"""

        logging.info("予測パイプラインを起動しました")
        max_sleep = max(sleep_seconds, 0.1)
        start_time = time.monotonic()
        try:
            while True:
                schedule.run_pending()

                if duration_minutes is not None:
                    elapsed_minutes = (time.monotonic() - start_time) / 60.0
                    if elapsed_minutes >= duration_minutes:
                        logging.info(
                            "指定された稼働時間 %.2f 分に達したためスケジューラを終了します",
                            duration_minutes,
                        )
                        break

                time.sleep(max_sleep)
        except KeyboardInterrupt:
            logging.info("ユーザー操作によりパイプラインを終了します")
        except Exception as exc:
            logging.error(f"パイプライン実行中にエラー: {exc}", exc_info=True)
            raise

    def run_cycle(
        self,
        *,
        run_prediction: bool = True,
        run_actuals: bool = True,
        run_review: bool = False,
    ) -> None:
        """単発実行で必要な処理のみを順次実行"""

        logging.info(
            "単発サイクル実行を開始します (prediction=%s, actuals=%s, review=%s)",
            run_prediction,
            run_actuals,
            run_review,
        )
        if run_prediction:
            self.predict_all_tickers()
        if run_actuals:
            self.collect_actual_price()
        if run_review:
            self.weekly_model_review()
        logging.info("単発サイクル実行が完了しました")


def main() -> None:
    """スクリプト起動時のメイン処理"""

    parser = argparse.ArgumentParser(
        description="NVDA 予測パイプラインの実行モードを制御します"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="設定ファイルのパス。未指定の場合は config/config.yaml を使用",
    )
    parser.add_argument(

        "--mode",
        choices=["daemon", "cycle"],
        default="daemon",
        help="daemon: 常駐監視 / cycle: 単発処理",
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=None,
        help="daemon モードで稼働させる分数。未指定なら無期限",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=60.0,
        help="スケジューラの待機間隔 (秒)",
    )
    parser.add_argument(
        "--run-prediction",
        action="store_true",
        help="cycle モードで予測を実行",
    )
    parser.add_argument(
        "--run-actuals",
        action="store_true",
        help="cycle モードで実績取り込みを実行",
    )
    parser.add_argument(
        "--run-review",
        action="store_true",
        help="cycle モードで週次レビューを実行",
    )

    args = parser.parse_args()

    config = Config(args.config)

    setup_logging(config._config)
    pipeline = PredictionPipeline(config)

    if args.mode == "daemon":
        pipeline.run(
            duration_minutes=args.duration_minutes,
            sleep_seconds=max(args.sleep_seconds, 0.1),
        )
    else:
        run_prediction = args.run_prediction
        run_actuals = args.run_actuals
        run_review = args.run_review
        if not any([run_prediction, run_actuals, run_review]):
            run_prediction = True
            run_actuals = True

        pipeline.run_cycle(
            run_prediction=run_prediction,
            run_actuals=run_actuals,
            run_review=run_review,
        )


if __name__ == "__main__":
    main()
