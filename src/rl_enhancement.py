"""Advanced reinforcement learning logging and review helpers."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class RLEntry:
    """Structured representation of a single RL log entry."""

    timestamp: datetime
    ticker: str
    predicted_price: float
    actual_price: float
    error_pct: float
    reward: float
    confidence: float
    model_version: str
    lookback_period: int
    learning_rate: float
    data_quality: str
    metadata: str

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> Optional["RLEntry"]:
        """Convert CSV dict row to :class:`RLEntry`. Return None if parsing fails."""

        try:
            timestamp = datetime.fromisoformat(row.get("timestamp", ""))
        except ValueError:
            return None

        def _to_float(value: str, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _to_int(value: str, default: int = 0) -> int:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

        return cls(
            timestamp=timestamp,
            ticker=row.get("ticker", "UNKNOWN"),
            predicted_price=_to_float(row.get("predicted_price")),
            actual_price=_to_float(row.get("actual_price")),
            error_pct=_to_float(row.get("error_pct")),
            reward=_to_float(row.get("reward"), 0.5),
            confidence=_to_float(row.get("confidence"), 0.5),
            model_version=row.get("model_version", "v1.0"),
            lookback_period=_to_int(row.get("lookback_period"), 365),
            learning_rate=_to_float(row.get("learning_rate"), 0.001),
            data_quality=row.get("data_quality", "unknown"),
            metadata=row.get("metadata", ""),
        )


class AdvancedReinforcementLogger:
    """Enhanced RL logging utilities handling CSV + JSON summaries."""

    csv_columns = [
        "timestamp",
        "ticker",
        "predicted_price",
        "actual_price",
        "error_pct",
        "reward",
        "confidence",
        "model_version",
        "lookback_period",
        "learning_rate",
        "data_quality",
        "metadata",
    ]

    def __init__(
        self,
        log_path: Path | str = Path("reinforcement_learning.log"),
        history_dir: Path | str = Path("data/rl_history"),
        daily_dir: Optional[Path | str] = Path("data/daily_summaries"),
    ) -> None:
        self.log_path = Path(log_path)
        self.history_dir = Path(history_dir)
        self.daily_dir = Path(daily_dir) if daily_dir is not None else None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        if self.daily_dir is not None:
            self.daily_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_log_file()

    # ------------------------------------------------------------------
    def _ensure_log_file(self) -> None:
        if not self.log_path.exists():
            LOGGER.info("Creating RL log file with extended schema: %s", self.log_path)
            with self.log_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.csv_columns)
                writer.writeheader()

    # ------------------------------------------------------------------
    def record_detailed_outcome(
        self,
        *,
        ticker: str,
        predicted_price: float,
        actual_price: float,
        confidence: float,
        model_version: str,
        lookback_period: int,
        learning_rate: float,
        data_quality: str = "good",
        reward: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append a detailed RL outcome to the CSV log."""

        timestamp = datetime.now(UTC)
        error_pct = 0.0
        if actual_price != 0:
            error_pct = (predicted_price - actual_price) / actual_price

        if reward is None:
            reward = self._compute_reward(predicted_price, actual_price)

        metadata_str = json.dumps(metadata or {}, ensure_ascii=False)
        row = {
            "timestamp": timestamp.isoformat(),
            "ticker": ticker,
            "predicted_price": f"{predicted_price:.6f}",
            "actual_price": f"{actual_price:.6f}",
            "error_pct": f"{error_pct:.6f}",
            "reward": f"{float(reward):.6f}",
            "confidence": f"{confidence:.6f}",
            "model_version": model_version,
            "lookback_period": str(int(lookback_period)),
            "learning_rate": f"{learning_rate:.6f}",
            "data_quality": data_quality,
            "metadata": metadata_str,
        }

        with self.log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_columns)
            writer.writerow(row)

        LOGGER.debug("Recorded RL outcome for %s", ticker)
        return row

    # ------------------------------------------------------------------
    def export_daily_summary(self, target_date: Optional[date] = None) -> Path:
        """Aggregate log entries for a day and store JSON summary."""

        target_date = target_date or datetime.now(UTC).date()
        entries = [entry for entry in self._iter_entries() if entry.timestamp.date() == target_date]
        summary = self._build_summary(entries, label=str(target_date))

        filename = f"daily_summary_{target_date.isoformat()}.json"
        history_path = self.history_dir / filename
        self._write_json(history_path, summary)

        if self.daily_dir is not None:
            daily_path = self.daily_dir / filename
            self._write_json(daily_path, summary)

        LOGGER.info("Saved daily summary: %s", history_path)
        return history_path

    # ------------------------------------------------------------------
    def load_recent_entries(self, days: int = 7) -> List[RLEntry]:
        """Return entries newer than now - days."""

        cutoff = datetime.now(UTC) - timedelta(days=days)
        return [entry for entry in self._iter_entries() if entry.timestamp >= cutoff]

    # ------------------------------------------------------------------
    def _iter_entries(self) -> Iterable[RLEntry]:
        if not self.log_path.exists():
            return []

        entries: List[RLEntry] = []
        with self.log_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = RLEntry.from_row(row)
                if parsed is not None:
                    entries.append(parsed)
        return entries

    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary(entries: List[RLEntry], label: str) -> Dict[str, Any]:
        def safe_mean(values: List[float]) -> float:
            return round(mean(values), 6) if values else 0.0

        rewards = [entry.reward for entry in entries]
        errors = [abs(entry.error_pct) for entry in entries]
        success_threshold = 0.95
        successes = [entry for entry in entries if entry.reward >= success_threshold]

        by_key = {}
        for entry in entries:
            ticker_stats = by_key.setdefault(entry.ticker, {"count": 0, "avg_reward": 0.0, "avg_error_pct": 0.0})
            ticker_stats.setdefault("_rewards", []).append(entry.reward)
            ticker_stats.setdefault("_errors", []).append(abs(entry.error_pct))
            ticker_stats["count"] += 1

        for stats in by_key.values():
            stats["avg_reward"] = safe_mean(stats.pop("_rewards", []))
            stats["avg_error_pct"] = safe_mean(stats.pop("_errors", []))

        return {
            "label": label,
            "generated_at": datetime.now(UTC).isoformat(),
            "total_predictions": len(entries),
            "average_reward": safe_mean(rewards),
            "average_error_pct": safe_mean(errors),
            "best_reward": max(rewards) if rewards else 0.0,
            "worst_reward": min(rewards) if rewards else 0.0,
            "success_rate": round((len(successes) / len(entries)) * 100, 2) if entries else 0.0,
            "by_ticker": by_key,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_reward(predicted_price: float, actual_price: float) -> float:
        if actual_price == 0:
            return 0.5
        error = abs(predicted_price - actual_price) / actual_price
        return max(0.0, min(1.0, 1.0 - error))


class WeeklyReviewGenerator:
    """Generate weekly reinforcement learning reviews."""

    def __init__(
        self,
        logger: AdvancedReinforcementLogger,
        weekly_dir: Path | str = Path("data/weekly_reviews"),
    ) -> None:
        self.logger = logger
        self.weekly_dir = Path(weekly_dir)
        self.weekly_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def generate_weekly_review(self, days: int = 7) -> Dict[str, Any]:
        entries = self.logger.load_recent_entries(days=days)
        summary = AdvancedReinforcementLogger._build_summary(entries, label=f"last_{days}_days")
        by_model = self._aggregate(entries, key=lambda entry: entry.model_version)
        by_ticker = self._aggregate(entries, key=lambda entry: entry.ticker)
        recommendations = self._build_recommendations(summary, by_ticker, by_model)

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": summary,
            "by_ticker": by_ticker,
            "by_model": by_model,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    def generate_and_save(self, days: int = 7) -> Path:
        review = self.generate_weekly_review(days=days)
        today = date.today()
        iso_year, iso_week, _ = today.isocalendar()
        filename = f"weekly_review_{iso_year}-W{iso_week:02d}.json"
        path = self.weekly_dir / filename
        AdvancedReinforcementLogger._write_json(path, review)
        LOGGER.info("Saved weekly review: %s", path)
        return path

    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate(entries: List[RLEntry], key) -> Dict[str, Any]:
        buckets: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            bucket_key = key(entry)
            bucket = buckets.setdefault(bucket_key, {
                "count": 0,
                "avg_reward": 0.0,
                "avg_error_pct": 0.0,
                "best_reward": 0.0,
                "worst_reward": 0.0,
            })
            rewards = bucket.setdefault("_rewards", [])
            errors = bucket.setdefault("_errors", [])
            rewards.append(entry.reward)
            errors.append(abs(entry.error_pct))
            bucket["count"] += 1

        for bucket in buckets.values():
            rewards = bucket.pop("_rewards", [])
            errors = bucket.pop("_errors", [])
            bucket["avg_reward"] = round(mean(rewards), 6) if rewards else 0.0
            bucket["avg_error_pct"] = round(mean(errors), 6) if errors else 0.0
            bucket["best_reward"] = max(rewards) if rewards else 0.0
            bucket["worst_reward"] = min(rewards) if rewards else 0.0

        return buckets

    # ------------------------------------------------------------------
    @staticmethod
    def _build_recommendations(
        summary: Dict[str, Any],
        by_ticker: Dict[str, Any],
        by_model: Dict[str, Any],
    ) -> List[str]:
        recommendations: List[str] = []

        if summary.get("average_reward", 0) < 0.95:
            recommendations.append(
                "平均報酬が 0.95 未満のため、学習データ期間の拡張や特徴量の再検討を推奨します"
            )

        for ticker, stats in by_ticker.items():
            if stats.get("avg_reward", 1.0) < 0.9:
                recommendations.append(
                    f"{ticker} の平均報酬が低下しています。lookback_period を {max(90, stats.get('count', 0) * 5)} 日まで延長して再学習してください"
                )

        for model_version, stats in by_model.items():
            if stats.get("avg_error_pct", 0) > 0.05:
                recommendations.append(
                    f"モデル {model_version} の誤差率が高いです。学習率を微調整し、最新データで再トレーニングを検討してください"
                )

        if not recommendations:
            recommendations.append("大きな問題は検出されませんでした。現在の戦略を維持してください。")

        return recommendations


def integrate_with_pipeline(pipeline: Any) -> Any:
    """Attach RL logging utilities to an existing pipeline instance."""

    if getattr(pipeline, "rl_logger", None) is None:
        pipeline.rl_logger = AdvancedReinforcementLogger(Path("reinforcement_learning.log"))
        LOGGER.info("Advanced RL logger integrated into pipeline")

    if getattr(pipeline, "weekly_review_generator", None) is None:
        pipeline.weekly_review_generator = WeeklyReviewGenerator(pipeline.rl_logger)
        LOGGER.info("Weekly review generator integrated into pipeline")

    return pipeline


def _demo() -> None:
    LOGGER.info("Running RL enhancement demo")
    logger = AdvancedReinforcementLogger(Path("reinforcement_learning.log"))
    record = logger.record_detailed_outcome(
        ticker="NVDA",
        predicted_price=500.0,
        actual_price=495.5,
        confidence=0.92,
        model_version="demo",
        lookback_period=365,
        learning_rate=0.001,
        metadata={"source": "demo", "note": "dry run"},
    )
    LOGGER.info("Recorded demo entry: %s", record)

    summary_path = logger.export_daily_summary()
    LOGGER.info("Daily summary saved to %s", summary_path)

    review_gen = WeeklyReviewGenerator(logger)
    review_path = review_gen.generate_and_save()
    LOGGER.info("Weekly review saved to %s", review_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _demo()
