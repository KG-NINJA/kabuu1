"""軽量な強化学習パイプラインのスタブ実装。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


@dataclass
class PredictionRecord:
    """予測結果を保持するデータクラス。"""

    ticker: str
    predicted_price: float
    actual_price: float
    reward: float
    timestamp: str


@dataclass
class ReinforcementLearningPipeline:
    """予測パイプラインと連携するスタブ用クラス。"""

    validation_dir: str
    prediction_dir: str
    reward_threshold: float = 0.0
    history: List[PredictionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        """出力ディレクトリを作成しログファイルを初期化する。"""
        self.validation_path = Path(self.validation_dir)
        self.prediction_path = Path(self.prediction_dir)
        self.validation_path.mkdir(parents=True, exist_ok=True)
        self.prediction_path.mkdir(parents=True, exist_ok=True)
        self.log_path = self.prediction_path / "reinforcement_learning.log"
        if not self.log_path.exists():
            self.log_path.write_text(
                "ticker,predicted,actual,reward\n", encoding="utf-8"
            )

    def record_prediction_result(
        self,
        ticker: str,
        predicted_price: float,
        actual_price: float,
        model_params: Dict[str, float],
    ) -> float:
        """予測結果を記録し単純な報酬を返す。"""
        del model_params
        if actual_price == 0:
            reward = 0.0
        else:
            error_ratio = abs(predicted_price - actual_price) / actual_price
            reward = max(1.0 - error_ratio, -1.0)
        record = PredictionRecord(
            ticker=ticker,
            predicted_price=predicted_price,
            actual_price=actual_price,
            reward=reward,
            timestamp="",
        )
        self.history.append(record)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{ticker},{predicted_price:.4f},{actual_price:.4f},{reward:.6f}\n"
            )
        return reward

    def should_improve_model(self) -> Tuple[bool, Dict[str, float]]:
        """履歴を基に再学習が必要かを判定する。"""
        if len(self.history) < 5:
            return False, {}
        recent_rewards = [record.reward for record in self.history[-5:]]
        avg_reward = mean(recent_rewards)
        if avg_reward < self.reward_threshold:
            strategy = {
                "learning_rate": 0.0005,
                "lookback_period": 180,
            }
            return True, strategy
        return False, {}

    def get_learning_insights(self) -> Dict[str, float]:
        """報酬履歴から簡易的なメトリクスを計算する。"""
        if not self.history:
            return {"count": 0, "average_reward": 0.0}
        rewards = [record.reward for record in self.history]
        return {
            "count": float(len(rewards)),
            "average_reward": float(mean(rewards)),
            "best_reward": float(max(rewards)),
            "worst_reward": float(min(rewards)),
        }
