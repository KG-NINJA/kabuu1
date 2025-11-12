"""NVDA 専用の強化学習オーケストレーター."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

from .reinforcement_learning import ReinforcementLearningPipeline

TARGET_SYMBOL = "NVDA"


@dataclass
class NvdaReinforcementHub:
    """NVDA の予測改善を担当する補助クラス."""

    base_dir: Path = Path("nvda_learning")
    reward_threshold: float = 0.0
    _pipeline: ReinforcementLearningPipeline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.validation_dir = self.base_dir / "validation"
        self.prediction_dir = self.base_dir / "predictions"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = ReinforcementLearningPipeline(
            validation_dir=str(self.validation_dir),
            prediction_dir=str(self.prediction_dir),
            reward_threshold=self.reward_threshold,
        )

    def record_outcome(
        self,
        *,
        predicted_price: float,
        actual_price: float,
        model_params: Dict[str, float],
        ticker: str = TARGET_SYMBOL,
    ) -> float:
        """NVDA の予測結果を記録し報酬を返す."""

        return self._pipeline.record_prediction_result(
            ticker=ticker,
            predicted_price=predicted_price,
            actual_price=actual_price,
            model_params=model_params,
        )

    def should_improve(self) -> Tuple[bool, Dict[str, float]]:
        """改善が必要かどうかを RL パイプラインに委譲."""

        return self._pipeline.should_improve_model()

    def get_learning_insights(self) -> Dict[str, float]:
        """学習メトリクスを返す."""

        return self._pipeline.get_learning_insights()

    @property
    def pipeline(self) -> ReinforcementLearningPipeline:
        """内部の RL パイプラインを参照用に返す."""

        return self._pipeline
