"""
毎日強くなるシステムの実装
報酬に基づいてハイパーパラメータを動的に調整
"""

from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# グローバル定数
TARGET_SYMBOL = "NVDA"


class AdaptiveNVDALearner:
    """報酬ベースの適応的学習システム"""
    
    def __init__(self, log_path: Path = Path("reinforcement_learning.log")):
        self.log_path = log_path
        self.rewards_history = []
        self.params_history = []
        self.load_history()
    
    def load_history(self) -> None:
        """報酬履歴を読み込む"""
        if not self.log_path.exists():
            return
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # ヘッダーをスキップ
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        try:
                            reward = float(parts[3])
                            self.rewards_history.append(reward)
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning(f"報酬履歴の読み込みに失敗しました: {e}")
    
    def get_average_reward(self, window: int = 5) -> float:
        """直近N日の平均報酬を計算"""
        if not self.rewards_history:
            return 0.5  # デフォルト
        
        recent = self.rewards_history[-window:]
        return sum(recent) / len(recent)
    
    def should_improve_model(self) -> Tuple[bool, Dict[str, Any]]:
        """モデル改善が必要か判定し、戦略を提案"""
        
        if len(self.rewards_history) < 3:
            return False, {}  # データが少なすぎる
        
        avg_reward = self.get_average_reward(window=5)
        recent_trend = self.rewards_history[-1] - self.rewards_history[-2] if len(self.rewards_history) > 1 else 0
        
        strategy = {
            "timestamp": datetime.now().isoformat(),
            "average_reward": avg_reward,
            "recent_trend": recent_trend,
            "adjustments": {}
        }
        
        # 高性能：現在の戦略を保持
        if avg_reward > 0.99:
            strategy["adjustments"] = {
                "action": "maintain",
                "reason": f"高性能を維持（平均報酬: {avg_reward:.4f}）"
            }
            return False, strategy
        
        # 改善傾向：学習率を少し上げて積極的に学習
        elif avg_reward > 0.95 and recent_trend > 0:
            strategy["adjustments"] = {
                "action": "increase_learning",
                "learning_rate": 0.002,  # 0.001 → 0.002
                "reason": f"改善傾向が見られます（トレンド: {recent_trend:.4f}）"
            }
            return True, strategy
        
        # 低性能：データ期間を拡大して再学習
        elif avg_reward < 0.95:
            strategy["adjustments"] = {
                "action": "retrain_with_more_data",
                "lookback_period": 180,  # 90 → 180 日
                "learning_rate": 0.001,
                "reason": f"性能低下を検出（平均報酬: {avg_reward:.4f}）"
            }
            return True, strategy
        
        # デフォルト：様子を見る
        return False, strategy
    
    def get_learning_report(self) -> Dict[str, Any]:
        """学習進捗レポートを生成"""
        
        if not self.rewards_history:
            return {"status": "No data"}
        
        report = {
            "total_predictions": len(self.rewards_history),
            "latest_reward": round(self.rewards_history[-1], 6),
            "average_reward_all": round(sum(self.rewards_history) / len(self.rewards_history), 6),
            "average_reward_7day": round(sum(self.rewards_history[-7:]) / min(7, len(self.rewards_history)), 6),
            "best_reward": round(max(self.rewards_history), 6),
            "worst_reward": round(min(self.rewards_history), 6),
            "trend": "improving" if (self.rewards_history[-1] > self.rewards_history[-2] if len(self.rewards_history) > 1 else False) else "stable/declining",
        }
        
        return report


class NvdaReinforcementHub:
    """強化学習ハブ - 予測結果を記録し、改善戦略を提案"""
    
    def __init__(self, base_dir: Optional[str] = None, reward_threshold: float = 0.0):
        """
        初期化
        
        Args:
            base_dir: ベースディレクトリ（デフォルト: nvda_learning）
            reward_threshold: 報酬閾値
        """
        self.base_dir = Path(base_dir or "nvda_learning")
        self.validation_dir = self.base_dir / "validation"
        self.prediction_dir = self.base_dir / "predictions"
        self.reward_threshold = reward_threshold
        self.log_path = Path("reinforcement_learning.log")
        
        # ディレクトリを作成
        for d in [self.validation_dir, self.prediction_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # ログファイルが存在しなければ作成
        if not self.log_path.exists():
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write("ticker,predicted,actual,reward\n")
        
        self.learner = AdaptiveNVDALearner(self.log_path)
        logger.info(f"NvdaReinforcementHub を初期化しました (base_dir: {self.base_dir})")
    
    def record_outcome(
        self, 
        ticker: str, 
        predicted_price: float, 
        actual_price: float, 
        model_params: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        予測結果を記録し、報酬を計算
        
        Args:
            ticker: 銘柄（例: NVDA）
            predicted_price: 予測価格
            actual_price: 実際の価格
            model_params: モデルパラメータ（オプション）
        
        Returns:
            計算された報酬スコア（0.0-1.0）
        """
        try:
            # 報酬を計算（誤差が小さいほど報酬が大きい）
            if actual_price == 0:
                reward = 0.5
            else:
                error = abs(predicted_price - actual_price) / actual_price
                reward = max(0.0, min(1.0, 1.0 - error))  # 0.0-1.0 の範囲に正規化
            
            # ログに記録
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"{ticker},{predicted_price:.4f},{actual_price:.4f},{reward:.6f}\n")
            
            logger.info(
                f"結果を記録しました: {ticker} - 予測: ${predicted_price:.2f}, "
                f"実績: ${actual_price:.2f}, 報酬: {reward:.6f}"
            )
            
            return reward
        
        except Exception as e:
            logger.error(f"結果の記録に失敗しました: {e}", exc_info=True)
            return 0.5
    
    def should_improve(self) -> Tuple[bool, Dict[str, Any]]:
        """
        モデル改善が必要か判定
        
        Returns:
            (改善が必要か, 改善戦略)
        """
        try:
            return self.learner.should_improve_model()
        except Exception as e:
            logger.error(f"改善判定に失敗しました: {e}", exc_info=True)
            return False, {}
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        学習インサイトを取得
        
        Returns:
            学習進捗レポート
        """
        try:
            return self.learner.get_learning_report()
        except Exception as e:
            logger.error(f"学習インサイトの取得に失敗しました: {e}", exc_info=True)
            return {"status": "Error"}


# 使用例
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # AdaptiveNVDALearner のテスト
    print("=" * 60)
    print("AdaptiveNVDALearner テスト")
    print("=" * 60)
    learner = AdaptiveNVDALearner()
    
    # 学習レポート
    report = learner.get_learning_report()
    print("\n📊 学習進捗レポート:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # 改善が必要か判定
    should_improve, strategy = learner.should_improve_model()
    print(f"\n🔧 改善が必要: {should_improve}")
    print(f"💡 推奨戦略: {json.dumps(strategy, indent=2, ensure_ascii=False)}")
    
    # NvdaReinforcementHub のテスト
    print("\n" + "=" * 60)
    print("NvdaReinforcementHub テスト")
    print("=" * 60)
    hub = NvdaReinforcementHub()
    
    # 結果を記録
    print("\n📝 結果を記録:")
    reward = hub.record_outcome(
        ticker=TARGET_SYMBOL,
        predicted_price=186.93,
        actual_price=186.86,
        model_params={"learning_rate": 0.001}
    )
    print(f"  報酬: {reward:.6f}")
    
    # 改善判定
    print("\n🔧 改善判定:")
    should_improve, strategy = hub.should_improve()
    print(f"  改善が必要: {should_improve}")
    
    # 学習インサイト
    print("\n📈 学習インサイト:")
    insights = hub.get_learning_insights()
    for key, value in insights.items():
        print(f"  {key}: {value}")
