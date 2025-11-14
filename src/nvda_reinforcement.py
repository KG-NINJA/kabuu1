"""
毎日強くなるシステムの実装例
報酬に基づいてハイパーパラメータを動的に調整
"""

from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Tuple

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
            "latest_reward": self.rewards_history[-1],
            "average_reward_all": sum(self.rewards_history) / len(self.rewards_history),
            "average_reward_7day": sum(self.rewards_history[-7:]) / min(7, len(self.rewards_history)),
            "best_reward": max(self.rewards_history),
            "worst_reward": min(self.rewards_history),
            "trend": "improving" if (self.rewards_history[-1] > self.rewards_history[-2] if len(self.rewards_history) > 1 else False) else "stable/declining",
        }
        
        return report

# 使用例
if __name__ == "__main__":
    learner = AdaptiveNVDALearner()
    
    # 学習レポート
    report = learner.get_learning_report()
    print("=== 学習進捗レポート ===")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # 改善が必要か判定
    should_improve, strategy = learner.should_improve_model()
    print(f"\n=== 改善判定 ===")
    print(f"改善が必要: {should_improve}")
    print(f"戦略: {json.dumps(strategy, indent=2, ensure_ascii=False)}")
