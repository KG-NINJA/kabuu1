import sys
from pathlib import Path

# プロジェクトルートをパスに追加（親ディレクトリを指定）
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from reinforcement_learning import test_rl_pipeline, test_rl_improvement

print("=" * 60)
print("強化学習パイプラインのテストを実行します")
print("=" * 60)

print("\n【テスト1】基本動作テスト")
print("-" * 60)
test_rl_pipeline()

print("\n【テスト2】精度改善テスト")
print("-" * 60)
test_rl_improvement()

print("\n" + "=" * 60)
print("✅ すべてのテストが完了しました")
print("=" * 60)
