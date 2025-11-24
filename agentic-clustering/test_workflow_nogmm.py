# GMM無効化テスト
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from agentic_workflow import AgenticClusteringWorkflow

print("=" * 60)
print("GMM無効化テスト")
print("=" * 60)

# テストデータ生成
np.random.seed(42)
n_samples = 500
n_features = 13

data = []
for i in range(5):
    center = np.random.randn(n_features) * 3
    cluster_data = np.random.randn(n_samples // 5, n_features) + center
    data.append(cluster_data)

X = np.vstack(data)

# DataFrameに変換
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)

print(f"データサイズ: {df.shape}")
print()

# Agenticワークフロー実行
workflow = AgenticClusteringWorkflow(df, feature_names)
workflow.run(quality_threshold=60.0, overlap_threshold=0.5)

print("\n" + "=" * 60)
print("実行された手法:")
print("=" * 60)
for method in workflow.evaluation_results.keys():
    result = workflow.evaluation_results[method]
    score = result.get('total_score', result.get('composite_score', 0))
    print(f"  ✓ {method}: 総合スコア = {score:.2f}")

print("\n" + "=" * 60)
if 'GMM' in workflow.evaluation_results:
    print("❌ GMM が実行されました (想定外)")
else:
    print("✅ GMM は実行されませんでした (正常)")
print("=" * 60)
