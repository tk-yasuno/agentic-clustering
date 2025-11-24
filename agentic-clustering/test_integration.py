# 統合テスト: alternative_methods.pyのtry_hdbscan()をテスト
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, r'C:\Users\yasun\LangChain\learning-langchain\agentic-clustering')

from alternative_methods import AlternativeClusteringMethods

print("=" * 60)
print("AlternativeClusteringMethods - HDBSCAN統合テスト")
print("=" * 60)

# テストデータ生成
np.random.seed(42)
n_samples = 1000
n_features = 13

data = []
for i in range(5):
    center = np.random.randn(n_features) * 3
    cluster_data = np.random.randn(n_samples // 5, n_features) + center
    data.append(cluster_data)

X = np.vstack(data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"データサイズ: {X_scaled.shape}")

# AlternativeClusteringMethodsのインスタンス化
alt_methods = AlternativeClusteringMethods(X_scaled)

# HDBSCANを実行
print("\nHDBSCANメソッドをテスト中...")
labels = alt_methods.try_hdbscan(target_clusters=10)

if labels is not None:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"\n結果:")
    print(f"  クラスタ数: {n_clusters}")
    print(f"  ノイズポイント: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    
    if 'HDBSCAN' in alt_methods.results:
        result = alt_methods.results['HDBSCAN']
        print(f"  最適パラメータ: {result['params']}")
        print(f"  スコア: {result['score']:.4f}")
        print("\n✅ HDBSCAN統合テスト成功!")
    else:
        print("\n⚠️ 結果がresultsに保存されていません")
else:
    print("\n❌ HDBSCANの実行に失敗しました")
