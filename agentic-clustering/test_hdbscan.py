# HDBSCANクラスタリングのテストスクリプト
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.metrics import silhouette_score

print("=" * 60)
print("HDBSCAN クラスタリングのテスト")
print("=" * 60)

# テストデータ生成（13次元、4292サンプル）
np.random.seed(42)
n_samples = 4292
n_features = 13

# 複数のクラスタを含むデータを生成
data = []
for i in range(5):
    center = np.random.randn(n_features) * 3
    cluster_data = np.random.randn(n_samples // 5, n_features) + center
    data.append(cluster_data)

X = np.vstack(data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"データサイズ: {X_scaled.shape}")

# HDBSCANクラスタリング実行
print("\nHDBSCANを実行中...")

min_cluster_size_values = [50, 100, 150]
min_samples_values = [10, 20]

for min_cluster_size in min_cluster_size_values:
    for min_samples in min_samples_values:
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method='eom',
                metric='euclidean'
            )
            
            labels = clusterer.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters >= 2:
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    score = silhouette_score(X_scaled[mask], labels[mask])
                    print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples}: "
                          f"クラスタ数={n_clusters}, ノイズ={n_noise} ({n_noise/len(labels)*100:.1f}%), "
                          f"スコア={score:.4f}")
                else:
                    print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples}: "
                          f"クラスタ数={n_clusters}, ノイズ={n_noise} - スコア計算不可")
            else:
                print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples}: "
                      f"クラスタ数={n_clusters} (不十分)")
        except Exception as e:
            print(f"  min_cluster_size={min_cluster_size}, min_samples={min_samples}: エラー - {e}")

print("\n✅ HDBSCANテスト完了")
