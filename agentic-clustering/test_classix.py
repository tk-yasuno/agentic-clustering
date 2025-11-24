# CLASSIXクラスタリングのテストスクリプト
import numpy as np
from sklearn.preprocessing import StandardScaler
from classix import CLASSIX
from sklearn.metrics import silhouette_score

print("=" * 60)
print("CLASSIX クラスタリングのテスト")
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

# CLASSIXクラスタリング実行
print("\nCLASSIXを実行中...")

radius_values = [0.5, 1.0, 1.5]
minPts_values = [10, 20]

for radius in radius_values:
    for minPts in minPts_values:
        try:
            classix = CLASSIX(radius=radius, minPts=minPts, verbose=0)
            classix.fit(X_scaled)
            labels = classix.labels_
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters >= 2:
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    score = silhouette_score(X_scaled[mask], labels[mask])
                    print(f"  radius={radius}, minPts={minPts}: "
                          f"クラスタ数={n_clusters}, ノイズ={n_noise}, スコア={score:.4f}")
        except Exception as e:
            print(f"  radius={radius}, minPts={minPts}: エラー - {e}")

print("\n✅ CLASSIXテスト完了")
