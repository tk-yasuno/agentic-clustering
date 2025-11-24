# -*- coding: utf-8 -*-
"""
Agenticワークフローの段階的テスト
"""

import pandas as pd
import numpy as np
import config
from data_preprocessing import preprocess_all_data

print("=" * 60)
print("📊 ステップ1: データ前処理")
print("=" * 60)

# データ前処理
df = preprocess_all_data()

if df is not None:
    print(f"\n✅ 前処理完了: {len(df)}件")
    print(f"📋 特徴量: {config.FEATURE_COLUMNS}")
    print(f"\n🔍 欠損値チェック:")
    missing = df[config.FEATURE_COLUMNS].isnull().sum()
    print(missing)
    
    print(f"\n📈 基本統計:")
    print(df[config.FEATURE_COLUMNS].describe())
    
    # 次のステップに進むか確認
    print("\n" + "=" * 60)
    print("📊 ステップ2: クラスタリング準備")
    print("=" * 60)
    
    from sklearn.preprocessing import StandardScaler
    
    X = df[config.FEATURE_COLUMNS].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ スケーリング完了: {X_scaled.shape}")
    print(f"   特徴量数: {X_scaled.shape[1]}")
    print(f"   データ数: {X_scaled.shape[0]}")
    
    # KMeansクラスタリングのテスト
    print("\n" + "=" * 60)
    print("📊 ステップ3: KMeansクラスタリング（Round 1）")
    print("=" * 60)
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # 最適なクラスタ数を探索
    best_k = None
    best_score = -1
    scores = []
    
    print("🔍 最適クラスタ数を探索中...")
    for k in range(config.MIN_CLUSTERS, config.MAX_CLUSTERS + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append((k, score))
        print(f"   k={k}: シルエットスコア={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"\n✅ 最適クラスタ数: k={best_k} (シルエットスコア={best_score:.4f})")
    
    # 最適なクラスタ数でクラスタリング
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"\n📊 クラスタ分布:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"   クラスタ{cluster_id}: {count}件 ({count/len(labels)*100:.1f}%)")
    
    # 評価
    print("\n" + "=" * 60)
    print("📊 ステップ4: クラスタリング品質評価")
    print("=" * 60)
    
    from cluster_evaluator import ClusterEvaluator
    
    evaluator = ClusterEvaluator(X_scaled, labels)
    evaluation = evaluator.evaluate_all()
    
    print(f"\n📈 評価結果:")
    print(f"   シルエットスコア: {evaluation['silhouette']:.4f}")
    print(f"   Davies-Bouldin指数: {evaluation['davies_bouldin']:.4f}")
    print(f"   Calinski-Harabasz指数: {evaluation['calinski_harabasz']:.2f}")
    print(f"   クラスタバランス: {evaluation['balance']:.4f}")
    print(f"   📊 総合スコア: {evaluation['overall']:.2f}/100")
    
    needs_improvement = evaluator.needs_improvement(evaluation['overall'])
    if needs_improvement:
        print(f"\n⚠️ 総合スコアが閾値60未満のため、代替手法を試行する必要があります")
    else:
        print(f"\n✅ 総合スコアが閾値60以上のため、KMeansを採用します")
    
    # PCA次元削減のテスト
    print("\n" + "=" * 60)
    print("📊 ステップ5: PCA次元削減（Round 1）")
    print("=" * 60)
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(X_scaled)
    
    print(f"✅ PCA完了: {embedding.shape}")
    print(f"   説明された分散: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    # オーバーラップ評価
    from cluster_evaluator import DimensionalityReductionEvaluator
    
    dim_evaluator = DimensionalityReductionEvaluator(embedding, labels)
    overlap_result = dim_evaluator.evaluate_overlap()
    
    print(f"\n📈 オーバーラップ評価:")
    print(f"   平均クラスタ中心間距離: {overlap_result['mean_center_distance']:.4f}")
    print(f"   平均クラスタ内分散: {overlap_result['mean_variance']:.4f}")
    print(f"   📊 オーバーラップスコア: {overlap_result['overlap']:.4f}")
    
    needs_alternative_dim = overlap_result['overlap'] >= 0.5
    if needs_alternative_dim:
        print(f"\n⚠️ オーバーラップスコアが閾値0.5以上のため、代替手法を試行する必要があります")
    else:
        print(f"\n✅ オーバーラップスコアが閾値0.5未満のため、PCAを採用します")
    
    print("\n" + "=" * 60)
    print("✅ 全ステップ完了")
    print("=" * 60)

else:
    print("❌ データ前処理に失敗しました")
