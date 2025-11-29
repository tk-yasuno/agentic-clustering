# -*- coding: utf-8 -*-
"""
代替クラスタリング手法モジュール: Agentic Clustering v0.2
GMM, DBSCANなどの代替アルゴリズムを提供
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import config

class AlternativeClusteringMethods:
    """代替クラスタリング手法を提供するクラス"""
    
    def __init__(self, X_scaled):
        """
        Parameters:
        -----------
        X_scaled : array-like
            標準化された特徴量
        """
        self.X_scaled = X_scaled
        self.results = {}
    
    def try_kmeans(self, n_clusters):
        """KMeansクラスタリング"""
        print(f"\n🔵 KMeans (k={n_clusters}) を実行中...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=config.RANDOM_STATE,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(self.X_scaled)
        
        # クラスタ分布を表示
        unique, counts = np.unique(labels, return_counts=True)
        print(f"   クラスタ数: {len(unique)}")
        for cluster_id, count in zip(unique, counts):
            print(f"   クラスタ {cluster_id}: {count}件")
        
        self.results['KMeans'] = {
            'model': kmeans,
            'labels': labels,
            'n_clusters': n_clusters
        }
        
        return labels
    
    def try_gmm(self, n_components_range=None):
        """ガウス混合モデル（GMM）"""
        print(f"\n🟣 GMM (Gaussian Mixture Model) を実行中...")
        
        if n_components_range is None:
            n_components_range = range(config.MIN_CLUSTERS, config.MAX_CLUSTERS + 1)
        
        best_gmm = None
        best_labels = None
        best_score = -1
        best_n = config.MIN_CLUSTERS
        
        for n in n_components_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='full',
                random_state=config.RANDOM_STATE,
                n_init=10
            )
            
            labels = gmm.fit_predict(self.X_scaled)
            
            # ノイズクラスタ（-1）がある場合は除外して評価
            if len(np.unique(labels)) > 1:
                score = silhouette_score(self.X_scaled, labels)
                print(f"   n_components={n}: シルエットスコア = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_gmm = gmm
                    best_labels = labels
                    best_n = n
        
        print(f"   ✓ 最適コンポーネント数: {best_n} (スコア: {best_score:.4f})")
        
        # クラスタ分布を表示
        unique, counts = np.unique(best_labels, return_counts=True)
        print(f"   クラスタ数: {len(unique)}")
        for cluster_id, count in zip(unique, counts):
            print(f"   クラスタ {cluster_id}: {count}件")
        
        self.results['GMM'] = {
            'model': best_gmm,
            'labels': best_labels,
            'n_clusters': best_n,
            'score': best_score
        }
        
        return best_labels
    
    def try_dbscan(self, eps_range=None, min_samples_range=None, target_clusters=50):
        """DBSCAN(密度ベースクラスタリング)"""
        print(f"\n🟢 DBSCAN (Density-Based Spatial Clustering) を実行中...")
        print(f"   目標クラスタ数: {target_clusters}程度")
        
        # デフォルトパラメータ範囲（クラスタ数50程度に調整）
        if eps_range is None:
            eps_range = [0.8, 1.0, 1.2, 1.4, 1.6]
        
        if min_samples_range is None:
            min_samples_range = [15, 20, 25, 30, 35]
        
        best_dbscan = None
        best_labels = None
        best_score = -1
        best_params = None
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.X_scaled)
                
                # ノイズポイント（-1）を除いたクラスタ数
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # クラスタが2つ以上ある場合のみ評価
                if n_clusters >= 2:
                    # ノイズポイントを除外してシルエットスコアを計算
                    mask = labels != -1
                    if mask.sum() > 0:
                        score = silhouette_score(self.X_scaled[mask], labels[mask])
                        
                        # 目標クラスタ数からの距離を計算（ペナルティ）
                        cluster_penalty = abs(n_clusters - target_clusters) / target_clusters
                        adjusted_score = score * (1 - cluster_penalty * 0.5)  # クラスタ数の差に応じてスコアを調整
                        
                        print(f"   eps={eps}, min_samples={min_samples}: "
                              f"クラスタ数={n_clusters}, ノイズ={n_noise}, "
                              f"スコア={score:.4f}, 調整後={adjusted_score:.4f}")
                        
                        # ノイズが少なく、調整後スコアが高く、クラスタ数が目標に近いものを選択
                        if (adjusted_score > best_score and 
                            n_noise < len(labels) * 0.35 and 
                            20 <= n_clusters <= 100):  # クラスタ数の妥当な範囲（60を中心に）
                            best_score = adjusted_score
                            best_dbscan = dbscan
                            best_labels = labels
                            best_params = {'eps': eps, 'min_samples': min_samples}
        
        if best_labels is not None:
            n_clusters_final = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise_final = list(best_labels).count(-1)
            
            print(f"   ✓ 最適パラメータ: eps={best_params['eps']}, "
                  f"min_samples={best_params['min_samples']} (調整後スコア: {best_score:.4f})")
            print(f"   ✓ クラスタ数: {n_clusters_final}, ノイズ: {n_noise_final}件")
            
            # クラスタ分布を表示（上位10件のみ）
            unique, counts = np.unique(best_labels, return_counts=True)
            sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
            print(f"   主要クラスタ分布（上位10件）:")
            for i, (cluster_id, count) in enumerate(sorted_clusters[:10]):
                label_name = "ノイズ" if cluster_id == -1 else f"クラスタ {cluster_id}"
                print(f"     {label_name}: {count}件")
            
            self.results['DBSCAN'] = {
                'model': best_dbscan,
                'labels': best_labels,
                'n_clusters': len(set(best_labels)) - (1 if -1 in best_labels else 0),
                'score': best_score,
                'params': best_params
            }
        else:
            print(f"   ⚠️ 適切なDBSCANパラメータが見つかりませんでした。")
            # フォールバック: デフォルトパラメータで実行
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            best_labels = dbscan.fit_predict(self.X_scaled)
            
            self.results['DBSCAN'] = {
                'model': dbscan,
                'labels': best_labels,
                'n_clusters': len(set(best_labels)) - (1 if -1 in best_labels else 0),
                'score': -1,
                'params': {'eps': 0.5, 'min_samples': 5}
            }
        
        return best_labels
    
    def try_hdbscan(self, min_cluster_size_range=None, min_samples_range=None, target_clusters=50):
        """HDBSCAN (Hierarchical Density-Based Spatial Clustering)
        
        DBSCANのクラスタ数が多すぎる場合の代替手法として使用。
        HDBSCANは階層的な密度ベースクラスタリングで、より適応的なクラスタを生成する。
        """
        print(f"\n🟡 HDBSCAN (Hierarchical DBSCAN) を実行中...")
        print(f"   目標クラスタ数: {target_clusters}程度")
        
        try:
            import hdbscan
        except ImportError:
            print(f"   ⚠️ HDBSCANがインストールされていません。")
            print(f"   'pip install hdbscan' でインストールしてください。")
            return None
        
        # デフォルトパラメータ範囲
        if min_cluster_size_range is None:
            min_cluster_size_range = [10, 15, 20, 30, 40]  # より細かいクラスタを生成
        
        if min_samples_range is None:
            min_samples_range = [5, 8, 10]  # min_samplesも小さく調整
        
        best_hdbscan = None
        best_labels = None
        best_score = -1
        best_params = None
        
        for min_cluster_size in min_cluster_size_range:
            for min_samples in min_samples_range:
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_method='eom',  # Excess of Mass
                        metric='euclidean'
                    )
                    
                    labels = clusterer.fit_predict(self.X_scaled)
                    
                    # クラスタ数とノイズポイント数
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    # クラスタが2つ以上ある場合のみ評価
                    if n_clusters >= 2:
                        # ノイズポイントを除外してシルエットスコアを計算
                        mask = labels != -1
                        if mask.sum() > 1 and len(set(labels[mask])) > 1:
                            score = silhouette_score(self.X_scaled[mask], labels[mask])
                            
                            # 目標クラスタ数からの距離を計算(ペナルティ)
                            cluster_penalty = abs(n_clusters - target_clusters) / target_clusters
                            # クラスタ数が目標に近いほど高スコア、ノイズ比率も考慮
                            noise_penalty = n_noise / len(labels)
                            adjusted_score = score * (1 - cluster_penalty * 0.5) * (1 - noise_penalty * 0.3)
                            
                            print(f"   min_cluster_size={min_cluster_size}, min_samples={min_samples}: "
                                  f"クラスタ数={n_clusters}, ノイズ={n_noise}, "
                                  f"スコア={score:.4f}, 調整後={adjusted_score:.4f}")
                            
                            # ノイズが少なく、調整後スコアが高く、クラスタ数が適切なものを選択
                            # 実用的なクラスタ数範囲: 20〜80個(意思決定に適した粒度)
                            # ノイズ比率: 40%以下
                            if (adjusted_score > best_score and 
                                n_noise < len(labels) * 0.40 and 
                                20 <= n_clusters <= 80):
                                best_score = adjusted_score
                                best_hdbscan = clusterer
                                best_labels = labels
                                best_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}
                
                except Exception as e:
                    print(f"   ⚠️ min_cluster_size={min_cluster_size}, min_samples={min_samples}でエラー: {e}")
                    continue
        
        if best_labels is not None:
            n_clusters_final = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise_final = list(best_labels).count(-1)
            
            print(f"   ✓ 最適パラメータ: min_cluster_size={best_params['min_cluster_size']}, "
                  f"min_samples={best_params['min_samples']} (調整後スコア: {best_score:.4f})")
            print(f"   ✓ クラスタ数: {n_clusters_final}, ノイズ: {n_noise_final}件")
            
            # クラスタ分布を表示（上位10件のみ）
            unique, counts = np.unique(best_labels, return_counts=True)
            sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
            print(f"   主要クラスタ分布（上位10件）:")
            for i, (cluster_id, count) in enumerate(sorted_clusters[:10]):
                label_name = "ノイズ" if cluster_id == -1 else f"クラスタ {cluster_id}"
                print(f"     {label_name}: {count}件")
            
            self.results['HDBSCAN'] = {
                'model': best_hdbscan,
                'labels': best_labels,
                'n_clusters': n_clusters_final,
                'score': best_score,
                'params': best_params
            }
        else:
            print(f"   ⚠️ 適切なHDBSCANパラメータが見つかりませんでした。")
            self.results['HDBSCAN'] = None
        
        return best_labels
    
    def get_results(self):
        """すべての結果を返す"""
        return self.results


class AlternativeDimensionalityReduction:
    """代替次元削減手法を提供するクラス"""
    
    def __init__(self, X_scaled):
        """
        Parameters:
        -----------
        X_scaled : array-like
            標準化された特徴量
        """
        self.X_scaled = X_scaled
        self.results = {}
    
    def try_tsne(self, n_components=2, perplexity_range=None):
        """t-SNE（t-distributed Stochastic Neighbor Embedding）"""
        print(f"\n🔴 t-SNE を実行中...")
        
        from sklearn.manifold import TSNE
        
        if perplexity_range is None:
            # データサイズに応じて適切なperplexityを選択
            n_samples = len(self.X_scaled)
            perplexity_range = [min(30, n_samples // 4), 
                               min(50, n_samples // 3)]
        
        best_tsne = None
        best_embedding = None
        best_perplexity = perplexity_range[0]
        
        for perplexity in perplexity_range:
            try:
                # scikit-learnのバージョンに応じてパラメータを調整
                tsne_params = {
                    'n_components': n_components,
                    'perplexity': perplexity,
                    'random_state': config.RANDOM_STATE
                }
                
                # バージョン互換性のため、max_iterとn_iterの両方を試す
                try:
                    tsne = TSNE(**tsne_params, n_iter=1000, n_iter_without_progress=300)
                except TypeError:
                    tsne = TSNE(**tsne_params, max_iter=1000, n_iter_without_progress=300)
                
                embedding = tsne.fit_transform(self.X_scaled)
                
                # KL divergenceが利用可能な場合は表示
                if hasattr(tsne, 'kl_divergence_'):
                    print(f"   perplexity={perplexity}: KL divergence = {tsne.kl_divergence_:.4f}")
                    if best_tsne is None or tsne.kl_divergence_ < best_tsne.kl_divergence_:
                        best_tsne = tsne
                        best_embedding = embedding
                        best_perplexity = perplexity
                else:
                    print(f"   perplexity={perplexity}: 完了")
                    if best_tsne is None:
                        best_tsne = tsne
                        best_embedding = embedding
                        best_perplexity = perplexity
            
            except Exception as e:
                print(f"   ⚠️ perplexity={perplexity}でエラー: {e}")
                continue
        
        if best_embedding is not None:
            print(f"   ✓ 最適perplexity: {best_perplexity}")
            
            self.results['t-SNE'] = {
                'model': best_tsne,
                'embedding': best_embedding,
                'perplexity': best_perplexity
            }
        else:
            print(f"   ⚠️ t-SNEの実行に失敗しました。")
            self.results['t-SNE'] = None
        
        return best_embedding
    
    def try_umap(self, n_components=2, n_neighbors_range=None):
        """UMAP（Uniform Manifold Approximation and Projection）"""
        print(f"\n🟠 UMAP を実行中...")
        
        try:
            import umap
        except ImportError:
            print(f"   ⚠️ UMAPがインストールされていません。")
            print(f"   'pip install umap-learn' でインストールしてください。")
            self.results['UMAP'] = None
            return None
        
        if n_neighbors_range is None:
            n_samples = len(self.X_scaled)
            n_neighbors_range = [min(15, n_samples // 10),
                                 min(30, n_samples // 5)]
        
        best_umap = None
        best_embedding = None
        best_n_neighbors = n_neighbors_range[0]
        
        for n_neighbors in n_neighbors_range:
            try:
                umap_model = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    random_state=config.RANDOM_STATE,
                    min_dist=0.1
                )
                
                embedding = umap_model.fit_transform(self.X_scaled)
                
                print(f"   n_neighbors={n_neighbors}: 完了")
                
                if best_umap is None:
                    best_umap = umap_model
                    best_embedding = embedding
                    best_n_neighbors = n_neighbors
            
            except Exception as e:
                print(f"   ⚠️ n_neighbors={n_neighbors}でエラー: {e}")
                continue
        
        if best_embedding is not None:
            print(f"   ✓ 最適n_neighbors: {best_n_neighbors}")
            
            self.results['UMAP'] = {
                'model': best_umap,
                'embedding': best_embedding,
                'n_neighbors': best_n_neighbors
            }
        else:
            print(f"   ⚠️ UMAPの実行に失敗しました。")
            self.results['UMAP'] = None
        
        return best_embedding
    
    def get_results(self):
        """すべての結果を返す"""
        return self.results
