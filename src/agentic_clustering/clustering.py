"""
Agentic Clustering Workflow Module
Implements self-improving clustering with v0.5 optimization
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")

warnings.filterwarnings('ignore')


class AgenticClusteringWorkflow:
    """
    Self-improving clustering workflow with agentic optimization.
    
    v0.5 Improvements:
    - DBSCAN exclusion rule (when clusters > 50)
    - HDBSCAN auto-triggering with parameter optimization
    - Adaptive parameter tuning
    """
    
    MAX_DBSCAN_CLUSTERS = 50  # v0.5: DBSCAN exclusion threshold
    
    def __init__(
        self,
        method: Optional[str] = 'auto',
        min_cluster_size: int = 5,
        random_state: int = 42
    ):
        """
        Initialize agentic clustering workflow.
        
        Args:
            method: Clustering method ('auto', 'dbscan', 'hdbscan')
            min_cluster_size: Minimum cluster size
            random_state: Random seed
        """
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.clusterer = None
        self.labels_ = None
        self.selected_method = None
        self.optimization_history = []
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit clustering model and predict labels with agentic optimization.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels of shape (n_samples,)
        """
        # Select method with agentic logic
        if self.method == 'auto':
            self.selected_method = self._select_clustering_method(X)
        else:
            self.selected_method = self.method
        
        # Apply clustering with optimization
        if self.selected_method == 'dbscan':
            self.labels_ = self._apply_dbscan(X)
        elif self.selected_method == 'hdbscan':
            self.labels_ = self._apply_hdbscan(X)
        else:
            raise ValueError(f"Unknown method: {self.selected_method}")
        
        # Check DBSCAN exclusion rule (v0.5)
        if self.selected_method == 'dbscan':
            n_clusters = len(np.unique(self.labels_[self.labels_ >= 0]))
            if n_clusters > self.MAX_DBSCAN_CLUSTERS:
                print(f"DBSCAN produced {n_clusters} clusters (> {self.MAX_DBSCAN_CLUSTERS})")
                print("Triggering HDBSCAN with optimized parameters...")
                self.selected_method = 'hdbscan'
                self.labels_ = self._apply_hdbscan(X, optimize=True)
        
        return self.labels_
    
    def _select_clustering_method(self, X: np.ndarray) -> str:
        """
        Automatically select optimal clustering method.
        
        Selection criteria:
        - Small datasets (< 500): Try DBSCAN first
        - Medium/Large datasets: Use HDBSCAN if available
        - Fallback to DBSCAN if HDBSCAN unavailable
        
        Args:
            X: Input data
            
        Returns:
            Selected method name
        """
        n_samples = X.shape[0]
        
        if HDBSCAN_AVAILABLE and n_samples >= 500:
            return 'hdbscan'
        elif HDBSCAN_AVAILABLE:
            # Try DBSCAN first, may switch to HDBSCAN if needed
            return 'dbscan'
        else:
            return 'dbscan'
    
    def _apply_dbscan(self, X: np.ndarray) -> np.ndarray:
        """
        Apply DBSCAN with adaptive parameter tuning.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        n_samples = X.shape[0]
        
        # Adaptive eps and min_samples
        eps = self._estimate_eps(X)
        min_samples = max(self.min_cluster_size, int(np.log(n_samples)))
        
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.clusterer.fit_predict(X)
        
        # Log optimization
        n_clusters = len(np.unique(labels[labels >= 0]))
        n_noise = np.sum(labels == -1)
        
        self.optimization_history.append({
            'method': 'dbscan',
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        })
        
        return labels
    
    def _apply_hdbscan(self, X: np.ndarray, optimize: bool = False) -> np.ndarray:
        """
        Apply HDBSCAN with auto-triggering and parameter optimization (v0.5).
        
        Args:
            X: Input data
            optimize: If True, perform extensive parameter optimization
            
        Returns:
            Cluster labels
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        if optimize:
            # v0.5: Optimized parameters when auto-triggered
            labels = self._optimize_hdbscan(X)
        else:
            # Standard HDBSCAN
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_cluster_size,
                cluster_selection_epsilon=0.0,
                metric='euclidean'
            )
            labels = self.clusterer.fit_predict(X)
        
        # Log optimization
        n_clusters = len(np.unique(labels[labels >= 0]))
        n_noise = np.sum(labels == -1)
        
        self.optimization_history.append({
            'method': 'hdbscan',
            'min_cluster_size': self.min_cluster_size,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'optimized': optimize
        })
        
        return labels
    
    def _optimize_hdbscan(self, X: np.ndarray) -> np.ndarray:
        """
        Optimize HDBSCAN parameters for best clustering (v0.5).
        
        Tries multiple parameter combinations and selects best based on:
        - Number of clusters (target: 10-50)
        - Silhouette score
        - Noise ratio (target: < 10%)
        
        Args:
            X: Input data
            
        Returns:
            Best cluster labels
        """
        n_samples = X.shape[0]
        
        # Parameter grid for optimization
        min_cluster_sizes = [
            max(5, n_samples // 200),
            max(5, n_samples // 100),
            max(5, n_samples // 50)
        ]
        
        best_score = -1
        best_labels = None
        best_params = None
        
        for mcs in min_cluster_sizes:
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=mcs,
                    cluster_selection_epsilon=0.0,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                labels = clusterer.fit_predict(X)
                
                # Evaluate clustering
                n_clusters = len(np.unique(labels[labels >= 0]))
                n_noise = np.sum(labels == -1)
                noise_ratio = n_noise / n_samples
                
                # Skip if too many/few clusters or too much noise
                if n_clusters < 2 or n_clusters > 50 or noise_ratio > 0.1:
                    continue
                
                # Calculate silhouette score
                if n_clusters >= 2 and n_noise < n_samples - 1:
                    valid_mask = labels >= 0
                    if valid_mask.sum() > n_clusters:
                        score = silhouette_score(X[valid_mask], labels[valid_mask])
                    else:
                        score = 0
                else:
                    score = 0
                
                # Composite score favoring 10-50 clusters
                cluster_penalty = abs(n_clusters - 30) / 30  # Favor ~30 clusters
                composite_score = score - cluster_penalty - noise_ratio
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_labels = labels
                    best_params = {
                        'min_cluster_size': mcs,
                        'n_clusters': n_clusters,
                        'noise_ratio': noise_ratio,
                        'silhouette': score
                    }
            
            except Exception as e:
                continue
        
        # Use best if found, otherwise use default
        if best_labels is not None:
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=best_params['min_cluster_size'],
                min_samples=best_params['min_cluster_size'],
                cluster_selection_epsilon=0.0,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            self.clusterer.fit(X)
            print(f"Optimized HDBSCAN: {best_params}")
            return best_labels
        else:
            # Fallback to default
            return self._apply_hdbscan(X, optimize=False)
    
    def _estimate_eps(self, X: np.ndarray) -> float:
        """
        Estimate optimal eps for DBSCAN using k-distance method.
        
        Args:
            X: Input data
            
        Returns:
            Estimated eps value
        """
        from sklearn.neighbors import NearestNeighbors
        
        k = min(self.min_cluster_size, X.shape[0] - 1)
        
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Use 90th percentile of k-distances
        eps = np.percentile(distances[:, -1], 90)
        
        return eps
    
    def evaluate_clustering(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate clustering quality with multiple metrics.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.labels_ is None:
            return {}
        
        n_clusters = len(np.unique(self.labels_[self.labels_ >= 0]))
        n_noise = np.sum(self.labels_ == -1)
        noise_ratio = n_noise / len(self.labels_)
        
        metrics = {
            'method': self.selected_method,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'n_samples': len(self.labels_)
        }
        
        # Silhouette score (only if we have valid clusters)
        if n_clusters >= 2 and n_noise < len(self.labels_) - 1:
            valid_mask = self.labels_ >= 0
            if valid_mask.sum() > n_clusters:
                try:
                    metrics['silhouette_score'] = silhouette_score(
                        X[valid_mask], self.labels_[valid_mask]
                    )
                    metrics['davies_bouldin_score'] = davies_bouldin_score(
                        X[valid_mask], self.labels_[valid_mask]
                    )
                except Exception:
                    pass
        
        return metrics
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        
        Returns:
            DataFrame with cluster statistics
        """
        if self.labels_ is None:
            return pd.DataFrame()
        
        unique_labels = np.unique(self.labels_)
        
        summary = []
        for label in unique_labels:
            mask = self.labels_ == label
            summary.append({
                'cluster_id': label,
                'size': mask.sum(),
                'percentage': mask.sum() / len(self.labels_) * 100
            })
        
        return pd.DataFrame(summary).sort_values('size', ascending=False)
