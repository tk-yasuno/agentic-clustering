"""
Dimensionality Reduction Module
Implements t-SNE/UMAP with v0.5 improvements
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Literal
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

warnings.filterwarnings('ignore')


class DimensionalityReducer:
    """
    Dimensionality reduction with automatic optimal method selection.
    
    v0.5 Improvements:
    - t-SNE operational fixes
    - UMAP operational fixes
    - Overlap threshold adjustment (0.10)
    - Automatic optimal method selection
    """
    
    OVERLAP_THRESHOLD = 0.10  # v0.5: Adjusted from default
    
    def __init__(
        self, 
        method: Optional[Literal['auto', 'tsne', 'umap', 'pca']] = 'auto',
        n_components: int = 2,
        random_state: int = 42
    ):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: Reduction method ('auto', 'tsne', 'umap', 'pca')
            n_components: Number of dimensions to reduce to (default: 2)
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.reducer = None
        self.selected_method = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data with optimal dimensionality reduction.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Reduced data of shape (n_samples, n_components)
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select method if auto
        if self.method == 'auto':
            self.selected_method = self._select_optimal_method(X_scaled)
        else:
            self.selected_method = self.method
        
        # Apply reduction
        if self.selected_method == 'tsne':
            X_reduced = self._apply_tsne(X_scaled)
        elif self.selected_method == 'umap':
            X_reduced = self._apply_umap(X_scaled)
        elif self.selected_method == 'pca':
            X_reduced = self._apply_pca(X_scaled)
        else:
            raise ValueError(f"Unknown method: {self.selected_method}")
        
        return X_reduced
    
    def _select_optimal_method(self, X: np.ndarray) -> str:
        """
        Automatically select optimal dimensionality reduction method.
        
        Selection criteria:
        - Small datasets (< 500): PCA
        - Medium datasets (500-5000) with UMAP available: UMAP
        - Medium datasets without UMAP: t-SNE
        - Large datasets (> 5000): UMAP if available, else t-SNE
        
        Args:
            X: Input data
            
        Returns:
            Selected method name
        """
        n_samples = X.shape[0]
        
        if n_samples < 500:
            return 'pca'
        elif n_samples < 5000:
            return 'umap' if UMAP_AVAILABLE else 'tsne'
        else:
            return 'umap' if UMAP_AVAILABLE else 'tsne'
    
    def _apply_tsne(self, X: np.ndarray) -> np.ndarray:
        """
        Apply t-SNE with v0.5 operational fixes.
        
        Fixes include:
        - Proper perplexity adjustment based on sample size
        - Increased iterations for convergence
        - Learning rate optimization
        
        Args:
            X: Scaled input data
            
        Returns:
            Reduced data
        """
        n_samples = X.shape[0]
        
        # Adjust perplexity based on sample size
        perplexity = min(30, max(5, n_samples // 10))
        
        # v0.5: Operational fixes
        self.reducer = TSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            n_iter=1000,  # Increased for better convergence
            learning_rate='auto',  # Automatic learning rate
            random_state=self.random_state,
            init='pca',  # Better initialization
            method='barnes_hut'  # Faster for large datasets
        )
        
        return self.reducer.fit_transform(X)
    
    def _apply_umap(self, X: np.ndarray) -> np.ndarray:
        """
        Apply UMAP with v0.5 operational fixes.
        
        Fixes include:
        - Optimized n_neighbors
        - Proper min_dist setting
        - Metric selection
        
        Args:
            X: Scaled input data
            
        Returns:
            Reduced data
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        n_samples = X.shape[0]
        
        # Adjust n_neighbors based on sample size
        n_neighbors = min(15, max(5, n_samples // 20))
        
        # v0.5: Operational fixes
        self.reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.OVERLAP_THRESHOLD,  # v0.5: Use overlap threshold
            metric='euclidean',
            random_state=self.random_state,
            n_epochs=500  # Sufficient for convergence
        )
        
        return self.reducer.fit_transform(X)
    
    def _apply_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Apply PCA for fast linear reduction.
        
        Args:
            X: Scaled input data
            
        Returns:
            Reduced data
        """
        from sklearn.decomposition import PCA
        
        self.reducer = PCA(
            n_components=self.n_components,
            random_state=self.random_state
        )
        
        return self.reducer.fit_transform(X)
    
    def calculate_overlap_score(self, X_reduced: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate cluster overlap score in reduced space.
        
        Uses overlap threshold of 0.10 (v0.5 adjustment).
        
        Args:
            X_reduced: Reduced dimensional data
            labels: Cluster labels
            
        Returns:
            Overlap score (0-1, lower is better separation)
        """
        if len(np.unique(labels)) < 2:
            return 0.0
        
        # Calculate pairwise distances within and between clusters
        unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
        
        if len(unique_labels) < 2:
            return 0.0
        
        within_dist = []
        between_dist = []
        
        for label in unique_labels:
            mask = labels == label
            cluster_points = X_reduced[mask]
            other_points = X_reduced[~mask]
            
            if len(cluster_points) > 1:
                # Within-cluster distances
                for i in range(len(cluster_points)):
                    for j in range(i + 1, len(cluster_points)):
                        within_dist.append(
                            np.linalg.norm(cluster_points[i] - cluster_points[j])
                        )
                
                # Between-cluster distances
                for point in cluster_points:
                    min_dist = np.min([
                        np.linalg.norm(point - other)
                        for other in other_points[:100]  # Sample for efficiency
                    ])
                    between_dist.append(min_dist)
        
        if not within_dist or not between_dist:
            return 0.0
        
        # Overlap score: ratio of mean within to mean between
        mean_within = np.mean(within_dist)
        mean_between = np.mean(between_dist)
        
        overlap = mean_within / (mean_between + 1e-10)
        
        return min(1.0, overlap)
    
    def get_method_info(self) -> dict:
        """
        Get information about the selected method.
        
        Returns:
            Dictionary with method information
        """
        return {
            'method': self.selected_method,
            'n_components': self.n_components,
            'overlap_threshold': self.OVERLAP_THRESHOLD,
            'reducer_type': type(self.reducer).__name__ if self.reducer else None
        }
