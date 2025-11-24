"""
Tests for dimensionality reduction (v0.5)
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agentic_clustering.reduction import DimensionalityReducer


def generate_test_data(n_samples=100, n_features=13):
    """Generate synthetic data for testing."""
    np.random.seed(42)
    return np.random.randn(n_samples, n_features)


def test_reducer_initialization():
    """Test DimensionalityReducer initialization."""
    reducer = DimensionalityReducer(method='auto', n_components=2)
    
    assert reducer.method == 'auto'
    assert reducer.n_components == 2
    assert reducer.OVERLAP_THRESHOLD == 0.10  # v0.5 threshold


def test_overlap_threshold_v05():
    """Test that overlap threshold is set to 0.10 (v0.5 feature)."""
    reducer = DimensionalityReducer()
    
    assert reducer.OVERLAP_THRESHOLD == 0.10


def test_tsne_reduction():
    """Test t-SNE dimensionality reduction."""
    X = generate_test_data(n_samples=100, n_features=13)
    
    reducer = DimensionalityReducer(method='tsne', n_components=2)
    X_reduced = reducer.fit_transform(X)
    
    assert X_reduced.shape == (100, 2)
    assert reducer.selected_method == 'tsne'


def test_pca_reduction():
    """Test PCA dimensionality reduction."""
    X = generate_test_data(n_samples=100, n_features=13)
    
    reducer = DimensionalityReducer(method='pca', n_components=2)
    X_reduced = reducer.fit_transform(X)
    
    assert X_reduced.shape == (100, 2)
    assert reducer.selected_method == 'pca'


def test_auto_method_selection_small():
    """Test auto method selection for small dataset."""
    X = generate_test_data(n_samples=100, n_features=13)
    
    reducer = DimensionalityReducer(method='auto', n_components=2)
    X_reduced = reducer.fit_transform(X)
    
    assert X_reduced.shape == (100, 2)
    # Small dataset should use PCA
    assert reducer.selected_method == 'pca'


def test_auto_method_selection_medium():
    """Test auto method selection for medium dataset."""
    X = generate_test_data(n_samples=1000, n_features=13)
    
    reducer = DimensionalityReducer(method='auto', n_components=2)
    X_reduced = reducer.fit_transform(X)
    
    assert X_reduced.shape == (1000, 2)
    assert reducer.selected_method in ['tsne', 'umap']


def test_get_method_info():
    """Test method info retrieval."""
    X = generate_test_data(n_samples=100, n_features=13)
    
    reducer = DimensionalityReducer(method='pca', n_components=2)
    reducer.fit_transform(X)
    
    info = reducer.get_method_info()
    
    assert 'method' in info
    assert 'n_components' in info
    assert 'overlap_threshold' in info
    assert info['overlap_threshold'] == 0.10


def test_calculate_overlap_score():
    """Test overlap score calculation."""
    X = generate_test_data(n_samples=100, n_features=13)
    
    reducer = DimensionalityReducer(method='pca', n_components=2)
    X_reduced = reducer.fit_transform(X)
    
    # Create simple labels
    labels = np.array([0] * 50 + [1] * 50)
    
    overlap_score = reducer.calculate_overlap_score(X_reduced, labels)
    
    assert 0.0 <= overlap_score <= 1.0


def test_calculate_overlap_score_with_noise():
    """Test overlap score with noise points (-1 labels)."""
    X = generate_test_data(n_samples=100, n_features=13)
    
    reducer = DimensionalityReducer(method='pca', n_components=2)
    X_reduced = reducer.fit_transform(X)
    
    # Create labels with noise
    labels = np.array([0] * 40 + [1] * 40 + [-1] * 20)
    
    overlap_score = reducer.calculate_overlap_score(X_reduced, labels)
    
    assert 0.0 <= overlap_score <= 1.0


def test_standardization():
    """Test that data is standardized before reduction."""
    X = generate_test_data(n_samples=100, n_features=13)
    # Add some features with different scales
    X[:, 0] = X[:, 0] * 1000
    X[:, 1] = X[:, 1] * 0.001
    
    reducer = DimensionalityReducer(method='pca', n_components=2)
    X_reduced = reducer.fit_transform(X)
    
    # Should still work properly despite scale differences
    assert X_reduced.shape == (100, 2)
    assert not np.any(np.isnan(X_reduced))


def test_three_components():
    """Test reduction to 3 components."""
    X = generate_test_data(n_samples=100, n_features=13)
    
    reducer = DimensionalityReducer(method='pca', n_components=3)
    X_reduced = reducer.fit_transform(X)
    
    assert X_reduced.shape == (100, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
