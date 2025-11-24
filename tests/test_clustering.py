"""
Tests for agentic clustering workflow (v0.5)
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agentic_clustering.clustering import AgenticClusteringWorkflow


def generate_test_data(n_samples=100, n_features=13, n_clusters=3):
    """Generate synthetic clustered data for testing."""
    np.random.seed(42)
    X = []
    for i in range(n_clusters):
        center = np.random.randn(n_features) * 10
        cluster = np.random.randn(n_samples // n_clusters, n_features) + center
        X.append(cluster)
    return np.vstack(X)


def test_workflow_initialization():
    """Test AgenticClusteringWorkflow initialization."""
    workflow = AgenticClusteringWorkflow(method='auto', min_cluster_size=5)
    
    assert workflow.method == 'auto'
    assert workflow.min_cluster_size == 5
    assert workflow.MAX_DBSCAN_CLUSTERS == 50


def test_dbscan_clustering():
    """Test DBSCAN clustering."""
    X = generate_test_data(n_samples=150, n_clusters=3)
    
    workflow = AgenticClusteringWorkflow(method='dbscan', min_cluster_size=5)
    labels = workflow.fit_predict(X)
    
    assert len(labels) == len(X)
    assert workflow.selected_method == 'dbscan'
    
    # Should have found some clusters
    n_clusters = len(np.unique(labels[labels >= 0]))
    assert n_clusters >= 1


def test_dbscan_exclusion_rule():
    """Test DBSCAN exclusion rule when clusters > 50 (v0.5 feature)."""
    # This is a conceptual test - in practice we'd need data that produces >50 clusters
    workflow = AgenticClusteringWorkflow(method='dbscan', min_cluster_size=5)
    
    # Check the threshold is set correctly
    assert workflow.MAX_DBSCAN_CLUSTERS == 50


def test_auto_method_selection():
    """Test automatic method selection."""
    X = generate_test_data(n_samples=100, n_clusters=3)
    
    workflow = AgenticClusteringWorkflow(method='auto', min_cluster_size=5)
    labels = workflow.fit_predict(X)
    
    assert workflow.selected_method in ['dbscan', 'hdbscan']
    assert len(labels) == len(X)


def test_evaluate_clustering():
    """Test clustering evaluation metrics."""
    X = generate_test_data(n_samples=150, n_clusters=3)
    
    workflow = AgenticClusteringWorkflow(method='dbscan', min_cluster_size=5)
    labels = workflow.fit_predict(X)
    
    metrics = workflow.evaluate_clustering(X)
    
    assert 'method' in metrics
    assert 'n_clusters' in metrics
    assert 'n_noise' in metrics
    assert 'noise_ratio' in metrics
    assert metrics['n_samples'] == len(X)


def test_cluster_summary():
    """Test cluster summary generation."""
    X = generate_test_data(n_samples=150, n_clusters=3)
    
    workflow = AgenticClusteringWorkflow(method='dbscan', min_cluster_size=5)
    labels = workflow.fit_predict(X)
    
    summary = workflow.get_cluster_summary()
    
    assert not summary.empty
    assert 'cluster_id' in summary.columns
    assert 'size' in summary.columns
    assert 'percentage' in summary.columns


def test_optimization_history():
    """Test that optimization history is recorded."""
    X = generate_test_data(n_samples=100, n_clusters=3)
    
    workflow = AgenticClusteringWorkflow(method='dbscan', min_cluster_size=5)
    workflow.fit_predict(X)
    
    assert len(workflow.optimization_history) > 0
    assert 'method' in workflow.optimization_history[0]


def test_small_dataset():
    """Test clustering with small dataset."""
    X = generate_test_data(n_samples=30, n_clusters=2)
    
    workflow = AgenticClusteringWorkflow(method='auto', min_cluster_size=3)
    labels = workflow.fit_predict(X)
    
    assert len(labels) == len(X)
    # Should have found at least one cluster
    assert len(np.unique(labels)) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
