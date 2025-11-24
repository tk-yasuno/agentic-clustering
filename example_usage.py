"""
Example Usage of Agentic Clustering v0.5

Demonstrates the 13-feature system, agentic workflow, and dimensionality reduction.

Note: This script uses src imports for development. After installing the package
(pip install -e . or pip install .), use:
    from agentic_clustering import GeospatialFeatureExtractor, ...
"""

import numpy as np
import pandas as pd

# Development import (for running directly from source)
from src.agentic_clustering import (
    GeospatialFeatureExtractor,
    AgenticClusteringWorkflow,
    DimensionalityReducer
)

# After installation, use this instead:
# from agentic_clustering import (
#     GeospatialFeatureExtractor,
#     AgenticClusteringWorkflow,
#     DimensionalityReducer
# )


def generate_sample_bridge_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic bridge maintenance data for demonstration.
    
    Args:
        n_samples: Number of bridge records to generate
        
    Returns:
        DataFrame with bridge data
    """
    np.random.seed(42)
    
    # Yamaguchi Prefecture approximate bounds
    lat_min, lat_max = 33.7, 34.8
    lon_min, lon_max = 130.8, 132.2
    
    data = pd.DataFrame({
        'bridge_id': range(n_samples),
        'latitude': np.random.uniform(lat_min, lat_max, n_samples),
        'longitude': np.random.uniform(lon_min, lon_max, n_samples),
        'elevation': np.random.uniform(0, 500, n_samples),
        'bridge_length': np.random.lognormal(3, 1, n_samples),  # meters
        'bridge_width': np.random.uniform(5, 20, n_samples),  # meters
        'year_built': np.random.randint(1960, 2020, n_samples),
        'last_inspection_year': np.random.randint(2018, 2024, n_samples),
        'damage_score': np.random.uniform(0, 10, n_samples),
        'traffic_volume': np.random.lognormal(8, 2, n_samples),  # vehicles/day
        'population_density': np.random.lognormal(5, 2, n_samples),  # people/km²
        'terrain_slope': np.random.uniform(0, 30, n_samples),  # degrees
    })
    
    # Add bridge names with some river bridges
    river_mask = np.random.rand(n_samples) < 0.3  # 30% are river bridges
    data['bridge_name'] = [
        f"River Bridge {i}" if river_mask[i] else f"Bridge {i}"
        for i in range(n_samples)
    ]
    
    return data


def main():
    """Main demonstration of v0.5 features."""
    
    print("=" * 80)
    print("Agentic Clustering v0.5 - Demonstration")
    print("=" * 80)
    print()
    
    # Step 1: Generate sample data
    print("Step 1: Generating sample bridge data...")
    data = generate_sample_bridge_data(n_samples=1000)
    print(f"Generated {len(data)} bridge records")
    print(f"Columns: {list(data.columns)}")
    print()
    
    # Step 2: Extract 13 geospatial features (NEW v0.5)
    print("Step 2: Extracting 13 geospatial features (v0.5)...")
    extractor = GeospatialFeatureExtractor(
        coastal_reference_point=(34.186, 131.472)  # Yamaguchi coast
    )
    features = extractor.extract_features(data)
    
    print(f"Features extracted: {list(features.columns)}")
    print(f"Feature shape: {features.shape}")
    
    # Validate features
    is_valid, message = extractor.validate_features(features)
    print(f"Validation: {message}")
    
    # Show NEW v0.5 features
    print("\nNEW v0.5 Features:")
    print(f"  - under_river: {features['under_river'].sum()} bridges over rivers")
    print(f"  - distance_to_coast_km: min={features['distance_to_coast_km'].min():.1f}, "
          f"max={features['distance_to_coast_km'].max():.1f}, "
          f"mean={features['distance_to_coast_km'].mean():.1f}")
    print()
    
    # Step 3: Dimensionality reduction (v0.5 improvements)
    print("Step 3: Dimensionality reduction with v0.5 improvements...")
    reducer = DimensionalityReducer(method='auto', n_components=2)
    X_reduced = reducer.fit_transform(features.values)
    
    method_info = reducer.get_method_info()
    print(f"Selected method: {method_info['method']}")
    print(f"Overlap threshold: {method_info['overlap_threshold']}")
    print(f"Reduced shape: {X_reduced.shape}")
    print()
    
    # Step 4: Agentic clustering workflow (v0.5 optimization)
    print("Step 4: Agentic clustering workflow with v0.5 optimization...")
    workflow = AgenticClusteringWorkflow(method='auto', min_cluster_size=5)
    labels = workflow.fit_predict(X_reduced)
    
    print(f"Selected method: {workflow.selected_method}")
    print(f"DBSCAN exclusion threshold: {workflow.MAX_DBSCAN_CLUSTERS} clusters")
    print()
    
    # Step 5: Evaluate clustering
    print("Step 5: Evaluating clustering quality...")
    metrics = workflow.evaluate_clustering(X_reduced)
    
    print("Clustering Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Step 6: Cluster summary
    print("Step 6: Cluster summary...")
    summary = workflow.get_cluster_summary()
    print(summary.head(10))
    print()
    
    # Step 7: Calculate overlap score
    print("Step 7: Calculating overlap score (v0.5 threshold=0.10)...")
    overlap_score = reducer.calculate_overlap_score(X_reduced, labels)
    print(f"Overlap score: {overlap_score:.4f}")
    print(f"Overlap threshold: {reducer.OVERLAP_THRESHOLD}")
    print(f"Separation quality: {'Good' if overlap_score < reducer.OVERLAP_THRESHOLD else 'Moderate'}")
    print()
    
    # Step 8: Show optimization history
    print("Step 8: Optimization history...")
    if workflow.optimization_history:
        print("Clustering attempts:")
        for i, entry in enumerate(workflow.optimization_history, 1):
            print(f"  Attempt {i}: {entry}")
    print()
    
    print("=" * 80)
    print("Demonstration complete!")
    print("=" * 80)
    print()
    print("Key v0.5 Features Demonstrated:")
    print("  ✓ 13-feature system (including under_river, distance_to_coast_km)")
    print("  ✓ DBSCAN exclusion rule (clusters > 50)")
    print("  ✓ HDBSCAN auto-triggering with parameter optimization")
    print("  ✓ t-SNE/UMAP operational fixes")
    print("  ✓ Overlap threshold adjustment (0.10)")
    print("  ✓ Automatic optimal method selection")


if __name__ == "__main__":
    main()
