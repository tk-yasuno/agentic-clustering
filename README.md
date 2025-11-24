# agentic-clustering

This project applies **self-improving (Agentic) clustering** to bridge maintenance data in Yamaguchi Prefecture, Japan, to automatically identify bridge groups with high maintenance priority.

## Version 0.5 Features

### 1. Geospatial Features (13-Feature System)
- **Basic Geospatial**: latitude, longitude, elevation
- **Structural**: bridge_length, bridge_width
- **Temporal**: year_built, last_inspection_year
- **Condition & Usage**: damage_score, traffic_volume, population_density
- **NEW v0.5 Geospatial Features**:
  - `under_river`: Binary flag indicating if bridge crosses over a river
  - `distance_to_coast_km`: Distance to coastline in kilometers
- **Terrain**: terrain_slope

### 2. Agentic Workflow Optimization
- **DBSCAN Exclusion Rule**: Automatically switches to HDBSCAN when DBSCAN produces > 50 clusters
- **HDBSCAN Auto-triggering**: Intelligent parameter optimization when auto-triggered
- **Adaptive Parameter Tuning**: Dynamic parameter selection based on data characteristics

### 3. Dimensionality Reduction Improvements
- **t-SNE Operational Fixes**: 
  - Proper perplexity adjustment
  - Improved convergence with increased iterations
  - Automatic learning rate optimization
- **UMAP Operational Fixes**: 
  - Optimized n_neighbors
  - Proper min_dist setting
- **Overlap Threshold**: Adjusted to 0.10 for better cluster separation
- **Automatic Optimal Method Selection**: Chooses best reduction method based on dataset size

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Usage

After installation:

```python
from agentic_clustering import (
    GeospatialFeatureExtractor,
    AgenticClusteringWorkflow,
    DimensionalityReducer
)

# Extract 13 geospatial features
extractor = GeospatialFeatureExtractor()
features = extractor.extract_features(bridge_data)

# Reduce dimensionality
reducer = DimensionalityReducer(method='auto')
X_reduced = reducer.fit_transform(features.values)

# Apply agentic clustering
workflow = AgenticClusteringWorkflow(method='auto')
labels = workflow.fit_predict(X_reduced)
```

## Example

Run the example script to see v0.5 features in action:

```bash
# For development (without installation)
python example_usage.py

# After installation
python -c "from agentic_clustering import GeospatialFeatureExtractor; print('Package installed successfully!')"
```

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

## Project Structure

```
agentic-clustering/
├── src/
│   └── agentic_clustering/
│       ├── __init__.py
│       ├── features.py         # 13-feature extraction
│       ├── clustering.py       # Agentic clustering workflow
│       └── reduction.py        # Dimensionality reduction
├── tests/
│   ├── test_features.py
│   ├── test_clustering.py
│   └── test_reduction.py
├── example_usage.py
├── requirements.txt
└── setup.py
```

## Key Improvements in v0.5

1. **Geospatial Features Added** (13-Feature System)
   - Under river flag (under_river)
   - Distance to coastline (distance_to_coast_km)

2. **Agentic Workflow Optimization**
   - DBSCAN exclusion rule (when clusters > 50)
   - HDBSCAN auto-triggering with parameter optimization

3. **Dimensionality Reduction Improvements**
   - t-SNE/UMAP operational fixes
   - Overlap threshold adjustment (0.10)
   - Automatic optimal method selection
