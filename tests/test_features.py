"""
Tests for geospatial feature extraction (v0.5)
"""

import numpy as np
import pandas as pd
import pytest

from src.agentic_clustering.features import GeospatialFeatureExtractor


def test_feature_extractor_initialization():
    """Test GeospatialFeatureExtractor initialization."""
    extractor = GeospatialFeatureExtractor()
    
    assert extractor.coastal_reference_point == (34.186, 131.472)
    assert len(extractor.feature_names) == 13


def test_extract_basic_features():
    """Test extraction of basic features."""
    data = pd.DataFrame({
        'latitude': [34.0, 34.1],
        'longitude': [131.0, 131.1],
        'elevation': [100, 200],
        'bridge_length': [50, 100],
        'bridge_width': [10, 15],
        'year_built': [1990, 2000],
        'last_inspection_year': [2020, 2021],
        'damage_score': [5.0, 3.0],
        'traffic_volume': [1000, 2000],
        'population_density': [500, 1000],
        'terrain_slope': [5, 10],
    })
    
    extractor = GeospatialFeatureExtractor()
    features = extractor.extract_features(data)
    
    assert features.shape == (2, 13)
    assert 'latitude' in features.columns
    assert 'under_river' in features.columns
    assert 'distance_to_coast_km' in features.columns


def test_under_river_flag_from_name():
    """Test under_river flag extraction from bridge name (v0.5 feature)."""
    data = pd.DataFrame({
        'bridge_name': ['River Bridge 1', 'Highway Bridge', 'Stream Bridge'],
        'latitude': [34.0, 34.1, 34.2],
        'longitude': [131.0, 131.1, 131.2],
    })
    
    extractor = GeospatialFeatureExtractor()
    features = extractor.extract_features(data)
    
    assert features['under_river'][0] == 1  # Contains 'river'
    assert features['under_river'][1] == 0  # No river keyword
    assert features['under_river'][2] == 1  # Contains 'stream'


def test_under_river_flag_explicit():
    """Test under_river flag when explicitly provided."""
    data = pd.DataFrame({
        'under_river': [1, 0, 1],
        'latitude': [34.0, 34.1, 34.2],
        'longitude': [131.0, 131.1, 131.2],
    })
    
    extractor = GeospatialFeatureExtractor()
    features = extractor.extract_features(data)
    
    assert features['under_river'][0] == 1
    assert features['under_river'][1] == 0
    assert features['under_river'][2] == 1


def test_distance_to_coast():
    """Test distance_to_coast_km calculation (v0.5 feature)."""
    # Yamaguchi Prefecture coordinates
    data = pd.DataFrame({
        'latitude': [34.186, 34.5],  # One at coast, one inland
        'longitude': [131.472, 131.8],
    })
    
    extractor = GeospatialFeatureExtractor(
        coastal_reference_point=(34.186, 131.472)
    )
    features = extractor.extract_features(data)
    
    # First point should be very close to coast (approximately 0)
    assert features['distance_to_coast_km'][0] < 1.0
    
    # Second point should be farther
    assert features['distance_to_coast_km'][1] > features['distance_to_coast_km'][0]


def test_validate_features_success():
    """Test feature validation with valid features."""
    data = pd.DataFrame({
        'latitude': [34.0],
        'longitude': [131.0],
        'bridge_name': ['Test Bridge'],
    })
    
    extractor = GeospatialFeatureExtractor()
    features = extractor.extract_features(data)
    
    is_valid, message = extractor.validate_features(features)
    
    assert is_valid
    assert "successfully" in message.lower()


def test_validate_features_missing():
    """Test feature validation with missing features."""
    features = pd.DataFrame({
        'latitude': [34.0],
        'longitude': [131.0],
    })
    
    extractor = GeospatialFeatureExtractor()
    is_valid, message = extractor.validate_features(features)
    
    assert not is_valid
    assert "Missing features" in message


def test_13_features_present():
    """Test that all 13 features are extracted (v0.5 requirement)."""
    data = pd.DataFrame({
        'latitude': [34.0],
        'longitude': [131.0],
    })
    
    extractor = GeospatialFeatureExtractor()
    features = extractor.extract_features(data)
    
    # Check all 13 features are present
    expected_features = [
        'latitude', 'longitude', 'elevation', 'bridge_length', 'bridge_width',
        'year_built', 'last_inspection_year', 'damage_score', 'traffic_volume',
        'population_density', 'under_river', 'distance_to_coast_km', 'terrain_slope'
    ]
    
    for feature in expected_features:
        assert feature in features.columns
    
    assert features.shape[1] == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
