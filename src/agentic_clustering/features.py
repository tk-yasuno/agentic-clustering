"""
Geospatial Feature Extraction Module
Implements 13-feature system for bridge maintenance clustering
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from geopy.distance import geodesic
from shapely.geometry import Point, LineString, Polygon
import warnings

warnings.filterwarnings('ignore')


class GeospatialFeatureExtractor:
    """
    Extracts 13 geospatial features for bridge maintenance prioritization.
    
    Features include:
    1. latitude
    2. longitude
    3. elevation (if available)
    4. bridge_length
    5. bridge_width
    6. year_built
    7. last_inspection_year
    8. damage_score
    9. traffic_volume
    10. population_density
    11. under_river (NEW in v0.5)
    12. distance_to_coast_km (NEW in v0.5)
    13. terrain_slope
    """
    
    def __init__(self, coastal_reference_point: Optional[Tuple[float, float]] = None):
        """
        Initialize the feature extractor.
        
        Args:
            coastal_reference_point: (latitude, longitude) of a coastal reference point.
                                    Defaults to Yamaguchi coast (34.186, 131.472)
        """
        self.coastal_reference_point = coastal_reference_point or (34.186, 131.472)
        self.feature_names = [
            'latitude', 'longitude', 'elevation', 'bridge_length', 'bridge_width',
            'year_built', 'last_inspection_year', 'damage_score', 'traffic_volume',
            'population_density', 'under_river', 'distance_to_coast_km', 'terrain_slope'
        ]
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all 13 features from input data.
        
        Args:
            data: DataFrame with bridge information
            
        Returns:
            DataFrame with 13 features extracted
        """
        features = pd.DataFrame()
        
        # Basic geospatial features (1-3)
        features['latitude'] = data.get('latitude', 0.0)
        features['longitude'] = data.get('longitude', 0.0)
        features['elevation'] = data.get('elevation', 0.0)
        
        # Bridge structural features (4-5)
        features['bridge_length'] = data.get('bridge_length', 0.0)
        features['bridge_width'] = data.get('bridge_width', 0.0)
        
        # Temporal features (6-7)
        features['year_built'] = data.get('year_built', 2000)
        features['last_inspection_year'] = data.get('last_inspection_year', 2023)
        
        # Condition and usage features (8-10)
        features['damage_score'] = data.get('damage_score', 0.0)
        features['traffic_volume'] = data.get('traffic_volume', 0.0)
        features['population_density'] = data.get('population_density', 0.0)
        
        # NEW v0.5: Geospatial features (11-12)
        features['under_river'] = self._extract_under_river_flag(data)
        features['distance_to_coast_km'] = self._calculate_distance_to_coast(
            features['latitude'], features['longitude']
        )
        
        # Terrain feature (13)
        features['terrain_slope'] = data.get('terrain_slope', 0.0)
        
        return features
    
    def _extract_under_river_flag(self, data: pd.DataFrame) -> pd.Series:
        """
        Extract under_river flag (NEW in v0.5).
        
        Determines if a bridge crosses over a river based on:
        - Explicit 'under_river' or 'over_river' column
        - Bridge type classification
        - Name patterns (contains 'river', '川', etc.)
        
        Args:
            data: Input DataFrame
            
        Returns:
            Series with binary flag (1 if under/over river, 0 otherwise)
        """
        # Check if explicitly provided
        if 'under_river' in data.columns:
            return data['under_river'].astype(int)
        
        if 'over_river' in data.columns:
            return data['over_river'].astype(int)
        
        # Infer from bridge type
        if 'bridge_type' in data.columns:
            river_types = ['river', 'stream', 'water', '河川']
            is_river = data['bridge_type'].astype(str).str.lower().apply(
                lambda x: any(t in x for t in river_types)
            )
            if is_river.sum() > 0:
                return is_river.astype(int)
        
        # Infer from bridge name
        if 'bridge_name' in data.columns:
            river_patterns = ['river', 'stream', '川', '河']
            is_river = data['bridge_name'].astype(str).apply(
                lambda x: any(p in x.lower() for p in river_patterns)
            )
            return is_river.astype(int)
        
        # Default: assume not over river
        return pd.Series(0, index=data.index)
    
    def _calculate_distance_to_coast(
        self, 
        latitudes: pd.Series, 
        longitudes: pd.Series
    ) -> pd.Series:
        """
        Calculate distance to coastline in kilometers (NEW in v0.5).
        
        Uses geodesic distance to a reference coastal point.
        
        Args:
            latitudes: Series of latitude values
            longitudes: Series of longitude values
            
        Returns:
            Series with distances in kilometers
        """
        distances = []
        
        for lat, lon in zip(latitudes, longitudes):
            if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
                distances.append(0.0)
            else:
                try:
                    dist = geodesic(
                        (lat, lon),
                        self.coastal_reference_point
                    ).kilometers
                    distances.append(dist)
                except Exception:
                    distances.append(0.0)
        
        return pd.Series(distances, index=latitudes.index)
    
    def validate_features(self, features: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate that all 13 features are present and valid.
        
        Args:
            features: DataFrame with extracted features
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check all features are present
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            return False, f"Missing features: {missing_features}"
        
        # Check for correct number of features
        if len(features.columns) != 13:
            return False, f"Expected 13 features, got {len(features.columns)}"
        
        # Check for NaN values
        nan_counts = features.isna().sum()
        if nan_counts.sum() > 0:
            return False, f"NaN values found in features: {nan_counts[nan_counts > 0].to_dict()}"
        
        return True, "All 13 features validated successfully"
