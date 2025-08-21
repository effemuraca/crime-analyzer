"""
Custom transformers and utilities for crime hotspot clustering analysis.

This module provides specialized transformers for spatial, temporal, and categorical features
used in clustering-based crime hotspot detection. All transformers follow scikit-learn
conventions and can be used in Pipeline objects.

Key Transformers:
-----------------
- CyclicalTransformer: Convert temporal features (HOUR, WEEKDAY, MONTH, DAY) to sin/cos components
- SpatialProjectionTransformer: Project lat/lon coordinates to meters (essential for DBSCAN/HDBSCAN)
- LatLonToRadiansTransformer: Convert lat/lon to radians for angular distance calculations
- FeatureSelector: Select and reorder features for consistent clustering input
- SchemaValidator: Validate temporal column ranges before processing
- PCATransformer: Dimensionality reduction wrapper with DataFrame output
"""

import logging
import warnings
from typing import Optional, Union, List, Tuple, Any, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA
from pyproj import Transformer
import numpy as np
import pandas as pd

# Set up logging for performance monitoring
logger = logging.getLogger(__name__)

def cyclical_transform(X):
    """
    Encodes cyclical features using sine and cosine transforms.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input X must be a pandas DataFrame.")

    X_transformed = pd.DataFrame(index=X.index)
    feature_names_out = []

    weekday_order = ['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
    weekday_map = {d: i for i, d in enumerate(weekday_order)}

    for col in X.columns:
        col_out_sin = f'{col}_SIN'
        col_out_cos = f'{col}_COS'
        feature_names_out += [col_out_sin, col_out_cos]

        if col == 'HOUR':
            num_val = pd.to_numeric(X[col], errors='raise')
            period = 24.0
            valid_range = (0, period - 1)
        elif col == 'WEEKDAY':
            if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col].dtype):
                num_val = X[col].map(weekday_map)
                if num_val.isnull().any():
                    bad = X[col][num_val.isnull()].unique()
                    warnings.warn(f"WEEKDAY contains invalid values: {bad}. These will be dropped.")
                    # Skip invalid weekdays
                    continue
            else:
                num_val = pd.to_numeric(X[col], errors='raise')
            period = 7.0
            valid_range = (0, period - 1)
        elif col == 'MONTH':
            num_val = pd.to_numeric(X[col], errors='raise')
            period = 12.0
            valid_range = (1, period)
        elif col == 'DAY':
            num_val = pd.to_numeric(X[col], errors='raise')
            period = X['MONTH'].map({
                2: 29.0,
                4: 30.0, 6: 30.0, 9: 30.0, 11: 30.0,
                1: 31.0, 3: 31.0, 5: 31.0, 7: 31.0, 8: 31.0, 10: 31.0, 12: 31.0
            })
            invalid = ~((num_val >= 1) & (num_val <= period))
            if invalid.any():
                bad_rows = X.loc[invalid, ['MONTH', 'DAY']]
                raise ValueError(
                    f"Invalid DAY values found for some MONTHs:\n{bad_rows}"
                )
            # valid_range non serve più qui
        else:
            raise ValueError(f"Unknown cyclical column: {col}")

        if col != 'DAY':  # Per DAY il controllo è già stato fatto sopra
            if not ((num_val >= valid_range[0]) & (num_val <= valid_range[1])).all():
                raise ValueError(
                    f"Values of '{col}' must be in [{valid_range[0]}, {valid_range[1]}], "
                    f"found {num_val.min()}–{num_val.max()}"
                )

        X_transformed[col_out_sin] = np.sin(2 * np.pi * num_val / period)
        X_transformed[col_out_cos] = np.cos(2 * np.pi * num_val / period)

    return X_transformed

class CyclicalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self
    def transform(self, X, y=None):
        return cyclical_transform(X)
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        out = []
        for col in input_features:
            out += [f"{col}_SIN", f"{col}_COS"]
        return out

class SpatialProjectionTransformer(BaseEstimator, TransformerMixin):
    """
    Transform latitude/longitude coordinates to projected coordinates in meters.
    
    This transformer is essential for DBSCAN and HDBSCAN clustering as these algorithms
    use Euclidean distance. Geographic coordinates (lat/lon) have non-uniform
    distances when using Euclidean calculations, so we need to project them
    to a metric coordinate system.
    
    Parameters:
    -----------
    crs : str, default="EPSG:32618"
        Target coordinate reference system. Default is UTM 18N for NYC.
        Alternative: "EPSG:3857" (Web Mercator) as fallback.
    lat_col : str, default="Latitude"
        Name of the latitude column.
    lon_col : str, default="Longitude" 
        Name of the longitude column.
    out_cols : tuple, default=("X_METERS", "Y_METERS")
        Names for the output projected coordinate columns.
    drop_latlon : bool, default=False
        Whether to drop the original lat/lon columns from output.
    """
    def __init__(self, crs="EPSG:32618", lat_col="Latitude", lon_col="Longitude",
                 out_cols=("X_METERS","Y_METERS"), drop_latlon=False):
        self.crs = crs
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.out_cols = out_cols
        self.drop_latlon = drop_latlon
        
    def fit(self, X, y=None):
        """Fit the transformer by setting up the coordinate transformation."""
        self._transformer_ = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self
        
    def transform(self, X, y=None):
        """Transform lat/lon coordinates to projected meters."""
        check_is_fitted(self, "_transformer_")
        
        if self.lat_col not in X.columns or self.lon_col not in X.columns:
            raise ValueError(f"Required columns {self.lat_col} and {self.lon_col} not found in input data")
            
        # Project coordinates using pyproj (extracted from STKDE logic)
        x, y_m = self._transformer_.transform(
            X[self.lon_col].to_numpy(), 
            X[self.lat_col].to_numpy()
        )
        
        Xo = X.copy()
        Xo[self.out_cols[0]] = x
        Xo[self.out_cols[1]] = y_m
        
        if self.drop_latlon:
            Xo = Xo.drop(columns=[self.lat_col, self.lon_col])
            
        return Xo
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_', [])
        
        output_features = list(input_features)
        
        if self.drop_latlon:
            # Remove lat/lon columns if dropped
            output_features = [f for f in output_features if f not in [self.lat_col, self.lon_col]]
            
        # Add projected coordinate columns
        output_features.extend(self.out_cols)
        
        return np.array(output_features)


class LatLonToRadiansTransformer(BaseEstimator, TransformerMixin):
    """
    Convert latitude and longitude columns from degrees to radians.
    
    Useful for algorithms that work with angular coordinates or 
    when computing great circle distances.
    
    Parameters:
    -----------
    lat_col : str, default="Latitude"
        Name of the latitude column.
    lon_col : str, default="Longitude"
        Name of the longitude column.
    out_cols : tuple, default=("LAT_RADIANS", "LON_RADIANS")
        Names for the output radian columns.
    drop_original : bool, default=False
        Whether to drop the original degree columns.
    """
    def __init__(self, lat_col="Latitude", lon_col="Longitude", 
                 out_cols=("LAT_RADIANS", "LON_RADIANS"), drop_original=False):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.out_cols = out_cols
        self.drop_original = drop_original
        
    def fit(self, X, y=None):
        """Fit the transformer (no-op)."""
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self
        
    def transform(self, X, y=None):
        """Transform degrees to radians."""
        if self.lat_col not in X.columns or self.lon_col not in X.columns:
            raise ValueError(f"Required columns {self.lat_col} and {self.lon_col} not found in input data")
            
        Xo = X.copy()
        Xo[self.out_cols[0]] = np.deg2rad(X[self.lat_col])
        Xo[self.out_cols[1]] = np.deg2rad(X[self.lon_col])
        
        if self.drop_original:
            Xo = Xo.drop(columns=[self.lat_col, self.lon_col])
            
        return Xo
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_', [])
        
        output_features = list(input_features)
        
        if self.drop_original:
            output_features = [f for f in output_features if f not in [self.lat_col, self.lon_col]]
            
        output_features.extend(self.out_cols)
        return np.array(output_features)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Simple feature selector that keeps only specified columns in a fixed order.
    
    Useful to ensure consistent feature ordering for clustering algorithms
    and to select only relevant features for the clustering task.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names to select and their desired order.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        """Fit the selector (no-op)."""
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self
        
    def transform(self, X, y=None):
        """Select and reorder features."""
        missing_cols = [col for col in self.feature_names if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return X[self.feature_names].copy()
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return np.array(self.feature_names)


class SchemaValidator(BaseEstimator, TransformerMixin):
    """
    Lightweight validator for temporal columns ranges.
    
    Validates that HOUR, WEEKDAY, MONTH, DAY are in expected ranges
    before applying cyclical transformations.
    
    Parameters:
    -----------
    strict : bool, default=True
        If True, raises errors for invalid values. If False, issues warnings.
    """
    def __init__(self, strict=True):
        self.strict = strict
        
    def fit(self, X, y=None):
        """Fit the validator (no-op)."""
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self
        
    def transform(self, X, y=None):
        """Validate column ranges."""
        X_validated = X.copy()
        
        # Define validation rules
        validation_rules = {
            'HOUR': (0, 23, int),
            'MONTH': (1, 12, int),
            'DAY': (1, 31, int),  # Simplified - could be more complex with month-specific validation
        }
        
        weekday_valid = ['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
        
        for col in X.columns:
            if col in validation_rules:
                min_val, max_val, dtype = validation_rules[col]
                try:
                    numeric_col = pd.to_numeric(X[col], errors='coerce')
                    invalid_mask = (numeric_col < min_val) | (numeric_col > max_val) | numeric_col.isna()
                    
                    if invalid_mask.any():
                        invalid_values = X.loc[invalid_mask, col].unique()
                        message = f"Column '{col}' contains invalid values: {invalid_values}. Expected range: [{min_val}, {max_val}]"
                        
                        if self.strict:
                            raise ValueError(message)
                        else:
                            warnings.warn(message)
                            
                except Exception as e:
                    message = f"Error validating column '{col}': {e}"
                    if self.strict:
                        raise ValueError(message)
                    else:
                        warnings.warn(message)
                        
            elif col == 'WEEKDAY':
                if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col].dtype):
                    invalid_mask = ~X[col].isin(weekday_valid)
                    
                    if invalid_mask.any():
                        invalid_values = X.loc[invalid_mask, col].unique()
                        message = f"Column 'WEEKDAY' contains invalid values: {invalid_values}. Expected: {weekday_valid}"
                        
                        if self.strict:
                            raise ValueError(message)
                        else:
                            warnings.warn(message)
                            
        return X_validated
        
    def get_feature_names_out(self, input_features=None):
        """Get output feature names (unchanged)."""
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_', [])
        return np.array(input_features)


class PCATransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper around sklearn PCA for consistent interface.
    
    Useful for dimensionality reduction before clustering,
    especially for high-dimensional categorical data after encoding.
    
    Parameters:
    -----------
    n_components : int, float or None, default=None
        Number of components to keep. If None, keep all components.
    **pca_kwargs : dict
        Additional keyword arguments for sklearn PCA.
    """
    def __init__(self, n_components=None, **pca_kwargs):
        self.n_components = n_components
        self.pca_kwargs = pca_kwargs
        
    def fit(self, X, y=None):
        """Fit PCA on the data."""
        self.pca_ = PCA(n_components=self.n_components, **self.pca_kwargs)
        self.pca_.fit(X)
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
            
        return self
        
    def transform(self, X, y=None):
        """Apply PCA transformation."""
        check_is_fitted(self, 'pca_')
        X_transformed = self.pca_.transform(X)
        
        # Return as DataFrame with meaningful column names
        n_components = X_transformed.shape[1]
        column_names = [f'PC{i+1}' for i in range(n_components)]
        
        if hasattr(X, 'index'):
            return pd.DataFrame(X_transformed, index=X.index, columns=column_names)
        else:
            return pd.DataFrame(X_transformed, columns=column_names)
            
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self, 'pca_')
        n_components = self.pca_.n_components_
        return np.array([f'PC{i+1}' for i in range(n_components)])
        
    @property
    def explained_variance_ratio_(self):
        """Access to PCA explained variance ratio."""
        check_is_fitted(self, 'pca_')
        return self.pca_.explained_variance_ratio_
