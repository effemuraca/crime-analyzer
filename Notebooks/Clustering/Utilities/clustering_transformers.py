import logging
import warnings
from typing import Dict
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
            # Treat object or pandas Categorical dtype as categorical weekday labels
            if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype):
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
            
        # Project coordinates using pyproj
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
                # Treat object or pandas Categorical dtype as categorical weekday labels
                if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype):
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


class CategoricalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for categorical features before K-Modes clustering.
    
    Handles missing values, data type conversion, and feature validation
    specifically for categorical data used in K-Modes clustering.
    """
    def __init__(self, handle_missing='drop', min_frequency=None):
        self.handle_missing = handle_missing
        self.min_frequency = min_frequency
        
    def fit(self, X, y=None):
        """Fit the preprocessor by collecting feature statistics."""
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        
        # Store category mappings and frequency filters if needed
        self.category_stats_ = {}
        for col in X.columns:
            self.category_stats_[col] = {
                'unique_count': X[col].nunique(),
                'value_counts': X[col].value_counts(),
                'most_frequent': X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown'
            }
        return self
        
    def transform(self, X, y=None):
        """Transform categorical features for K-Modes clustering.

        When handle_missing='drop', rows containing NaN in any processed column
        are removed to ensure valid categorical inputs.
        """
        X_processed = X.copy()

        # Handle missing values
        if self.handle_missing == 'drop':
            # Drop rows with any NaNs in the categorical columns
            X_processed = X_processed.dropna(axis=0, how='any')
        elif self.handle_missing == 'most_frequent':
            for col in X_processed.columns:
                most_frequent = self.category_stats_[col]['most_frequent']
                X_processed[col] = X_processed[col].fillna(most_frequent)

        # Ensure all values are strings (K-Modes requirement)
        for col in X_processed.columns:
            X_processed[col] = X_processed[col].astype(str)

        return X_processed
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_', [])
        return np.array(input_features)


class MCATransformer(BaseEstimator, TransformerMixin):
    """
    Multiple Correspondence Analysis transformer for categorical data.
    
    This transformer applies MCA (Multiple Correspondence Analysis) to categorical data,
    reducing dimensionality while preserving the relationships between categories.
    Compatible with sklearn pipelines.
    """
    
    def __init__(self, n_components=10, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.mca_ = None
        
    def fit(self, X, y=None):
        """Fit MCA on the categorical data."""
        try:
            import prince
        except ImportError:
            raise ImportError("prince library is required for MCA. Install with: pip install prince")
            
        X = self._validate_input(X)
        
        self.mca_ = prince.MCA(
            n_components=self.n_components, 
            random_state=self.random_state
        )
        self.mca_.fit(X)
        return self
    
    def transform(self, X):
        """Transform categorical data using fitted MCA."""
        if self.mca_ is None:
            raise ValueError("MCA transformer not fitted. Call fit() first.")
            
        X = self._validate_input(X)
        transformed = self.mca_.transform(X)
        
        # Return as DataFrame with proper column names
        column_names = [f'MCA_Component_{i+1}' for i in range(transformed.shape[1])]
        return pd.DataFrame(transformed, columns=column_names, index=X.index)
    
    def get_explained_variance(self):
        """Get explained variance ratio for each component."""
        if self.mca_ is None:
            return None
        return self.mca_.explained_inertia_
    
    def _validate_input(self, X):
        """Validate input data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        return X


class MixedFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for mixed categorical and numerical features for clustering.
    
    This transformer handles both categorical and numerical features, applying
    appropriate preprocessing to each type while maintaining pipeline compatibility.
    """
    
    def __init__(self, max_categorical_features=20, max_numerical_features=10, 
                 categorical_strategy='drop'):
        self.max_categorical_features = max_categorical_features
        self.max_numerical_features = max_numerical_features
        self.categorical_strategy = categorical_strategy
        self.categorical_features_ = []
        self.numerical_features_ = []
        self.categorical_preprocessor_ = None
        self.scaler_ = None
        self._category_mappings_: Dict[str, Dict[str, int]] = {}
        
    def fit(self, X, y=None):
        """Fit the mixed feature preprocessor."""
        from sklearn.preprocessing import StandardScaler
        
        X = self._validate_input(X)
        
        # Identify categorical and numerical features
        categorical_cols = []
        numerical_cols = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype):
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        # Limit features based on parameters
        self.categorical_features_ = categorical_cols[:self.max_categorical_features]
        self.numerical_features_ = numerical_cols[:self.max_numerical_features]
        
        # Fit categorical preprocessor if we have categorical features
        if self.categorical_features_:
            self.categorical_preprocessor_ = CategoricalPreprocessor(
                handle_missing=self.categorical_strategy
            )
            X_cat = self.categorical_preprocessor_.fit_transform(X[self.categorical_features_])
            # Build stable mappings from the fitted data
            self._category_mappings_ = {}
            for col in X_cat.columns:
                # Preserve order of appearance for stability
                seen = []
                mapping: Dict[str, int] = {}
                for val in X_cat[col].astype(str).tolist():
                    if val not in mapping:
                        mapping[val] = len(mapping)
                        seen.append(val)
                self._category_mappings_[col] = mapping
        
        # Fit numerical scaler if we have numerical features
        if self.numerical_features_:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X[self.numerical_features_].fillna(0))  # Simple fillna for numerical
        
        return self
    
    def transform(self, X):
        """Transform mixed features."""
        X = self._validate_input(X)
        
        transformed_parts = []
        
        # Process categorical features
        if self.categorical_features_ and self.categorical_preprocessor_:
            cat_transformed = self.categorical_preprocessor_.transform(X[self.categorical_features_])
            # Encode using stable mappings from fit; unseen -> -1
            cat_encoded = pd.DataFrame(index=cat_transformed.index)

            for col in cat_transformed.columns:
                mapping = self._category_mappings_.get(col, {})
                cat_encoded[f'{col}_encoded'] = cat_transformed[col].astype(str).map(mapping).fillna(-1).astype(int)

            transformed_parts.append(cat_encoded)
        
        # Process numerical features
        if self.numerical_features_ and self.scaler_:
            num_data = X[self.numerical_features_].fillna(0)
            num_scaled = self.scaler_.transform(num_data)
            num_df = pd.DataFrame(num_scaled, 
                                columns=[f'{col}_scaled' for col in self.numerical_features_],
                                index=X.index)
            transformed_parts.append(num_df)
        
        # Combine all parts
        if transformed_parts:
            result = pd.concat(transformed_parts, axis=1)
            return result
        else:
            raise ValueError("No features available for transformation")
    
    def _validate_input(self, X):
        """Validate input data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        return X


class CategoricalDimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Categorical Dimensionality Reduction transformer using OneHot + PCA.
    
    This transformer provides a robust alternative to MCA for categorical data
    dimensionality reduction. It uses OneHot encoding followed by PCA, which
    is more numerically stable and doesn't produce NaN values.
    """
    
    def __init__(self, n_components=5, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.encoder_ = None
        self.pca_ = None
        
    def fit(self, X, y=None):
        """Fit encoder and PCA."""        
        X = self._validate_input(X)
        
        # Fit one-hot encoder
        from sklearn.preprocessing import OneHotEncoder
        # Support sklearn<1.2 (uses 'sparse' instead of 'sparse_output')
        try:
            self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            self.encoder_ = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_encoded = self.encoder_.fit_transform(X)
        
        # Fit PCA
        from sklearn.decomposition import PCA
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_.fit(X_encoded)
        
        return self
    
    def transform(self, X):
        """Transform using fitted encoder and PCA."""
        if self.encoder_ is None or self.pca_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
            
        X = self._validate_input(X)
        
        # One-hot encode
        X_encoded = self.encoder_.transform(X)
        
        # Apply PCA
        X_pca = self.pca_.transform(X_encoded)
        
        # Return as DataFrame
        column_names = [f'CatDimRed_Component_{i+1}' for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=column_names, index=X.index)
    
    def get_explained_variance_ratio(self):
        """Get explained variance ratio for each component."""
        if self.pca_ is None:
            return None
        return self.pca_.explained_variance_ratio_
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self, ['encoder_', 'pca_'])
        return np.array([f'CatDimRed_Component_{i+1}' for i in range(self.n_components)])
    
    def _validate_input(self, X):
        """Validate input data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        return X

# Simple identity preprocessor for categorical data
class IdentityPreprocessor(BaseEstimator, TransformerMixin):
    """Simple identity preprocessor that just passes data through."""
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.copy()


class ColumnBinner(BaseEstimator, TransformerMixin):
    """
    Generic column binning transformer for creating categorical bins from numeric features.

    Supports multiple strategies per column:
    - kind='distance': domain cut with optional bins/labels, fallback to 3-quantile ['Near','Mid','Far']
    - kind='count'   : quantile bins with zero-aware handling (maps all-zero to 'Zero')
    - kind='score'   : generic quantile bins
    - kind='quantile': explicit quantile-based binning
    - kind='cut'     : explicit pd.cut with bins/labels
    """

    def __init__(self, config: Dict[str, Dict], suffix: str = "_BIN", fill_unknown: str = "Unknown"):
        self.config = config
        self.suffix = suffix
        self.fill_unknown = fill_unknown

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        # No statistics needed; keep track of what we will (likely) create
        self._planned_cols_ = [f"{col}{self.suffix}" for col in self.config.keys() if col in getattr(self, 'feature_names_in_', [])]
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        Xo = X.copy()
        created = []

        for col, cfg in self.config.items():
            if col not in Xo.columns:
                continue
            kind = (cfg.get('kind') or '').lower()
            out_col = f"{col}{self.suffix}"

            s = pd.to_numeric(Xo[col], errors='coerce')

            # If all values are NaN, fill with Unknown
            if s.notna().sum() == 0:
                Xo[out_col] = self.fill_unknown
                created.append(out_col)
                continue

            try:
                if kind == 'distance':
                    bins = cfg.get('bins')
                    labels = cfg.get('labels', ['Near', 'Mid', 'Far'])
                    if bins is not None:
                        binned = pd.cut(s, bins=bins, labels=labels)
                        Xo[out_col] = binned.astype(str).fillna(self.fill_unknown)
                    else:
                        # Fallback: 3-quantile using rank for robustness
                        qlabels = labels if labels else ['Near', 'Mid', 'Far']
                        q = int(cfg.get('quantiles', 3))
                        binned = pd.qcut(s.rank(method='first'), q, labels=qlabels[:q])
                        Xo[out_col] = binned.astype(str).fillna(self.fill_unknown)

                elif kind == 'count':
                    # Special case: all zeros -> zero_label
                    zero_label = cfg.get('zero_label', 'Zero')
                    if (s.fillna(0) == 0).all():
                        Xo[out_col] = zero_label
                    else:
                        q = int(cfg.get('quantiles', 4))
                        labels = cfg.get('labels', ['Low', 'Medium', 'High', 'VeryHigh'])
                        # Align labels length with q if needed
                        if len(labels) < q:
                            # pad labels
                            labels = labels + [labels[-1]] * (q - len(labels))
                        binned = pd.qcut(s.rank(method='first'), q, labels=labels[:q])
                        Xo[out_col] = binned.astype(str).fillna(self.fill_unknown)

                elif kind == 'score':
                    q = int(cfg.get('quantiles', 4))
                    labels = cfg.get('labels', [f'Q{i+1}' for i in range(q)])
                    if len(labels) < q:
                        labels = labels + [labels[-1]] * (q - len(labels))
                    binned = pd.qcut(s.rank(method='first'), q, labels=labels[:q])
                    Xo[out_col] = binned.astype(str).fillna(self.fill_unknown)

                elif kind == 'quantile':
                    q = int(cfg.get('q', 4))
                    labels = cfg.get('labels', [f'Q{i+1}' for i in range(q)])
                    if len(labels) < q:
                        labels = labels + [labels[-1]] * (q - len(labels))
                    binned = pd.qcut(s.rank(method='first'), q, labels=labels[:q])
                    Xo[out_col] = binned.astype(str).fillna(self.fill_unknown)

                elif kind == 'cut':
                    bins = cfg.get('bins')
                    labels = cfg.get('labels')
                    if bins is None or labels is None:
                        raise ValueError("'cut' kind requires 'bins' and 'labels'")
                    binned = pd.cut(s, bins=bins, labels=labels)
                    Xo[out_col] = binned.astype(str).fillna(self.fill_unknown)

                else:
                    # Unknown kind: pass-through as Unknown
                    Xo[out_col] = self.fill_unknown

                created.append(out_col)

            except Exception:
                # On any failure, mark as Unknown
                Xo[out_col] = self.fill_unknown
                created.append(out_col)

        # Track created feature names for downstream use
        self.created_bins_ = created
        return Xo

    def get_feature_names_out(self, input_features=None):
        """Get output feature names (created bins)."""
        return np.array(getattr(self, 'created_bins_', []))