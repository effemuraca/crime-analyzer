import logging
import warnings
from typing import Optional, Union, List, Tuple, Any, Dict
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import BallTree
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KDTree
from pyproj import Transformer
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
import jenkspy

# Set up logging for performance monitoring
logger = logging.getLogger(__name__)

def apply_fixed_thresholds_to_intensities(intensities: Union[pd.Series, np.ndarray], 
                                          thresholds: Union[List[float], Tuple[float, ...]]) -> np.ndarray:
    """
    Applies fixed thresholds to intensity values to create labels.
    Labels are 0, 1, 2, ..., len(thresholds).
    
    Args:
        intensities: The STKDE intensity values.
        thresholds: A list of N-1 threshold values, sorted ascending,
                   to create N classes.
                   

    Returns:
        The generated integer labels.
        
    Raises:
        ValueError: If thresholds are invalid or not sorted.
    """
    if not isinstance(thresholds, (list, tuple, np.ndarray)) or not all(isinstance(t, (int, float, np.integer, np.floating)) for t in np.array(thresholds).flatten()):
        raise ValueError("Thresholds must be a list, tuple, or numpy array of numbers.")
    # Allow empty thresholds list for the case where no breaks are found or n_classes=1
    thresholds_arr = np.array(thresholds).flatten()
    if thresholds_arr.size > 1 and np.any(thresholds_arr[:-1] > thresholds_arr[1:]):
        raise ValueError("Thresholds must be sorted in non-decreasing order.")
  
    # If thresholds is empty, all intensities will be in class 0.
    return np.digitize(intensities, bins=list(thresholds), right=True)


class BinarizeSinCosTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that binarizes sin/cos cyclical features based on a threshold.
    
    Parameters:
        threshold: Threshold value for binarization (default: 0.0).
    """
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'BinarizeSinCosTransformer':
        """
        Fit the transformer and store feature names.
        
        Args:
            X: Input DataFrame.
            y: Target values (ignored).
            
        Returns:
            Self for method chaining.
        """
        # Store feature names if X is a DataFrame, for get_feature_names_out
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        # No fitted state required; simply binarize sin/cos columns
        X_transformed = X.copy()
        # The input X to this transformer will be the output of the ColumnTransformer's 'cyc_passthrough' part
        # which contains the sin/cos transformed columns.
        # Only binarize sin/cos columns
        for col in X_transformed.columns:
            if '_SIN' in col or '_COS' in col:
                X_transformed[col] = (X_transformed[col] > self.threshold).astype(int)
        return X_transformed

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
            else:
                # Fallback for when fit wasn't called on a DataFrame
                raise ValueError("Cannot determine output feature names without input_features.")
        # The transformer modifies columns in place, so names don't change.
        return list(input_features)
    
class STKDEAndRiskLabelTransformer(BaseEstimator, TransformerMixin):
    """
    Computes STKDE intensity and derives a risk label based on dynamic or fixed thresholds.
    Supports multiple threshold strategies to prevent data leakage during cross-validation:
    - 'fixed': Use predefined thresholds (may cause data leakage if computed from full dataset)
    - 'dynamic_jenks': Calculate Jenks natural breaks on training fold intensities
    - 'dynamic_quantile': Use quantile-based thresholds on training fold intensities  
    - 'dynamic_median': Use median-based threshold on training fold intensities
    """
    
    def __init__(self, year_col: str = 'YEAR', month_col: str = 'MONTH', 
                        day_col: str = 'DAY', hour_col: str = 'HOUR',
                        lat_col: str = 'Latitude', lon_col: str = 'Longitude', 
                        hs: float = 200.0, ht: float = 60.0,
                        fixed_thresholds: Optional[List[float]] = None,
                        threshold_strategy: str = 'dynamic_jenks',
                        n_classes: int = 2, n_jobs: int = -1, random_state: int = 42,
                        intensity_col_name: str = 'stkde_intensity_engineered',
                        label_col_name: str = 'RISK_LEVEL_engineered') -> None:
        # Validate bandwidth parameters
        if hs <= 0:
            raise ValueError("Spatial bandwidth 'hs' must be positive.")
        if ht <= 0:
            raise ValueError("Temporal bandwidth (ht) must be positive.")
            
        self.year_col = year_col
        self.month_col = month_col
        self.day_col = day_col
        self.hour_col = hour_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.hs = hs
        self.ht = ht
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.intensity_col_name = intensity_col_name
        self.label_col_name = label_col_name
        self.threshold_strategy = threshold_strategy
        self.n_classes = n_classes
        self.n_classes_init_ = n_classes
        self.n_classes_eff_ = n_classes

        # Validate threshold strategy and fixed_thresholds
        if threshold_strategy == 'fixed':
            if fixed_thresholds is None:
                raise ValueError("fixed_thresholds must be provided when threshold_strategy is 'fixed'.")
            if not isinstance(fixed_thresholds, (list, tuple)) or not all(isinstance(t, (int, float)) for t in fixed_thresholds):
                raise ValueError("fixed_thresholds must be a list or tuple of numbers.")
            if len(fixed_thresholds) != n_classes - 1:
                raise ValueError(f"For {n_classes} classes, expected {n_classes - 1} thresholds, but got {len(fixed_thresholds)}.")
        elif threshold_strategy not in ['dynamic_jenks', 'dynamic_quantile', 'dynamic_median']:
            raise ValueError(f"Invalid threshold_strategy: {threshold_strategy}")
        
        if threshold_strategy != 'fixed' and fixed_thresholds is not None:
            warnings.warn("fixed_thresholds is ignored when threshold_strategy is not 'fixed'.")
        
        self.fixed_thresholds = fixed_thresholds
        self.calculated_thresholds_: Optional[np.ndarray] = None

        # Attributes to be learned during fit
        self.train_reference_coords_rad_: Optional[np.ndarray] = None
        self.train_reference_datetime_np_: Optional[np.ndarray] = None
        self.t0_: Optional[pd.Timestamp] = None # Add t0_ to store the reference time
        self.fitted_ = False

    def _calculate_dynamic_thresholds(self, intensities: np.ndarray) -> np.ndarray:
        """
        Calculate dynamic thresholds based on the specified strategy.
        """
        if self.threshold_strategy == 'dynamic_jenks':
            try:
                import jenkspy
                # n_classes-1 thresholds are needed. jenks_breaks returns n_classes+1 values (min, thresholds..., max)
                breaks = jenkspy.jenks_breaks(intensities, n_classes=self.n_classes_init_)
                return np.array(breaks[1:-1])
            except ImportError:
                warnings.warn("jenkspy library not found. Please install it for 'dynamic_jenks' strategy. Falling back to 'dynamic_quantile'.")
                return self._calculate_quantile_thresholds(intensities)
        elif self.threshold_strategy == 'dynamic_quantile':
            return self._calculate_quantile_thresholds(intensities)
        elif self.threshold_strategy == 'dynamic_median':
            return self._calculate_median_thresholds(intensities)
        else: # Should not be reached due to __init__ validation
            raise ValueError(f"Invalid threshold strategy: {self.threshold_strategy}")
    
    def _calculate_quantile_thresholds(self, intensities: np.ndarray) -> np.ndarray:
        """Calculate quantile-based thresholds."""
        quantiles = np.linspace(0, 1, self.n_classes_init_ + 1)[1:-1]  # Exclude 0 and 1
        return np.quantile(intensities, quantiles)
    
    def _calculate_median_thresholds(self, intensities: np.ndarray) -> np.ndarray:
        """Calculate median-based threshold (only works for 2 classes)."""
        if self.n_classes_init_ != 2:
            raise ValueError("Median thresholding is only supported for n_classes=2.")
        return np.array([np.median(intensities)])

    def _prepare_data_for_stkde(self, X_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """Helper to prepare datetime and coordinates from an input DataFrame X_df."""
        df_copy = X_df.copy()
        temp_datetime_col = '__temp_stkde_datetime_internal__'
        
        required_cols = [self.year_col, self.month_col, self.day_col, self.hour_col, self.lat_col, self.lon_col]
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for STKDE: {missing_cols}")

        # Coerce date/time parts to numeric, forcing errors to NaN
        for col in [self.year_col, self.month_col, self.day_col, self.hour_col]:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        df_copy[temp_datetime_col] = pd.to_datetime(
            df_copy[[self.year_col, self.month_col, self.day_col, self.hour_col]].rename(
                columns={self.year_col:'year', self.month_col:'month', self.day_col:'day', self.hour_col:'hour'}
            ),
            errors='coerce'
        )
        
        df_copy[self.lat_col] = pd.to_numeric(df_copy[self.lat_col], errors='coerce')
        df_copy[self.lon_col] = pd.to_numeric(df_copy[self.lon_col], errors='coerce')

        df_copy.dropna(subset=[temp_datetime_col, self.lat_col, self.lon_col], inplace=True)
        
        if df_copy.empty:
            return df_copy, None, None

        coords = df_copy[[self.lat_col, self.lon_col]].values.astype(float)
        coords_rad = np.deg2rad(coords)
        datetime_np = df_copy[temp_datetime_col].values
        
        return df_copy, coords_rad, datetime_np

    def _calculate_stkde_for_set(self, target_coords_rad: Optional[np.ndarray], 
                                 target_datetime_np: Optional[np.ndarray], 
                                 reference_coords_rad: Optional[np.ndarray], 
                                 reference_datetime_np: Optional[np.ndarray],
                                 t0: pd.Timestamp,
                                 is_self_calculation: bool = False) -> np.ndarray:
        """
        Calculates STKDE for target points using reference points.
        """
        # Early exit if no data
        if target_coords_rad is None or reference_coords_rad is None or \
           target_coords_rad.size == 0 or reference_coords_rad.size == 0:
            return np.array([])

        # Convert radians to degrees for projection
        target_coords_deg = np.rad2deg(target_coords_rad)
        reference_coords_deg = np.rad2deg(reference_coords_rad)

        # Project to meters (EPSG:3857)
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        target_x, target_y = transformer.transform(target_coords_deg[:,1], target_coords_deg[:,0])
        ref_x, ref_y = transformer.transform(reference_coords_deg[:,1], reference_coords_deg[:,0])
        target_xy = np.column_stack([target_x, target_y])
        ref_xy = np.column_stack([ref_x, ref_y])

        # Compute relative time in days (relative to the provided t0)
        ref_t_days = (pd.to_datetime(reference_datetime_np) - t0) / np.timedelta64(1, 'D')
        target_t_days = (pd.to_datetime(target_datetime_np) - t0) / np.timedelta64(1, 'D')

        # Build KDTree on reference
        tree = KDTree(ref_xy, metric='euclidean')
        hs, ht = self.hs, self.ht

        # Pre-calculate constants for kernels
        hs_squared = hs**2
        spatial_norm = 1 / (2 * np.pi * hs_squared)
        temporal_norm = 1 / (2 * ht)

        def process_event(i):
            # Query neighbors in space and get distances
            neigh_idx, neigh_dist = tree.query_radius(target_xy[i:i+1], r=5*hs, return_distance=True)
            neigh_idx, neigh_dist = neigh_idx[0], neigh_dist[0]

            if neigh_idx.size == 0:
                return 1e-12

            # Calculate temporal differences
            dt = np.abs(target_t_days[i] - ref_t_days[neigh_idx])
            
            # Exclude self-comparison if applicable
            if is_self_calculation:
                # In self-calculation, the i-th target point corresponds to the i-th reference point.
                # We need to find where neigh_idx is equal to i and remove it.
                self_mask = (neigh_idx != i)
                neigh_idx = neigh_idx[self_mask]
                neigh_dist = neigh_dist[self_mask]
                dt = dt[self_mask]

            # Filter by temporal bandwidth
            time_mask = (dt <= 5 * ht)
            if not np.any(time_mask):
                return 1e-12
            
            valid_dist = neigh_dist[time_mask]
            valid_dt = dt[time_mask]

            # Vectorized kernel calculation
            spatial_kernel_vals = spatial_norm * np.exp(-0.5 * (valid_dist / hs)**2)
            temporal_kernel_vals = temporal_norm * np.exp(-valid_dt / ht)
            
            s = np.sum(spatial_kernel_vals * temporal_kernel_vals)
            return max(s, 1e-12)

        n = len(target_xy)
        try:
            intensities = Parallel(n_jobs=self.n_jobs)(
                delayed(process_event)(i) for i in range(n)
            )
            return np.array(intensities)
        except Exception as e:
            warnings.warn(f"Parallel STKDE error: {e}")
            return np.full(n, np.nan)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'STKDEAndRiskLabelTransformer':
        _X = X.copy()
        if isinstance(_X, pd.DataFrame):
            self.feature_names_in_ = _X.columns.tolist()

        X_prepared_df, self.train_reference_coords_rad_, self.train_reference_datetime_np_ = self._prepare_data_for_stkde(_X)

        if self.train_reference_coords_rad_ is None or self.train_reference_coords_rad_.size == 0:
            warnings.warn("No valid data points found in training data for STKDE. The transformer will not be fitted.")
            self.fitted_ = True
            return self

        # Store the reference time t0 from the training set
        self.t0_ = pd.to_datetime(self.train_reference_datetime_np_).min()
        
        stkde_intensities_train = self._calculate_stkde_for_set(
            self.train_reference_coords_rad_, self.train_reference_datetime_np_,
            self.train_reference_coords_rad_, self.train_reference_datetime_np_,
            t0=self.t0_,
            is_self_calculation=True
        )

        if stkde_intensities_train.size == 0 or np.all(stkde_intensities_train == 0):
             warnings.warn("STKDE intensities for training data are all zero. Thresholds may not be meaningful.")

        if self.threshold_strategy == 'fixed':
            self.calculated_thresholds_ = np.array(self.fixed_thresholds)
        else:
            self.calculated_thresholds_ = self._calculate_dynamic_thresholds(stkde_intensities_train)
            
        self.n_classes_eff_ = len(self.calculated_thresholds_) + 1
        self.n_classes_ = self.n_classes_eff_  # Add this line for sklearn compatibility
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.Series]:
        check_is_fitted(self)
        _X = X.copy()

        X_prepared_df, target_coords_rad, target_datetime_np = self._prepare_data_for_stkde(_X)

        if target_coords_rad is None or target_coords_rad.size == 0:
            X_out = _X.copy()
            X_out[self.intensity_col_name] = np.nan
            X_out[self.label_col_name] = np.nan
            return X_out, X_out[self.label_col_name]

        stkde_intensities = self._calculate_stkde_for_set(
            target_coords_rad, target_datetime_np,
            self.train_reference_coords_rad_, self.train_reference_datetime_np_,
            t0=self.t0_ # Use the t0 from fitting
        )
        
        X_out = X_prepared_df.copy()
        X_out[self.intensity_col_name] = stkde_intensities
        
        labels = apply_fixed_thresholds_to_intensities(stkde_intensities, self.calculated_thresholds_)
        X_out[self.label_col_name] = labels
        y_out = pd.Series(labels, index=X_out.index, name=self.label_col_name)

        X_out.drop(columns=['__temp_stkde_datetime_internal__'], inplace=True, errors='ignore')
        
        return X_out, y_out

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and then transform, returning DataFrame and label series.

        Returns:
            X_out (pd.DataFrame): Transformed feature DataFrame including intensity and label columns.
            y_out (pd.Series): The engineered label series.
        """
        return self.fit(X, y).transform(X, y)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        check_is_fitted(self)
        if input_features is None:
            if not hasattr(self, 'feature_names_in_'):
                 raise ValueError("Input features not known. Please fit on a DataFrame or provide `input_features`.")
            input_features = self.feature_names_in_
        
        return input_features + [self.intensity_col_name, self.label_col_name]

class TargetEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    A pipeline that first applies a target engineering step (like STKDEAndRiskLabelTransformer)
    to generate new target-related features (and potentially the actual target for the model),
    and then applies a feature processing pipeline to the features *before* they were augmented
    by the target engineer, using the *newly generated target* for fitting the main classifier.
    
    Parameters:
        target_engineer: Transformer for target engineering. Must have a 'label_col_name' attribute
                         and 'fit_transform' method that returns a DataFrame containing this column.
        feature_pipeline: Pipeline for feature processing.
    """
    def __init__(self, target_engineer: TransformerMixin, feature_pipeline: Pipeline) -> None:
        self.target_engineer = target_engineer
        self.feature_pipeline = feature_pipeline
        self.y_engineered_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'TargetEngineeringPipeline':
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()

        X_target_engineered, y_engineered = self.target_engineer.fit_transform(X.copy(), y)
        
        if not hasattr(self.target_engineer, 'label_col_name'):
            raise AttributeError("target_engineer must have a 'label_col_name' attribute.")
        
        self.y_engineered_ = y_engineered

        X_for_fp_fit = X.loc[X_target_engineered.index]
        self.feature_pipeline.fit(X_for_fp_fit, self.y_engineered_)
        
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        # Note: This transform does not re-apply target engineering. It only applies the feature pipeline.
        # This is intentional, as the engineered target is only needed for fitting.
        return self.feature_pipeline.transform(X.copy())

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Fit and then transform, returning DataFrame and label series.
        Returns:
            X_out (pd.DataFrame): Transformed feature DataFrame including intensity and label columns.
            y_out (pd.Series): The engineered label series.
        """
        # Fit both target_engineer and feature_pipeline, then return transformed features
        self.fit(X, y)
        return self.transform(X)

    def get_target(self) -> Optional[pd.Series]:
        check_is_fitted(self)
        return self.y_engineered_

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        check_is_fitted(self)
        # The output features are those from the feature_pipeline
        return self.feature_pipeline.get_feature_names_out(input_features)

    # Ensure get_params and set_params work correctly for nested estimators
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self._get_params('target_engineer', 'feature_pipeline', deep=deep)

    def set_params(self, **params) -> 'TargetEngineeringPipeline':
        self._set_params('target_engineer', 'feature_pipeline', **params)
        return self

    def _get_params(self, *estimator_names: str, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=False)
        if deep:
            for name in estimator_names:
                estimator = getattr(self, name)
                if estimator is not None:
                    for key, value in estimator.get_params(deep=True).items():
                        params[f'{name}__{key}'] = value
        return params

    def _set_params(self, *estimator_names: str, **params) -> None:
        my_params = {}
        nested_params = {name: {} for name in estimator_names}
        
        for key, value in params.items():
            if '__' in key:
                estimator_name, param_name = key.split('__', 1)
                if estimator_name in nested_params:
                    nested_params[estimator_name][param_name] = value
                else:
                    my_params[key] = value # Or raise an error for unknown estimator
            else:
                my_params[key] = value
        
        # Set params for the main object
        super().set_params(**my_params)
        
        for name, p in nested_params.items():
            if hasattr(self, name) and getattr(self, name) is not None:
                getattr(self, name).set_params(**p)


class CustomModelPipeline(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """
    A complete modeling pipeline that combines STKDE transformation, feature processing, and classification.
    """
    def __init__(self, stkde_transformer: Optional[STKDEAndRiskLabelTransformer] = None, 
                 feature_processor: Optional[Pipeline] = None, 
                 classifier: Optional[ClassifierMixin] = None) -> None:
        self.stkde_transformer = stkde_transformer
        self.feature_processor = feature_processor
        self.classifier = classifier

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'CustomModelPipeline':
        """Fit the complete pipeline."""
        if self.stkde_transformer is None or self.feature_processor is None or self.classifier is None:
            raise ValueError("All components (stkde_transformer, feature_processor, classifier) must be provided")
        
        X_aug, y_engineered = self.stkde_transformer.fit_transform(X.copy(), y)
        X_processed = self.feature_processor.fit_transform(X_aug, y_engineered)
        self.classifier.fit(X_processed, y_engineered)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        X_aug, _ = self.stkde_transformer.transform(X.copy())
        X_processed = self.feature_processor.transform(X_aug)
        return self.classifier.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self)
        if not hasattr(self.classifier, 'predict_proba'):
            raise AttributeError("The classifier does not support probability prediction")
        X_aug, _ = self.stkde_transformer.transform(X.copy())
        X_processed = self.feature_processor.transform(X_aug)
        return self.classifier.predict_proba(X_processed)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = super().get_params(deep=False)
        if deep:
            for name in ['stkde_transformer', 'feature_processor', 'classifier']:
                estimator = getattr(self, name)
                if estimator is not None:
                    for key, value in estimator.get_params(deep=True).items():
                        params[f'{name}__{key}'] = value
        return params

    def set_params(self, **params) -> 'CustomModelPipeline':
        my_params = {}
        nested_params = {name: {} for name in ['stkde_transformer', 'feature_processor', 'classifier']}
        
        for key, value in params.items():
            if '__' in key:
                estimator_name, param_name = key.split('__', 1)
                if estimator_name in nested_params:
                    nested_params[estimator_name][param_name] = value
                else:
                    my_params[key] = value
            else:
                my_params[key] = value

        super().set_params(**my_params)
        
        for name, p in nested_params.items():
            if hasattr(self, name) and getattr(self, name) is not None:
                getattr(self, name).set_params(**p)
        return self

    def get_selected_features_mask(self) -> Optional[np.ndarray]:
        """Helper to get selected features if RFE is used in the classifier or feature_processor."""
        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_ > 0
        if hasattr(self.classifier, 'coef_'):
            return np.sum(np.abs(self.classifier.coef_), axis=0) > 0
        return None

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Access feature importances from the classifier if available."""
        return getattr(self.classifier, 'feature_importances_', None)

    @property
    def coef_(self) -> Optional[np.ndarray]:
        """Access coefficients from the classifier if available."""
        return getattr(self.classifier, 'coef_', None)

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
                    raise ValueError(f"WEEKDAY contains invalid values: {bad}")
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


class SlidingWindowSplit(BaseCrossValidator):
    """
    Cross-validator for time series data that uses a sliding window approach.
    
    In each split, the training set is a fixed-size window that "slides"
    forward in time, and the test set is the block of data immediately
    following the training set.
    
    This is useful for models where only recent history is relevant (concept drift).
    """
    def __init__(self, n_splits=5, train_size=None, test_size=None):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if self.train_size is None or self.test_size is None:
            raise ValueError("train_size and test_size must be specified")
        
        if self.train_size + self.test_size * self.n_splits > n_samples:
            raise ValueError("train_size and test_size are too large for the number of samples and splits.")
        
        # Start index of the first test fold
        start_test = n_samples - (self.n_splits * self.test_size)
        
        for i in range(self.n_splits):
            test_start_idx = start_test + i * self.test_size
            train_end_idx = test_start_idx
            train_start_idx = train_end_idx - self.train_size
            
            if train_start_idx < 0:
                raise ValueError(f"train_start_idx is negative ({train_start_idx}) on fold {i}. "
                                 "Reduce train_size, test_size, or n_splits.")

            yield (
                np.arange(train_start_idx, train_end_idx),
                np.arange(test_start_idx, test_start_idx + self.test_size)
            )