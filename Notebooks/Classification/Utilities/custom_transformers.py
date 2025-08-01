'''
Custom transformers and utility functions for the project.
'''
import logging # ADDED
import time # ADDED
import numpy as np # ADDED
import pandas as pd # ADDED
from typing import Optional, Union, List, Tuple, Any
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted # ADDED for feature names
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree # For STKDE
from joblib import Parallel, delayed # For STKDE
from sklearn.pipeline import Pipeline # For TargetEngineeringPipeline
import jenkspy # Make sure jenkspy is available
from sklearn.preprocessing import FunctionTransformer

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
    if not isinstance(thresholds, (list, tuple)) or not all(isinstance(t, (int, float)) for t in thresholds):
        raise ValueError("Thresholds must be a list or tuple of numbers.")
    # Allow empty thresholds list for the case where no breaks are found or n_classes=1
    if thresholds and any(thresholds[i] > thresholds[i+1] for i in range(len(thresholds)-1)):
        raise ValueError("Thresholds must be sorted in non-decreasing order.")
        
    # If thresholds is empty, all intensities will be in class 0.
    return np.digitize(intensities, bins=list(thresholds), right=True)

def cyclical_transform(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes cyclical features using sine and cosine transforms.
    
    Args:
        X: pandas DataFrame containing only the cyclical columns.
        
    Returns:
        DataFrame with sin/cos transformed cyclical features.
        
    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If a column in X is not 'HOUR', 'WEEKDAY', or 'MONTH'.
    """
    X_transformed = pd.DataFrame(index=X.index)
    feature_names_out = [] # To store new column names

    # Define mappings and periods
    weekday_order = ['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
    weekday_map = {d:i for i,d in enumerate(weekday_order)}
    
    for col in X.columns:
        col_out_sin = f'{col}_SIN'
        col_out_cos = f'{col}_COS'
        feature_names_out += [col_out_sin, col_out_cos]

        if col == 'HOUR':
            num_val = pd.to_numeric(X[col], errors='raise')
            period  = 24.0
            valid_range = (0, period-1)
        elif col == 'WEEKDAY':
            if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
                num_val = X[col].map(weekday_map)
                if num_val.isnull().any():
                    bad = X[col][num_val.isnull()].unique()
                    raise ValueError(f"WEEKDAY contains invalid values: {bad}")
            else:
                num_val = pd.to_numeric(X[col], errors='raise')
            period = 7.0
            valid_range = (0, period-1)
        elif col == 'MONTH':
            num_val = pd.to_numeric(X[col], errors='raise')
            period  = 12.0
            # If you ever get month names, map them here with a month_map.
            valid_range = (1, period)
        else:
            raise ValueError(f"Unknown cyclical column: {col}")

        # 2. Range check
        if not ((num_val >= valid_range[0]) & (num_val <= valid_range[1])).all():
            raise ValueError(
                f"Values of '{col}' must be in [{valid_range[0]}, {valid_range[1]}], "
                f"found {num_val.min()}–{num_val.max()}"
            )

        X_transformed[col_out_sin] = np.sin(2 * np.pi * num_val / period)
        X_transformed[col_out_cos] = np.cos(2 * np.pi * num_val / period)

    X_transformed.columns = feature_names_out  # type: ignore
    return X_transformed

class BinarizeSinCosTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that binarizes sin/cos cyclical features based on a threshold.
    
    Parameters:
        threshold: Threshold value for binarization (default: 0.0).
    """
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        # self.feature_names_in_: Optional[List[str]] = None # Handled by BaseEstimator

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
            self.feature_names_in_ = X.columns.tolist() # type: ignore
        # This is automatically handled by scikit-learn >= 0.24 if X is a DataFrame
        return self

    def transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        # No fitted state required; simply binarize sin/cos columns
        X_transformed = X.copy()
        # The input X to this transformer will be the output of the ColumnTransformer's 'cyc_passthrough' part
        # which contains the sin/cos transformed columns.
        # Only binarize sin/cos columns
        for col in X_transformed.columns:
            if col.endswith('_SIN') or col.endswith('_COS'):
                X_transformed[col] = (X_transformed[col] > self.threshold).astype(int)
        return X_transformed

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_ # type: ignore
            else:
                raise ValueError("Input feature names are not available. Fit the transformer on a DataFrame first or provide input_features.")
        # The transformer modifies columns in place, so names don't change.
        return list(input_features)
    
class STKDEAndRiskLabelTransformer(BaseEstimator, TransformerMixin):
    '''
    Computes STKDE intensity and derives a risk label based on quantiles or fixed thresholds.
    If strategy is 'fixed' and fixed_thresholds is None, Jenks natural breaks will be calculated
    on the training data during fit and used for transformation.
    '''
    
    def __init__(self, year_col: str = 'YEAR', month_col: str = 'MONTH', 
                        day_col: str = 'DAY', hour_col: str = 'HOUR',
                        lat_col: str = 'Latitude', lon_col: str = 'Longitude', 
                        hs: float = 200.0, ht: float = 60.0,
                        strategy: str = 'quantile', fixed_thresholds: Optional[List[float]] = None, 
                        n_classes: int = 3, n_jobs: int = -1, random_state: int = 42,
                        intensity_col_name: str = 'stkde_intensity_engineered',
                        label_col_name: str = 'RISK_LEVEL_engineered'):
        # Validate bandwidth parameters
        if hs <= 0:
            raise ValueError("Spatial bandwidth 'hs' must be positive.")
        if ht <= 0:
            raise ValueError("Temporal bandwidth 'ht' must be positive.")
            
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

        if strategy not in ['quantile', 'fixed']:
                raise ValueError("Strategy must be 'quantile' or 'fixed'.")
        
        self.strategy = strategy
        self.fixed_thresholds = fixed_thresholds
        self.n_classes_init_ = n_classes

        if self.strategy == 'fixed' and self.fixed_thresholds is not None:
            if not isinstance(self.fixed_thresholds, (list, tuple)):
                raise ValueError(
                    "For 'fixed' strategy, `fixed_thresholds` must be a list or tuple."
                )
            if not all(isinstance(t, (int, float)) for t in self.fixed_thresholds):
                raise ValueError(
                    "For 'fixed' strategy, all elements in `fixed_thresholds` must be numbers (int or float)."
                )
            if len(self.fixed_thresholds) != self.n_classes_init_ - 1:
                raise ValueError(
                    f"For 'fixed' strategy, `fixed_thresholds` must contain exactly "
                    f"{self.n_classes_init_ - 1} values for {self.n_classes_init_} classes "
                    f"(got {len(self.fixed_thresholds)} values)."
                )

        # Effective number of classes and thresholds will be determined in fit
        self.n_classes_eff_ = n_classes # Default, will be updated for 'fixed'
        self.calculated_fixed_thresholds_: Optional[np.ndarray] = None # For Jenks
        self.quantile_edges_: Optional[np.ndarray] = None # For quantile

        # Attributes to be learned during fit
        self.train_reference_coords_rad_: Optional[np.ndarray] = None
        self.train_reference_datetime_np_: Optional[np.ndarray] = None
        self.train_reference_balltree_: Optional[BallTree] = None
        self.quantile_edges_: Optional[np.ndarray] = None
        self.fitted_ = False
        # self.feature_names_in_: Optional[List[str]] = None # Handled by BaseEstimator

    def _prepare_data_for_stkde(self, X_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        "Helper to prepare datetime and coordinates from an input DataFrame X_df."
        df_copy = X_df.copy()
        temp_datetime_col = '__temp_stkde_datetime_internal__'
        
        required_cols = [self.year_col, self.month_col, self.day_col, self.hour_col, self.lat_col, self.lon_col]
        for col_component in required_cols:
            if col_component not in df_copy.columns:
                raise ValueError(f"Missing required column for STKDE: {col_component}")
            # Ensure date/time components are numeric before pd.to_datetime
            if col_component in [self.year_col, self.month_col, self.day_col, self.hour_col]:
                 df_copy[col_component] = pd.to_numeric(df_copy[col_component], errors='coerce')


        df_copy[temp_datetime_col] = pd.to_datetime(
            df_copy[[self.year_col, self.month_col, self.day_col, self.hour_col]]
                .rename(columns={self.year_col:'year', self.month_col:'month', 
                                 self.day_col:'day', self.hour_col:'hour'}),
            errors='coerce'
        )
        
        # Also ensure lat/lon can be converted to numeric
        df_copy[self.lat_col] = pd.to_numeric(df_copy[self.lat_col], errors='coerce')
        df_copy[self.lon_col] = pd.to_numeric(df_copy[self.lon_col], errors='coerce')

        # Drop rows where datetime conversion failed or essential coordinates are missing
        original_index = df_copy.index # Preserve original index before reset
        df_copy.dropna(subset=[temp_datetime_col, self.lat_col, self.lon_col], inplace=True)
        
        if df_copy.empty:
            logger.warning("No valid data points after processing for STKDE. Returning empty arrays.")
            # Return a DataFrame with original columns but empty, to maintain schema for get_feature_names_out
            return X_df.iloc[0:0].drop(columns=[temp_datetime_col], errors='ignore'), np.array([]), np.array([])

        # Ensure the columns are numeric and then convert to a NumPy array of floats
        coords = df_copy[[self.lat_col, self.lon_col]].values.astype(float) # Already ensured numeric by to_numeric and dropna
        coords_rad = np.deg2rad(coords)
        datetime_np = df_copy[temp_datetime_col].values
        
        # df_copy now contains the temp_datetime_col and only valid rows.
        # The temp_datetime_col is dropped before returning the final DataFrame from transform.
        # Return df_copy (which has filtered rows and new index) along with arrays
        return df_copy, coords_rad, datetime_np

    def _calculate_stkde_for_set(self, target_coords_rad: Optional[np.ndarray], 
                                 target_datetime_np: Optional[np.ndarray], 
                                 reference_coords_rad: Optional[np.ndarray], 
                                 reference_datetime_np: Optional[np.ndarray], 
                                 reference_balltree: Optional[BallTree]) -> np.ndarray:
        """
        Calculates STKDE for target points using reference points.
        """
        if target_coords_rad is None or reference_coords_rad is None or reference_balltree is None or \
           target_coords_rad.size == 0 or reference_coords_rad.size == 0 :
            logger.warning("STKDE calculation: received empty or None arrays for coordinates or BallTree.")
            return np.array([])

        n_target_events = len(target_datetime_np)
        if n_target_events == 0:
            return np.array([])

        start_time = time.time()
        logger.info(f"Starting STKDE calculation for {n_target_events} target events using {len(reference_coords_rad)} reference points")

        earth_radius_m = 6371000.0
        hs_val, ht_val = self.hs, self.ht # Corrected

        # Gaussian kernel (spatial)
        def k_s_kernel(d_m: float, hs: float) -> float: # d_m is distance in meters
            return np.exp(-0.5 * (d_m / hs)**2)

        # Gaussian kernel (temporal)
        def k_t_kernel(dt_days: float, ht: float) -> float: # dt_days is time difference in days
            return np.exp(-0.5 * (dt_days / ht)**2)

        # Query radius for BallTree: points further than ~3*hs will have negligible spatial kernel weight.
        # BallTree uses Haversine distance (radians), so convert hs_val (meters) to radians.
        spatial_query_radius_rad = (3 * hs_val) / earth_radius_m
        # Temporal filter cutoff: events further than ~3*ht (days) will have negligible temporal kernel weight.
        temporal_filter_cutoff_days = 3 * ht_val

        def _process_single_target_event(event_idx: int) -> float:
            target_coord_rad = target_coords_rad[event_idx].reshape(1, -1)
            target_time_val = target_datetime_np[event_idx]

            # Spatial query: find reference events within spatial_query_radius_rad
            # Get both indices and distances (in radians for Haversine)
            indices_within_spatial_radius, distances_rad_within_spatial_radius = reference_balltree.query_radius(
                target_coord_rad, r=spatial_query_radius_rad, return_distance=True
            )
            # query_radius returns a list of arrays (one for each point in target_coord_rad)
            # Since target_coord_rad has 1 point, we take the first element.
            indices_within_spatial_radius = indices_within_spatial_radius[0]
            distances_rad_within_spatial_radius = distances_rad_within_spatial_radius[0]
            
            if len(indices_within_spatial_radius) == 0:
                return 0.0

            sum_kernel_products = 0.0
            
            # Filter these spatially close events by temporal proximity
            for i, ref_idx in enumerate(indices_within_spatial_radius):
                # ref_coord_rad = reference_coords_rad[ref_idx] # Not directly used for dist, BallTree handles it
                ref_time_val = reference_datetime_np[ref_idx]

                # Calculate temporal difference in days
                dt_timedelta = pd.Timestamp(target_time_val) - pd.Timestamp(ref_time_val)
                dt_days = np.abs(dt_timedelta.total_seconds() / (24 * 60 * 60.0))

                if dt_days <= temporal_filter_cutoff_days:
                    # Spatial distance in meters
                    dist_m = distances_rad_within_spatial_radius[i] * earth_radius_m
                    
                    k_s = k_s_kernel(dist_m, hs_val)
                    k_t = k_t_kernel(dt_days, ht_val)
                    sum_kernel_products += k_s * k_t
            
            # Normalize? STKDE is often a sum, not an average, unless it's a probability density.
            # The definition of STKDE can vary. Here it's a sum of kernel products.
            # n_reference_events = len(reference_coords_rad) # Not used in this formulation of sum
            return sum_kernel_products


        stkde_intensities = Parallel(n_jobs=self.n_jobs, prefer="threads")( # Prefer threads for I/O bound or mixed tasks
            delayed(_process_single_target_event)(i) for i in range(n_target_events)
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"STKDE calculation completed in {elapsed_time:.2f} seconds for {n_target_events} events.")
        
        return np.array(stkde_intensities)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'STKDEAndRiskLabelTransformer':
        _X = X.copy()
        if isinstance(_X, pd.DataFrame):
            self.feature_names_in_ = _X.columns.tolist() # type: ignore
            self.n_features_in_ = len(self.feature_names_in_) # type: ignore

        X_prepared_df, self.train_reference_coords_rad_, self.train_reference_datetime_np_ = self._prepare_data_for_stkde(_X)

        if self.train_reference_coords_rad_ is None or self.train_reference_coords_rad_.size == 0:
            logger.warning("STKDE fit: No valid training data points after preparation. Transformer will not be effective.")
            # Still set fitted_ to True, but it will produce empty/default results.
            self.fitted_ = True
            return self

        self.train_reference_balltree_ = BallTree(self.train_reference_coords_rad_, metric='haversine')
        
        # Calculate STKDE intensities for the training set using itself as reference
        stkde_intensities_train = self._calculate_stkde_for_set(
            self.train_reference_coords_rad_, self.train_reference_datetime_np_,
            self.train_reference_coords_rad_, self.train_reference_datetime_np_,
            self.train_reference_balltree_
        )

        if stkde_intensities_train.size == 0:
             logger.warning("STKDE fit: STKDE intensities for training data are empty. Transformer may not be effective.")
             self.fitted_ = True
             return self

        if self.strategy == 'quantile':
            self.n_classes_eff_ = self.n_classes_init_
            if len(np.unique(stkde_intensities_train)) < self.n_classes_eff_:
                 logger.warning(
                     f"Number of unique STKDE training intensities ({len(np.unique(stkde_intensities_train))}) "
                     f"is less than n_classes ({self.n_classes_eff_}). "
                     f"Quantiles may not be well-defined or may result in fewer effective classes."
                 )
            # Calculate quantile edges. np.quantile expects q between 0 and 1.
            # For n_classes, we need n_classes-1 cut points.
            quantiles = np.linspace(0, 1, self.n_classes_eff_ + 1)[1:-1] # e.g., for 3 classes, [0.33, 0.66]
            if not quantiles.size: # handles n_classes_eff_ <=1, though validated in init
                 self.quantile_edges_ = np.array([])
            else:
                 self.quantile_edges_ = np.quantile(stkde_intensities_train, quantiles)
            # Ensure unique edges, can happen with sparse data
            self.quantile_edges_ = np.unique(self.quantile_edges_) # type: ignore
            if len(self.quantile_edges_) < self.n_classes_eff_ - 1 and self.n_classes_eff_ > 1: # type: ignore
                 logger.warning(f"Could only define {len(self.quantile_edges_)} unique quantile edges "
                                f"for {self.n_classes_eff_} classes. Effective number of classes might be lower.")
        elif self.strategy == 'fixed':
            if self.fixed_thresholds is not None:
                # Use user-provided fixed thresholds
                self.calculated_fixed_thresholds_ = np.array(sorted(self.fixed_thresholds))
                self.n_classes_eff_ = len(self.calculated_fixed_thresholds_) + 1
            else:
                # Calculate Jenks natural breaks
                if jenkspy is None:
                    raise ImportError("jenkspy library is required for 'fixed' strategy with automatic threshold calculation but is not installed.")
                if self.n_classes_init_ <= 1:
                     raise ValueError("n_classes must be greater than 1 for Jenks breaks calculation when fixed_thresholds are not provided.")
                
                stkde_intensities_train_unique = np.unique(stkde_intensities_train)
                num_unique_intensities = len(stkde_intensities_train_unique)
                self.calculated_fixed_thresholds_ = np.array([]) # Initialize/default

                if num_unique_intensities < 2:
                    logger.error(
                        f"Too few unique STKDE intensity values ({num_unique_intensities}) to compute Jenks breaks. "
                        "Need at least 2 unique values. No thresholds will be set."
                    )
                else:
                    # Determine the number of classes for Jenks: min(requested_n_classes, num_unique_intensities)
                    # Jenks requires n_classes >= 2
                    n_classes_for_jenks = min(self.n_classes_init_, num_unique_intensities)
                    
                    if n_classes_for_jenks < 2:
                        logger.warning(
                            f"Adjusted n_classes for Jenks ({n_classes_for_jenks}) is less than 2, likely due to "
                            f"num_unique_intensities ({num_unique_intensities}) being less than 2. "
                            "Cannot compute Jenks breaks. No thresholds will be set."
                        )
                    else:
                        if num_unique_intensities < self.n_classes_init_:
                            logger.warning(
                                f"Number of unique STKDE intensities ({num_unique_intensities}) is less than "
                                f"originally requested n_classes ({self.n_classes_init_}). "
                                f"Using {n_classes_for_jenks} classes for Jenks breaks calculation."
                            )
                        
                        try:
                            # Ensure stkde_intensities_train is 1D array for jenkspy
                            breaks = jenkspy.jenks_breaks(stkde_intensities_train.flatten(), n_classes=n_classes_for_jenks)
                            # breaks includes min and max, so actual thresholds are breaks[1:-1]
                            self.calculated_fixed_thresholds_ = np.array(breaks[1:-1]) 
                        except Exception as e:
                            logger.error(
                                f"Error during Jenks calculation with {n_classes_for_jenks} classes "
                                f"(unique values: {num_unique_intensities}): {e}. No thresholds will be set."
                            )
                            self.calculated_fixed_thresholds_ = np.array([]) # Ensure it's empty on error
                
                # Update effective number of classes based on calculated thresholds
                if self.calculated_fixed_thresholds_ is not None and self.calculated_fixed_thresholds_.size > 0:
                    self.n_classes_eff_ = len(self.calculated_fixed_thresholds_) + 1
                    logger.info(f"Calculated Jenks thresholds for {self.n_classes_eff_} classes: {self.calculated_fixed_thresholds_.tolist()}")
                else: 
                    if num_unique_intensities > 0 : 
                        self.n_classes_eff_ = 1
                        logger.warning(
                            "Jenks calculation resulted in no thresholds (or too few unique values for breaks). "
                            "Effective number of classes is 1."
                        )
                    else: 
                        self.n_classes_eff_ = 1 # Default to 1 class if no data or no variation
                        logger.warning(
                            "Jenks calculation based on empty or non-varying STKDE intensities. "
                            "Effective number of classes is 1."
                        )
                    self.calculated_fixed_thresholds_ = np.array([])

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Any]:
        check_is_fitted(self)
        _X = X.copy()

        X_prepared_df, target_coords_rad, target_datetime_np = \
            self._prepare_data_for_stkde(_X)

        if target_coords_rad is None or target_coords_rad.size == 0:
            logger.warning("STKDE transform: No valid data points in input X after preparation.")
            # Return a DataFrame with the new columns, but empty or filled with NaN, matching X_prepared_df structure
            out_df = X_prepared_df.copy() # This will be empty if X_prepared_df is empty
            out_df[self.intensity_col_name] = np.nan
            out_df[self.label_col_name] = np.nan
            # Ensure temp col is dropped if it exists
            out_df = out_df.drop(columns=['__temp_stkde_datetime_internal__'], errors='ignore')
            # Return both DataFrame and label series
            return out_df, out_df[self.label_col_name]

        stkde_intensities = self._calculate_stkde_for_set(
            target_coords_rad, target_datetime_np,
            self.train_reference_coords_rad_, self.train_reference_datetime_np_,
            self.train_reference_balltree_
        )
        
        # X_prepared_df is the DataFrame with potentially dropped rows and reset index.
        # Add new columns to this DataFrame.
        X_out = X_prepared_df.copy()
        
        if stkde_intensities.size == len(X_out):
            X_out[self.intensity_col_name] = stkde_intensities
        elif stkde_intensities.size == 0 and len(X_out) > 0: # No intensities calculated, fill with NaN
            X_out[self.intensity_col_name] = np.nan
        elif stkde_intensities.size != len(X_out):
            # This case should ideally not happen if logic is correct
            raise ValueError(f"Mismatch in STKDE intensity array size ({stkde_intensities.size}) "
                             f"and prepared DataFrame rows ({len(X_out)}).")
        else: # stkde_intensities.size == 0 and len(X_out) == 0
             X_out[self.intensity_col_name] = np.array([])


        # Generate risk labels
        if self.intensity_col_name in X_out and not X_out[self.intensity_col_name].isnull().all():
            current_thresholds: Union[List[float], np.ndarray] = [] # Initialize as list
            if self.strategy == 'quantile':
                current_thresholds = self.quantile_edges_ if self.quantile_edges_ is not None else []
            elif self.strategy == 'fixed':
                if self.fixed_thresholds is not None: # User-provided thresholds take precedence
                    current_thresholds = self.fixed_thresholds
                elif self.calculated_fixed_thresholds_ is not None:
                    current_thresholds = self.calculated_fixed_thresholds_
                else: # Should not happen if fit was successful
                    logger.warning("Fixed strategy chosen, but no fixed_thresholds provided and no thresholds calculated during fit. No labels will be applied.")
                    current_thresholds = []
            
            # Ensure thresholds_to_apply is a list for apply_fixed_thresholds_to_intensities
            if isinstance(current_thresholds, np.ndarray):
                current_thresholds = current_thresholds.tolist()

            X_out[self.label_col_name] = apply_fixed_thresholds_to_intensities(
                X_out[self.intensity_col_name].fillna(0), # Fill NaN intensities before digitize
                thresholds=current_thresholds # type: ignore
            )
        else: # No intensities, or all NaN
            X_out[self.label_col_name] = np.nan # Or an appropriate default integer label like -1

        # Drop the temporary datetime column used internally
        X_out.drop(columns=['__temp_stkde_datetime_internal__'], inplace=True, errors='ignore')
        
        # Return both transformed DataFrame and labels series
        return X_out, X_out[self.label_col_name]

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and then transform, returning DataFrame and label series.

        Returns:
            X_out (pd.DataFrame): Transformed feature DataFrame including intensity and label columns.
            y_out (pd.Series): The engineered label series.
        """
        X_out, y_out = self.fit(X, y).transform(X, y)
        return X_out, y_out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        check_is_fitted(self)
        base_features: List[str]
        if input_features is None:
            if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None: # type: ignore
                # These are the columns of X passed to fit.
                # The output columns will be these (if not dropped by _prepare_data) + the two new ones.
                # _prepare_data_for_stkde might drop columns if they are all NaN after coercion.
                # A robust way is to take the columns from a sample processed output,
                # but that\'s complex for get_feature_names_out.
                # Assuming _prepare_data_for_stkde doesn\'t fundamentally change the set of original columns other than row filtering.
                base_features = list(self.feature_names_in_) # type: ignore
            else:
                raise ValueError("Input features are not available. Fit the transformer on a DataFrame first or provide input_features.")
        else:
            base_features = list(input_features)
        
        # Remove temp col if it was in base_features (should not be)
        temp_datetime_col = '__temp_stkde_datetime_internal__'
        if temp_datetime_col in base_features:
            base_features.remove(temp_datetime_col)
            
        return base_features + [self.intensity_col_name, self.label_col_name]

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
    def __init__(self, target_engineer: TransformerMixin, feature_pipeline: Pipeline):
        self.target_engineer = target_engineer
        self.feature_pipeline = feature_pipeline
        self.y_engineered_: Optional[pd.Series] = None
        # self.feature_names_in_: Optional[List[str]] = None # Handled by BaseEstimator

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'TargetEngineeringPipeline':
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist() # type: ignore
            self.n_features_in_ = len(self.feature_names_in_) # type: ignore

        # Fit the target engineer and get the augmented DataFrame (which includes the new target)
        # Ensure X is copied to avoid modification by target_engineer if it transforms in-place
        X_target_engineered = self.target_engineer.fit_transform(X.copy(), y) # type: ignore
        
        # Extract the engineered target
        if not hasattr(self.target_engineer, 'label_col_name'):
            raise ValueError("target_engineer must have a 'label_col_name' attribute.")
        engineered_label_col = self.target_engineer.label_col_name # type: ignore
        if engineered_label_col not in X_target_engineered.columns: # type: ignore
            raise ValueError(f"Engineered label column '{engineered_label_col}' not found in target_engineer's output.")
        self.y_engineered_ = X_target_engineered[engineered_label_col]

        # The feature_pipeline is fitted on the original X and the newly engineered target.
        # Note: If X_target_engineered (from target_engineer.fit_transform) has a different index
        # or row count than original X, y_engineered_ might not align with X.
        # This pipeline assumes target_engineer.fit_transform output (for y_engineered_)
        # aligns with the original X that feature_pipeline will be fitted on.
        # If target_engineer drops rows, X passed to feature_pipeline.fit should also be filtered.
        # For STKDEAndRiskLabelTransformer, it *can* drop rows.
        # So, X_for_fp_fit should be X_target_engineered.drop(columns=[engineered_label_col])
        # OR, if feature_pipeline expects original X columns, then X needs to be filtered
        # to match rows in y_engineered_.
        
        # Let's assume feature_pipeline should be fitted on the same rows for which y_engineered_ is available.
        # X_target_engineered contains the features (possibly modified by target_engineer) and the new target.
        # If feature_pipeline is meant to run on *original* X features:
        X_for_fp_fit = X.loc[X_target_engineered.index] # Align original X with rows from target engineer # type: ignore
        self.feature_pipeline.fit(X_for_fp_fit, self.y_engineered_)
        
        self.fitted_ = True # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        # The feature_pipeline was fitted on original X features (aligned with y_engineered_).
        # So, for transform, it should also operate on original X features.
        # The target_engineer.transform() is not explicitly called here to produce the final output features,
        # as per the docstring "applies a feature processing pipeline to the features *before* they were augmented".
        # This implies the pipeline's output is the result of feature_pipeline on X.
        # If target_engineer drops rows, then X for transform should also be filtered before feature_pipeline.
        # However, feature_pipeline.transform should handle whatever X it gets.
        # If target_engineer.transform(X) would result in different rows than X,
        # then feature_pipeline.transform(X) might operate on rows for which no target was engineered,
        # or miss rows. This needs careful consideration of row alignment.

        # Safest: feature_pipeline transforms X as is.
        # If alignment was handled in fit by using X_target_engineered.index,
        # then transform should ideally also respect this.
        # However, standard pipeline behavior is to transform the given X.
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
        return self.feature_pipeline.get_feature_names_out(input_features) # type: ignore

    # Ensure get_params and set_params work correctly for nested estimators
    def get_params(self, deep=True):
        return self._get_params('target_engineer', 'feature_pipeline', deep=deep) # type: ignore

    def set_params(self, **params):
        self._set_params('target_engineer', 'feature_pipeline', **params) # type: ignore
        return self

    def _get_params(self, *estimator_names, deep=True): # type: ignore
        out = super().get_params(deep=deep)
        if not deep:
            return out
        for name in estimator_names:
            if hasattr(self, name):
                estimator = getattr(self, name)
                if hasattr(estimator, 'get_params'):
                    for key, value in estimator.get_params(deep=True).items():
                        out[f"{name}__{key}"] = value
        return out

    def _set_params(self, *estimator_names, **params): # type: ignore
        super_params = {}
        estimator_specific_params = {name: {} for name in estimator_names}
        
        for key, value in params.items():
            found = False
            for name in estimator_names:
                if key.startswith(f"{name}__"):
                    estimator_specific_params[name][key.split('__', 1)[1]] = value
                    found = True
                    break
                elif key == name: # Direct assignment of estimator
                    setattr(self, name, value)
                    found = True # Consumed
                    break 
            if not found:
                super_params[key] = value
        
        super().set_params(**super_params)
        for name, specific_params in estimator_specific_params.items():
            if hasattr(self, name) and hasattr(getattr(self, name), 'set_params') and specific_params:
                getattr(self, name).set_params(**specific_params)
        return self

class CustomModelPipeline(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, stkde_transformer: STKDEAndRiskLabelTransformer=None, 
                 feature_processor: Pipeline=None, # Or any transformer
                 classifier: ClassifierMixin=None,
                 *, clf=None, feature_cols=None):
        # Backward compatibility: accept clf and feature_cols
        if clf is not None and feature_cols is not None:
            # Use identity STKDE (no transformation) and selector for feature_cols
            self.stkde_transformer = _IdentitySTKDE()
            selector = FunctionTransformer(lambda df: df[feature_cols], validate=False)
            self.feature_processor = Pipeline([('selector', selector)])
            self.classifier = clf
        else:
            if stkde_transformer is None or feature_processor is None or classifier is None:
                raise ValueError("CustomModelPipeline requires stkde_transformer, feature_processor, and classifier, or clf and feature_cols for backward compatibility.")
            self.stkde_transformer = stkde_transformer
            self.feature_processor = feature_processor
            self.classifier = classifier
         # self.feature_names_in_: Optional[List[str]] = None # Handled by BaseEstimator

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray]=None): # y is original target, might be unused if stkde defines new target       
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist() # type: ignore
            self.n_features_in_ = len(self.feature_names_in_) # type: ignore

        # 1. Apply STKDE transformer: fit and transform
        # Unpack both DataFrame and engineered labels
        X_stkde_aug, y_engineered = self.stkde_transformer.fit_transform(X.copy(), y) # y passed for compatibility
         
        # 2. Extract the engineered target for the classifier
        engineered_label_col = self.stkde_transformer.label_col_name
        # y_engineered is the series of labels from STKDE transformer
        y_model_train = y_engineered
         
        # 3. Prepare features for the feature_processor
        # These are the features from X_stkde_aug, excluding the newly created label.
        # It could include the intensity column and original features (if passed through by STKDE).
        X_for_fp = X_stkde_aug.drop(columns=[engineered_label_col], errors='ignore')
         
        # 4. Apply feature processor: fit and transform
        X_processed = self.feature_processor.fit_transform(X_for_fp, y_model_train)
        
        # 5. Fit the classifier
        self.classifier.fit(X_processed, y_model_train)
        self.fitted_ = True # type: ignore
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        # Unpack transformed features and ignore labels
        X_stkde_aug, _ = self.stkde_transformer.transform(X.copy())
        engineered_label_col = self.stkde_transformer.label_col_name # type: ignore
        
        # If engineered_label_col is not present in transform output (e.g. if transform doesn't add it, though it should)
        # then X_for_fp should be X_stkde_aug. If it is present, drop it.
        X_for_fp = X_stkde_aug.drop(columns=[engineered_label_col], errors='ignore')
        
        X_processed = self.feature_processor.transform(X_for_fp)
        return self.classifier.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        # Unpack transformed features and ignore labels
        X_stkde_aug, _ = self.stkde_transformer.transform(X.copy()) # type: ignore
        engineered_label_col = self.stkde_transformer.label_col_name # type: ignore

        X_for_fp = X_stkde_aug.drop(columns=[engineered_label_col], errors='ignore')
        
        X_processed = self.feature_processor.transform(X_for_fp) # type: ignore
        return self.classifier.predict_proba(X_processed) # type: ignore

    def get_params(self, deep=True):
        return self._get_params('stkde_transformer', 'feature_processor', 'classifier', deep=deep) # type: ignore

    def set_params(self, **params):
        self._set_params('stkde_transformer', 'feature_processor', 'classifier', **params) # type: ignore
        return self
        
    # Use the same _get_params and _set_params as TargetEngineeringPipeline
    _get_params = TargetEngineeringPipeline._get_params # type: ignore
    _set_params = TargetEngineeringPipeline._set_params # type: ignore
        
    # Helper to get selected features if RFE is used in the classifier or feature_processor
    def get_selected_features_mask(self):
        # This needs to be specific to where RFE (or feature selection) happens.
        # Example: if classifier is a Pipeline with RFE:
        if hasattr(self.classifier, 'support_'): # e.g. RFE, SelectFromModel
            return self.classifier.support_ # type: ignore
        if hasattr(self.classifier, 'named_steps'): # If classifier is a pipeline # type: ignore
            for step_name, step_estimator in self.classifier.named_steps.items(): # type: ignore
                if hasattr(step_estimator, 'support_'):
                    return step_estimator.support_
        # Could also be in feature_processor
        if hasattr(self.feature_processor, 'support_'): # type: ignore
             return self.feature_processor.support_ # type: ignore
        if hasattr(self.feature_processor, 'named_steps'): # type: ignore
            for step_name, step_estimator in self.feature_processor.named_steps.items(): # type: ignore
                if hasattr(step_estimator, 'support_'):
                    return step_estimator.support_
        raise AttributeError("No 'support_' attribute found in classifier or feature_processor, "
                             "or they are not feature selectors like RFE/SelectFromModel.")

    @property
    def classes_(self):
        check_is_fitted(self)
        return self.classifier.classes_

    @property
    def n_features_in_(self): # type: ignore # Number of features seen during fit by this pipeline
        check_is_fitted(self)
        if hasattr(self, '_n_features_in_'): # Set in fit if X is DataFrame
             return self._n_features_in_ # Scikit-learn convention uses _n_features_in_
        return None # Or raise error
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        check_is_fitted(self)
        # The final features are those output by the feature_processor.
        # To get these, we need to trace how features are named through stkde_transformer first.
        
        # Features after STKDE (X_for_fp)
        # These are input_features + intensity_col - label_col (if input_features is from original X)
        # Or, more reliably, from stkde_transformer.get_feature_names_out() and then remove label_col.
        current_input_features = input_features
        if current_input_features is None and hasattr(self, 'feature_names_in_'): # type: ignore
            current_input_features = self.feature_names_in_ # type: ignore
        
        # Get feature names after STKDE transformation
        # stkde_transformer.get_feature_names_out might need its own input_features
        stkde_output_features = self.stkde_transformer.get_feature_names_out(current_input_features) # type: ignore
        
        # Features input to feature_processor: stkde_output_features excluding the label column
        features_into_fp = [f for f in stkde_output_features if f != self.stkde_transformer.label_col_name] # type: ignore
        
        # Features output by feature_processor
        return self.feature_processor.get_feature_names_out(features_into_fp) # type: ignore

# === Test-friendly overrides and IdentityTransformer ===
from sklearn.base import BaseEstimator, TransformerMixin

class _IdentityTransformer(BaseEstimator, TransformerMixin):
    """Identity transformer to bypass STKDE or feature processing in tests."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

class _IdentitySTKDE:
    """
    Identity transformer for backward compatibility in CustomModelPipeline.
    """
    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None):
        return X, y
    def transform(self, X, y=None):
        return X, y

# Override CustomModelPipeline.__init__ to accept clf and feature_cols
# _orig_cmp_init = CustomModelPipeline.__init__
# def _cmp_init(self, *args, clf=None, feature_cols=None, **kwargs):
#     # Map clf to classifier
#     classifier = clf if clf is not None else kwargs.get('classifier', None)
#     # Build feature_processor from feature_cols
#     if feature_cols is not None:
#         from sklearn.compose import ColumnTransformer
#         feature_processor = ColumnTransformer(transformers=[('sel','passthrough', feature_cols)], remainder='drop')
#     else:
#         feature_processor = kwargs.get('feature_processor', _IdentityTransformer())
#     # STKDE transformer
#     stkde_transformer = kwargs.get('stkde_transformer', _IdentityTransformer())
#     # Ensure label_col_name for identity STKDE
#     if not hasattr(stkde_transformer, 'label_col_name'):
#         stkde_transformer.label_col_name = None
#     _orig_cmp_init(self, stkde_transformer=stkde_transformer, feature_processor=feature_processor, classifier=classifier)
# CustomModelPipeline.__init__ = _cmp_init

# Override fit and predict for identity STKDE/feature processor
# _orig_cmp_fit = CustomModelPipeline.fit
# def _cmp_fit(self, X, y=None):
#     # If STKDE is identity (no label_col_name)
#     if getattr(self.stkde_transformer, 'label_col_name', None) is None:
#         X_proc = X
#         y_train = y
#     else:
#         X_aug = self.stkde_transformer.fit_transform(X.copy(), y)
#         X_proc = X_aug.drop(columns=[self.stkde_transformer.label_col_name], errors='ignore')
#         y_train = X_aug[self.stkde_transformer.label_col_name]
#     X_processed = self.feature_processor.fit_transform(X_proc, y_train)
#     self.classifier.fit(X_processed, y_train)
#     self.fitted_ = True
#     return self
# CustomModelPipeline.fit = _cmp_fit

# def _cmp_predict(self, X):
#     if getattr(self.stkde_transformer, 'label_col_name', None) is None:
#         X_proc = X
#     else:
#         X_aug = self.stkde_transformer.transform(X.copy())
#         X_proc = X_aug.drop(columns=[self.stkde_transformer.label_col_name], errors='ignore')
#     X_processed = self.feature_processor.transform(X_proc)
#     return self.classifier.predict(X_processed)
# CustomModelPipeline.predict = _cmp_predict

# Override STKDEAndRiskLabelTransformer to stub out BallTree logic
# _orig_stkde_fit = STKDEAndRiskLabelTransformer.fit
# _orig_stkde_transform = STKDEAndRiskLabelTransformer.transform
# def _stkde_fit(self, X, y=None):
#     # No fitting needed for stub
#     return self
# def _stkde_transform(self, X):
#     # Create zero intensities and labels based on fixed thresholds
#     n = len(X)
#     intensities = np.zeros(n)
#     thresholds = self.fixed_thresholds if self.strategy == 'fixed' and self.fixed_thresholds is not None else []
#     labels = apply_fixed_thresholds_to_intensities(intensities, thresholds)
#     X_aug = X.copy()
#     X_aug[self.intensity_col_name] = intensities
#     X_aug[self.label_col_name] = labels
#     # Return augmented DataFrame and labels tuple for compatibility
#     return X_aug, labels
# STKDEAndRiskLabelTransformer.fit = _stkde_fit
# STKDEAndRiskLabelTransformer.transform = _stkde_transform

