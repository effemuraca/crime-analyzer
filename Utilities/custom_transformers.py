'''
Custom transformers and utility functions for the project.
'''
import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import BallTree # For STKDE
from joblib import Parallel, delayed # For STKDE
from sklearn.pipeline import Pipeline # For TargetEngineeringPipeline

def cyclical_transform(X):
    # This function encodes cyclical features using sine and cosine transforms.
    # It expects a pandas DataFrame containing only the cyclical columns.
    import numpy as np
    import pandas as pd

    if not isinstance(X, pd.DataFrame):
         raise TypeError(f"cyclical_transform: input must be a pandas DataFrame. Got {type(X)}.")

    X = X.copy()
    out = pd.DataFrame(index=X.index)
    feature_names_out = [] # To store new column names

    # Define mappings and periods
    weekday_order = ['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
    weekday_map = {d:i for i,d in enumerate(weekday_order)}
    season_order = ['WINTER','SPRING','SUMMER','AUTUMN']
    season_map = {s:i for i,s in enumerate(season_order)}
    time_bucket_order = ['MORNING','AFTERNOON','EVENING','NIGHT']
    time_bucket_map = {t:i for i,t in enumerate(time_bucket_order)}

    # Process each column present in the input DataFrame X
    for col in X.columns:
        col_out_sin = f'{col}_SIN'
        col_out_cos = f'{col}_COS'
        feature_names_out.extend([col_out_sin, col_out_cos])

        if col == 'HOUR':
            # Ensure numeric, fill missing with 0 (or another strategy if needed)
            num_val = pd.to_numeric(X[col], errors='coerce').fillna(0)
            period = 24
        elif col == 'WEEKDAY':
            num_val = X[col].astype(str).map(weekday_map)
            num_val = pd.to_numeric(num_val, errors='coerce').fillna(0) # Fill unknown weekdays
            period = 7
        elif col == 'MONTH':
            num_val = pd.to_numeric(X[col], errors='coerce').fillna(1) # Fill missing month with Jan
            period = 12
        elif col == 'SEASON':
            num_val = X[col].astype(str).map(season_map)
            num_val = pd.to_numeric(num_val, errors='coerce').fillna(0) # Fill unknown seasons
            period = 4
        elif col == 'TIME_BUCKET':
            num_val = X[col].astype(str).map(time_bucket_map)
            num_val = pd.to_numeric(num_val, errors='coerce').fillna(0) # Fill unknown buckets
            period = 4
        else:
            # Should not happen if cyclical_cols is defined correctly
            warnings.warn(f"Column '{col}' not recognized for cyclical encoding. Skipping.")
            continue

        # Apply sin/cos transformation
        out[col_out_sin] = np.sin(2 * np.pi * num_val / period)
        out[col_out_cos] = np.cos(2 * np.pi * num_val / period)

    # Set feature names for the transformer output
    out.columns = feature_names_out
    return out

class BinarizeSinCosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store feature names if X is a DataFrame, for get_feature_names_out
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        # The input X to this transformer will be the output of the ColumnTransformer's 'cyc_passthrough' part
        # which contains the sin/cos transformed columns.
        for col in X_transformed.columns: # Iterate over all columns passed to this transformer
            if col.endswith('_SIN') or col.endswith('_COS'): # Apply only to sin/cos columns
                 X_transformed[col] = (X_transformed[col] > self.threshold).astype(np.uint8)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        raise ValueError("Input feature names are not available. Fit the transformer on a DataFrame first.")

class STKDEAndRiskLabelTransformer(BaseEstimator, TransformerMixin):
    '''
    Computes STKDE intensity and derives a risk label based on quantiles.
    This transformer is designed to be used *within* each fold of a cross-validation 
    or *after* a train-test split on the training data only, to prevent data leakage.

    On fit(X_train, y_train=None):
    - Calculates STKDE intensities for X_train using its own data points.
    - Determines quantile bins for RISK_LEVEL based on these X_train intensities.
    - Stores X_train (or its relevant parts for STKDE calculation) as a reference.

    On transform(X_new, y_new=None):
    - Calculates STKDE intensities for X_new using the *stored X_train reference data*.
    - Applies the *stored quantile bins* from X_train to assign RISK_LEVEL to X_new.
    - Returns X_new with two new columns: 'stkde_intensity_engineered' and 'RISK_LEVEL_engineered'.
    '''
    def __init__(self, year_col='YEAR', month_col='MONTH', day_col='DAY', hour_col='HOUR',
                 lat_col='Latitude', lon_col='Longitude', hs=200.0, ht=60.0, n_classes=3, 
                 n_jobs=-1, random_state=42, 
                 intensity_col_name='stkde_intensity_engineered', 
                 label_col_name='RISK_LEVEL_engineered'):
        self.year_col = year_col
        self.month_col = month_col
        self.day_col = day_col
        self.hour_col = hour_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.hs = hs
        self.ht = ht
        self.n_classes = n_classes
        self.n_jobs = n_jobs
        self.random_state = random_state # For reproducibility if any internal sampling occurs
        self.intensity_col_name = intensity_col_name
        self.label_col_name = label_col_name

        # Attributes to be learned during fit
        self.train_reference_coords_rad_ = None
        self.train_reference_datetime_np_ = None
        self.train_reference_balltree_ = None
        self.quantile_edges_ = None
        self.fitted_ = False

    def _prepare_data_for_stkde(self, X_df):
        "Helper to prepare datetime and coordinates from an input DataFrame X_df."
        df_copy = X_df.copy()
        temp_datetime_col = '__temp_stkde_datetime_internal__'
        
        # Ensure date/time components are numeric before pd.to_datetime
        for col_component in [self.year_col, self.month_col, self.day_col, self.hour_col]:
            if col_component in df_copy.columns:
                df_copy[col_component] = pd.to_numeric(df_copy[col_component], errors='coerce')
            else:
                raise ValueError(f"Missing required datetime column for STKDE: {col_component}")

        df_copy[temp_datetime_col] = pd.to_datetime(
            df_copy[[self.year_col, self.month_col, self.day_col, self.hour_col]]
                .rename(columns={self.year_col:'year', self.month_col:'month', 
                                 self.day_col:'day', self.hour_col:'hour'}),
            errors='coerce' # Important: coerce errors to NaT
        )
        
        # Drop rows where datetime conversion failed or essential coordinates are missing
        # The original indices are preserved before reset_index for potential later alignment if needed,
        # though the transformer primarily works with the transformed data directly.
        df_copy.dropna(subset=[temp_datetime_col, self.lat_col, self.lon_col], inplace=True)
        df_copy.reset_index(drop=True, inplace=True)

        if df_copy.empty:
            # If no valid data points remain, return None for coords and datetimes
            # This will be handled by the calling _calculate_stkde_for_set function
            return None, None, None, df_copy # Return empty df as well

        # Ensure the columns are numeric and then convert to a NumPy array of floats
        coords = df_copy[[self.lat_col, self.lon_col]].apply(pd.to_numeric, errors='coerce').values.astype(float)
        coords_rad = np.deg2rad(coords)
        datetime_np = df_copy[temp_datetime_col].values
        
        # df_copy now contains the temp_datetime_col and only valid rows.
        # The temp_datetime_col is dropped before returning the final DataFrame from transform.
        return coords_rad, datetime_np, df_copy # Return the processed df_copy

    def _calculate_stkde_for_set(self, target_coords_rad, target_datetime_np, 
                                 reference_coords_rad, reference_datetime_np, reference_balltree):
        "Calculates STKDE for target points using reference points."
        # from sklearn.neighbors import BallTree # Local import if not already available

        if target_coords_rad is None or reference_coords_rad is None or reference_balltree is None:
            # This can happen if _prepare_data_for_stkde returned None due to all rows being invalid
            # or if fit was called on an empty dataset.
            # Return an array of NaNs or zeros matching the expected output shape if target_datetime_np exists.
            if target_datetime_np is not None:
                return np.full(len(target_datetime_np), np.nan)
            return np.array([]) # Or handle as an error upstream

        n_target_events = len(target_datetime_np)
        if n_target_events == 0:
            return np.array([])

        earth_radius_m = 6371000.0
        hs_val, ht_val = self.hs, self.ht

        def k_s_kernel(d, hs):
            if hs == 0: return 0
            return np.exp(-0.5 * (d / hs)**2) / (2 * np.pi * hs**2)

        def k_t_kernel(dt_days, ht):
            if ht == 0: return 0
            # Ensure dt_days is a float for np.exp and np.abs
            dt_days_float = np.abs(float(dt_days)) 
            return np.exp(-dt_days_float / ht) / ht

        spatial_query_radius_rad = (3 * hs_val) / earth_radius_m
        temporal_filter_cutoff_days = 3 * ht_val

        def _process_single_target_event(event_idx):
            current_target_coord_rad = target_coords_rad[event_idx:event_idx+1]
            current_target_datetime = target_datetime_np[event_idx]

            # Query the *reference* BallTree
            neighbor_indices_in_ref, neighbor_distances_rad_from_ref = reference_balltree.query_radius(
                current_target_coord_rad, r=spatial_query_radius_rad, return_distance=True
            )
            spatial_neighbor_indices_in_ref = neighbor_indices_in_ref[0]
            spatial_neighbor_distances_rad = neighbor_distances_rad_from_ref[0]

            current_stkde_sum = 0.0
            if len(spatial_neighbor_indices_in_ref) == 0:
                return 0.0

            for i_loop, ref_neighbor_original_idx in enumerate(spatial_neighbor_indices_in_ref):
                ref_neighbor_datetime = reference_datetime_np[ref_neighbor_original_idx]
                
                time_delta = current_target_datetime - ref_neighbor_datetime
                time_delta_in_seconds = time_delta / np.timedelta64(1, 's')
                time_delta_days = time_delta_in_seconds / (24.0 * 3600.0)

                if abs(time_delta_days) > temporal_filter_cutoff_days:
                    continue

                dist_meters = spatial_neighbor_distances_rad[i_loop] * earth_radius_m
                weight_spatial = k_s_kernel(dist_meters, hs_val)
                weight_temporal = k_t_kernel(time_delta_days, ht_val)
                current_stkde_sum += weight_spatial * weight_temporal
            return current_stkde_sum

        stkde_intensities = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(_process_single_target_event)(i) for i in range(n_target_events)
        )
        return np.array(stkde_intensities)

    def fit(self, X, y=None):
        # from sklearn.neighbors import BallTree # Ensure BallTree is available
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X to STKDEAndRiskLabelTransformer.fit must be a pandas DataFrame.")

        # 1. Prepare data from X (this is X_train)
        # df_train_prepared will have a '__temp_stkde_datetime_internal__' column and only valid rows
        train_coords_rad, train_datetime_np, df_train_prepared = self._prepare_data_for_stkde(X)

        if train_coords_rad is None or df_train_prepared.empty:
            warnings.warn("STKDEAndRiskLabelTransformer.fit called with data that results in no valid points for STKDE. Transformer may not function correctly.")
            # Set defaults for a non-functional state to avoid errors in transform if called
            self.train_reference_coords_rad_ = None
            self.train_reference_datetime_np_ = None
            self.train_reference_balltree_ = None
            self.quantile_edges_ = np.array([-np.inf, np.inf]) # Default to one bin if no data
            self.fitted_ = True
            return self

        # 2. Store reference data for use in transform()
        self.train_reference_coords_rad_ = train_coords_rad
        self.train_reference_datetime_np_ = train_datetime_np
        self.train_reference_balltree_ = BallTree(self.train_reference_coords_rad_, metric='haversine')

        # 3. Calculate STKDE intensities for this training data (X_train) using itself as reference
        #    This is crucial: fit uses X_train to calculate intensities on X_train.
        train_stkde_intensities = self._calculate_stkde_for_set(
            target_coords_rad=self.train_reference_coords_rad_, 
            target_datetime_np=self.train_reference_datetime_np_,
            reference_coords_rad=self.train_reference_coords_rad_, 
            reference_datetime_np=self.train_reference_datetime_np_,
            reference_balltree=self.train_reference_balltree_
        )
        
        # Add these intensities temporarily to df_train_prepared to compute quantiles
        if len(train_stkde_intensities) == len(df_train_prepared):
            df_train_prepared[self.intensity_col_name] = train_stkde_intensities
        else:
            warnings.warn(f"Mismatch in lengths for train_stkde_intensities ({len(train_stkde_intensities)}) and df_train_prepared ({len(df_train_prepared)}). Quantiles might be incorrect. Filling with 0.")
            temp_series = pd.Series(train_stkde_intensities, index=df_train_prepared.index[:len(train_stkde_intensities)])
            df_train_prepared[self.intensity_col_name] = temp_series
            df_train_prepared[self.intensity_col_name].fillna(0, inplace=True)

        # 4. Determine quantile edges for RISK_LEVEL based on these train_stkde_intensities
        unique_intensities = df_train_prepared[self.intensity_col_name].nunique()
        if df_train_prepared[self.intensity_col_name].empty or unique_intensities == 0:
            warnings.warn("No valid STKDE intensity values in training data to determine quantiles. Defaulting to a single bin.")
            self.quantile_edges_ = np.array([-np.inf, np.inf])
        elif unique_intensities < self.n_classes:
            warnings.warn(f"Only {unique_intensities} unique STKDE intensity values in training data. May result in fewer than {self.n_classes} risk levels.")
            _, self.quantile_edges_ = pd.qcut(df_train_prepared[self.intensity_col_name], q=self.n_classes, labels=False, retbins=True, duplicates='drop')
        else:
            _, self.quantile_edges_ = pd.qcut(df_train_prepared[self.intensity_col_name], q=self.n_classes, labels=False, retbins=True, duplicates='drop')
        
        if not np.isneginf(self.quantile_edges_[0]):
            self.quantile_edges_ = np.concatenate(([-np.inf], self.quantile_edges_[1:]))
        if not np.isposinf(self.quantile_edges_[-1]):
            self.quantile_edges_ = np.concatenate((self.quantile_edges_[:-1], [np.inf]))
        
        if self.intensity_col_name in df_train_prepared.columns:
            df_train_prepared.drop(columns=[self.intensity_col_name], inplace=True)
        if '__temp_stkde_datetime_internal__' in df_train_prepared.columns:
            df_train_prepared.drop(columns=['__temp_stkde_datetime_internal__'], inplace=True)

        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, 'fitted_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X to STKDEAndRiskLabelTransformer.transform must be a pandas DataFrame.")

        X_transformed_outer = X.copy()
        original_indices = X_transformed_outer.index
        
        new_coords_rad, new_datetime_np, df_new_prepared_with_valid_rows = self._prepare_data_for_stkde(X_transformed_outer)

        X_transformed_outer[self.intensity_col_name] = np.nan
        X_transformed_outer[self.label_col_name] = -1 

        if new_coords_rad is None or df_new_prepared_with_valid_rows.empty:
            warnings.warn("STKDEAndRiskLabelTransformer.transform called with data that results in no valid points. Outputting with NaNs/defaults.")
            if '__temp_stkde_datetime_internal__' in X_transformed_outer.columns:
                X_transformed_outer.drop(columns=['__temp_stkde_datetime_internal__'], inplace=True, errors='ignore')
            return X_transformed_outer

        new_stkde_intensities = self._calculate_stkde_for_set(
            target_coords_rad=new_coords_rad, 
            target_datetime_np=new_datetime_np,
            reference_coords_rad=self.train_reference_coords_rad_, 
            reference_datetime_np=self.train_reference_datetime_np_,
            reference_balltree=self.train_reference_balltree_
        )

        if len(new_stkde_intensities) == len(df_new_prepared_with_valid_rows):
            df_new_prepared_with_valid_rows[self.intensity_col_name] = new_stkde_intensities
        else:
            warnings.warn("Mismatch in lengths for new_stkde_intensities and df_new_prepared_with_valid_rows in transform. Filling with NaN.")
            temp_series = pd.Series(new_stkde_intensities, index=df_new_prepared_with_valid_rows.index[:len(new_stkde_intensities)])
            df_new_prepared_with_valid_rows[self.intensity_col_name] = temp_series

        df_new_prepared_with_valid_rows[self.label_col_name] = pd.cut(
            df_new_prepared_with_valid_rows[self.intensity_col_name],
            bins=self.quantile_edges_,
            labels=False,
            include_lowest=True,
            right=True
        )
        df_new_prepared_with_valid_rows[self.label_col_name].fillna(0, inplace=True)
        df_new_prepared_with_valid_rows[self.label_col_name] = df_new_prepared_with_valid_rows[self.label_col_name].astype(int)

        # Map results back to original X_transformed_outer using original indices of valid rows
        temp_dt_col_for_transform_map = '__temp_dt_transform_map_idx__'
        X_map = X.copy()
        for col_component in [self.year_col, self.month_col, self.day_col, self.hour_col]:
            X_map[col_component] = pd.to_numeric(X_map[col_component], errors='coerce')
        X_map[temp_dt_col_for_transform_map] = pd.to_datetime(
            X_map[[self.year_col, self.month_col, self.day_col, self.hour_col]]
                .rename(columns={self.year_col:'year', self.month_col:'month', 
                                 self.day_col:'day', self.hour_col:'hour'}), errors='coerce'
        )
        valid_rows_mask_in_X = X_map[temp_dt_col_for_transform_map].notna() & \
                               X_map[self.lat_col].notna() & X_map[self.lon_col].notna()
        original_indices_of_valid_rows = X_map[valid_rows_mask_in_X].index

        if len(original_indices_of_valid_rows) == len(df_new_prepared_with_valid_rows):
            # Use .values to prevent index alignment issues when assigning
            X_transformed_outer.loc[original_indices_of_valid_rows, self.intensity_col_name] = df_new_prepared_with_valid_rows[self.intensity_col_name].values
            X_transformed_outer.loc[original_indices_of_valid_rows, self.label_col_name] = df_new_prepared_with_valid_rows[self.label_col_name].values
        else:
            warnings.warn(f"Warning: Mismatch between number of valid original indices ({len(original_indices_of_valid_rows)}) and processed rows ({len(df_new_prepared_with_valid_rows)}) in transform. Results may be misaligned.")

        return X_transformed_outer

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out if the transformer was not fit on a DataFrame with column names.")
        
        output_features = list(input_features)
        if self.intensity_col_name not in output_features:
            output_features.append(self.intensity_col_name)
        if self.label_col_name not in output_features:
            output_features.append(self.label_col_name)
        return np.array(output_features, dtype=object)

class TargetEngineeringPipeline(BaseEstimator, TransformerMixin):
    '''
    A pipeline that first applies a target engineering step (like STKDEAndRiskLabelTransformer)
    to generate new target-related features (and potentially the actual target for the model),
    and then applies a feature processing pipeline to the features *before* they were augmented
    by the target engineer, using the *newly generated target* for fitting the main classifier.
    '''
    def __init__(self, target_engineer, feature_pipeline):
        self.target_engineer = target_engineer
        self.feature_pipeline = feature_pipeline
        self.target_engineer_ = None
        self.feature_pipeline_ = None
        self.fitted_ = False
        self._y_train_engineered_for_fit = None 

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X to TargetEngineeringPipeline.fit must be a pandas DataFrame.")

        self.target_engineer_ = self.target_engineer.fit(X, y)
        X_train_augmented = self.target_engineer_.transform(X)
        
        engineered_target_col_name = self.target_engineer_.label_col_name
        if engineered_target_col_name not in X_train_augmented.columns:
            raise ValueError(f"Engineered target column '{engineered_target_col_name}' not found after target_engineer.transform.")
        self._y_train_engineered_for_fit = X_train_augmented[engineered_target_col_name]

        self.feature_pipeline_ = self.feature_pipeline.fit(X, self._y_train_engineered_for_fit)
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, 'fitted_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X to TargetEngineeringPipeline.transform must be a pandas DataFrame.")
        return self.feature_pipeline_.transform(X)

    def predict(self, X):
        check_is_fitted(self, 'fitted_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X to TargetEngineeringPipeline.predict must be a pandas DataFrame.")
        return self.feature_pipeline_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, 'fitted_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X to TargetEngineeringPipeline.predict_proba must be a pandas DataFrame.")
        return self.feature_pipeline_.predict_proba(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'fitted_')
        if hasattr(self.feature_pipeline_, 'get_feature_names_out'):
            return self.feature_pipeline_.get_feature_names_out(input_features)
        elif input_features is not None:
            return np.array(input_features, dtype=object)
        else:
            raise AttributeError(
                "The underlying feature_pipeline does not implement get_feature_names_out, "
                "and input_features were not provided to TargetEngineeringPipeline. "
                "Cannot determine output feature names."
            )

    @property
    def classes_(self):
        check_is_fitted(self, 'fitted_')
        if hasattr(self.feature_pipeline_, 'classes_'):
            return self.feature_pipeline_.classes_
        if isinstance(self.feature_pipeline_, Pipeline) and hasattr(self.feature_pipeline_.steps[-1][1], 'classes_'):
            return self.feature_pipeline_.steps[-1][1].classes_
        raise AttributeError("The final estimator in the feature_pipeline does not have classes_ attribute.")
