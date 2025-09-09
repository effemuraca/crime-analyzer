# Crime Hotspot Detection - Clustering Analysis

This document provides a structured approach for identifying crime hotspots using clustering techniques. The primary objective is to discover spatial-temporal crime concentration patterns in NYC data.

---

## 1. ClusteringTendency.ipynb

**Objective:** Pre-clustering validation for crime hotspot and categorical pattern analysis, assessing clustering tendency for both spatial and mixed-type features

### Sections:
1. **Setup**
   - Import data manipulation libraries (pandas, numpy)
   - Import clustering validation libraries (sklearn.neighbors for Hopkins statistic, sklearn.preprocessing for scaling)
   - Import visualization libraries (matplotlib) for exploratory analysis
   - Configure random seeds for reproducibility

2. **Path Definition**
   - Define paths for loading final preprocessed crime dataset
   - Configure save directory for validation results
   - Load dataset from `JupyterOutputs/Final/final_crime_data.csv`

3. **Data Loading & Validation**
   - Load and validate preprocessed crime dataset
   - Perform data integrity assessment (shape, memory usage, columns)
   - Check for missing values and data quality issues
   - Validate dataset completeness for clustering analysis

4. **Clustering Tendency Assessment**
   - **Hopkins Statistic Implementation**: Custom implementation for spatial-temporal features
   - **Multi-repetition Stability**: Average Hopkins values over multiple repetitions
   - **Robust Preprocessing**: Z-score standardization followed by min-max scaling
   - **Spatio-Temporal Analysis**: Assess clustering tendency for (Latitude, Longitude, HOUR, WEEKDAY)
   - **Mixed-Type Analysis**: Validate clustering tendency for categorical + numeric subspace

5. **Spatial Features Hopkins Testing**
   - Test clustering tendency specifically for spatial coordinates
   - Validate geographic clustering potential
   - Assess micro-hotspot structure in crime data

6. **Categorical + Numeric Subspace Testing**
   - Assess clustering tendency for mixed-type feature combinations
   - Test offense, premise, temporal bucket, and demographic features
   - Evaluate context scores and limited numeric features

7. **Data Integrity Validation**
   - Check for duplicates and data consistency
   - Validate feature availability and completeness
   - Ensure inputs are suitable for subsequent clustering stages

8. **Results Export and Documentation**
   - Export Hopkins statistics and validation results
   - Document clustering readiness assessment
   - Prepare recommendations for clustering approach

---

## 2. MultidimensionalClusteringAnalysis.ipynb

**Objective:** Comprehensive multidimensional clustering analysis for crime pattern discovery using spatial, temporal, and categorical features with advanced pipeline construction

### Sections:
1. **Setup**
   - Import core data manipulation libraries (pandas, numpy, statistics)
   - Import geographic libraries (shapely.geometry, pyproj for coordinate transformations)
   - Import clustering libraries (sklearn.cluster for KMeans/SpectralClustering, kmodes for categorical clustering)
   - Import custom transformers from Utilities module (CategoricalPreprocessor, MixedFeaturePreprocessor, etc.) located at `Notebooks/Clustering/Utilities/clustering_transformers.py`
   - Configure Google Colab support (optional)

2. **Configure Paths and Custom Utilities**
   - Set up project directory structure and output paths
   - Configure data directories and create output directories
   - Add custom utilities to Python path for transformer access
   - Validate directory structure and accessibility

3. **Configure Analysis Parameters**
   - **Spatial Features**: Define primary spatial coordinates (Latitude, Longitude)
   - **Temporal Features**: Configure temporal analysis features (HOUR, WEEKDAY, MONTH)
   - **Extended Temporal Features**: Full temporal feature set including IS_WEEKEND, SEASON, TIME_BUCKET, holidays
   - **Categorical Features**: Core categorical features (BORO_NM, OFNS_DESC, PREM_TYP_DESC)
   - **Extended Categorical Features**: Complete categorical set including demographics and crime descriptors
   - **Spatial Context Features**: POI-based features (distances, counts, diversity, density scores)
   - **Social Features**: Demographic interaction features (SAME_AGE_GROUP, SAME_SEX)

4. **Cross-Validation Functions**
   - **Custom CV Implementation**: Clustering-specific cross-validation with appropriate metrics
   - **Silhouette Score Evaluation**: Quality assessment using intra vs inter-cluster distances
   - **Multi-fold Validation**: K-fold cross-validation adapted for clustering evaluation
   - **Composite Scoring**: Combined metrics for robust clustering quality assessment

5. **Data Loading & Feature Preparation**
   - Load preprocessed crime dataset with comprehensive feature set
   - Validate feature availability and completeness
   - **Feature Selection Strategy**: Select appropriate features for different clustering approaches
   - **Data Quality Assessment**: Check for missing values and data consistency

6. **Spatial Clustering Pipeline**
   - **Geographic Coordinate Clustering**: Primary spatial clustering using Latitude/Longitude
   - **Spatial Preprocessing**: Coordinate standardization and outlier handling
   - **Spatial Cluster Validation**: Geographic coherence and hotspot identification
   - **Spatial Context Integration**: Incorporate POI features for enhanced spatial analysis

7. **Temporal Clustering Analysis**
   - **Time Pattern Clustering**: Cluster based on temporal features (hour, day, season)
   - **Temporal Preprocessing**: Cyclical encoding and temporal standardization
   - **Extended Temporal Features**: Full temporal feature set clustering
   - **Temporal Cluster Interpretation**: Time-based crime pattern discovery

8. **Categorical Clustering (K-Modes)**
   - **K-Modes Implementation**: Categorical clustering for crime type and location features
   - **Categorical Feature Selection**: Choose relevant categorical features for clustering (use `CategoricalPreprocessor` from utilities)
   - **Mode-based Distance**: Categorical distance metrics for cluster formation
   - **Categorical Cluster Validation**: Assess categorical pattern quality

9. **Mixed-Type Clustering**
   - **Spectral Clustering**: Non-convex clustering for mixed spatial-temporal-categorical features
   - **Mixed Feature Preprocessing**: Handle combined numerical and categorical features (use `MixedFeaturePreprocessor` from utilities)
   - **Affinity Matrix Construction**: Build similarity matrices for spectral clustering
   - **Mixed-Type Validation**: Assess quality of multi-modal clustering

10. **Advanced Clustering Methods**
    - **Spectral Clustering (NJW Algorithm)**: Ng-Jordan-Weiss implementation for non-convex patterns
    - **Multidimensional Feature Integration**: Combine spatial, temporal, and categorical features
    - **Advanced Preprocessing Pipelines**: Custom transformers for mixed-type data
    - **Parameter Optimization**: Grid search for optimal clustering parameters

11. **Pipeline Construction and Optimization**
    - **Modular Pipeline Design**: Sklearn pipelines for preprocessing + clustering
    - **Custom Transformer Integration**: Use domain-specific preprocessing transformers
    - **Cross-Validation Integration**: Pipeline evaluation with clustering-specific CV
    - **Parameter Grid Search**: Systematic optimization of clustering parameters

12. **Clustering Evaluation and Comparison**
    - **Multiple Metric Assessment**: Silhouette, inertia, and custom clustering metrics
    - **Method Comparison**: Compare spatial, temporal, categorical, and mixed approaches
    - **Stability Analysis**: Assess clustering consistency across different parameters
    - **Performance Analysis**: Computational efficiency and scalability assessment

13. **Results Export and Visualization**
    - **Cluster Assignment Export**: Save cluster labels and membership information
    - **Performance Metrics Export**: Document clustering quality metrics
    - **Pipeline Serialization**: Save trained clustering pipelines for deployment
    - **Results Documentation**: Comprehensive analysis documentation

---

## 3. SpatialHotspotAnalysis.ipynb

**Objective:** Specialized geographic crime hotspot identification using density-based clustering methods and spatial validation

### Sections:
1. **Setup**
   - Import spatial clustering libraries (DBSCAN, HDBSCAN, KMeans from sklearn.cluster)
   - Import validation libraries (sklearn.metrics for silhouette analysis)
   - Import visualization libraries (matplotlib, seaborn, plotly/folium for mapping)
   - Import data manipulation libraries (pandas, numpy)
   - Import pipeline libraries (sklearn.pipeline)
   - Configure geographic coordinate systems

2. **Data Loading & Feature Preparation**
   - Load preprocessed crime dataset with spatial-temporal features
   - Validate coordinate accuracy and feature completeness
   - **Feature selection**: Select geographic coordinates for spatial clustering
   - **Scaling**: Apply appropriate scaling to coordinate and distance features

3. **Density-Based Spatial Clustering (DBSCAN)**
   - **Algorithm**: DBSCAN for spatial crime concentration detection
   - **Parameters**: Epsilon (distance threshold) and MinPts optimization
   - **Features**: Latitude-longitude coordinates as primary features
   - **Validation**: Geographic visualization and hotspot coherence

4. **Hierarchical Density Clustering (HDBSCAN)**
   - **Objective**: Handle varying density hotspots
   - **Algorithm**: HDBSCAN for multi-density spatial patterns
   - **Parameter**: MinClusterSize optimization
   - **Comparison**: DBSCAN vs HDBSCAN performance

5. **K-Means for Geographic Zones**
   - **Objective**: Identify fixed number of crime zones
   - **Algorithm**: K-means on spatial coordinates
   - **Optimization**: Elbow method for optimal K
   - **Validation**: Silhouette analysis

6. **Temporal Pattern Integration**
   - **Hour-based clustering**: Peak crime hours identification
   - **Day-of-week patterns**: Weekly crime concentration
   - **Seasonal analysis**: Monthly/quarterly hotspot variations

7. **Hotspot Validation & Characterization**
   - **Geographic validation**: Map visualization of detected hotspots
   - **Crime type profiling**: Dominant crimes per hotspot
   - **Temporal profiling**: Time patterns within hotspots
   - **Statistical validation**: Cluster quality metrics

8. **Production Pipeline Setup**
   - **Spatial hotspot pipeline**: Combine preprocessing + DBSCAN/HDBSCAN for deployment
   - **Geographic zone pipeline**: Standardized K-means pipeline for zone identification
   - **Pipeline validation**: Cross-validation and performance testing
   - **Export models**: Save trained pipelines for operational use

---

## General Notes
- Data source: expects `JupyterOutputs/Final/final_crime_data.csv`. Generate it by running the General notebooks up to DataFinal.
- Custom utilities: clustering pipelines use `Notebooks/Clustering/Utilities/clustering_transformers.py` which provides:
   - `CyclicalTransformer` and `SchemaValidator` for safe temporal encoding (HOUR 0–23, MONTH 1–12, DAY 1–31, WEEKDAY as names).
   - `SpatialProjectionTransformer` to project lat/lon to meters (default CRS EPSG:32618 for NYC; Web Mercator 3857 is an alternative).
   - `LatLonToRadiansTransformer` for angular-distance workflows.
   - `FeatureSelector`, `PCATransformer`, `CategoricalPreprocessor`, `MixedFeaturePreprocessor`, `CategoricalDimensionalityReducer`, and `ColumnBinner` for advanced pipelines.
- Algorithm choices: DBSCAN/HDBSCAN for density hotspots, K-Means for fixed-zone partitioning, K-Modes for categorical patterns, Spectral Clustering for mixed non-convex structures.
- Preprocessing tips:
   - Project to meters before Euclidean-based clustering (DBSCAN/KMeans).
   - Scale numeric features; encode/compact categorical features (OneHot+PCA or MCA) for spectral methods.
   - Validate temporal columns before cyclical transforms.
- Parameter guidance:
   - DBSCAN eps should be in meters if using projected coords; tune with grid or k-distance plot (k≈MinPts).
   - HDBSCAN handles variable density; focus on `min_cluster_size`.
   - For K-Means, use elbow and silhouette; run multiple `n_init`.
- Evaluation and exports: write labels, metrics, and serialized pipelines under `JupyterOutputs/Clustering/` subfolders per notebook; include map-ready outputs when applicable.
- Dependencies: pandas, numpy, scikit-learn, pyproj, shapely (optional for geometry), kmodes (for categorical), hdbscan (optional), matplotlib/seaborn/plotly/folium for visualization.
- Reproducibility: fix `random_state` where supported; document CRS and scaling choices in outputs.