# Crime Hotspot Detection - Clustering Analysis

This document describes the three clustering notebooks and aligns their documentation with the implemented code. Goals: (1) pre-clustering tendency checks, (2) multidimensional pattern discovery across categorical/mixed features, and (3) spatial micro‑hotspot detection with temporal profiling.

---

## 1. ClusteringTendency.ipynb

Objective: Pre‑clustering validation for crime hotspot and categorical pattern analysis, assessing clustering tendency for both spatio‑temporal and mixed‑type subspaces.

### Sections:
1. **Setup**
   - Import data manipulation libraries (pandas, numpy)
   - Import clustering validation libraries (sklearn.neighbors for Hopkins statistic, sklearn.preprocessing for scaling)
   - Import visualization libraries (matplotlib) for exploratory analysis
   - Configure random seeds for reproducibility

2. **Path Definition**
   - Define paths for loading the feature‑engineered dataset
   - Configure save directory for any optional artifacts
   - Load dataset from `JupyterOutputs/Final/final_crime_data.csv`

3. **Data Loading & Validation**
   - Load and validate preprocessed crime dataset
   - Perform data integrity assessment (shape, memory usage, columns)
   - Check for missing values and data quality issues
   - Validate dataset completeness for clustering analysis

4. **Clustering Tendency Assessment**
   - Hopkins statistic (custom): mixed‑type compatible, computed via Gower distance
   - Multi‑repetition stability: average over 5 runs with mean ± std and qualitative label
   - Spatio‑temporal subspace: Latitude, Longitude + cyclical(HOUR, WEEKDAY)
   - Mixed‑type subspace: categorical crime context + limited numeric context (Gower)

5. **Categorical + Numeric Subspace Testing**
   - Assess clustering tendency on categorical (e.g., BORO_NM, OFNS_DESC, PREM_TYP_DESC, TIME_BUCKET, weekend/holiday flags, demographics) plus compact numeric context (e.g., METRO_DISTANCE, POI_DENSITY_SCORE)
   - Treat intended binary/time‑bucket fields as categorical dtype before Gower

6. **Data Integrity Validation**
   - Check for duplicates and data consistency
   - Validate feature availability and completeness
   - Ensure inputs are suitable for subsequent clustering stages

7. **Results and Documentation**
   - Results are printed to console (no CSV/JSON export in code). Optional persistence can be added.
   - Includes qualitative spatial distribution diagnostics (scatter, histograms, heatmap, time/borough summaries)

---

## 2. MultidimensionalClusteringAnalysis.ipynb

Objective: Multidimensional clustering for categorical and mixed features, producing actionable crime pattern profiles and an executive summary for police operations.

### Sections:
1. **Setup**
    - Import core data manipulation libraries (pandas, numpy, statistics)
    - Import clustering libraries (sklearn.cluster for KMeans/SpectralClustering; kmodes for categorical)
    - Import custom transformers from Utilities at `Notebooks/Clustering/Utilities/clustering_transformers.py`:
       - CategoricalPreprocessor, ColumnBinner, CategoricalDimensionalityReducer, GroupBalancedOneHotEncoder

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

4. **Data Loading & Feature Preparation**
   - Load preprocessed crime dataset with comprehensive feature set
   - Validate feature availability and completeness
   - **Feature Selection Strategy**: Select appropriate features for different clustering approaches
   - **Data Quality Assessment**: Check for missing values and data consistency

5. **Categorical Clustering (K‑Modes)**
   - Pipeline: CategoricalPreprocessor → KModes
   - Features: base (BORO_NM, OFNS_DESC, PREM_TYP_DESC) + operational categorical (TIME_BUCKET, IS_WEEKEND, IS_HOLIDAY) + demographics (sex/age only) + POI/context bins via ColumnBinner (e.g., METRO_DISTANCE_BIN, POI_DENSITY_SCORE_BIN)
   - Grid search: n_clusters 6–20, init ∈ {Huang, Cao}, n_init ∈ {5, 10}
   - Evaluation: custom composite score combining dissimilarity and cluster balance; select best and profile clusters

6. **Categorical Dimensionality Reduction + KMeans**
   - Pipeline: CategoricalDimensionalityReducer (OneHot + PCA with group‑balanced encoding) → KMeans
   - Grid: dimred__n_components ∈ {10, 20, 30}; cluster__n_clusters ∈ {8, 12, 16, 20}; cluster__n_init ∈ {10, 20}
   - Metric: silhouette over transformed embedding

7. **Spectral Clustering (mixed representation)**
   - Pipeline: ColumnTransformer(GroupBalancedOneHotEncoder for categoricals; StandardScaler for numericals) → SpectralClustering
   - Grid: n_clusters ∈ {8, 12, 16}, affinity ∈ {rbf, nearest_neighbors} with gamma/n_neighbors sweeps; assign_labels='kmeans'
   - Metric: silhouette on the embedded numeric matrix

8. **Pipeline Construction and Optimization**
   - Modular sklearn pipelines; parameter sweeps via ParameterGrid and custom loops
   - Balanced one‑hot encoding prevents high‑cardinality dominance

9. **Clustering Evaluation and Comparison**
   - Metrics: composite score (K‑Modes), silhouette (DimRed+KMeans, Spectral)
   - Method comparison JSON: best scores/params for K‑Modes, CatDimRed+KMeans, Spectral

10. **Results Export and Visualization**
      - Exports:
         - `JupyterOutputs/Clustering (MultidimensionalClusteringAnalysis)/pipeline_methods_comparison.json`
         - `JupyterOutputs/Clustering (MultidimensionalClusteringAnalysis)/executive_crime_summary.json`
         - Optional: `.../executive_crime_summary_enriched.json`, `.../police_operational_intelligence_enriched.csv`
      - Operational dashboards: concise priority tables and executive summary printed to console; optional CSV/JSON saved

---

## 3. SpatialHotspotAnalysis.ipynb

Objective: Specialized geographic crime hotspot identification using density‑based clustering with temporal encoding, plus optional CLARANS comparison and temporal pattern profiling.

### Sections:
1. **Setup**
   - Import spatial clustering libraries (DBSCAN from sklearn.cluster) and a custom CLARANS (k-medoids) implementation
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

3. **Spatio‑Temporal DBSCAN with k‑distance heuristic**
   - Algorithm: DBSCAN to detect spatial crime concentrations (micro‑hotspots)
   - k‑distance pre‑stage: compute descending k‑distance curves and estimate ε via bucketed slope ≈ −1; build focused ε grid with spreads
   - Projected coordinates: SpatialProjectionTransformer to EPSG:32618 (meters) + CyclicalTransformer for (HOUR, WEEKDAY) with StandardScaler and a temporal weight
   - Parameter sweep: grid over ε (meters), min_samples, and temporal weight; skip configs with <2 clusters
   - Selection criteria (implemented): minimize noise ratio; tie‑break by maximizing silhouette, then number of clusters
   - Outputs: Best pipeline refit; labeled DataFrame `df_labeled` with X_METERS/Y_METERS and cluster IDs

4. **CLARANS (k‑medoids) comparative evaluation**
   - Explore k over a wide grid; multiple local minima searches; select k by silhouette on projected spatial coords; retain total medoid cost as auxiliary
   - Outputs: optional `df_labeled_clar`

5. **Temporal Pattern Integration**
   - Hour/day cyclical features are part of the DBSCAN feature space; downstream temporal profiling classifies clusters (e.g., concentrated_night, weekend_focused)

6. **Hotspot Validation & Characterization**
   - Static scatter plots in meter space, with legends summarizing n, radius, hour mean, weekday mode
   - By zone/borough faceting (BORO_NM) with focus filters (min points, max radius)
   - Temporal pattern analysis: time slots, weekday dominance, diversity, weekend share; operational hotspot recommendations

7. **Production Pipeline Setup**
   - DBSCAN pipeline as above; CLARANS optional comparison
   - Source selection via `CLUSTER_SOURCE` ∈ {'dbscan','clarans'}
   - Exports: interactive map saved to `JupyterOutputs/Clustering (SpatialHotspots)/cluster_temporal_patterns_map.html`; figures/logs inline

8. **Notes**
   - The notebook includes k-distance plots, best-config selection logs, and cluster visualizations by area
   - CLARANS is implemented inline for comparative analysis; use with caution on very large k ranges

---

## General Notes
- Data sources:
   - SpatialHotspotAnalysis, MultidimensionalClusteringAnalysis load `Data/final_crime_data.csv`
   - ClusteringTendency loads `JupyterOutputs/Final/final_crime_data.csv`
- Custom utilities: `Notebooks/Clustering/Utilities/clustering_transformers.py` provides:
   - CyclicalTransformer, SchemaValidator (safe temporal encoding)
   - SpatialProjectionTransformer (project lat/lon → meters, default EPSG:32618)
   - LatLonToRadiansTransformer (angular workflows)
   - FeatureSelector, PCATransformer, CategoricalPreprocessor, CategoricalDimensionalityReducer, ColumnBinner, GroupBalancedOneHotEncoder
- Algorithm choices:
   - DBSCAN for spatio‑temporal hotspots; CLARANS for medoid‑based spatial comparison
   - K‑Modes for categorical patterns; OneHot+PCA→KMeans and Spectral for mixed representations
- Preprocessing tips:
   - Always project to meters for Euclidean spatial clustering
   - Use cyclical encoding for time; treat TIME_BUCKET/weekend/holiday as categorical where appropriate
   - Use group‑balanced one‑hot for categorical features feeding numeric methods to avoid high‑cardinality dominance
- Parameter guidance:
   - DBSCAN: estimate ε via k‑distance slope≈−1; sweep ε spreads, min_samples, temporal weights; prefer ≥2 clusters; selection prioritizes low noise
   - CLARANS: explore k with multiple local minima; select by silhouette; keep cost as diagnostic
   - K‑Means (DimRed pipeline): grid n_clusters, n_init; select by silhouette
- Evaluation and exports:
   - Multidimensional: comparison JSON + executive summaries; optional enriched CSV
   - SpatialHotspot: interactive map HTML + plots; labeled DataFrame in‑kernel
   - Tendency: console outputs for Hopkins + visuals (no export by default)
- Dependencies: pandas, numpy, scikit‑learn, kmodes, pyproj, shapely (optional), gower (for mixed‑type Hopkins), matplotlib/seaborn/plotly/folium
- Reproducibility: fixed `random_state`; document CRS (EPSG:32618) and temporal filtering (e.g., YearMonth ≥ 202401 or ≥ 202411 depending on notebook)