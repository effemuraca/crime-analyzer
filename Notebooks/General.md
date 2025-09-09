# Notebook Index - General Folder

This document provides a structured index for each notebook in the General folder, following the data preprocessing pipeline order.

---

## 1. PrePreProcessing.ipynb

**Objective:** Initial preparation and merging of NYC Crime datasets

### Sections:
1. **Setup**
   - Import base libraries (pandas, numpy, os, warnings, matplotlib, seaborn)
   - Define dataset paths for historic and year-to-date datasets
   - Configure output directories

2. **Data Loading and Merging**
   - Load NYPD historical dataset from `JupyterOutputs/Raw/NYPD_Complaint_Data_Historic_*.csv`
   - Load current year dataset from `JupyterOutputs/Raw/NYPD_Complaint_Data_Current__Year_To_Date__*.csv`
   - Align columns between datasets (handle missing columns with NaN)
   - Concatenate datasets with proper indexing
   - Convert CMPLNT_FR_DT to datetime format
   - Filter for dates from 2020 onwards
   - Save unified dataset to `JupyterOutputs/Merged/NYPD_Complaints_Merged.csv`

3. **Data Quality Check**
   - Verify unified dataset structure and column alignment
   - Check for duplicates and data integrity
   - Preliminary size analysis and shape validation
   - Export to `JupyterOutputs/PrePreProcessed/cleaned_crime_data.csv`

---

## 2. Preprocessing_(Data_Cleaning).ipynb

**Objective:** Comprehensive data cleaning and standardization

### Sections:
1. **Setup**
   - Import libraries (pandas, numpy, matplotlib, seaborn, warnings, re, datetime, Counter)
   - Import sklearn utilities (Pipeline, StandardScaler, LabelEncoder, SimpleImputer)
   - Configure environment and suppress warnings

2. **Helper Functions**
   - `analyze_missing_values()`: Comprehensive missing values analysis with percentages
   - `detect_placeholder_values()`: Detect common placeholder values ('NA', 'NULL', 'UNKNOWN', etc.)
   - `detect_string_anomalies()`: Identify empty strings, whitespace, and special characters
   - Define analysis utilities for data profiling

3. **Data Loading & Initial Analysis**
   - Load unified dataset from `JupyterOutputs/PrePreProcessed/cleaned_crime_data.csv`
   - Initial structural analysis and data validation
   - Identify problematic columns and data quality issues

4. **Missing Values Analysis**
   - Comprehensive missing values analysis with helper functions
   - Missing values pattern visualizations
   - Management strategies for each column type

5. **Data Type Optimization**
   - Convert to appropriate data types for memory efficiency
   - Memory optimization and data type validation
   - Handle dates and timestamps properly

6. **Categorical Data Cleaning**
   - Standardize categories to uppercase
   - Handle inconsistent values and placeholder detection
   - Preliminary mapping and encoding strategies

7. **Geographical Data Cleaning**
   - Validate geographical coordinates (Latitude, Longitude)
   - Correct anomalous lat/long values outside NYC bounds
   - Handle missing spatial data with appropriate strategies

8. **Temporal Data Cleaning**
   - Parse and validate dates using datetime functions
   - Handle temporal inconsistencies and invalid dates
   - Create base temporal features for analysis

9. **Outlier Detection & Treatment**
   - Identify statistical outliers in numerical features
   - Analyze geographical outliers outside expected ranges
   - Treatment strategies for different outlier types

10. **Data Validation & Quality Assurance**
    - Final integrity checks and validation rules
    - Validate business rules and data consistency
    - Generate comprehensive data quality report

11. **Export Cleaned Data**
    - Save cleaned dataset to `JupyterOutputs/Processed/cleaned_crime_data_processed.csv`
    - Document all applied transformations and cleaning steps
   - Downstream notebooks expect final dataset at `JupyterOutputs/Final/final_crime_data.csv`

---

## 3. DataIntegration.ipynb

**Objective:** Integration with external data and information enrichment

### Sections:
1. **Setup**
   - Import base libraries (pandas, numpy, os, warnings, json)
   - Define paths for processed data and output directories
   - Configure integration workflow

2. **Load Cleaned Data**
   - Load cleaned crime dataset from `JupyterOutputs/Processed/cleaned_crime_data_processed.csv`
   - Perform basic data validation and error handling
   - Display dataset overview, statistics, and sample data

3. **OSM-based Spatial Enrichment Design**
   - **Projection**: EPSG:32618 (UTM 18N) for meter-accurate distances
   - **Context Unit**: Regular 100m grid to avoid MAUP bias
   - **POI Sets**: Bars, nightclubs, ATMs, bus stops, metro/train stations, schools
   - **Features**: Distances to nearest POI by type (meters), counts per grid cell
   - **Tooling**: QGIS + QuickOSM + NNJoin workflow
   - **Reproducibility**: Enriched table exported as CSV for notebook consumption

4. **Data Integration Process**
   - Load QGIS-enriched dataset from `JupyterOutputs/DataIntegrated/integrated_crime_data.csv`
   - Validate spatial and POI features from external enrichment
   - Standardize columns and create model-agnostic derived features
   - Intelligent column detection and validation

5. **Spatial Feature Validation**
   - Validate POI distance features (BAR_DISTANCE, NIGHTCLUB_DISTANCE, etc.)
   - Check POI count features (ATMS_COUNT, BARS_COUNT, etc.)
   - Assess spatial context features and density metrics
   - Ensure geographic coordinate consistency

6. **Feature Processing and Aggregation**
   - Basic feature engineering for distance aggregations
   - Density metrics calculation and validation
   - Safe column standardization with error handling
   - Data validation and quality checks

7. **Export Integrated Data**
   - Final validation and comprehensive quality checks
   - Save cleaned integrated dataset to `JupyterOutputs/DataIntegrated/cleaned_integrated_crime_data.csv`
   - Create processing log at `JupyterOutputs/DataIntegrated/integration_processing_log.json`
   - Document integration methodology and reproducibility notes

---

## 4. FeatureEngineering.ipynb

**Objective:** Advanced feature creation for machine learning

### Sections:
1. **Setup**
   - Import ML libraries (sklearn preprocessing, pipelines, feature_selection)
   - Import specialized libraries (holidays, openpyxl for PD code mapping)
   - Import feature engineering utilities (OneHotEncoder, StandardScaler, PCA, etc.)
   - Configure preprocessing pipeline and transformers

2. **Path Configuration**
   - Set up input path: `JupyterOutputs/DataIntegrated/cleaned_integrated_crime_data.csv`
   - Configure PD codes mapping: `Documents/PDCode_PenalLaw.xlsx`
   - Set output path: `JupyterOutputs/FeatureEngineered/feature_engineered_crime_data.csv`
   - Validate file existence and create directories

3. **Load Integrated Data**
   - Load integrated dataset with comprehensive validation
   - Analyze existing feature structure and data quality
   - Perform initial data profiling and validation

4. **Offense Description Enhancement**
   - Impute OFNS_DESC from PD_CD using official codebook mapping
   - Handle '(NULL)' values with official crime code descriptions
   - Validate and standardize offense descriptions

5. **Temporal Feature Engineering**
   - Extract temporal features from CMPLNT_FR_DT/CMPLNT_FR_TM (HOUR, WEEKDAY, SEASON)
   - Create TIME_BUCKET for period-of-day analysis
   - Cyclical encoding for temporal variables using custom transformers
   - Holiday features using holidays.us library
   - Payday features with heuristics (1st/15th of month)

6. **Demographic Feature Engineering**
   - Suspect-victim interaction features (SAME_AGE_GROUP, SAME_SEX)
   - Advanced demographic encoding with robust placeholder handling
   - Age group and demographic standardization
   - Interaction flags for demographic analysis

7. **Location and Spatial Feature Engineering**
   - Location enrichment using distance-based imputation
   - PREM_TYP_DESC imputation when nearest POI ≤ 30m
   - Consolidate PARKS_NM into PREM_TYP_DESC
   - Fallback rules for street crimes and public spaces
   - Advanced proximity features and spatial context

8. **Feature Transformation and Scaling**
   - Categorical encoding strategies (OneHot, Ordinal)
   - Numerical feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
   - Feature selection using SelectKBest and mutual information
   - Remove correlated features to reduce multicollinearity

9. **Pipeline Creation and Validation**
   - Build complete ML preprocessing pipeline
   - Custom transformers integration and validation
   - Pipeline testing and robustness checks
   - Feature engineering validation

10. **Export Engineered Features**
    - Save feature-engineered dataset with comprehensive validation
    - Export preprocessing artifacts and transformers
    - Document feature engineering process and decisions
    - Create feature metadata and documentation

---

## 5. VisualizationPreprocessing.ipynb

**Objective:** Comprehensive visual analysis and publication-quality visualizations

### Sections:
1. **Setup and Data Loading**
   - Import visualization libraries (matplotlib, seaborn, folium, plotly)
   - Import analysis libraries (scipy.stats, pandas, numpy)
   - Configure plotting styles and parameters for publication quality
   - Load final dataset from `JupyterOutputs/Final/final_crime_data.csv`
   - Comprehensive data validation and quality summary

2. **Crime Type Distribution Analysis**
   - Enhanced pie charts with inequality metrics (Gini coefficient)
   - Crime type bar charts with Pareto analysis
   - Distribution inequality analysis and dominance detection
   - Export to `crime_types_pie_chart.png` and `crime_types_bar_chart.png`

3. **Location Analysis Visualizations**
   - Crime location distribution with auto-adaptation to available columns
   - Location-based pie charts and bar charts
   - Geographic distribution analysis
   - Export to `crime_location_pie_chart_*.png` and `crime_location_bar_chart_*.png`

4. **Temporal Analysis Visualizations**
   - Crime trends over time with adaptive time resolution
   - Seasonal patterns and temporal trends analysis
   - Monthly and yearly distribution visualizations
   - Export to `crime_trends_over_time.png` or `crime_by_month.png`

5. **Hourly and Daily Pattern Analysis**
   - Crime distribution by hour with peak identification
   - Day-hour heatmaps for spatio-temporal patterns
   - Weekday patterns and temporal clustering
   - Export to `crime_by_hour.png` and `crime_heatmap_day_hour.png`

6. **Interactive Spatial Visualizations**
   - Geographic crime density maps using Folium
   - Interactive heatmaps with HeatMap plugin
   - Enhanced spatial visualizations with hotspot identification
   - Export to `crime_heatmap.html`, `enhanced_crime_heatmap.html`, `hotspot_map.html`

7. **Statistical Analysis and Insights**
   - Chi-square contingency testing for categorical associations
   - Distribution analysis with statistical significance testing
   - Non-obvious insights discovery (Pareto effects, geographic hotspots)
   - Comprehensive statistical reporting

8. **Publication-Ready Output Generation**
   - High-resolution figure export (300 DPI)
   - Consistent styling and color schemes
   - Auto-adaptive visualizations based on available columns
   - Comprehensive error handling and safe fallbacks
   - Output directory: `JupyterOutputs/VisualizationPreprocessing/`

---

## 6. DataFinal.ipynb

**Objective:** Final dataset preparation and validation for downstream analysis

### Sections:
1. **Setup**
   - Import final libraries (pandas, numpy, matplotlib, seaborn, json)
   - Import statistical libraries (scipy.stats) for validation testing
   - Configure final environment for production dataset preparation

2. **Load Feature Engineered Data**
   - Load dataset from `JupyterOutputs/FeatureEngineered/feature_engineered_crime_data.csv`
   - Fallback to enhanced variant if primary artifact unavailable
   - Comprehensive data validation and shape verification
   - Memory usage analysis and optimization

3. **Final Data Validation**
   - Comprehensive quality checks and data integrity validation
   - Verify absence of critical missing values
   - Validate feature distributions and data consistency
   - Statistical validation of data quality

4. **Data Export and Artifacts**
   - Export final dataset to `JupyterOutputs/Final/final_crime_data.csv`
   - Optional compression (.gz) for large datasets (>100MB)
   - Generate dataset metadata in `JupyterOutputs/Final/final_dataset_metadata.json`
   - Create sample dataset: `JupyterOutputs/Final/final_crime_data_sample_10000.csv`

5. **Final Dataset Documentation**
   - Complete dataset summary with feature descriptions
   - Data quality metrics and validation results
   - Export statistics and metadata for reproducibility
   - Final dataset preparation summary and recommendations

6. **Pipeline Summary & Documentation**
   - Complete preprocessing pipeline summary
   - Final dataset metrics and characteristics
   - Data lineage and transformation documentation
   - Modeling readiness assessment and recommendations

---

## General Notes

- **Execution Order:** Notebooks must be executed in the indicated order to ensure data availability in subsequent phases
- **Dependencies:** Each notebook depends on the outputs of the previous one
- **Output:** Each notebook produces intermediate datasets saved for subsequent phases
- **Documentation:** Each section includes detailed documentation of applied transformations
