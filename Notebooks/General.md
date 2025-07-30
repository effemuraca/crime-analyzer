# Notebook Index - General Folder

This document provides a structured index for each notebook in the General folder, following the data preprocessing pipeline order.

---

## 1. PrePreProcessing.ipynb

**Objective:** Initial preparation and merging of NYC Crime datasets

### Sections:
1. **Setup**
   - Mount Google Drive
   - Import base libraries (pandas, numpy, os, warnings)
   - Define dataset paths

2. **Data Loading and Merging**
   - Load NYPD historical dataset
   - Load current year dataset
   - Align columns
   - Concatenate datasets
   - Filter for dates from 2020 onwards
   - Save unified dataset

3. **Data Quality Check**
   - Verify unified dataset structure
   - Check for duplicates
   - Preliminary size analysis

---

## 2. Preprocessing_(Data_Cleaning).ipynb

**Objective:** Comprehensive data cleaning and standardization

### Sections:
1. **Setup**
   - Mount Google Drive
   - Import libraries (pandas, numpy, matplotlib, seaborn, warnings, re, datetime, Counter)
   - Define helper functions for analysis

2. **Data Loading & Initial Analysis**
   - Load unified dataset
   - Initial structural analysis
   - Identify problematic columns

3. **Missing Values Analysis**
   - Comprehensive missing values analysis
   - Missing values pattern visualizations
   - Management strategies for each column

4. **Data Type Optimization**
   - Convert to appropriate data types
   - Memory optimization
   - Handle dates and timestamps

5. **Categorical Data Cleaning**
   - Standardize categories
   - Handle inconsistent values
   - Preliminary mapping and encoding

6. **Geographical Data Cleaning**
   - Validate geographical coordinates
   - Correct anomalous lat/long values
   - Handle missing spatial data

7. **Temporal Data Cleaning**
   - Parse and validate dates
   - Handle temporal inconsistencies
   - Create base temporal features

8. **Outlier Detection & Treatment**
   - Identify statistical outliers
   - Analyze geographical outliers
   - Treatment strategies

9. **Data Validation & Quality Assurance**
   - Final integrity checks
   - Validate business rules
   - Data quality report

10. **Export Cleaned Data**
    - Save cleaned dataset
    - Document transformations

---

## 3. DataIntegration.ipynb

**Objective:** Integration with external data and information enrichment

### Sections:
1. **Setup**
   - Mount Google Drive
   - Import base libraries (pandas, numpy, os, warnings)
   - Define cleaned data paths

2. **Load Cleaned Data**
   - Load cleaned crime dataset
   - Perform basic data validation
   - Display dataset overview and statistics

3. **Data Integration – Enrichment with OpenStreetMap POI**
   - Construction of spatial grid and POI extraction from OSM
   - Spatial feature generation (distance and density features)
   - Integration with bars, nightclubs, ATMs, metro stations, schools

4. **Data Processing and Feature Engineering**
   - Intelligent column detection and validation
   - Basic feature engineering (distance aggregations, density metrics)
   - Safe column standardization and data validation

5. **Export Integrated Data**
   - Final validation and quality checks
   - Save cleaned integrated dataset
   - Create processing log and documentation

---

## 4. FeatureEngineering.ipynb

**Objective:** Advanced feature creation for machine learning

### Sections:
1. **Setup**
   - Import ML libraries (sklearn, pandas, numpy)
   - Import specialized libraries (holidays, openpyxl)
   - Configure preprocessing pipeline

2. **Load Integrated Data**
   - Load integrated dataset
   - Analyze existing feature structure

3. **Temporal Feature Engineering**
   - Extract temporal features (hour, weekday, month, season)
   - Cyclical encoding for temporal variables
   - Weekend/holiday features
   - Custom time buckets

4. **Demographic Feature Engineering**
   - Suspect-victim interaction features
   - Advanced demographic encoding
   - Matching features (age, gender)

5. **Geographical Feature Engineering**
   - Geographical density features
   - Spatial clustering
   - Advanced proximity features
   - Area-based aggregations

6. **Categorical Encoding**
   - One-Hot Encoding for nominal categories
   - Ordinal Encoding for ordinal categories
   - Target Encoding when appropriate

7. **Feature Scaling & Normalization**
   - StandardScaler for numerical features
   - RobustScaler for features with outliers
   - MinMaxScaler when needed

8. **Dimensionality Reduction**
   - PCA for high-dimensional features
   - Feature selection with mutual information
   - Remove correlated features

9. **Pipeline Creation**
   - Build complete ML pipeline
   - Validate pipeline
   - Export engineered features

---

## 5. VisualizationPreprocessing.ipynb

**Objective:** Comprehensive visual analysis and ML insights

### Sections:
1. **Comprehensive Data Visualization Report**
   - Feature engineering process overview
   - Introduction to visualizations

2. **Feature Engineering Pipeline Summary**
   - Temporal features summary
   - Demographic features summary
   - Geographical features summary
   - Advanced preprocessing summary

3. **Temporal Analysis Visualizations**
   - Crime distribution by hour/day/month
   - Seasonal patterns and temporal trends
   - Temporal heatmaps

4. **Geographical Analysis Visualizations**
   - Crime density maps
   - Borough/precinct analysis
   - POI and proximity visualizations

5. **Categorical Analysis Visualizations**
   - Crime type distributions
   - Demographic analysis
   - Cross-tabulations and correlations

6. **Feature Correlation Analysis**
   - Correlation matrix
   - Feature importance plots
   - Multicollinearity analysis

7. **ML Readiness Assessment**
   - Feature distributions for ML
   - Target variable balance check
   - Final data quality report

---

## 6. DataFinal.ipynb

**Objective:** Final dataset preparation for machine learning

### Sections:
1. **Setup**
   - Import final libraries
   - Configure ML environment

2. **Load Feature Engineered Data**
   - Load dataset with complete features
   - Validate completeness

3. **Final Data Validation**
   - Final quality checks
   - Verify absence of critical missing values
   - Validate feature distributions

4. **Train-Test Split Preparation**
   - Temporal splitting strategies
   - Class balancing if necessary
   - Split validation

5. **Export ML-Ready Dataset**
   - Save final dataset for ML
   - Create feature metadata
   - Final documentation

6. **Pipeline Summary & Documentation**
   - Complete preprocessing summary
   - Final dataset metrics
   - Modeling recommendations

---

## General Notes

- **Execution Order:** Notebooks must be executed in the indicated order to ensure data availability in subsequent phases
- **Dependencies:** Each notebook depends on the outputs of the previous one
- **Output:** Each notebook produces intermediate datasets saved for subsequent phases
- **Documentation:** Each section includes detailed documentation of applied transformations
