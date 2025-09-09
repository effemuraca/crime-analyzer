# Crime Risk Classification - Binary Classification Analysis

This document provides a structured approach for developing a binary crime risk classification system. The primary objective is to predict crime risk levels (HIGH RISK vs LOW RISK) using Spatio-Temporal Kernel Density Estimation (STKDE) and machine learning techniques.

---

## 1. AlternativePreprocessing.ipynb

**Objective:** Robust preprocessing strategy for classification with emphasis on preventing data leakage

### Sections:
1. **Setup**
   - Import data manipulation libraries (pandas, numpy, typing)
   - Import preprocessing libraries (sklearn.preprocessing, sklearn.compose)
   - Import custom transformers from `Utilities/custom_transformers.py` (file: `Notebooks/Classification/Utilities/custom_transformers.py`)
   - Import model selection utilities (train_test_split, RandomizedSearchCV)
   - Configure file paths, working directories, and random seeds

2. **Path Definition**
   - Configure input path: `JupyterOutputs/Final/final_crime_data.csv`
   - Set up output directory: `JupyterOutputs/Classification (Preprocessing)/`
   - Validate file paths and create necessary directories

3. **Data Loading & Basic Validation**
   - Load feature-engineered dataset with comprehensive validation
   - Perform basic data info analysis and column inspection
   - Validate data integrity and feature completeness
   - Initial data profiling and quality assessment

4. **Feature Engineering Pipeline Design**
   - **Temporal Features**: Cyclical transformations using CyclicalTransformer for hour, day, month
   - **Categorical Encoding**: OneHot encoding for linear models, Ordinal for tree-based models
   - **Numerical Scaling**: StandardScaler and RobustScaler for continuous variables
   - **Custom Transformations**: Domain-specific transformers for crime data processing

5. **STKDE Parameter Optimization**
   - **Bandwidth Selection**: Optimize spatial and temporal bandwidth parameters
   - **Intensity Calculation**: Spatio-Temporal Kernel Density Estimation setup using STKDEAndRiskLabelTransformer
   - **Risk Threshold Determination**: Jenks natural breaks for LOW/HIGH risk classification
   - **Training-Only Optimization**: Ensure parameters derived only from training data to prevent leakage
   - Output column names (from code defaults): intensity_col=`stkde_intensity_engineered`, label_col=`RISK_LEVEL_engineered`

6. **Train-Test Split and Data Leakage Prevention**
   - Perform train/test split BEFORE any transformation to prevent data leakage
   - Temporal considerations for time-based splitting
   - Stratification strategies for balanced risk classes
   - Data validation post-split

7. **Preprocessing Pipeline Construction**
   - **Modular Pipeline Design**: Separate pipelines for different model types (general vs trees)
   - **STKDE Integration**: Include STKDE computation in preprocessing workflow
   - **Pipeline Validation**: Test preprocessing consistency and robustness
   - **Custom Transformer Integration**: Seamless integration of domain-specific transformers

8. **Artifact Export and Documentation**
   - Export preprocessing pipelines: `preprocessing_pipeline_general.joblib`, `preprocessing_pipeline_trees.joblib`
   - Save train-test splits: `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`
   - Export metadata: `feature_names.json`, `scaler_info.json`, `stkde_optimal_params.json`
   - Document preprocessing decisions and parameter choices

---

## 2. Modeling.ipynb

**Objective:** Comprehensive model selection and evaluation for binary crime risk classification

### Sections:
1. **Setup and Initialization**
   - Import machine learning libraries (sklearn ensemble, linear models, metrics)
   - Import specialized libraries (xgboost, lightgbm, imblearn for imbalanced learning)
   - Import custom transformers (STKDEAndRiskLabelTransformer, SlidingWindowSplit)
   - Import visualization libraries (matplotlib, seaborn)
   - Configure random seeds and environment

2. **Path Definition and Artifact Loading**
   - Set up directory structure: `JupyterOutputs/Classification (Preprocessing)/` for inputs
   - Configure output directory: `JupyterOutputs/Classification (Modeling)/`
   - Load preprocessing artifacts from previous stage
   - Load train/test splits and preprocessing pipelines

3. **Data Loading & Target Engineering**
   - Load training data: `X_train.pkl`, `y_train.pkl`
   - **STKDE Label Generation**: Apply STKDEAndRiskLabelTransformer for HIGH/LOW risk labels (label column: `RISK_LEVEL_engineered`)
   - **Windowed Target Engineering**: Create temporal windows for risk assessment
   - **Feature-Label Integration**: Combine features with engineered risk labels
   - **Data Validation**: Ensure label distribution and feature consistency

4. **Model Configuration and Selection**
   - **Baseline Models**: DummyClassifier, GaussianNB, LogisticRegression
   - **Tree-Based Models**: RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier
   - **Ensemble Methods**: BaggingClassifier, AdaBoostClassifier
   - **Gradient Boosting**: XGBClassifier, LGBMClassifier for enhanced performance
   - **Imbalanced Learning**: BalancedRandomForestClassifier, EasyEnsembleClassifier, BalancedBaggingClassifier
   - **Support Vector Machines**: LinearSVC (linear SVM; class_weight='balanced')
   - **Distance-Based**: KNeighborsClassifier with appropriate scaling

5. **Cross-Validation Framework**
   - **Stratified K-Fold**: Balanced validation across risk classes with fixed random_state
   - **Time-Series Split**: Temporal validation using SlidingWindowSplit to respect time order
   - **Custom Metrics**: f1, precision, recall, accuracy, matthews_corrcoef, roc_auc, average_precision_score
   - **Imbalance Handling**: class_weight='balanced' and SMOTE resampling within CV folds
   - **Leakage Prevention**: Scalers/encoders fitted only on train folds

6. **Model Evaluation and Comparison**
   - **Comprehensive Metrics**: Multi-metric evaluation including neg_log_loss
   - **Cross-Validation Results**: Robust evaluation across multiple folds
   - **Performance Analysis**: Statistical significance testing and model comparison
   - **Pipeline Integration**: Preprocessing + model in single pipeline to prevent leakage

7. **Results Export and Documentation**
   - Export metrics per model: `model_selection_results.csv`
   - Per-model summaries: `model_selection_*.json` for each algorithm
   - Auxiliary artifacts: `precomputed_targets.pkl`, `window_config.json`
   - Model performance documentation and comparison analysis

8. **Best Model Selection**
   - **Performance Ranking**: Multi-criteria model selection based on validation metrics
   - **Computational Efficiency**: Training and prediction time analysis
   - **Robustness Assessment**: Consistency across different validation strategies
   - **Model Shortlisting**: Identify top candidates for hyperparameter tuning

---

## 3. TuningAndTraining.ipynb

**Objective:** Advanced hyperparameter optimization and model interpretability analysis

### Sections:
1. **Setup and Configuration**
   - Import optimization libraries (scipy.optimize, sklearn.model_selection for RandomizedSearchCV)
   - Import interpretability libraries (SHAP, LIME for model explanation)
   - Import advanced metrics and visualization tools
   - Load best model and preprocessing artifacts from Modeling.ipynb

2. **Hyperparameter Optimization Strategy**
   - **Parameter Space Definition**: Define search ranges for selected best model
   - **RandomizedSearchCV**: Efficient parameter space exploration with cross-validation
   - **Scoring Strategy**: Multi-metric optimization focusing on F1-score and AUC
   - **Validation Integration**: Use same CV strategy as modeling phase for consistency

3. **Advanced Parameter Tuning**
   - **Grid Search Refinement**: Fine-tune around best RandomizedSearch results
   - **Cross-Validation Consistency**: Maintain stratified K-fold validation approach
   - **Overfitting Prevention**: Monitor validation curves and learning curves
   - **Parameter Sensitivity Analysis**: Assess parameter impact on model performance

4. **Threshold Optimization**
   - **ROC Curve Analysis**: Optimal threshold selection for binary classification
   - **Precision-Recall Trade-off**: Balance precision and recall for crime prediction
   - **Cost-Sensitive Optimization**: Consider misclassification costs for practical deployment
   - **Business Metric Optimization**: Optimize for crime prevention effectiveness metrics

5. **Model Interpretability Analysis**
   - **SHAP Analysis**: Comprehensive Shapley value analysis for feature importance
   - **Global Interpretability**: Overall feature importance and model behavior patterns
   - **Local Interpretability**: Individual prediction explanations for specific cases
   - **Feature Interaction Effects**: Understanding feature combinations and dependencies

6. **Performance Validation and Diagnostics**
   - **Learning Curves**: Model performance vs training set size analysis
   - **Validation Curves**: Parameter sensitivity and model complexity assessment
   - **Bias-Variance Analysis**: Model complexity and generalization assessment
   - **Cross-Validation Stability**: Performance consistency across different data subsets

7. **Production Model Preparation**
   - **Final Model Training**: Train optimized model on full training set
   - **Pipeline Integration**: Combine optimized preprocessing and modeling pipelines
   - **Model Serialization**: Save tuned model and artifacts for deployment
   - **Performance Documentation**: Comprehensive tuning results and model characteristics

8. **Artifact Export and Documentation**
   - Export tuned model: `best_model_info.json` with optimal parameters
   - Save performance metrics and validation results
   - Export SHAP analysis results and interpretability artifacts
   - Document hyperparameter tuning process and optimal configuration

---

## 4. Final.ipynb

**Objective:** Final model training, comprehensive evaluation, and deployment preparation

### Sections:
1. **Setup and Initialization**
   - Import deployment libraries (joblib for model persistence, pickle for data loading)
   - Import comprehensive evaluation tools (sklearn.metrics, matplotlib, seaborn)
   - Import SHAP for final interpretability analysis
   - Load optimized model configuration and parameters from TuningAndTraining.ipynb

2. **Model and Data Loading**
   - Load test data: `X_test.pkl`, `y_test.pkl` from preprocessing stage
   - Load tuned model configuration: `best_model_info.json` and parameter files
   - Validate model artifacts and preprocessing pipeline compatibility
   - Ensure consistent data preprocessing for final evaluation

3. **Final Model Training**
   - **Production Training**: Train final model using optimal parameters on complete training dataset
   - **Pipeline Integration**: Combine preprocessing and modeling into single production pipeline
   - **Model Validation**: Ensure model quality, consistency, and convergence
   - **Training Diagnostics**: Monitor training process and validate model stability

4. **Comprehensive Test Set Evaluation**
   - **Hold-out Testing**: Unbiased evaluation on completely unseen test data
   - **Binary Classification Metrics**: Accuracy, precision, recall, F1-score, specificity
   - **Ranking Metrics**: AUC-ROC, AUC-PR for threshold-independent evaluation
   - **Confusion Matrix Analysis**: Detailed classification performance breakdown
   - **Statistical Significance**: Confidence intervals and performance reliability assessment

5. **Model Diagnostics and Analysis**
   - **Prediction Distribution**: Analyze prediction confidence patterns and score distributions
   - **Error Analysis**: Investigate misclassification patterns and failure modes
   - **Decision Boundary Analysis**: Understand model decision-making patterns
   - **Model Behavior Validation**: Ensure model makes sensible predictions

6. **Interpretability and Explainability**
   - **SHAP Analysis**: Final feature importance analysis using Shapley values
   - **Feature Contribution**: Global and local feature importance assessment
   - **Model Explanation**: Generate interpretable explanations for model decisions
   - **Business Insight Generation**: Translate model insights into actionable crime prevention strategies

7. **Production Artifacts and Deployment**
   - **Model Persistence**: Save final trained model as `*_production_model.joblib`
   - **Evaluation Report**: Export comprehensive evaluation results as JSON
   - **SHAP Visualizations**: Save interpretability plots and summary visualizations
   - **Performance Documentation**: Create detailed model evaluation report

8. **Deployment Readiness Assessment**
   - **Production Validation**: Validate model readiness for operational deployment
   - **Performance Benchmarking**: Compare against baseline and business requirements
   - **Monitoring Setup**: Guidelines for model performance monitoring in production
   - **Deployment Documentation**: Instructions for model integration and usage

9. **Business Impact Assessment**
   - **Crime Risk Prediction Accuracy**: Evaluate effectiveness for law enforcement planning
   - **Resource Allocation Support**: Assess utility for patrol optimization and resource allocation
   - **Operational Recommendations**: Actionable insights for crime prevention strategies
   - **ROI Analysis**: Quantify potential impact and value of model deployment

---

## General Notes
- Data source: expects `JupyterOutputs/Final/final_crime_data.csv` created by the General pipeline. Execute `Notebooks/General` through DataFinal if missing.
- Custom utilities: classifiers rely on `Notebooks/Classification/Utilities/custom_transformers.py` with:
   - `STKDEAndRiskLabelTransformer`: computes `stkde_intensity_engineered` and labels `RISK_LEVEL_engineered` using Jenks thresholds; requires `YEAR, MONTH, DAY, HOUR, Latitude, Longitude`.
   - `CyclicalTransformer`: encodes HOUR/WEEKDAY/MONTH/DAY to sin/cos components with rigorous range validation.
   - `SlidingWindowSplit`: cross-validator for time-respecting evaluation.
- Data leakage control: always split train/test before fitting any transformers; fit STKDE, scalers, and encoders only on training folds.
- STKDE parameters: spatial bandwidth `hs` (meters) and temporal `ht` (days) are tunable; default label thresholds come from Jenks on train intensities. For deterministic labels across runs, fix the random_state and retain learned thresholds.
- Artifacts & paths:
   - Preprocessing exports under `JupyterOutputs/Classification (Preprocessing)/`: pipelines, train/test splits, metadata (feature names, scaler info, STKDE params)
   - Modeling outputs under `JupyterOutputs/Classification (Modeling)/`: per-model JSONs, aggregated CSV, window configs
   - Tuning and Final outputs under `JupyterOutputs/Classification (Tuning)/` and `JupyterOutputs/Classification (Final)/`: `best_model_info.json`, SHAP plots, evaluation report, and `*_production_model.joblib`
- Metrics: use multi-metric scoring (F1, precision, recall, MCC, ROC-AUC, PR-AUC, accuracy; optionally log loss) with stratified/time-series CV.
- Imbalance handling: prefer `class_weight='balanced'` for linear/SVM/tree models; integrate SMOTE only within CV folds to avoid leakage.
- Explainability: SHAP is expected for tree/boosting models; store global and local explanations under the phase output folders.
- Reproducibility: set `random_state=42` consistently; pin package versions in `Application/Tourists/requirements.txt` or project-level requirements if extended.