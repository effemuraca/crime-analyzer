# Crime Pattern Analysis - Association Rule Mining

This document provides a structured approach for discovering spatio-temporal crime patterns using association rule mining techniques. The primary objective is to identify significant patterns in crime data to inform features and policy insights through interpretable association rules.

---

## 1. PatternAnalysis.ipynb

**Objective:** Comprehensive spatio-temporal crime pattern discovery using FP-Growth association rule mining with temporal validation and stability analysis

### Sections:
1. **Setup and Dependencies**
   - Import data manipulation libraries (pandas, numpy, pathlib)
   - Import association rule mining libraries (mlxtend.frequent_patterns for FP-Growth)
   - Import visualization libraries (plotly for interactive charts)
   - Import progress tracking libraries (tqdm for mining progress)
   - Configure deterministic options and environment

2. **Paths, Constants, and Parameters**
   - Configure input/output paths for data pipeline integration
   - Define Top-N category caps (LOC_OF_OCCUR: 20, SUSP_AGE: 10, VIC_AGE: 10)
   - Set distance binning parameters ([0, 250, 1000, ∞] with labels ['<250m', '250-1000m', '>1000m'])
   - Configure FP-Growth parameters (GLOBAL_MIN_SUPPORT: 0.5, MIN_CONFIDENCE: 0.40)
   - Set auto-tuning thresholds (MIN_RULES: 150, MAX_RULES: 800, TOP_K: 50)
    - Optional diagnostics helper: run `tools/pattern_analysis_profile.py` (default input `JupyterOutputs/Final/final_crime_data.csv`) to auto-suggest binning and Top-N caps

3. **Load and Validate Data**
   - Load finalized dataset from `JupyterOutputs/Final/final_crime_data.csv`
   - Apply temporal filter for 2024 data only
   - Create HAS_POI binary feature from TOTAL_POI_COUNT
   - Construct DATE column from YEAR/MONTH/DAY components
   - Validate date construction and handle invalid dates
   - Drop unused columns for performance optimization

4. **Override Parameters from Diagnostics**
   - Load diagnostics profile from `profile_summary.json` if available
   - Apply Top-N suggestions from diagnostics analysis
   - Adopt optimized binning configurations from diagnostics
   - Track and report applied overrides for transparency

5. **Preprocess Categorical Features**
   - **Category Normalization**: Standardize categorical values to uppercase strings
   - **Top-N Capping**: Apply configurable caps with "OTHER" category for long-tail values
   - **Borough Standardization**: Normalize BORO_NM to consistent format
   - **Missing Value Handling**: Replace missing values with "UNKNOWN" category
   - **Feature Mapping**: Create standardized feature set with retained categories tracking

6. **Engineer Time, Distance, and Count Buckets**
   - **Temporal Features**: Standardize TIME_BUCKET from existing column or derive from timestamps
   - **Distance Binning**: Apply configurable distance bins to POI distance features
   - **Numeric Discretization**: Convert continuous features to categorical bins
   - **Feature Validation**: Ensure all engineered features have meaningful distributions

7. **Build Transactional One-Hot Encoding**
   - **Transactional Format**: Convert to "KEY=VALUE" item format for FP-Growth
   - **One-Hot Matrix**: Create boolean matrix suitable for association rule mining
   - **Context Columns**: Add special context columns (__DATE__, __BORO__, __TIME_BUCKET__) for conditional mining
   - **UNKNOWN Filtering**: Remove items with UNKNOWN values to reduce noise

8. **FP-Growth Mining with Auto-Tuning**
   - **Auto-Support Tuning**: Binary search approach to find optimal minimum support
   - **Target Range Control**: Automatically adjust support to achieve 150-800 rules
   - **Robust Mining**: Handle edge cases and small datasets gracefully
   - **Performance Optimization**: Limit itemset length and use column filtering

9. **Generate Association Rule Metrics**
   - **Standard Metrics**: Support, confidence, lift, leverage from mlxtend
   - **Advanced Metrics**: Conviction and Kulczynski for enhanced evaluation
   - **Metric Validation**: Handle edge cases and numerical stability

10. **Prune Redundant Rules and Rank Top-K**
    - **Duplicate Removal**: Eliminate exact duplicate rules using canonical representation
    - **Subsumption Pruning**: Remove rules that are subsumed by more general, stronger rules
    - **Trivial Rule Filtering**: Remove rules containing only trivial terms (HAS_POI, DIST_BIN)
    - **Maximal Marginal Significance (MMS)**: Select diverse top-K rules with marginal coverage
    - **Composite Scoring**: Rank by Kulczynski × √(support) with quality thresholds

11. **Global Mining, Reporting, and Export**
    - **End-to-End Pipeline**: Orchestrate complete mining process with auto-tuning
    - **Multiple Export Formats**: JSON catalog, CSV for analysis, human-readable insights
    - **Quality Metrics**: Track rules found, pruning effectiveness, ranking quality
    - **Artifact Management**: Systematic export of all mining results

12. **Visualize Top-K Global Rules**
    - **Interactive Bar Charts**: Display rules ranked by composite score
    - **Rule Visualization**: Show confidence, lift, and support in hover data
    - **Dynamic Sizing**: Adjust chart height based on number of rules
    - **Color Mapping**: Use confidence for color-coding rule strength

13. **Per-Borough Conditional Rules**
    - **Contextual Mining**: Mine rules specific to each borough
    - **Borough Filtering**: Extract meaningful patterns for individual boroughs
    - **Comparative Analysis**: Enable cross-borough pattern comparison
    - **Context Exclusion**: Avoid trivial rules (e.g., IF X THEN BORO=BROOKLYN when mining Brooklyn)

14. **Per-Time-Bucket Conditional Rules**
    - **Temporal Context Mining**: Discover time-specific crime patterns
    - **Time Bucket Analysis**: Mine rules for different periods of day/week
    - **Temporal Pattern Discovery**: Identify time-dependent crime associations

15. **Temporal Hold-out Validation**
    - **Time-based Split**: Split data chronologically (80% train, 20% test)
    - **Rule Generalization**: Test rule validity on unseen future data
    - **Performance Tracking**: Compare train vs test confidence and support
    - **Temporal Stability**: Assess rule reliability over time

16. **Rolling Window Evaluation and Stability Tracking**
    - **Rolling Windows**: Create overlapping 90-day windows with 30-day steps
    - **Temporal Consistency**: Mine rules in each window for stability analysis
    - **Rule Persistence**: Track which rules appear consistently across windows
    - **Evolution Tracking**: Monitor how rule strength changes over time

17. **Stability Analysis and Comparative Reporting**
    - **Jaccard Similarity**: Measure rule set overlap between consecutive windows
    - **Stability Metrics**: Calculate mean overlap and variation in rule consistency
    - **Temporal Reliability**: Identify most stable vs volatile patterns

18. **Visualize Rule Stability Over Time**
    - **Stability Charts**: Line plots showing Jaccard similarity between windows
    - **Trend Analysis**: Visualize temporal stability patterns
    - **Window Transitions**: Label transitions between time periods

19. **Export Final Rule Catalog and Consolidated Insights**
    - **Master Catalog**: Consolidate global, borough, and time-specific rules
    - **Metadata Tracking**: Include creation timestamps, parameters, and data sources
    - **Validation Results**: Integrate hold-out and stability analysis results
    - **Unified Documentation**: Combine all insights into consolidated readable format

20. **Final Results and Comparative Analysis**
    - **Project Overview**: Document complete pattern mining pipeline
    - **Literature Comparison**: Compare approach with related association rule mining studies
    - **Method Validation**: Highlight adaptive support tuning and temporal validation
    - **Interpretation Guidelines**: Provide context for understanding discovered patterns

---

## General Notes
- Data source: expects `JupyterOutputs/Final/final_crime_data.csv` (from the General pipeline). If missing, run the notebooks in `Notebooks/General` through DataFinal to produce it.
- Year filter: default analysis filters `YEAR == 2024`. You can disable or change the year; the diagnostics tool mirrors this behavior.
- Diagnostics helper: `tools/pattern_analysis_profile.py` profiles the input and writes suggestions to `JupyterOutputs/PatternAnalysis/Diagnostics/`:
    - `profile_summary.json`: presence by feature groups, suggested Top-N caps, TIME_BUCKET status, proposed bins for distances/counts/diversity/density
    - `numeric_stats.csv` and `category_value_counts.json` for quick audits
    - Use these to override caps/bins before mining for better support distribution
- Canonicalization and capping:
    - Uppercase string normalization; missing mapped to `UNKNOWN`
    - Borough normalized to `BORO`
    - Long-tail capping with `OTHER` for high-cardinality categoricals (typical caps suggested by diagnostics: `OFNS_DESC≈30`, `PREM_TYP_DESC≈25`, `LOC_OF_OCCUR_DESC≈20`, `SUSP_AGE_GROUP≈10`, `VIC_AGE_GROUP≈10`)
    - `LAW_CAT_CD` normalized to a compact `LAW_CAT`
- Binning defaults and fallbacks (overridable by diagnostics):
    - Distances: bins `[0, 250, 1000, inf]` → labels `['<250m','250-1000m','>1000m']`
    - Counts: fixed bins `[0, 5, 20, inf]` with labels `['0-4','5-19','20+']`; auto-switch to quantile bins if one bin >90% coverage
    - Diversity: `POI_DIVERSITY` into `['Very Low','Low','Medium','High']`
    - Density: `POI_DENSITY_SCORE` via quartiles (Q1–Q4)
    - `TOTAL_POI_COUNT` also produces `HAS_POI` boolean for trivial-rule filtering
- Transactional encoding: items are `KEY=VALUE` strings, with context columns `__DATE__`, `__BORO__`, `__TIME_BUCKET__` added for conditional mining; drop items with `UNKNOWN` to reduce noise.
- Mining tuning: support is auto-tuned to reach ~150–800 rules; confidence threshold typically ≥0.40; limit itemset length for performance.
- Metrics: compute support, confidence, lift, leverage, conviction, Kulczynski; prune duplicates/subsumed and trivial rules; rank by a composite score (e.g., Kulczynski × sqrt(support)) and select Top-K (e.g., 50) with marginal coverage diversity.
- Exports (suggested): write catalogs and visuals under `JupyterOutputs/PatternAnalysis/` (JSON catalog, CSV, human-readable summaries, Plotly figures).
- Dependencies: pandas, numpy, mlxtend (frequent_patterns), plotly, tqdm; optional: `tools/pattern_analysis_profile.py` (pandas, numpy).
- Reproducibility & performance: fix random seeds where applicable; large datasets benefit from column filtering and limiting max itemset length.