## Process Landscape — Crime Analyzer (Tourist Classifier)

This diagram riassume il flusso end‑to‑end del progetto, seguendo la struttura dell'immagine fornita e mappandola agli artifact presenti nella repo (preprocessing, modeling, tuning, API di predizione).

> Riferimenti chiave in repo
> - Preprocessing e integrazione dati: `Notebooks/General`, `Notebooks/Classification` → output in `JupyterOutputs/Classification (Preprocessing)/`
> - Model selection/tuning/test: `Notebooks/Classification` → output in `JupyterOutputs/Classification (Modeling|Tuning|Final)/`
> - Pattern mining per insight: `Notebooks/PatternAnalysis` + `tools/pattern_analysis_profile.py` → output in `JupyterOutputs/PatternAnalysis/`
> - Servizio inference: `Application/Tourists/app.py` (FastAPI) + `pattern_insights.py`

```mermaid
%%{init: {"flowchart": {"curve": "stepBefore", "htmlLabels": false}} }%%
flowchart TB
  %% Process Landscape for Crime Analyzer (Tourist Classifier)
  %% Top-level stages as subgraphs mirroring the provided image

  %% 1) CONFIGURE SERVICE
  subgraph CFG["CONFIGURE SERVICE"]
    direction LR
    cfg1(["Receive order / request"]) -->|1:1| cfg2(["Configure service\n- docker-compose + .env\n- Mount JupyterOutputs volume\n- Set MODEL_PATH / PREPROCESSOR_PATH / FINAL_PIPELINE_PATH / OPTIMAL_THRESHOLD / PATTERN_OUTPUT_DIR"])
  end

  %% 2) PREPARE SESSION (DATA)
  subgraph PREP["PREPARE SESSION"]
    direction LR
    p1(["Receive records\n- NYPD + POI + temporal"]) -->|1:1| p2(["Create raw session\n- Integrate & clean (Notebooks/General)\n- Scrivi JupyterOutputs/Final/final_crime_data.csv"]) -->|1:M| p3(["Create prepared session\n- Classification (Preprocessing)\n- Salva preprocessing_pipeline_general.joblib, feature_names.json, scaler_info.json, window_config.json"])
  end

  %% 3) GENERATE LEARNING SETS
  subgraph GLS["GENERATE LEARNING SETS"]
    direction LR
    g1(["Collect prepared sessions\n- final_crime_data.csv"]) -->|1:1| g2(["Check session quality\n- Balance classi, leakage check\n- feature_selection_benchmark_all_folds.csv"])
    g2 -->|1:M| g3(["Generate learning sets\n- Train/Test split + CV folds\n- Persist STKDE labels, window_config, scaler_info"])
  end

  %% 4) DEVELOP CLASSIFIER
  subgraph DEV["DEVELOP CLASSIFIER"]
    direction LR
    d1(["Train classifier\n- Notebooks/Classification (Modeling)\n- model_selection_*.json, feature ranking"]) -->|N:1| d2(["Validate classifier\n- Cross-val metrics\n- model_selection_results.csv"])
    d2 -->|1:1| d3(["Test & package classifier\n- Hold-out test (Final)\n- Export LogisticRegression_production_model.joblib, LogisticRegression_final_pipeline.joblib, LogisticRegression_optimal_threshold.json"])
  end

  %% 5) CLASSIFY SESSION (INFERENCE SERVICE)
  subgraph CLS["CLASSIFY SESSION"]
    direction LR
    c1(["Receive session\n- POST /api/v1/predict\n- Batch o singolo record"]) -->|1:1| c2(["Classify session\n- Application/Tourists/app.py\n- Carica pipeline o MODEL+PREPROCESSOR\n- Applica OPTIMAL_THRESHOLD"])
    c2 -->|1:M| c3(["Provide label\n- HIGH_RISK/LOW_RISK + confidence\n- Top features (coeff-based)\n- Context rules da PatternAnalysis (pattern_insights.py)"])
  end

  %% 6) EVALUATE CLASSIFIER PERFORMANCE
  subgraph EVAL["EVALUATE CLASSIFIER PERFORMANCE"]
    direction LR
    e1(["Compare labels\n- Predictions vs y_test"]) -->|1:1| e2(["Evaluate classifier\n- LogisticRegression_evaluation_report.json\n- Dashboard + SHAP *.png"])
  end

  %% Stage flow
  CFG ===> PREP ===> GLS ===> DEV ===> CLS ===> EVAL
```

Note:
- Gli step e gli artefatti citati corrispondono a file/cartelle già presenti in `JupyterOutputs/` (es. `preprocessing_pipeline_general.joblib`, `feature_names.json`, `LogisticRegression_production_model.joblib`, `LogisticRegression_optimal_threshold.json`).
- Gli insight contestuali (regole per BORO_NM/TIME_BUCKET) provengono da `JupyterOutputs/PatternAnalysis/` (`rule_catalog.json`, `global_insights.txt`, file per borough/time bucket) e sono caricati da `pattern_insights.py`.
- Il servizio FastAPI restituisce anche spiegazioni (top features) e trend contestuali (regole per BORO_NM/TIME_BUCKET) tramite `pattern_insights.py`.
