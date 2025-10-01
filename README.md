# Crime Analyzer

A streamlined crime analysis platform focused on tourist safety risk prediction. The project combines a production-ready FastAPI service, machine learning artifacts, and notebooks for analysis and model development.

## Features

- Tourist Safety Prediction API: FastAPI service that predicts crime risk for tourists using a Logistic Regression model with lightweight explanations and contextual trends
- Machine Learning Pipeline: End-to-end workflow including data preprocessing, feature engineering, model training, and evaluation
- Pattern Analysis: Rule-based insights enriching predictions with neighborhood and temporal trends
- Notebooks and Outputs: Reproducible notebooks and exported artifacts for development and validation

## Repository Structure

```
crime-analyzer/
├── Application/
│   └── Tourists/
│       ├── app.py                  # FastAPI application
│       ├── docker-compose.yml      # Docker configuration (exposes :8000)
│       ├── Dockerfile              # Container definition
│       ├── pattern_insights.py     # Pattern insights module (uses JupyterOutputs)
│       ├── requirements.txt        # Python dependencies
│       └── README.md               # Tourist API documentation
├── Data/                           # Raw data (optional/local)
├── Documents/                      # Project documentation
│   ├── UserGuideTourist.pdf
│   ├── NYPD_Complaint_Historic_DataDictionary.xlsx
│   ├── PDCode_PenalLaw.xlsx
│   └── figures/
├── JupyterOutputs/                 # ML model artifacts and results
│   ├── Classification (Final)/
│   ├── Classification (Modeling)/
│   ├── Classification (Preprocessing)/
│   ├── Classification (Tuning)/
│   ├── PatternAnalysis/
│   ├── VisualizationPreprocessing/
│   ├── DataIntegrated/
│   ├── FeatureEngineered/
│   ├── Final/
│   ├── Merged/
│   ├── PrePreProcessed/
│   ├── Processed/
│   └── Raw/
├── Notebooks/                      # Jupyter notebooks
│   ├── Classification/
│   ├── General/
│   └── PatternAnalysis/
├── tools/
│   └── pattern_analysis_profile.py
└── README.md
```

Note: The previous Police Intelligence Dashboard and clustering components have been removed.

## Installation and Setup

### Prerequisites

- Python 3.10+
- Docker Desktop (for running the Tourist API)
- Dataset can be downloaded from: [NYPD Complaint Historic Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data) and [NYPD Complaint Data Current (Year To Date)](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243/about_data)

### Tourist Safety API (Docker)

1. Navigate to `Application/Tourists/`
2. Ensure Docker Desktop is running
3. Build and start the service:

   ```powershell
   docker compose up --build
   ```

4. Open `http://localhost:8000/docs` for interactive API docs

The container mounts the repo at `/workspace` to load model artifacts from `JupyterOutputs`.

### Python Dependencies (local dev)

For working with notebooks or running utilities locally:

```powershell
pip install -r Application/Tourists/requirements.txt
```

## Usage

### Tourist Safety API

Send POST requests to `/api/v1/predict` with incident context to receive:

- Risk classification (HIGH_RISK/LOW_RISK)
- Confidence score (positive-class probability)
- Top contributing features (best-effort for linear models)
- Neighborhood/time-bucket trend insights

Example (PowerShell, single record):

```powershell
$body = @{
  BORO_NM = "BROOKLYN"
  LOC_OF_OCCUR_DESC = "OUTSIDE"
  VIC_AGE_GROUP = "25-44"
  VIC_RACE = "WHITE"
  VIC_SEX = "M"
  Latitude = 40.6782
  Longitude = -73.9442
  BAR_DISTANCE = 120
  NIGHTCLUB_DISTANCE = 500
  ATM_DISTANCE = 80
  ATMS_COUNT = 2
  BARS_COUNT = 3
  BUS_STOPS_COUNT = 1
  METROS_COUNT = 0
  NIGHTCLUBS_COUNT = 0
  SCHOOLS_COUNT = 1
  METRO_DISTANCE = 300
  MIN_POI_DISTANCE = 30
  AVG_POI_DISTANCE = 150
  MAX_POI_DISTANCE = 600
  TOTAL_POI_COUNT = 7
  POI_DIVERSITY = 4
  POI_DENSITY_SCORE = 0.45
  HOUR = 13
  DAY = 15
  WEEKDAY = "MONDAY"
  IS_WEEKEND = 0
  MONTH = 5
  YEAR = 2023
  SEASON = "SPRING"
  TIME_BUCKET = "AFTERNOON"
  IS_HOLIDAY = 0
  IS_PAYDAY = 0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body $body -ContentType "application/json"
```

### Jupyter Notebooks

Use notebooks in `Notebooks/` for:

- Data exploration and preprocessing
- Model training and evaluation
- Pattern discovery and validation
- Generating analysis figures and reports

## Project Components

### Machine Learning Pipeline

- Preprocessing: Data cleaning, feature engineering, encoding
- Classification: Logistic Regression model for risk prediction
- Pattern Analysis: Association rule mining for crime correlations (used for trend enrichment)

### Data Sources

- NYPD Complaint Historic Data
- Geographic and POI (Points of Interest) data
- Temporal and seasonal indicators

### Output Formats

- Trained models (`.joblib`)
- Preprocessing pipelines and feature name lists
- JSON configuration and report files
- Evaluation reports and visualizations