# Crime Analyzer

A comprehensive crime analysis platform that provides insights and predictions for both law enforcement and tourists in urban environments. The project combines machine learning models, data visualization, and web applications to analyze crime patterns and assess safety risks.

## Features

- **Police Intelligence Dashboard**: Static web dashboard for law enforcement with hotspot maps, crime pattern analysis, and operational insights
- **Tourist Safety Prediction API**: FastAPI service that predicts crime risk for tourists using machine learning models with SHAP explanations
- **Machine Learning Pipeline**: Complete workflow including data preprocessing, feature engineering, model training, and evaluation
- **Spatial and Temporal Analysis**: Clustering algorithms for identifying crime hotspots and temporal patterns
- **Pattern Analysis**: Rule-based insights for understanding crime trends and correlations
- **Interactive Visualizations**: Jupyter notebooks with comprehensive analysis and visualization capabilities

## Repository Structure

```
crime-analyzer/
├── Application/                    # Web applications
│   ├── Police/                     # Police intelligence dashboard
│   │   ├── app.js                  # Dashboard JavaScript
│   │   ├── index.html              # Main dashboard page
│   │   ├── styles.css              # Dashboard styling
│   │   ├── logo_police.png         # Police logo
│   │   └── README.md               # Police app documentation
│   └── Tourists/                   # Tourist safety API
│       ├── app.py                  # FastAPI application
│       ├── docker-compose.yml      # Docker configuration
│       ├── Dockerfile              # Container definition
│       ├── pattern_insights.py     # Pattern insights module
│       ├── requirements.txt        # Python dependencies
│       └── README.md               # Tourist API documentation
├── Data/                           # Raw data files
├── DataIntegrated/                 # Integrated datasets
├── Documents/                      # Project documentation
│   ├── Project Guidelines.pdf      # Project guidelines
│   ├── Project Proposal.pptx       # Initial proposal
│   ├── UserGuidePolice.pdf         # Police user guide
│   ├── UserGuideTourist.pdf        # Tourist user guide
│   ├── NYPD_Complaint_Historic_DataDictionary.xlsx  # Data dictionary
│   ├── PDCode_PenalLaw.xlsx        # Penal law codes
│   └── figures/                    # Documentation figures
├── JupyterOutputs/                 # ML model outputs and results
│   ├── Classification (Final)/     # Final trained models
│   ├── Classification (Modeling)/  # Model selection results
│   ├── Classification (Preprocessing)/  # Preprocessing artifacts
│   ├── Classification (Tuning)/    # Hyperparameter tuning results
│   ├── Clustering/                 # Clustering analysis outputs
│   ├── PatternAnalysis/            # Pattern analysis results
│   └── VisualizationPreprocessing/ # Visualization data
├── Notebooks/                      # Jupyter notebooks
│   ├── Classification/             # Classification notebooks
│   ├── Clustering/                 # Clustering notebooks
│   ├── General/                    # General analysis notebooks
│   └── PatternAnalysis/            # Pattern analysis notebooks
├── tools/                          # Utility scripts
│   └── pattern_analysis_profile.py # Pattern analysis profiling
├── GENERAL_TODO.md                 # Project task list
└── README.md                       # This file
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- Docker Desktop (for Tourist API)
- Dataset can be downloaded from [NYPD Complaint Historic Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data) and [NYPD Complaint Data Current (Year To Date)](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243/about_data)

### Police Dashboard Setup

1. Navigate to the repository root
2. Start a static HTTP server:

   **Option A: Python**
   ```powershell
   python -m http.server 8000
   ```

3. Open `http://localhost:8000/Application/Police/index.html` in your browser

### Tourist Safety API Setup

1. Navigate to `Application/Tourists/`
2. Ensure Docker Desktop is running
3. Build and start the service:

   ```powershell
   docker compose up --build
   ```

4. Access the API documentation at `http://localhost:8000/docs`

### Python Dependencies

For development work with notebooks or tools:

```powershell
pip install -r Application/Tourists/requirements.txt
```

## Usage

### Police Dashboard

- View crime hotspot maps
- Analyze temporal patterns
- Review key performance indicators (KPIs)
- Access clustering insights and recommendations

### Tourist Safety API

Send POST requests to `/api/v1/predict` with crime incident data to receive:

- Risk classification (HIGH_RISK/LOW_RISK)
- Confidence scores
- SHAP-based feature explanations
- Neighborhood and temporal trend insights

**Example API Usage:**

```powershell
# Single prediction
$body = @{
  BORO_NM = "BROOKLYN"
  LOC_OF_OCCUR_DESC = "OUTSIDE"
  VIC_AGE_GROUP = "25-44"
  VIC_RACE = "WHITE"
  VIC_SEX = "M"
  Latitude = 40.6782
  Longitude = -73.9442
  # ... additional fields
}

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body ($body | ConvertTo-Json) -ContentType "application/json"
```

### Jupyter Notebooks

Run the notebooks in `Notebooks/` for:

- Data exploration and preprocessing
- Model training and evaluation
- Clustering analysis
- Pattern discovery
- Visualization generation

## Project Components

### Machine Learning Pipeline

- **Preprocessing**: Data cleaning, feature engineering, encoding
- **Classification**: Logistic Regression model for risk prediction
- **Clustering**: Spatial and temporal crime pattern identification
- **Pattern Analysis**: Association rule mining for crime correlations

### Data Sources

- NYPD Complaint Historic Data
- Geographic and POI (Points of Interest) data
- Temporal and seasonal indicators

### Output Formats

- Trained models (`.joblib`)
- Preprocessing pipelines
- Interactive HTML maps
- JSON configuration files
- Evaluation reports and visualizations