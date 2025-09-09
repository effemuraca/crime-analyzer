# Tourist Safety Prediction API

FastAPI service for tourist-oriented crime risk predictions using the final Logistic Regression model. Inputs are validated, preprocessed, and scored; responses include explanations and contextual trends.

## Endpoint

- POST /api/v1/predict — Unified prediction with explanations and trends
  - Response (single record):
    - label: HIGH_RISK | LOW_RISK
    - confidence: float (positive class probability)
    - threshold: float
    - explanations.top_features: array of { feature, shap }
    - trends.neighborhood: top rules for the input borough (BORO_NM)
    - trends.time_bucket: top rules for the input time bucket (TIME_BUCKET)

For batch, the endpoint returns: `{ "count": number, "results": [<single-record-response>, ...] }`.

## Request schema

Single record fields (all required):
- BORO_NM: string (e.g., BRONX | BROOKLYN | MANHATTAN | QUEENS | STATEN ISLAND)
- LOC_OF_OCCUR_DESC: string
- VIC_AGE_GROUP: string
- VIC_RACE: string
- VIC_SEX: string
- Latitude: number
- Longitude: number
- BAR_DISTANCE: number
- NIGHTCLUB_DISTANCE: number
- ATM_DISTANCE: number
- ATMS_COUNT: number
- BARS_COUNT: number
- BUS_STOPS_COUNT: number
- METROS_COUNT: number
- NIGHTCLUBS_COUNT: number
- SCHOOLS_COUNT: number
- METRO_DISTANCE: number
- MIN_POI_DISTANCE: number
- AVG_POI_DISTANCE: number
- MAX_POI_DISTANCE: number
- TOTAL_POI_COUNT: number
- POI_DIVERSITY: integer
- POI_DENSITY_SCORE: number
- HOUR: integer [0..23]
- DAY: integer [1..31]
- WEEKDAY: string (MONDAY..SUNDAY)
- IS_WEEKEND: integer 0/1
- MONTH: integer [1..12]
- YEAR: integer
- SEASON: string (e.g., SPRING | SUMMER | FALL | WINTER)
- TIME_BUCKET: string (e.g., MORNING | AFTERNOON | EVENING | NIGHT)
- IS_HOLIDAY: integer 0/1
- IS_PAYDAY: integer 0/1

Batch payload: `{ "records": [<InputRecord>, ...] }`.

## Run with Docker

- Prerequisites: Docker Desktop
- From `Application/Tourists` run:

```powershell
docker compose up --build
Start-Process http://localhost:8000/docs
```

The container binds the project root to `/workspace` so the service can load:
- /workspace/JupyterOutputs/Classification (Preprocessing)/preprocessing_pipeline_general.joblib
- /workspace/JupyterOutputs/Classification (Preprocessing)/feature_names.json
- /workspace/JupyterOutputs/Classification (Final)/LogisticRegression_production_model.joblib
- /workspace/JupyterOutputs/Classification (Tuning)/LogisticRegression_optimal_threshold.json (optional)
- /workspace/JupyterOutputs/PatternAnalysis/rule_catalog.json (for trends)

Environment overrides: `MODEL_PATH`, `PREPROCESSOR_PATH`, `FEATURE_NAMES_PATH`, `OPTIMAL_THRESHOLD`, `PATTERN_OUTPUT_DIR`.

## Example requests

PowerShell (single record):

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

Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/predict -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

curl (single record):

```bash
curl -X POST "http://localhost:8000/api/v1/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"BORO_NM\":\"BROOKLYN\",\"LOC_OF_OCCUR_DESC\":\"OUTSIDE\",\"VIC_AGE_GROUP\":\"25-44\",\"VIC_RACE\":\"WHITE\",\"VIC_SEX\":\"M\",\"Latitude\":40.6782,\"Longitude\":-73.9442,\"BAR_DISTANCE\":120,\"NIGHTCLUB_DISTANCE\":500,\"ATM_DISTANCE\":80,\"ATMS_COUNT\":2,\"BARS_COUNT\":3,\"BUS_STOPS_COUNT\":1,\"METROS_COUNT\":0,\"NIGHTCLUBS_COUNT\":0,\"SCHOOLS_COUNT\":1,\"METRO_DISTANCE\":300,\"MIN_POI_DISTANCE\":30,\"AVG_POI_DISTANCE\":150,\"MAX_POI_DISTANCE\":600,\"TOTAL_POI_COUNT\":7,\"POI_DIVERSITY\":4,\"POI_DENSITY_SCORE\":0.45,\"HOUR\":13,\"DAY\":15,\"WEEKDAY\":\"MONDAY\",\"IS_WEEKEND\":0,\"MONTH\":5,\"YEAR\":2023,\"SEASON\":\"SPRING\",\"TIME_BUCKET\":\"AFTERNOON\",\"IS_HOLIDAY\":0,\"IS_PAYDAY\":0}"
```

PowerShell (batch):

```powershell
$batch = @{ records = @(
  @{ BORO_NM = "BROOKLYN"; LOC_OF_OCCUR_DESC = "OUTSIDE"; VIC_AGE_GROUP = "25-44"; VIC_RACE = "WHITE"; VIC_SEX = "M"; Latitude = 40.6782; Longitude = -73.9442; BAR_DISTANCE = 120; NIGHTCLUB_DISTANCE = 500; ATM_DISTANCE = 80; ATMS_COUNT = 2; BARS_COUNT = 3; BUS_STOPS_COUNT = 1; METROS_COUNT = 0; NIGHTCLUBS_COUNT = 0; SCHOOLS_COUNT = 1; METRO_DISTANCE = 300; MIN_POI_DISTANCE = 30; AVG_POI_DISTANCE = 150; MAX_POI_DISTANCE = 600; TOTAL_POI_COUNT = 7; POI_DIVERSITY = 4; POI_DENSITY_SCORE = 0.45; HOUR = 13; DAY = 15; WEEKDAY = "MONDAY"; IS_WEEKEND = 0; MONTH = 5; YEAR = 2023; SEASON = "SPRING"; TIME_BUCKET = "AFTERNOON"; IS_HOLIDAY = 0; IS_PAYDAY = 0 }
) } | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/predict -ContentType 'application/json' -Body $batch | ConvertTo-Json -Depth 6