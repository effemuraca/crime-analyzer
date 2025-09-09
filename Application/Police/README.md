Police Intelligence Dashboard (static)

What it is
- A static dashboard for police operations under Application/Police
- Embeds the hotspot map from JupyterOutputs/Clustering (SpatialHotspots)/cluster_temporal_patterns_map.html
- Loads insights from JupyterOutputs/Clustering (MultidimensionalClusteringAnalysis)/*.json and renders:
  - KPIs (tot crimes, patterns, high-priority share)
  - Priority patterns (most concentrated, highest volume)
  - Recommendations (derived)
  - Clustering quality + methods comparison

How to run (recommended)
- Serve the repo root with a static server to avoid browser CORS limits on file://

Python (from repo root)
  - PowerShell:
    - python -m http.server 8000
  - Then open: http://localhost:8000/Application/Police/index.html

Notes
- Relative paths assume this directory structure is kept intact.
- If the map iframe shows a fallback message, check the file exists:
  JupyterOutputs/Clustering (SpatialHotspots)/cluster_temporal_patterns_map.html
- If JSONs are missing, the panel will degrade gracefully (—).
