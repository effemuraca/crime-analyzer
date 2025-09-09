import os
import sys
from typing import List, Literal, Union, Optional, Tuple, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import types
import importlib.util
import importlib
import json
from scipy import sparse as sp


# Ensure custom transformers are importable when unpickling the model
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UTILS_PATH = os.path.join(REPO_ROOT, "Notebooks", "Classification", "Utilities")
if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def _ensure_custom_transformers_alias():
    """Ensure both 'custom_transformers' and 'Utilities.custom_transformers' are importable."""
    ct = None
    try:
        import custom_transformers as _ct  # type: ignore
        ct = _ct
    except Exception:
        # Try to load directly from file if not importable by name
        ct_path = os.path.join(UTILS_PATH, "custom_transformers.py")
        if os.path.isfile(ct_path):
            spec = importlib.util.spec_from_file_location("custom_transformers", ct_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["custom_transformers"] = module
                spec.loader.exec_module(module)
                ct = module

    # Create 'Utilities' package alias and bind custom_transformers under it
    if "Utilities" not in sys.modules:
        pkg = types.ModuleType("Utilities")
        pkg.__path__ = [UTILS_PATH]  # mark as a namespace/package
        sys.modules["Utilities"] = pkg
    if ct is not None:
        sys.modules["Utilities.custom_transformers"] = ct

_ensure_custom_transformers_alias()

# Many sklearn pipelines rely on imbalanced-learn components; ensure available for unpickling
try:
    import imblearn  # type: ignore  # noqa: F401
except Exception:
    pass


def _ensure_sklearn_pickle_shims():
    """Provide missing private attributes expected by older/newer sklearn pickles."""
    try:
        mod = importlib.import_module("sklearn.compose._column_transformer")
        if not hasattr(mod, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            setattr(mod, "_RemainderColsList", _RemainderColsList)
    except Exception:
        # Best effort; if it fails we'll see the original error and can adjust version
        pass

_ensure_sklearn_pickle_shims()


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(
        REPO_ROOT,
        "JupyterOutputs",
        "Classification (Final)",
        "LogisticRegression_production_model.joblib",
    ),
)

PREPROCESSOR_PATH = os.getenv(
    "PREPROCESSOR_PATH",
    os.path.join(
        REPO_ROOT,
        "JupyterOutputs",
        "Classification (Preprocessing)",
        "preprocessing_pipeline_general.joblib",
    ),
)

FEATURE_NAMES_PATH = os.getenv(
    "FEATURE_NAMES_PATH",
    os.path.join(
        REPO_ROOT,
        "JupyterOutputs",
        "Classification (Preprocessing)",
        "feature_names.json",
    ),
)

THRESHOLD_JSON_PATH = os.path.join(
    REPO_ROOT,
    "JupyterOutputs",
    "Classification (Tuning)",
    "LogisticRegression_optimal_threshold.json",
)

# Optional: prefer a fully-fitted end-to-end pipeline when available
FINAL_PIPELINE_ENV = os.getenv("FINAL_PIPELINE_PATH")
FINAL_PIPELINE_DEFAULT = os.path.join(
    REPO_ROOT,
    "JupyterOutputs",
    "Classification (Tuning)",
    "LogisticRegression_final_pipeline.joblib",
)


def _resolve_pipeline_path() -> Optional[str]:
    # Priority: explicit env -> default -> derive from MODEL/PREPROCESSOR -> common /workspace path
    candidates: List[str] = []
    if FINAL_PIPELINE_ENV:
        candidates.append(FINAL_PIPELINE_ENV)
    candidates.append(FINAL_PIPELINE_DEFAULT)
    # Derive from MODEL_PATH root
    model_root = os.path.dirname(os.path.dirname(os.path.dirname(MODEL_PATH))) if MODEL_PATH else None
    if model_root:
        candidates.append(os.path.join(model_root, "Classification (Tuning)", "LogisticRegression_final_pipeline.joblib"))
    # Derive from PREPROCESSOR_PATH root
    pre_root = os.path.dirname(os.path.dirname(os.path.dirname(PREPROCESSOR_PATH))) if PREPROCESSOR_PATH else None
    if pre_root:
        candidates.append(os.path.join(pre_root, "Classification (Tuning)", "LogisticRegression_final_pipeline.joblib"))
    # Common mount root
    candidates.append("/workspace/JupyterOutputs/Classification (Tuning)/LogisticRegression_final_pipeline.joblib")

    for p in candidates:
        try:
            if p and os.path.isfile(p):
                return p
        except Exception:
            continue
    return None


def _load_optimal_threshold(default: float = 0.64) -> float:
    env_val = os.getenv("OPTIMAL_THRESHOLD")
    if env_val is not None:
        try:
            return float(env_val)
        except Exception:
            pass
    try:
        if os.path.isfile(THRESHOLD_JSON_PATH):
            with open(THRESHOLD_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Try a few common keys
            for key in ("optimal_threshold", "threshold", "best_threshold", "value"):
                if key in data:
                    return float(data[key])
    except Exception:
        pass
    return float(default)


OPTIMAL_THRESHOLD = _load_optimal_threshold()


class InputRecord(BaseModel):
    # Operational input schema (fields available at prediction time)
    BORO_NM: str
    LOC_OF_OCCUR_DESC: str
    VIC_AGE_GROUP: str
    VIC_RACE: str
    VIC_SEX: str
    Latitude: float
    Longitude: float
    BAR_DISTANCE: float
    NIGHTCLUB_DISTANCE: float
    ATM_DISTANCE: float
    ATMS_COUNT: float
    BARS_COUNT: float
    BUS_STOPS_COUNT: float
    METROS_COUNT: float
    NIGHTCLUBS_COUNT: float
    SCHOOLS_COUNT: float
    METRO_DISTANCE: float
    MIN_POI_DISTANCE: float
    AVG_POI_DISTANCE: float
    MAX_POI_DISTANCE: float
    TOTAL_POI_COUNT: float
    POI_DIVERSITY: int
    POI_DENSITY_SCORE: float
    HOUR: int = Field(ge=0, le=23)
    DAY: int = Field(ge=1, le=31)
    WEEKDAY: Literal[
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
        "SUNDAY",
    ]
    IS_WEEKEND: int = Field(ge=0, le=1)
    MONTH: int = Field(ge=1, le=12)
    YEAR: int
    SEASON: str
    TIME_BUCKET: str
    IS_HOLIDAY: int = Field(ge=0, le=1)
    IS_PAYDAY: int = Field(ge=0, le=1)


class BatchRequest(BaseModel):
    records: List[InputRecord]


app = FastAPI(title="Crime Analyzer - LogisticRegression API", version="1.0.0")


def _load_model(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


"""Prefer the production model joblib (Final) as the primary deployment artifact."""
# Load pipeline/model at startup with status diagnostics
PIPELINE = None
PIPELINE_STATUS = {
    "resolved_path": MODEL_PATH,
    "exists": os.path.isfile(MODEL_PATH),
    "loaded": False,
    "has_predict_proba": False,
    "load_error": None,
}

MODEL = None
try:
    obj = _load_model(MODEL_PATH)
    # Detect sklearn/imblearn Pipeline-like objects first
    if hasattr(obj, "named_steps") or hasattr(obj, "steps"):
        PIPELINE = obj
        PIPELINE_STATUS["loaded"] = True
        # Heuristic: pipeline exposes predict_proba or final estimator does
        has_pp = hasattr(PIPELINE, "predict_proba")
        if not has_pp and hasattr(PIPELINE, "named_steps"):
            try:
                last_name, last_step = list(getattr(PIPELINE, "steps", []) or getattr(PIPELINE, "named_steps", {}).items())[-1]
                has_pp = hasattr(last_step, "predict_proba")
            except Exception:
                pass
        PIPELINE_STATUS["has_predict_proba"] = bool(has_pp)
    else:
        # Not a pipeline; keep as bare model
        MODEL = obj
except Exception as e:
    PIPELINE_STATUS["load_error"] = str(e)
    PIPELINE = None
    MODEL = None

# If no pipeline from MODEL_PATH and FINAL_PIPELINE_PATH is a true estimator pipeline, try it
if PIPELINE is None:
    FINAL_PIPELINE_PATH = _resolve_pipeline_path()
    if FINAL_PIPELINE_PATH and os.path.isfile(FINAL_PIPELINE_PATH):
        PIPELINE_STATUS["resolved_path"] = FINAL_PIPELINE_PATH
        PIPELINE_STATUS["exists"] = True
        try:
            maybe_pipe = joblib.load(FINAL_PIPELINE_PATH)
            if hasattr(maybe_pipe, "named_steps") or hasattr(maybe_pipe, "steps"):
                PIPELINE = maybe_pipe
                PIPELINE_STATUS["loaded"] = True
                has_pp = hasattr(PIPELINE, "predict_proba")
                if not has_pp and hasattr(PIPELINE, "named_steps"):
                    try:
                        last_name, last_step = list(getattr(PIPELINE, "steps", []) or getattr(PIPELINE, "named_steps", {}).items())[-1]
                        has_pp = hasattr(last_step, "predict_proba")
                    except Exception:
                        pass
                PIPELINE_STATUS["has_predict_proba"] = bool(has_pp)
            else:
                # Likely a dict of components; not suitable for direct inference
                PIPELINE_STATUS["load_error"] = "Not a Pipeline object (components bundle)"
        except Exception as e:
            PIPELINE_STATUS["load_error"] = str(e)

if PIPELINE is None and MODEL is None:
    # As last resort, try loading bare model again from MODEL_PATH (already attempted)
    try:
        MODEL = _load_model(MODEL_PATH)
    except Exception:
        MODEL = None


def _load_preprocessor(preprocessor_path: str):
    if not os.path.isfile(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
    try:
        obj = joblib.load(preprocessor_path)
        # If a Pipeline was saved, extract its 'preprocessor' step to avoid double feature selection
        if hasattr(obj, "named_steps") and "preprocessor" in obj.named_steps:
            pre = obj.named_steps["preprocessor"]
        else:
            pre = obj
        # Expect transform method
        if not hasattr(pre, "transform"):
            raise RuntimeError("Loaded preprocessor has no transform method")
        return pre
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessor: {e}")


def _load_feature_names(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Feature names file not found at {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = json.load(f)
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise ValueError("feature_names.json must be a list of strings")
        return names
    except Exception as e:
        raise RuntimeError(f"Failed to load feature names: {e}")


PREPROCESSOR = _load_preprocessor(PREPROCESSOR_PATH)
FEATURE_NAMES: List[str] = _load_feature_names(FEATURE_NAMES_PATH)


def _expected_input_columns() -> List[str]:
    # Priority: pipeline's recorded input schema
    try:
        if PIPELINE is not None:
            if hasattr(PIPELINE, "feature_names_in_"):
                return list(getattr(PIPELINE, "feature_names_in_"))
            if hasattr(PIPELINE, "named_steps") and "preprocessor" in PIPELINE.named_steps:
                pre = PIPELINE.named_steps["preprocessor"]
                if hasattr(pre, "feature_names_in_"):
                    return list(getattr(pre, "feature_names_in_"))
    except Exception:
        pass
    # Fallback: standalone preprocessor schema
    try:
        if hasattr(PREPROCESSOR, "feature_names_in_"):
            return list(getattr(PREPROCESSOR, "feature_names_in_"))
    except Exception:
        pass
    # Final fallback: static feature_names.json
    return FEATURE_NAMES


EXPECTED_INPUT_COLUMNS: List[str] = _expected_input_columns()
PREPROCESSOR_STATUS = {
    "path": PREPROCESSOR_PATH,
    "appears_fitted": False,
}
try:
    if hasattr(PREPROCESSOR, "transformers_"):
        PREPROCESSOR_STATUS["appears_fitted"] = True
    elif hasattr(PREPROCESSOR, "fitted_") and bool(getattr(PREPROCESSOR, "fitted_", False)):
        PREPROCESSOR_STATUS["appears_fitted"] = True
except Exception:
    pass


@app.get("/health")
def health():
    return {
        "status": "ok",
    "mode": "pipeline" if PIPELINE is not None else "preprocessor+model",
    "pipeline": PIPELINE_STATUS,
    "model_path": MODEL_PATH if MODEL is not None else None,
    "preprocessor": PREPROCESSOR_STATUS,
    "features_expected": len(EXPECTED_INPUT_COLUMNS),
    "expected_input_columns_sample": EXPECTED_INPUT_COLUMNS[:10],
        "transformed_features_expected": (
            int(getattr(PIPELINE.named_steps.get("preprocessor"), "get_feature_names_out")().__len__())
            if (PIPELINE is not None and hasattr(PIPELINE, "named_steps") and "preprocessor" in PIPELINE.named_steps and hasattr(PIPELINE.named_steps["preprocessor"], "get_feature_names_out"))
            else None
        ),
        "threshold": OPTIMAL_THRESHOLD,
    }
_CATEGORICAL_COLS = {
    "BORO_NM",
    "LOC_OF_OCCUR_DESC",
    "VIC_AGE_GROUP",
    "VIC_RACE",
    "VIC_SEX",
    "WEEKDAY",
    "SEASON",
    "TIME_BUCKET",
}


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize categorical columns as strings (uppercase for stable matching)
    for c in _CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()
    # Cast numerics where appropriate
    for c in df.columns:
        if c not in _CATEGORICAL_COLS:
            # Try numeric cast; if fails, leave as is and let preprocessing handle errors
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df


def _validate_and_align(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce input types to match training expectations
    df = _coerce_types(df.copy())
    provided = set(df.columns.tolist())
    expected = set(EXPECTED_INPUT_COLUMNS)
    missing = sorted(list(expected - provided))
    extra = sorted(list(provided - expected))
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required fields",
                "missing": missing,
            },
        )
    # Drop extra fields silently to be lenient
    df = df[[c for c in EXPECTED_INPUT_COLUMNS if c in df.columns]]
    return df


def _predict_df(df: pd.DataFrame):
    # Validate and align input columns
    df = _validate_and_align(df)

    # Prefer full fitted pipeline when available
    if PIPELINE is not None:
        try:
            proba = PIPELINE.predict_proba(df)
        except Exception as e:
            # Fallback: attempt manual forward pass with padding/truncation to satisfy selector expectations
            try:
                if not hasattr(PIPELINE, "named_steps"):
                    raise e
                pre = PIPELINE.named_steps.get("preprocessor")
                sel = PIPELINE.named_steps.get("feature_selector")
                clf = PIPELINE.named_steps.get("classifier")
                if pre is None or sel is None or clf is None:
                    raise e

                X_pre = pre.transform(df)
                expected_in = None
                if hasattr(sel, "estimator_") and hasattr(sel.estimator_, "n_features_in_"):
                    expected_in = int(sel.estimator_.n_features_in_)
                elif hasattr(sel, "n_features_in_"):
                    expected_in = int(getattr(sel, "n_features_in_"))

                if expected_in is not None:
                    current = X_pre.shape[1]
                    if current < expected_in:
                        diff = expected_in - current
                        if sp.issparse(X_pre):
                            zeros = sp.csr_matrix((X_pre.shape[0], diff))
                            X_pre = sp.hstack([X_pre, zeros], format="csr")
                        else:
                            import numpy as np
                            zeros = np.zeros((X_pre.shape[0], diff), dtype=X_pre.dtype if hasattr(X_pre, "dtype") else float)
                            X_pre = np.hstack([X_pre, zeros])
                    elif current > expected_in:
                        # Truncate extra columns if present
                        if sp.issparse(X_pre):
                            X_pre = X_pre[:, :expected_in]
                        else:
                            X_pre = X_pre[:, :expected_in]

                X_sel = sel.transform(X_pre)
                proba = clf.predict_proba(X_sel)
            except Exception:
                raise HTTPException(status_code=500, detail=f"Pipeline prediction failed: {e}")
    else:
        # Using separate preprocessor + model
        # Check that the preprocessor appears to be fitted (common attributes for ColumnTransformer)
        try:
            needs_fit = False
            if hasattr(PREPROCESSOR, "transformers_"):
                # Fitted ColumnTransformer usually has transformers_
                pass
            elif hasattr(PREPROCESSOR, "fitted_"):
                # Some custom transformers expose fitted_ flag
                if not getattr(PREPROCESSOR, "fitted_", False):
                    needs_fit = True
            # Attempt a dry-run with zero rows to detect fit state without mutating
            if not needs_fit:
                _ = PREPROCESSOR.transform(df.head(0))
        except Exception:
            needs_fit = True
        if needs_fit:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Configured preprocessor appears unfitted. Prefer using the final pipeline "
                    "artifact (LogisticRegression_final_pipeline.joblib) or ensure a fitted preprocessor."
                ),
            )

        # Apply preprocessing pipeline before prediction
        try:
            X = PREPROCESSOR.transform(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

        # Expecting a sklearn-compatible estimator with predict_proba
        if MODEL is None or not hasattr(MODEL, "predict_proba"):
            raise HTTPException(status_code=500, detail="Loaded model has no predict_proba method")
        proba = MODEL.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] > 1:
        pos_proba = proba[:, 1]
    else:
        # Some estimators return single-column proba for binary
        pos_proba = proba.ravel()

    preds = (pos_proba >= OPTIMAL_THRESHOLD).astype(int)
    return pos_proba.tolist(), preds.tolist()


# --- Feature contribution extraction (best-effort for linear models) ---
def _get_top_features_single(df_row: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    # This function tries to compute feature contributions using a linear classifier downstream of
    # the preprocessor and feature selector. It gracefully degrades to empty list if unavailable.
    if PIPELINE is None:
        return []
    try:
        if not hasattr(PIPELINE, "named_steps"):
            return []
        pre = PIPELINE.named_steps.get("preprocessor")
        sel = PIPELINE.named_steps.get("feature_selector")
        clf = PIPELINE.named_steps.get("classifier")
        if pre is None or clf is None:
            return []

        X_pre = pre.transform(df_row)
        if sel is not None:
            # Align dimensions if needed, as in the manual fallback
            expected_in = None
            if hasattr(sel, "estimator_") and hasattr(sel.estimator_, "n_features_in_"):
                expected_in = int(sel.estimator_.n_features_in_)
            elif hasattr(sel, "n_features_in_"):
                expected_in = int(getattr(sel, "n_features_in_"))
            if expected_in is not None and X_pre.shape[1] != expected_in:
                current = X_pre.shape[1]
                if current < expected_in:
                    diff = expected_in - current
                    if sp.issparse(X_pre):
                        zeros = sp.csr_matrix((X_pre.shape[0], diff))
                        X_pre = sp.hstack([X_pre, zeros], format="csr")
                    else:
                        import numpy as np
                        zeros = np.zeros((X_pre.shape[0], diff), dtype=X_pre.dtype if hasattr(X_pre, "dtype") else float)
                        X_pre = np.hstack([X_pre, zeros])
                else:
                    X_pre = X_pre[:, :expected_in]
            X = sel.transform(X_pre)
        else:
            X = X_pre

        # Try to get feature names after preprocessing/selection
        feat_names: List[str] = []
        try:
            if hasattr(pre, "get_feature_names_out"):
                feat_names = list(pre.get_feature_names_out())
            elif hasattr(pre, "get_feature_names"):
                feat_names = list(pre.get_feature_names())  # type: ignore
        except Exception:
            feat_names = []

        # If selector exists and supports get_support, reduce feature names accordingly
        if sel is not None and hasattr(sel, "get_support"):
            try:
                support_mask = sel.get_support()
                if feat_names and len(feat_names) == len(support_mask):
                    feat_names = [n for n, keep in zip(feat_names, support_mask) if keep]
            except Exception:
                pass

        # Classifier coefficients (assume 2-class LR or linear model)
        coef = None
        if hasattr(clf, "coef_"):
            coef = getattr(clf, "coef_")
        elif hasattr(clf, "feature_importances_"):
            coef = getattr(clf, "feature_importances_")
        if coef is None:
            return []

        import numpy as np
        x = X.toarray()[0] if sp.issparse(X) else np.array(X)[0]
        w = coef[0] if isinstance(coef, (list, tuple)) or (hasattr(coef, "shape") and len(getattr(coef, "shape")) > 1) else coef
        w = np.array(w)
        # Align lengths defensively
        m = min(len(x), len(w))
        x = x[:m]
        w = w[:m]
        contrib = x * w

        # Build pairs with names where available
        if feat_names and len(feat_names) >= m:
            names = feat_names[:m]
        else:
            names = [f"f{i}" for i in range(m)]

        idx = np.argsort(-np.abs(contrib))[:max(1, int(top_n))]
        results = [
            {"feature": str(names[i]), "contribution": float(contrib[i])} for i in idx
        ]
        return results
    except Exception:
        return []


# legacy predict endpoints removed; use /api/v1/predict


# --- Unified v1 endpoint with trends enrichment ---
try:
    from . import pattern_insights as pin
except Exception:
    import importlib
    pin = importlib.import_module("pattern_insights")


def _context_rules_for_input(df_row: pd.Series, top_k: int = 5) -> Dict[str, Any]:
    boro = str(df_row.get("BORO_NM", "")).strip().upper()
    tbucket = str(df_row.get("TIME_BUCKET", "")).strip().upper()
    out = {"neighborhood": [], "time_bucket": []}
    try:
        if boro:
            res_b = pin.get_borough_top_k(boro, top_k)
            if res_b.get("available"):
                out["neighborhood"] = res_b.get("rules", [])
        if tbucket:
            res_t = pin.get_time_bucket_top_k(tbucket, top_k)
            if res_t.get("available"):
                out["time_bucket"] = res_t.get("rules", [])
    except Exception:
        pass
    return out


@app.post("/api/v1/predict")
def v1_predict(payload: Union[InputRecord, BatchRequest]):
    # Accept both single and batch; we’ll return a single enriched object when single, or array when batch
    if isinstance(payload, InputRecord):
        records = [payload.model_dump()]
    else:
        if not payload.records:
            raise HTTPException(status_code=400, detail="records must be a non-empty list")
        records = [rec.model_dump() for rec in payload.records]

    df = pd.DataFrame(records)
    try:
        probas, labels = _predict_df(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    outputs = []
    for i, row in df.iterrows():
        prob = float(probas[i])
        lbl = int(labels[i])
        label_name = "HIGH_RISK" if lbl == 1 else "LOW_RISK"
        # Top features for this record
        top_feats = _get_top_features_single(df.iloc[[i]], top_n=5)
        # Trends for borough/time bucket
        trends = _context_rules_for_input(row, top_k=5)
        outputs.append({
            "label": label_name,
            "confidence": prob,
            "threshold": OPTIMAL_THRESHOLD,
            "explanations": {"top_features": top_feats},
            "trends": trends,
        })

    return outputs[0] if len(outputs) == 1 else {"count": len(outputs), "results": outputs}


"""Dedicated insights endpoints removed; insights are embedded in /api/v1/predict output."""



@app.get("/")
def root():
    return {
        "message": "Tourist Safety API. Send POST /api/v1/predict with an InputRecord or BatchRequest to get predictions.",
        "threshold": OPTIMAL_THRESHOLD,
        "response_format": {
            "label": "HIGH_RISK | LOW_RISK",
            "confidence": "float in [0,1] for positive class",
            "threshold": "decision threshold used",
            "explanations": {"top_features": "array of {feature, contribution}"},
            "trends": {
                "neighborhood": "top rules for BORO_NM",
                "time_bucket": "top rules for TIME_BUCKET",
            },
        },
        "example_single": {
            "BORO_NM": "BROOKLYN",
            "LOC_OF_OCCUR_DESC": "OUTSIDE",
            "VIC_AGE_GROUP": "25-44",
            "VIC_RACE": "WHITE",
            "VIC_SEX": "M",
            "Latitude": 40.6782,
            "Longitude": -73.9442,
            "BAR_DISTANCE": 120.0,
            "NIGHTCLUB_DISTANCE": 500.0,
            "ATM_DISTANCE": 80.0,
            "ATMS_COUNT": 2.0,
            "BARS_COUNT": 3.0,
            "BUS_STOPS_COUNT": 1.0,
            "METROS_COUNT": 0.0,
            "NIGHTCLUBS_COUNT": 0.0,
            "SCHOOLS_COUNT": 1.0,
            "METRO_DISTANCE": 300.0,
            "MIN_POI_DISTANCE": 30.0,
            "AVG_POI_DISTANCE": 150.0,
            "MAX_POI_DISTANCE": 600.0,
            "TOTAL_POI_COUNT": 7.0,
            "POI_DIVERSITY": 4,
            "POI_DENSITY_SCORE": 0.45,
            "HOUR": 13,
            "DAY": 15,
            "WEEKDAY": "MONDAY",
            "IS_WEEKEND": 0,
            "MONTH": 5,
            "YEAR": 2023,
            "SEASON": "SPRING",
            "TIME_BUCKET": "MORNING",
            "IS_HOLIDAY": 0,
            "IS_PAYDAY": 0,
        },
    }
