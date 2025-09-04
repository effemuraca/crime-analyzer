import os
import json
from typing import Any, Dict, List, Optional, Tuple


# Resolve repo root relative to this file
HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _resolve_output_dir() -> str:
    # 1) Explicit env override
    env_dir = os.getenv("PATTERN_OUTPUT_DIR")
    candidates = []
    if env_dir:
        candidates.append(env_dir)
    # 2) Preferred workspace mount used by docker-compose
    candidates.append(os.path.join("/workspace", "JupyterOutputs", "PatternAnalysis"))
    # 3) Repo-root relative (works when running without container and cwd is project root)
    candidates.append(os.path.join(REPO_ROOT, "JupyterOutputs", "PatternAnalysis"))
    # 4) Local dev fallback: try two levels up (if Application/ nested deeper)
    candidates.append(os.path.abspath(os.path.join(HERE, "..", "..", "JupyterOutputs", "PatternAnalysis")))

    for d in candidates:
        try:
            if d and os.path.isdir(d):
                return d
        except Exception:
            continue
    # If none exist yet, return the first candidate to keep paths consistent
    return candidates[0] if candidates else os.path.join(REPO_ROOT, "JupyterOutputs", "PatternAnalysis")


def _paths() -> tuple[str, str]:
    base = _resolve_output_dir()
    return (
        os.path.join(base, "rule_catalog.json"),
        os.path.join(base, "global_insights.txt"),
    )


_CATALOG_CACHE: Optional[Dict[str, Any]] = None
_CATALOG_MTIME: Optional[float] = None


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _safe_read_text(path: str) -> Optional[str]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        return None
    return None


def load_catalog(force: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    global _CATALOG_CACHE, _CATALOG_MTIME
    rule_catalog_path, _ = _paths()
    try:
        mtime = os.path.getmtime(rule_catalog_path) if os.path.isfile(rule_catalog_path) else None
    except Exception:
        mtime = None

    if not force and _CATALOG_CACHE is not None and _CATALOG_MTIME == mtime:
        return _CATALOG_CACHE, _CATALOG_MTIME

    data = _safe_read_json(rule_catalog_path)
    _CATALOG_CACHE = data
    _CATALOG_MTIME = mtime
    return data, mtime


def _simplify_rule(r: Dict[str, Any]) -> Dict[str, Any]:
    ant = r.get("antecedent", [])
    cons = r.get("consequent", [])
    text = None
    try:
        ant_str = " AND ".join(str(x) for x in ant)
        cons_str = " AND ".join(str(x) for x in cons)
        text = f"IF ({ant_str}) THEN ({cons_str})"
    except Exception:
        text = None
    return {
        "antecedent": ant,
        "consequent": cons,
        "support": r.get("support"),
        "confidence": r.get("confidence"),
        "lift": r.get("lift"),
    "leverage": r.get("leverage"),
    "conviction": r.get("conviction"),
    "zhangs_metric": r.get("zhangs_metric"),
        "score": r.get("score"),
        "text": text,
        "context": r.get("context"),
    }


def _top_k(rules: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    k = max(1, int(k))
    if not rules:
        return []
    # Sort by score desc then support desc when available
    try:
        sorted_rules = sorted(
            rules,
            key=lambda r: (
                float(r.get("score", 0.0) or 0.0),
                float(r.get("support", 0.0) or 0.0),
            ),
            reverse=True,
        )
    except Exception:
        sorted_rules = rules
    return [_simplify_rule(r) for r in sorted_rules[:k]]


def get_global_top_k(k: int = 10) -> Dict[str, Any]:
    catalog, _ = load_catalog()
    if not catalog:
        return {"available": False, "error": "rule_catalog.json not found"}
    glb = catalog.get("global", {})
    rules = glb.get("top_k_rules", [])
    return {
        "available": True,
        "scope": "global",
        "top_k": int(k),
        "rules": _top_k(rules, k),
        "metadata": catalog.get("metadata", {}),
    }


def get_borough_top_k(borough: str, k: int = 10) -> Dict[str, Any]:
    catalog, _ = load_catalog()
    if not catalog:
        return {"available": False, "error": "rule_catalog.json not found"}
    target = (borough or "").strip().upper()
    items = catalog.get("by_borough", [])
    entry = next((e for e in items if str(e.get("value", "")).strip().upper() == target), None)
    if not entry:
        return {"available": False, "error": f"borough '{borough}' not found in catalog"}
    return {
        "available": True,
        "scope": "borough",
        "value": entry.get("value"),
        "records_count": entry.get("records_count"),
        "top_k": int(k),
        "rules": _top_k(entry.get("top_k_rules", []), k),
        "metadata": catalog.get("metadata", {}),
    }


def get_time_bucket_top_k(time_bucket: str, k: int = 10) -> Dict[str, Any]:
    catalog, _ = load_catalog()
    if not catalog:
        return {"available": False, "error": "rule_catalog.json not found"}
    target = (time_bucket or "").strip().upper()
    items = catalog.get("by_time_bucket", [])
    entry = next((e for e in items if str(e.get("value", "")).strip().upper() == target), None)
    if not entry:
        return {"available": False, "error": f"time_bucket '{time_bucket}' not found in catalog"}
    return {
        "available": True,
        "scope": "time_bucket",
        "value": entry.get("value"),
        "records_count": entry.get("records_count"),
        "top_k": int(k),
        "rules": _top_k(entry.get("top_k_rules", []), k),
        "metadata": catalog.get("metadata", {}),
    }


def get_text_insights(scope: str, value: Optional[str] = None) -> Dict[str, Any]:
    scope_l = (scope or "").strip().lower()
    path: Optional[str] = None
    if scope_l == "global":
        _, global_text_path = _paths()
        path = global_text_path
    elif scope_l == "borough" and value:
        base, _ = _paths()
        base_dir = os.path.dirname(base)
        path = os.path.join(base_dir, f"Borough_{value.strip().upper()}_insights.txt")
    elif scope_l == "time_bucket" and value:
        base, _ = _paths()
        base_dir = os.path.dirname(base)
        path = os.path.join(base_dir, f"TimeBucket_{value.strip().upper()}_insights.txt")
    else:
        return {"available": False, "error": "invalid scope or missing value"}

    content = _safe_read_text(path)
    if content is None:
        return {"available": False, "error": f"insights text not found at {os.path.basename(path)}"}

    # Return lines excluding empty ones for easier consumption
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    return {
        "available": True,
        "scope": scope_l,
        "value": (value or None),
        "file": os.path.basename(path),
        "lines": lines,
    }


def status() -> Dict[str, Any]:
    catalog, mtime = load_catalog()
    available = catalog is not None
    meta = catalog.get("metadata", {}) if catalog else {}
    contexts = {
        "global": bool(catalog and catalog.get("global")),
        "by_borough": bool(catalog and catalog.get("by_borough")),
        "by_time_bucket": bool(catalog and catalog.get("by_time_bucket")),
    }
    rule_catalog_path, _ = _paths()
    return {
        "available": available,
        "rule_catalog_path": rule_catalog_path,
        "rule_catalog_mtime": mtime,
        "contexts": contexts,
        "metadata": meta,
    }
