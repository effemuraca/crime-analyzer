#!/usr/bin/env python3
"""
Pattern Analysis Input Profiler

Purpose
- Inspect the finalized crime dataset used by the PatternAnalysis notebook
- Report feature presence, dtypes, value distributions, and nulls
- Propose binning plans for distances, counts, diversity, and density
- Suggest top-N caps for key categoricals and TRUE/FALSE mappings for boolean-like columns
- Export a JSON and CSV summary that we can use to finalize the notebook before running
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# --- Expected columns and groups (aligned with the notebook) ---
EXPECTED_CATEGORICAL = [
    'OFNS_DESC', 'PREM_TYP_DESC', 'LAW_CAT_CD', 'LOC_OF_OCCUR_DESC', 'BORO_NM',
    'SEASON', 'TIME_BUCKET', 'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX',
    'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX'
]
EXPECTED_BOOLEAN_LIKE = [
    'IS_WEEKEND', 'IS_HOLIDAY', 'IS_PAYDAY', 'SAME_AGE_GROUP', 'SAME_SEX', 'TO_CHECK_CITIZENS'
]
EXPECTED_NUMERIC_DISTANCES = [
    'MIN_POI_DISTANCE', 'AVG_POI_DISTANCE', 'MAX_POI_DISTANCE',
    'BAR_DISTANCE', 'METRO_DISTANCE', 'NIGHTCLUB_DISTANCE', 'ATM_DISTANCE'
]
EXPECTED_NUMERIC_COUNTS = [
    'TOTAL_POI_COUNT', 'BARS_COUNT', 'ATMS_COUNT', 'BUS_STOPS_COUNT',
    'METROS_COUNT', 'NIGHTCLUBS_COUNT', 'SCHOOLS_COUNT'
]
EXPECTED_NUMERIC_OTHER = [
    'POI_DIVERSITY', 'POI_DENSITY_SCORE'
]

# Default bins used by the notebook
DISTANCE_BINS = [0, 250, 1000, math.inf]
DISTANCE_LABELS = ['<250m', '250-1000m', '>1000m']
COUNT_BINS = [0, 5, 20, math.inf]
COUNT_LABELS = ['0-4', '5-19', '20+']
DIVERSITY_BINS = [0, 0.25, 0.5, 0.75, math.inf]
DIVERSITY_LABELS = ['Very Low', 'Low', 'Medium', 'High']

# Columns the notebook ultimately items for FP mining
ITEM_COLS_FOR_MINING = [
    'OFNS_DESC', 'PREM_TYP_DESC', 'LAW_CAT', 'LOC_OF_OCCUR',
    'BORO', 'TIME_BUCKET', 'SEASON',
    'SUSP_AGE', 'SUSP_RACE', 'SUSP_SEX',
    'VIC_AGE', 'VIC_RACE', 'VIC_SEX',
    'DIST_BIN', 'COUNT_BIN', 'DIVERSITY_BIN', 'DENSITY_BIN',
    'IS_WEEKEND', 'IS_HOLIDAY', 'IS_PAYDAY', 'SAME_AGE', 'SAME_SEX', 'TO_CHECK_CITIZENS'
]


@dataclass
class NumericStats:
    column: str
    non_null: int
    nulls: int
    null_pct: float
    dtype: str
    min: Optional[float]
    p01: Optional[float]
    p05: Optional[float]
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    std: Optional[float]


TRUE_TOKENS = {'true', 't', '1', 'y', 'yes'}
FALSE_TOKENS = {'false', 'f', '0', 'n', 'no'}


def coerce_bool_like(series: pd.Series) -> pd.Series:
    s = series.astype('string').str.strip().str.lower()
    mapped = np.where(s.isin(TRUE_TOKENS), 'TRUE',
             np.where(s.isin(FALSE_TOKENS), 'FALSE', 'UNKNOWN'))
    return pd.Series(mapped, index=series.index, dtype='string')


def safe_quantiles(s: pd.Series, qs: List[float]) -> List[Optional[float]]:
    s_num = pd.to_numeric(s, errors='coerce').dropna()
    if s_num.empty:
        return [None] * len(qs)
    return [float(s_num.quantile(q)) for q in qs]


def numeric_profile(df: pd.DataFrame, cols: List[str]) -> List[NumericStats]:
    out: List[NumericStats] = []
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce')
        non_null = int(s.notna().sum())
        nulls = int(s.isna().sum())
        total = non_null + nulls
        qs = safe_quantiles(s, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        out.append(NumericStats(
            column=col,
            non_null=non_null,
            nulls=nulls,
            null_pct=(nulls / total * 100.0) if total else 0.0,
            dtype=str(df[col].dtype),
            min=float(s.min()) if non_null else None,
            p01=qs[0], p05=qs[1], p25=qs[2], p50=qs[3], p75=qs[4], p95=qs[5], p99=qs[6],
            max=float(s.max()) if non_null else None,
            mean=float(s.mean()) if non_null else None,
            std=float(s.std()) if non_null else None,
        ))
    return out


def bin_coverage(s: pd.Series, bins: List[float], labels: List[str]) -> Dict[str, float]:
    s_num = pd.to_numeric(s, errors='coerce')
    cat = pd.cut(s_num, bins=bins, labels=labels, include_lowest=True)
    dist = cat.value_counts(dropna=False, normalize=True).to_dict()
    # include NaN share as UNKNOWN
    unknown_share = float(dist.get(np.nan, 0.0))
    cov = {str(k): float(v) for k, v in dist.items() if not (isinstance(k, float) and math.isnan(k))}
    cov['UNKNOWN'] = unknown_share
    return cov


def quartile_labels(n: int) -> List[str]:
    return [f'Q{i+1}' for i in range(n)]


def propose_bins(df: pd.DataFrame) -> Dict[str, Any]:
    proposal: Dict[str, Any] = {'distances': {}, 'counts': {}, 'diversity': {}, 'density': {}}

    # Distances: use defaults and report coverage if present
    for col in [c for c in EXPECTED_NUMERIC_DISTANCES if c in df.columns]:
        proposal['distances'][col] = {
            'bins': DISTANCE_BINS,
            'labels': DISTANCE_LABELS,
            'coverage': bin_coverage(df[col], DISTANCE_BINS, DISTANCE_LABELS)
        }

    # Auto-detect any additional *_DISTANCE columns
    for col in df.columns:
        if col not in proposal['distances'] and col.upper().endswith('_DISTANCE'):
            proposal['distances'][col] = {
                'bins': DISTANCE_BINS,
                'labels': DISTANCE_LABELS,
                'coverage': bin_coverage(df[col], DISTANCE_BINS, DISTANCE_LABELS)
            }

    # Counts: use defaults and report coverage
    for col in [c for c in EXPECTED_NUMERIC_COUNTS if c in df.columns]:
        proposal['counts'][col] = {
            'bins': COUNT_BINS,
            'labels': COUNT_LABELS,
            'coverage': bin_coverage(df[col], COUNT_BINS, COUNT_LABELS)
        }

    # Auto-detect *_COUNT columns
    for col in df.columns:
        up = col.upper()
        if col not in proposal['counts'] and (up.endswith('_COUNT') or up.endswith('_COUNTS')):
            proposal['counts'][col] = {
                'bins': COUNT_BINS,
                'labels': COUNT_LABELS,
                'coverage': bin_coverage(df[col], COUNT_BINS, COUNT_LABELS)
            }

    # Diversity
    if 'POI_DIVERSITY' in df.columns:
        proposal['diversity']['POI_DIVERSITY'] = {
            'bins': DIVERSITY_BINS,
            'labels': DIVERSITY_LABELS,
            'coverage': bin_coverage(df['POI_DIVERSITY'], DIVERSITY_BINS, DIVERSITY_LABELS)
        }

    # Density: propose quartiles
    if 'POI_DENSITY_SCORE' in df.columns:
        s = pd.to_numeric(df['POI_DENSITY_SCORE'], errors='coerce')
        try:
            qcut = pd.qcut(s, q=4, labels=quartile_labels(4), duplicates='drop')
            coverage = qcut.value_counts(normalize=True, dropna=False).to_dict()
            unknown_share = float(coverage.get(np.nan, 0.0))
            cov = {str(k): float(v) for k, v in coverage.items() if not (isinstance(k, float) and math.isnan(k))}
            cov['UNKNOWN'] = unknown_share
            proposal['density']['POI_DENSITY_SCORE'] = {
                'strategy': 'quantiles',
                'q': 4,
                'labels': quartile_labels(len(set(qcut.dropna().unique()))) if qcut.notna().any() else quartile_labels(4),
                'coverage': cov
            }
        except Exception:
            proposal['density']['POI_DENSITY_SCORE'] = {
                'strategy': 'quantiles', 'q': 4, 'labels': quartile_labels(4), 'coverage': {'UNKNOWN': 1.0}
            }

    return proposal


def topn_coverage(series: pd.Series, n: int) -> Dict[str, Any]:
    vc = series.value_counts(dropna=False)
    total = int(vc.sum())
    top = vc.head(n)
    share = float(top.sum() / total) if total else 0.0
    return {
        'top_n': n,
        'coverage_pct': share * 100.0,
        'top_values': [{ 'value': str(idx), 'count': int(cnt) } for idx, cnt in top.items()]
    }


def build_profile(df: pd.DataFrame) -> Dict[str, Any]:
    present = df.columns.tolist()

    # Column presence by expectation groups
    presence = {
        'categorical_present': [c for c in EXPECTED_CATEGORICAL if c in df.columns],
        'categorical_missing': [c for c in EXPECTED_CATEGORICAL if c not in df.columns],
        'boolean_like_present': [c for c in EXPECTED_BOOLEAN_LIKE if c in df.columns],
        'boolean_like_missing': [c for c in EXPECTED_BOOLEAN_LIKE if c not in df.columns],
        'numeric_distances_present': [c for c in EXPECTED_NUMERIC_DISTANCES if c in df.columns],
        'numeric_distances_missing': [c for c in EXPECTED_NUMERIC_DISTANCES if c not in df.columns],
        'numeric_counts_present': [c for c in EXPECTED_NUMERIC_COUNTS if c in df.columns],
        'numeric_counts_missing': [c for c in EXPECTED_NUMERIC_COUNTS if c not in df.columns],
        'numeric_other_present': [c for c in EXPECTED_NUMERIC_OTHER if c in df.columns],
        'numeric_other_missing': [c for c in EXPECTED_NUMERIC_OTHER if c not in df.columns],
    }

    # Dtypes snapshot
    dtypes_map = {c: str(dt) for c, dt in df.dtypes.items()}

    # Numeric stats for all numeric-like columns (auto-detected)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_stats = [asdict(ns) for ns in numeric_profile(df, numeric_cols)]

    # Proposed bins and coverage
    bins = propose_bins(df)

    # Boolean-like normalization preview
    bool_previews: Dict[str, Dict[str, int]] = {}
    for col in presence['boolean_like_present']:
        mapped = coerce_bool_like(df[col])
        bool_previews[col] = {k: int(v) for k, v in mapped.value_counts(dropna=False).to_dict().items()}

    # TIME_BUCKET presence and suggestion from HOUR
    time_bucket_info: Dict[str, Any] = {}
    if 'TIME_BUCKET' in df.columns:
        time_bucket_info['present'] = True
        vc = df['TIME_BUCKET'].astype('string').str.strip().str.upper().value_counts(dropna=False).to_dict()
        time_bucket_info['value_counts'] = {str(k): int(v) for k, v in vc.items()}
    elif 'HOUR' in df.columns:
        hours = pd.to_numeric(df['HOUR'], errors='coerce').dropna().astype(int).clip(0, 23)
        hb = {'NIGHT': range(0, 6), 'MORNING': range(6, 12), 'AFTERNOON': range(12, 18), 'EVENING': range(18, 24)}
        bucket_map = {h: name for name, rng in hb.items() for h in rng}
        buckets = hours.map(bucket_map)
        time_bucket_info['present'] = False
        time_bucket_info['derived_from_hour'] = {k: int(v) for k, v in buckets.value_counts().to_dict().items()}
    else:
        time_bucket_info['present'] = False
        time_bucket_info['note'] = 'TIME_BUCKET and HOUR not found'

    # Top-N coverage suggestions
    topn_suggestions: Dict[str, Any] = {}
    topn_plan = {
        'OFNS_DESC': 30,
        'PREM_TYP_DESC': 25,
        'LOC_OF_OCCUR_DESC': 20,
        'SUSP_AGE_GROUP': 10,
        'VIC_AGE_GROUP': 10,
    }
    for col, n in topn_plan.items():
        if col in df.columns:
            s = df[col].astype('string').str.strip().str.upper().fillna('UNKNOWN')
            topn_suggestions[col] = topn_coverage(s, n)

    # Mining item columns presence snapshot (after notebook renames/caps)
    mining_presence = {
        'desired': ITEM_COLS_FOR_MINING,
        'will_be_available_after_processing': [
            # Map direct-to-processed assumptions
            'OFNS_DESC', 'PREM_TYP_DESC', 'LOC_OF_OCCUR', 'LAW_CAT',
            'BORO', 'TIME_BUCKET', 'SEASON',
            'SUSP_AGE', 'SUSP_RACE', 'SUSP_SEX',
            'VIC_AGE', 'VIC_RACE', 'VIC_SEX',
            'DIST_BIN', 'COUNT_BIN', 'DIVERSITY_BIN', 'DENSITY_BIN',
            'IS_WEEKEND', 'IS_HOLIDAY', 'IS_PAYDAY', 'SAME_AGE', 'SAME_SEX', 'TO_CHECK_CITIZENS'
        ]
    }

    return {
        'row_count': int(len(df)),
        'columns_present': present,
        'dtypes': dtypes_map,
        'presence_by_group': presence,
        'numeric_stats': num_stats,
        'proposed_bins': bins,
        'boolean_like_previews': bool_previews,
        'time_bucket': time_bucket_info,
        'topn_suggestions': topn_suggestions,
        'mining_item_columns': mining_presence,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def write_numeric_csv(path: Path, stats: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats).to_csv(path, index=False)


def write_category_value_counts(path: Path, df: pd.DataFrame, max_uniques: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, int]] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        vc = df[col].astype('string').str.strip().str.upper().value_counts()
        if len(vc) <= max_uniques:
            out[col] = {str(k): int(v) for k, v in vc.to_dict().items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)


def find_default_input() -> Optional[Path]:
    # Prefer repo-root relative JupyterOutputs path
    cand = Path('JupyterOutputs') / 'Final' / 'final_crime_data.csv'
    return cand if cand.exists() else None


def main():
    parser = argparse.ArgumentParser(description='Profile PatternAnalysis inputs and propose binning/caps.')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input CSV (default: JupyterOutputs/Final/final_crime_data.csv if exists)')
    parser.add_argument('--out', type=str, default=str(Path('JupyterOutputs') / 'PatternAnalysis' / 'Diagnostics'),
                        help='Output directory for diagnostics artifacts')
    parser.add_argument('--year', type=int, default=2024,
                        help='Optional YEAR filter to mirror the notebook. Use 0 to disable filtering.')
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_default_input()
    if not input_path or not input_path.exists():
        raise FileNotFoundError('Input CSV not found. Provide --input or place file at JupyterOutputs/Final/final_crime_data.csv')

    out_dir = Path(args.out)

    print(f'Reading: {input_path}')
    df = pd.read_csv(input_path, low_memory=False)

    # YEAR filter like the notebook
    if args.year and 'YEAR' in df.columns:
        pre = len(df)
        df = df[df['YEAR'] == int(args.year)].copy()
        print(f'Applied YEAR == {args.year} filter: {pre} -> {len(df)} rows')
    else:
        print('YEAR filter disabled or YEAR column not present')

    # Construct DATE if possible (for info only)
    for c in ['YEAR', 'MONTH', 'DAY']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    if 'DAY' in df.columns:
        df['DAY'] = df['DAY'].fillna(1)
    df['DATE'] = pd.to_datetime(
        df[['YEAR', 'MONTH', 'DAY']].rename(columns={'YEAR': 'year', 'MONTH': 'month', 'DAY': 'day'}),
        errors='coerce'
    )

    profile = build_profile(df)

    # Write artifacts
    json_path = out_dir / 'profile_summary.json'
    write_json(json_path, profile)
    write_numeric_csv(out_dir / 'numeric_stats.csv', profile['numeric_stats'])
    write_category_value_counts(out_dir / 'category_value_counts.json', df)

    print(f'Wrote: {json_path}')
    print(f'Wrote: {out_dir / "numeric_stats.csv"}')
    print(f'Wrote: {out_dir / "category_value_counts.json"}')

    # Human friendly hints
    missing_any = {
        'categorical': profile['presence_by_group']['categorical_missing'],
        'boolean_like': profile['presence_by_group']['boolean_like_missing'],
        'numeric_distances': profile['presence_by_group']['numeric_distances_missing'],
        'numeric_counts': profile['presence_by_group']['numeric_counts_missing'],
        'numeric_other': profile['presence_by_group']['numeric_other_missing'],
    }
    print('\nQuick summary:')
    print(f"Rows: {profile['row_count']}")
    for k, v in missing_any.items():
        if v:
            print(f'  Missing {k}: {v}')
    if 'time_bucket' in profile and not profile['time_bucket'].get('present', False):
        print('  TIME_BUCKET missing; HOUR-based derivation available in summary')


if __name__ == '__main__':
    main()
