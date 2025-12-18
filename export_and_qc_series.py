#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 00:28:37 2025

@author: seung-hwan.kim
"""

# export_and_qc_series.py

"""
Deterministically generate Supplementary Data S6 (series_auto.csv) and
Supplementary Data S7 (series_auto_debug.csv) from Supplementary Data S1
(kmt2a_longitudinal_clean.xlsx).

S7 column list matches your current table:
Patient_ID, Group, Disease, Day, Sample name, Type, Tissue, Material,
KMT2A-fusion detected, Rescued Samples, ClinicalData,
Leukemic Cells Detected at CR, t_years, x_value, n_value

Usage:
  python export_and_qc_series.py \
    --excel "/path/to/kmt2a_longitudinal_clean.xlsx" \
    --sheet "Series" \
    --out_s6 "/path/to/series_auto.csv" \
    --out_s7 "/path/to/series_auto_debug.csv"

Notes:
- Assumes S1 contains a tidy "Series" sheet with columns: Patient_ID, series, t, value
- Assumes S1 also contains a raw/metadata sheet (default: "Samples") used to build S7.
  If your workbook uses a different name, pass --samples_sheet.
- Sorting + type normalization are fixed to ensure identical output across runs.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DAYS_PER_YEAR = 365.25

# ---------- utilities ----------
def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _norm_yes_no(x):
    s = "" if pd.isna(x) else str(x).strip()
    sl = s.lower()
    if sl in ("yes", "y", "true", "1"):
        return "Yes"
    if sl in ("no", "n", "false", "0", ""):
        return "No"
    # preserve original (but stripped) if nonstandard
    return s

def _coerce_str(df, col):
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df

def _coerce_num(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ---------- S6 from Series ----------
def make_s6_from_series(series_df: pd.DataFrame) -> pd.DataFrame:
    df = _strip_cols(series_df)

    need = {"Patient_ID", "series", "t", "value"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Series sheet missing columns: {missing}. Found: {df.columns.tolist()}")

    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    df["series"]     = df["series"].astype("string").str.strip().str.lower()
    df["t"]          = pd.to_numeric(df["t"], errors="coerce")
    df["value"]      = pd.to_numeric(df["value"], errors="coerce")

    # Keep only x/n, drop incomplete rows
    df = df.loc[df["series"].isin(["x", "n"])].dropna(subset=["Patient_ID", "series", "t", "value"]).copy()

    # Deterministic ordering
    df = df.sort_values(["Patient_ID", "series", "t", "value"], kind="mergesort").reset_index(drop=True)

    # Canonical dtypes (helps stable CSV)
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    df["series"]     = df["series"].astype(str)
    df["t"]          = df["t"].astype(float)
    df["value"]      = df["value"].astype(float)

    return df[["Patient_ID", "series", "t", "value"]]

# ---------- S7 from Samples + Series ----------
def make_s7_debug(samples_df: pd.DataFrame, s6_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the exact S7 wide table by taking the metadata rows (one per sample/timepoint),
    computing t_years from Day, and attaching x_value/n_value at that timepoint.

    Deterministic behavior:
    - stable sort by Patient_ID then Day then Sample name
    - stable merge for x_value/n_value
    """

    df = _strip_cols(samples_df)

    # Normalize possible column variants to exact S7 headers
    rename_map = {
        "Patient": "Patient_ID",
        "patient": "Patient_ID",
        "patient_id": "Patient_ID",
        "Patient ID": "Patient_ID",

        "Sample_name": "Sample name",
        "Sample Name": "Sample name",
        "sample name": "Sample name",

        "KMT2A_fusion_detected": "KMT2A-fusion detected",
        "KMT2A-fusion detected": "KMT2A-fusion detected",
        "KMT2A fusion detected": "KMT2A-fusion detected",

        "Rescued_Samples": "Rescued Samples",
        "rescued samples": "Rescued Samples",

        "Clinical_Data": "ClinicalData",
        "clinicaldata": "ClinicalData",

        "Leukemic_Cells_Detected_at_CR": "Leukemic Cells Detected at CR",
        "Leukemic cells detected at CR": "Leukemic Cells Detected at CR",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Required metadata columns for your S7
    required = [
        "Patient_ID", "Group", "Disease", "Day", "Sample name", "Type", "Tissue", "Material",
        "KMT2A-fusion detected", "Rescued Samples", "ClinicalData", "Leukemic Cells Detected at CR"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Samples sheet missing required columns for S7:\n"
            f"Missing: {missing}\n"
            f"Found: {df.columns.tolist()}"
        )

    # Normalize types (deterministic)
    df = _coerce_str(df, "Patient_ID")
    df = _coerce_str(df, "Group")
    df = _coerce_str(df, "Disease")
    df = _coerce_str(df, "Sample name")
    df = _coerce_str(df, "Type")
    df = _coerce_str(df, "Tissue")
    df = _coerce_str(df, "Material")
    df = _coerce_str(df, "ClinicalData")
    df = _coerce_str(df, "Leukemic Cells Detected at CR")

    df["KMT2A-fusion detected"] = df["KMT2A-fusion detected"].map(_norm_yes_no)
    df["Rescued Samples"]       = df["Rescued Samples"].map(_norm_yes_no)

    df = _coerce_num(df, "Day")

    # Compute t_years
    df["t_years"] = df["Day"] / DAYS_PER_YEAR

    # Attach x_value and n_value from S6 by matching (Patient_ID, t_years) to (Patient_ID, t)
    # Use rounded matching to avoid floating drift (deterministic). Keep 12 decimals.
    meta = df.copy()
    meta["t_key"] = meta["t_years"].round(12)

    s6 = s6_df.copy()
    s6["t_key"] = s6["t"].round(12)

    x = s6.loc[s6["series"] == "x", ["Patient_ID", "t_key", "value"]].rename(columns={"value": "x_value"})
    n = s6.loc[s6["series"] == "n", ["Patient_ID", "t_key", "value"]].rename(columns={"value": "n_value"})

    meta = meta.merge(x, on=["Patient_ID", "t_key"], how="left", sort=False)
    meta = meta.merge(n, on=["Patient_ID", "t_key"], how="left", sort=False)

    # If missing, leave NaN (deterministic); then cast n_value to Int64 if possible
    meta["x_value"] = pd.to_numeric(meta["x_value"], errors="coerce")
    meta["n_value"] = pd.to_numeric(meta["n_value"], errors="coerce")

    # Deterministic ordering (stable sort)
    meta = meta.sort_values(["Patient_ID", "Day", "Sample name"], kind="mergesort").reset_index(drop=True)

    # Output exact S7 column order
    out_cols = [
        "Patient_ID", "Group", "Disease", "Day", "Sample name", "Type", "Tissue", "Material",
        "KMT2A-fusion detected", "Rescued Samples", "ClinicalData", "Leukemic Cells Detected at CR",
        "t_years", "x_value", "n_value"
    ]
    out = meta[out_cols].copy()

    # Canonical dtypes for stable CSV writing
    out["Patient_ID"] = out["Patient_ID"].astype(str)
    for c in ["Group", "Disease", "Sample name", "Type", "Tissue", "Material", "ClinicalData", "Leukemic Cells Detected at CR"]:
        out[c] = out[c].astype(str)
    out["KMT2A-fusion detected"] = out["KMT2A-fusion detected"].astype(str)
    out["Rescued Samples"] = out["Rescued Samples"].astype(str)

    out["Day"] = pd.to_numeric(out["Day"], errors="coerce")
    out["t_years"] = pd.to_numeric(out["t_years"], errors="coerce")
    out["x_value"] = pd.to_numeric(out["x_value"], errors="coerce")
    out["n_value"] = pd.to_numeric(out["n_value"], errors="coerce")

    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to kmt2a_longitudinal_clean.xlsx (Supplementary Data S1)")
    ap.add_argument("--sheet", default="Series", help='Series sheet name (default: "Series")')
    ap.add_argument("--samples_sheet", default="Samples",
                    help='Raw/metadata sheet name used to build S7 (default: "Samples"). '
                         'Set this to the sheet that contains Day/Type/ClinicalData/etc.')
    ap.add_argument("--out_s6", default="series_auto.csv", help="Output path for Supplementary Data S6")
    ap.add_argument("--out_s7", default="series_auto_debug.csv", help="Output path for Supplementary Data S7")
    args = ap.parse_args()

    xlsx = Path(args.excel)
    if not xlsx.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx}")

    # Load sheets
    series_sheet = pd.read_excel(xlsx, sheet_name=args.sheet)
    s6 = make_s6_from_series(series_sheet)

    # Load samples sheet for S7
    try:
        samples_sheet = pd.read_excel(xlsx, sheet_name=args.samples_sheet)
    except ValueError as e:
        raise ValueError(
            f"Could not load samples sheet '{args.samples_sheet}'. "
            f"Available sheets: {pd.ExcelFile(xlsx).sheet_names}"
        ) from e

    s7 = make_s7_debug(samples_sheet, s6)

    # Write outputs deterministically (fixed float format)
    out_s6 = Path(args.out_s6).resolve()
    out_s7 = Path(args.out_s7).resolve()
    out_s6.parent.mkdir(parents=True, exist_ok=True)
    out_s7.parent.mkdir(parents=True, exist_ok=True)

    s6.to_csv(out_s6, index=False, float_format="%.12g")
    s7.to_csv(out_s7, index=False, float_format="%.15g")

    print(f"[OK] Wrote S6 → {out_s6} (rows={len(s6)})")
    print(f"[OK] Wrote S7 → {out_s7} (rows={len(s7)})")
    print("\nS6 preview:")
    print(s6.head(10))
    print("\nS7 preview:")
    print(s7.head(5))

if __name__ == "__main__":
    main()