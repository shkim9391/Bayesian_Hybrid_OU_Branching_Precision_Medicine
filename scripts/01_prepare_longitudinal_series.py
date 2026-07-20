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
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()
    sl = s.lower()

    if sl in ("yes", "y", "true", "1"):
        return "Yes"
    if sl in ("no", "n", "false", "0"):
        return "No"
    if sl == "":
        return pd.NA

    return s

def _coerce_str(df, col):
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()
    return df

def _coerce_num(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ---------- canonical model input from Series ----------
def make_series_input(series_df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.sort_values(
    ["Patient_ID", "series", "t"],
    kind="mergesort",
    ).reset_index(drop=True)

    # Canonical dtypes (helps stable CSV)
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    df["series"]     = df["series"].astype(str)
    df["t"]          = df["t"].astype(float)
    df["value"]      = df["value"].astype(float)

    return df[["Patient_ID", "series", "t", "value"]]

# ---------- optional debug table from metadata + Series ----------
def make_debug_table(samples_df: pd.DataFrame, s6_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the optional sample-level debug table by taking the metadata rows (one per sample/timepoint),
    computing t_years from Day, and attaching x_value/n_value at that timepoint.

    Deterministic behavior:
    - stable sort by Patient_ID then Day then Sample name
    - stable merge for x_value/n_value
    """

    df = _strip_cols(samples_df)

    # Normalize possible column variants to canonical debug-table headers
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

    # Required metadata columns for the debug table
    required = [
        "Patient_ID", "Group", "Disease", "Day", "Sample name", "Type", "Tissue", "Material",
        "KMT2A-fusion detected", "Rescued Samples", "ClinicalData", "Leukemic Cells Detected at CR"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Metadata sheet missing required columns for the debug table:\n"
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

    missing_key_rows = df[["Patient_ID", "Day"]].isna().any(axis=1)
    
    if missing_key_rows.any():
        raise ValueError(
            f"Metadata contains {int(missing_key_rows.sum())} rows with "
            "missing Patient_ID or Day."
        )

    # Compute t_years
    df["t_years"] = df["Day"] / DAYS_PER_YEAR

    # Preserve original metadata-row order and construct integer day keys.
    meta = df.copy()
    meta["_source_order"] = np.arange(len(meta))
    meta["day_key"] = np.rint(meta["Day"]).astype("Int64")
    
    meta = meta.sort_values(
        ["Patient_ID", "day_key", "_source_order"],
        kind="mergesort",
    )
    
    meta["sample_index"] = (
        meta.groupby(["Patient_ID", "day_key"], sort=False)
        .cumcount()
    )
    
    # Construct the corresponding day keys from the canonical Series times.
    series = s6_df.copy()
    series["_source_order"] = np.arange(len(series))
    series["day_key"] = np.rint(
        series["t"] * DAYS_PER_YEAR
    ).astype("Int64")
    
    series = series.sort_values(
        ["Patient_ID", "series", "day_key", "_source_order"],
        kind="mergesort",
    )
    
    series["sample_index"] = (
        series.groupby(
            ["Patient_ID", "series", "day_key"],
            sort=False,
        )
        .cumcount()
    )
    
    counts = (
        series.groupby(["Patient_ID", "day_key", "series"])
        .size()
        .unstack("series", fill_value=0)
    )
    
    for series_name in ["x", "n"]:
        if series_name not in counts.columns:
            counts[series_name] = 0
    
    mismatched_xn = counts.loc[counts["x"] != counts["n"]]
    
    if not mismatched_xn.empty:
        raise ValueError(
            "The numbers of x and n observations differ for some "
            "patient-day combinations:\n"
            f"{mismatched_xn.head(20)}"
        )
    
    metadata_counts = (
        meta.groupby(["Patient_ID", "day_key"])
        .size()
        .rename("metadata_rows")
    )
    
    series_counts = counts["x"].rename("series_rows")
    
    comparison = pd.concat(
        [metadata_counts, series_counts],
        axis=1,
    ).fillna(0)
    
    count_mismatches = comparison.loc[
        comparison["metadata_rows"] != comparison["series_rows"]
    ]
    
    if not count_mismatches.empty:
        raise ValueError(
            "Metadata-row counts do not match Series-row counts for some "
            "patient-day combinations:\n"
            f"{count_mismatches.head(20)}"
        )
    
    x = (
        series.loc[
            series["series"] == "x",
            ["Patient_ID", "day_key", "sample_index", "value"],
        ]
        .rename(columns={"value": "x_value"})
    )
    
    n = (
        series.loc[
            series["series"] == "n",
            ["Patient_ID", "day_key", "sample_index", "value"],
        ]
        .rename(columns={"value": "n_value"})
    )
    
    meta = meta.merge(
        x,
        on=["Patient_ID", "day_key", "sample_index"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    
    meta = meta.merge(
        n,
        on=["Patient_ID", "day_key", "sample_index"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    
    meta["x_value"] = pd.to_numeric(
        meta["x_value"],
        errors="coerce",
    )
    meta["n_value"] = pd.to_numeric(
        meta["n_value"],
        errors="coerce",
    )
    
    unmatched = meta.loc[
        meta["x_value"].isna() | meta["n_value"].isna(),
        [
            "Patient_ID",
            "Day",
            "Sample name",
            "day_key",
            "sample_index",
        ],
    ]
    
    if not unmatched.empty:
        raise ValueError(
            "Some metadata rows could not be matched to both x and n values:\n"
            f"{unmatched.head(20)}"
        )
    
    # Deterministic final ordering for the debug output.
    meta = meta.sort_values(
        ["Patient_ID", "Day", "Sample name"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Canonical debug-table column order
    out_cols = [
        "Patient_ID", "Group", "Disease", "Day", "Sample name", "Type", "Tissue", "Material",
        "KMT2A-fusion detected", "Rescued Samples", "ClinicalData", "Leukemic Cells Detected at CR",
        "t_years", "x_value", "n_value"
    ]
    out = meta[out_cols].copy()

    # Canonical dtypes for stable CSV writing
    string_columns = [
        "Patient_ID",
        "Group",
        "Disease",
        "Sample name",
        "Type",
        "Tissue",
        "Material",
        "ClinicalData",
        "Leukemic Cells Detected at CR",
        "KMT2A-fusion detected",
        "Rescued Samples",
    ]

    for column in string_columns:
        out[column] = out[column].astype("string").str.strip()

    out["Day"] = pd.to_numeric(out["Day"], errors="coerce")
    out["t_years"] = pd.to_numeric(out["t_years"], errors="coerce")
    out["x_value"] = pd.to_numeric(out["x_value"], errors="coerce")
    out["n_value"] = pd.to_numeric(out["n_value"], errors="coerce")

    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--clean_excel",
        required=True,
        help=(
            "Workbook containing the canonical tidy longitudinal "
            "Series sheet."
        ),
    )

    ap.add_argument(
        "--series_sheet",
        default="Series",
        help='Tidy longitudinal-series sheet name (default: "Series").',
    )

    ap.add_argument(
        "--metadata_excel",
        default=None,
        help=(
            "Optional original workbook containing sample-level metadata. "
            "Required only when generating the debug CSV."
        ),
    )

    ap.add_argument(
        "--metadata_sheet",
        default="Samples",
        help='Metadata sheet name (default: "Samples").',
    )

    ap.add_argument(
        "--out_series",
        default="series_auto.csv",
        help="Output path for the canonical tidy model-input CSV.",
    )

    ap.add_argument(
        "--out_debug",
        default=None,
        help=(
            "Optional output path for the sample-level debug CSV. "
            "Requires --metadata_excel."
        ),
    )

    args = ap.parse_args()

    clean_path = Path(args.clean_excel)
    
    if not clean_path.exists():
        raise FileNotFoundError(clean_path)
        
    series_df = pd.read_excel(
        clean_path,
        sheet_name=args.series_sheet,
    )
    
    series_out = make_series_input(series_df)
    
    out_series = Path(args.out_series).resolve()
    out_series.parent.mkdir(parents=True, exist_ok=True)
    
    series_out.to_csv(
        out_series,
        index=False,
        float_format="%.12g",
    )
    
    print(
        f"[OK] Wrote canonical series file: "
        f"{out_series} (rows={len(series_out)})"
    )
    
    if args.out_debug:
        if not args.metadata_excel:
            raise ValueError(
                "--metadata_excel is required when --out_debug is specified."
            )
    
        metadata_path = Path(args.metadata_excel)
    
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
    
        metadata_df = pd.read_excel(
            metadata_path,
            sheet_name=args.metadata_sheet,
        )
    
        debug_out = make_debug_table(
            metadata_df,
            series_out,
        )
    
        out_debug = Path(args.out_debug).resolve()
        out_debug.parent.mkdir(parents=True, exist_ok=True)
    
        debug_out.to_csv(
            out_debug,
            index=False,
            float_format="%.15g",
        )
    
        print(
            f"[OK] Wrote debug file: "
            f"{out_debug} (rows={len(debug_out)})"
        )

if __name__ == "__main__":
    main()
