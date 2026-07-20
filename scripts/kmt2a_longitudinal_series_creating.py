import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# CONFIGURABLE DEFAULTS
# =========================
DAYS_PER_YEAR      = 365.25

# ---- x (trait) defaults (edit if you prefer)
X_DIAGNOSIS        = 1.00
X_RELAPSE          = 1.00
X_CR               = 0.05
X_FUSION_YES       = 0.50
X_LONGITUDINAL_DEF = 0.10

# ---- n (clone count) defaults
N_FUSION_YES       = 2
N_MRD_OR_LEUKEMIC  = 1
N_REMISSION        = 0

RELAPSE_TOKENS     = ("relapse", "relapse1", "relapse2")
CR_TOKENS          = ("cr",)
LEUKEMIC_TOKENS    = ("yes",)

# =========================
# HELPERS
# =========================
def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def extract_mrd_fraction(clinical_text: str):
    s = norm_str(clinical_text)
    m = re.search(r"mrd\s*([0-9]{1,3})", s)
    if m:
        frac = int(m.group(1)) / 100.0
        return float(np.clip(frac, 0.0, 1.0))
    return None

def is_relapse(type_text: str) -> bool:
    s = norm_str(type_text)
    return any(tok in s for tok in RELAPSE_TOKENS)

def is_cr(clinical_text: str) -> bool:
    s = norm_str(clinical_text)
    return "cr" in re.split(r"[\s,;/]+", s)

def yes_no_to_bool(x):
    s = norm_str(x)
    if s in ("yes", "y", "true", "1"):
        return True
    if s in ("no", "n", "false", "0", ""):
        return False
    return False

# =========================
# INFERENCE RULES
# =========================
def infer_x_value(row):
    typ   = norm_str(row.get("Type", ""))
    clin  = norm_str(row.get("ClinicalData", ""))
    kmt2a = yes_no_to_bool(row.get("KMT2A-fusion detected", row.get("KMT2A_fusion_detected", "")))

    mrd_frac = extract_mrd_fraction(clin)
    if mrd_frac is not None:
        return mrd_frac
    if "diagnosis" in typ:
        return X_DIAGNOSIS
    if is_relapse(typ):
        return X_RELAPSE
    if is_cr(clin):
        return X_CR
    if kmt2a:
        return X_FUSION_YES
    if "longitudinal" in typ or typ:
        return X_LONGITUDINAL_DEF
    return None

def infer_n_value(row):
    clin   = norm_str(row.get("ClinicalData", ""))
    leuk   = norm_str(row.get("Leukemic Cells Detected at CR", row.get("Leukemic_Cells_Detected_at_CR", "")))
    kmt2a  = yes_no_to_bool(row.get("KMT2A-fusion detected", row.get("KMT2A_fusion_detected", "")))

    if kmt2a:
        return N_FUSION_YES
    if extract_mrd_fraction(clin) is not None:
        return N_MRD_OR_LEUKEMIC
    if leuk in LEUKEMIC_TOKENS:
        return N_MRD_OR_LEUKEMIC
    return N_REMISSION

# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to kmt2a_longitudinal_data.xlsx")
    ap.add_argument("--sheet", default=None, help="Worksheet name (default: first sheet)")
    ap.add_argument("--out",   default="series_auto.csv", help="Output tidy CSV")
    ap.add_argument("--debug", default="series_auto_debug.csv", help="Output QA CSV with all columns")
    args = ap.parse_args()

    xlsx_path = Path(args.excel)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    # 1) Load Excel
    df = pd.read_excel(xlsx_path, sheet_name=args.sheet)
    df.columns = [c.strip() for c in df.columns]
    
   # ---- If the sheet is already a tidy "Series" table, just export it
    tidy_need = {"Patient_ID", "series", "t", "value"}
    if tidy_need.issubset(df.columns):
        df2 = df.copy()
    
        # normalize
        df2["Patient_ID"] = df2["Patient_ID"].astype("string").str.strip()
        df2["series"]     = df2["series"].astype("string").str.strip().str.lower()
        df2["t"]          = pd.to_numeric(df2["t"], errors="coerce")
        df2["value"]      = pd.to_numeric(df2["value"], errors="coerce")
    
        # keep only valid rows
        series_df = (df2.loc[df2["series"].isin(["x", "n"])]
                       .dropna(subset=["Patient_ID", "series", "t", "value"])
                       .sort_values(["Patient_ID", "series", "t"])
                       .reset_index(drop=True))
    
        # create a TRUE debug table with QA fields
        debug_df = series_df.copy()
    
        out_path   = Path(args.out).resolve()
        debug_path = Path(args.debug).resolve()
        print("Writing OUT  :", out_path)
        print("Writing DEBUG:", debug_path)
        if out_path == debug_path:
            raise ValueError("out and debug paths are the same! Choose different filenames.")
    
        out_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
    
        series_df.to_csv(out_path, index=False)
        debug_df.to_csv(debug_path, index=False)
    
        print(f"[OK] Input sheet already tidy. Wrote → {out_path} (rows={len(series_df)})")
        print(f"[OK] Wrote debug table → {debug_path} (rows={len(debug_df)})")
        print(series_df.head(12))
        return

    # 2) Normalize column names
    rename_map = {
        "Patient": "Patient_ID",
        "patient": "Patient_ID",
        "patient_id": "Patient_ID",
        "Day": "Day",
        "day": "Day",
        "Type": "Type",
        "type": "Type",
        "ClinicalData": "ClinicalData",
        "clinicaldata": "ClinicalData",
        "KMT2A-fusion detected": "KMT2A-fusion detected",
        "KMT2A_fusion_detected": "KMT2A-fusion detected",
        "Leukemic Cells Detected at CR": "Leukemic Cells Detected at CR",
        "Leukemic_Cells_Detected_at_CR": "Leukemic Cells Detected at CR",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    for r in ("Patient_ID", "Day"):
        if r not in df.columns:
            raise ValueError(f"Required column '{r}' not found. Columns: {df.columns.tolist()}")

    # 3) Compute t (years)
    df["t_years"] = df["Day"].astype(float) / DAYS_PER_YEAR

    # 4) Infer values
    df["x_value"] = df.apply(infer_x_value, axis=1)
    df["n_value"] = df.apply(infer_n_value, axis=1)

    # 5) Build tidy series table
    rows = []
    for _, row in df.iterrows():
        pid = row["Patient_ID"]
        t   = float(row["t_years"])

        xv = row["x_value"]
        if xv is not None:
            rows.append({"Patient_ID": pid, "series": "x", "t": t, "value": float(xv)})

        nv = row["n_value"]
        if nv is not None:
            rows.append({"Patient_ID": pid, "series": "n", "t": t, "value": int(nv)})

    series_df = pd.DataFrame(rows).sort_values(["Patient_ID", "series", "t"]).reset_index(drop=True)

    out_path = Path(args.out).resolve()
    debug_path = Path(args.debug).resolve()
    print("Writing OUT  :", out_path)
    print("Writing DEBUG:", debug_path)
    if out_path == debug_path:
        raise ValueError("out and debug paths are the same! Choose different filenames.")
    
    # 6) Write outputs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.debug).parent.mkdir(parents=True, exist_ok=True)

    series_df.to_csv(args.out, index=False)
    df.to_csv(args.debug, index=False)

    print(f"[OK] Wrote tidy series → {args.out}  (rows={len(series_df)})")
    print(f"[OK] Wrote debug table → {args.debug} (rows={len(df)})")
    print("\nPreview:")
    print(series_df.head(12))

if __name__ == "__main__":
    main()
