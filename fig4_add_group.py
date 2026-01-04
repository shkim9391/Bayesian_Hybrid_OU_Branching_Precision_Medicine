#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#fig4_add_group.py

"""
Add clinical group labels to cohort_diagnostics.csv using Clinical_Data
(Ahlgren et al. 2025 Nat Commun) as the canonical source.

Inputs
------
1) Figure 4/cohort_diagnostics.csv
   (created by fig4_cohort_diagnostics.py)

2) kmt2a_clinical_data.xlsx
   - Sheet "Clinical_Data" (header row at index 3) with columns:
       Patient_ID, Group, ...

Output
------
1) Figure 4/kmt2a_cohort_diagnostics.csv
   Same as cohort_diagnostics.csv, plus a 'group' column.

This file is used by the R script:
  fig4_model_diagnostics_bayesian_ou_branching_cohort.R
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Paths – EDIT THESE IF YOUR LAYOUT CHANGES
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(
    "/Users/seung-hwan.kim/Desktop/Bayesian_Hybrid_OU_Branching_Precision_Medicine"
)

CLINICAL_PATH = PROJECT_ROOT / "kmt2a_clinical_data.xlsx"
FIG4_DIR = PROJECT_ROOT / "Figure 4"
FIG4_DIR.mkdir(parents=True, exist_ok=True)

COHORT_CSV = FIG4_DIR / "cohort_diagnostics.csv"
OUT_CSV    = FIG4_DIR / "kmt2a_cohort_diagnostics.csv"


# ---------------------------------------------------------------------
# Helper: load groups from Clinical_Data
# ---------------------------------------------------------------------
def load_groups_from_clinical(clin_path: Path) -> pd.DataFrame:
    """
    Load canonical (Patient_ID, Group) from Clinical_Data sheet.

    Header row is at index 3. If a patient has multiple group labels,
    we prioritize any that contain 'refractory'; otherwise we use the
    first non-missing label.
    """
    if not clin_path.exists():
        raise FileNotFoundError(f"Clinical metadata not found at {clin_path}")

    df = pd.read_excel(clin_path, sheet_name="Clinical_Data", header=3)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    if "Patient_ID" not in df.columns or "Group" not in df.columns:
        raise ValueError(
            f"'Clinical_Data' must contain Patient_ID and Group columns. "
            f"Found: {list(df.columns)}"
        )

    df = df[["Patient_ID", "Group"]].dropna(subset=["Patient_ID"]).copy()
    df["Patient_ID"] = df["Patient_ID"].astype(str)

    def pick_group(series: pd.Series) -> str:
        vals = [str(v).strip() for v in series.dropna()]
        if not vals:
            return np.nan
        refractory = [v for v in vals if "refractory" in v.lower()]
        return refractory[0] if refractory else vals[0]

    grouped = (
        df.groupby("Patient_ID", as_index=False)["Group"]
        .apply(pick_group)
        .rename(columns={"Group": "group"})
    )

    return grouped


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # 1) Load cohort diagnostics
    if not COHORT_CSV.exists():
        raise FileNotFoundError(f"Base cohort diagnostics not found at {COHORT_CSV}")

    diag = pd.read_csv(COHORT_CSV)
    diag["Patient_ID"] = diag["Patient_ID"].astype(str)

    # 2) Load canonical group labels from Clinical_Data
    groups = load_groups_from_clinical(CLINICAL_PATH)

    # 3) Merge group into diagnostics
    merged = diag.merge(groups, on="Patient_ID", how="left")

    # 4) Save
    merged.to_csv(OUT_CSV, index=False)
    print(f"[+] Saved kmt2a_cohort_diagnostics.csv to:\n    {OUT_CSV}")

    # 5) Quick sanity check for the four key patients
    print("\n[CHECK] Key patients P28, P68, P135, P136:")
    print(merged[merged["Patient_ID"].isin(["P28", "P68", "P135", "P136"])])


if __name__ == "__main__":
    main()
