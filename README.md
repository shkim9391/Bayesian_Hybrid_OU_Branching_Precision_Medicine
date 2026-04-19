# Stage-specific evolutionary dynamics in pediatric leukemia inferred using Bayesian stochastic modeling

Code and processed supporting data for the PLOS ONE submission  
**Stage-specific evolutionary dynamics in pediatric leukemia inferred using Bayesian stochastic modeling**

This repository contains the analysis scripts, processed input tables, and figure-supporting data used for the PLOS ONE submission. The repository is organized to make it straightforward to reproduce the main figures, supplementary figures, and supporting tables from the processed longitudinal dataset.

## Repository overview

The repository includes:

- main figure scripts
- supplementary figure and table scripts
- processed input datasets
- posterior summary and diagnostics tables
- exported longitudinal series used for plotting and quality control

## Main analysis scripts

- `fig1_v3.py`  
  Generates the final version of Figure 1.

- `fig2_final.py`  
  Generates the final version of Figure 2.

- `fit_fig2_posterior.py`  
  Fits or summarizes the posterior quantities used in Figure 2.

- `fig3_final.py`  
  Generates the final version of Figure 3.

- `fig4_final_P15.py`  
  Generates the final version of Figure 4.

- `fig5_add_group.py`  
  Produces grouped outputs used in Figure 5.

- `fig5_cohort_diagnostics.py`  
  Produces cohort-level diagnostic outputs used in Figure 5.

- `fig5_model_diagnostics_bayesian_ou_branching.py`  
  Produces Bayesian OU-branching model diagnostic outputs associated with Figure 5.

## Supplementary scripts

- `Sfig1-2_table1_ou_longitudinal_compare.py`  
  Generates Supplementary Figures S1-S2 and Table 1.

- `Sfig3-7_final_all_patients.py`  
  Generates Supplementary Figures S3-S7 across all patients.

- `figs8_remission_summary.py`  
  Generates Supplementary Figure S8.

- `make_S9_table.py`  
  Generates Supplementary Table S9.

## Processing and helper scripts

- `export_and_qc_series.py`  
  Exports processed longitudinal series and quality-control outputs used downstream.

- `phase_ou_fit.py`  
  Runs phase-based OU fitting/calibration steps used in the analysis workflow.

- `phase_ou_plots_P15_P28.py`  
  Generates phase-specific plots for selected cases.

## Data files in this repository

- `kmt2a_longitudinal_clean.xlsx`  
  Cleaned longitudinal dataset used as the main processed input for the analysis.

- `kmt2a_clinical_data.xlsx`  
  Clinical metadata used alongside the longitudinal dataset.

- `phase_ou_data.csv`  
  Phase-level calibration or fitting input/output table used in the Figure 2 workflow.

- `posterior_summary.csv`  
  Full posterior summary table from the OU-based modeling workflow.

- `patient_group.csv`  
  Patient/group assignment table used in grouped analyses.

- `series_auto.csv`  
  Preprocessed longitudinal series used for Figure 4-level analyses.

- `series_auto_debug.csv`  
  Debug version of the longitudinal series for verification and troubleshooting.

- `cohort_diagnostics.csv`  
  Cohort-level diagnostic summary table.

- `kmt2a_cohort_diagnostics.csv`  
  Additional cohort diagnostics table for KMT2A-focused summaries.

- `ou_methods_comparison_real.csv`  
  Comparison table used for OU-method benchmarking/comparison.

## Supporting data files for journal submission

For the PLOS ONE submission, the supporting datasets are organized as separate supporting data files:

- **S1 Data.** Complete cleaned longitudinal dataset (input to all analyses)
- **S2 Data.** Raw calibration outputs for Figure 2
- **S3 Data.** Group-level OU parameter means underlying Figure 3
- **S4 Data.** Group-level OU parameter HDIs underlying Figure 3
- **S5 Data.** Full OU posterior summaries
- **S6 Data.** Preprocessed longitudinal series for Figure 4
- **S7 Data.** Debug version of the longitudinal series for Figure 4
- **S8 Data.** Cohort-level diagnostics underlying Figure 5

These supporting data files correspond to the processed tables included in this repository and/or their journal-packaged counterparts.

## Typical loading in Python

A typical starting point is:

```python
import pandas as pd

longitudinal = pd.read_excel("kmt2a_longitudinal_clean.xlsx")
clinical = pd.read_excel("kmt2a_clinical_data.xlsx")

phase_data = pd.read_csv("phase_ou_data.csv")
posterior = pd.read_csv("posterior_summary.csv")

series = pd.read_csv("series_auto.csv")
series_debug = pd.read_csv("series_auto_debug.csv")

cohort_diag = pd.read_csv("cohort_diagnostics.csv")
kmt2a_cohort_diag = pd.read_csv("kmt2a_cohort_diagnostics.csv")
patient_group = pd.read_csv("patient_group.csv")

## Example execution

python export_and_qc_series.py
python phase_ou_fit.py
python fit_fig2_posterior.py

python fig1_v3.py
python fig2_final.py
python fig3_final.py
python fig4_final_P15.py
python fig5_add_group.py
python fig5_cohort_diagnostics.py
python fig5_model_diagnostics_bayesian_ou_branching.py

python Sfig1-2_table1_ou_longitudinal_compare.py
python Sfig3-7_final_all_patients.py
python figs8_remission_summary.py
python make_S9_table.py

## Software environment

This repository is intended to run in Python 3.x with the standard scientific Python stack used for the manuscript analyses, typically including:
	•	pandas
	•	numpy
	•	matplotlib
	•	scipy
	•	openpyxl

Additional packages may be required depending on the exact modeling code used in your local environment.

## Reproducibility note

This repository is designed for reproducibility of the PLOS ONE submission using processed analysis-ready data. The files provided here are the processed inputs and derived outputs used to generate the figures, supplementary figures, and supporting tables included with the manuscript.

[![DOI](https://zenodo.org/badge/1097191765.svg)](https://doi.org/10.5281/zenodo.19618927)
