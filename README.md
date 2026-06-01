# ML_scikit-survival
how to handle right-censored data, train a machine learning survival model, and evaluate it using clinical metrics.
## Project 2: Oncology Survival Analysis & Patient Risk Stratification

### Objective
To build an automated machine learning workflow capable of predicting Progression-Free Survival (PFS) in breast cancer cohorts using both standard linear and non-linear survival algorithms.

### Methodology
* **Data Ingestion:** Utilized the German Breast Cancer Study Group (GBSG2) clinical cohort.
* **Pre-processing:** Implemented dummy variable encoding for clinical staging and applied standard scaling across longitudinal numerical covariates (e.g., progesterone/estrogen receptor values).
* **Modeling:** Trained a Baseline Cox Proportional Hazards Model and an Advanced Random Survival Forest (RSF).
* **Evaluation:** Quantified predictive accuracy using the Concordance Index (C-index) to manage right-censored trial metrics natively.
