# Project Scope and Requirements — az-ml-experiments
## Three-Sprint Implementation Plan

**Date:** April 2026  
**Scope:** Full pipeline execution: infrastructure provisioning → data pull (19 sources) → feature engineering → feature selection → XGBoost training → HyperDrive sweeps → inference + SHAP explainability  
**Resource:** 1 data scientist, ~3 × 2-week sprints alongside other work (~50–60% capacity)  
**Tech stack:** Azure Machine Learning, ADLS Gen2, Key Vault, MLflow, HyperDrive  
**Reference:** See `azure-ml-usage-plan.md` for VM sizing, cluster configuration, and compute cost estimates

---

## TL;DR

The pipeline code is complete. All 19 data pull notebooks, feature engineering notebooks, model development notebooks, and the HyperDrive training script (`src/train/train_outcome.py`) are already implemented. A single data scientist's work across three sprints is infrastructure setup, running and debugging the pipeline, iterating on HyperDrive sweeps, and interpreting results — not building from scratch. The pipeline will be fully operational by end of Sprint 2; Sprint 3 is refinement, interpretation, and documentation.

---

## Pipeline Overview

```
01_data_pull/          19 parallel notebooks → raw parquet per source
        ↓
02_feature_engineering/
   01_prio_grid_spatial_features   (requires 01/09 + 01/04)
   02_build_feature_matrix         (requires all of 01_data_pull)
   03_engineer_derived_features    (requires 02/02)
        ↓
03_model_development/
   01_feature_selection_lasso      (LASSO CV + MI rescue, per outcome)
   02_train_xgboost_outcomes       (5 models × RandomizedSearchCV)
   03_interrogate_nonlinearity     (PDP, SHAP interactions, linearity audit)
        ↓
04_inference/
   01_xgboost_inference            (per-country SHAP artifacts)
```

**Five target outcomes:** `civil_war_onset`, `coup_attempt`, `regime_backsliding`, `mass_unrest_onset`, `humanitarian_crisis_onset`

**Temporal design:** Train ≤ 2018 | Validation 2019–2021 | Test ≥ 2022. All outcomes forward-shifted +1 year (features at *t* predict label at *t+1*). Custom `ExpandingYearCV` enforces no future leakage.

**Primary metric:** AUPRC (area under precision-recall curve). Class imbalance is extreme — civil_war_onset ~0.2% base rate. AUROC will look deceptively high; always evaluate AUPRC and Brier score together.

---

## Sprint 1 — Infrastructure + Data Pull + Feature Engineering
**Available: ~40–48 hours | Estimated active work: 26–36 hours**

### Phase 1a — Infrastructure (Days 1–3) | 8–12 hours

1. Provision AML workspace, ADLS Gen2 container (`data`), Key Vault, compute instance (Standard_DS4_v2 recommended — 8 vCPU / 28 GB), and CPU cluster (5 × Standard_DS3_v2, `min_instances=0` for scale-to-zero). **Start this day 1 — IT/Azure permissions are the most common blocker.**

2. Install `requirements.txt` onto compute instance; configure the 6 required environment variables in AML Studio → Compute → Edit → Environment variables:

   | Variable | Value |
   |---|---|
   | `ADLS_ACCOUNT_NAME` | Your ADLS Gen2 storage account name |
   | `ADLS_CONTAINER` | `data` |
   | `KEY_VAULT_URL` | `https://<vault-name>.vault.azure.net` |
   | `AZURE_SUBSCRIPTION_ID` | Output of `az account show --query id -o tsv` |
   | `AZURE_RESOURCE_GROUP` | Resource group containing the AML workspace |
   | `AZUREML_WORKSPACE_NAME` | AML workspace name |

3. Store Key Vault secrets:

   | Secret name | Source |
   |---|---|
   | `acled-api-key` | Register at acleddata.com (same-day) |
   | `acled-email` | Same registration |
   | `gcp-service-account-json` | GCP Console → IAM → Service Accounts → create key → JSON |
   | `gcp-project-id` | GCP Console → project settings |

   ACLED registration is same-day. GCP service account creation takes ~30 minutes. Skip GCP secrets if GDELT data is not needed — notebook `01/06` will error at the KV cell but all other notebooks are unaffected.

4. Run `setup/onboarding.ipynb` top-to-bottom — validates all environment variables, Key Vault access, ADLS write permissions, and uploads `data/country_crosswalk.csv`. Fix any issues here before the data pull phase.

5. **Initiate CNTS commercial data procurement immediately.** The Cross-National Time Series dataset (notebook `01/12`) requires a CQ Press / Databanks International subscription. If procurement takes weeks, notebook `01/12` skips gracefully — it is not required for the four core outcomes.

### Phase 1b — Data Pull (Days 3–7) | 10–14 hours

6. Run all 19 `01_data_pull/` notebooks. Each executes in 2–10 minutes; the bottleneck is data quality debugging, not execution time. Run in credential-sorted batches:

   **Batch 1 — Public APIs, no credentials required (run first):**
   `02` World Bank, `03` V-Dem/Polity, `05` FSI/UNHCR, `07` FAO, `08` SIPRI, `10` Archigos, `13` NELDA, `14` RSUI, `15` Freedom House, `16` EPR, `17` PTS, `18` V-Dem ERT, `19` Cline Coup

   **Batch 2 — ACLED credentials required:**
   `01` ACLED, `04` UCDP-GED conflict labels, `11` Africa leadership change

   **Batch 3 — GCP BigQuery credentials required:**
   `06` GDELT

   **Batch 4 — Manual upload required:**
   `12` CNTS (run after commercial data arrives; skip on first pass)

7. After completing Batch 1, inspect the first 3–4 parquet outputs for schema correctness: column dtypes, null rates, country coverage. Expect 2–3 notebooks to need minor debugging on first run (date parsing, API pagination limits, rate limiting).

8. Verify raw outputs exist in ADLS: `raw/<source>/{RUN_DATE}/` for each completed source.

### Phase 1c — Feature Engineering (Days 8–10) | 8–10 hours

9. Run `02/01_prio_grid_spatial_features` — constructs spatial weights matrix (libpysal), computes Moran's I, and derives 15 spatial feature families from PRIO-GRID 30-arc-minute grid cells. Slowest notebook in this phase (~5–15 min). Review derived feature distributions before proceeding.

10. Run `02/02_build_feature_matrix` — joins all 19 sources on `(iso3, year)` key, codes 12 binary outcome labels (forward-shifted +1 year), applies temporal train/val/test flags. Watch for: country code mismatches (use `country_crosswalk.csv` for resolution), temporal alignment gaps across sources, unexpected null rates from sources with shorter historical coverage.

11. Run `02/03_engineer_derived_features` — adds log1p/sqrt transforms, HP-filter residuals, pairwise interactions, spatial spillover lags (KNN lag, LISA). Toggle sections A (transforms), B (interactions), C (spillover) independently. Start with all enabled; note which produce large null columns for downstream investigation.

---

### Sprint 1 Deliverables

| # | Output | Location | Description |
|---|---|---|---|
| 1 | 19 raw source parquets | `raw/<source>/{RUN_DATE}/` | All public data sources pulled (CNTS pending procurement) |
| 2 | `priogrid_engineered_features.parquet` | `raw/prio_grid/{RUN_DATE}/` | 15 spatial feature families for ~167 countries |
| 3 | `feature_matrix.parquet` | `processed/feature_matrix/{RUN_DATE}/` | ~4,000 rows × ~120–180 features; 12 outcome labels coded |
| 4 | `feature_matrix_engineered.parquet` | `processed/feature_matrix_engineered/{RUN_DATE}/` | Log/sqrt transforms, HP-filter residuals, spatial spillover lags |
| 5 | AML infrastructure live | Azure Portal | Compute instance, CPU cluster (scale-to-zero), KV, ADLS all validated |

---

## Sprint 2 — Feature Selection + Training + HyperDrive Round 1
**Available: ~40–48 hours | Estimated active work: 20–28 hours**

### Phase 2a — Feature Selection (Days 1–2) | 4–6 hours

12. Run `03/01_feature_selection_lasso` — fits `LogisticRegressionCV(penalty='l1', solver='saga', n_cs=30)` × 5 folds × 5 outcomes = 750 total fits. Runs in ~8–15 minutes on Standard_DS4_v2. Outputs one `selected_{outcome}.json` per outcome.

13. Review the audit outputs carefully:
    - Which features were selected by LASSO vs. rescued by mutual information? MI rescues indicate genuinely non-linear predictors that L1 regularisation would otherwise discard.
    - Flag any outcome where fewer than 10 features survive — this signals data sparsity or multicollinearity requiring investigation before training.
    - Expected selection counts: 15–40 features per outcome from an initial 80–150 candidate pool.

14. Run `03/03_interrogate_nonlinearity` against the LASSO audit results — PDP/ICE plots and SHAP interaction values confirm LASSO did not discard non-linear predictors. If the linearity audit flags surprises, adjust the MI rescue threshold in `03/01` and re-run before proceeding.

### Phase 2b — Baseline XGBoost Training (Days 3–5) | 6–8 hours

15. Run `03/02_train_xgboost_outcomes` interactively on the compute instance — `RandomizedSearchCV(n_iter=40)` per outcome on the LASSO-selected features. Total execution: ~20–40 minutes for all 5 outcomes.

16. Inspect MLflow results in AML Studio → Experiments. Key metrics to evaluate:

    | Outcome | Realistic AUPRC range | Notes |
    |---|---|---|
    | `mass_unrest_onset` | 0.45–0.65 | ~15% base rate — most tractable |
    | `regime_backsliding` | 0.35–0.55 | V-Dem indices are strong predictors |
    | `coup_attempt` | 0.20–0.40 | ~0.2% base rate; use Brier score alongside AUPRC |
    | `civil_war_onset` | 0.15–0.35 | ~0.2% base rate; AUROC misleadingly high |
    | `humanitarian_crisis_onset` | 0.30–0.50 | ~39 FEWS countries; smaller N |

17. Identify the two lowest-AUPRC outcomes — prioritise these in HyperDrive sweeps with 40 trials each.

### Phase 2c — HyperDrive Sweep Round 1 (Days 6–9) | 6–8 hours active; ~1.5–2.5 hours wall-clock per submission

18. Submit sweep jobs sequentially, one outcome at a time. Cluster auto-scales on first job submission:

    ```bash
    az ml job create -f jobs/sweep_outcome.yml \
      --set inputs.outcome=civil_war_onset \
      --set inputs.adls_account_name=$ADLS_ACCOUNT \
      --name sweep-civil-war-r1-$(date +%Y%m%d)
    ```

    While each sweep runs (~20–30 min wall-clock, 5 concurrent trials), submit the next outcome's job. Total wall-clock for all 5 outcomes if staggered: ~2–3 hours.

    Sweep configuration (from `jobs/sweep_outcome.yml`):
    - 40 trials per outcome (20 for `humanitarian_crisis_onset`)
    - 5 concurrent trials on Standard_DS3_v2 cluster
    - Bandit early stopping: `slack_factor=0.15`, evaluation begins at trial 5
    - Expected early stop rate: 25–35% of trials

19. Compare sweep best vs. notebook baseline — expect 5–15% AUPRC improvement from HyperDrive search. Note which hyperparameters the sweep converged on; `max_depth`, `learning_rate`, and `subsample` are typically the most sensitive.

### Phase 2d — Sweep Round 1 Interpretation (Days 9–10) | 4–6 hours

20. Review bandit early termination patterns:
    - If >40% of trials were terminated early: bandit is too aggressive — lower `slack_factor` to 0.10 for Round 2.
    - If <15% were terminated: search space is too narrow — widen `learning_rate` or `max_depth` bounds for Round 2.

21. Check for overfitting: compare `val_auprc` vs. `test_auprc`. A gap > 0.10 signals overfitting — increase `reg_alpha`, `reg_lambda`, or `min_child_weight` in the Round 2 search space.

22. Update `jobs/sweep_outcome.yml` search space bounds based on findings before Sprint 3.

---

### Sprint 2 Deliverables

| # | Output | Location | Description |
|---|---|---|---|
| 1 | `selected_{outcome}.json` × 5 | `feature_selection/{RUN_DATE}/` | LASSO + MI rescue selected features per outcome |
| 2 | Baseline XGBoost models × 5 | MLflow model registry | First trained models; AUPRC baseline per outcome established |
| 3 | HyperDrive Round 1 results | MLflow experiment | Best hyperparams + AUPRC per outcome; sweep convergence patterns |
| 4 | Refined `sweep_outcome.yml` | Repo | Search space tightened for Round 2 |
| 5 | Calibration + beeswarm plots × 5 | MLflow artifacts | Initial SHAP global importance per outcome |

---

## Sprint 3 — Sweep Round 2 + Inference + Documentation
**Available: ~40–48 hours | Estimated active work: 18–26 hours**

### Phase 3a — HyperDrive Sweep Round 2 (Days 1–4) | 6–8 hours active

23. Submit Round 2 sweeps with refined search space. Prioritise the two hardest outcomes at 40 trials each; re-run easier outcomes at 20 trials to conserve compute budget.

24. Compare Round 1 vs. Round 2 AUPRC per outcome. Expect diminishing returns (2–5% further improvement). If Round 2 matches Round 1 within noise, the model has converged — additional sweeps will not materially improve results without new data or features.

25. Register the final best model per outcome in the MLflow model registry with tags:

    ```
    outcome: civil_war_onset
    val_auprc: 0.XX
    test_auprc: 0.XX
    feature_count: N
    best_iteration: N
    sweep_round: 2
    ```

### Phase 3b — Inference + SHAP (Days 5–7) | 4–6 hours

26. Run `04/01_xgboost_inference` with the best model run ID from Round 2 for each outcome. Outputs per country-year in the test set (2022+):
    - `waterfall.png` — top drivers pushing risk above/below baseline
    - `force.png` — individual observation force plot
    - `feature_contributions.csv` — numerical SHAP values
    - `prediction_summary.json` — predicted probability, top-3 risk factors, model metadata

27. From inference output, identify the top-5 highest-risk country-years per outcome. For each, examine the waterfall plot to identify the 3–5 dominant risk drivers. This is the primary analytical deliverable from the pipeline.

28. Spot-check SHAP global importance against domain expectations:
    - V-Dem liberal democracy index → `regime_backsliding`
    - ACLED protest event count → `mass_unrest_onset`
    - Archigos leadership tenure / irregular exit → `coup_attempt`
    - Conflict battle deaths lag → `civil_war_onset`
    - FEWS IPC phase / UNHCR refugee flows → `humanitarian_crisis_onset`

    Unexpected top features are worth investigating before reporting — they may indicate data leakage, collinearity, or genuinely surprising findings.

### Phase 3c — Documentation + Reporting (Days 8–10) | 6–8 hours

29. Produce model summary covering:
    - AUPRC by outcome and sweep round (baseline → R1 → R2 improvement curve)
    - Calibration assessment (Brier score, reliability diagram)
    - Top 10 features per outcome by mean |SHAP|
    - Highest-risk countries for each outcome in 2022–2023
    - Known limitations (CNTS absence if not acquired, GDELT coverage gaps, temporal panel end)

30. Update MLflow experiment notes and model registry descriptions for reproducibility.

---

### Sprint 3 Deliverables

| # | Output | Location | Description |
|---|---|---|---|
| 1 | Final models × 5 | MLflow model registry (tagged) | Best hyperparams, Round 2 sweep winners, registered with full metadata |
| 2 | Per-country SHAP artifacts | `inference_outputs/{country_year}/` | Waterfall + force plots; `prediction_summary.json` per country-year |
| 3 | Model summary report | `.claude/model-results-summary.md` | AUPRC ranges, top features, highest-risk country-years, limitations |
| 4 | Documented MLflow experiment | AML Studio → Experiments | All runs tagged, annotated, best model per outcome registered |

---

## Man-Hours Summary

| Phase | Task | Hours |
|---|---|---|
| **Sprint 1** | | |
| 1a | Infrastructure provisioning and validation | 8–12 |
| 1b | Data pull — 19 notebooks + debugging | 10–14 |
| 1c | Feature engineering — spatial, matrix build, derived features | 8–10 |
| **Sprint 1 subtotal** | | **26–36 h (midpoint ~31 h)** |
| **Sprint 2** | | |
| 2a | Feature selection (LASSO + MI rescue, 5 outcomes) | 4–6 |
| 2b | Baseline XGBoost training + MLflow review | 6–8 |
| 2c | HyperDrive Round 1 submission and monitoring | 6–8 |
| 2d | Round 1 interpretation + search space refinement | 4–6 |
| **Sprint 2 subtotal** | | **20–28 h (midpoint ~24 h)** |
| **Sprint 3** | | |
| 3a | HyperDrive Round 2 submission and comparison | 6–8 |
| 3b | Inference + SHAP artifact generation | 4–6 |
| 3c | Documentation and model registry | 6–8 |
| **Sprint 3 subtotal** | | **16–22 h (midpoint ~19 h)** |
| **Total** | | **62–86 h (midpoint ~74 h)** |

At 50–60% capacity across 3 × 2-week sprints (~120–144 available hours total), the midpoint estimate of ~74 hours leaves **46–70 hours of buffer** — roughly 40–50% of available capacity. The pipeline will be fully operational well inside the three-sprint window.

**Highest schedule-risk items:** IT/Azure permissions (Phase 1a), CNTS commercial procurement, and ACLED/GCP credential setup. Start all procurement and access requests on Day 1.

---

## What Remains After Sprint 3

At end of Sprint 3 there will be five trained, evaluated, and documented instability classifiers with SHAP explainability and full MLflow tracking. The following items will **not** be complete:

| Remaining Item | Estimated Effort | Notes |
|---|---|---|
| CNTS data integration | 0.5 sprint | Notebooks already written; blocked on commercial data procurement |
| Production serving endpoint (AML online endpoint) | 0.5 sprint | Requires `score.py` + endpoint YAML; not in current codebase |
| Dashboard / reporting UI (`requirements-web.txt` exists, nothing built) | 0.5–1 sprint | Frontend for non-technical consumers |
| Automated retraining pipeline (AML Pipeline with schedule trigger) | 0.5 sprint | Currently all notebooks are manual; no pipeline YAML yet written |
| Additional outcome models beyond the 5 core | Variable | 12 outcomes coded in `02_build_feature_matrix`; 7 are not yet trained |
| Ensemble / stacking across outcomes | 0.5 sprint | Cross-outcome correlations could improve joint prediction |
| Integration with PEA pipeline outputs as real-time features | 1 sprint | PEA protest events as a feature feed is analytically valuable but not yet wired |
| Post-2022 realized-event validation (ACLED / GLOCON) | 0.5 sprint | Ground-truth check of highest-risk predictions against recorded events |

---

## Key Implementation Notes

1. **Temporal integrity is critical.** Train ≤ 2018, Val 2019–2021, Test ≥ 2022. Never let test-period data influence feature engineering or feature selection. The `ExpandingYearCV` class in `src/train/train_outcome.py` enforces this during HyperDrive sweeps; interactive notebook training must enforce it manually.

2. **Use AUPRC, not AUROC**, for all primary evaluation. AUROC is misleadingly high for events with 0.2% base rate (e.g., civil_war_onset). Brier score monitors calibration quality alongside discrimination.

3. **Class imbalance handling.** `scale_pos_weight = n_negatives / n_positives` is calculated per outcome in `train_outcome.py` — verify this is applied correctly in interactive notebook training in `03/02`.

4. **Feature selection is per-outcome.** LASSO + MI rescue runs independently for each of the five outcomes. There is no shared feature set; each model trains on its own selected feature subset.

5. **ADLS path convention.**
   - Raw sources: `raw/<source>/{RUN_DATE}/`
   - Processed: `processed/feature_matrix/{RUN_DATE}/`
   - Models: `models/{RUN_DATE}/{outcome}/`
   - Inference: `inference_outputs/{country_year}/`

6. **Compute costs.** See `azure-ml-usage-plan.md` for full cost breakdown. HyperDrive sweeps are the primary compute cost driver (~$30–40 per full round across all 5 outcomes on Standard_DS3_v2). The compute instance should be **stopped when not in use** — no idle charges if stopped.

7. **GDELT is optional.** If GCP credentials are unavailable, skip notebook `01/06`. The feature matrix will be missing GDELT media tone and protest coverage features, but all four non-GDELT-dependent outcomes train normally.
