# Project Reference — Instability Classifiers

State-of-play reference for the country-year instability-forecasting pipeline. Consolidated from:
- `archive/azure-ml-usage-plan.md` — VM sizing, HyperDrive estimates, budget
- `archive/meta-plan-instability-classifiers.md` — original 4–6 month project plan (most data/architecture sections now live here, in `data-and-predictors.md`, or in the notebooks themselves)

For dataset/predictor/label material see `data-and-predictors.md`. For metric reading and threshold selection see `metric-interpretation-guide.md`.

---

## 1. Pipeline overview

Four stages, each with its own notebook directory. Code that runs in production (HyperDrive sweeps, train script) lives under `src/train/`.

| Stage | Notebooks | Compute | Output |
|---|---|---|---|
| 1. Data pull | `01_data_pull/01–26_*.ipynb` | Compute instance | Date-partitioned parquets at `raw/<source>/<YYYYMMDD>/` |
| 2. Feature engineering | `02_feature_engineering/01_panel_spine.ipynb`, `02_build_feature_matrix.ipynb`, `03_engineer_derived_features.ipynb` | Compute instance | `processed/feature_matrix_engineered/<YYYYMMDD>/feature_matrix_engineered.parquet` + `feature_catalog_additions.csv` |
| 3. Model development | `03_model_development/01_lasso_feature_selection.ipynb`, `02_train_xgboost_outcomes.ipynb`; `src/train/train_outcome.py` + `src/train/sweep_outcome.yml` | Instance for LASSO; CPU cluster for HyperDrive | Per-outcome XGBoost models + SHAP artifacts logged to MLflow |
| 4. Inference | `04_inference/01_score_country_year.ipynb` | Compute instance | Scored country-year predictions + SHAP narratives |

**Boundary contract:** stage 3 reads `FEATURE_MATRIX_PREFIX` from the engineered output of stage 2, not the raw matrix. Update this prefix whenever stage 2 is rerun.

---

## 2. Architecture decisions

The design is fixed for the current iteration. Revisit only if performance plateaus.

| Decision | Choice | Rationale |
|---|---|---|
| Outcome framing | One binary classifier per outcome (12 outcomes total) | Enables per-outcome metric, threshold, and SHAP interpretation; avoids the multilabel coupling problem |
| Algorithm | XGBoost (`tree_method="hist"`) | Strong on small tabular country-year data; no GPU needed at this scale |
| Feature selection | LASSO (`LogisticRegressionCV`, saga, n_cs=30, 5-fold) per outcome before XGBoost | Reduces 200+ engineered columns to ~80 per outcome before sweep |
| Hyperparameter tuning | HyperDrive Bandit (slack 0.15, eval starts trial 5) on CPU cluster | Allows early stopping of bad trials; MLflow tracks all runs |
| Temporal split | Train ≤2018 / Val ≤2021 / Test 2022–2024 | No shuffling; respects temporal causality; matches deployment scenario |
| Imbalance handling | `scale_pos_weight` per outcome (computed from training base rate) | Better calibration than oversampling for tree models |
| Interpretation | SHAP per outcome on test set; narrative generation downstream | Per-prediction explanations meet stakeholder requirements |
| Forecast horizon | 1 year ahead (`LABEL_HORIZON=1`) | Matches available evaluation period and stakeholder use case |
| Geography | Africa, ~54 countries, 2000–2024 | Initial scope; design generalises to other regions if outcomes match |

**Three things explicitly NOT in scope** (live in `archive/meta-plan-instability-classifiers.md`):
- LSTM / sequence-model baselines
- LLM-generated narrative reports
- Stacking ensemble across thematic sub-models

These were Phase IV extensions in the original plan; revisit only if the per-outcome XGBoost approach hits a ceiling.

---

## 3. Compute resources

### 3.1 Resource map — what runs where

| Workload | Resource | Why |
|---|---|---|
| Data pull (`01_*`) | Compute instance | Network-bound, light pandas |
| Feature engineering (`02_*`) | Compute instance | Spatial weights (libpysal), HP-filter; single-node, memory-moderate |
| LASSO selection (`03/01`) | Compute instance | ~750 fits across all outcomes; fast on ~1,100 rows |
| XGBoost sweep (`sweep_outcome.yml`) | CPU cluster (auto-scale) | HyperDrive parallelises 5 concurrent trials |
| Inference (`04/01`) | Compute instance | Single-pass predict + SHAP |

**No GPU is warranted for current scope.** XGBoost `tree_method="hist"` does not benefit from GPU on a ~1,200-row dataset. Reconsider only if neural-network baselines are added.

### 3.2 Compute instance (interactive)

Stop when not in use — no idle charges if stopped. Set an auto-shutdown schedule in AML Studio.

| Tier | VM | vCPU | RAM | When |
|---|---|---|---|---|
| Min | Standard_DS3_v2 | 4 | 14 GB | Light feature eng; most training on cluster |
| **Recommended** | **Standard_DS4_v2** | **8** | **28 GB** | **Comfortable for libpysal, LASSO CV** |
| Generous | Standard_DS5_v2 | 16 | 56 GB | Full pipeline interactively in parallel |

### 3.3 HyperDrive CPU cluster

Each trial = one `train_outcome.py` run (load parquet, fit, SHAP, write artifacts, log MLflow). Sweep YAML sets `max_concurrent_trials: 5`.

| Tier | VM per node | vCPU | RAM | Trade-off |
|---|---|---|---|---|
| Budget | Standard_DS2_v2 | 2 | 7 GB | ~30–60 sec longer per trial |
| **Recommended** | **Standard_DS3_v2** | **4** | **14 GB** | **All 4 cores used by `n_jobs=-1`** |
| Fast | Standard_F8s_v2 | 8 | 16 GB | ~30–45 sec faster; ~60% more per node |

**Cluster config:**
```yaml
min_instances: 0          # scale to zero — no idle charges
max_instances: 5          # matches max_concurrent_trials
idle_seconds_before_scaledown: 120
```

---

## 4. Runtime estimates

### LASSO selection (notebook `03/01`, instance)

5 outcomes × 30-CS path × 5-fold CV = 750 fits. **Wall clock on DS4_v2: 8–15 min** for all outcomes together.

### HyperDrive sweep (cluster, per outcome)

| Outcome | Max trials | Concurrent | Expected early stops | Wall clock |
|---|---|---|---|---|
| civil_war_onset | 40 | 5 | ~10–15 | 20–30 min |
| coup_attempt | 40 | 5 | ~10–15 | 20–30 min |
| regime_backsliding | 40 | 5 | ~8–12 | 20–30 min |
| mass_unrest_onset | 40 | 5 | ~5–10 | 25–35 min |
| humanitarian_crisis_onset | 20 | 5 | ~5–8 | 15–20 min |

**One full sweep across all 5 core outcomes:** ~1.5–2.5 hours wall clock; ~12–15 node-hours.

**Iteration plan over project lifetime:**
- Round 1 (wide search): 5 sweeps, 12–15 node-hr
- Round 2 (refined): 5 sweeps, 10–12 node-hr
- Round 3 (targeted): 5 sweeps, 8–10 node-hr
- Total: 15 sweeps, 30–37 node-hr

---

## 5. Budget

Approximate Azure PAYG, US East, Linux, 2025. Verify via the [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/). AML workspace, experiments, and registry are free — cost is VM + storage only.

### VM reference prices

| VM | $/hr (PAYG, US East) |
|---|---|
| Standard_DS2_v2 | ~$0.14 |
| Standard_DS3_v2 | ~$0.28 |
| Standard_DS4_v2 | ~$0.57 |
| Standard_DS5_v2 | ~$1.14 |
| Standard_F8s_v2 | ~$0.40 |

### All-in for the initial 3-round development cycle (~3 months)

Assumes 8 hr/day × 15 working days of instance time, plus 30–37 node-hr of cluster sweeps and ~25–55 GB ADLS storage.

| Scenario | Instance | Cluster | Storage | **Total** |
|---|---|---|---|---|
| Minimum (DS3 inst + DS2 cluster) | $34 | $5 | $4 | **~$43** |
| Recommended (DS4 inst + DS3 cluster) | $68 | $10 | $4 | **~$82** |
| Performance (DS5 inst + F8s cluster) | $137 | $12 | $4 | **~$153** |

### Caveats

- The dominant cost is human time, not compute.
- The main budget risk is a running compute instance left on. DS4 24/7 for a week ≈ $95.
- Reserved-instance pricing (~40% discount) is not worth committing to during build-out.

---

## 6. Open work (deferred follow-ups)

Tracked formally in `docs/refactor-backlog.md` (sibling branch `claude/refactor-backlog-followups`). Top three:

1. **Align ENG_CFG column references** with actual feature-matrix names (e.g. `gdp_per_capita` → `gdp_per_capita_const_usd`, `polity_score` → `polity`, `archigos_*` → `arch_*`). The validation block in `02/03` warns about mismatches but does not auto-rename.
2. **Consolidate ACLED/GDELT/ICEWS monthly→annual aggregation** into a shared helper. Currently each notebook re-implements the same monthly-roll-up boilerplate.
3. **Vectorise Section C spatial spillover** in `02/03`. Current per-row `.at` writes are O(n²); replace with `concat + merge` for a meaningful speedup on the full panel.

---

## 7. Quick command reference

```bash
# Run a sweep for one outcome
az ml job create -f src/train/sweep_outcome.yml \
    --set inputs.outcome=civil_war_onset

# Pull all sweep results
python -c "
import mlflow
runs = mlflow.search_runs(experiment_names=['instability_xgboost'])
print(runs[['tags.outcome', 'metrics.val_auprc', 'metrics.test_auprc']]
      .sort_values('metrics.val_auprc', ascending=False).head(20))
"

# After updating stage 2, point stage 3 at the new prefix
# Edit FEATURE_MATRIX_PREFIX in notebooks/03_model_development/01_lasso_feature_selection.ipynb:
#   processed/feature_matrix_engineered/<RUN_DATE>/feature_matrix_engineered.parquet
```
