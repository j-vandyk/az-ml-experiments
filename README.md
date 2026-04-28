# az-ml-experiments

Predicts national-level political and humanitarian instability events from a
country-year panel dataset. The pipeline ingests data from nineteen international
sources, builds a unified feature matrix augmented with PRIO-GRID spatial features
and derived transformations, selects features via LASSO with a mutual-information
rescue pass, trains one XGBoost classifier per outcome, and produces full
SHAP-based explainability at both the model level and — during inference — at the
level of individual country-year observations.

All compute runs on Azure Machine Learning. Experiment tracking, model
registration, and artifact storage use MLflow integrated with the AML workspace.
Raw and processed data live in Azure Data Lake Storage Gen2.

---

## Pipeline overview

```
 01_data_pull/
 Notebooks 01–19
 19 sources → ADLS parquets
        │
        ├── 01 ACLED          09 PRIO-GRID raw        14 RSUI
        ├── 02 World Bank     10 Archigos              15 Freedom House
        ├── 03 V-Dem/Polity   11 Africa leadership     16 EPR
        ├── 04 Conflict lbls  12 CNTS                  17 PTS
        ├── 05 Fragility/Hum  13 NELDA                 18 V-Dem ERT
        ├── 06 GDELT                                   19 Cline coup
        ├── 07 FAO food
        └── 08 SIPRI
               │
               ▼
 02_feature_engineering/
   01_prio_grid_spatial_features  ← eight spatial feature families from PRIO-GRID cells
   02_build_feature_matrix        ← join all 19 sources + spatial features; code labels
   03_engineer_derived_features   ← transforms · interactions · spatial spillover
               │
               ▼
 03_model_development/
   01_feature_selection_lasso     ← LASSO screening + mutual-information rescue
   02_train_xgboost_outcomes      ← one XGBoost classifier per outcome + SHAP
   03_interrogate_nonlinearity    ← PDP/ICE · SHAP interactions · linearity audit
               │
               ▼
 04_inference/
   01_xgboost_inference           ← per-observation SHAP artifacts on new data
```

---

## Notebook stages

### Stage 01 — Data pull (`01_data_pull/`)

Each notebook pulls one source, writes raw parquet(s) to ADLS under
`raw/<source>/{RUN_DATE}/`, and produces a cleaned country-year aggregate
ready for joining in `02/02_build_feature_matrix`.

| # | Source | Key variables |
|---|---|---|
| 01 | ACLED | Conflict event counts and fatalities by type (country-month → country-year) |
| 02 | World Bank WDI + WGI | GDP per capita, growth, inflation, trade, debt, governance indicators |
| 03 | V-Dem + Polity5 | Democracy indices, electoral quality, civil liberties, Polity score |
| 04 | Conflict labels | UCDP-GED (civil war), Powell-Thyne (coups), PITF (state failure) |
| 05 | Fragility & Humanitarian | FSI sub-scores, UNHCR refugee flows, UNDP HDI |
| 06 | GDELT | Media-derived event tone, protest coverage, cross-national tension indices |
| 07 | FAO food prices | Domestic food price volatility and deviation from trend |
| 08 | SIPRI | Military expenditure as share of GDP and government budget |
| 09 | PRIO-GRID | Geographic/environmental risk factors aggregated to country-year |
| 10 | Archigos | Leadership tenure, irregular exit history, succession type |
| 11 | Africa leadership change | Continent-specific leadership event coding for African states |
| 12 | CNTS | Cross-National Time-Series political instability indicators |
| 13 | NELDA | Election timing, competitiveness, and irregularity scores |
| 14 | RSUI | IMF Reported Social Unrest Index — monthly, 130 countries, 1996–present |
| 15 | Freedom House | Freedom in the World annual scores and status transitions |
| 16 | EPR-Core | Ethnic Power Relations — ethnic group exclusion and status at country-year |
| 17 | PTS | Political Terror Scale — government repression coded 1–5, three sources |
| 18 | V-Dem ERT | Episodes of Regime Transformation — autocratization episode coding |
| 19 | Cline Center Coup | Global coup registry 1945–present; extends and supersedes Powell-Thyne |

Notebooks 14–19 provide supplementary label sources (additional dependent variables
for backsliding, repression, ethnic exclusion, and coup models) as well as predictors.
All write to ADLS; `02/02` joins them via `RAW_PREFIXES` using the same mechanism as
the core sources.

---

### Stage 02 — Feature engineering (`02_feature_engineering/`)

#### `01_prio_grid_spatial_features`

Depends on `01/09` (PRIO-GRID pull) and `01/04` (UCDP-GED events). Builds eight
feature families from grid-cell-level spatial data that capture *within-country
variation* — distributional shape, spatial concentration, and temporal shocks that
flat country-level averages erase.

| Method | Feature prefix | Instability mechanism |
|---|---|---|
| 1 — Distributional aggregates | `prio_ntl_*`, `prio_pop_*` | Dispersion of NTL/pop captures urban–rural fractures |
| 2 — NTL Gini / darkness fraction | `prio_ntl_gini`, `prio_ntl_darkness_frac` | Within-country economic inequality proxy |
| 3 — Pop-weighted climate stress | `prio_pop_wtd_rainfall_sd`, `prio_high_stress_pop_frac` | Which populations face rainfall/temp volatility |
| 4 — Conflict cell density | `prio_conflict_cell_*` | Spatial breadth of active violence (UCDP-GED join) |
| 5 — Lootable resource exposure | `prio_petroleum_*`, `prio_drug_*` | Grievance potential of resource enclaves |
| 6 — Core/periphery polarization | `prio_ntl_core_*`, `prio_ntl_cv` | Primate city concentration, regional neglect |
| 7 — NTL economic shock | `prio_ntl_yoy_change`, `prio_ntl_shock_zscore` | Sudden economic deterioration signal |
| 8 — GeoEPR ethnic territory | `prio_geoepr_ntl_excl_gap`, `prio_geoepr_excl_pop_share` | Economic gap inside excluded group homelands (EPR-Core × GeoEPR) |

Writes to `raw/prio_grid/{RUN_DATE}/priogrid_engineered_features.parquet`.

#### `02_build_feature_matrix`

Joins all 19 source parquets and the PRIO-GRID spatial features (`02/01`) on a
common `(iso3, year)` key to produce a single country-year panel
(~167 countries × 25 years ≈ 4,000 rows). Codes twelve binary outcome labels,
each forward-shifted one year so that features at year *t* predict events at year *t+1*.

**Core outcomes** (sources 01–13)

| Label | Source | Coding |
|---|---|---|
| `civil_war_onset` | UCDP-GED | First year of state-based conflict after ≥2-year peace spell |
| `coup_attempt` | Powell-Thyne | Any coup attempt in the year (success or failure) |
| `regime_backsliding` | V-Dem | `v2x_libdem` drops ≥0.05, or transition to closed autocracy |
| `mass_unrest_onset` | ACLED | Annual protest+riot events exceed country 90th-percentile |
| `humanitarian_crisis_onset` | FEWS NET | IPC phase ≤3 → ≥4 (~39 FEWS countries; NA elsewhere) |

**Supplementary outcomes** (sources 14–19)

| Label | Source notebook | Coding |
|---|---|---|
| `unrest_binary` | RSUI (14) | Annual max RSUI score exceeds country-specific 75th-percentile |
| `fh_status_decline` | Freedom House (15) | FH status category worsened vs. prior year |
| `ethnic_exclusion_any` | EPR (16) | Any large ethnic group (>10%) coded Powerless or Discriminated |
| `epr_state_collapse` | EPR (16) | EPR codes the country as State Collapse this year |
| `pts_high` | PTS (17) | Government repression score ≥ 4 across any source |
| `pts_escalation` | PTS (17) | PTS max score increased ≥1 point year-on-year |
| `aut_ep` | V-Dem ERT (18) | Country is in an active autocratization episode |

Adding a new source requires adding its prefix to `RAW_PREFIXES` and extending `OUTCOME_COLS`.

#### `03_engineer_derived_features`

Reads the assembled feature matrix, adds derived columns, and writes an augmented
parquet consumed by `03/01` in place of the raw matrix. Each section is independently
toggled in `ENG_CFG`.

| Section | What it adds |
|---|---|
| A — Transformations | `log1p`, `sqrt`, first differences (`_diff1`), HP-filter trend/cycle |
| B — Interactions | Configured pairwise products encoding domain hypotheses |
| C — Spatial spillover | KNN spatial lag (`_slag`), Local Moran's I, LISA quadrant |

A **feature catalog CSV** is written alongside the output parquet documenting every
derived column, its source variables, transformation applied, and missingness rate.

---

### Stage 03 — Model development (`03_model_development/`)

#### `01_feature_selection_lasso`

Screens features in two passes before any XGBoost model sees them, producing a
separate selected feature set per outcome.

**Pass 1 — LASSO screening.** `LogisticRegressionCV(penalty='l1', solver='saga')`
with five expanding temporal folds (2000–2018). λ chosen by the 1-SE rule.

**Pass 2 — Mutual information rescue.** `mutual_info_classif` detects features with
U-shaped, threshold, or interaction-only effects silenced by LASSO. Features in the
MI top-50 with a zero LASSO coefficient and a large MI-vs-LASSO rank gap are rescued.

Outputs per outcome: `feature_selection/{RUN_DATE}/selected_{outcome}.json` and an
audit parquet consumed by `03/03`.

#### `02_train_xgboost_outcomes`

Trains one `XGBClassifier` per outcome using the feature set from `03/01`.

- Temporal train/val/test split (train ≤ 2018, val 2019–2021, test ≥ 2022)
- `RandomizedSearchCV` with expanding temporal CV folds — no cross-time leakage
- Class imbalance handled per-outcome via `scale_pos_weight`
- Early stopping on a held-out 10% of train+val
- All models, metrics, and SHAP plots logged to MLflow (one run per outcome)
- Global SHAP: beeswarm, bar chart, waterfall (high/low risk), dependence plots, heatmap

A matching AML sweep job is defined in `jobs/sweep_outcome.yml` for running
HyperDrive hyperparameter search at scale. The standalone training script at
`src/train/train_outcome.py` is the trial command called by each sweep trial.

#### `03_interrogate_nonlinearity`

Validates that feature selection (`03/01`) did not discard non-linear signals and
characterises the non-linearity of features the trained models actually use.

1. Linearity trap audit — LASSO vs. MI rank divergence per feature and outcome
2. Partial Dependence Plots + Individual Conditional Expectation for top LASSO features
3. PDP for MI-rescued features — confirms non-linear pattern is real
4. SHAP interaction values — joint effects LASSO cannot detect (top-20 features)
5. Monotonicity check — SHAP sign vs. domain expectation for key features
6. Direction concordance — LASSO coefficient sign vs. SHAP mean sign

---

### Stage 04 — Inference (`04_inference/`)

#### `01_xgboost_inference`

Loads a trained XGBoost model from MLflow (by run ID or registered model URI),
applies it to new country-year observations, and produces a self-contained SHAP
artifact directory for every observation.

```
inference_outputs/
  BWA_2025/
    waterfall.png              ← primary explainability plot
    force.png                  ← horizontal push/pull diagram
    feature_contributions.csv  ← all features with raw value and SHAP value
    prediction_summary.json    ← top risk drivers and stabilisers
  SOM_2025/
    ...
  inference_results.csv        ← all countries ranked by P(instability)
  inference_results.json       ← same with full driver detail
```

The notebook reads the model's logged signature to get the exact feature list used
at training time and filters to `inference_years` (earlier years supply lag/rolling
features only). Model artifacts are produced by `03/02_train_xgboost_outcomes`.

---

### Exploratory (`00_exploratory/`)

#### `01_xgboost_prototype`

> **Not part of the main pipeline.**

A self-contained notebook that trains a single binary XGBoost classifier directly
from raw CSV/JSON without depending on the numbered pipeline. Useful for rapid
experimentation with new data slices or modelling assumptions before integrating
into the full workflow. Uses inline feature engineering and `StratifiedKFold`
(not temporal CV — treat performance metrics as optimistic). MLflow artifacts
produced here are loadable by `04/01_xgboost_inference`.

---

## Two-stage modelling design

The pipeline separates *which features to use* from *how to use them*.

**Stage 1 — Feature selection (`03/01`):** LASSO with temporal CV produces a sparse,
outcome-specific feature list. The MI rescue pass adds back non-linear predictors
silenced by L1 regularisation. The result is a compact, theoretically motivated
feature set per outcome — not a single shared set applied uniformly to all models.

**Stage 2 — Training (`03/02`):** Each outcome model is trained only on its stage-1
feature set. Narrower, curated inputs concentrate signal, make SHAP attributions
interpretable in terms of substantively meaningful variables, and speed up
hyperparameter search. Feature selection can be re-run as new data years arrive
without retraining all models, and vice versa.

---

## Infrastructure

| Component | Technology |
|---|---|
| Compute | Azure Machine Learning compute clusters |
| Data storage | Azure Data Lake Storage Gen2 (raw and processed parquet) |
| Experiment tracking | MLflow integrated with AML workspace |
| Model registry | MLflow Model Registry via AML |
| Hyperparameter sweep | AML HyperDrive (`jobs/sweep_outcome.yml`) |
| Sweep trial script | `src/train/train_outcome.py` |
| Authentication | Managed Identity (AML compute) / `DefaultAzureCredential` (local) |

---

## Setup

```bash
pip install -r requirements.txt
```

Set the following environment variables before running any notebook locally:

```bash
ADLS_ACCOUNT_NAME   # Storage account name
ADLS_CONTAINER      # Container name (default: data)
ADLS_SAS_TOKEN      # or ADLS_ACCOUNT_KEY
```

On AML compute, authentication is handled automatically via Managed Identity.

### ADLS path conventions

| Stage | Path pattern |
|---|---|
| Raw source data | `raw/<source>/{RUN_DATE}/<source>_<frequency>.parquet` |
| PRIO-GRID spatial features | `raw/prio_grid/{RUN_DATE}/priogrid_engineered_features.parquet` |
| Feature matrix (raw) | `processed/feature_matrix/{RUN_DATE}/feature_matrix.parquet` |
| Feature matrix (engineered) | `processed/feature_matrix_engineered/{RUN_DATE}/feature_matrix.parquet` |
| Feature selection manifests | `feature_selection/{RUN_DATE}/selected_{outcome}.json` |
| Trained models | `models/{RUN_DATE}/{outcome}/model.json` |
| SHAP values | `models/{RUN_DATE}/{outcome}/shap_values.parquet` |
| Inference outputs | `inference_outputs/{RUN_DATE}/` |
