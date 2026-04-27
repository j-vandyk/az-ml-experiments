# az-ml-experiments

Predicts national-level political and humanitarian instability events from a
country-year panel dataset. The pipeline ingests data from thirteen international
sources, builds a unified feature matrix, selects features via LASSO with a
mutual-information rescue pass, trains one XGBoost classifier per outcome, and
produces full SHAP-based explainability at both the model level and — during
inference — at the level of individual country-year observations.

All compute runs on Azure Machine Learning. Experiment tracking, model
registration, and artifact storage use MLflow integrated with the AML workspace.
Raw and processed data live in Azure Data Lake Storage Gen2.

---

## Pipeline overview

```
Notebooks 01–13          Notebook 14             Notebook 15
Data ingestion      →    Feature matrix    →    Feature selection
(13 sources)             country-year panel      LASSO + MI rescue
                         5 outcome labels        per outcome

        ↓
Notebook 16             Notebook 17
XGBoost training   →    Non-linearity audit
one model/outcome        PDP, ICE, SHAP interactions


        ↓                               ↓
xgboost_instability_classifier.ipynb    xgboost_inference.ipynb
Self-contained prototype notebook       Per-observation inference
(model dev + global SHAP)               with SHAP artifacts per country-year
```

---

## Notebooks

### Data ingestion — notebooks 01–13

Each notebook pulls one data source, writes raw parquet to ADLS, and produces a
cleaned country-year aggregate ready for joining in notebook 14.

| Notebook | Source | Key variables |
|---|---|---|
| 01 | ACLED | Conflict event counts and fatalities by type (country-month → country-year) |
| 02 | World Bank WDI | GDP per capita, growth, inflation, trade, debt, health, education indicators |
| 03 | V-Dem / Polity5 | Democracy indices, electoral quality, civil liberties, Polity score |
| 04 | UCDP-GED conflict labels | Civil war onset, conflict intensity, peace spell duration |
| 05 | FSI, UNHCR, UNDP | Fragile States Index sub-scores, refugee flows, Human Development Index |
| 06 | GDELT | Media-derived event tone, protest coverage, cross-national tension indices |
| 07 | FAO food prices | Domestic food price volatility and deviation from trend |
| 08 | SIPRI | Military expenditure as share of GDP and government budget |
| 09 | PRIO-Grid | Geographic and environmental risk factors aggregated to country level |
| 10 | Archigos | Leadership tenure, irregular exit history, succession type |
| 11 | Africa leadership change | Continent-specific leadership event coding for African states |
| 12 | CNTS | Cross-National Time-Series political instability indicators |
| 13 | NELDA | National election timing, competitiveness, and irregularity scores |

### Notebook 14 — Build feature matrix

Joins all cleaned source parquets on a common ISO3 country key to produce a single
country-year panel (~167 countries × 25 years). Codes five binary outcome labels,
each forward-shifted by one year so that features at year *t* predict events at
year *t+1*:

| Label | Source | Coding |
|---|---|---|
| `civil_war_onset` | UCDP-GED | First year of state-based conflict after a ≥2-year peace spell |
| `coup_attempt` | Powell-Thyne | Any coup attempt in the year (success or failure) |
| `humanitarian_collapse` | FSI / UNHCR | Composite threshold on FSI scores and displacement flows |
| `mass_protest` | ACLED / CNTS | Large-scale protest exceeding country-specific threshold |
| `regime_change` | Archigos / Polity | Irregular leadership exit or Polity score shift ≥3 points |

### Notebook 15 — LASSO feature selection

Screens features in two passes before any XGBoost model sees them, producing a
separate selected feature set per outcome.

**Pass 1 — LASSO screening.** `LogisticRegressionCV(penalty='l1', solver='saga')`
with five expanding temporal folds inside the training window (2000–2018). The
regularisation parameter λ is chosen by the 1-SE rule — the most regularised λ
within one standard error of the best AUPRC — to prefer sparser solutions.

**Pass 2 — Mutual information rescue.** LASSO cannot detect U-shaped, threshold,
or pure interaction effects and silently zeroes those features. `mutual_info_classif`
is run on all features; any feature in the MI top-50 that received a zero LASSO
coefficient and shows a large MI-vs-LASSO rank gap is rescued and added back to
the selected set.

Selected feature lists are written to ADLS and consumed by notebook 16.

### Notebook 16 — XGBoost training (multi-outcome)

Trains one `XGBClassifier` per outcome using the feature set produced by notebook 15
for that outcome. Models are independent — no stacking or cascading.

- Temporal train/val/test split (train ≤ 2018, val 2019–2021, test ≥ 2022) — no
  data from future periods leaks into training via `StratifiedKFold` shuffling
- `RandomizedSearchCV` over the full hyperparameter grid with temporal CV folds
- Early stopping on a held-out 10% of train+val to select the best iteration
- Class imbalance handled per-outcome via `scale_pos_weight`
- Every model, its metrics, and its SHAP plots are logged to a dedicated MLflow run

A matching AML sweep job is defined in `jobs/sweep_outcome.yml` for running
HyperDrive hyperparameter search at scale on AML compute clusters.

Global SHAP outputs per model: beeswarm, bar chart, waterfall (highest- and
lowest-risk samples), dependence plots for the top 3 features, and a heatmap.

### Notebook 17 — Interrogate non-linearity

Validates that the feature selection stage did not permanently discard non-linear
signals and characterises the non-linearity of the features the trained models
actually use.

1. Linearity trap audit table — LASSO vs. MI rank divergence per feature and outcome
2. Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) curves
   for top LASSO-selected features
3. PDP for MI-rescued features — confirms the non-linear pattern is real
4. SHAP interaction values — joint effects LASSO cannot detect (top-20 features)
5. Monotonicity check — SHAP sign vs. domain expectation for key features
6. Direction concordance table — LASSO coefficient sign vs. SHAP mean sign

---

## Self-contained classifier notebook

**`notebooks/xgboost_instability_classifier.ipynb`** is an end-to-end prototype
that trains a single binary instability classifier directly from raw CSV without
depending on the numbered pipeline. It is useful for rapid experimentation, testing
modelling assumptions, and exploring a new data slice before integrating it into
the full pipeline.

It covers the full cycle in one file: data loading from ADLS, temporal feature
engineering (lags, rolling windows, event history), train/val/test split,
hyperparameter search, final model training with early stopping, test-set
evaluation plots (ROC, PR curve, confusion matrix), and a complete set of global
SHAP explainability plots — all logged to MLflow.

Use this notebook to understand the model at the population level. It is not
intended for generating predictions on new data.

---

## Two-stage modeling process

The pipeline separates *which features to use* from *how to use them*, which
matters because raw feature sets are wide (200+ columns after lags and rolling
aggregates) and different outcomes are driven by different variable subsets.

**Stage 1 — Feature selection (notebook 15)**

LASSO with temporal cross-validation produces a sparse, outcome-specific feature
list. The mutual-information rescue pass then adds back non-linear predictors that
L1 regularisation silences. The result is a compact, theoretically motivated
feature set per outcome — not a single shared set applied uniformly to all models.

**Stage 2 — XGBoost training (notebook 16)**

Each outcome model is trained only on its stage-1 feature set. Using a narrower,
curated input space means the tree search concentrates signal rather than learning
to ignore noise, the SHAP attributions are interpretable in terms of substantively
meaningful variables, and hyperparameter search is faster.

The separation also means feature selection can be re-run (e.g. as new data years
arrive) without retraining all models, and vice versa.

---

## Inference notebook

**`notebooks/xgboost_inference.ipynb`** is the operational complement to the
training pipeline. It is designed for one purpose: running a trained model on new
country-year data that was not seen during training and producing a full
explanation for every prediction.

### What it does

1. Loads a trained model from MLflow by run ID (or registered model URI).
2. Reads the model's logged signature to get the exact feature list and column order
   used at training time — no manual alignment needed.
3. Loads new raw data from ADLS (same schema as training data) and applies the same
   feature engineering pipeline (lags, rolling windows, event history features).
4. Filters to the configured `inference_years` — historical years in the data are
   used only for computing lag and rolling features and are not predicted on.
5. Runs predictions across all observations, then calls
   `generate_observation_artifacts()` for each one.

### Per-observation artifacts

For every country-year prediction the notebook writes a self-contained directory:

```
inference_outputs/
  BWA_2024/
    waterfall.png              ← primary explainability plot
    force.png                  ← horizontal push/pull diagram
    feature_contributions.csv  ← all features with raw value and SHAP value
    prediction_summary.json    ← structured summary with top drivers
  SOM_2024/
    ...
  inference_results.csv        ← all countries ranked by P(collapse)
  inference_results.json       ← same, with full driver detail
```

**`waterfall.png`** — starts at the model's baseline log-odds (average across
training data) and adds one bar per feature in order of influence magnitude. Red
bars push the prediction toward collapse; blue bars push toward stability. The
final value maps to the reported probability via the logistic function. This is
the primary artifact for communicating *why* the model assessed a country the way
it did.

**`force.png`** — a horizontal representation of the same information, useful for
side-by-side comparison across countries.

**`feature_contributions.csv`** — every feature in the model with its raw value
for this country-year and its SHAP value, sorted by absolute SHAP magnitude.
Suitable for downstream analysis or inclusion in reports.

**`prediction_summary.json`** — machine-readable summary containing the country,
year, probability, predicted label, and the top five risk drivers and top five
stabilising features with their values and SHAP contributions.

### Data requirements

The input data must use the same schema as the training data and must include
enough historical years per country for lag and rolling features to be computed
(at minimum `max(lag_years)` years of history before the first inference year).
The target column is optional — the notebook runs without it.

---

## Infrastructure

| Component | Technology |
|---|---|
| Compute | Azure Machine Learning compute clusters |
| Data storage | Azure Data Lake Storage Gen2 (raw and processed parquet) |
| Experiment tracking | MLflow integrated with AML workspace |
| Model registry | MLflow Model Registry via AML |
| Hyperparameter sweep | AML HyperDrive (`jobs/sweep_outcome.yml`) |
| Authentication | Managed Identity (AML compute) / DefaultAzureCredential (local) |

## Requirements

```
pip install -r requirements.txt
```

Set the following environment variables before running any notebook locally:

```
ADLS_ACCOUNT_NAME   # Storage account name
ADLS_CONTAINER      # Container name (default: data)
ADLS_SAS_TOKEN      # or ADLS_ACCOUNT_KEY
```

On AML compute, authentication is handled automatically via Managed Identity.
