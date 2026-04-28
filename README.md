# az-ml-experiments

Predicts national-level political and humanitarian instability events from a
country-year panel dataset. The pipeline ingests data from eighteen international
sources (thirteen core + five supplementary), builds a unified feature matrix
augmented with derived features, selects
features via LASSO with a mutual-information rescue pass, trains one XGBoost
classifier per outcome, and produces full SHAP-based explainability at both the
model level and — during inference — at the level of individual country-year
observations.

All compute runs on Azure Machine Learning. Experiment tracking, model
registration, and artifact storage use MLflow integrated with the AML workspace.
Raw and processed data live in Azure Data Lake Storage Gen2.

---

## Pipeline overview

```
 Notebooks 01–13                    Notebooks 14c–14g
 Core data ingestion          +     Supplementary label sources
 (13 sources → ADLS parquets)       (RSUI, FH, EPR, PTS, ERT → ADLS parquets)
        │                                       │
        └───────────────────┬───────────────────┘
                            ↓
                       Notebook 14
                  Feature matrix assembly
                  All sources joined on ISO3
                  Outcome labels coded + forward-shifted
                            │
                            ↓
                       Notebook 14b
                  Derived feature engineering
                  Transformations · Interactions · Spatial spillover
                            │
                            ↓
                       Notebook 15
                  Feature selection (LASSO + MI rescue)
                  Outcome-specific feature lists
                            │
                            ↓
               ┌────────────────────────────┐
          Notebook 16                  Notebook 17
          XGBoost training        →    Non-linearity audit
          One model per outcome        PDP · ICE · SHAP interactions
               │
               ↓
  ┌────────────────────────────────────────────────────┐
  │                                                    │
18_xgboost_instability_classifier.ipynb  19_xgboost_inference.ipynb
Self-contained prototype                 Operational inference
Model dev + global SHAP                  Per-observation SHAP artifacts
```

---

## Notebooks

### Core data ingestion — notebooks 01–13

Each notebook pulls one data source, writes raw parquet to ADLS, and produces a
cleaned country-year aggregate ready for joining in notebook 14.

| Notebook | Source | Key variables |
|---|---|---|
| 01 | ACLED | Conflict event counts and fatalities by type (country-month → country-year) |
| 02 | World Bank WDI | GDP per capita, growth, inflation, trade, debt, health, education indicators |
| 03 | V-Dem / Polity5 | Democracy indices, electoral quality, civil liberties, Polity score |
| 04 | UCDP-GED + Powell-Thyne + PITF | Civil war onset, coup attempts, ethnic/revolutionary war, adverse regime change |
| 05 | FSI, UNHCR, UNDP | Fragile States Index sub-scores, refugee flows, Human Development Index |
| 06 | GDELT | Media-derived event tone, protest coverage, cross-national tension indices |
| 07 | FAO food prices | Domestic food price volatility and deviation from trend |
| 08 | SIPRI | Military expenditure as share of GDP and government budget |
| 09 | PRIO-Grid | Geographic and environmental risk factors aggregated to country level |
| 10 | Archigos | Leadership tenure, irregular exit history, succession type |
| 11 | Africa leadership change | Continent-specific leadership event coding for African states |
| 12 | CNTS | Cross-National Time-Series political instability indicators |
| 13 | NELDA | National election timing, competitiveness, and irregularity scores |

### Supplementary label sources — notebooks 14c–14g

These notebooks pull additional datasets that provide dependent variables not
covered by notebooks 01–13. They run independently and write parquets to ADLS;
notebook 14 joins them onto the panel spine alongside the core sources.

**`14c` — RSUI (IMF Reported Social Unrest Index)**

Source: Barrett, Appendino, Nguyen, De la Torre (2021), IMF WP/21/291.
Monthly country-level index of newspaper-derived unrest intensity, covering
~130 countries from 1996 to present.

| Label / Variable | Description |
|---|---|
| `unrest_binary` | 1 if annual max RSUI score exceeds the country's historical 75th-percentile threshold — the operationalisation in Barrett et al. 2021 |
| `rsui_mean`, `rsui_max` | Continuous annual aggregates, usable as predictors |
| `rsui_gov`, `rsui_election`, `rsui_violent` | Sub-type monthly scores for government-, election-, and violence-related unrest |

RSUI is the primary dependent variable in the IMF WP social unrest model and is
constructed independently of ACLED and GDELT, making it a cross-validating label
source for mass-protest prediction.

**`14d` — Freedom House (Freedom in the World)**

Source: Freedom House annual reports, 1972–present, ~210 countries.

| Label / Variable | Description |
|---|---|
| `fh_status_decline` | 1 if Freedom House status category worsened vs. prior year (Free→Partly Free or Partly Free→Not Free) |
| `fh_pr_decline` | 1 if Political Rights score increased by ≥1 point (more restrictive) |
| `fh_pr`, `fh_cl`, `fh_total` | Raw scores, usable as predictors in any outcome model |

Freedom House scores appear as predictors in the IMF WP/21/291 unrest model and
the Harff 2003 genocide model. `fh_status_decline` is a democratic backsliding
label that complements V-Dem ERT autocratization episodes.

**`14e` — EPR-Core (Ethnic Power Relations)**

Source: Cederman, Wimmer & Min — ETH Zurich ICR group, 1946–2020, global.

Raw data is at the ethnic group × year level; this notebook aggregates to country-year.

| Label / Variable | Description |
|---|---|
| `ethnic_exclusion_any` | 1 if any group >10% of population has Powerless or Discriminated status |
| `n_excluded_groups` | Count of politically excluded groups (any size) |
| `excluded_pop_share` | Sum of population shares of excluded groups |
| `n_relevant_groups` | Count of all politically relevant ethnic groups |
| `max_group_size` | Population share of the largest single group |
| `dominant_group_flag` | 1 if any group holds Monopoly or Dominant status |
| `epr_state_collapse` | 1 if EPR codes the country as State Collapse this year |

`ethnic_exclusion_any` is the primary predictor in ethnic war and genocide models
(Goldstone 2010, Harff 2003, Goldsmith 2013) and a key variable in the ViEWS model.

**`14f` — PTS (Political Terror Scale)**

Source: Gibney et al. — politicalterrorscale.org, 1976–present, ~180 countries.

Government repression coded 1–5 from three independent sources: Amnesty International
(PTS-A), US State Department (PTS-S), and Human Rights Watch (PTS-H).

| Label / Variable | Description |
|---|---|
| `pts_high` | 1 if pts_max ≥ 4 (systematic terror affecting large populations) |
| `pts_escalation` | 1 if pts_max increased ≥1 point vs. prior year |
| `pts_a`, `pts_s`, `pts_h` | Raw per-source scores (1–5), usable as predictors |
| `pts_mean`, `pts_max` | Cross-source summary scores |

`pts_max` is a predictor in the Harff 2003 genocide model. `pts_high` serves as
a dependent variable for state-terror onset and is a complement to mass killing
onset labels derived from UCDP.

**`14g` — V-Dem ERT (Episodes of Regime Transformation)**

Source: Edgell, Lührmann, Maerz et al. — V-Dem Institute, 1900–present, ~180 countries.

Notebook 03 pulls V-Dem core democracy indices but not ERT. ERT codes discrete
autocratization and democratization *episodes* — directional regime trajectories —
rather than index levels. A country can be autocratizing even when its absolute
democracy score remains moderate.

| Label / Variable | Description |
|---|---|
| `aut_ep` | 1 if country is in an autocratization episode this year |
| `aut_ep_start_year` | Year the current episode began (NA if not in episode) |
| `aut_ep_duration` | Years elapsed since episode start |
| `dem_ep` | 1 if country is in a democratization episode |
| `reg_trans_type` | Episode type string (gradual/rapid × aut/dem) |
| `edi`, `edi_change_3y` | Electoral Democracy Index and 3-year trend |

`aut_ep` is the primary dependent variable for democratic backsliding prediction
and complements `fh_status_decline` from notebook 14d.

### Notebook 14 — Build feature matrix

Joins all source parquets — core (01–13) and supplementary (14c–14g) — on a
common ISO3 country key to produce a single country-year panel (~167 countries ×
25 years). All five supplementary sources are wired into `RAW_PREFIXES` and joined
via a shared `_join_iso3_source` helper. Codes twelve binary outcome labels, each
forward-shifted by one year so that features at year *t* predict events at year *t+1*:

**Core outcomes**

| Label | Source | Coding |
|---|---|---|
| `civil_war_onset` | UCDP-GED | First year of state-based conflict after a ≥2-year peace spell |
| `coup_attempt` | Powell-Thyne | Any coup attempt in the year (success or failure) |
| `regime_backsliding` | V-Dem | `v2x_libdem` drops ≥0.05, or regime transitions to closed autocracy |
| `mass_unrest_onset` | ACLED | Annual protest+riot events exceed country 90th-percentile (train years only) |
| `humanitarian_crisis_onset` | FEWS NET | IPC phase ≤3 → ≥4 in year *t+1* (~39 FEWS countries; NA elsewhere) |

**Supplementary outcomes**

| Label | Source | Coding |
|---|---|---|
| `unrest_binary` | RSUI (14c) | Annual max RSUI score exceeds country-specific 75th-percentile |
| `fh_status_decline` | Freedom House (14d) | Freedom House status category worsened vs. prior year |
| `ethnic_exclusion_any` | EPR (14e) | Any large ethnic group (>10%) coded Powerless or Discriminated |
| `epr_state_collapse` | EPR (14e) | EPR codes the country as State Collapse |
| `pts_high` | PTS (14f) | Government repression score ≥ 4 across any source |
| `pts_escalation` | PTS (14f) | PTS max score increased ≥1 point year-on-year |
| `aut_ep` | V-Dem ERT (14g) | Country is in an active autocratization episode |

All supplementary labels are pre-computed in their respective source notebooks and
forward-shifted here. Adding a new source requires only adding its prefix to
`RAW_PREFIXES` and extending `OUTCOME_COLS`.

### Notebook 14b — Derived feature engineering

Sits between notebook 14 and notebook 15. Reads the assembled feature matrix,
adds three categories of derived columns, runs a pre-LASSO audit, and writes the
augmented matrix back to ADLS. Notebook 15 reads from this output rather than
directly from notebook 14.

Each section is independently toggled in `ENG_CFG` and skipped gracefully if
its dependencies are absent.

**Section A — Variable transformations**

| Transform | Applied to | Purpose |
|---|---|---|
| `log1p` | GDP, population, fatality counts, refugee flows | Compresses right-skewed distributions; linearises LASSO penalty |
| `sqrt` | Bounded event counts | Milder compression for zero-inclusive counts |
| First difference (`_diff1`) | GDP, democracy indices, FSI, CPI | Year-on-year change within-country — captures deterioration rather than level |
| HP-filter trend / cycle | GDP, inflation, military spend, FSI | Separates secular trend from cyclical deviation using the Hodrick-Prescott filter (λ=6.25 for annual data) |

**Section B — Interaction effects**

Pairwise products from a configured list, standardised (z-scored) before
multiplication so the product is in interpretable standard-deviation units.
Interactions are explicitly configured rather than enumerated — exhaustive
enumeration over 200+ features would produce ~20,000 noisy candidates and
overwhelm the LASSO screening pass. Configured pairs encode domain hypotheses:
economic stress × political exclusion, food insecurity × conflict history,
military capacity × state fragility, and others.

**Section C — Spatial spillover features**

Builds a K-nearest-neighbour spatial weights matrix once from capital city
coordinates (`data/geo_coords.csv`, not included — see notebook header for
construction instructions). For each configured variable and each year:

| Feature suffix | What it captures |
|---|---|
| `_slag` | Neighbor-weighted mean — direct regional contagion signal |
| `_local_moran` | Local Moran's I — whether the country clusters with or diverges from its neighbors |
| `_lisa_quad` | LISA quadrant (1=High-High hotspot, 2=Low-High, 3=Low-Low, 4=High-Low) |

A **feature catalog CSV** is written alongside the output parquet, documenting
every derived column, its source variables, the transform applied, missingness
rate, and point-biserial correlation with each outcome label.

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

## Self-contained classifier notebook (18)

**`notebooks/18_xgboost_instability_classifier.ipynb`** is an end-to-end prototype
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

## Inference notebook (19)

**`notebooks/19_xgboost_inference.ipynb`** is the operational complement to the
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
