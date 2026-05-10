> **Archived.** The original 4–6 month project plan, written before notebooks 01–26 existed. Forward-looking sections (architecture, decisions, open work) have been merged into `../project-reference.md`. Data and predictor sections have been merged into `../data-and-predictors.md`. The notebooks themselves are now the canonical record of what was actually built. Phase IV extensions (LSTM, LLM narratives, stacking ensemble) are explicitly out of scope for the current iteration; consult before reviving.

---

# Meta-Plan: Nation-State Instability Forecasting
## 4–6 Month Single Data Scientist Project on Azure ML

**Compiled:** April 2026
**Compute & tracking:** Azure ML (HyperDrive, MLflow, Managed Endpoints)
**Architecture target:** Four thematic sub-models per outcome → stacking ensemble
**Primary evaluation metric:** AUPRC per outcome (not AUROC — rare events; per Ward, Greenhill & Bakke 2010)
**Prediction horizon:** 6-month ahead (primary); 12-month ahead (stretch goal)

---

## Problem Redefinition

This is a **nation-state level structural forecasting** problem, not an event classification problem. The distinction matters for every design decision downstream:

| Dimension | Event classification (old) | Structural forecasting (this plan) |
|-----------|---------------------------|-------------------------------------|
| Unit of analysis | Individual event record | Country-month or country-year |
| Labels | Event type (protest, riot, etc.) | Onset of civil war / coup / regime change / mass unrest |
| Predictors | Event text, date, location | Structural indicators: governance, economy, fragility, demographics, displacement |
| Event data role | Labels to predict | One predictor domain among many |
| Time horizon | Same month | 6–12 months ahead |
| Model architecture | Single classifier | Multiple thematic sub-models fused into ensemble |
| Key challenge | Extraction quality | Rare event imbalance + temporal leakage |

ACLED, PEA outputs, and GDELT enter as **predictor features** (aggregated to country-month counts, fatality rates, protest intensity scores) — not as labels.

---

## Target Outcomes (Four Parallel Models)

Four outcome models are trained independently then fused. Each is a binary onset prediction.

### 1. Civil War Onset
- **Definition:** First month of armed intrastate conflict (≥25 battle deaths) after ≥24-month peace spell
- **Base rate:** ~5–8 onsets/year globally; extremely rare per country-year
- **Ground truth:** UCDP/PRIO Armed Conflict Dataset (ACD); available annually 1946–present; monthly from UCDP GED
- **Key literature:** Fearon & Laitin (2003); Muchlinski et al. (2016); Mueller & Rauh (2019)
- **Lead predictors:** Regime type (anocracy gap), GDP per capita, mountainous terrain, ethnic fractionalization, prior conflict history

### 2. Coup Attempt
- **Definition:** Any illegal, overt attempt to seize executive power by a portion of the state military or security forces
- **Base rate:** ~10–15 attempts/year globally; ~3–5 successes
- **Ground truth:** Powell-Thyne Coup Dataset (1950–present, updated annually); CoupCast auxiliary data
- **Key literature:** Beger, Dorff & Ward (2014/2017); Powell (2012); CoupCast (Powell/UCF)
- **Lead predictors:** Executive constraints, prior coup history, economic contraction, military size/spending, regional coup wave

### 3. Regime Change / Democratic Backsliding
- **Definition:** Substantial decline in democratic quality (V-Dem LDI drop ≥0.1 within 2 years) OR transition from electoral autocracy to closed autocracy
- **Base rate:** ~8–12 meaningful backsliding episodes/year; relatively higher base rate than civil war or coup
- **Ground truth:** V-Dem Liberal Democracy Index; Varieties of Democracy `v2x_libdem`; episodes dataset `v2x_regime`
- **Key literature:** Morgan, Beger & Glynn (2019); Lührmann & Lindberg (2019); V-Dem project
- **Lead predictors:** Executive aggrandizement indicators (V-Dem `v2exrescon`), civil society repression, judicial independence, election manipulation, economic inequality

### 4. Mass Social Unrest
- **Definition:** Country-month with ACLED protest/riot event count exceeding the 90th percentile of that country's historical distribution, OR ≥3 events with reported fatalities in a 30-day window
- **Base rate:** Highest of the four — ~15–20% of country-months for countries with active civil society
- **Ground truth:** ACLED disorder events (demonstrations + political violence, aggregated monthly); threshold defined per country to account for baseline differences
- **Key literature:** Ramakrishnan/EMBERS (2014); Choi & Pieniążek (2023); IMF WP Barrett et al. (2021)
- **Lead predictors:** Unemployment, food price shocks, austerity events, protest contagion from neighbors, social media penetration, prior protest history

---

## Predictor Feature Set

This is the core differentiator from prior event-classification work. Features are organized into six thematic domains that mirror the sub-model architecture.

### Domain 1 — Structural / Economic
| Feature | Operationalization | Source | Frequency |
|---------|-------------------|--------|-----------|
| GDP per capita (log) | Log-transformed PPP constant | World Bank WDI `NY.GDP.PCAP.PP.KD` | Annual |
| GDP growth rate | YoY % change | WDI `NY.GDP.MKTP.KD.ZG` | Annual |
| Inflation rate | CPI annual % | WDI `FP.CPI.TOTL.ZG` | Annual |
| Unemployment rate | % of labor force | WDI `SL.UEM.TOTL.ZS` | Annual |
| Food price index | FAO FPFI; country-level WDI food inflation | FAO / WDI | Monthly |
| Gini coefficient | Income inequality | WDI `SI.POV.GINI` | Annual (interpolated) |
| Resource rents | Oil + mineral rents % of GDP | WDI `NY.GDP.TOTL.RT.ZS` | Annual |
| Trade openness | Exports + imports % GDP | WDI | Annual |
| External debt | % of GNI | WDI | Annual |

### Domain 2 — Governance / Political
| Feature | Operationalization | Source | Frequency |
|---------|-------------------|--------|-----------|
| Liberal democracy index | V-Dem `v2x_libdem` | V-Dem v14 | Annual |
| Electoral democracy index | V-Dem `v2x_polyarchy` | V-Dem | Annual |
| Executive constraints | Polity5 `XCONST`; V-Dem `v2exrescon` | Polity5 / V-Dem | Annual |
| Regime type | Polity5 `polity2` score (−10 to +10); anocracy flag (−5 to +5) | Polity5 | Annual |
| Political stability | WGI `PV.EST` (Political Stability and Absence of Violence) | World Bank WGI | Annual |
| Govt effectiveness | WGI `GE.EST` | WGI | Annual |
| Rule of law | WGI `RL.EST` | WGI | Annual |
| Control of corruption | WGI `CC.EST` | WGI | Annual |
| Election proximity | Months to/from last/next legislative election | NELDA / V-Dem | Monthly |
| Executive turnover | Leader change in prior 12 months (binary) | Archigos / GDELT | Monthly |

### Domain 3 — Fragility / Humanitarian
| Feature | Operationalization | Source | Frequency |
|---------|-------------------|--------|-----------|
| Fragile States Index (total) | FSI composite score (0–120) | Fund for Peace | Annual |
| FSI sub-scores | Security apparatus, factionalised elites, group grievance, economy, state legitimacy, public services, human rights, demographic pressures, refugees/IDPs, external intervention | Fund for Peace | Annual |
| Refugees hosted | UNHCR total refugees in country (log) | UNHCR | Annual |
| IDPs | Internally displaced persons (log) | UNHCR / IDMC | Annual |
| Refugee outflow | New refugee departures from country (log) | UNHCR | Annual |
| Humanitarian operations | UN OCHA active operations (binary) | UN OCHA | Annual |
| Human Development Index | UNDP HDI composite | UNDP | Annual |
| Child mortality | Under-5 mortality rate | WDI | Annual |

### Domain 4 — Event History (ACLED as predictor)
| Feature | Operationalization | Source | Frequency |
|---------|-------------------|--------|-----------|
| Protest event count | ACLED demonstrations events, country-month | ACLED | Monthly |
| Political violence count | ACLED violence events, country-month | ACLED | Monthly |
| Fatalities | Total reported fatalities, country-month (log+1) | ACLED | Monthly |
| 3-month rolling event trend | Change in event count vs prior 3-month mean | ACLED | Monthly |
| 12-month event intensity | Rolling annual sum (normalised by population) | ACLED | Monthly |
| State repression events | ACLED "violence against civilians" by state forces | ACLED | Monthly |
| Spatial contagion | Neighbor-country event count, inverse-distance weighted | ACLED | Monthly |

### Domain 5 — Technology / Information
| Feature | Operationalization | Source | Frequency |
|---------|-------------------|--------|-----------|
| Mobile phone subscriptions | Per 100 people | ITU / WDI `IT.CEL.SETS.P2` | Annual |
| Internet users | % of population | WDI `IT.NET.USER.ZS` | Annual |
| Social media penetration | Facebook/Meta users as % population (where available) | Meta Data for Good / NapoleonCat | Annual |
| Press freedom index | RSF World Press Freedom Index score | RSF | Annual |
| Internet shutdowns | Binary: documented shutdown in prior 6 months | NetBlocks / Access Now | Monthly |

### Domain 6 — GDELT Signals
| Feature | Operationalization | Source | Frequency |
|---------|-------------------|--------|-----------|
| Goldstein score | Monthly average tone of all events involving country | GDELT | Monthly |
| Conflict event volume | GDELT conflict/material conflict event count | GDELT | Monthly |
| Mediation/dialogue events | GDELT verbal cooperation volume | GDELT | Monthly |

---

## Architectural Approach: Thematic Sub-Models + Stacking Ensemble

Each of the four outcome models (civil war, coup, regime change, mass unrest) is built using the same three-layer architecture:

```
Layer 1: Thematic sub-models (one per domain)
  ├── Sub-model A: Structural/Economic features → XGBoost
  ├── Sub-model B: Governance/Political features → XGBoost
  ├── Sub-model C: Fragility/Humanitarian features → XGBoost
  ├── Sub-model D: Event history (ACLED) features → XGBoost + optional LSTM
  └── Sub-model E: Technology + GDELT signals → XGBoost

Layer 2: Stacking ensemble
  └── Meta-learner (logistic regression) on out-of-fold sub-model predictions

Layer 3: Calibration
  └── Platt scaling or isotonic regression → calibrated probability output
```

This mirrors the EMBERS, CoupCast, and ViEWS architectures. Each sub-model is independently tunable and inspectable via SHAP. The meta-learner learns which domains are most predictive for each outcome.

---

## Phase I — Infrastructure & Data Assembly (Month 1)

### Weeks 1–2: Azure ML Workspace + Data Procurement

**Infrastructure**
- Azure ML Workspace via Bicep: Workspace, Storage Account, CPU cluster (Standard_DS3_v2, 0–4 nodes autoscale)
- MLflow tracking URI → Azure ML workspace
- Repo structure: `data/{raw,processed,splits,external}`, `notebooks/`, `src/{features,training,inference,monitoring}`, `configs/`, `infra/`, `docs/`
- `.env.template`: `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_WORKSPACE_NAME`, `ACLED_API_KEY`, `MLFLOW_TRACKING_URI`

**Data procurement priority order:**

| Priority | Dataset | Access | Notes |
|----------|---------|--------|-------|
| 1 | ACLED (1997–present) | Free registration | Monthly aggregation to country-level; used as predictor domain |
| 1 | V-Dem v14 | Free download | Core regime type + democratic backsliding labels and features |
| 1 | World Bank WDI | `wbdata` Python package | Economic + development predictors |
| 1 | World Bank WGI | WB API | Governance indicators |
| 2 | Polity5 | Free download (CSP) | Executive constraints; anocracy gap |
| 2 | Fragile States Index | Fund for Peace (annual PDF/CSV) | All 12 sub-indicators |
| 2 | UCDP/PRIO ACD | Free download | Civil war onset labels |
| 2 | Powell-Thyne Coup Dataset | Free download | Coup attempt labels |
| 2 | UNHCR population data | UNHCR data portal | Refugees, IDPs |
| 3 | ITU ICT data | ITU free download | Cell phone + internet penetration |
| 3 | UNDP HDI | UNDP data portal | Human development composite |
| 3 | GDELT (monthly aggregates) | BigQuery | Goldstein scores, event volumes |
| 3 | FAO Food Price Index | FAO API | Monthly food prices |

**Decisions to lock in Week 2 — document in `docs/schema_definition.md`:**
1. **Unit of analysis:** country-month (primary); country-year for annual-only features (interpolated or forward-filled to monthly)
2. **Geographic scope:** Global (all sovereign states with sufficient data coverage, ~160 countries); Africa-first for validation
3. **Temporal scope:** 2000–2024 for primary dataset; extend to 1990 if coup/civil war labels are too sparse
4. **Missing data strategy:** Forward-fill annual indicators to monthly; flag missingness as binary feature for fragile states where data gaps are informative

**Outputs:** All raw datasets downloaded + versioned as Azure ML Data Assets, `docs/schema_definition.md`

---

### Weeks 3–4: Panel Construction + EDA

**Panel construction** (`src/features/build_panel.py`):
- Master panel: one row per country-month; ~160 countries × ~288 months (2000–2024) = ~46,000 rows
- Left-join all feature domains onto country-month index
- Resolve country-code harmonisation (ISO3, COW codes, ACLED codes — all different; build crosswalk table)
- Forward-fill annual indicators within-country; document which indicators are interpolated
- Output: `data/processed/master_panel.parquet`

**Label construction** (`src/features/build_labels.py`):
- `civil_war_onset`: 1 in the first month of UCDP episode after ≥24-month peace spell; 0 elsewhere; NaN during active conflict (excluded from training)
- `coup_attempt`: 1 in month of Powell-Thyne coup attempt; 0 elsewhere
- `regime_change`: 1 if V-Dem LDI drops ≥0.05 in the following 12 months (forward-looking — shift label by prediction horizon); 0 elsewhere
- `mass_unrest`: 1 if ACLED protest+riot count exceeds country-specific 90th percentile in a given month; 0 elsewhere
- **Critical:** shift all labels forward by prediction horizon (6 months default) relative to features. Verify no leakage.

**EDA notebook** (`notebooks/01_eda.ipynb`):
- Class distribution per outcome: expect civil_war_onset ~0.2%, coup ~0.08%, regime_change ~1%, mass_unrest ~12%
- Missing data heatmap by country-year — highest missingness in fragile states (document)
- Correlation matrix of predictors (flag collinear pairs >0.85)
- Temporal event distribution: how many onsets per year? Spikes around 2011 (Arab Spring), 2020–2022?
- Geographic distribution: Africa, MENA, Central Asia likely overrepresented

**Outputs:** `data/processed/master_panel.parquet`, `data/processed/labels.parquet`, `notebooks/01_eda.ipynb`, country-code crosswalk

---

## Phase II — Baseline + Sub-Model Training (Months 2–3)

### Weeks 5–6: Temporal Splits + Baselines

**Strict temporal splits** (`src/features/splitter.py`):
- Train: 2000–2020
- Validation: 2021–2022
- Test: 2023–2024 (held out until final evaluation only)
- **Never shuffle. Never use k-fold across time.** Assert at runtime that no test-set timestamp appears in training index.
- For civil war and coup models: train period may need extending to 1990 given low base rates — evaluate empirically

**Rule-based baselines** (logged to MLflow as `baseline_rules_{outcome}`):
- Civil war: IF Polity2 score between −5 and +5 (anocracy) AND GDP per capita <$2000 → onset=1
- Coup: IF prior coup in last 5 years AND executive constraints low → attempt=1
- Regime change: IF V-Dem LDI declined last 2 years AND opposition repressed → change=1
- Mass unrest: IF prior month ACLED count > country mean × 1.5 AND unemployment rising → unrest=1

**Logistic regression baselines** (Firth rare-event logistic regression for civil war + coup; standard for regime change + unrest):
- 5–8 features per model; log AUPRC, F1, per-class precision/recall to MLflow

---

### Weeks 7–10: Thematic Sub-Models (XGBoost per Domain per Outcome)

For each of the 4 outcomes × 5 feature domains = up to 20 sub-models. In practice, not all domain/outcome combinations are meaningful — prioritise:

**Civil war:** Economic + Governance + Fragility domains most predictive (per Fearon-Laitin, Muchlinski)
**Coup:** Governance + Economic + Event history most predictive (per Beger-Ward, CoupCast)
**Regime change:** Governance + Technology (information control) most predictive (per V-Dem backsliding literature)
**Mass unrest:** Economic + Event history + Technology most predictive (per EMBERS, IMF WP)

Each sub-model is an XGBoost binary classifier with:
- `scale_pos_weight = n_negative / n_positive` (severe imbalance requires this)
- `eval_metric = "aucpr"` (AUPRC tracked during training)
- HyperDrive job on Azure ML CPU cluster: 30–50 iterations per sub-model, Bandit early stopping
- Log to MLflow: AUPRC, F1, confusion matrix, SHAP summary plot

Sub-model scripts: `src/training/train_submodel.py --outcome civil_war --domain economic`

**SHAP analysis for each sub-model** (`notebooks/03_shap_{outcome}.ipynb`):
- Feature importance rankings per domain per outcome
- Partial dependence plots for top 5 features
- Identify which features are driving predictions vs. which add noise
- These SHAP values feed the LLM narrative layer in Phase IV

---

### Weeks 11–12: Stacking Ensemble

**Stacking ensemble** (`src/training/train_ensemble.py --outcome {outcome}`):
- Out-of-fold sub-model predictions on training set → meta-features (one column per sub-model)
- Meta-learner: logistic regression with L2 regularisation (simple, interpretable; avoids overfitting on small meta-feature matrix)
- Also evaluate: weighted averaging of calibrated sub-model probabilities (simpler; often within 2–3 AUPRC of stacking)
- Probability calibration: Platt scaling on validation set

**Ensemble evaluation** (`notebooks/04_ensemble_comparison.ipynb`):
- Compare: each individual sub-model vs. ensemble on validation AUPRC
- Expected gain: 5–15 AUPRC points over best sub-model (per EMBERS, ViEWS, CoupCast literature)
- If ensemble does not beat best sub-model: check sub-model correlation — if all sub-models agree, stacking adds nothing; investigate which domain is driving divergence
- Per-country and per-year breakdown of predictions: where does the ensemble fail?

**Champion selection per outcome:**
- Primary: AUPRC on validation set
- Secondary: false negative rate at high-stakes threshold (e.g., P(coup) > 0.20 → alert)
- Register 4 champion ensemble models in MLflow Registry under status `Staging`

**Outputs:** 4 registered ensemble models, ensemble comparison notebook, `models/champion_models.json`

---

## Phase III — Deployment + Monitoring (Month 4)

### Weeks 13–14: Inference Package

**Inference code** (`src/inference/`):
- `model_loader.py` — fetch all 4 champion models from MLflow registry; cache locally
- `preprocessor.py` — same feature pipeline as training; validates input schema; handles missing values identically to training
- `predictor.py` — for a given country + reference month → run all 4 sub-model chains → ensemble → output JSON per outcome
- `app.py` — FastAPI:
  - `POST /predict` — single country, single reference month
  - `POST /predict/batch` — list of countries, returns ranked risk table
  - `GET /health`

Output schema for `POST /predict`:
```json
{
  "country": "Mali",
  "reference_month": "2024-06",
  "prediction_horizon_months": 6,
  "predictions": {
    "civil_war_onset":  {"probability": 0.04, "risk_level": "low"},
    "coup_attempt":     {"probability": 0.18, "risk_level": "elevated"},
    "regime_change":    {"probability": 0.31, "risk_level": "high"},
    "mass_unrest":      {"probability": 0.67, "risk_level": "high"}
  },
  "model_versions": {
    "civil_war": "instability-civil-war-v1",
    "coup": "instability-coup-v1",
    "regime_change": "instability-regime-change-v1",
    "mass_unrest": "instability-unrest-v1"
  }
}
```

**Risk level thresholds** (calibrate on validation set):
- `low`: P < 0.10
- `elevated`: 0.10 ≤ P < 0.25
- `high`: 0.25 ≤ P < 0.50
- `critical`: P ≥ 0.50

**Tests** (`tests/`):
- `test_preprocessor.py` — known inputs → known feature values; missing data handled identically to training
- `test_predictor.py` — mock models, verify output schema for all 4 outcomes
- Integration test: 20 country-months from test set, spot-check predictions vs. historical outcomes

**Azure ML Managed Endpoint:**
```bash
az ml online-endpoint create --name instability-forecaster \
  --resource-group <rg> --workspace-name <ws>

az ml online-deployment create --endpoint instability-forecaster \
  --name v1 --model <model-uri>
```

**Outputs:** Inference package, all tests passing, endpoint live, 4-outcome health check succeeds

---

### Weeks 15–16: Monitoring + Documentation

**Data freshness monitoring** (`src/monitoring/freshness_checker.py`):
- Monthly job: verify ACLED pull is current, WDI/V-Dem versions match registered model training versions
- Alert if any predictor domain is >30 days stale for monthly features, or >14 months for annual features

**Prediction monitoring** (`src/monitoring/prediction_monitor.py`):
- Monthly: compare prediction probability distributions to historical baseline
- Alert if country-level risk scores shift dramatically without corresponding feature change (possible data ingestion error)
- Log all inference requests + predictions to Application Insights

**Retraining trigger criteria** (document in `docs/retraining_policy.md`):
- Annual retraining: incorporate new ACLED, WDI, V-Dem, FSI data
- Event-triggered retraining: if a major global shock (pandemic, war) affects >20 countries simultaneously, retrain immediately with updated data
- Performance-triggered: if retrospective evaluation (comparing predictions to outcomes 6 months later) shows AUPRC decline >0.05

**Documentation:**
- `docs/schema_definition.md` — all features, labels, operationalisations, data sources
- `docs/monitoring_guide.md` — freshness checks, Application Insights queries
- `docs/retraining_policy.md` — when and how to retrain
- `docs/rollback_playbook.md` — revert to previous model version via MLflow registry
- `docs/runbook.md` — end-to-end data refresh → retrain → evaluate → promote → deploy

**Outputs:** Monitoring dashboard, all runbooks, MLflow audit, go-live checklist

---

## Phase IV — Optional Extensions (Months 5–6)

### Month 5: LSTM Temporal Branch for Unrest + Coup Models

Mass unrest and coup models have the richest monthly time-series signal (ACLED monthly counts; GDELT Goldstein scores). An LSTM branch processes these as sequences, capturing dynamics that tabular lag features miss:

- **Architecture:** Hybrid Bi-LSTM + XGBoost (Chitengu et al. 2025)
  - Bi-LSTM: 12-month sequences of [ACLED count, ACLED fatalities, GDELT Goldstein, food price index] per country
  - XGBoost: structural features (regime type, GDP, FSI)
  - Fusion: concatenate LSTM hidden state + XGBoost leaf embeddings → logistic classification head
- **Azure ML GPU cluster** (Standard_NC6): only for LSTM training; autoscale to zero otherwise
- Only implement if tabular ensemble AUPRC is below 0.55 for unrest or 0.45 for coup

### Month 6: LLM Narrative Explanations

Following the USHMM Early Warning Project and Mueller/Rauh ConflictForecast.org pattern, an LLM generates analyst-facing risk narratives. This uses Azure AI Foundry (same infrastructure as the PEA pipeline):

- **Input:** SHAP feature attributions for a country-month prediction + most recent ACLED event summaries + FSI sub-scores
- **Output:** 3-paragraph narrative:
  1. Risk summary: what is elevated and why
  2. Key drivers: top 3 SHAP features in plain language
  3. Historical context: last comparable episode for this country (from training data lookup)
- `src/inference/narrator.py` — zero-shot prompt; no fine-tuning required
- LLMs are **not** the classifier; they explain the classifier's output

---

## Azure ML Architecture Map

```
Azure ML Workspace
├── Compute
│   ├── cpu-cluster (Standard_DS3_v2, 0–4 nodes)    sub-model training, HyperDrive
│   └── gpu-cluster (Standard_NC6, 0–2 nodes)        LSTM branch (Phase IV only)
│
├── Data Assets (versioned)
│   ├── acled-global-2000-present
│   ├── vdem-v14-panel
│   ├── wdi-panel
│   ├── wgi-panel
│   ├── polity5-panel
│   ├── fragile-states-index
│   ├── ucdp-acd-episodes
│   ├── powell-thyne-coups
│   ├── unhcr-population
│   └── master-panel-vN          (joined, versioned monthly)
│
├── Experiments (MLflow — one experiment per outcome)
│   ├── civil-war-onset/
│   │   ├── baseline-rules, baseline-logreg
│   │   ├── submodel-economic, submodel-governance, submodel-fragility
│   │   └── ensemble-stacking
│   ├── coup-attempt/    (same structure)
│   ├── regime-change/   (same structure)
│   └── mass-unrest/     (same structure)
│
├── Model Registry
│   ├── instability-civil-war-v1      (Staging → Production)
│   ├── instability-coup-v1
│   ├── instability-regime-change-v1
│   └── instability-mass-unrest-v1
│
└── Managed Endpoint
    └── instability-forecaster        (FastAPI, 4-outcome output, autoscale 0–2 replicas)
```

---

## Single Data Scientist Scope Adjustments

**Annotation bottleneck eliminated.** All four outcome labels are sourced from established expert-coded datasets (UCDP, Powell-Thyne, V-Dem, ACLED). No manual annotation required.

**Data harmonisation is the real Month 1 bottleneck.** Country-code crosswalking across ACLED (ISO3), COW numeric, V-Dem string, Polity5 country names, and Powell-Thyne codes is the single most time-consuming data task. Budget the full two weeks for it. Build and test `data/crosswalks/country_codes.csv` before touching any model code.

**Run sub-models in parallel with HyperDrive.** A single data scientist cannot tune 20 sub-models manually. Submit overnight HyperDrive jobs for each domain/outcome combination. Budget ~$50–80/month for CPU cluster time; set Azure spending alert at $150/month.

**Start with 2 outcomes.** If Month 1 runs over, prioritise **coup** (rarest, highest policy value, best-established literature benchmark) and **mass unrest** (most data, monthly signal, easiest to evaluate). Civil war and regime change can be added in months 3–4 once the pipeline is proven.

---

## Success Criteria

| Milestone | Target | Notes |
|-----------|--------|-------|
| Month 1 complete | Master panel built, 4 label series constructed, EDA notebook runnable | Country-code crosswalk is the gating dependency |
| Month 2 complete | All baselines logged; at least one sub-model beats logistic regression on AUPRC | |
| Month 3 complete | Coup ensemble AUPRC ≥ 0.45 on validation; mass unrest AUPRC ≥ 0.55 | Per CoupCast benchmark and IMF WP benchmarks |
| Month 4 complete | All 4 ensemble models registered; endpoint live; monitoring dashboard active | |
| Month 5 *(optional)* | LSTM hybrid improves unrest AUPRC by ≥ 0.05 over tabular ensemble | Only pursue if tabular ensemble plateaus |

---

## Timeline Summary

| Month | Phase | Key Deliverables | Azure ML Usage |
|-------|-------|-----------------|----------------|
| 1 | Infrastructure + Data | Workspace live, master panel built, 4 label series, EDA | Data Assets versioned, first MLflow run |
| 2 | Baselines + Splits | Temporal splits locked, rule-based + logit baselines for all 4 outcomes | MLflow experiment tracking |
| 3 | Sub-model tuning | XGBoost sub-models per domain per outcome, SHAP analysis | HyperDrive on CPU cluster |
| 4 | Ensemble + Deployment | Stacking ensembles, 4-outcome endpoint live, monitoring, runbooks | Managed Endpoint, Application Insights |
| 5 *(optional)* | LSTM branch | Hybrid architecture for coup + unrest if tabular plateaus | GPU cluster |
| 6 *(optional)* | LLM narratives | Analyst-facing risk summaries via Azure Foundry | Foundry API (shared with PEA) |

---

## Key Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Country-code harmonisation takes >2 weeks | High | Start with ACLED + V-Dem + Polity5 only; add other datasets incrementally; build crosswalk as first deliverable |
| Civil war + coup labels too sparse for ML (< 200 positive cases) | Medium | Extend temporal coverage to 1990; consider pooling conflict-prone regions; evaluate whether base rates support the approach |
| Annual predictors (FSI, V-Dem) have large missingness for fragile states | High | Add binary missingness indicator per feature; forward-fill within-country; document imputation in `schema_definition.md` |
| Temporal leakage invalidates results | High | Assert at runtime that test-set timestamps never appear in training index; review label construction for forward-looking features |
| Azure ML compute costs exceed budget | Medium | CPU cluster autoscales to zero; HyperDrive Bandit early stopping; GPU cluster disabled except Phase IV |
| Sub-models highly correlated → ensemble adds nothing | Medium | Evaluate sub-model correlation matrix before building meta-learner; if correlation > 0.90 across domains, investigate whether feature domains are actually distinct |

---

## Source Documents

This meta-plan synthesizes four planning documents in this repo:

| File | Contents |
|------|----------|
| `plan-socialInstabilityClassifiers.prompt.md` | Original 4-month phased plan (data procurement, annotation, tuning, deployment) |
| `model_types_instability_prediction.md` | Literature synthesis across model generations (logistic regression → XGBoost → LSTM → LLMs); ensemble fusion methods |
| `# Dataset & Variable Synthesis.md` | Source dataset profiles (ACLED, V-Dem, WDI, UCDP, Powell-Thyne, FSI, UNHCR, ITU); dependent and independent variable taxonomies |
| `# Snowball Literature Review: Machine Le.md` | Annotated bibliography of 30+ studies across five instability domains (EMBERS, ViEWS, CoupCast, IMF WP, Muchlinski, Fearon-Laitin, Beger-Ward) |
