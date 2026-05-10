> **Archived.** Compute, runtime, and budget content has been merged into `../project-reference.md` (sections 3–5). This file is preserved for the original phrasing only — do not update it; update the active reference instead.

---

# Azure ML Usage Plan, VM Recommendations, and Budget Estimates

## Context

Five XGBoost instability classifiers trained on a country-year panel (~54 African countries,
2000–2022, ~1,100–1,400 rows per outcome after filtering). Pipeline runs in four stages:
data pull → feature engineering → model development → inference. Only the HyperDrive
sweep jobs run on an auto-scaling cluster; all other work runs interactively on a
single compute instance.

---

## 1. Resource Map — What Runs Where

| Workload | Resource type | Why |
|---|---|---|
| Data pull notebooks (`01_*`) | Compute instance | Light I/O, pandas; network-bound, not CPU-bound |
| Feature engineering (`02_*`) | Compute instance | Spatial weights (libpysal), HP-filter, Moran's I — single-node, memory-moderate |
| LASSO feature selection (`03/01`) | Compute instance | `LogisticRegressionCV(solver='saga', n_cs=30)` × 5 folds × 5 outcomes ≈ 750 fits, fast on ~1,100 rows |
| XGBoost sweep (`sweep_outcome.yml`) | CPU cluster (auto-scale) | HyperDrive parallelises 5 concurrent trials across cluster nodes |
| Inference (`04/01`) | Compute instance | Single-pass predict + SHAP on saved model artifact |

**Nothing in this pipeline requires GPU.** XGBoost with `tree_method="hist"` is optimally fast on CPU for datasets of this size.

---

## 2. Compute Recommendations

### 2a. Compute Instance (interactive work)

The instance runs all day while working. Stop it when not in use — no idle charges if stopped.

| Tier | VM | vCPU | RAM | When to choose |
|---|---|---|---|---|
| Minimum | Standard_DS3_v2 | 4 | 14 GB | Light feature eng; most training on cluster |
| **Recommended** | **Standard_DS4_v2** | **8** | **28 GB** | **Comfortable for libpysal spatial ops, LASSO CV** |
| Generous | Standard_DS5_v2 | 16 | 56 GB | Running full pipeline interactively, multiple outcomes in parallel |

Standard_DS4_v2 is the recommended instance. The 8-vCPU / 28 GB profile handles
`LogisticRegressionCV(n_jobs=-1)` across all 5 outcomes, the spatial weights matrix
construction in `02/01`, and the feature matrix join without memory pressure.

### 2b. HyperDrive CPU Cluster

Each trial is one `train_outcome.py` run: load parquet from ADLS, XGBoost fit, SHAP,
write artifacts, log to MLflow. The sweep YAML sets `max_concurrent_trials: 5`.

| Tier | VM per node | vCPU | RAM | Trade-off |
|---|---|---|---|---|
| Budget | Standard_DS2_v2 | 2 | 7 GB | Slower tree building; ~30–60 sec longer per trial |
| **Recommended** | **Standard_DS3_v2** | **4** | **14 GB** | **Sweet spot — all 4 cores used by `n_jobs=-1`** |
| Fast | Standard_F8s_v2 | 8 | 16 GB | ~30–45 sec faster per trial; ~60% more per node |

Standard_DS3_v2 at 5 nodes is the recommended cluster configuration. For a ~1,200-row
dataset, moving to 8 vCPU saves little real wall-clock time relative to cost.

**Cluster config:**
```yaml
min_instances: 0          # scale to zero when idle — no idle charges
max_instances: 5          # matches max_concurrent_trials in sweep_outcome.yml
idle_seconds_before_scaledown: 120
```

---

## 3. Estimated Runs Per Model Type

### 3a. LASSO Feature Selection (notebook `03/01`, runs on compute instance)

Runs once per development cycle. Fits are fast on ~1,100 rows.

| Outcome | Approx. rows | Approx. features in | CS path | CV folds | Total fits |
|---|---|---|---|---|---|
| civil_war_onset | ~1,100 | 80–150 | 30 | 5 | 150 |
| coup_attempt | ~1,100 | 80–150 | 30 | 5 | 150 |
| regime_backsliding | ~1,100 | 80–150 | 30 | 5 | 150 |
| mass_unrest_onset | ~1,100 | 80–150 | 30 | 5 | 150 |
| humanitarian_crisis_onset | ~700 (FEWS countries) | 80–150 | 30 | 5 | 150 |
| **Total** | | | | | **750 fits** |

Estimated wall clock on Standard_DS4_v2: **8–15 minutes** (all 5 outcomes together).

### 3b. HyperDrive XGBoost Sweeps (cluster, `sweep_outcome.yml`)

Per outcome, per sweep run:

| Outcome | Max trials | Concurrent | Expected early stops | Effective trials | Wall clock (est.) |
|---|---|---|---|---|---|
| civil_war_onset | 40 | 5 | ~10–15 | ~25–30 | 20–30 min |
| coup_attempt | 40 | 5 | ~10–15 | ~25–30 | 20–30 min |
| regime_backsliding | 40 | 5 | ~8–12 | ~28–32 | 20–30 min |
| mass_unrest_onset | 40 | 5 | ~5–10 | ~30–35 | 25–35 min |
| humanitarian_crisis_onset | 20 | 5 | ~5–8 | ~12–15 | 15–20 min |

Bandit early stopping (`slack_factor: 0.15`, evaluation begins at trial 5) terminates
roughly 25–35% of trials before completion. Each trial takes ~90–150 seconds on
Standard_DS3_v2 (including ADLS I/O and MLflow logging overhead).

**One full sweep run across all 5 outcomes (sequential job submissions):**
- Total wall clock: ~1.5–2.5 hours
- Total node-hours consumed: ~12–15

**Anticipated iteration rounds over project lifetime:**

| Round | Purpose | Sweeps | Node-hours (est.) |
|---|---|---|---|
| Round 1 | Initial discovery — wide search space | 5 | 12–15 |
| Round 2 | Refine search space based on Round 1 results | 5 | 10–12 |
| Round 3 (optional) | Targeted narrow sweep around best region | 5 | 8–10 |
| **Total** | | **15** | **30–37** |

---

## 4. Budget Estimates

Prices are approximate Azure Pay-as-you-go, US East, Linux, 2025.
Verify current rates via the [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/).
AML workspace, experiments, and model registry have no separate charge — cost is VM + storage only.

### VM reference prices

| VM | vCPU | RAM | $/hr (PAYG, US East) |
|---|---|---|---|
| Standard_DS2_v2 | 2 | 7 GB | ~$0.14 |
| Standard_DS3_v2 | 4 | 14 GB | ~$0.28 |
| Standard_DS4_v2 | 8 | 28 GB | ~$0.57 |
| Standard_DS5_v2 | 16 | 56 GB | ~$1.14 |
| Standard_F4s_v2 | 4 | 8 GB | ~$0.20 |
| Standard_F8s_v2 | 8 | 16 GB | ~$0.40 |

### Compute instance cost

Assuming 8 hours/day active use, 15 working days for initial development build-out.
Instance must be stopped when not in use.

| VM | $/hr | Hours (8h × 15d) | Total |
|---|---|---|---|
| Standard_DS3_v2 | $0.28 | 120 | ~$34 |
| Standard_DS4_v2 | $0.57 | 120 | ~$68 |
| Standard_DS5_v2 | $1.14 | 120 | ~$137 |

### HyperDrive sweep cluster cost (3 rounds total)

Cluster scales to zero between jobs — charged only when trials are running.

| Scenario | VM per node | $/hr per node | Node-hours | Total |
|---|---|---|---|---|
| Budget | Standard_DS2_v2 | $0.14 | 37 | ~$5 |
| Recommended | Standard_DS3_v2 | $0.28 | 37 | ~$10 |
| Fast | Standard_F8s_v2 | $0.40 | 30 | ~$12 |

### Storage — ADLS Gen2 (LRS)

| Content | Est. size | $/GB/month | Monthly cost |
|---|---|---|---|
| Raw source data (01_* parquet outputs) | 20–50 GB | $0.023 | $0.46–$1.15 |
| Processed feature matrix | < 1 GB | $0.023 | < $0.03 |
| Models + SHAP (5 outcomes × 3 rounds) | < 1 GB | $0.023 | < $0.03 |
| **Total** | ~25–55 GB | | **~$0.50–$1.20/month** |

### All-in total (initial 3-round development cycle, ~3 months)

| Scenario | Instance | Cluster | Storage (3 mo.) | **Total** |
|---|---|---|---|---|
| Minimum (DS3 instance + DS2 cluster) | $34 | $5 | $4 | **~$43** |
| Recommended (DS4 instance + DS3 cluster) | $68 | $10 | $4 | **~$82** |
| Performance (DS5 instance + F8s cluster) | $137 | $12 | $4 | **~$153** |

---

## 5. Key Caveats

**Why costs are low.** The country-year panel for Africa is a small dataset. XGBoost
trains in seconds per trial. The dominant cost is human time, not compute.

**The main budget risk is a running compute instance.** A DS4 left on 24/7 for a week
costs ~$95. Set an auto-shutdown schedule: AML Studio → Compute → your instance →
Edit → Auto-shutdown.

**Data pull notebooks are not CPU-bound.** ACLED and GDELT notebooks are network-bound
(API calls, BigQuery reads). A DS2 instance is adequate for initial data pulls; upgrade
to DS4 only when running LASSO and spatial feature engineering.

**No GPU warranted for current scope.** XGBoost `tree_method="hist"` does not benefit
from GPU on a ~1,200-row dataset. If neural network baselines are added later (TabNet,
FFN), NC-series GPU instances would be appropriate.

**HyperDrive vs. notebook `RandomizedSearchCV`.** The notebook `03/02` contains its own
`RandomizedSearchCV(n_iter=40)` that runs on the compute instance. Use that for
interactive development. Use the HyperDrive sweep (`sweep_outcome.yml`) when you want
MLflow tracking, reproducible run records, and the ability to compare rounds.

**Reserved instances.** If the pipeline runs on a regular monthly schedule after initial
development, 1-year reserved pricing cuts the compute instance rate by ~40%. Not
worth committing to during initial build-out.
