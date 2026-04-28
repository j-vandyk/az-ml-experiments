# Metric Interpretation Guide — Instability Classifiers

Reference for understanding model outputs, avoiding overfitting, and iterating
effectively on the five XGBoost instability classifiers.

---

## 1. Start with your baseline — AUPRC is relative, not absolute

AUPRC for a random classifier equals the outcome's prevalence. A model achieving
AUPRC = 0.10 sounds weak, but for `civil_war_onset` it represents ~33× the random
baseline — that's a strong signal.

**Skill ratio** = AUPRC / prevalence. Clear 10× before trusting a model operationally.

| Outcome | Approx. prevalence | Random AUPRC baseline | Target (10× skill) | Good result |
|---|---|---|---|---|
| civil_war_onset | ~0.3% | ~0.003 | 0.03 | 0.08–0.15 |
| coup_attempt | ~0.2% | ~0.002 | 0.02 | 0.06–0.12 |
| regime_backsliding | ~1.0% | ~0.010 | 0.10 | 0.15–0.25 |
| mass_unrest_onset | ~15% | ~0.150 | 0.45 | 0.40–0.60 |
| humanitarian_crisis_onset | ~7% | ~0.070 | 0.21 | 0.20–0.35 |

**On AUROC:** at <1% prevalence a model can reach AUROC 0.85 while barely beating
random on precision. Use it as a sanity check, not the headline number for rare onset
outcomes. AUPRC is the primary metric for `civil_war_onset`, `coup_attempt`, and
`regime_backsliding`.

---

## 2. Reading the train / val / test gap

Without `train_auprc` logged, a large val→test drop is ambiguous. The code changes
in section 8 add it. Once you have all three, use this table:

| Pattern | Likely cause | Response |
|---|---|---|
| `train ≈ val >> test` | Overfitting to train+val period | More regularisation; lower `max_depth`; higher `min_child_weight`; reduce `n_estimators` |
| `train >> val ≈ test` | Overfitting to training split only | Same regularisation levers; check for data leakage |
| `train ≈ val ≈ test` (all moderate) | Genuine signal difficulty | Accept it; look at feature quality; more events data |
| `val >> test`, `train ≈ val` | Structural temporal shift post-2021 | Not a model problem — conflict dynamics changed; document and report |

A `train_auprc` of 0.70 and `val_auprc` of 0.18 means overfitting. Both at 0.18
means a feature/data problem — different fix entirely.

**Temporal shift vs. overfitting:** if all trials in a sweep show `val >> test` by a
consistent margin (~0.10+), and the gap doesn't shrink with more regularisation, it's
structural shift. Document it in the model card rather than chasing it with tuning.

---

## 3. Early stopping — the missing anti-overfitting lever

The current code passes `eval_set` to XGBoost but no `early_stopping_rounds`, so the
model always trains to the full `n_estimators` even after the validation curve plateaus.
Trees added after the plateau memorise noise.

**Fix:** add `early_stopping_rounds=30` to `.fit()` and log `best_iteration`.

Once early stopping is active, `n_estimators` in the hyperparameter search becomes
irrelevant — XGBoost stops itself. Consider removing it from the HyperDrive search
space and fixing it at 800. This frees one search axis for the parameters that actually
matter: `max_depth`, `learning_rate`, `min_child_weight`, `gamma`, `reg_alpha`.

**Signal to watch:** if `best_iteration` is consistently hitting the ceiling
(e.g., 299 out of 300), your `n_estimators` ceiling is too low or early stopping isn't
triggering — loosen `slack_factor` or raise the ceiling.

---

## 4. Calibration — reading the plots

The calibration plot compares predicted probability bins (x-axis) to observed positive
rate (y-axis). The diagonal = perfect calibration.

**Common patterns in rare-event models with `scale_pos_weight`:**

- **Curve below diagonal** (overconfident): model says P=0.20, but only 5% of those
  observations are actual positives. This is the norm with high `scale_pos_weight` —
  XGBoost inflates probabilities to compensate for class weighting.
- **Flat S-curve**: model discriminates well (ranks correctly) but probabilities aren't
  meaningful as absolute risk estimates.

**When it matters:**

- **Just rank ordering** (who's highest risk): calibration doesn't matter. The model
  is doing its job.
- **Communicating absolute probability** ("15% chance of civil war onset"): calibration
  matters. Add Platt scaling post-hoc on the validation set:

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
calibrated.fit(X_v, y_v)
```

**Brier score** is the single-number calibration summary (lower = better). The
naive constant-predictor baseline is `prevalence × (1 - prevalence)`. A model should
beat this — if it doesn't, its probability estimates are actively harmful.

---

## 5. Threshold selection

The default 0.5 threshold is wrong for all five outcomes. At extreme imbalance, almost
no country-year will exceed 0.5 even with a strong model.

**Two principled approaches** — choose by use case:

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_v, val_prob)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

# Option A: maximise F1 on validation set
thresh_f1 = thresholds[f1[:-1].argmax()]

# Option B: fix recall at 80% (catch most onset events, accept more false alarms)
idx_r80   = next(i for i, r in enumerate(recall) if r <= 0.80)
thresh_r80 = thresholds[idx_r80]
p_at_r80   = precision[idx_r80]
```

**Which to use per outcome:**

| Outcome | Recommended threshold strategy | Why |
|---|---|---|
| civil_war_onset | Recall-anchored (≥80%) | Missing an onset is the costly error |
| coup_attempt | Recall-anchored (≥80%) | Same — rare, high-stakes event |
| regime_backsliding | Either; F-beta (β=1.5) | Moderate frequency; both errors matter |
| mass_unrest_onset | F1-max | Common enough that precision matters |
| humanitarian_crisis_onset | Recall-anchored (≥75%) | FEWS use case prioritises coverage |

Store the chosen threshold per outcome in the model artifact so inference uses it
consistently.

---

## 6. Precision@K — the most actionable metric

An early warning system has finite analyst capacity. "Of the 20 country-years we flag
as highest-risk, how many are true positives?" is more operationally meaningful than
any threshold-based metric, and it's stable at extreme imbalance.

```python
k = 20
top_k_idx     = val_prob.argsort()[::-1][:k]
precision_at_k = float(y_v[top_k_idx].mean())
```

If P@20 = 0.40 for `civil_war_onset`, analysts reviewing the top 20 flagged countries
find 8 genuine cases. That's what to present to stakeholders — not an AUPRC value they
can't interpret.

Tune K to your actual analyst capacity. P@10 is more demanding; P@50 is more forgiving.

---

## 7. F1 — when it's useful and when it's not

F1 only makes sense when both precision and recall matter roughly equally AND your
threshold choice is well-motivated.

| Outcome | Is F1 useful? | Why |
|---|---|---|
| civil_war_onset | Not directly | Too rare; F1 dominated by threshold sensitivity |
| coup_attempt | Not directly | Same |
| regime_backsliding | Marginally | Can be used with F-beta (β=1.5, weighting recall) |
| mass_unrest_onset | Yes | ~15% prevalence; F1-max on val set is meaningful |
| humanitarian_crisis_onset | Marginally | FEWS subset; treat like regime_backsliding |

For the rare outcomes, use `val_f1_max` (the best F1 achievable at any threshold)
as a diagnostic upper bound, not an operational metric.

---

## 8. Code additions (priority order)

These three additions are the highest-leverage changes to the training pipeline.
Apply them to both `src/train/train_outcome.py` and `notebooks/03_model_development/02_train_xgboost_outcomes.ipynb`.

### 8a. Add `train_auprc` — enables overfitting diagnosis

```python
train_prob = model.predict_proba(X_tr)[:, 1]
metrics["train_auprc"] = _safe_auprc(y_tr, train_prob)
```

### 8b. Add `early_stopping_rounds=30` — prevents trees from memorising noise

```python
model.fit(
    X_tr, y_tr,
    eval_set=[(X_v, y_v)],
    verbose=False,
    early_stopping_rounds=30,
)
metrics["best_iteration"] = model.best_iteration
```

In the notebook (which uses `RandomizedSearchCV`), refit the best model manually
after the search so early stopping applies to the final artifact:

```python
best_params = search.best_params_
# Refit with early stopping — replaces search.best_estimator_
best_model = xgb.XGBClassifier(**XGB_BASE_PARAMS, scale_pos_weight=spw, **best_params)
best_model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], early_stopping_rounds=30, verbose=False)
```

### 8c. Add operational metrics — threshold-aware and precision@k

```python
from sklearn.metrics import precision_recall_curve, brier_score_loss

precision_c, recall_c, thresholds_c = precision_recall_curve(y_v, val_prob)
f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-9)

top20_idx = val_prob.argsort()[::-1][:20]
metrics["val_precision_at_20"] = float(y_v[top20_idx].mean())
metrics["val_f1_max"]          = float(f1_c[:-1].max())
metrics["val_thresh_f1"]       = float(thresholds_c[f1_c[:-1].argmax()])
metrics["val_brier"]           = float(brier_score_loss(y_v, val_prob))
metrics["test_brier"]          = float(brier_score_loss(y_te, test_prob))
```

---

## 9. Iterative sweep workflow

### After Round 1 — read the results

Pull all runs in MLflow Studio or via the SDK:

```python
import mlflow
runs = mlflow.search_runs(experiment_names=["instability_xgboost"])

for outcome in OUTCOME_COLS:
    s = runs[runs["tags.outcome"] == outcome]
    print(f"\n{outcome}")
    print(f"  val_auprc:   best={s['metrics.val_auprc'].max():.4f}  std={s['metrics.val_auprc'].std():.4f}")
    print(f"  test_auprc:  best={s['metrics.test_auprc'].max():.4f}")
    print(f"  train_auprc: best={s['metrics.train_auprc'].max():.4f}")
```

**What to look for:**
- High `val_auprc` variance → search space too wide; most of the space is bad
- Consistent `val >> test` gap → temporal shift; tighten regularisation but don't
  expect to close the gap fully
- `train >> val` by >0.20 → overfitting; add more regularisation in Round 2

### Narrowing for Round 2

Look at the top 10 trials by `val_auprc`. What do their hyperparameters cluster around?
Narrow the search space to those regions. 40 trials in a narrow space finds better
optima than 40 trials across the full space.

Example: if top trials cluster around `max_depth=4–5`, `learning_rate=0.01–0.05`,
`min_child_weight=5–10`, tighten the Round 2 distributions to those ranges.

### Signs to trust a result

- `test_auprc` within ~0.05 of `val_auprc`
- `train_auprc` within ~0.15 of `val_auprc`
- `val_brier` < prevalence × (1 - prevalence) (beats naive constant predictor)
- Calibration curve not wildly off the diagonal
- `best_iteration` well below `n_estimators` ceiling (early stopping fired)

### Signs to distrust a result

- All top trials show `val >> test` by >0.10 and it doesn't improve with regularisation → structural temporal shift, not a tuning problem
- `train_auprc >> val_auprc` by >0.20 → overfitting; increase `reg_alpha`, `gamma`, `min_child_weight`
- `best_iteration` always at ceiling → early stopping not firing; raise `n_estimators` or reduce `early_stopping_rounds`
- `val_brier` > prevalence × (1 - prevalence) → the model's probabilities are worse than just predicting the base rate
