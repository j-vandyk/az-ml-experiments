> **Archived.** Label-construction patterns (peace-spell onset, deterioration deltas, country-relative percentile, weak supervision, LLM extraction). Merged into `../data-and-predictors.md` §5. Kept here for the longer worked examples (Snorkel labeling functions, GDELT → LLM pipeline).

---

# Labeled Data Sources and Construction Methods
## Country-Year Instability Prediction

---

## Layer 1 — Existing Curated Country-Year Label Sources

These datasets deliver a binary or ordinal country-year label maintained by researchers with documented coding rules.

### Conflict and Violence

| Dataset | Outcomes Available | Coverage | Notes |
|---|---|---|---|
| UCDP Dyadic/PRIO Armed Conflict | Civil war onset, interstate war, ethnic conflict, conflict termination | 1946–present | Already in pipeline via GED; the dyadic dataset gives richer conflict-type disaggregation |
| Global Terrorism Database (GTD) | Terrorism event onset, surge | 1970–present | Threshold on annual event count per country |
| UCDP One-Sided Violence | State or non-state mass killing | 1989–present | Distinct from armed conflict — targets civilians |
| Mass Mobilization Project (Clark & Regan) | Government–opposition mobilization, state repression responses | 1990–2020 | Country-year and country-month available |

### Coups and Leader Instability

| Dataset | Outcomes Available | Notes |
|---|---|---|
| REIGN (Rulers, Elections, Irregular Governance) | Coup attempts, irregular leader removal, government type transition | Monthly, converts cleanly to country-year; more recent updates than Powell-Thyne |
| CSP Polity5 coup codes | Coup attempts, successful coups | Already partially covered via Archigos; Polity includes `COUPTRY` variable |

### Human Rights and Repression

| Dataset | Outcomes Available | Notes |
|---|---|---|
| Fariss Human Rights Scores | Continuous latent score; threshold for "repression onset" | Latent variable model — explicitly models measurement uncertainty; better than CIRI ordinal scale |
| Political Terror Scale (PTS) | 5-point ordinal repression scale | Coded from Amnesty and State Dept reports; can threshold to binary |

### Economic Crises

| Dataset | Outcomes Available | Notes |
|---|---|---|
| Reinhart-Rogoff (Ilzetzki et al.) | Banking crisis, currency crisis, sovereign default, inflation crisis | 1800–present; clean binary country-year labels with onset/end dates |
| Laeven-Valencia (IMF) | Banking crisis systemic onset | Cross-validates Reinhart-Rogoff; includes fiscal cost measures |

### Elections and Democratic Erosion

| Dataset | Outcomes Available | Notes |
|---|---|---|
| NELDA | Electoral fraud, opposition boycotts, violence around elections | Already in pipeline |
| Electoral Integrity Project (EIP) | Election quality scores | Can threshold — low quality election = label positive |
| Varieties of Democracy (V-Dem) | Electoral democracy backsliding, executive takeover, autocratization | `v2x_regime` transitions already used; `v2x_autocratization` is a direct label |

---

## Layer 2 — Construction Methods for Labeling from Raw Event Data

When no curated label exists, these are the established approaches.

### Onset with Peace-Spell Filter

The same pattern used for `civil_war_onset` generalises to any outcome:

```
label_t = 1  if  event_t >= threshold  AND  no event in [t-k, t-1]
```

The peace-spell window `k` is a design parameter encoding "what counts as a new crisis vs. continuation." `k=2` years is common for armed conflict. This prevents the model from being dominated by ongoing-conflict rows.

### Delta / Deterioration Labels from Continuous Indicators

For slow-moving indices (V-Dem, FSI, Fariss), create a label that fires on meaningful year-over-year change rather than level:

```
label_t = 1  if  indicator_t - indicator_{t-1} < -threshold
```

Conceptually cleaner than level-based labels because it captures *deterioration events* — more analogous to onset than to state. Already used for `regime_backsliding`.

### Percentile-Exceedance Labels

For count-based outcomes where base rates vary dramatically across countries:

```
label_t = 1  if  events_t > q90(events_{country, training_years})
```

Already used for `mass_unrest_onset`. The percentile is country-specific, making the label measure *relative to that country's history* rather than global absolute counts. Important for highly heterogeneous outcomes like protest events.

### Cross-Source Triangulation

For disputed or ambiguous events, require concordance across two independent sources:

```
label_t = 1  if  source_A codes event  AND  source_B codes event
```

Reduces false positives at the cost of increased false negatives. Appropriate when the downstream cost of wrongly predicting instability is high (e.g., humanitarian resource allocation).

---

## Layer 3 — Programmatic Labeling for Novel Outcomes

When no curated dataset exists and you want to define a new outcome, two approaches work at country-year resolution.

### Weak Supervision (Snorkel-Style Labeling Functions)

Write 3–6 imperfect labeling functions (LFs), each coding a proxy signal. A label model combines them into a probabilistic label, handling conflicts and abstentions.

Example for a hypothetical `economic_instability_onset` outcome:

```python
# Each LF returns: 1 (positive), -1 (negative), 0 (abstain)

def lf_gdp_contraction(row):
    return 1 if row["gdp_growth"] < -3.0 else 0

def lf_currency_crash(row):
    return 1 if row["fx_depreciation_pct"] > 20 else 0

def lf_imf_program(row):
    return 1 if row["imf_program_new"] == 1 else 0

def lf_acled_economic_protests(row):
    return 1 if row["acled_economic_protest_count"] > row["acled_q75"] else 0

def lf_sovereign_spread(row):
    return 1 if row["sovereign_spread_bp"] > 800 else 0
```

The label model learns which LFs are reliable, their correlation structure, and produces `P(label=1 | LFs)`. You can threshold that probability at 0.5 for a binary label, or pass soft labels directly to XGBoost via `sample_weight`. The key advantage: you can iterate on the outcome definition without any hand-coding of individual rows.

### LLM-Based Extraction from News Archives

GDELT publishes article URLs with country-event codes. For a novel outcome, the pipeline is:

1. Filter GDELT to country + relevant CAMEO event codes (e.g., CAMEO 17x for "coerce")
2. Fetch article text from GDELT URLs
3. Prompt an LLM with a structured extraction schema: *"Does this article describe [specific phenomenon] in [country] during [year]? Answer yes/no with confidence."*
4. Aggregate article-level extractions to country-year (majority vote or mean confidence)
5. Validate a sample against a hand-coded gold standard

This approach is useful for outcomes that don't fit existing coding schemes — for example, "state capture by business oligarchs" or "judicial independence collapse" — where no curated dataset exists but news coverage is dense.

---

## Documentation Checklist for Any New Label

Whatever method is used, record these four things before including a new outcome in the model:

| Item | Why It Matters |
|---|---|
| **Base rate** (positive country-years / total) | Determines `scale_pos_weight`; base rates below ~0.1% make AUPRC the only meaningful metric |
| **Coverage** (which countries, which years) | Determines whether to treat missing as `0` (assume no event) or `NA` (unknown); FEWS NET-style partial coverage needs the NA treatment |
| **Onset definition** (event threshold + peace-spell window) | Small changes have large effects on class balance with rare events; document the rule explicitly so it can be reproduced |
| **Source agreement rate** (if cross-source triangulated) | If two sources agree 85% of the time, the 15% disagreement rows are the hardest and most informative cases — worth inspecting manually |
