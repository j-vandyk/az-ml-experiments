# Data Sources and Predictors — Reference

Consolidated from:
- `archive/DatasetVariableSynthesis.md` — full codebook for 10 core political-science sources, dependent-variable taxonomy, and the variable-usage matrix across 10 prior studies
- `archive/predictor-catalog-extended.md` — 13 aspirational predictor domains beyond the core toolkit (satellite, financial markets, food security, displacement, digital, environment, etc.)
- `archive/labeled-data-sources-and-methods.md` — labeling methods (curated sources, construction patterns, weak supervision)
- `archive/meta-plan-instability-classifiers.md` — original feature-domain plan (six domains × ~80 features)

This file is the canonical "what data do we have, what does it mean, and how are labels built" reference. Anything not covered here lives in `archive/` and should be promoted back if it becomes operational.

---

## 1. Implemented sources (notebooks 01–26)

Every source below is pulled by a numbered notebook in `notebooks/01_data_pull/` and joined onto the country-year panel in `02_feature_engineering/02_build_feature_matrix.ipynb`. All write parquet to ADLS Gen2 under `raw/<source>/<YYYYMMDD>/`.

### Core sources (notebooks 01–13)

| # | Source | Notebook | Cadence | Key fields | Role |
|---|---|---|---|---|---|
| 01 | ACLED (event data, monthly aggregate) | `01_pull_acled.ipynb` | Monthly → annual | event counts by type, fatalities, actors | Predictor + label material for `mass_unrest_onset` |
| 02 | World Bank WDI | `02_pull_wdi.ipynb` | Annual | gdp_per_capita_const_usd, gdp_growth_pct, inflation, unemployment, population, urbanization | Structural / economic predictors |
| 03 | World Bank WGI | `03_pull_wgi.ipynb` | Annual | voice_accountability, political_stability, gov_effectiveness, regulatory_quality, rule_of_law, control_of_corruption | Governance predictors |
| 04 | V-Dem | `04_pull_vdem.ipynb` | Annual | v2x_polyarchy, v2x_libdem, v2x_partipdem, v2x_egaldem, v2x_regime, v2x_civlib | Democracy / regime predictors and `regime_backsliding` label source |
| 05 | Polity5 | `05_pull_polity5.ipynb` | Annual | polity (composite), durable, exrec, exconst, parcomp, COUPTRY | Regime durability + alternative coup label |
| 06 | UCDP-GED (country-year aggregate) | `06_pull_ucdp_ged.ipynb` | Annual | battle_deaths, conflict_episodes, dyad_count, conflict_type | `civil_war_onset` label source (≥25 battle deaths threshold) |
| 07 | Powell-Thyne coups | `07_pull_powell_thyne.ipynb` | Event-level | coup attempts, success/failure, plotter type | `coup_attempt` label source |
| 08 | PITF (Political Instability Task Force) | `08_pull_pitf.ipynb` | Annual | adverse regime change, ethnic war, revolutionary war, genocide flags | Multi-type instability labels |
| 09 | FSI (Fragile States Index) | `09_pull_fsi.ipynb` | Annual | total + 12 sub-scores (cohesion, economic, political, social) | Composite fragility predictor |
| 09b | PRIO-GRID engineered features | `09b_engineer_priogrid.ipynb` | Annual | population-weighted aggregates of cell-level conflict, terrain, infrastructure | Spatial/structural predictors |
| 10 | UNHCR | `10_pull_unhcr.ipynb` | Annual | refugees_in, refugees_out, idps, asylum_seekers | Forced displacement |
| 11 | UNDP HDI | `11_pull_undp_hdi.ipynb` | Annual | hdi, life_expectancy, mean_years_schooling, gni_per_capita | Human development |
| 12 | GDELT | `12_pull_gdelt.ipynb` | Daily → annual | event counts by CAMEO root code, average tone, Goldstein scale aggregates | High-frequency event signals |
| 13 | FAO (FFPI + country CPI) | `13_pull_fao.ipynb` | Monthly → annual | food_price_index, country-level CPI for cereals/oils/sugar | Food-price shocks |

### Supplementary sources (notebooks 14–19)

| # | Source | Notebook | Adds | Label outcomes |
|---|---|---|---|---|
| 14 | RSUI (IMF Reported Social Unrest Index) | `14_pull_rsui.ipynb` | rsui_mean/max/std, sub-component aggregates | `unrest_binary` |
| 15 | Freedom House | `15_pull_freedom_house.ipynb` | fh_pr, fh_cl, fh_total, fh_status_ord | `fh_status_decline`, `fh_pr_decline` |
| 16 | EPR-Core | `16_pull_epr.ipynb` | n_excluded_groups, excluded_pop_share, dominant_group_flag | `ethnic_exclusion_any`, `epr_state_collapse` |
| 17 | Political Terror Scale | `17_pull_pts.ipynb` | pts_a, pts_s, pts_h, pts_mean, pts_max | `pts_high`, `pts_escalation` |
| 18 | V-Dem ERT (autocratization episodes) | `18_pull_vdem_ert.ipynb` | aut_ep_duration, dem_ep, edi, edi_change_3y | `aut_ep` |
| 19 | Cline coup (event + country-year) | `19_pull_cline_coup.ipynb` | cline_* coup-attempt features | Cross-source coup validation |

### New supplementary sources (notebooks 20–26)

| # | Source | Notebook | Adds |
|---|---|---|---|
| 20 | WBI (Worldwide Bureaucracy Indicators) | `20_pull_wbi.ipynb` | wage_bill_gdp_pct, public_employment_pct, pub_priv_wage_ratio |
| 21 | ICEWS (event data) | `21_pull_icews.ipynb` | CAMEO event counts, avg Goldstein, conflict/cooperation balance |
| 22 | WEO (IMF World Economic Outlook forecasts) | `22_pull_weo.ipynb` | weo_gdp_growth_pct, weo_inflation_*, weo_current_account_pct (prefixed to avoid WDI collision) |
| 23 | IDMC (internal displacement) | `23_pull_idmc.ipynb` | conflict_/disaster_new_displacements, stock_displacement |
| 24 | BTI (Bertelsmann Transformation Index) | `24_pull_bti.ipynb` | governance/political/economic transformation scores, months_since_bti decay |
| 25 | V-Party | `25_pull_vparty.ipynb` | gov_populism_mean/max, any_populist_gov, opp_populism_max |
| 26 | ND-GAIN | `26_pull_ndgain.ipynb` | vulnerability, readiness, ndgain_score |

**Pipeline contract:** every source-pull notebook must write a parquet keyed by `(iso3, year)` (or be aggregated into one). The hardened `_join_iso3_source` helper in `02/02` will derive `iso3` from `country_name`/`country` via `name_to_iso3` if missing, but `year` is required. Use `feature_prefix=` if column names will collide with another source (the WEO/WDI overlap is the canonical case).

**Vintage / look-ahead policy (WEO).** IMF WEO (source 22) is handled with full vintage-keyed joins to prevent look-ahead leakage:

- **Storage:** `22_pull_weo_forecasts.ipynb` writes to `raw/weo/vintage={YYYY}-{MM}/weo_panel.parquet` (e.g. `vintage=2025-10`). The vintage is inferred from the pull date: October release if `month >= 10`, April release if `month >= 4`, otherwise previous October. Each row carries a `weo_vintage` column and an `is_projection` flag (`1` if `year >= VINTAGE_YEAR`).
- **Join:** `_join_weo_vintage_keyed()` in `02/02` performs a time-valid join: for each panel year *t*, it selects the latest vintage where `vintage_year ≤ t` and joins only the `year == t` row from that vintage. No future projections (`year > t`) ever enter the panel. Years that predate all stored vintages receive NaN — the diagnostic log names which vintage was used per year.
- **Backtesting integrity:** to eliminate vintage-drift bias for historical years, backfill old WEO vintage files from the IMF WEO archive (`https://www.imf.org/en/Publications/WEO/weo-database`) and re-run notebook 22 for each historical vintage. Until backfill is done, all panel years will use the earliest available vintage, which is better than the latest (less look-ahead).

---

## 2. Dependent variables (12 outcomes)

The model trains a separate XGBoost classifier per outcome. All are binary at country-year resolution. Onset definitions use a peace-spell filter: `label_t = 1 if event_t ≥ threshold AND no event in [t-k, t-1]`, with `k` chosen per outcome.

### Core outcomes

| Outcome | Source | Definition | Approx. base rate | Random AUPRC |
|---|---|---|---|---|
| `civil_war_onset` | UCDP-GED | ≥25 battle deaths in year, no qualifying year in prior 2 | ~0.3% | ~0.003 |
| `coup_attempt` | Powell-Thyne | Any coup attempt (success or failure), no attempt in prior 2 | ~0.2% | ~0.002 |
| `regime_backsliding` | V-Dem | `v2x_polyarchy` drops by >0.05 year-over-year | ~1.0% | ~0.010 |
| `mass_unrest_onset` | ACLED protests/riots | Country-year above 90th percentile of country's own training-period distribution | ~15% | ~0.150 |
| `humanitarian_crisis_onset` | FEWS NET / UNHCR (subset) | IPC Phase 3+ population surge or large refugee outflow | ~7% (FEWS countries) | ~0.070 |

### Supplementary outcomes (notebooks 14–18)

| Outcome | Source | Definition |
|---|---|---|
| `unrest_binary` | RSUI | Country-year flagged as unrest event in IMF index |
| `fh_status_decline` | Freedom House | Status drops one ordinal class year-over-year |
| `ethnic_exclusion_any` | EPR | Any politically excluded ethnic group present |
| `epr_state_collapse` | EPR | State-collapse flag set |
| `pts_high` | PTS | Mean PTS score ≥ 4 |
| `pts_escalation` | PTS | PTS score increases by ≥1 from prior year |
| `aut_ep` | V-Dem ERT | Autocratization episode active |

### Threshold choice is theory, not bookkeeping

Every label embeds substantive claims:
- UCDP's 25-battle-death threshold defines what counts as "civil war" — moving it to 100 changes both base rate and which countries enter the positive class.
- The peace-spell window `k=2` says "a new civil war within 2 years of the last one is a continuation, not a new onset." Mueller & Rauh's "hard onset" (k=5) cuts AUPRC nearly in half.
- The 90th-percentile rule for `mass_unrest_onset` is country-specific; absolute thresholds would be dominated by a few high-event countries.

Document the threshold and peace-spell rule with each label. Sensitivity to `k` is a real result, not a tuning artifact.

---

## 3. Independent variables — taxonomy by domain

The current pipeline organises predictors into eight domains. Each domain is sourced from one or more of the implemented notebooks (§1). After feature engineering (notebook `02/03`) the count grows by ~30–60 derived columns.

### 3.1 Regime / governance
**Sources:** V-Dem (04), Polity5 (05), WGI (03), V-Party (25), BTI (24), Freedom House (15)
**Variables:** `v2x_polyarchy`, `v2x_libdem`, `polity`, `durable`, governance scores (six WGI pillars), populism indices, BTI political-transformation score
**Lag:** structural — use level and 3-year change

### 3.2 Economic
**Sources:** WDI (02), WEO (22), UNDP HDI (11)
**Variables:** `gdp_per_capita_const_usd`, `gdp_growth_pct`, `inflation_pct`, `unemployment_pct`, `weo_*` forecasts
**Lag:** annual; engineered HP-filter trend/cycle decomposition (λ=6.25)

### 3.3 Demographic / human development
**Sources:** WDI (02), UNDP HDI (11)
**Variables:** `population_total`, `urban_population_pct`, `youth_bulge_pct`, `life_expectancy`, `mean_years_schooling`
**Lag:** very slow-moving; first-difference rarely informative

### 3.4 Conflict event history
**Sources:** ACLED (01), UCDP-GED (06), GDELT (12), ICEWS (21)
**Variables:** event counts by type, fatalities, average Goldstein, count of distinct dyads
**Lag:** monthly aggregates → annual mean/max/std; use 3-year rolling history

### 3.5 Displacement / humanitarian
**Sources:** UNHCR (10), IDMC (23)
**Variables:** `refugees_in/out`, `idps`, `conflict_new_displacements`, `disaster_new_displacements`
**Lag:** annual stocks + flows; flows are leading indicators

### 3.6 Spatial / structural
**Sources:** PRIO-GRID engineered (09b), spatial-spillover features from `02/03` Section C
**Variables:** population-weighted cell aggregates, KNN-neighbor weights, local Moran's I, LISA quadrants
**Requires:** `geo_coords_path` CSV with `iso3, lat, lon` (Section C skips gracefully if absent)

### 3.7 Coup / leader history
**Sources:** Powell-Thyne (07), Polity5 (05), Cline coup (19), Archigos (CY aggregate)
**Variables:** prior coup count (5-year rolling), years_since_last_coup, leader-tenure, irregular-removal flag

### 3.8 Composite fragility / repression
**Sources:** FSI (09), PTS (17), EPR (16), V-Dem ERT (18), ND-GAIN (26)
**Variables:** FSI total + 12 sub-scores, PTS mean/max, ethnic exclusion share, ND-GAIN vulnerability/readiness

### Engineered columns (notebook `02/03`)

Section A — `log1p`, `sqrt`, first differences (`_diff1`), HP-filter trend/cycle.
Section B — Configured pairwise interactions (z-scored), e.g. anocracy × economic contraction.
Section C — Spatial spillover (Moran's I, LISA cluster indicators).
Section D — Audit: NZV flags, missingness, point-biserial correlation against each of the 12 outcome labels.

---

## 4. Interaction effects worth engineering

These are the documented mechanisms most often called out in the prior studies catalogued in `archive/predictor-catalog-extended.md`. They become Section B interactions in the engineered feature matrix.

| Interaction | Mechanism |
|---|---|
| `anocracy × gdp_contraction` | Goldstone et al. (2010) — partial regimes most vulnerable to economic shocks |
| `food_price_shock × urban_pct` | Bellemare (2015) — urban dependence on imported staples |
| `refugee_outflow × neighbor_conflict` | Salehyan & Gleditsch — spillover from conflict next door |
| `youth_bulge × unemployment` | Cincotta — demographic stress amplified by labour-market failure |
| `prior_coup × low_durable` | Coup recidivism in low-tenure regimes |
| `ethnic_exclusion × resource_dependence` | Resource curse conditioned on horizontal inequality |
| `pts × press_freedom_change` | Repression escalation under closing civic space |
| `acled_intensity × spatial_lag_acled` | Local conflict + regional contagion |
| `inflation × fx_depreciation` | Joint macro stress signal |
| `aut_ep_active × upcoming_election` | Backsliding episodes intensify around elections |

---

## 5. Constructing labels for new outcomes

When no curated dataset exists, these are the established patterns. (Detailed examples in `archive/labeled-data-sources-and-methods.md`.)

### Onset with peace-spell filter
`label_t = 1 if event_t ≥ threshold AND no event in [t-k, t-1]`. Used for `civil_war_onset`, `coup_attempt`. Choose `k` to encode "new vs. continuation"; `k=2` is conventional for armed conflict.

### Delta / deterioration labels
`label_t = 1 if indicator_t − indicator_{t-1} < −threshold`. Used for `regime_backsliding`. Captures change events rather than levels — appropriate for slow-moving indices (V-Dem, FSI, Fariss).

### Country-relative percentile
`label_t = 1 if events_t > q90(events_{country, training_years})`. Used for `mass_unrest_onset`. Avoids global-absolute thresholds that would be dominated by a few high-event countries.

### Cross-source triangulation
`label_t = 1 if source_A AND source_B`. Reduces false positives at the cost of recall. Appropriate for high-stakes downstream decisions.

### Weak supervision (Snorkel-style)
3–6 labeling functions, each returning {1, −1, 0}. A label model resolves conflicts and produces P(label=1). Useful when iterating on a novel outcome definition without hand-coding rows. Soft labels can pass to XGBoost via `sample_weight`.

### LLM extraction from news
GDELT URLs → fetch text → structured-extraction prompt → aggregate to country-year. Useful for outcomes that don't fit existing coding schemes ("state capture", "judicial-independence collapse"). Always validate against a hand-coded gold sample.

### Documentation checklist for any new label

| Item | Why |
|---|---|
| Base rate (positives / total) | Determines `scale_pos_weight`; <0.1% makes AUPRC the only meaningful metric |
| Coverage (which countries, which years) | Determines whether missing → `0` (assume no event) or `NA` (unknown) |
| Onset definition (threshold + peace-spell window) | Documents reproducibility; small changes have large effects on rare-event balance |
| Source-agreement rate (if cross-sourced) | The disagreement rows are the most informative cases |

---

## 6. Pointer — extended (not-yet-implemented) catalog

The 13 aspirational predictor domains that go beyond the current pipeline live in `archive/predictor-catalog-extended.md`. Headlines worth knowing about:

- **Satellite imagery** — VIIRS nightlights month-over-month change as economic-shock proxy; Sentinel-1 SAR for building destruction; FIRMS for arson/conflict signals
- **Financial markets** — sovereign bond spreads, currency-crash flags, IMF-program activations (Bloomberg-dependent series excluded for cost)
- **Food security** — FEWS NET IPC Phase 3+ population (immediate unrest trigger), pastoral terms-of-trade (Sahel-specific)
- **Digital** — NetBlocks shutdown frequency, Tor-usage spikes, Freedom on the Net scores
- **Civil society** — NAVCO 3.5% participation threshold, MMP government-response taxonomy, Carnegie protest tracker
- **Africa-specific** — IIAG security sub-score (often outperforms WGI for Africa), Afrobarometer trust deficits

Promote a domain back into this file (and into a new `01_data_pull/` notebook) once it's implemented.
