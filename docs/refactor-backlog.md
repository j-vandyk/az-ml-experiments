# Refactor backlog

Engineering follow-ups that are **known but deliberately deferred** so individual
PRs stay reviewable. Each item documents the issue, why it's deferred, and what
"done" looks like.

Keep entries small and self-contained. When you tackle one, delete it from this
file (the git history is the audit trail).

---

## 1. Align `ENG_CFG` column references with the actual feature matrix

**Where:** `notebooks/02_feature_engineering/03_engineer_derived_features.ipynb`,
`ENG_CFG` cell (`log_cols`, `sqrt_cols`, `diff_cols`, `hp_cols`,
`interaction_pairs`, `spatial_features`).

**Problem.** Many configured column names don't match what `02/02` actually
produces, so the corresponding transforms / interactions / spatial lags are
silently no-op'd. Concrete mismatches identified during the May 2026 audit:

| `ENG_CFG` reference | Actual panel column(s) | Why it drifts |
|---|---|---|
| `gdp_per_capita` | `gdp_per_capita_const_usd`, `gdp_per_capita_ppp` | WDI's friendly-name map (`WDI_INDICATORS` in `01/02`) |
| `population` | `population_total` | WDI |
| `gdp_total` | (not pulled) | not in `WDI_INDICATORS` |
| `inflation_cpi` | `inflation_cpi_pct` | WDI |
| `military_expenditure_usd` | `sipri_*` (SIPRI gets prefix `sipri`) | `_join_*_source` prefixes |
| `polity_score` | `polity`, `polity2` (Polity5 is unprefixed) | Polity5 raw column names |
| `archigos_tenure_years` / `archigos_irreg_entry` | `arch_tenure_years` / `arch_irreg_entry` | Archigos joined with prefix `arch` |
| `fao_food_price_index` | `ffpi_food_price_idx` (or similar) | FFPI broadcast block prefixes with `ffpi_` |
| `hist_conflict_years` | (not produced anywhere) | needs to be derived |
| `ethnic_fractionalization` | check V-Dem actuals | possibly under a different V-Dem variable |

**Why deferred.** The cheap fix (rename strings in `ENG_CFG`) is one decision per
column — what *do* you want, `gdp_per_capita_const_usd` or the PPP version? — and
needs a quick check against the live panel. The recently added pre-flight audit
in `03_engineer_derived_features.ipynb` will print every missing reference at
notebook startup, so this can be done in one sitting after the next matrix build.

**Done when:** running `03_engineer_derived_features` prints
`ENG_CFG column references all resolve against the feature matrix.` (the
"missing" warning block is empty).

---

## 2. Consolidate the monthly-source aggregation blocks (ACLED / GDELT / ICEWS)

**Where:** `notebooks/02_feature_engineering/02_build_feature_matrix.ipynb`,
the "Aggregate monthly sources to annual and join" cell (cell id `71087dec`).

**Problem.** Three near-duplicate scaffolds, ~110 lines total, that all do:
identify the country identifier → resolve it to ISO3 → derive `year` from
`year_month` → pick numeric columns → group `(iso3, year)` → sum the count-like
columns and mean+std the rest → merge onto the panel. They differ in only two
ways: (1) ID-resolution method, and (2) the rule for "this is a count column."

**Risks of leaving as-is.**
- Drift: a fix applied to one block (e.g., the `is_numeric_dtype` check) doesn't
  propagate to the others. Today GDELT and ICEWS use the older
  `c.dtype in ("float64", "int64", "Int64")` pattern, which silently misses
  `float32`, `Int32`, etc.
- Adding a fourth monthly source means another ~40 lines of boilerplate.

**Suggested shape.**
```python
def _aggregate_monthly_to_annual(prefix_key, source_label, *,
                                 resolve_iso3,           # callable(df) -> df with iso3
                                 count_predicate=None):  # callable(col_name) -> bool
    raw = read_latest_parquet(RAW_PREFIXES[prefix_key])
    if raw is None: return None
    raw.columns = [c.lower().strip() for c in raw.columns]
    raw = resolve_iso3(raw)
    if "year" not in raw.columns:
        raw["year"] = pd.to_numeric(raw["year_month"].str[:4], errors="coerce")
    numeric = [c for c in raw.columns
               if c not in ("iso3", "year", "year_month")
               and pd.api.types.is_numeric_dtype(raw[c])]
    is_count = count_predicate or (lambda c: "count" in c or c.startswith("events_"))
    spec = {}
    for c in numeric:
        if is_count(c):
            spec[f"{source_label}_{c}_annual"] = (c, "sum")
        else:
            spec[f"{source_label}_{c}_mean"] = (c, "mean")
            spec[f"{source_label}_{c}_std"]  = (c, "std")
    return (raw[["iso3", "year"] + numeric]
            .dropna(subset=["iso3", "year"])
            .groupby(["iso3", "year"], as_index=False).agg(**spec))
```
…then three short call sites with source-specific `resolve_iso3` lambdas.

**Why deferred.** Pure refactor with non-trivial blast radius (changes the
column inventory the FE / model notebooks see) — best done as its own commit so
a column-by-column diff against a reference run is reviewable.

**Done when:** the three monthly blocks are reduced to (helper + 3 call sites)
and a sample run produces the same `agg.columns` set as before.

---

## 3. Vectorise Section C spatial spillover

**Where:** `notebooks/02_feature_engineering/03_engineer_derived_features.ipynb`,
`run_spatial(...)` (cell id `73d861a2`).

**Problem.** For each (year × spatial feature), the per-row mapping of spatial
lag / Moran's I / LISA quadrant back into the panel uses a Python loop with
`df.at[idx, country_col]` and `series.at[idx] = …` — i.e. `O(n_rows × n_years
× n_features)` scalar `.at` calls. For ~4k rows × ~25 years × ~7 features that's
hundreds of thousands of scalar ops; profiling will show this is the dominant
cost of `03_engineer_derived_features`.

**Suggested shape.** Build per-year results as small `(iso3, slag, moran, quad)`
DataFrames, concatenate them, and merge once on `(iso3, year)`:
```python
yearly_frames = []
for yr in years:
    ...  # compute slag_vec, moran_vec, quad_vec
    yearly_frames.append(pd.DataFrame({
        "iso3": ordered_countries,
        "year": yr,
        slag_col: slag_vec,
        moran_col: moran_vec,
        quad_col: quad_vec,
    }))
out = pd.concat(yearly_frames, ignore_index=True)
df = df.merge(out, on=["iso3", "year"], how="left")
```
Two merges per spatial feature instead of N scalar assignments per row.

**Why deferred.** Correctness, not perf, was the priority for the May 2026
audit. The current code is slow but right; refactoring it is a separate change
that should be benchmarked against the existing output (LISA quadrants from
`esda.Moran_Local` need to round-trip identically).

**Done when:** Section C runs without per-row `.at` assignment; output of
`spatial_features × {_slag, _local_moran, _lisa_quad}` matches the previous
implementation row-for-row on a fixed input.

---

## How to use this file

- Add new entries under their own `## N. Title` heading; renumber if needed.
- Each entry: **Where**, **Problem**, **Why deferred**, **Done when**.
- Delete the entry in the same PR that resolves it; don't strike it out.

---
