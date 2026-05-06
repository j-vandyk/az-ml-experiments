---
applyTo: "**"
---
# GitHub Copilot Instructions — az-ml-experiments

## Project Context

This repository predicts national-level political and humanitarian instability
events from a country-year panel dataset. The pipeline ingests data from nineteen
international sources (World Bank WDI/WGI, V-Dem, ACLED, UCDP, FAO, etc.),
builds a unified feature matrix with PRIO-GRID spatial features, selects features
via LASSO + mutual-information rescue, trains one XGBoost classifier per outcome,
and produces SHAP-based explainability at model and observation level.

All compute runs on **Azure Machine Learning**. Experiment tracking, model
registration, and artifact storage use **MLflow** integrated with the AML workspace.
Raw and processed data live in **Azure Data Lake Storage Gen2**.

Algorithms are frequently drawn from recent arXiv papers — cite the arXiv ID in
docstrings when implementing paper-derived logic. Primary runtime: Python 3.10,
AML-managed compute, and AML-managed environments.

---

## Python Standards

- Follow **PEP 8** (max line length 120 for notebook cells, 88 elsewhere)
- Use **type hints** on every function signature; use `Optional[X]` not `X | None`
  (Python 3.9 compat — `X | Y` union syntax requires 3.10+)
- Prefer **dataclasses** over plain dicts for structured data passed between functions
- Every UDF must have a **docstring** with: purpose, args, returns, and one usage example
- Use **early returns** to handle error/empty cases before the main logic
- Validate inputs at function boundaries; raise `ValueError` with descriptive messages

## UDF Design Principles

- **Single responsibility**: each function does one thing
- **No side effects by default**: UDFs return values; they do not mutate shared state
- **Name verbs clearly**: `build_feature_matrix()` not `process()` or `run()`
- **Keep UDFs short**: if a function exceeds ~40 lines, decompose it
- Prefer **keyword arguments** for any function with more than 2 parameters
- Do not add error handling for scenarios that cannot happen; only validate at boundaries

## Comments and Documentation

- Comment **why**, not **what** — assume the reader knows Python
- For complex algorithmic sections, add a plain-English summary before the block
- Use `# ──` section dividers inside long cells/functions to delineate logical passes
- Do not add docstrings, comments, or type annotations to code you did not change

---

## Data Conventions

- Raw source data lives in ADLS; access via `adlfs` / `fsspec` using the
  `abfs://` protocol — not `wasbs://` and not local file paths
- Prefer **Parquet** for intermediate feature matrices; **CSV** for human-readable
  exports and small reference tables
- Use `pandas` for feature engineering; `scikit-learn` pipelines for transforms
- **Never hardcode credentials** — use `azure-identity` `DefaultAzureCredential`
  (resolves to managed identity on AML compute) or `python-dotenv` locally
- Secrets (API keys for ACLED, UCDP, etc.) must come from Azure Key Vault;
  retrieve via `SecretClient` not environment variable literals

---

## Additional Context Files

When working in **agent mode**, read the following files for project-specific plans,
active work items, and resource references **before proposing changes**:

- `.claude/meta-plan-instability-classifiers.md` — overall modeling strategy and
  experiment sequencing
- `.claude/project_scope_requirements.md` — outcome definitions, scope constraints
- `.claude/predictor-catalog-extended.md` — full feature list with sources and
  transformation notes
- `.claude/azure-ml-usage-plan.md` — AML workspace layout, compute targets,
  environment names, and job submission patterns
- `.claude/labeled-data-sources-and-methods.md` — data source details and
  known quality issues

Do **not** modify files under `.claude/` directly unless explicitly instructed;
propose the change and explain the downstream impact first.

> Scoped rules for algorithms, notebooks, and Azure ML patterns live in
> `.github/instructions/` and are applied automatically by file-path pattern.
