---
applyTo: "**/*.ipynb"
---
# Notebook Conventions

Rules specific to Jupyter notebook cells in this repository.
Notebooks run on AML compute instances (JupyterLab) or locally.

## Cell Structure

- Every notebook must have a **master config cell** near the top that defines all
  ADLS paths, AML workspace references, environment names, and tuning constants.
  Nothing else should hardcode those values.
- Use `# ╔══...══╗ / ║ ... ║ / ╚══...══╝` box headers to visually separate
  major sections within a long cell.
- Use `# ── Section label ──────` dividers between logical passes inside a cell.
- Limit code cells to one coherent task. If a cell exceeds ~60 lines, split it.

## Cell Ordering and Side Effects

- Cells must be runnable **top-to-bottom** without hidden state from prior
  partial runs. If a cell depends on outputs from a prior cell, add a guard:
  ```python
  if "variable_name" not in dir():
      raise RuntimeError("Run the previous cell first.")
  ```
- Do not define functions in the same cell where they are called for the
  first time in the main workflow — keep definitions and invocations separate.
- After `pip install` cells, add a comment directing the user to restart the
  kernel before continuing.

## Outputs and Logging

- Every major pipeline stage cell must print a one-line summary on success:
  `print(f"✓ Stage complete: {n} records written to {path}")`
- Log MLflow metrics and params with `mlflow.log_metric()` / `mlflow.log_param()`
  at the end of each training or evaluation cell — not scattered mid-cell.
- Do not leave exploratory `display()` / `print()` calls in production cells;
  wrap them in `if DEBUG:` guards using a `DEBUG = False` flag in the config cell.

## Notebook-Level Documentation

- The first cell must be a markdown cell with:
  - Title and subtitle (model type, data sources, runtime)
  - "What this notebook covers" numbered list
  - Run order relative to other notebooks in the pipeline
  - ADLS paths read and written (table format)
  - AML experiment name and registered model name (if applicable)
- Section headers use `##` (not `#`) so the notebook outline is navigable.
- Remove exploratory/tutorial cells once a workflow is validated; they add
  noise and make re-runs slower.

## AML-Specific

- Use `MLClient` from `azure.ai.ml` for all AML SDK operations; do not mix
  the legacy `azureml-core` SDK unless a specific feature requires it.
- Retrieve the AML workspace handle once at the top of the notebook via
  `MLClient.from_config()` (on AML compute) or `MLClient(credential, ...)` locally.
- Submit long-running training steps as AML jobs (`ml_client.jobs.create_or_update()`)
  rather than running them inline in a notebook cell.
- Always log the AML job name/URL after submission so runs are traceable:
  `print(f"Job submitted: {job.name} — {job.studio_url}")`
