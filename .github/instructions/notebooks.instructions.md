---
applyTo: "**/*.ipynb"
---
# Notebook Conventions

Rules specific to Jupyter / Synapse notebook cells in this repository.

## Cell Structure

- Every notebook must have a **master config cell** near the top that defines all
  blob paths, deployment names, thresholds, and tuning constants. Nothing else
  should hardcode those values.
- All imports must be in the top 1-2 cells; do not scatter imports throughout the notebook. 
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
- After `pip install` cells, include `notebookutils.session.restartPython()`
  (Synapse) or a comment directing the user to restart the kernel.

## Outputs and Logging

- Every major pipeline stage cell must print a one-line summary on success:
  `print(f"✓ Stage complete: {n} records written to {path}")`
- Use structured print blocks (not bare `print("")`) so log lines are
  parseable in Synapse run history.
- Do not leave exploratory `display()` / `print()` calls in production cells;
  wrap them in `if DEBUG:` guards using a `DEBUG = False` flag in the config cell.

## Notebook-Level Documentation

- The first cell must be a markdown cell with:
  - Title and subtitle (method names, runtime)
  - "What this notebook covers" numbered list
  - Run order relative to other notebooks in the pipeline
  - ADLS blob paths written and consumed (table format)
- Section headers use `##` (not `#`) so the notebook outline is navigable.
- Remove explainer/tutorial cells once the workflow is validated end-to-end;
  they add noise for production runs.

## Synapse-Specific

- Import `mssparkutils` inside a `try/except ImportError` block so the notebook
  can be tested locally without a Synapse session.
- All ADLS credential calls must go through
  `mssparkutils.credentials.getSecret(KEY_VAULT_URL, SECRET_NAME)`.
- Broadcast-heavy Spark operations (e.g., `mapInPandas` UDFs) must use
  `spark.sparkContext.broadcast()` for large shared objects; do not close
  over Python module-level state directly.
