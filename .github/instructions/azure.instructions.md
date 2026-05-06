---
applyTo: "infra/**,**/pipeline*,**/storage*,**/*azure*,**/*adls*,**/*aml*,jobs/**,setup/**"
---
# Azure ML Patterns

Rules for any code that interacts with Azure services in this repository.

## Credential Handling

- **Never hardcode** API keys, connection strings, SAS tokens, or passwords.
- On AML compute: use `DefaultAzureCredential()` — it resolves automatically to
  the compute instance's managed identity.
- Locally: load from `.env` via `python-dotenv`; `.env` is gitignored.
- External API keys (ACLED, UCDP, FAO, V-Dem) must be stored in Azure Key Vault
  and retrieved via `SecretClient`:
  ```python
  from azure.keyvault.secrets import SecretClient
  client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)
  secret = client.get_secret(SECRET_NAME).value
  ```
  Never read these from environment variable literals in committed code.

## ADLS Gen2 / Data Access

- Access ADLS via `adlfs` / `fsspec` using the `abfs://` protocol:
  ```python
  import adlfs
  fs = adlfs.AzureBlobFileSystem(account_name=STORAGE_ACCOUNT, credential=credential)
  df = pd.read_parquet("abfs://container/path/file.parquet", filesystem=fs)
  ```
- Do **not** use `wasbs://` paths or `BlobServiceClient` for tabular reads —
  `adlfs` integrates cleanly with pandas/fsspec and avoids an extra dependency layer.
- Respect the medallion layer discipline:
  | Operation | Layer |
  |-----------|-------|
  | Read raw source data | `bronze/` |
  | Write cleaned / joined feature matrices | `silver/` |
  | Write validated model inputs / outputs | `gold/` |
- Prefix all gold-layer filenames with a UTC timestamp:
  `f"{prefix}/{artifact_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"`

## Azure ML SDK (azure-ai-ml)

- Use the `azure-ai-ml` SDK (v2) exclusively; do not introduce `azureml-core`
  (v1) patterns unless a feature has no v2 equivalent.
- Obtain the `MLClient` once at the notebook/script top level; pass it down
  rather than re-instantiating per function.
- All training jobs must be submitted via `ml_client.jobs.create_or_update()`
  using a YAML job definition in `jobs/`; do not run training inline in notebooks
  for any job that takes more than a few minutes.
- Environment definitions live in `setup/`; do not inline conda/pip specs inside
  job submission code.

## MLflow Experiment Tracking

- Every training run must log at minimum:
  - All hyperparameters (`mlflow.log_param`)
  - Primary evaluation metrics per outcome (`mlflow.log_metric`)
  - SHAP summary plot as an artifact (`mlflow.log_artifact`)
  - The feature list used (`mlflow.log_dict` or `log_artifact`)
- Use `mlflow.set_experiment(EXPERIMENT_NAME)` at the top of every training
  notebook/script so runs are grouped correctly in AML Studio.
- Register models via `mlflow.register_model()` only after validation passes;
  do not register prototype/exploratory runs.

## Infrastructure as Code

- Bicep / ARM templates live in `infra/`; do not inline resource definitions in
  pipeline scripts or notebooks.
- All resource names must follow the naming convention established in `infra/`
  and documented in `.claude/azure-ml-usage-plan.md`.
