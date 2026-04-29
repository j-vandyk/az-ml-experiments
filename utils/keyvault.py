"""
utils/keyvault.py — Key Vault credential helpers for the instability pipeline.

Reads secrets from Azure Key Vault using DefaultAzureCredential and populates
os.environ so that existing notebook config cells (which read from os.environ)
work without modification.

Authentication:
  - On AML compute: managed identity (automatic, no action needed)
  - Local dev: `az login` then run notebook

Required environment variable:
  KEY_VAULT_URL  — e.g. https://<vault-name>.vault.azure.net

Usage in notebooks:
  import sys; sys.path.insert(0, "../..")   # repo root
  from utils.keyvault import load_acled_credentials
  load_acled_credentials()
"""

from __future__ import annotations

import json
import os
import tempfile

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


def get_kv_client() -> SecretClient:
    """Return a SecretClient authenticated via DefaultAzureCredential."""
    vault_url = os.environ["KEY_VAULT_URL"]
    return SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())


def load_acled_credentials(client: SecretClient | None = None) -> None:
    """
    Populate ACLED_API_KEY and ACLED_EMAIL from Key Vault.

    Key Vault secrets required:
      acled-api-key   — API key from acleddata.com
      acled-email     — email registered with ACLED
    """
    c = client or get_kv_client()
    os.environ["ACLED_API_KEY"] = c.get_secret("acled-api-key").value
    os.environ["ACLED_EMAIL"]   = c.get_secret("acled-email").value
    print("ACLED credentials loaded from Key Vault.")


def load_gcp_credentials(client: SecretClient | None = None) -> str:
    """
    Load GCP service account credentials from Key Vault.

    Writes the service account JSON to a temporary file and sets:
      GOOGLE_APPLICATION_CREDENTIALS — path to the temp file
      GCP_PROJECT_ID                 — from the JSON or a separate secret

    Returns the temp file path (kept alive for the process lifetime).

    Key Vault secrets required:
      gcp-service-account-json — full content of the GCP service account JSON file
      gcp-project-id           — GCP project ID (used as fallback if not in JSON)
    """
    c = client or get_kv_client()

    json_content = c.get_secret("gcp-service-account-json").value
    parsed = json.loads(json_content)

    # Write to a named temp file that persists until the process exits
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        prefix="gcp_sa_",
    )
    json.dump(parsed, tmp)
    tmp.flush()
    tmp_path = tmp.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_path

    # Resolve GCP_PROJECT_ID: env var > JSON > separate KV secret
    if not os.environ.get("GCP_PROJECT_ID"):
        project_id = parsed.get("project_id", "")
        if not project_id:
            try:
                project_id = c.get_secret("gcp-project-id").value
            except Exception:
                pass
        if project_id:
            os.environ["GCP_PROJECT_ID"] = project_id

    print(
        f"GCP credentials loaded from Key Vault → {tmp_path}\n"
        f"GCP_PROJECT_ID = {os.environ.get('GCP_PROJECT_ID', '(not set)')}"
    )
    return tmp_path
