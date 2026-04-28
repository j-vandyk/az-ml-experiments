"""
Standalone XGBoost training script for one instability outcome.

Called by Azure ML sweep jobs (HyperDrive). Reads feature matrix and
feature manifest from ADLS, trains one XGBoost model, logs to MLflow,
saves model and SHAP values to ADLS.

Usage:
    python train_outcome.py \
        --outcome civil_war_onset \
        --max-depth 5 \
        --learning-rate 0.05 \
        --n-estimators 400 \
        --subsample 0.8 \
        --colsample-bytree 0.8 \
        --min-child-weight 5 \
        --gamma 0.1 \
        --reg-alpha 0.01 \
        --reg-lambda 1.0
"""

import argparse
import json
import os
import re
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import BaseCrossValidator
import xgboost as xgb
import shap

from azure.identity import DefaultAzureCredential
import adlfs
import mlflow
import mlflow.xgboost

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants (must match notebooks 02/02, 03/01, 03/02) ─────────────────────

TRAIN_END_YEAR = 2018
VAL_END_YEAR   = 2021
RANDOM_STATE   = 42

OUTCOME_COLS = [
    "civil_war_onset",
    "coup_attempt",
    "regime_backsliding",
    "mass_unrest_onset",
    "humanitarian_crisis_onset",
]
ID_COLS = ["iso3", "year"]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outcome", required=True, choices=OUTCOME_COLS)

    # Hyperparameters (all optional — defaults match XGB_BASE_PARAMS in nb16)
    p.add_argument("--max-depth",         type=int,   default=5)
    p.add_argument("--learning-rate",     type=float, default=0.05)
    p.add_argument("--n-estimators",      type=int,   default=300)
    p.add_argument("--subsample",         type=float, default=0.8)
    p.add_argument("--colsample-bytree",  type=float, default=0.8)
    p.add_argument("--min-child-weight",  type=int,   default=5)
    p.add_argument("--gamma",             type=float, default=0.0)
    p.add_argument("--reg-alpha",         type=float, default=0.0)
    p.add_argument("--reg-lambda",        type=float, default=1.0)

    # Azure / paths
    p.add_argument("--adls-account",    default=os.environ.get("ADLS_ACCOUNT_NAME", ""))
    p.add_argument("--adls-container",  default=os.environ.get("ADLS_CONTAINER", "data"))
    p.add_argument("--crosswalk-path",  default="data/country_crosswalk.csv",
                   help="Local path to country_crosswalk.csv")
    p.add_argument("--run-date",        default=datetime.utcnow().strftime("%Y%m%d"))

    return p.parse_args()

# ── ADLS helpers ──────────────────────────────────────────────────────────────

def make_storage_opts(account: str, credential) -> dict:
    return {"account_name": account, "credential": credential}

def adls_url(container: str, account: str, subpath: str) -> str:
    return f"abfss://{container}@{account}.dfs.core.windows.net/{subpath}"

def read_latest_parquet(fs, container: str, account: str, prefix: str,
                        storage_opts: dict, filename: str) -> pd.DataFrame:
    full_prefix = f"{container}/{prefix}"
    entries = fs.ls(full_prefix, detail=False)
    date_dirs = sorted(
        [e for e in entries if re.search(r"/\d{8}(/|$)", e)], reverse=True
    )
    if not date_dirs:
        raise FileNotFoundError(f"No date-partitioned dirs under {full_prefix}")
    files = fs.glob(f"{date_dirs[0]}/{filename}")
    if not files:
        raise FileNotFoundError(f"{filename} not found under {date_dirs[0]}")
    url = adls_url(container, account, files[0].replace(f"{container}/", "", 1))
    return pd.read_parquet(url, storage_options=storage_opts)

def load_feature_manifest(fs, container: str, outcome: str) -> list[str]:
    prefix = f"{container}/feature_selection"
    entries = fs.ls(prefix, detail=False)
    date_dirs = sorted(
        [e for e in entries if re.search(r"/\d{8}(/|$)", e)], reverse=True
    )
    if not date_dirs:
        raise FileNotFoundError("No feature_selection date dirs in ADLS")
    path = f"{date_dirs[0]}/selected_{outcome}.json"
    with fs.open(path, "r") as f:
        manifest = json.load(f)
    return manifest.get("feature_names", [])

def write_parquet_adls(df: pd.DataFrame, container: str, account: str,
                       subpath: str, storage_opts: dict) -> None:
    url = adls_url(container, account, subpath)
    df.to_parquet(url, storage_options=storage_opts, index=False, engine="pyarrow")

def write_model_adls(model: xgb.XGBClassifier, fs, container: str,
                     subpath: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save_model(tmp_path)
        with open(tmp_path, "rb") as src:
            with fs.open(f"{container}/{subpath}", "wb") as dst:
                dst.write(src.read())
    finally:
        os.unlink(tmp_path)

# ── Temporal CV ───────────────────────────────────────────────────────────────

class ExpandingYearCV(BaseCrossValidator):
    def __init__(self, years: np.ndarray, n_splits: int = 5, val_years: int = 2,
                 min_val_positives: int = 2):
        self.years             = years
        self.n_splits          = n_splits
        self.val_years         = val_years
        self.min_val_positives = min_val_positives

    def split(self, X, y=None, groups=None):
        unique_years = np.sort(np.unique(self.years))
        candidates   = unique_years[: -self.val_years]
        step         = max(1, len(candidates) // self.n_splits)
        cutpoints    = candidates[step - 1 :: step][: self.n_splits]
        for cutpoint in cutpoints:
            train_idx = np.where(self.years <= cutpoint)[0]
            val_idx   = np.where(
                (self.years > cutpoint) &
                (self.years <= cutpoint + self.val_years)
            )[0]
            if y is not None and len(val_idx):
                if np.sum(np.array(y)[val_idx] == 1) < self.min_val_positives:
                    continue
            if len(train_idx) and len(val_idx):
                yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, outcome: str, feat_cols: list[str]):
    train = df[df["year"] <= TRAIN_END_YEAR]
    val   = df[(df["year"] > TRAIN_END_YEAR) & (df["year"] <= VAL_END_YEAR)]
    test  = df[df["year"] > VAL_END_YEAR]

    def _xy(subset):
        return (
            subset[feat_cols].values.astype(float),
            subset[outcome].values.astype(int),
        )

    return (*_xy(train), *_xy(val), *_xy(test))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    credential   = DefaultAzureCredential()
    storage_opts = make_storage_opts(args.adls_account, credential)
    fs = adlfs.AzureBlobFileSystem(
        account_name=args.adls_account, credential=credential
    )

    # Load feature matrix and labels
    print(f"Loading feature matrix for {args.outcome}...")
    df_features = read_latest_parquet(
        fs, args.adls_container, args.adls_account,
        "processed/feature_matrix", storage_opts, "feature_matrix.parquet",
    )
    df_labels = read_latest_parquet(
        fs, args.adls_container, args.adls_account,
        "processed/feature_matrix", storage_opts, "labels.parquet",
    )

    # Load feature manifest from Notebook 15
    try:
        selected_features = load_feature_manifest(fs, args.adls_container, args.outcome)
        feat_cols = [f for f in selected_features if f in df_features.columns]
        print(f"  Feature manifest: {len(selected_features)} selected, "
              f"{len(feat_cols)} present in matrix")
    except FileNotFoundError:
        feat_cols = [c for c in df_features.columns if c not in ID_COLS + OUTCOME_COLS]
        print(f"  No feature manifest — using all {len(feat_cols)} columns")

    # Merge and filter
    merged = df_features[ID_COLS + feat_cols].merge(
        df_labels[["iso3", "year", args.outcome]],
        on=["iso3", "year"], how="inner",
    )
    merged = merged[merged[args.outcome].notna()].copy()

    if args.outcome == "humanitarian_crisis_onset":
        cw = pd.read_csv(args.crosswalk_path, dtype=str)
        fews_countries = set(cw.loc[cw["fews_monitored"] == "1", "iso3"])
        merged = merged[merged["iso3"].isin(fews_countries)]
        print(f"  Restricted to {len(fews_countries)} FEWS countries")

    # Impute (medians from training rows only)
    train_mask = merged["year"] <= TRAIN_END_YEAR
    medians = merged.loc[train_mask, feat_cols].median()
    merged[feat_cols] = merged[feat_cols].fillna(medians)

    n_pos = int((merged[args.outcome] == 1).sum())
    print(f"  Rows: {len(merged):,} | Positives: {n_pos} ({n_pos/len(merged):.3f})")

    # Temporal split
    X_tr, y_tr, X_v, y_v, X_te, y_te = temporal_split(merged, args.outcome, feat_cols)

    if int(y_tr.sum()) < 5:
        raise ValueError(f"Too few training positives: {int(y_tr.sum())}")

    spw = float((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    print(f"  scale_pos_weight = {spw:.1f}")

    # Build model from CLI hyperparameters
    params = {
        "max_depth":        args.max_depth,
        "learning_rate":    args.learning_rate,
        "n_estimators":     args.n_estimators,
        "subsample":        args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "gamma":            args.gamma,
        "reg_alpha":        args.reg_alpha,
        "reg_lambda":       args.reg_lambda,
    }
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        scale_pos_weight=spw,
        **params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        verbose=False,
    )

    # Metrics
    def _safe_auprc(y_true, y_prob):
        return average_precision_score(y_true, y_prob) if y_true.sum() > 0 else float("nan")

    def _safe_auroc(y_true, y_prob):
        return roc_auc_score(y_true, y_prob) if y_true.sum() > 1 else float("nan")

    val_prob  = model.predict_proba(X_v)[:, 1]
    test_prob = model.predict_proba(X_te)[:, 1]

    metrics = {
        "val_auprc":  _safe_auprc(y_v,  val_prob),
        "val_auroc":  _safe_auroc(y_v,  val_prob),
        "test_auprc": _safe_auprc(y_te, test_prob),
        "test_auroc": _safe_auroc(y_te, test_prob),
    }
    print(f"  Val  AUPRC={metrics['val_auprc']:.4f}  AUROC={metrics['val_auroc']:.4f}")
    print(f"  Test AUPRC={metrics['test_auprc']:.4f}  AUROC={metrics['test_auroc']:.4f}")

    # SHAP (capped at 500 rows)
    explainer   = shap.TreeExplainer(model)
    X_te_sample = X_te[:500]
    shap_values = explainer.shap_values(X_te_sample)

    shap_df = pd.DataFrame(shap_values, columns=feat_cols)
    test_rows = merged[merged["year"] > VAL_END_YEAR].head(500)
    shap_df["iso3"]    = test_rows["iso3"].values
    shap_df["year"]    = test_rows["year"].values
    shap_df["outcome"] = args.outcome

    # SHAP beeswarm figure
    fig_bee, _ = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_te_sample, feature_names=feat_cols,
                      show=False, max_display=20, plot_size=None)
    plt.title(f"SHAP beeswarm — {args.outcome}")
    plt.tight_layout()

    # Calibration figure
    fig_cal = None
    if y_te.sum() > 0:
        fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
        CalibrationDisplay.from_predictions(
            y_te, test_prob, n_bins=10, ax=ax_cal, name=args.outcome,
        )
        ax_cal.set_title(f"Calibration — {args.outcome}")
        fig_cal.tight_layout()

    # Save to ADLS
    model_subpath = f"models/{args.run_date}/{args.outcome}/model.json"
    shap_subpath  = f"models/{args.run_date}/{args.outcome}/shap_values.parquet"
    write_model_adls(model, fs, args.adls_container, model_subpath)
    write_parquet_adls(shap_df, args.adls_container, args.adls_account,
                       shap_subpath, storage_opts)
    print(f"  Model → {model_subpath}")
    print(f"  SHAP  → {shap_subpath}")

    # MLflow logging
    mlflow_uri = os.getenv("AZUREML_MLFLOW_URI", "")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("instability_xgboost")

    with mlflow.start_run(tags={"outcome": args.outcome}):
        mlflow.log_params({**params, "scale_pos_weight": spw,
                           "n_features": len(feat_cols)})
        mlflow.log_metrics(metrics)
        mlflow.log_figure(fig_bee, f"shap_beeswarm_{args.outcome}.png")
        if fig_cal is not None:
            mlflow.log_figure(fig_cal, f"calibration_{args.outcome}.png")
        mlflow.xgboost.log_model(model, artifact_path=f"xgb_{args.outcome}")

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
