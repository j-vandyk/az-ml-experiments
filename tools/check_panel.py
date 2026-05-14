#!/usr/bin/env python3
"""Structural sanity checks for an engineered feature-matrix parquet.

Usage:
    check_panel.py <local_path_or_abfss_url> [--options ...]

Exit code:
    0  — no ERROR-severity findings
    1  — at least one ERROR finding
    2  — argument / load error
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ERROR = "ERROR"
WARN  = "WARN"
INFO  = "INFO"

DEFAULT_OUTCOME_COLS = [
    "civil_war_onset", "coup_attempt", "regime_backsliding",
    "mass_unrest_onset", "humanitarian_crisis_onset",
    "unrest_binary", "fh_status_decline",
    "ethnic_exclusion_any", "epr_state_collapse",
    "pts_high", "pts_escalation", "aut_ep",
]


def load_parquet(path: str):
    import pandas as pd
    if path.startswith("abfss://"):
        try:
            from azure.identity import DefaultAzureCredential  # noqa: F401
        except ImportError as e:
            print(f"error: ADLS read requires azure-identity: {e}", file=sys.stderr)
            raise
        try:
            import adlfs  # noqa: F401
        except ImportError as e:
            print(f"error: ADLS read requires adlfs: {e}", file=sys.stderr)
            raise
        import re
        m = re.match(r"abfss://[^@]+@([^.]+)\.dfs\.core\.windows\.net/", path)
        if not m:
            raise ValueError(f"could not parse account name from {path!r}")
        from azure.identity import DefaultAzureCredential  # noqa: F811
        storage_options = {
            "account_name": m.group(1),
            "credential":   DefaultAzureCredential(),
        }
        return pd.read_parquet(path, storage_options=storage_options)
    return pd.read_parquet(path)


def emit(findings: list, severity: str, message: str) -> None:
    findings.append((severity, message))


def check_keys(df, findings: list) -> None:
    for col in ("iso3", "year"):
        if col not in df.columns:
            emit(findings, ERROR, f"missing required key column: {col}")
            return
    if df["iso3"].isna().any():
        emit(findings, ERROR, f"iso3 has {df['iso3'].isna().sum()} NaN values")
    if df["year"].isna().any():
        emit(findings, ERROR, f"year has {df['year'].isna().sum()} NaN values")
    n_dup = df.duplicated(["iso3", "year"]).sum()
    if n_dup:
        emit(findings, ERROR, f"{n_dup} duplicate (iso3, year) rows")


def check_country_year_coverage(df, findings: list, *, c_min: int, c_max: int,
                                 years_expected: tuple[int, int] | None) -> None:
    if "iso3" not in df.columns or "year" not in df.columns:
        return
    n_countries = df["iso3"].nunique()
    if n_countries < c_min:
        emit(findings, WARN, f"only {n_countries} distinct countries (expected ≥ {c_min})")
    if n_countries > c_max:
        emit(findings, WARN, f"{n_countries} distinct countries (expected ≤ {c_max})")
    y_min, y_max = int(df["year"].min()), int(df["year"].max())
    emit(findings, INFO, f"countries={n_countries}, year range={y_min}–{y_max}")
    if years_expected is not None:
        ey_min, ey_max = years_expected
        if y_min != ey_min or y_max != ey_max:
            emit(findings, WARN, f"year range {y_min}–{y_max} differs from expected {ey_min}–{ey_max}")
    n_years = df["year"].nunique()
    n_total = len(df)
    expected_rows = n_countries * n_years
    coverage = n_total / expected_rows if expected_rows else 0
    emit(findings, INFO, f"rows={n_total:,}, full grid would be {expected_rows:,} ({coverage:.1%} coverage)")
    by_country = df.groupby("iso3")["year"].nunique()
    sparse_countries = by_country[by_country < 5].index.tolist()
    if sparse_countries:
        sample = sparse_countries[:10]
        emit(findings, WARN, f"{len(sparse_countries)} country/countries with <5 years of data: {sample}{'...' if len(sparse_countries) > 10 else ''}")
    by_year = df.groupby("year")["iso3"].nunique()
    sparse_years = by_year[by_year < 40].index.tolist()
    if sparse_years:
        emit(findings, WARN, f"{len(sparse_years)} year(s) with <40 countries: {sparse_years[:10]}")


def check_outcome_labels(df, findings: list, outcome_cols: list[str]) -> None:
    missing = [c for c in outcome_cols if c not in df.columns]
    if missing:
        emit(findings, ERROR, f"required outcome columns missing: {missing}")
    present = [c for c in outcome_cols if c in df.columns]
    for col in present:
        s = df[col]
        n_pos = int((s == 1).sum())
        n_total = int(s.notna().sum())
        rate = n_pos / n_total if n_total else 0
        emit(findings, INFO, f"{col}: positives={n_pos:,}/{n_total:,} ({rate:.2%})")
        if n_pos == 0 and n_total > 0:
            emit(findings, ERROR, f"{col} has zero positives — label likely broken")


def check_features(df, findings: list, *, missingness_threshold: float,
                   nzv_threshold: float, top_missing: int) -> None:
    feature_cols = [c for c in df.columns
                    if c not in ("iso3", "year") and c not in DEFAULT_OUTCOME_COLS]
    if not feature_cols:
        emit(findings, WARN, "no feature columns detected (only keys + outcomes)")
        return
    emit(findings, INFO, f"feature columns: {len(feature_cols)}")
    miss_frac = df[feature_cols].isna().mean()
    all_nan = miss_frac[miss_frac >= 1.0].index.tolist()
    if all_nan:
        emit(findings, WARN, f"{len(all_nan)} feature columns are 100% NaN: {all_nan[:10]}")
    high_miss = miss_frac[(miss_frac > missingness_threshold) & (miss_frac < 1.0)]
    if not high_miss.empty:
        sample = high_miss.sort_values(ascending=False).head(top_missing)
        emit(findings, WARN, f"{len(high_miss)} columns >{missingness_threshold:.0%} missing — top entries: {[(c, f'{v:.0%}') for c, v in sample.items()]}")
    top = miss_frac.sort_values(ascending=False).head(top_missing)
    emit(findings, INFO, f"top-{top_missing} most-missing features: {[(c, f'{v:.0%}') for c, v in top.items() if v > 0]}")
    numeric = df[feature_cols].select_dtypes(include="number")
    if not numeric.empty:
        stds = numeric.std()
        nzv = stds[stds < nzv_threshold].index.tolist()
        nzv_pct = len(nzv) / len(feature_cols) if feature_cols else 0
        if nzv:
            level = WARN if nzv_pct > 0.05 else INFO
            emit(findings, level, f"{len(nzv)} features with std<{nzv_threshold} ({nzv_pct:.1%} of features)")
            emit(findings, INFO, f"NZV columns: {nzv[:15]}{'...' if len(nzv) > 15 else ''}")
    suspicious = [c for c in feature_cols if c.endswith("_x") or c.endswith("_y")]
    if suspicious:
        emit(findings, WARN, f"{len(suspicious)} columns end with _x / _y — likely a silent merge collision: {suspicious[:10]}")


def report(findings: list[tuple[str, str]]) -> bool:
    severity_order = {ERROR: 0, WARN: 1, INFO: 2}
    findings.sort(key=lambda f: severity_order.get(f[0], 99))
    has_error = False
    for sev, msg in findings:
        if sev == ERROR:
            has_error = True
        print(f"  [{sev:5s}] {msg}")
    counts = {ERROR: 0, WARN: 0, INFO: 0}
    for sev, _ in findings:
        counts[sev] = counts.get(sev, 0) + 1
    print()
    print(f"Summary: {counts[ERROR]} ERROR, {counts[WARN]} WARN, {counts[INFO]} INFO")
    return has_error


def parse_year_range(s: str) -> tuple[int, int]:
    a, b = s.split("-", 1)
    return int(a), int(b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Local parquet path or abfss:// URL")
    parser.add_argument("--countries-min", type=int, default=30)
    parser.add_argument("--countries-max", type=int, default=200)
    parser.add_argument("--years-expected", type=parse_year_range, default=None)
    parser.add_argument("--outcome-cols", default=",".join(DEFAULT_OUTCOME_COLS))
    parser.add_argument("--missingness-threshold", type=float, default=0.75)
    parser.add_argument("--nzv-threshold", type=float, default=1e-6)
    parser.add_argument("--top-missing", type=int, default=20)
    args = parser.parse_args()

    if not args.path.startswith("abfss://") and not Path(args.path).exists():
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2

    try:
        df = load_parquet(args.path)
    except Exception as e:
        print(f"error: failed to load parquet: {e}", file=sys.stderr)
        return 2

    print(f"Loaded: {args.path}")
    print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print()

    outcome_cols = [c.strip() for c in args.outcome_cols.split(",") if c.strip()]
    findings: list = []
    check_keys(df, findings)
    check_country_year_coverage(df, findings, c_min=args.countries_min, c_max=args.countries_max, years_expected=args.years_expected)
    check_outcome_labels(df, findings, outcome_cols)
    check_features(df, findings, missingness_threshold=args.missingness_threshold, nzv_threshold=args.nzv_threshold, top_missing=args.top_missing)

    has_error = report(findings)
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
