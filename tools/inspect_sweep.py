#!/usr/bin/env python3
"""Inspect HyperDrive sweep results from MLflow and emit the diagnostic
table from .claude/metric-interpretation-guide.md §9.

Usage:
    inspect_sweep.py --experiment instability_xgboost
    inspect_sweep.py --tracking-uri "azureml://..." --experiment ... --top-n 10
    inspect_sweep.py --outcome civil_war_onset --json
"""
import argparse
import json
import math
import sys

PREVALENCE = {
    "civil_war_onset":          0.003,
    "coup_attempt":             0.002,
    "regime_backsliding":       0.010,
    "mass_unrest_onset":        0.150,
    "humanitarian_crisis_onset":0.070,
    "unrest_binary":            0.080,
    "fh_status_decline":        0.030,
    "ethnic_exclusion_any":     0.300,
    "epr_state_collapse":       0.005,
    "pts_high":                 0.250,
    "pts_escalation":           0.080,
    "aut_ep":                   0.040,
}

TUNABLES = ["max_depth", "learning_rate", "min_child_weight", "gamma", "reg_alpha", "reg_lambda", "subsample", "colsample_bytree"]


def fmt(x, fmt_spec=".4f"):
    if x is None:
        return "—"
    try:
        if x != x:
            return "—"
    except Exception:
        pass
    try:
        return format(float(x), fmt_spec)
    except (TypeError, ValueError):
        return str(x)


def metric_col(df, name):
    import pandas as pd
    if f"metrics.{name}" in df.columns:
        return df[f"metrics.{name}"]
    if name in df.columns:
        return df[name]
    return pd.Series([float("nan")] * len(df))


def param_col(df, name):
    if f"params.{name}" in df.columns:
        return df[f"params.{name}"]
    if name in df.columns:
        return df[name]
    return None


_LOWER_IS_BETTER = {"val_brier", "test_brier", "train_brier"}
_NOT_AN_OBJECTIVE = {"best_iteration"}


def summarise_metric(df, name):
    empty = {"best": None, "min": None, "max": None, "median": None, "std": None}
    s = metric_col(df, name).dropna()
    try:
        s = s.astype(float)
    except (TypeError, ValueError):
        return empty
    if s.empty:
        return empty
    mn, mx = float(s.min()), float(s.max())
    best = mn if name in _LOWER_IS_BETTER else mx
    return {"best": best, "min": mn, "max": mx, "median": float(s.median()), "std": float(s.std())}


def summarise_outcome(df_all, outcome, *, top_n, ceiling):
    df = df_all[df_all.get("tags.outcome", "") == outcome] if "tags.outcome" in df_all else df_all
    n_runs = len(df)
    if n_runs == 0:
        return {"outcome": outcome, "n_runs": 0}

    metrics = {m: summarise_metric(df, m) for m in ("val_auprc", "test_auprc", "train_auprc", "val_brier", "test_brier", "val_precision_at_20", "best_iteration")}

    val_s = metric_col(df, "val_auprc").astype(float, errors="ignore")
    df_sorted = df.assign(_val_auprc=val_s).sort_values("_val_auprc", ascending=False).head(top_n)

    top_trials = []
    for _, row in df_sorted.iterrows():
        trial = {
            "run_id":         row.get("run_id", ""),
            "val_auprc":      float(row.get("_val_auprc", float("nan"))),
            "test_auprc":     float(metric_col(df_sorted, "test_auprc").get(row.name, float("nan"))),
            "train_auprc":    float(metric_col(df_sorted, "train_auprc").get(row.name, float("nan"))),
            "val_brier":      float(metric_col(df_sorted, "val_brier").get(row.name, float("nan"))),
            "best_iteration": metric_col(df_sorted, "best_iteration").get(row.name, None),
            "params":         {p: row.get(f"params.{p}", row.get(p)) for p in TUNABLES if f"params.{p}" in df_sorted.columns or p in df_sorted.columns},
        }
        top_trials.append(trial)

    best = top_trials[0] if top_trials else None
    diagnostics = []
    if best:
        v, t, tr, bi = best["val_auprc"], best["test_auprc"], best["train_auprc"], best["best_iteration"]

        if not (math.isnan(v) or math.isnan(t)):
            gap = v - t
            label = "within noise" if abs(gap) <= 0.05 else ("mild" if abs(gap) <= 0.10 else "STRUCTURAL SHIFT")
            diagnostics.append(("val→test gap (best)", f"{fmt(v)} → {fmt(t)} (Δ={gap:+.3f}) — {label}"))

        vb = best.get("val_brier")
        if vb is not None and math.isnan(vb):
            vb = None
        prev = PREVALENCE.get(outcome)
        if vb is not None and prev is not None:
            baseline = prev * (1 - prev)
            verdict = "beats baseline ✓" if vb < baseline else "WORSE THAN BASELINE — probabilities harmful"
            diagnostics.append(("val_brier vs. prevalence baseline", f"{fmt(vb)} vs {fmt(baseline)} (prev={prev:.1%}) — {verdict}"))
        elif prev is None:
            diagnostics.append(("val_brier vs. prevalence baseline", f"prevalence unknown for {outcome!r}; add to PREVALENCE dict"))

        if bi is not None:
            try:
                bi_v = float(bi)
                pct = bi_v / ceiling if ceiling else 0
                verdict = ("early stopping fired ✓" if pct < 0.8 else "EARLY STOPPING NOT FIRING — raise n_estimators or reduce early_stopping_rounds")
                diagnostics.append(("best_iteration", f"{bi_v:.0f} / {ceiling} ({pct:.0%}) — {verdict}"))
            except (TypeError, ValueError):
                pass

        if not (math.isnan(tr) or math.isnan(v)):
            gap = tr - v
            label = ("minimal overfit" if gap <= 0.10 else "mild overfit" if gap <= 0.20 else "SERIOUS OVERFIT — raise reg_alpha / gamma / min_child_weight")
            diagnostics.append(("train→val gap (best)", f"{fmt(tr)} → {fmt(v)} (Δ={gap:+.3f}) — {label}"))

    clustering = {}
    for p in TUNABLES:
        col = param_col(df_sorted, p)
        if col is None:
            continue
        try:
            vals = col.astype(float).dropna().tolist()
        except (TypeError, ValueError):
            continue
        if not vals:
            continue
        clustering[p] = {"min": min(vals), "median": sorted(vals)[len(vals) // 2], "max": max(vals), "values": vals}

    return {"outcome": outcome, "n_runs": n_runs, "metrics": metrics, "top_trials": top_trials, "diagnostics": diagnostics, "clustering": clustering}


def render_markdown(result: dict) -> str:
    if result["n_runs"] == 0:
        return f"## {result['outcome']}\n\n_No runs found._\n"
    lines = [f"## {result['outcome']} ({result['n_runs']} runs)\n"]
    lines.append("|              | best   | median | std    |")
    lines.append("|--------------|--------|--------|--------|")
    for m in ("val_auprc", "test_auprc", "train_auprc", "val_brier", "val_precision_at_20", "best_iteration"):
        s = result["metrics"][m]
        spec = ".4f" if "auprc" in m or "brier" in m or "precision" in m else ".0f"
        label = m if m != "best_iteration" else "best_iter*"
        lines.append(f"| {label:<12s} | {fmt(s['best'], spec)} | {fmt(s['median'], spec)} | {fmt(s['std'], spec)} |")
    lines.append("\n_*best_iter \"best\" = max iteration reached (saturation indicator, not an objective)._")
    lines.append(f"\n### Top {len(result['top_trials'])} trials by val_auprc\n")
    if result["top_trials"]:
        params_present = sorted({k for t in result["top_trials"] for k in t["params"]})
        header = ["run_id", "val_auprc", "test_auprc", "train_auprc", "best_iter"] + params_present
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for t in result["top_trials"]:
            row = [str(t["run_id"])[:8], fmt(t["val_auprc"]), fmt(t["test_auprc"]), fmt(t["train_auprc"]), fmt(t["best_iteration"], ".0f")]
            for p in params_present:
                row.append(str(t["params"].get(p, "—")))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("\n### Diagnostics\n")
    for label, finding in result["diagnostics"]:
        lines.append(f"- **{label}:** {finding}")
    if result["clustering"]:
        lines.append("\n### Hyperparameter clustering (top-N) — Round 2 narrowing\n")
        for p, c in result["clustering"].items():
            lines.append(f"- `{p}`: min={c['min']}, median={c['median']}, max={c['max']} → tighten next round to roughly [{c['min']}, {c['max']}]")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="instability_xgboost")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--n-estimators-ceiling", type=int, default=300)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        import mlflow
    except ImportError:
        print("error: mlflow not installed. Run `pip install mlflow`.", file=sys.stderr)
        return 2

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    try:
        df = mlflow.search_runs(experiment_names=[args.experiment])
    except Exception as e:
        print(f"error: mlflow.search_runs failed: {e}", file=sys.stderr)
        return 2

    if df.empty:
        print(f"No runs found in experiment {args.experiment!r}.", file=sys.stderr)
        return 2

    if not args.include_failed and "status" in df.columns:
        df = df[df["status"] == "FINISHED"]

    if "tags.outcome" not in df.columns:
        print("warning: runs not tagged with `outcome` — treating all as '(untagged)'.", file=sys.stderr)
        df["tags.outcome"] = "(untagged)"

    outcomes = [args.outcome] if args.outcome else sorted(df["tags.outcome"].dropna().unique().tolist())
    results = [summarise_outcome(df, o, top_n=args.top_n, ceiling=args.n_estimators_ceiling) for o in outcomes]

    if args.json:
        def clean(v):
            if isinstance(v, dict):
                return {k: clean(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            try:
                json.dumps(v)
                return v
            except TypeError:
                return str(v)
        print(json.dumps([clean(r) for r in results], indent=2))
        return 0

    print(f"# Sweep inspection — experiment `{args.experiment}`\n")
    print(f"_{len(df)} total runs across {len(outcomes)} outcome(s)._\n")
    for r in results:
        print(render_markdown(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
