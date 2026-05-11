#!/usr/bin/env python3
"""Static checks for the feature-engineering notebooks.

Exits non-zero if any ERROR-severity finding is reported. WARN and INFO findings
do not affect exit code (so this is safe to run as an advisory pre-commit step).

Usage:
    check_fe.py --all
    check_fe.py path/to/02_build_feature_matrix.ipynb [path/to/03_engineer_derived_features.ipynb]
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_BUILD_MATRIX = Path("notebooks/02_feature_engineering/02_build_feature_matrix.ipynb")
DEFAULT_ENGINEER     = Path("notebooks/02_feature_engineering/03_engineer_derived_features.ipynb")

# ANSI-free severity tags so output stays grep-able.
ERROR = "ERROR"
WARN  = "WARN"
INFO  = "INFO"


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def find_cells(nb: dict, predicate) -> list[tuple[int, dict]]:
    return [(i, c) for i, c in enumerate(nb.get("cells", [])) if predicate(c)]


def cell_by_id(nb: dict, cell_id: str) -> dict | None:
    for cell in nb.get("cells", []):
        if cell.get("id") == cell_id:
            return cell
    return None


def emit(findings: list, severity: str, cell_id: str, message: str) -> None:
    findings.append((severity, cell_id, message))


# ─── checks for 02_build_feature_matrix.ipynb ──────────────────────────────

def check_build_matrix(nb: dict, findings: list) -> None:
    # 1. `_log_join` helper should not be present.
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        if re.search(r"\bdef\s+_log_join\s*\(", src):
            emit(findings, WARN, cell.get("id", f"#{i}"),
                 "_log_join helper is back — was removed in the audit refactor; check for a stale revert")

    # 2. _join_iso3_source("X", ...) keys must be in RAW_PREFIXES (cell-3 is the canonical home).
    raw_prefix_keys: set[str] = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        if "RAW_PREFIXES" in src and "{" in src and "raw/" in src:
            for m in re.finditer(r'"([a-z][a-z0-9_]*)"\s*:\s*"raw/', src):
                raw_prefix_keys.add(m.group(1))

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        for m in re.finditer(r'_join_iso3_source\(\s*"([a-z][a-z0-9_]*)"', src):
            key = m.group(1)
            if raw_prefix_keys and key not in raw_prefix_keys:
                emit(findings, ERROR, cell.get("id", "?"),
                     f'_join_iso3_source("{key}", ...) — "{key}" is not a key in RAW_PREFIXES; '
                     f"will raise KeyError at runtime")

    # 3. Stale comment string.
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        if "(notebooks 14c–14g sources)" in src or "(notebooks 14c-14g sources)" in src:
            emit(findings, WARN, cell.get("id", "?"),
                 'stale comment "(notebooks 14c–14g sources)" — replace with "(notebooks 14–18 sources)"')

    # 4. WEO must be joined with feature_prefix=.
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        for m in re.finditer(r'_join_iso3_source\(\s*"weo"[^)]*\)', src):
            call = m.group(0)
            if "feature_prefix" not in call:
                emit(findings, ERROR, cell.get("id", "?"),
                     "WEO joined without feature_prefix= — will collide with WDI on "
                     "gdp_growth_pct / inflation_* and produce silent _x/_y columns")


# ─── checks for 03_engineer_derived_features.ipynb ─────────────────────────

def check_engineer(nb: dict, findings: list) -> None:
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        cid = cell.get("id", "?")

        # 1. df.merge inside a for-loop in the Section D audit cell.
        if "audit_rows" in src or "for col in derived_cols" in src:
            in_loop = False
            for line in src.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("for col in derived_cols"):
                    in_loop = True
                    continue
                # naive but adequate: any de-indented top-level statement after the loop
                # ends our window. We only flag merges *inside* the loop.
                if in_loop and stripped and not line.startswith((" ", "\t")):
                    in_loop = False
                if in_loop and re.search(r"\b\w+\.merge\(", line) and "df_labels" in line:
                    emit(findings, ERROR, cid,
                         f"df.merge() inside the per-feature loop ({stripped[:60]}...) — "
                         "hoist the merge before the loop")

        # 2. HP-filter row alignment.
        if "hpfilter" in src:
            if re.search(r"trend_vals\.extend\(|cycle_vals\.extend\(|\.extend\(\s*trend\b|\.extend\(\s*cycle\b", src):
                emit(findings, ERROR, cid,
                     "HP filter populated via .extend() — relies on groupby iteration order matching df rows. "
                     "Use df.loc[grp.index, col] = trend / cycle instead")
            if re.search(r"\.fillna\(\s*0\s*\).*hpfilter|hpfilter.*\.fillna\(\s*0\s*\)", src, re.DOTALL):
                emit(findings, WARN, cid,
                     "HP-filter input uses .fillna(0) — produces spurious leading downturns "
                     "for countries with NaN early years; prefer ffill().bfill()")

        # 3. Parquet-as-CSV: write_parquet on something that ends .csv.
        for m in re.finditer(r"write_parquet\(\s*[^,]+,\s*[^)]*\.csv", src):
            emit(findings, ERROR, cid,
                 "write_parquet(..., '...csv') — parquet payload with .csv suffix; use to_csv for catalog writes")

        # 4. Section C per-row .at writes (deferred refactor; flag as WARN).
        if "Moran" in src or "spatial_lag" in src:
            if re.search(r"\.at\[[^\]]+\]\s*=", src):
                emit(findings, WARN, cid,
                     "Section C uses per-row .at[] assignment — vectorise with concat+merge "
                     "(see docs/refactor-backlog.md item 3)")


# ─── cross-notebook check ──────────────────────────────────────────────

def extract_outcome_cols(nb: dict, *, ident: str) -> set[str] | None:
    """Find a list literal assigned to `ident` and return its string entries."""
    for cell in nb.get("cells", []):
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        if ident == "OUTCOME_COLS":
            m = re.search(r"OUTCOME_COLS\s*=\s*\[(.*?)\]", src, re.DOTALL)
        else:
            m = re.search(r"'outcome_cols'\s*:\s*\[(.*?)\]", src, re.DOTALL)
        if m:
            return set(re.findall(r"['\"]([a-z][a-z0-9_]*)['\"]" , m.group(1)))
    return None


def check_outcome_alignment(build_nb: dict, eng_nb: dict, findings: list) -> None:
    build_cols = extract_outcome_cols(build_nb, ident="OUTCOME_COLS")
    eng_cols   = extract_outcome_cols(eng_nb, ident="outcome_cols")
    if build_cols is None or eng_cols is None:
        emit(findings, INFO, "(cross)",
             "could not locate OUTCOME_COLS / ENG_CFG outcome_cols list — skipping alignment check")
        return
    only_build = build_cols - eng_cols
    only_eng   = eng_cols - build_cols
    if only_build:
        emit(findings, WARN, "(cross)",
             f"OUTCOME_COLS has labels missing from ENG_CFG['outcome_cols']: {sorted(only_build)}")
    if only_eng:
        emit(findings, WARN, "(cross)",
             f"ENG_CFG['outcome_cols'] has labels missing from OUTCOME_COLS: {sorted(only_eng)}")


# ─── runner ────────────────────────────────────────────────────────────────────────────

def report(findings: list[tuple[str, str, str]], label: str) -> None:
    print(f"\n=== {label} ===")
    if not findings:
        print("  (no findings)")
        return
    severity_order = {ERROR: 0, WARN: 1, INFO: 2}
    findings.sort(key=lambda f: (severity_order.get(f[0], 99), f[1]))
    for sev, cid, msg in findings:
        print(f"  [{sev:5s}] {cid:14s}  {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Notebooks to check")
    parser.add_argument("--all", action="store_true",
                        help=f"Check both default targets ({DEFAULT_BUILD_MATRIX} and {DEFAULT_ENGINEER})")
    args = parser.parse_args()

    targets: list[Path] = list(args.notebooks)
    if args.all:
        targets += [DEFAULT_BUILD_MATRIX, DEFAULT_ENGINEER]
    if not targets:
        parser.print_help()
        return 2

    seen: set[Path] = set()
    targets = [p for p in targets if not (p in seen or seen.add(p))]

    nbs: dict[str, dict] = {}
    all_findings: dict[str, list] = {}
    has_error = False

    for path in targets:
        if not path.exists():
            print(f"error: notebook not found: {path}", file=sys.stderr)
            return 2
        nb = json.loads(path.read_text())
        nbs[str(path)] = nb
        findings: list = []

        name = path.name
        if name == "02_build_feature_matrix.ipynb":
            check_build_matrix(nb, findings)
        elif name == "03_engineer_derived_features.ipynb":
            check_engineer(nb, findings)
        else:
            emit(findings, INFO, "(file)",
                 f"no specific checks for {name} — running both rule sets defensively")
            check_build_matrix(nb, findings)
            check_engineer(nb, findings)

        all_findings[str(path)] = findings
        report(findings, str(path))
        if any(f[0] == ERROR for f in findings):
            has_error = True

    build_path = next((p for p in targets if p.name == "02_build_feature_matrix.ipynb"), None)
    eng_path   = next((p for p in targets if p.name == "03_engineer_derived_features.ipynb"), None)
    if build_path and eng_path:
        cross_findings: list = []
        check_outcome_alignment(nbs[str(build_path)], nbs[str(eng_path)], cross_findings)
        report(cross_findings, "cross-notebook checks")
        if any(f[0] == ERROR for f in cross_findings):
            has_error = True

    print()
    summary = (
        f"Summary: "
        f"{sum(1 for findings in all_findings.values() for f in findings if f[0] == ERROR)} ERROR, "
        f"{sum(1 for findings in all_findings.values() for f in findings if f[0] == WARN)} WARN, "
        f"{sum(1 for findings in all_findings.values() for f in findings if f[0] == INFO)} INFO"
    )
    print(summary)
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
