#!/usr/bin/env python3
"""Apply structured edits to a Jupyter notebook keyed by cell ID.

Usage:
    edit_notebook.py <notebook.ipynb> <edits.json>      # read spec from file
    edit_notebook.py <notebook.ipynb> -                 # read spec from stdin

Edits spec (JSON):
{
  "edits": [
    {"cell_id": "cell-3", "new_source": "..."},
    {"cell_id": "abcd1234", "old_substring": "foo", "new_substring": "bar"},
    {"cell_id": "ef005",    "old_substring": "X",   "new_substring": "Y", "optional": true},
    {"cell_id": "1234abcd", "old_substring": "x",   "new_substring": "y", "replace_all": true},
    {"cell_id": "deadbeef", "append": "\\n# trailing comment"}
  ]
}

Each edit specifies exactly ONE of:
- new_source       — replace cell source verbatim
- old_substring + new_substring — exact substring replace
- append           — append text to existing source

Errors if a cell_id is not found (fail loud — do not silently create cells).
Writes back with indent=1 to match the project convention.
"""
import argparse
import json
import sys
from pathlib import Path


def find_cell(nb: dict, cell_id: str) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("id") == cell_id:
            return cell
    raise KeyError(f"cell_id not found: {cell_id!r}")


def apply_edit(cell: dict, edit: dict) -> str:
    cid = edit["cell_id"]
    keys = {k for k in ("new_source", "old_substring", "append") if k in edit}
    if len(keys) != 1:
        raise ValueError(f"edit for {cid} must specify exactly one of new_source / old_substring / append; got {sorted(keys)}")

    if "new_source" in edit:
        cell["source"] = edit["new_source"].splitlines(keepends=True)
        return f"  {cid}: replaced source ({len(edit['new_source'])} chars)"

    src = "".join(cell.get("source", []))
    if "old_substring" in edit:
        if "new_substring" not in edit:
            raise ValueError(f"edit for {cid}: old_substring requires new_substring")
        count = src.count(edit["old_substring"])
        if count == 0:
            if edit.get("optional", False):
                return f"  {cid}: old_substring not found (optional, skipped)"
            raise ValueError(f"edit for {cid}: old_substring not found verbatim")
        if count > 1 and not edit.get("replace_all", False):
            raise ValueError(f"edit for {cid}: old_substring matches {count} sites — make it more specific, or set 'replace_all': true")
        new_src = src.replace(edit["old_substring"], edit["new_substring"])
        cell["source"] = new_src.splitlines(keepends=True)
        suffix = f" ×{count}" if count > 1 else ""
        return f"  {cid}: substring replaced{suffix}"

    appended = src.rstrip("\n") + "\n" + edit["append"]
    cell["source"] = appended.splitlines(keepends=True)
    return f"  {cid}: appended {len(edit['append'])} chars"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("spec", help="Path to edits.json, or '-' for stdin")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not args.notebook.exists():
        print(f"error: {args.notebook} does not exist", file=sys.stderr)
        return 2

    spec_text = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text()
    try:
        spec = json.loads(spec_text)
    except json.JSONDecodeError as e:
        print(f"error: spec is not valid JSON: {e}", file=sys.stderr)
        return 2

    edits = spec.get("edits", [])
    if not edits:
        print("error: spec has no 'edits' array or it is empty", file=sys.stderr)
        return 2

    nb = json.loads(args.notebook.read_text())
    log_lines: list[str] = []
    for edit in edits:
        cid = edit.get("cell_id")
        if not cid:
            print(f"error: edit missing cell_id: {edit!r}", file=sys.stderr)
            return 2
        try:
            cell = find_cell(nb, cid)
        except KeyError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        try:
            log_lines.append(apply_edit(cell, edit))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    if args.check_only:
        print("check-only — no write. Would apply:")
        for line in log_lines:
            print(line)
        return 0

    args.notebook.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Updated {args.notebook}:")
    for line in log_lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
