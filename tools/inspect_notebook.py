#!/usr/bin/env python3
"""Inspect a Jupyter notebook and print one line per cell.

Output format: [idx] cell_type id  first-90-chars-of-source

Used alongside edit_notebook.py to verify cell IDs before applying edits.
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print one line per notebook cell (index, type, id, source preview).")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--preview", type=int, default=90)
    args = parser.parse_args()

    if not args.notebook.exists():
        print(f"error: {args.notebook} does not exist", file=sys.stderr)
        return 2

    nb = json.loads(args.notebook.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        head = src.split("\n", 1)[0][: args.preview] if src else "(empty)"
        cid = cell.get("id", "")
        print(f"  [{i:3d}] {cell['cell_type']:8s} id={cid[:14]:14s}  {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
