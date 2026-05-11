#!/usr/bin/env python3
"""Scaffold a label-card markdown stub at docs/labels/<outcome>.md."""
import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

VALID_FAMILIES = {"onset", "delta", "percentile", "triangulated", "weak-supervision", "llm"}
PLACEHOLDERS = ("OUTCOME", "FAMILY", "SOURCE", "OWNER", "TODAY")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outcome", required=True, help="Outcome column name (snake_case)")
    parser.add_argument("--family", required=True, choices=sorted(VALID_FAMILIES))
    parser.add_argument("--source", required=True)
    parser.add_argument("--owner", default="unassigned")
    parser.add_argument("--label-dir", type=Path, default=Path("docs/labels"))
    parser.add_argument("--template", type=Path, default=Path(__file__).parent / "templates" / "label_card_template.md")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.outcome):
        print(f"error: --outcome must be snake_case, got {args.outcome!r}", file=sys.stderr)
        return 2
    if not args.template.exists():
        print(f"error: template not found at {args.template}", file=sys.stderr)
        return 2

    args.label_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.label_dir / f"{args.outcome}.md"
    if out_path.exists() and not args.force:
        print(f"error: {out_path} already exists (use --force to overwrite)", file=sys.stderr)
        return 2

    text = args.template.read_text()
    subs = {"OUTCOME": args.outcome, "FAMILY": args.family, "SOURCE": args.source,
            "OWNER": args.owner, "TODAY": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    for ph in PLACEHOLDERS:
        text = text.replace("{{" + ph + "}}", subs[ph])

    leftover = re.findall(r"\{\{[A-Z_]+\}\}", text)
    if leftover:
        print(f"error: unsubstituted placeholders remain: {sorted(set(leftover))}", file=sys.stderr)
        return 2

    out_path.write_text(text)
    print(f"Scaffolded: {out_path} ({text.count('TODO')} TODO markers to fill in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
