#!/usr/bin/env python3
"""Append a consistently-formatted entry to docs/refactor-backlog.md.

Auto-numbers the entry by counting existing `## <N>. ` headings.
Creates the file with a standard header if it doesn't exist.

Usage:
    add_backlog_entry.py --title "Fix X" --priority medium \\
        --problem "..." --fix "..." --blast-radius "..."
"""
import argparse
import re
import sys
from pathlib import Path
from textwrap import dedent

DEFAULT_PATH = Path("docs/refactor-backlog.md")

HEADER = dedent("""\
    # Refactor Backlog

    Tracks deferred refactors and known issues that didn't fit the immediate
    fix scope. Items here are real, bounded, and have a proposed fix —
    speculative "we might want to refactor X someday" notes don't belong here.

    Promote an item to a PR when picked up; remove the entry in the same PR
    that resolves it (don't strike-through).

    ---
    """)

ENTRY_TEMPLATE = dedent("""\

    ## {n}. {title}

    **Priority:** {priority}

    **Problem.**
    {problem}

    **Proposed fix.**
    {fix}

    **Blast radius.**
    {blast_radius}

    ---
    """)


def next_entry_number(text: str) -> int:
    nums = [int(m.group(1)) for m in re.finditer(r"^##\s+(\d+)\.\s", text, re.MULTILINE)]
    return (max(nums) + 1) if nums else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", required=True)
    parser.add_argument("--priority", choices=["high", "medium", "low"], default="medium")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--fix", required=True)
    parser.add_argument("--blast-radius", required=True)
    parser.add_argument("--backlog-path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    path = args.backlog_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER)

    body = path.read_text()
    n = next_entry_number(body)
    entry = ENTRY_TEMPLATE.format(n=n, title=args.title.strip(), priority=args.priority,
                                   problem=args.problem.strip(), fix=args.fix.strip(),
                                   blast_radius=args.blast_radius.strip())
    if not body.endswith("\n"):
        body += "\n"
    path.write_text(body + entry)
    print(f"Appended entry {n} to {path}: {args.title!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
