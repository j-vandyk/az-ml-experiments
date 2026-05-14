"""Shared fixtures and helpers for the test suite."""

import json
import re
from pathlib import Path
from typing import List, Tuple

import pytest

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOKS = sorted(REPO_ROOT.rglob("*.ipynb"))
# Python source lives in src/, utils/, and tools/
SRC_FILES = sorted([
    *REPO_ROOT.glob("src/**/*.py"),
    *(REPO_ROOT / "utils").glob("*.py"),
    *(REPO_ROOT / "tools").glob("*.py"),
])

# ── Low-level helpers ──────────────────────────────────────────────────────────


def load_nb(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def cells(nb: dict) -> list:
    return nb.get("cells", [])


def src(cell: dict) -> str:
    s = cell.get("source", [])
    return "".join(s) if isinstance(s, list) else s


def code_cells(nb: dict) -> List[Tuple[int, dict]]:
    """Return (notebook_index, cell) for every code cell."""
    return [(i, c) for i, c in enumerate(cells(nb)) if c.get("cell_type") == "code"]


# ── Parametrised fixture ───────────────────────────────────────────────────────


def _nb_ids() -> List[str]:
    return [nb.relative_to(REPO_ROOT).as_posix() for nb in NOTEBOOKS]


@pytest.fixture(params=NOTEBOOKS, ids=_nb_ids())
def notebook(request) -> Tuple[dict, str]:
    """Yields (parsed_nb_dict, short_name) for every .ipynb in the repo."""
    path: Path = request.param
    return load_nb(path), path.name
