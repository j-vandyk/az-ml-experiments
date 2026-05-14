"""Critical notebook structure checks (notebooks.instructions.md)."""

import re
from typing import Tuple

import pytest

from conftest import cells, code_cells, src


# ── Helpers ────────────────────────────────────────────────────────────────────

_CREDENTIAL_PATTERNS = [
    re.compile(r"AccountKey\s*=\s*['\"]\w{10,}"),
    re.compile(r"DefaultEndpointsProtocol\s*=\s*https"),
    re.compile(r"(?i)sas_token\s*=\s*['\"][^{'\"]"),
    re.compile(r"(?i)password\s*=\s*['\"][^{'\"]"),
]


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_first_cell_is_markdown(notebook: Tuple[dict, str]) -> None:
    """First cell must be a markdown documentation block."""
    nb, name = notebook
    cell_list = cells(nb)
    assert cell_list, f"{name}: notebook has no cells"
    assert cell_list[0].get("cell_type") == "markdown", (
        f"{name}: first cell must be markdown, got '{cell_list[0].get('cell_type')}'"
    )
    first_src = src(cell_list[0])
    assert re.search(r"^#{1,2} ", first_src, re.MULTILINE), (
        f"{name}: first markdown cell must have a title heading (## Title)"
    )


def test_master_config_cell_present(notebook: Tuple[dict, str]) -> None:
    """A master config cell with DEBUG and a box header must exist in the first 5 code cells."""
    nb, name = notebook
    code = code_cells(nb)
    found = any(
        "DEBUG" in src(cell) and ("\u2554" in src(cell) or "MASTER CONFIG" in src(cell).upper())
        for _, cell in code[:5]
    )
    assert found, (
        f"{name}: no master config cell in the first 5 code cells "
        f"(expected `DEBUG = False` + \u2554\u2550\u2550\u2557 box header)"
    )


def test_no_hardcoded_credentials(notebook: Tuple[dict, str]) -> None:
    """No raw connection strings or API keys (azure.instructions.md)."""
    nb, name = notebook
    for i, cell in enumerate(cells(nb)):
        cell_src = src(cell)
        for pattern in _CREDENTIAL_PATTERNS:
            assert not pattern.search(cell_src), (
                f"{name} cell {i + 1}: potential hardcoded credential — `{pattern.pattern}`"
            )


def test_no_wasbs_protocol(notebook: Tuple[dict, str]) -> None:
    """Use abfs:// not wasbs:// for ADLS Gen2 (azure.instructions.md)."""
    nb, name = notebook
    for i, cell in enumerate(cells(nb)):
        assert "wasbs://" not in src(cell), (
            f"{name} cell {i + 1}: found `wasbs://` — use `abfs://` instead"
        )


def test_no_bare_display_calls(notebook: Tuple[dict, str]) -> None:
    """display() calls must be inside `if DEBUG:` (notebooks.instructions.md)."""
    nb, name = notebook
    violations = []
    for i, cell in enumerate(cells(nb)):
        if cell.get("cell_type") != "code":
            continue
        cell_src = src(cell)
        if re.search(r"^\s*display\(", cell_src, re.MULTILINE) and "if DEBUG" not in cell_src:
            violations.append(i + 1)
    assert not violations, (
        f"{name}: bare display() calls in cells {violations} — wrap in `if DEBUG:`"
    )
