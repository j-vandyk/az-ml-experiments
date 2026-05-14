"""Critical src/ quality checks (copilot-instructions.md + azure.instructions.md)."""

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from conftest import REPO_ROOT, SRC_FILES


_CREDENTIAL_PATTERNS = [
    re.compile(r"AccountKey\s*=\s*['\"]\w{10,}"),
    re.compile(r"(?i)password\s*=\s*['\"][^{'\"]"),
    re.compile(r"(?i)api_key\s*=\s*['\"][^{'\"]"),
]

_SRC_IDS = [f.relative_to(REPO_ROOT).as_posix() for f in SRC_FILES]


def _public_functions(path: Path) -> List[ast.FunctionDef]:
    """Top-level public functions in a .py file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_flake8_passes() -> None:
    """PEP 8 compliance across src/, utils/, and tools/."""
    result = subprocess.run(
        [sys.executable, "-m", "flake8", "src/", "utils/", "tools/"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"flake8 violations:\n{result.stdout}"


@pytest.mark.parametrize("src_file", SRC_FILES, ids=_SRC_IDS)
def test_public_functions_have_docstrings(src_file: Path) -> None:
    """Every public function needs a docstring (copilot-instructions.md)."""
    if src_file.name == "__init__.py":
        pytest.skip("__init__.py is exempt")
    missing = [
        node.name
        for node in _public_functions(src_file)
        if not ast.get_docstring(node)
    ]
    assert not missing, (
        f"{src_file.relative_to(REPO_ROOT)}: missing docstrings on: {', '.join(missing)}"
    )


@pytest.mark.parametrize("src_file", SRC_FILES, ids=_SRC_IDS)
def test_no_credentials_or_banned_protocols(src_file: Path) -> None:
    """No hardcoded secrets or wasbs:// in library code (azure.instructions.md)."""
    text = src_file.read_text(encoding="utf-8")
    assert "wasbs://" not in text, (
        f"{src_file.relative_to(REPO_ROOT)}: found `wasbs://` — use `abfs://`"
    )
    for pattern in _CREDENTIAL_PATTERNS:
        assert not pattern.search(text), (
            f"{src_file.relative_to(REPO_ROOT)}: potential hardcoded credential"
            f" — `{pattern.pattern}`"
        )
