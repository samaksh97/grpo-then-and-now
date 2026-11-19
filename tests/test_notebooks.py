"""Every notebook must execute top to bottom without errors (outputs are not saved)."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_notebooks import run  # noqa: E402

NOTEBOOKS = sorted((ROOT / "notebook").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_executes(path):
    _, _, errors, _ = run(path, write=False)
    assert not errors, errors
