"""Notebook conventions: every linked image exists, and no visible code cell is longer than 25 lines."""
import pathlib
import re

import nbformat
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebook").glob("*.ipynb"))
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_image_links_resolve(path):
    nb = nbformat.read(path, as_version=4)
    links = [m for c in nb.cells if c.cell_type == "markdown" for m in IMAGE.findall(c.source)
             if not m.startswith(("http://", "https://", "attachment:"))]
    missing = [link for link in links if not (path.parent / link).resolve().exists()]
    assert not missing, missing
    unattached = [(m, c.source[:40]) for c in nb.cells if c.cell_type == "markdown"
                  for m in IMAGE.findall(c.source) if m.startswith("attachment:")
                  and m.split(":", 1)[1] not in c.get("attachments", {})]
    assert not unattached, unattached


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_code_cells_stay_short(path):
    nb = nbformat.read(path, as_version=4)
    long_cells = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "code"
                  and len([line for line in c.source.splitlines() if line.strip()]) > 25]
    assert not long_cells, long_cells
