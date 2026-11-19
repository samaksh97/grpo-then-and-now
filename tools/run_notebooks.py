"""Execute notebooks headlessly and report errors, figures and runtime.

Usage:
    python tools/run_notebooks.py                 # execute every notebook in notebook/ and save outputs
    python tools/run_notebooks.py --check A.ipynb # execute without writing (used by the tests)
Exits with status 1 if any cell raises.
"""
import argparse
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(path, write=True, timeout=600):
    """Execute one notebook from its own folder; return (n_code_cells, n_figures, errors, seconds)."""
    nb = nbformat.read(path, as_version=4)
    start = time.time()
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    seconds = time.time() - start
    if write:
        nbformat.write(nb, path)
    code = [c for c in nb.cells if c.cell_type == "code"]
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(code)
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    figures = sum("image/png" in o.get("data", {}) for c in code for o in c.get("outputs", []))
    return len(code), figures, errors, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=pathlib.Path)
    parser.add_argument("--check", action="store_true", help="execute without saving outputs")
    args = parser.parse_args()
    paths = [p.resolve() for p in args.notebooks] or sorted((ROOT / "notebook").glob("*.ipynb"))
    failed = False
    for path in paths:
        n_code, n_figs, errors, seconds = run(path, write=not args.check)
        print(f"{path.name}: {n_code} code cells, {n_figs} figures, {len(errors)} errors, {seconds:.1f} s")
        for error in errors:
            print("   ERROR", error)
        failed |= bool(errors)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
