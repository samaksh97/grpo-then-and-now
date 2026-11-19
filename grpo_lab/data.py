"""Locate the repository and load the KernelBench task specifications."""
import json
import pathlib

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
TASKS_PATH = REPO_ROOT / "data" / "kernelbench" / "tasks.json"
FIGURES_DIR = REPO_ROOT / "figures" / "set"


def load_tasks(path=TASKS_PATH):
    """Return the KernelBench task dicts: real problem specifications with illustrative metrics.

    See data/kernelbench/README.md for what is real and what is illustrative.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def matmul_task(tasks=None):
    """The Level 1 square-matmul task (KernelBench problem 1) used by the bandit environment."""
    tasks = load_tasks() if tasks is None else tasks
    return next(t for t in tasks if t["kernelbench_problem_id"] == 1 and t["level"] == 1)
