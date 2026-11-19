"""grpo_lab: the small NumPy library behind the 'GRPO, Then and Now' notebooks.

Modules
    data        locate the repository and load the KernelBench task specifications
    envs        toy verifiable-reward environments (Level A bandit, Level B sequence)
    objectives  group-relative advantages, clipped objectives and GRPO update steps
    plots       every plot the notebooks draw, so notebook cells stay short

Equation numbers in docstrings refer to DeepSeekMath (Shao et al., 2024), PPO (Schulman et al., 2017)
and DAPO (Yu et al., 2025).
"""
from . import data, envs, objectives, plots

__all__ = ["data", "envs", "objectives", "plots"]
