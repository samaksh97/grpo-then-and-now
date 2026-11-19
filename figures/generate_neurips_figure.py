from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(12, 1.05))
ax.set_xlim(0, 12)
ax.set_ylim(0, 1.05)
ax.axis("off")


def box(x, w, text, edge, face, fontsize=13):
    ax.add_patch(FancyBboxPatch(
        (x, 0.16), w, 0.72, boxstyle="round,pad=0.018,rounding_size=0.05",
        linewidth=1.35, edgecolor=edge, facecolor=face,
    ))
    ax.text(x + w / 2, 0.52, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#172b4d")


def arrow(x1, x2):
    ax.add_patch(FancyArrowPatch(
        (x1, 0.52), (x2, 0.52), arrowstyle="-|>", mutation_scale=15,
        linewidth=1.7, color="#53657d",
    ))


box(0.05, 1.2, "GRPO", "#286fb4", "#edf5ff", 16)
arrow(1.3, 1.55)

failures = ["F1\ngroups", "F2\nentropy", "F3\nlength", "F4\ntrunc.", "F5\ndiff.", "F6\nover-think"]
for i, label in enumerate(failures):
    box(1.65 + i * 0.9, 0.78, label, "#d9534f", "#fff4f4", 10.5)

arrow(7.02, 7.27)
box(7.37, 1.3, "DAPO\nF1–F4", "#23895d", "#effbf5", 13)
box(8.83, 1.3, "DisCO\nF2, F5", "#6846b8", "#f7f3ff", 13)
box(10.29, 1.55, "S-GRPO\nF6", "#b85a20", "#fff7ef", 13)

plt.tight_layout(pad=0.05)
fig.savefig(ROOT / "neurips_figure_simple.png", dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(ROOT / "neurips_figure_simple.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
