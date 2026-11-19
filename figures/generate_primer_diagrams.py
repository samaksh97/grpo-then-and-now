"""Draw the two schematic diagrams shown in notebook/00_pg_ppo_refresher.ipynb.

They contain no data, so they are drawn once here rather than by code cells in the tutorial:

    00a_rlvr_kernelbench_loop.png   §1.2   the RL loop with a verifiable reward on KernelBench
    00b_ppo_update_loop.png         §1.4   one PPO iteration: sample once, reuse the batch, clip

The PPO-RLHF versus GRPO comparison once drawn here is figure F2 of the curated set in figures/set/.

Usage:  python figures/generate_primer_diagrams.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DPI = 170

INK = "#172b4d"                      # titles
MUTED = "#44546a"                    # body text
LINE = "#53657d"                     # arrows
BLUE = ("#286fb4", "#edf5ff")        # the policy and its updates
GRAY = ("#6b7785", "#f1f3f5")        # frozen models, prompts
ORANGE = ("#b85a20", "#fff7ef")      # learned reward model (replaced by a verifier under RLVR)
PURPLE = ("#6846b8", "#f7f3ff")      # baseline source: critic in PPO, group statistics in GRPO
GREEN = ("#23895d", "#effbf5")       # verifier, pass
RED = ("#c9403b", "#fff4f4")         # fail, removed
SLATE = ("#53657d", "#ffffff")       # data and computation steps
DASH = (0, (5, 3))


def canvas(width, height):
    """A figure whose data coordinates are inches, so layout numbers are physical sizes."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, colors=SLATE, dashed=False, lw=1.6):
    edge, face = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                linestyle=DASH if dashed else "-"))


def text_block(ax, cx, cy, title, body=None, title_size=12.5, body_size=10.5, color=INK, body_lines=None):
    """A bold title with optional body lines, centred as one block on (cx, cy).

    `body_lines` overrides the line count used for centring (a stacked fraction is about two lines tall).
    """
    title_h = (title.count("\n") + 1) * title_size / 72 * 1.25
    gap = 0.07 if body else 0.0
    n_body = body_lines or (body.count("\n") + 1 if body else 0)
    body_h = n_body * body_size / 72 * 1.38
    top = cy + (title_h + gap + body_h) / 2
    ax.text(cx, top, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=color,
            linespacing=1.15)
    if body:
        ax.text(cx, top - title_h - gap, body, ha="center", va="top", fontsize=body_size,
                color=MUTED, linespacing=1.38)


def labelled_box(ax, x, y, w, h, title, body=None, colors=SLATE, dashed=False, **kw):
    box(ax, x, y, w, h, colors, dashed)
    text_block(ax, x + w / 2, y + h / 2, title, body, **kw)


def arrow(ax, points, color=LINE, lw=1.8, dashed=False, head=0.2):
    """Polyline through `points` ending in a solid arrowhead (also when the line is dashed)."""
    pts = [np.asarray(p, dtype=float) for p in points]
    direction = pts[-1] - pts[-2]
    neck = pts[-1] - direction / np.linalg.norm(direction) * head
    xs, ys = zip(*(pts[:-1] + [neck]))
    ax.plot(xs, ys, color=color, lw=lw, linestyle=DASH if dashed else "-",
            solid_capstyle="butt", dash_capstyle="butt", solid_joinstyle="miter")
    ax.add_patch(FancyArrowPatch(tuple(neck), tuple(pts[-1]), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0))


def badge(ax, x, y, label, color):
    ax.text(x, y, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.25", facecolor=color, edgecolor="none"))


def tag(ax, x, y, label, color=MUTED, ha="center", size=9.5):
    ax.text(x, y, label, ha=ha, va="bottom", fontsize=size, fontweight="bold", color=color)


def save(fig, name):
    fig.savefig(ROOT / name, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def rlvr_kernelbench_loop():
    fig, ax = canvas(14, 5.7)
    ax.text(0.15, 5.6, "RL with a verifiable reward, on one KernelBench task", ha="left", va="top",
            fontsize=14, fontweight="bold", color=INK)

    y, h = 2.95, 1.35
    cy = y + h / 2
    labelled_box(ax, 0.15, y, 2.3, h, r"Task prompt $s$",
                 "PyTorch reference code\n(e.g. square matmul)\n+ “make it faster”", GRAY)
    tag(ax, 1.3, y + h + 0.08, "STATE")
    labelled_box(ax, 2.9, y, 2.2, h, r"Policy $\pi_\theta$", "the LLM writes a\nCUDA kernel", BLUE)
    tag(ax, 4.0, y + h + 0.08, "POLICY")
    arrow(ax, [(2.45, cy), (2.9, cy)])
    arrow(ax, [(5.1, cy), (6.45, cy)])
    ax.text(5.67, cy + 0.12, r"action $a$", ha="center", va="bottom", fontsize=10.5, color=INK,
            fontweight="bold")
    ax.text(5.67, cy - 0.12, "kernel code\n(tokens)", ha="center", va="top", fontsize=9, color=MUTED)

    # The verifier: three gates in sequence; failing any gate gives reward 0
    box(ax, 6.25, 1.25, 5.95, 3.75, ("#2f8f5b", "#f7fbf8"), dashed=True, lw=1.4)
    ax.text(6.42, 4.85, "Verifier: the KernelBench harness (a program, not a learned model)",
            ha="left", va="top", fontsize=10.5, fontweight="bold", color=GREEN[0])
    gates = [("Compiles?", "nvcc builds the\nCUDA source"),
             ("Correct?", "same outputs as\nPyTorch on random\ninputs (allclose)"),
             ("Fast enough?", "speedup\n" + r"$t_{\mathrm{PyTorch}}\,/\,t_{\mathrm{kernel}} > p$")]
    gate_w, gate_y, gate_h = 1.7, 2.75, 1.55
    for i, (title, body) in enumerate(gates):
        gx = 6.45 + i * 1.92
        labelled_box(ax, gx, gate_y, gate_w, gate_h, title, body, SLATE, title_size=11.5, body_size=9.5)
        badge(ax, gx + 0.05, gate_y + gate_h - 0.02, str(i + 1), LINE)
        arrow(ax, [(gx + gate_w / 2, gate_y), (gx + gate_w / 2, 2.12)], color=RED[0], lw=1.5)
        ax.text(gx + gate_w / 2 + 0.08, 2.45, "fail", ha="left", va="center", fontsize=9, color=RED[0],
                fontweight="bold")
        if i < 2:
            arrow(ax, [(gx + gate_w, cy + 0.15), (gx + 1.92, cy + 0.15)], lw=1.5)
    labelled_box(ax, 6.45, 1.42, 5.54, 0.7, r"$R = 0$: fails any of the three checks",
                 None, RED, title_size=11.5)
    labelled_box(ax, 12.55, gate_y, 1.3, gate_h, r"$R = 1$", "passes all\nthree checks", GREEN,
                 title_size=13, body_size=9.5)
    tag(ax, 13.2, gate_y + gate_h + 0.08, "REWARD")
    arrow(ax, [(6.45 + 2 * 1.92 + gate_w, cy + 0.15), (12.55, cy + 0.15)], color=GREEN[0], lw=1.5)

    # The update closes the loop
    arrow(ax, [(13.2, gate_y), (13.2, 0.6), (4.0, 0.6), (4.0, y)], color=BLUE[0], lw=2.0)
    ax.plot([9.22, 9.22], [1.42, 0.6], color=BLUE[0], lw=2.0)
    ax.text(8.7, 0.48, r"policy-gradient update of $\theta$ (Section 1.2): make $R=1$ kernels more likely, "
            r"$R=0$ kernels less likely; then sample again", ha="center", va="top", fontsize=10.5,
            color=BLUE[0], fontweight="bold")
    save(fig, "00a_rlvr_kernelbench_loop.png")


def ppo_update_loop():
    fig, ax = canvas(14, 5.9)
    ax.text(0.15, 5.8, "One PPO iteration: sample a batch once, then reuse it for several updates",
            ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    y, h = 2.9, 1.45
    cy = y + h / 2
    labelled_box(ax, 0.15, y, 2.45, h, "Freeze a snapshot",
                 r"$\pi_{\mathrm{old}} \leftarrow \pi_\theta$" + "\n(a frozen copy of\nthe current policy)", GRAY)
    labelled_box(ax, 3.1, y, 2.55, h, "Sample once",
                 r"$\pi_{\mathrm{old}}$ generates rollouts" + "\n" + r"$(s, a, R)$: the expensive" + "\nLLM generation step",
                 SLATE)
    labelled_box(ax, 6.15, y, 2.55, h, "Score with the critic",
                 r"advantage $A = R - V_\psi(s)$" + "\n" + r"(learned critic $V_\psi$," + "\nusually via GAE)", PURPLE)
    for x0, x1 in [(2.6, 3.1), (5.65, 6.15), (8.7, 9.2)]:
        arrow(ax, [(x0, cy), (x1, cy)])
    for i, x in enumerate([0.15, 3.1, 6.15]):
        badge(ax, x + 0.05, y + h - 0.02, str(i + 1), LINE)

    # Step 4: K epochs over the same batch; the ratio drifts away from 1 and clipping stops it
    box(ax, 9.2, 0.95, 4.65, 4.1, BLUE)
    badge(ax, 9.25, 5.03, "4", LINE)
    ax.text(11.52, 4.88, r"Reuse the batch: $K$ epochs of" + "\n" + r"minibatch updates of $\pi_\theta$",
            ha="center", va="top", fontsize=11.5, fontweight="bold", color=INK, linespacing=1.3)
    inset = fig.add_axes([10.05 / 14, 1.55 / 5.9, 3.55 / 14, 2.45 / 5.9])
    steps = np.linspace(0, 12, 300)
    ratio = 1 + 0.36 * (1 - np.exp(-steps / 3.4))
    eps = 0.2
    inset.axhspan(1 - eps, 1 + eps, color="#e3eefb", zorder=0)
    for level in (1 - eps, 1 + eps):
        inset.axhline(level, color=BLUE[0], lw=1.1, ls="--", zorder=1)
    inside = ratio <= 1 + eps
    inset.plot(steps[inside], ratio[inside], color=GREEN[0], lw=2.6, zorder=3)
    inset.plot(steps[~inside], ratio[~inside], color=GRAY[0], lw=2.6, ls=(0, (3, 2)), zorder=3)
    inset.plot([0], [1], "o", color=GREEN[0], ms=6, zorder=4)
    inset.set_xlim(0, 12)
    inset.set_ylim(0.72, 1.62)
    inset.set_yticks([1 - eps, 1, 1 + eps], [r"$1-\epsilon$", "1", r"$1+\epsilon$"], fontsize=9.5)
    inset.set_xticks([])
    inset.set_xlabel("gradient steps on the same batch", fontsize=9.5, color=MUTED)
    inset.set_ylabel(r"$r = \pi_\theta / \pi_{\mathrm{old}}$", fontsize=10.5, color=INK)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    inset.patch.set_alpha(0)
    inset.text(0.3, 1.6, "one token with $A > 0$\n(illustrative)", fontsize=8.5, color=MUTED,
               ha="left", va="top", style="italic")
    inset.text(0.4, 0.86, r"starts at 1: $\pi_\theta = \pi_{\mathrm{old}}$", fontsize=9, color=GREEN[0],
               va="center")
    inset.text(11.8, 1.6, "clipped: zero gradient\nfrom this sample", fontsize=9, color=GRAY[0],
               ha="right", va="top")

    arrow(ax, [(11.52, 0.95), (11.52, 0.55), (1.37, 0.55), (1.37, y)], color=BLUE[0], lw=2.0)
    ax.text(6.3, 0.43, r"after $K$ epochs: take a new snapshot and sample a fresh batch "
            r"(the critic $V_\psi$ is also refit to the observed rewards)", ha="center", va="top",
            fontsize=10.5, color=BLUE[0], fontweight="bold")
    save(fig, "00b_ppo_update_loop.png")


if __name__ == "__main__":
    rlvr_kernelbench_loop()
    ppo_update_loop()
