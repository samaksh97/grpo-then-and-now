"""Every plot drawn by the notebooks, so that notebook cells show the computation and not the styling.

Each function takes the arrays a notebook cell computed and draws one figure. Colours follow one convention
throughout: green = positive advantage / reinforced, red = negative advantage / suppressed, blue = reward,
orange = entropy or ratios, purple = clipping.
"""
import matplotlib.pyplot as plt
import numpy as np

GREEN, RED, BLUE, ORANGE, PURPLE, GRAY = "#2ecc71", "#e74c3c", "#3498db", "#e67e22", "#9b59b6", "#7f8c8d"

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.2, "axes.spines.top": False,
                     "axes.spines.right": False})


# ----------------------------------------------------------------------------- refresher and core
def variance_reduction(reinforce_grads, baselined_grads):
    """Side-by-side histograms of per-sample gradient estimates without and with a mean-reward baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, grads, color, title in [(axes[0], reinforce_grads, RED, "REINFORCE (no baseline)"),
                                    (axes[1], baselined_grads, GREEN, "With baseline (mean reward)")]:
        ax.hist(grads, bins=40, alpha=0.7, color=color, edgecolor="white")
        ax.axvline(grads.mean(), color="black", linestyle="--", linewidth=2, label=f"Mean = {grads.mean():.3f}")
        ax.set(title=title, xlabel="Gradient estimate")
        ax.legend()
        ax.annotate(f"Std = {grads.std():.3f}", xy=(0.95, 0.85), xycoords="axes fraction", ha="right",
                    fontsize=11, bbox=dict(boxstyle="round", facecolor=color, alpha=0.2))
    axes[0].set_ylabel("Count")
    fig.suptitle("Same expected gradient, much lower variance", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def advantage_bars(advantages, labels):
    """Bar chart of one group's advantages; labels: one short name per rollout."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(range(len(advantages)), advantages, color=[GREEN if a > 0 else RED for a in advantages],
           edgecolor="white", linewidth=1.5)
    for i, adv in enumerate(advantages):
        ax.text(i, adv + (0.06 if adv >= 0 else -0.06), f"{adv:+.2f}", ha="center",
                va="bottom" if adv >= 0 else "top", fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(len(advantages)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set(xlabel=f"Rollout in the group (G = {len(advantages)})", ylabel="Group-relative advantage $A_i$",
           title="Only the passing kernel is reinforced; every other rollout is suppressed")
    ax.margins(y=0.18)
    plt.tight_layout()
    plt.show()


def bandit_training(history, n_actions, target_name):
    """Mean reward, P(target) and policy entropy over Bandit-GRPO training steps."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot([h["reward_mean"] for h in history], color=BLUE, linewidth=1.5)
    axes[0].set(title="Mean group reward", xlabel="Step", ylabel="reward", ylim=(-0.05, 1.05))
    axes[1].plot([h["p_correct"] for h in history], color=GREEN, linewidth=1.5)
    axes[1].set(title=f"P({target_name})", xlabel="Step", ylabel="probability", ylim=(-0.05, 1.05))
    axes[2].plot([h["entropy"] for h in history], color=ORANGE, linewidth=1.5)
    axes[2].axhline(np.log(n_actions), color="gray", linestyle="--", alpha=0.5, label="max entropy")
    axes[2].set(title="Policy entropy", xlabel="Step", ylabel="nats")
    axes[2].legend()
    fig.suptitle("Bandit-GRPO learns the passing kernel configuration", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def sequence_training(history, clip_bound=1.2):
    """Mean reward, largest sampled-token ratio and final-epoch clipped fraction per rollout batch."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot([h["reward_mean"] for h in history], color=BLUE)
    axes[0].set(title="Mean match reward", xlabel="Rollout batch", ylabel="reward", ylim=(-0.05, 1.05))
    axes[1].plot([h["max_ratio"] for h in history], color=ORANGE, label="largest token ratio")
    axes[1].axhline(clip_bound, color="#c0392b", linestyle="--", label="upper clip bound")
    axes[1].set(title="Ratios move away from 1", xlabel="Rollout batch", ylabel=r"$\pi_\theta / \pi_{old}$")
    axes[1].legend(fontsize=8)
    axes[2].plot([h["clip_fraction"] for h in history], color=PURPLE)
    axes[2].set(title="The clip bound activates", xlabel="Rollout batch", ylabel="clipped fraction (last epoch)",
                ylim=(-0.02, 1.02))
    fig.suptitle("Sequence-GRPO: one sampled group, eight optimizer epochs per batch", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------- failure lab
def zero_grad_sweep(success_rates, fractions_by_group):
    """Fraction of zero-gradient groups against success rate, one curve per group size."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (G, fracs), color in zip(fractions_by_group.items(), [RED, ORANGE, BLUE, GREEN, PURPLE]):
        ax.plot(success_rates, fracs, color=color, linewidth=2, label=f"G = {G}")
    ax.axvspan(0, 0.1, alpha=0.1, color="red")
    ax.axvspan(0.9, 1.0, alpha=0.1, color="red")
    ax.annotate("all wrong\n(too hard)", xy=(0.05, 0.85), ha="center", color="#c0392b", fontweight="bold")
    ax.annotate("all correct\n(too easy)", xy=(0.95, 0.85), ha="center", color="#c0392b", fontweight="bold")
    ax.annotate("rewards differ:\nthe group teaches", xy=(0.5, 0.05), ha="center", color="#27ae60",
                fontweight="bold")
    ax.set(xlabel="Per-response success probability p", ylabel="Fraction of zero-gradient groups",
           ylim=(-0.02, 1.02), title="Failure mode 1: when every rollout agrees, the group adds no gradient")
    ax.legend(title="Group size")
    plt.tight_layout()
    plt.show()


def clip_walkthrough(symmetric, clip_higher):
    """Rare winning token's ratio and probability, and the clipped fraction, over epochs on one frozen group."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for hist, label, color in [(symmetric, "symmetric 0.20/0.20", RED), (clip_higher, "Clip-Higher 0.20/0.50", GREEN)]:
        axes[0].plot([h["ratios"][5] for h in hist], color=color, label=label)
        axes[1].plot([h["probs"][1] for h in hist], color=color, label=label)
        axes[2].plot([h["clip_fraction"] for h in hist], color=color, label=label)
    axes[0].axhline(1.2, color=RED, linestyle="--", alpha=0.6)
    axes[0].axhline(1.5, color=GREEN, linestyle="--", alpha=0.6)
    axes[0].set(title="Rare winning token: actual ratio", xlabel="Optimizer epoch", ylabel=r"$p_{new} / p_{old}$")
    axes[1].set(title="Rare winning token: probability", xlabel="Optimizer epoch", ylabel="probability")
    axes[2].set(title="Surrogate terms clipped", xlabel="Optimizer epoch", ylabel="fraction")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("One frozen group, many optimizer epochs: the upper clip stops the rare token", fontweight="bold")
    plt.tight_layout()
    plt.show()


def clip_settings_comparison(results):
    """Entropy, clipped fraction and reward per batch; results maps a label to (history, color)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, (hist, color) in results.items():
        axes[0].plot([h["entropy"] for h in hist], color=color, label=label)
        axes[1].plot([h["clip_fraction"] for h in hist], color=color, label=label)
        axes[2].plot([h["reward_mean"] for h in hist], color=color, label=label)
    axes[0].set(title="Policy entropy (toy)", xlabel="Rollout batch", ylabel="nats")
    axes[1].set(title="Clipped fraction (last epoch)", xlabel="Rollout batch", ylabel="fraction")
    axes[2].set(title="Shaped match reward", xlabel="Rollout batch", ylabel="mean reward", ylim=(-0.02, 1.02))
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Clip-Higher on the sequence toy (seeded; observed behavior, not a guarantee)", fontweight="bold")
    plt.tight_layout()
    plt.show()


def normalization_batch(lengths, sample_per_token, token_per_token):
    """Per-token coefficient and total mass per response under sample-level and token-level normalization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x, width = np.arange(len(lengths)), 0.38
    axes[0].bar(x - width / 2, sample_per_token, width, label=r"sample: $1/(G|o_i|)$", color=RED)
    axes[0].bar(x + width / 2, token_per_token, width, label=r"token: $1/\sum_j|o_j|$", color=GREEN)
    axes[0].set(title="Per-token coefficient", xlabel="Response (tick = its length in tokens)", ylabel="coefficient")
    axes[1].bar(x - width / 2, lengths * sample_per_token, width, label="sample-level", color=RED)
    axes[1].bar(x + width / 2, lengths * token_per_token, width, label="token-level", color=GREEN)
    axes[1].set(title="Total objective mass per response", xlabel="Response", ylabel="sum of its coefficients")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(lengths, rotation=45)
        ax.legend(fontsize=8)
    fig.suptitle("Variable lengths: the two normalizations allocate the same total mass differently",
                 fontweight="bold")
    plt.tight_layout()
    plt.show()


def quartile_mass(masses):
    """Mean objective mass of the shortest and longest quartile under each normalization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    labels = ["shortest quartile", "longest quartile"]
    axes[0].bar(labels, masses["sample"], color=["#f5b7b1", RED])
    axes[0].set(title="Sample-level: equal mass per response", ylabel="mean objective mass", ylim=(0, 0.55))
    axes[1].bar(labels, masses["token"], color=["#abebc6", GREEN])
    axes[1].set(title="Token-level: equal mass per token")
    fig.suptitle("Weight allocation over 2,000 variable-length batches (toy)", fontweight="bold")
    plt.tight_layout()
    plt.show()


def dapo_mechanisms(success_rates, vanilla, dynamic, cost, clip_walks, masses):
    """Three DAPO mechanisms side by side: Dynamic Sampling, Clip-Higher, token-level mass."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(success_rates, vanilla, label="one attempt", color=RED)
    axes[0].plot(success_rates, dynamic, label="up to five attempts", color=GREEN)
    ax_cost = axes[0].twinx()
    ax_cost.plot(success_rates, cost, ":", color="#34495e")
    ax_cost.set_ylabel("responses generated per update (dotted)")
    ax_cost.grid(False)
    axes[0].set(title="Dynamic Sampling: failures and cost", xlabel="Success probability",
                ylabel="zero-gradient fraction")
    axes[0].legend(fontsize=8, loc="upper left")
    for label, (hist, color) in clip_walks.items():
        axes[1].plot([h["ratios"][5] for h in hist], color=color, label=label)
    axes[1].set(title="Clip-Higher: rare-token ratio (toy)", xlabel="Optimizer epoch", ylabel="actual ratio")
    axes[1].legend(fontsize=8)
    axes[2].bar(["sample\nshortest", "sample\nlongest", "token\nshortest", "token\nlongest"],
                [*masses["sample"], *masses["token"]], color=["#f5b7b1", RED, "#abebc6", GREEN])
    axes[2].set(title="Token-level loss: response mass", ylabel="mean objective mass")
    fig.suptitle("DAPO's mechanisms on seeded CPU toys (not an end-to-end benchmark)", fontweight="bold")
    plt.tight_layout()
    plt.show()


def overlong(lengths, penalties, l_max, l_cache):
    """DAPO's soft length penalty alone, and added to the task reward of a correct and a wrong response."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].plot(lengths, penalties, color=ORANGE, linewidth=2.5)
    axes[0].axvline(l_max - l_cache, color="gray", linestyle=":", label=r"$L_{max}-L_{cache}$")
    axes[0].axvline(l_max, color="#c0392b", linestyle="--", label=r"$L_{max}$")
    axes[0].set(title="Soft length penalty", xlabel="Response length (tokens)", ylabel=r"$r_{len}$", ylim=(-1.1, 0.1))
    axes[0].legend()
    for task_reward, color in [(1.0, GREEN), (0.0, BLUE)]:
        axes[1].plot(lengths, task_reward + penalties, color=color, linewidth=2,
                     label=fr"$r_{{task}}={task_reward:.0f}$")
    axes[1].axvline(l_max, color="#c0392b", linestyle="--")
    axes[1].set(title=r"Shaped reward $r_{task}+r_{len}$", xlabel="Response length (tokens)", ylabel="shaped reward")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def reward_hacking(history, probs_grid, target, vocab):
    """Reward under a flawed verifier, and the learned per-position action probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot([h["reward_mean"] for h in history], color=RED, linewidth=2)
    axes[0].set(title="Reward (looks great)", xlabel="Step", ylabel="mean reward", ylim=(-0.05, 1.05))
    im = axes[1].imshow(probs_grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    for t, token in enumerate(target):
        axes[1].plot(token, t, "g*", markersize=15)
    axes[1].set_xticks(range(len(vocab)))
    axes[1].set_xticklabels(vocab, rotation=30, ha="right", fontsize=8)
    axes[1].set(ylabel="Position", title="Learned policy (green stars = target)")
    axes[1].grid(False)
    plt.colorbar(im, ax=axes[1], label="probability")
    plt.tight_layout()
    plt.show()


def drgrpo(toy):
    """Per-question weight in expectation, and per-token penalty on an incorrect response, GRPO vs Dr. GRPO."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for (label, w), color in zip(toy["weights"].items(), [RED, PURPLE, GREEN]):
        axes[0].plot(toy["p"], w / w.max(), color=color, linewidth=2.5, label=label)
    axes[0].set(title="Per-question weight in expectation (scaled to max 1)", ylim=(0, 1.08),
                xlabel="Question accuracy $p$ under the current policy", ylabel="relative weight")
    axes[0].legend()
    axes[1].plot(toy["lengths"], np.abs(toy["per_token_grpo"]), "o-", color=RED, linewidth=2.5,
                 label=r"GRPO: $|A_i|\,/\,(G\,|o_i|)$")
    axes[1].plot(toy["lengths"], np.abs(toy["per_token_drgrpo"]), "s--", color=PURPLE, linewidth=2.5,
                 label=r"Dr. GRPO: $|A_i|\,/\,(G\,L_{\max})$")
    axes[1].set(xscale="log", yscale="log", title="Penalty per token on an incorrect response",
                xlabel=r"Length $|o_i|$ of the incorrect response (tokens)", ylabel="|weight| per token")
    axes[1].set_xticks(toy["lengths"], [str(L) for L in toy["lengths"]])
    axes[1].minorticks_off()
    axes[1].legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------- optional sliders
def _widgets():
    """(interact, FloatSlider, IntSlider) if ipywidgets is installed, else None after a short note."""
    try:
        from ipywidgets import FloatSlider, IntSlider, interact
    except ImportError:
        print("Optional: `pip install ipywidgets` for the slider version; the static plot above shows the same idea.")
        return None
    return interact, FloatSlider, IntSlider


def interactive_zero_grad():
    """Sliders for p and G: share of groups with identical rewards, and rollouts per informative group."""
    widgets = _widgets()
    if widgets:
        interact, FloatSlider, IntSlider = widgets

        def show(p=0.95, G=8):
            p_zero = p ** G + (1 - p) ** G
            print(f"p = {p:.2f}, G = {G}: {p_zero:.1%} of groups have identical rewards; "
                  f"about {G / max(1 - p_zero, 1e-12):.1f} rollouts per informative group")

        interact(show, p=FloatSlider(value=0.95, min=0.0, max=1.0, step=0.01), G=IntSlider(value=8, min=2, max=64))


def interactive_clip_toy():
    """Slider for eps_high on the frozen-group toy: how far the rare winning token's ratio can climb."""
    from .objectives import fixed_rollout_clip_toy
    widgets = _widgets()
    if widgets:
        interact, FloatSlider, _ = widgets

        def show(eps_high=0.28):
            hist = fixed_rollout_clip_toy(0.20, eps_high)
            fig, ax = plt.subplots(figsize=(7, 3.4))
            ax.plot([h["ratios"][5] for h in hist], color=GREEN, label="rare winning token")
            ax.axhline(1 + eps_high, color=RED, linestyle="--", label="upper clip bound")
            ax.set(xlabel="Optimizer epoch", ylabel=r"ratio $\pi_\theta/\pi_{old}$", title=f"eps_high = {eps_high:.2f}")
            ax.legend()
            plt.show()

        interact(show, eps_high=FloatSlider(value=0.28, min=0.10, max=0.80, step=0.02))


def interactive_normalization():
    """Slider for one response's length (seven others have 200 tokens): per-token weight under each scheme."""
    from .objectives import token_weights
    widgets = _widgets()
    if widgets:
        interact, _, IntSlider = widgets

        def show(length=20):
            lengths = np.array([length] + [200] * 7)
            for scheme in ("sample", "token", "constant"):
                w = token_weights(lengths, scheme, max_length=1000)
                print(f"{scheme:>8}: varied response {w[0]:.6f} per token, a 200-token response {w[1]:.6f} "
                      f"({w[0] / w[1]:.2f}x)")

        interact(show, length=IntSlider(value=20, min=5, max=1000, step=5))


# ----------------------------------------------------------------------------- S-GRPO notebook
def serial_group(labels, rewards, advantages):
    """Decaying reward and advantage for each exit of one S-GRPO serial group."""
    x, w = np.arange(len(labels)), 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, rewards, w, label="decaying reward $r_i$", color="#3a86c8")
    ax.bar(x + w / 2, advantages, w, label=r"advantage $\hat A_i$",
           color=[GREEN if a > 0 else (RED if a < 0 else "#95a5a6") for a in advantages])
    for xi, ai in zip(x, advantages):
        ax.text(xi + w / 2, ai + (0.03 if ai >= 0 else -0.03), f"{ai:+.2f}", ha="center",
                va="bottom" if ai >= 0 else "top", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set(ylabel="value", title="S-GRPO serial group: the earliest correct exit gets the largest advantage")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def exit_policy_training(depths, p_correct, exit_depth, hist_depth, hist_acc):
    """Accuracy by exit depth, expected exit depth over training, and the policy's accuracy over training."""
    n = depths[-1]
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4))
    a0.plot(depths, p_correct * 100, "o-", color="#8e44ad", lw=2)
    a0.axvline(exit_depth, ls="--", color=ORANGE, label=f"S-GRPO exit ~ {exit_depth:.1f}")
    a0.axvline(n, ls=":", color=GRAY, label="full chain")
    a0.set(xlabel="exit depth (reasoning length)", ylabel="P(correct) %",
           title="Accuracy is non-monotonic:\nsufficient, then overthinking")
    a0.legend()
    a1.plot(hist_depth, color=ORANGE, lw=2)
    a1.axhline(n, ls=":", color=GRAY, label=f"full chain = {n}")
    a1.set(xlabel="training step", ylabel="expected exit depth", title="Reasoning gets shorter")
    a1.legend()
    a2.plot(np.array(hist_acc) * 100, color="#3a86c8", lw=2, label="S-GRPO policy")
    a2.axhline(p_correct[-1] * 100, ls=":", color=GRAY, label="always full chain")
    a2.set(ylim=(0, 105), xlabel="training step", ylabel="accuracy (%)",
           title="...and accuracy is preserved (here, improved)")
    a2.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------- DisCO notebook
def disco_weights(p_grid, w_grpo_scaled, w_disco):
    """GRPO's per-question weight (scaled to 1 at p = 0.5) against DisCO's constant weight."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(p_grid, w_grpo_scaled, linewidth=3, label=r"GRPO: $\omega(q)=\sqrt{p(1-p)}$ (scaled to 1 at $p=0.5$)")
    ax.plot(p_grid, w_disco, linewidth=3, label=r"DisCO: $\omega(q)=1$, no weighting at all")
    ax.axvline(0.5, color="grey", linestyle=":", alpha=0.7)
    ax.set(xlabel="Question difficulty $p(q)$ = accuracy under the current policy",
           ylabel="Relative weight on the discriminative objective", ylim=(0, 1.15),
           title="GRPO's weight vanishes right where the hardest and easiest questions live")
    ax.legend()
    plt.show()


def question_race(trajectories):
    """Accuracy over training per question; trajectories maps label -> (p0, weighted, flat)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, (p0, weighted, flat) in trajectories.items():
        axes[0].plot(weighted, linewidth=2.5, label=f"{label} ($p_0={p0}$)")
        axes[1].plot(flat, linewidth=2.5, label=f"{label} ($p_0={p0}$)")
    axes[0].set_title(r"GRPO: push scaled by $\omega(p(t))$")
    axes[1].set_title("DisCO: push is flat, always")
    for ax in axes:
        ax.set(xlabel="training step", ylabel="question accuracy $p(q)$", ylim=(0, 1.02))
        ax.legend()
    plt.tight_layout()
    plt.show()


def score_gradients(x, grpo_clipped, eps):
    """d(score)/dx for GRPO's clipped score and DisCO's L-ratio and log-L scores."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, grpo_clipped, linewidth=3, label=r"GRPO: clipped L-ratio, $\min(x,1+\varepsilon)$")
    ax.plot(x, np.ones_like(x), linewidth=3, label="DisCO: L-ratio score, $x$")
    ax.plot(x, 1.0 / x, linewidth=3, label=r"DisCO: log-L score, $\log x$")
    ax.axvline(1 + eps, color="grey", linestyle=":", alpha=0.7)
    ax.set(xlabel=r"Probability ratio $x=\pi_\theta/\pi_{\mathrm{old}}$", ylabel="d(score)/dx",
           title="Clipped scores go flat past the window. Non-clipping scores never do.")
    ax.legend()
    plt.show()


def kl_penalty_vs_constraint(tt, hist_reg, hist_con, delta):
    """Divergence z and KL = z^2 / 2 over time: always-on penalty versus squared-hinge constraint."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for series, label in [(hist_reg, "KL regularization (always on)"),
                          (hist_con, "Squared-hinge constraint (only past the wall)")]:
        axes[0].plot(tt, series, linewidth=3, label=label)
        axes[1].plot(tt, series ** 2 / 2, linewidth=3, label=label)
    axes[1].axhline(delta, color="grey", linestyle=":", label=r"trust region $\delta$")
    axes[0].set(xlabel="training time", ylabel="$z$ (divergence from old policy)", title="Divergence over training")
    axes[1].set(xlabel="training time", ylabel="KL(old || theta)", title="KL divergence over training")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()


def dro_temperature(tau_grid, weights, j2_values, j1, n_negatives):
    """Softmax weight on the confuser and the DRO objective J2 against temperature tau."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(tau_grid, weights, linewidth=3)
    axes[0].axhline(1 / n_negatives, color="grey", linestyle=":", label=f"uniform weight, $1/{n_negatives}$")
    axes[0].set(xscale="log", xlabel=r"temperature $\tau$", ylabel="softmax weight on the confuser",
                title=r"Small $\tau$: almost all gradient attention on the hardest negative")
    axes[0].legend()
    axes[1].plot(tau_grid, j2_values, linewidth=3, label=r"$\mathcal{J}_2(\tau)$ (DRO / partial-AUC)")
    axes[1].axhline(j1, color="C1", linestyle="--", label=r"$\mathcal{J}_1$ (plain average)")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set(xscale="log", xlabel=r"temperature $\tau$", ylabel="objective value",
                title=r"$\mathcal{J}_2$ exposes the ranking violation, $\mathcal{J}_1$ hides it")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
    plt.tight_layout()
    plt.show()


def confuser_suppression(h_j1, h_j2, s_pos):
    """The confuser's score over training under the plain average (J1) and the DRO objective (J2)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(h_j1, linewidth=3, label=r"$\mathcal{J}_1$: plain average (uniform weight)")
    ax.plot(h_j2, linewidth=3, label=r"$\mathcal{J}_2$: DRO, $\tau=0.1$ (hard-negative weight)")
    ax.axhline(s_pos, color="grey", linestyle=":", label="the positive's score")
    ax.set(xlabel="training step", ylabel="confuser's score", title="Suppressing the one dangerous negative")
    ax.legend()
    plt.show()


def disco_training(runs, delta=0.01):
    """Accuracy per question for the GRPO-style control and DisCO, then DisCO's per-step KL (hard question)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (label, run) in zip(axes, runs.items()):
        ax.plot(run["grpo"][0], linewidth=2.5, label=r"GRPO-style ($\omega$-weighted, flat negatives)")
        ax.plot(run["disco"][0], linewidth=2.5, label=r"DisCO (no $\omega$, DRO negatives)")
        ax.axhline(0.9, color="grey", linestyle=":", label="90% accuracy")
        ax.set(title=f"{label} question", xlabel="training step", ylabel="accuracy $p(q)$", ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(runs["hard"]["disco"][1], linewidth=2, label=r"DisCO: KL(old || theta) per step")
    ax.axhline(delta, color="crimson", linestyle="--", label=rf"trust region $\delta={delta}$")
    ax.set(xlabel="training step", ylabel="KL divergence", title="The constraint only bites when it is crossed")
    ax.legend()
    plt.show()


# ----------------------------------------------------------------------------- DAPO notebook
def dapo_surrogates(ratios, grpo_objective, dapo_objective):
    """Positive-advantage clipped objective under GRPO's and DAPO's upper bounds."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ratios, grpo_objective, label="GRPO: upper bound 1.2", linewidth=3)
    ax.plot(ratios, dapo_objective, label="DAPO: upper bound 1.5", linewidth=3)
    ax.axvline(1.2, color="C0", linestyle="--", alpha=0.6)
    ax.axvline(1.5, color="C1", linestyle="--", alpha=0.6)
    ax.set(xlabel="Probability ratio: new / old", ylabel="Objective for positive advantage",
           title="Clip-Higher lets a successful token improve further")
    ax.legend()
    plt.show()


def rare_token_probability(grpo_history, dapo_history, p_old):
    """The rare token's probability over epochs; dashed lines mark each threshold p_old (1 + eps)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for hist, label in [(grpo_history, "GRPO: symmetric clipping"), (dapo_history, "DAPO: Clip-Higher")]:
        ax.plot([x["epoch"] for x in hist], [x["probability"] for x in hist], label=label, linewidth=3)
    ax.axhline(p_old * 1.2, color="C0", linestyle="--", alpha=0.6)
    ax.axhline(p_old * 1.5, color="C1", linestyle="--", alpha=0.6)
    ax.set(xlabel="Optimization epoch on the same rollout", ylabel='Probability of rare successful token "Draw"',
           title="Where does the positive-advantage update stop?")
    ax.legend()
    plt.show()


def informative_share(accuracy, prob_informative, group_sizes=(4, 8, 16, 64)):
    """P(group is informative) against per-response accuracy, one curve per group size."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for group_size in group_sizes:
        ax.plot(accuracy, prob_informative(accuracy, group_size), linewidth=2.5, label=f"G = {group_size}")
    ax.axvline(0.5, color="grey", linestyle=":", alpha=0.7)
    ax.set(xlabel="Per-response accuracy $p$ on this prompt", ylabel="P(group is informative)",
           title="A group teaches nothing unless its responses disagree", ylim=(-0.02, 1.02))
    ax.legend(title="responses per prompt")
    plt.show()


def effective_batch(run, prompt_batch):
    """Effective batch, relative gradient noise and generation cost over a run (output of dapo.effective_batch_run)."""
    steps, base = run["steps"], run["dynamic_noise"][0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(steps, run["naive_effective"], linewidth=3, label="Naive GRPO")
    axes[0].plot(steps, run["dynamic_effective"], linewidth=3, label="Dynamic Sampling")
    axes[0].axhline(prompt_batch, color="grey", linestyle=":", alpha=0.7)
    axes[0].set(ylabel="Prompts contributing gradient", title="Effective batch size", ylim=(0, prompt_batch * 1.08))
    axes[1].plot(steps, run["naive_noise"] / base, linewidth=3, label="Naive GRPO")
    axes[1].plot(steps, run["dynamic_noise"] / base, linewidth=3, label="Dynamic Sampling")
    axes[1].set(ylabel="Relative gradient noise", title=r"Gradient noise ($\propto 1/\sqrt{n_{\mathrm{eff}}}$)")
    axes[2].plot(steps, np.ones_like(steps, dtype=float), linewidth=3, label="Naive GRPO")
    axes[2].plot(steps, run["cost_multiplier"], linewidth=3, label="Dynamic Sampling")
    axes[2].set(ylabel="Responses generated / baseline", title="Generation cost per step")
    for ax in axes:
        ax.set_xlabel("Rollout step (policy improving ->)")
        ax.legend()
    plt.tight_layout()
    plt.show()


def interactive_sampling(prob_informative, prompt_batch=512):
    """Sliders for accuracy and group size: effective batch and Dynamic Sampling's generation multiplier.

    Without ipywidgets, prints a static sweep instead.
    """
    def sampling_report(accuracy=0.95, group_size=16):
        share = float(prob_informative(accuracy, group_size))
        n_eff = share * prompt_batch
        print(f"accuracy p = {accuracy:.2f}   G = {group_size}   prompt batch = {prompt_batch}")
        print(f"  P(group informative)          {share:6.1%}")
        print(f"  Naive GRPO effective batch    {n_eff:6.0f} / {prompt_batch} prompts")
        print(f"  Responses yielding no signal  {1.0 - share:6.1%}")
        if share > 0:
            print(f"  Dynamic Sampling generation   {1 / share:6.2f}x to refill the batch")
            print(f"  Naive GRPO noise penalty      {np.sqrt(1 / share):6.2f}x")
        else:
            print("  Dynamic Sampling cannot fill the batch: no informative groups exist.")
        fig, ax = plt.subplots(figsize=(7.5, 2.4))
        ax.barh(["Dynamic Sampling", "Naive GRPO"], [prompt_batch, n_eff], color=["C1", "C0"])
        ax.axvline(prompt_batch, color="grey", linestyle=":", alpha=0.8)
        ax.set(xlim=(0, prompt_batch * 1.05), xlabel="Prompts contributing gradient",
               title="Effective batch at this operating point")
        plt.tight_layout()
        plt.show()

    widgets = _widgets()
    if widgets:
        interact, FloatSlider, IntSlider = widgets
        interact(sampling_report,
                 accuracy=FloatSlider(value=0.95, min=0.0, max=1.0, step=0.01, description="accuracy p",
                                      continuous_update=False),
                 group_size=IntSlider(value=16, min=2, max=64, step=2, description="group size G",
                                      continuous_update=False))
        return
    print(f"{'p':>6s}" + "".join(f"{f'G={g}':>14s}" for g in (8, 16, 32)))
    for p in (0.50, 0.70, 0.90, 0.95, 0.99):
        cells = []
        for g in (8, 16, 32):
            share = float(prob_informative(p, g))
            cells.append(f"{share:5.1%} ({1 / share:.1f}x)" if share > 0 else "n/a")
        print(f"{p:6.2f}" + "".join(f"{c:>14s}" for c in cells))
    print("\nEach cell: P(informative), and in parentheses the generation multiplier to refill the batch.")


def length_weights(L_range, growing_sample, fixed_sample, token_level):
    """Per-token weights as one response grows: sample-level (its own tokens and a groupmate's) vs token-level."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(L_range, growing_sample, linewidth=3, label="sample-level: the growing response's own tokens")
    ax.plot(L_range, fixed_sample, linewidth=3, linestyle="--", label="sample-level: a fixed-length groupmate's tokens")
    ax.plot(L_range, token_level, linewidth=3, label="token-level: any token in the group (shared pool)")
    ax.set(xlabel="Length of the growing response (tokens)", ylabel="Per-token weight", yscale="log",
           title="Sample-level isolates length's cost to the response itself; token-level spreads it to everyone")
    ax.legend()
    ax.grid(alpha=0.2, which="both")
    plt.show()


def suppression_race(curves, p_old):
    """Probability of a repeated filler token over epochs; curves maps label -> history."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, hist in curves.items():
        ax.plot([x["epoch"] for x in hist], [x["probability"] for x in hist], linewidth=3, label=label)
    ax.axhline(p_old, color="grey", linestyle=":", alpha=0.6)
    ax.set(xlabel="Optimization epoch on the same rollout", ylabel='Probability of the repeated filler token "um"',
           title="A bad token buried in a long response is punished slowest under sample-level loss")
    ax.legend()
    plt.show()


def length_penalty_schemes(lengths, r_length, l_max, l_cache):
    """The length penalty alone, and a correct response's total reward, with and without it."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(lengths, np.zeros_like(lengths), linewidth=3, label="old: no length term")
    axes[0].plot(lengths, r_length, linewidth=3, label=r"new: $R_{\mathrm{length}}$ (Eq. 13)")
    axes[0].set(xlabel="Response length $|o_i|$", ylabel=r"$R_{\mathrm{length}}$", title="The length penalty by itself")
    axes[1].plot(lengths, np.ones_like(lengths), linewidth=3, label="old: correct response, any length")
    axes[1].plot(lengths, 1.0 + r_length, linewidth=3, label="new: correct response, shaped")
    axes[1].set(xlabel="Response length $|o_i|$", ylabel="Total reward $R_i$", title="A correct response's total reward")
    for ax in axes:
        ax.axvline(l_max - l_cache, color="grey", linestyle=":", alpha=0.7)
        ax.axvline(l_max, color="grey", linestyle=":", alpha=0.7)
        ax.legend()
    plt.tight_layout()
    plt.show()


def length_zones(steps, safe, cache, truncated):
    """Share of the batch in the safe zone, the cache window and at truncation, as lengths drift upward."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(steps, safe, cache, truncated,
                 labels=["safe: length term is 0", "cache window: graded penalty", "truncated: hard $-1$"],
                 colors=["C0", "C1", "C3"], alpha=0.85)
    ax.set(xlabel="Training step (mean response length drifting upward ->)", ylabel="Fraction of the batch",
           title="Under Soft Overlong Punishment, the cache window catches drift before truncation does", ylim=(0, 1))
    ax.legend(loc="upper left")
    plt.show()


# ----------------------------------------------------------------------------- method testbed
def testbed_comparison(results, difficulties=None):
    """Same-seed runs of several configs: accuracy (all prompts and the hardest), truncation, length,
    entropy, and responses generated per step. results: {name: history from testbed.run}."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
    panels = [("acc", "Mean accuracy over prompts", "P(correct)"),
              ("hardest", "Accuracy on the hardest prompt", "P(correct)"),
              ("truncated", "Responses truncated at L_max", "P(truncated)"),
              ("length", "Mean response length", "tokens"),
              ("entropy", "Mean token entropy", "nats"),
              ("generated", "Responses generated per step", "responses")]
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        for name, h in results.items():
            series = h["acc_by_prompt"][:, -1] if key == "hardest" else h[key]
            ax.plot(series, linewidth=2, label=name)
        hardest = f" (d = {difficulties[-1]})" if key == "hardest" and difficulties else ""
        ax.set(title=title + hardest, xlabel="training step", ylabel=ylabel)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Same seed, same toy, one flag set per method", fontweight="bold")
    plt.tight_layout()
    plt.show()
