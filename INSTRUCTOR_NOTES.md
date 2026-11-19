# Instructor notes

Session plans, the misconceptions each topic has to correct, a short activity per topic, and assessment
ideas. The misconceptions are collected from the reviews of this resource and from the places where the
notebooks have to correct a natural but wrong reading.

## Planning a session

| Format | Path | Time |
|---|---|---|
| Lecture slot | GRPO deck, then the DAPO deck's results slide | 30 min |
| Lab session | GRPO deck → `01_grpo_core` → `02_failure_lab` §1–4 → `03_exercises` 1 and 6 | 90 min |
| Two sessions | add `00_pg_ppo_refresher` before, `04_method_testbed` and one focused notebook after | 2 × 90 min |

Everything runs on a laptop CPU or on Colab (badge at the top of each notebook); nothing needs a GPU or an
API key. Each notebook states its reading time and runtime in its first cell.

For undergraduates, keep the refresher and the by-hand calculations. Practitioners who already know PPO can
start at `02_failure_lab` and spend the time saved in `04_method_testbed`.

## Topic notes

### Policy gradients and PPO (`00_pg_ppo_refresher`, deck `#policy-gradient`, `#ppo-refresher`)

- **"PPO's clip caps the probability."** It caps the *ratio* to the old policy within one update, so a rare
  token can only grow by a factor of $1+\varepsilon$ per batch. This sets up failure mode 2 (F5).
- **"A baseline changes what the gradient points at."** It leaves the expected gradient unchanged and lowers
  its variance (the 90.8% variance reduction in §1.3).
- **"The KL penalty and the clip do the same job."** The clip bounds each update; the KL term anchors the
  policy to a reference over the whole run.

One-sentence intuition for the importance ratio: it says how much more (or less) likely the current policy
makes a sampled token than the policy that generated it, and the clip removes the incentive to move that
token further once the ratio leaves $[1-\varepsilon, 1+\varepsilon]$.

### GRPO core (`01_grpo_core`, deck `#grpo-step`, `#score-rollouts`)

- **"GRPO is RL with a verifier."** These are two independent choices: GRPO removes the critic (the group
  mean is the baseline); RLVR replaces the learned reward model with a program (F2).
- **"Each token gets its own advantage."** One advantage per response, shared by all of its tokens.
- **"The clip is active from the first update."** All ratios equal 1 on the first epoch over a batch; the
  clip binds only when the same rollouts are reused (01 §3, 02 §2).
- **"Normalizing by the group's std is harmless."** It weights each question by $\sqrt{p(1-p)}$, the
  difficulty bias DisCO removes (F7).

At the board: work the 2-of-8 group (F3) by hand. The mean reward is 0.25 and the standard deviation about
0.433, so the passes get +1.73 and the fails −0.58, and the eight advantages sum to zero.

### Failure 1: zero-gradient groups (`02_failure_lab` §1, deck `#failure-zero-gradient`)

- **"A larger group fixes it."** It delays the problem: at $p=0.95$, 66% of groups are uninformative with
  $G=8$ and 4% with $G=64$, at eight times the generation per group (F4).
- **"Dynamic Sampling is free."** Discarded groups were still generated; the DAPO toy needs up to 2.3× the
  rollouts per kept prompt late in a run (F9), and the testbed about 3× (`04_method_testbed`).

Activity: before showing F4, ask learners to guess what share of groups still carries a gradient at
$p = 0.95$ with $G = 8$ (34%).

### Failure 2: the upper clip and exploration (`02_failure_lab` §2, deck `#failure-clip-headroom`)

- **"Clip-Higher raises entropy."** It gives rare, rewarded tokens more headroom per update (F5). Whether
  entropy stays higher is an empirical claim from DAPO's runs; in the testbed toy it does not.

Activity: have learners compute the largest gain one update allows for a token at $p_\text{old} = 0.01$ and
at $0.5$ with $\varepsilon = 0.2$ (+0.002 and +0.10, F5) before introducing Clip-Higher.

### Failure 3: length-dependent token weights (`02_failure_lab` §3, deck `#failure-length-weighting`)

- **"Sample- vs token-level averaging only matters for very long reasoning."** It matters whenever response
  lengths differ within a batch, and the two coincide exactly when lengths are equal (F6).
- **"Longer wrong answers are penalized more."** Under sample-level averaging each of their tokens is
  penalized *less*, which is how length can drift upward.

Activity: in a group where seven responses have 200 tokens and one has 20, how much more does each token of
the short response weigh under sample-level averaging? (10 times, F6.)

### DAPO (`dapo_four_fixes`, DAPO deck)

- **"Four independent tricks."** Each targets one failure, but they interact: Dynamic Sampling changes which
  groups the token-level average runs over, and overlong *filtering* (drop truncated responses) is a
  different mechanism from overlong *shaping* (graded penalty, F8).
- **"DAPO's 50 points come from one trick."** Every component adds points in the paper's ablation; Dynamic
  Sampling adds the most (F14a).

Activity: read the ablation (F14a) as a table of marginal gains and ask which single change learners would
keep if they could keep only one, and what it costs.

### DisCO (`disco_three_fixes`, DisCO deck)

- **"DisCO is a different kind of algorithm."** With binary rewards, GRPO's objective already *is* a
  difficulty-weighted discriminative objective (F7); DisCO removes the weight, the clipping and the uniform
  averaging over negatives.
- **"Removing the clip makes training unstable."** DisCO replaces it with an explicit KL constraint that
  acts only when the policy moves too far.

Contrast with DAPO on the same symptom (F5): DAPO widens the clip, while DisCO removes it and constrains the
policy with KL instead.

### S-GRPO (`sgrpo_early_exit`, S-GRPO deck)

- **"S-GRPO penalizes long reasoning."** The decaying reward ranks *correct* exits by position; a wrong
  early exit still earns 0, so stopping too early is never rewarded (F12).

At the board: work the F12 example. The rewards 0, 1, 1/2, 1/4 and 1/8 have mean 0.375, so only the first
two correct exits get a positive advantage (+0.625 and +0.125).

### The method testbed (`04_method_testbed`)

- **"These are five algorithms to memorize."** They are settings of one objective; the `diff` view lists
  the two to four switches that separate any two of them.
- **"The toy's ranking predicts LLM results."** It shows mechanisms on a 60-entry table; scale effects come
  from the papers (F14).

Group activity: each group takes one flag, predicts its effect on accuracy, response length and entropy,
then runs it against GRPO on the same seed and explains any surprise.

## Assessment ideas

- Exercises 1 and 6 in `03_exercises` work as a five-minute exit ticket.
- Ask learners to name the flag (testbed §2) that addresses a symptom: "all-correct groups", "entropy
  collapse", "length drift", "truncated answers scored as wrong".
- The misconceptions above can seed a pre/post quiz: state each one and ask whether it is true and why.
