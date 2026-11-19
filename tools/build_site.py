"""Build the static site published on GitHub Pages: a landing page, the four self-contained decks, their PDF
handouts and the poster PDF.

Usage:
    python tools/build_site.py [OUTPUT_DIR]      (default: build/site)
"""
import html
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "slides"))
from build_standalone import build  # noqa: E402

REPO = "https://github.com/samaksh97/grpo-then-and-now"
COLAB = "https://colab.research.google.com/github/samaksh97/grpo-then-and-now/blob/main/notebook/"
DECKS = [
    ("grpo", "GRPO", "Start here. How a group of sampled answers replaces PPO's critic, and three places where the "
     "learning signal breaks.", "about 15 min"),
    ("dapo", "DAPO", "Four fixes to GRPO for long reasoning: Clip-Higher, Dynamic Sampling, token-level loss and "
     "overlong reward shaping.", "about 15 min"),
    ("disco", "DisCO", "GRPO read as discriminative learning: no difficulty weight, no clipping, and a softmax over "
     "hard negatives.", "about 15 min"),
    ("sgrpo", "S-GRPO", "One chain of thought, several exits: a decaying reward that teaches a model when to stop.",
     "about 10 min"),
]
NOTEBOOKS = [
    ("00_pg_ppo_refresher", "Optional refresher: policy gradients, PPO, and what GRPO removes", "20 min"),
    ("01_grpo_core", "GRPO by hand: group-relative advantages and the clipped update", "30 min"),
    ("02_failure_lab", "Where the learning signal breaks, and how DAPO responds", "40 min"),
    ("03_exercises", "Six exercises with reference solutions", "15-45 min"),
    ("04_method_testbed", "Every method as a configuration of one objective, on the same seed", "25 min"),
    ("dapo_four_fixes", "DAPO's four fixes, each with its own experiment", "45 min"),
    ("disco_three_fixes", "DisCO's three fixes, with the code behind them", "45 min"),
    ("sgrpo_early_exit", "S-GRPO by hand: teaching a model to exit early", "20 min"),
]

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GRPO, Then and Now</title>
<style>
body{{margin:0;background:#08111f;color:#eaf2ff;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:48px 24px 64px}}
h1{{font-size:2.4rem;margin:.2em 0}} h2{{margin-top:2.2em;color:#58d8ff;font-size:1.1rem;letter-spacing:.12em;text-transform:uppercase}}
a{{color:#58d8ff}} .muted{{color:#9bb0ca}}
.decks{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}
.deck{{background:#101d31;border:1px solid #23395a;border-radius:16px;padding:18px}}
.deck h3{{margin:0 0 6px}} .deck p{{margin:0 0 12px;color:#c3d2e6;font-size:.95rem}}
table{{width:100%;border-collapse:collapse}} td{{border-bottom:1px solid #23395a;padding:10px 8px;vertical-align:top}}
code{{background:#101d31;padding:2px 6px;border-radius:6px}}
</style></head><body><main>
<p class="muted">NeurIPS 2026 Education Track</p>
<h1>GRPO, Then and Now</h1>
<p>An interactive guide to modern RL for reasoning: four slide decks and eight CPU-only notebooks on Group
Relative Policy Optimization, its failure modes, and the methods that revise it (DAPO, DisCO, S-GRPO).</p>
<p class="muted">Samaksh Gulati and Rachel Shalom &middot; <a href="{repo}">source on GitHub</a> &middot;
<a href="grpo_then_and_now_poster.pdf">NeurIPS poster (PDF)</a> &middot; code Apache-2.0, content CC BY 4.0</p>
<h2>Slide decks</h2>
<p class="muted">Each deck works offline; press T to switch between dark and light. The PDF handouts show
every slide in its default state.</p>
<div class="decks">{decks}</div>
<h2>Notebooks</h2>
<p class="muted">Run them on Colab with one click, or locally with <code>pip install -r requirements.txt</code>.
Suggested 50-minute path: GRPO deck, then 01, 02, and Exercises 1 and 6 in 03.</p>
<table>{notebooks}</table>
</main></body></html>
"""


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build" / "site")
    if out.exists():
        shutil.rmtree(out)
    build(out / "slides")
    shutil.copytree(ROOT / "slides" / "handouts", out / "slides" / "handouts")
    shutil.copyfile(ROOT / "poster" / "grpo_then_and_now_poster.pdf", out / "grpo_then_and_now_poster.pdf")
    decks = "".join(
        f'<div class="deck"><h3>{name}</h3><p>{html.escape(blurb)}</p><p class="muted">{minutes}</p>'
        f'<a href="slides/{key}_intro.html">Open the deck</a> &middot; '
        f'<a href="slides/handouts/{key}_handout.pdf">PDF handout</a></div>'
        for key, name, blurb, minutes in DECKS)
    notebooks = "".join(
        f'<tr><td><a href="{COLAB}{key}.ipynb">{key}</a></td><td>{html.escape(blurb)}</td>'
        f'<td class="muted">{minutes}</td></tr>' for key, blurb, minutes in NOTEBOOKS)
    (out / "index.html").write_text(PAGE.format(repo=REPO, decks=decks, notebooks=notebooks), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"site written to {out}")


if __name__ == "__main__":
    main()
