#!/usr/bin/env python3
"""Build the NeurIPS 2026 Education Track poster: 61 cm wide x 85 cm high, portrait.

poster/poster.html holds the layout and the text. This script fills in what is computed: the six failure
charts and the DAPO ablation chart (drawn with Matplotlib from the grpo_lab functions the notebooks use), the
method table (from grpo_lab.testbed.PRESETS), KaTeX and the QR codes. It then prints the page with Chrome.

Outputs (in poster/):
    grpo_then_and_now_poster.pdf          the print file, 610 mm x 850 mm
    grpo_then_and_now_poster_virtual.png  neurips.cc virtual poster: a landscape PNG within 5120 x 2880 px
    grpo_then_and_now_thumbnail.png       neurips.cc thumbnail, 320 x 256 px
and build/poster/poster.html, the self-contained page that was printed.

Needs the packages in requirements.txt and Chrome, Chromium or Edge (on WSL the Windows browser is used, as
for the handouts). The QR codes (poster/qr_site.svg, poster/qr_code.svg) are committed; if a link changes,
redraw them with `--qr`, which needs segno (pip install segno==1.6.6).

Usage:
    .venv/bin/python poster/build_poster.py          (or: make poster)
    .venv/bin/python poster/build_poster.py --html   only write build/poster/poster.html (no browser)
    python poster/build_poster.py --qr               redraw the QR codes
"""
import html
import io
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = ROOT / "poster"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "slides"))

TEMPLATE = HERE / "poster.html"
OUT_HTML = ROOT / "build" / "poster" / "poster.html"
PDF = HERE / "grpo_then_and_now_poster.pdf"
VIRTUAL_PNG = HERE / "grpo_then_and_now_poster_virtual.png"
THUMBNAIL = HERE / "grpo_then_and_now_thumbnail.png"
LINKS = {"site": "https://samaksh97.github.io/grpo-then-and-now/",
         "code": "https://github.com/samaksh97/grpo-then-and-now"}
PAGE_PX = (2306, 3213)        # 610 mm x 850 mm at 96 CSS px per inch
VIRTUAL_MAX = (5120, 2880)    # neurips.cc virtual poster limit (landscape PNG, at most 10 MB)
THUMB_SIZE = (320, 256)       # neurips.cc thumbnail limit
CANVAS = "#dfe5ee"            # solid background around the portrait page on the landscape canvas

INK, MUTED, GRID = "#0f1b2d", "#4b5a6e", "#e3e8ef"
RED, LIGHT_RED, DAPO, DISCO, SGRPO, GREY = "#b42318", "#e8a29a", "#157347", "#5b45c9", "#b45309", "#7b8794"
FIGSIZE = (6.6, 2.6)          # inches; shown about 660 CSS px wide, so 16 pt text is about 22 px on the page

# Short TRL GRPOConfig names for the table's last row; tests/test_poster.py checks them against
# grpo_lab.testbed.LIBRARY_SETTINGS. None = no built-in TRL flag.
TRL_NAMES = {"adv_norm": "scale_rewards", "aggregation": "loss_type", "eps": "epsilon, epsilon_high",
             "ratio_level": "importance_sampling_level", "clip_mode": 'loss_type="cispo"',
             "dynamic_sampling": None, "overlong": "get_soft_overlong_punishment"}
TABLE_COLUMNS = [("adv_norm", "advantage"), ("aggregation", "loss average"), ("eps", "clip ε<sub>low</sub> / ε<sub>high</sub>"),
                 ("ratio_level", "ratio"), ("clip_mode", "past the clip"), ("dynamic_sampling", "dyn. sampling"),
                 ("overlong", "overlong")]
TABLE_WORDS = {"adv_norm": {"std": "÷ std", "none": "mean only"},
               "aggregation": {"sample": "per answer", "token": "per token", "constant": "constant"},
               "ratio_level": {"token": "token", "sequence": "sequence"},
               "clip_mode": {"drop": "no gradient", "is_weight": "IS-weighted"},
               "dynamic_sampling": {False: "—", True: "on"},
               "overlong": {"none": "—", "shape": "soft penalty", "filter": "masked"}}


# ----------------------------------------------------------------------------------------------- charts
def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "svg.fonttype": "none", "svg.hashsalt": "grpo-then-and-now-poster",
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Inter", "Arial", "DejaVu Sans"],
        "font.size": 16, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 15, "legend.frameon": False, "legend.handlelength": 1.6,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9, "lines.linewidth": 3.2,
        "lines.solid_capstyle": "round",
    })
    return plt


def _svg(fig, name):
    """The figure as an inline <svg> that scales with its box, with ids scoped to this chart."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True, metadata={"Date": None})
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    root, rest = svg.split(">", 1)
    root = re.sub(r'\s(width|height)="[^"]*"', "", root)
    svg = f'{root} role="img" aria-label="{html.escape(name)}">{rest}'
    for ident in sorted(set(re.findall(r'\sid="([^"]+)"', svg)), key=len, reverse=True):
        for old, new in ((f'id="{ident}"', f'id="{name}-{ident}"'), (f'#{ident}"', f'#{name}-{ident}"'),
                         (f'#{ident})', f'#{name}-{ident})')):
            svg = svg.replace(old, new)
    return svg


def chart_no_gradient(plt, np):
    """Failure 1: share of groups whose G binary rewards are all equal, p^G + (1 - p)^G (F4)."""
    from grpo_lab.objectives import zero_gradient_probability
    p = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(p, zero_gradient_probability(p, 8), color=RED, label="GRPO, G = 8")
    ax.plot(p, zero_gradient_probability(p, 64), color=LIGHT_RED, lw=2.4, label="GRPO, G = 64")
    ax.plot(p, np.zeros_like(p), color=DAPO, ls="--", label="after Dynamic Sampling")
    share = zero_gradient_probability(0.95, 8)
    ax.scatter([0.95], [share], s=60, color=RED, zorder=5)
    ax.annotate(f"p = 0.95, G = 8:\n{share:.0%} of groups add nothing", xy=(0.95, share), xytext=(0.36, 0.70),
                fontsize=15, ha="left", arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4})
    ax.set(xlabel="success rate p on the prompt", ylabel="no-gradient groups", xlim=(0, 1), ylim=(-0.04, 1.04))
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.33))
    return fig


def chart_clip(plt, np):
    """Failure 2: largest probability gain one update allows before the clip binds, min(eps p, 1 - p) (F5)."""
    p = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(p, np.minimum(0.2 * p, 1 - p), color=RED, label="GRPO, ε = 0.2")
    ax.plot(p, np.minimum(0.28 * p, 1 - p), color=DAPO, label="Clip-Higher, upper ε = 0.28")
    ax.scatter([0.01, 0.5], [0.2 * 0.01, 0.2 * 0.5], s=60, color=RED, zorder=5)
    ax.annotate("rare token, p = 0.01: +0.002", xy=(0.01, 0.002), xytext=(0.2, 0.012), fontsize=15,
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4})
    ax.annotate("p = 0.5: +0.10", xy=(0.5, 0.10), xytext=(0.58, 0.062), fontsize=15,
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4})
    ax.set(xlabel="token probability before the update", ylabel="max gain per update", xlim=(0, 1),
           ylim=(0, 0.25))
    ax.legend(loc="upper left")
    return fig


def chart_length(plt, np):
    """Failure 3: per-token weight of one response whose length varies; the other 7 have 200 tokens (F6)."""
    from grpo_lab.objectives import token_weights
    lengths = np.unique(np.round(np.geomspace(10, 2000, 160)).astype(int))

    def relative(scheme, length):
        group = [length] + [200] * 7
        return token_weights(group, scheme, 2048)[0] / token_weights([200] * 8, scheme, 2048)[0]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(lengths, [relative("sample", n) for n in lengths], color=RED, label="GRPO: average per answer")
    ax.plot(lengths, [relative("token", n) for n in lengths], color=DAPO, label="DAPO: average per token")
    ax.plot(lengths, [relative("constant", n) for n in lengths], color=GREY, ls="--", lw=2.6,
            label="Dr. GRPO: constant divisor")
    ax.scatter([20], [relative("sample", 20)], s=60, color=RED, zorder=5)
    ax.annotate("a 20-token answer:\n10× weight per token", xy=(20, 10), xytext=(26, 0.25), fontsize=15, va="center",
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.4})
    ax.set(xscale="log", yscale="log", xlabel="length of one answer (the other 7 have 200 tokens)",
           ylabel="weight per token", xlim=(10, 2000), ylim=(0.05, 400))
    ax.set_yticks([0.1, 1, 10], ["0.1×", "1×", "10×"])
    ax.set_xticks([10, 100, 1000], ["10", "100", "1,000"])
    ax.legend(loc="upper right", fontsize=14)
    return fig


def chart_overlong(plt, np):
    """Failure 4: DAPO's overlong reward shaping (Eq. 13) with L_max = 20,480 and a 4,096-token window (F8)."""
    from grpo_lab.objectives import overlong_penalty
    l_max, l_cache = 20480, 4096
    lengths = np.linspace(9000, 23000, 700)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axvspan((l_max - l_cache) / 1000, l_max / 1000, color=DAPO, alpha=0.10, lw=0)
    ax.plot(lengths / 1000, [overlong_penalty(n, l_max, l_cache) for n in lengths], color=DAPO,
            label="DAPO: graded penalty")
    ax.axvline(l_max / 1000, color=RED, ls="--", lw=2.6, label="hard cut: truncated = wrong")
    ax.text((l_max - l_cache / 2) / 1000, 0.13, "4,096-token\nwarning zone", ha="center", fontsize=15, color=DAPO)
    ax.set(xlabel="response length (thousand tokens)", ylabel="length penalty", xlim=(9, 23), ylim=(-1.2, 0.42))
    ax.set_xticks([9, 16.384, 20.48], ["9", "16.4", "20.5"])
    ax.legend(loc="lower left")
    return fig


def chart_difficulty(plt, np):
    """Failure 5: per-question weight against accuracy p: GRPO sqrt(p(1-p)), Dr. GRPO p(1-p), DisCO 1 (F7)."""
    p = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(p, np.ones_like(p), color=DISCO)
    ax.plot(p, np.sqrt(p * (1 - p)), color=RED)
    ax.plot(p, p * (1 - p), color=GREY, ls="--", lw=2.6)
    for y, text, color in ((1.04, "DisCO: every question counts the same", DISCO),
                           (0.55, "GRPO: √(p(1−p))", RED), (0.29, "Dr. GRPO, no std: p(1−p)", GREY)):
        ax.text(0.5, y, text, ha="center", va="bottom", fontsize=15, color=color, fontweight="bold")
    ax.set(xlabel="question accuracy p", ylabel="question weight", xlim=(0, 1), ylim=(0, 1.2))
    return fig


def chart_serial_group(plt, np):
    """Failure 6: S-GRPO's serial group (F12): exits along one chain earn 0, 1, 1/2, 1/4, 1/8."""
    exits = ["20%\nwrong", "40%\nright", "60%\nright", "80%\nright", "100%\nright"]
    decaying, labels = [0, 1, 0.5, 0.25, 0.125], ["0", "1", "1/2", "1/4", "1/8"]
    x = np.arange(len(exits))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x, decaying, width=0.58, color=SGRPO, label="S-GRPO: decaying reward", zorder=3)
    ax.plot(x, [0, 1, 1, 1, 1], color=RED, ls="--", marker="o", ms=8, lw=2.4, label="outcome reward", zorder=4)
    for xi, value, text in zip(x, decaying, labels, strict=True):
        ax.text(xi, value + 0.05, text, ha="center", fontsize=16, fontweight="bold", color=SGRPO)
    ax.set_xticks(x, exits)
    ax.set(xlabel="exit point along one chain of thought", ylabel="reward", ylim=(0, 1.45))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0)
    return fig


def chart_dapo_ablation(plt, np):
    """DAPO's cumulative ablation on AIME 2024, avg@32, Qwen2.5-32B base (DAPO Table 1; F14a)."""
    steps = ["naive GRPO", "+ overlong filtering", "+ Clip-Higher", "+ soft overlong punishment",
             "+ token-level loss", "+ Dynamic Sampling"]
    scores = [30, 36, 38, 41, 42, 50]
    y = np.arange(len(steps))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.barh(y, scores, height=0.7, color=[GREY] + [DAPO] * 5, zorder=3)
    for yi, step, score in zip(y, steps, scores, strict=True):
        ax.text(0.8, yi, step, va="center", fontsize=14, color="white", zorder=4)
        ax.text(score + 0.6, yi, str(score), va="center", fontsize=16, fontweight="bold", zorder=4)
    ax.axvline(47, color=INK, ls=":", lw=2, zorder=2)
    ax.text(46.4, y[0] + 0.55, "DeepSeek-R1-Zero-Qwen-32B: 47", ha="right", va="bottom", fontsize=13)
    ax.set_yticks([])
    ax.set(xlabel="AIME 2024, avg@32", xlim=(0, 55), ylim=(-0.5, len(steps) - 0.05))
    ax.grid(axis="y", visible=False)
    return fig


CHARTS = {"no_gradient": chart_no_gradient, "clip": chart_clip, "length": chart_length,
          "overlong": chart_overlong, "difficulty": chart_difficulty, "serial_group": chart_serial_group,
          "dapo_ablation": chart_dapo_ablation}


def charts():
    import numpy as np
    plt = _pyplot()
    out = {}
    for name, draw in CHARTS.items():
        fig = draw(plt, np)
        fig.tight_layout(pad=0.4)
        out[name] = _svg(fig, name)
        plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------ method table
def _cell(cfg, key):
    if key == "eps":
        return f"{cfg.eps_low:g} / {cfg.eps_high:g}"
    return TABLE_WORDS[key][getattr(cfg, key)]


def testbed_table():
    """Every preset of grpo_lab.testbed as one row; cells that differ from GRPO are highlighted."""
    from grpo_lab.testbed import PRESETS
    base = PRESETS["GRPO"]
    head = "<tr><th>method</th>" + "".join(f"<th>{label}</th>" for _, label in TABLE_COLUMNS) + "</tr>"
    rows = []
    for name, cfg in PRESETS.items():
        cells = []
        for key, _ in TABLE_COLUMNS:
            value = _cell(cfg, key)
            changed = name != "GRPO" and value != _cell(base, key)
            cells.append(f'<td{" class=chg" if changed else ""}>{html.escape(value)}</td>')
        rows.append(f"<tr><th>{html.escape(name)}</th>{''.join(cells)}</tr>")
    trl = "".join(f"<td><code>{html.escape(TRL_NAMES[key])}</code></td>" if TRL_NAMES[key] else "<td>no flag</td>"
                  for key, _ in TABLE_COLUMNS)
    rows.append(f'<tr class="trl"><th>TRL option</th>{trl}</tr>')
    return f'<table class="testbed"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


# -------------------------------------------------------------------------------------------- QR codes
def write_qr_codes():
    import segno
    for key, url in LINKS.items():
        segno.make(url, error="m").save(str(HERE / f"qr_{key}.svg"), kind="svg", border=2, dark=INK, light="#fff",
                                       xmldecl=False, omitsize=True, nl=False)
        print(f"  qr_{key}.svg -> {url}")


def qr_svg(key):
    svg = (HERE / f"qr_{key}.svg").read_text(encoding="utf-8")
    return svg[svg.index("<svg"):].replace("<svg ", f'<svg role="img" aria-label="QR code: {LINKS[key]}" ', 1)


# ------------------------------------------------------------------------------------------------ page
def render_html():
    """The self-contained poster page: template + charts + table + QR codes + KaTeX."""
    from build_standalone import KATEX, LICENSE_NOTE, inline_fonts, inline_script
    js = (KATEX / "katex.min.js").read_text(encoding="utf-8")
    note = LICENSE_NOTE.format(version=re.search(r'version:"([0-9.]+)"', js).group(1))
    katex = (f"<style>/*! {note} */\n{inline_fonts((KATEX / 'katex.min.css').read_text(encoding='utf-8'))}\n</style>\n"
             + inline_script(f"/*! {note} */\n{js}"))
    fills = {"KATEX": katex, "TESTBED_TABLE": testbed_table(), "QR:site": qr_svg("site"), "QR:code": qr_svg("code")}
    fills.update({f"CHART:{name}": svg for name, svg in charts().items()})
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in fills.items():
        token = "{{" + key + "}}"
        if page.count(token) != 1:
            raise RuntimeError(f"poster.html must contain {token} exactly once")
        page = page.replace(token, value)
    left = re.findall(r"\{\{[A-Za-z_:]+\}\}", page)
    if left:
        raise RuntimeError(f"unfilled placeholders: {left}")
    return page


def print_page(page_html):
    """Print the page to PDF and take a full-page screenshot; returns the screenshot as a PIL image."""
    from PIL import Image
    from build_handouts import find_browser, windows_staging
    browser = find_browser()
    staging, host = windows_staging() if browser.endswith(".exe") else (tempfile.TemporaryDirectory(), str)
    with staging as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "poster.html").write_text(page_html, encoding="utf-8")
        url = "file:///" + host(tmp / "poster.html").replace("\\", "/").lstrip("/")
        common = [browser, "--headless=new", "--disable-gpu", "--no-first-run",
                  f"--user-data-dir={host(tmp / 'profile')}", "--virtual-time-budget=5000"]
        subprocess.run(common + ["--no-pdf-header-footer", f"--print-to-pdf={host(tmp / 'poster.pdf')}", url],
                       check=True, capture_output=True)
        subprocess.run(common + ["--hide-scrollbars", "--force-device-scale-factor=1",
                                 f"--window-size={PAGE_PX[0]},{PAGE_PX[1]}",
                                 f"--screenshot={host(tmp / 'poster.png')}", url], check=True, capture_output=True)
        shutil.copyfile(tmp / "poster.pdf", PDF)
        return Image.open(tmp / "poster.png").convert("RGB")


def write_pngs(shot):
    """The neurips.cc uploads: the portrait page centred on a landscape canvas, and a thumbnail of the title."""
    from PIL import Image
    scale = min(VIRTUAL_MAX[0] / shot.width, VIRTUAL_MAX[1] / shot.height, 1)
    page = shot.resize((round(shot.width * scale), round(shot.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", VIRTUAL_MAX, CANVAS)
    canvas.paste(page, ((canvas.width - page.width) // 2, (canvas.height - page.height) // 2))
    canvas.save(VIRTUAL_PNG, optimize=True)
    top = shot.crop((0, 0, shot.width, round(shot.width * THUMB_SIZE[1] / THUMB_SIZE[0])))
    top.resize(THUMB_SIZE, Image.LANCZOS).save(THUMBNAIL, optimize=True)


def main(args):
    if "--qr" in args:
        write_qr_codes()
        return
    page = render_html()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")
    print(f"  {OUT_HTML.relative_to(ROOT)}: {len(page) / 1e6:.2f} MB, self-contained")
    if "--html" in args:
        return
    write_pngs(print_page(page))
    for path in (PDF, VIRTUAL_PNG, THUMBNAIL):
        print(f"  {path.relative_to(ROOT)}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1:])
