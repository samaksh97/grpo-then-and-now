#!/usr/bin/env python3
"""Build self-contained copies of the four slide decks.

In this folder each deck loads KaTeX from ``slides/vendor/katex/``, the shared deck code from
``slides/assets/`` and its figures from ``figures/set/``. That works whenever the whole repository is
present, but a deck opened on its own -- for example straight out of a ZIP preview, which extracts only
the clicked file -- cannot find them and shows raw LaTeX and broken images.

This script inlines KaTeX's JavaScript, its CSS and its woff2 fonts, the shared deck CSS/JS, and every
figure (as base64 data URIs) into each deck. Every output file then works with no other files and no
network access.

Usage:
    python3 slides/build_standalone.py OUTPUT_DIR
"""
import base64
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
KATEX = HERE / "vendor" / "katex"
ASSETS = HERE / "assets"
DECKS = ["grpo_intro.html", "dapo_intro.html", "disco_intro.html", "sgrpo_intro.html"]

# The tags each source deck uses to load its dependencies (any attribute order, with or without "/>").
def tag(element, attr, value):
    return re.compile(rf'<{element}\b[^>]*\b{attr}="{re.escape(value)}"[^>]*>(?:</{element}>)?')


CSS_TAG = tag("link", "href", "vendor/katex/katex.min.css")
JS_TAG = tag("script", "src", "vendor/katex/katex.min.js")
AUTORENDER_TAG = tag("script", "src", "vendor/katex/contrib/auto-render.min.js")
COMMON_CSS_TAG = tag("link", "href", "assets/deck_common.css")
COMMON_JS_TAG = tag("script", "src", "assets/deck_common.js")
FIGURE_SRC = re.compile(r'src="(\.\./figures/[A-Za-z0-9_/.-]+\.png)"')

LICENSE_NOTE = ("KaTeX v{version} | MIT License | Copyright (c) 2013-2020 Khan Academy and other "
                "contributors | https://katex.org | full text: slides/vendor/katex/LICENSE")

# One @font-face "src:" list: the woff2 entry followed by woff/ttf fallbacks.
FONT_SRC = re.compile(
    r'src:url\(fonts/(KaTeX_[A-Za-z0-9_-]+)\.woff2\) format\("woff2"\)'
    r'(?:,url\([^)]*\) format\("[a-z]+"\))*'
)


def inline_fonts(css: str) -> str:
    """Replace every @font-face source list with a single embedded woff2 font."""
    def to_data_uri(match):
        font = KATEX / "fonts" / f"{match.group(1)}.woff2"
        data = base64.b64encode(font.read_bytes()).decode("ascii")
        return f'src:url(data:font/woff2;base64,{data}) format("woff2")'

    out, n = FONT_SRC.subn(to_data_uri, css)
    if n != 20 or "url(fonts/" in out:
        raise RuntimeError(f"expected to inline 20 KaTeX fonts, inlined {n}")
    return out


def inline_script(js: str) -> str:
    """Wrap library code in a <script> block (it must not contain a closing tag)."""
    if "</script" in js.lower():
        raise RuntimeError("library code contains '</script'; cannot inline safely")
    return "<script>\n" + js + "\n</script>"


def inline_figures(html: str, name: str) -> str:
    """Replace every src="../figures/....png" with a data URI of that PNG."""
    def to_data_uri(match):
        path = (HERE / match.group(1)).resolve()
        if not path.exists():
            raise RuntimeError(f"{name}: figure {match.group(1)} does not exist")
        return f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode("ascii")}"'

    return FIGURE_SRC.sub(to_data_uri, html)


def build(out_dir: pathlib.Path) -> None:
    js = (KATEX / "katex.min.js").read_text(encoding="utf-8")
    version = re.search(r'version:"([0-9.]+)"', js).group(1)
    note = LICENSE_NOTE.format(version=version)
    css = inline_fonts((KATEX / "katex.min.css").read_text(encoding="utf-8"))
    autorender = (KATEX / "contrib" / "auto-render.min.js").read_text(encoding="utf-8")

    replacements = {
        CSS_TAG: f"<style>/*! {note} */\n{css}\n</style>",
        JS_TAG: inline_script(f"/*! {note} */\n{js}"),
        AUTORENDER_TAG: inline_script(autorender),
        COMMON_CSS_TAG: "<style>\n" + (ASSETS / "deck_common.css").read_text(encoding="utf-8") + "</style>",
        COMMON_JS_TAG: inline_script((ASSETS / "deck_common.js").read_text(encoding="utf-8")),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in DECKS:
        html = (HERE / name).read_text(encoding="utf-8")
        for pattern, inline in replacements.items():
            found = pattern.findall(html)
            if len(found) != 1:
                raise RuntimeError(f"{name}: expected exactly one tag matching {pattern.pattern!r}")
            # str.replace, not re.sub: the inlined code contains backslashes.
            html = html.replace(found[0], inline)
        html = inline_figures(html, name)
        if re.search(r'(src|href)="(vendor|assets|\.\./figures)/', html):
            raise RuntimeError(f"{name}: a local file is still loaded by reference")
        (out_dir / name).write_text(html, encoding="utf-8")
        print(f"  {name}: {len(html) / 1e6:.2f} MB, self-contained")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(pathlib.Path(sys.argv[1]))
