"""Deck conventions: stable slide slugs, no numbered cross-references, resolvable links and figures,
and a standalone build that loads nothing by reference."""
import pathlib
import re
import subprocess
import sys

import pytest
from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parents[1]
DECKS = ["grpo_intro.html", "dapo_intro.html", "disco_intro.html", "sgrpo_intro.html"]


def soup(name):
    return BeautifulSoup((ROOT / "slides" / name).read_text(encoding="utf-8"), "html.parser")


SLUGS = {name: [s.get("data-slug") for s in soup(name).select("section.slide")] for name in DECKS}


@pytest.mark.parametrize("name", DECKS)
def test_every_slide_has_a_unique_slug(name):
    slugs = SLUGS[name]
    assert all(slugs), f"slides without data-slug: {[i + 1 for i, s in enumerate(slugs) if not s]}"
    assert len(set(slugs)) == len(slugs)


@pytest.mark.parametrize("name", DECKS)
def test_no_numbered_kickers_or_slide_number_references(name):
    deck = soup(name)
    numbered = [k.get_text(strip=True) for k in deck.select(".kicker") if re.match(r"\d{2}\s*·", k.get_text(strip=True))]
    assert not numbered, numbered
    text = " ".join(s.get_text(" ") for s in deck.select("section.slide"))
    assert not re.search(r"\bslides? \d", text, flags=re.I), re.findall(r".{30}\bslides? \d.{20}", text, flags=re.I)


@pytest.mark.parametrize("name", DECKS)
def test_links_and_figures_resolve(name):
    deck = soup(name)
    for a in deck.select("a[href]"):
        href = a["href"]
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        target, _, slug = href.partition("#")
        if target:
            assert (ROOT / "slides" / target).exists(), href
        if slug and target in DECKS + [""]:
            assert slug in SLUGS[target or name], href
    for img in deck.select("img[src]"):
        assert (ROOT / "slides" / img["src"]).resolve().exists(), img["src"]
        assert img.get("alt"), img["src"]


def test_deck_links_from_readme_and_notebooks_resolve():
    sources = [ROOT / "README.md", *sorted((ROOT / "notebook").glob("*.ipynb"))]
    bad = []
    for path in sources:
        for deck, slug in re.findall(r"(\w+_intro\.html)#([a-z0-9-]+)", path.read_text(encoding="utf-8")):
            if deck not in SLUGS or slug not in SLUGS[deck]:
                bad.append((path.name, f"{deck}#{slug}"))
    assert not bad, bad


def test_standalone_build_is_self_contained(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "slides" / "build_standalone.py"), str(tmp_path)], check=True,
                   capture_output=True)
    for name in DECKS:
        built = BeautifulSoup((tmp_path / name).read_text(encoding="utf-8"), "html.parser")
        # Links to the notebooks (<a href="../notebook/...">) are fine; nothing may be *loaded* by reference.
        loaded = [t.get("src") for t in built.find_all(src=True)] + [t.get("href") for t in built.find_all("link")]
        local = [u for u in loaded if u and not u.startswith(("data:", "http://", "https://"))]
        assert not local, (name, local)
