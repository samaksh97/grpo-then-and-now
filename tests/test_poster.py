"""The poster page: self-contained, every placeholder filled, the method table taken from the testbed presets,
and the TRL row consistent with the testbed's library table. (Printing needs a browser and is not tested.)"""
import importlib.util
import pathlib
import re

from bs4 import BeautifulSoup

from grpo_lab.testbed import LIBRARY_SETTINGS, PRESETS

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_poster", ROOT / "poster" / "build_poster.py")
build_poster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_poster)

LIBRARY_KEYS = {"eps": "eps_low, eps_high", "overlong": "overlong = shape"}


def test_trl_names_match_the_testbed_library_table():
    for key, name in build_poster.TRL_NAMES.items():
        trl = LIBRARY_SETTINGS[LIBRARY_KEYS.get(key, key)][0]
        if name is None:
            assert trl == "no built-in flag", key
        else:
            for part in name.split(", "):
                assert part in trl, (key, part, trl)


def test_page_is_self_contained_and_complete():
    page = build_poster.render_html()
    assert not re.search(r"\{\{[A-Za-z_:]+\}\}", page)
    soup = BeautifulSoup(page, "html.parser")
    loaded = [t.get("src") for t in soup.find_all(src=True)] + [t.get("href") for t in soup.find_all("link")]
    assert not [u for u in loaded if u and not u.startswith(("data:", "http://", "https://"))]
    assert len(soup.select(".chart svg")) == len(build_poster.CHARTS)
    ids = [t["id"] for t in soup.find_all(id=True)]
    assert len(ids) == len(set(ids)), "element ids collide between the inline charts"
    assert all(url in page for url in build_poster.LINKS.values())
    rows = [th.get_text() for th in soup.select("table.testbed tbody th")]
    assert rows == [*PRESETS, "TRL option"]
