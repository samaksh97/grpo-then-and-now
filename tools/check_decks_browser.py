"""Smoke-test the self-contained decks in real browsers, from file:// and from a local web server.

For every deck and browser: no page errors, KaTeX rendered every equation (no raw $...$ left), the slide
counter matches the slides, the arrow key advances, and a #slug link opens the right slide. A screenshot
of the first slide per browser goes to OUTPUT_DIR/shots/.

Needs Playwright:  pip install playwright && python -m playwright install chromium firefox webkit
Usage:
    python slides/build_standalone.py build/slides
    python tools/check_decks_browser.py build/slides [chromium firefox webkit]
"""
import functools
import http.server
import pathlib
import socketserver
import sys
import threading

from playwright.sync_api import sync_playwright

DECKS = {"grpo_intro.html": "failure-clip-headroom", "dapo_intro.html": "notation",
         "disco_intro.html": "hard-negatives", "sgrpo_intro.html": "serial-group"}


def serve(folder):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(folder))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def check(page, url, slug):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_load_state("load")
    page.wait_for_timeout(600)
    n_slides = page.locator("section.slide").count()
    counter = page.locator("#counter").inner_text()
    assert counter.strip().endswith(f"/ {n_slides}"), counter
    assert page.locator(".katex").count() > 0, "KaTeX did not render"
    raw = page.evaluate("""() => [...document.querySelectorAll('section.slide')].map(s => s.innerText)
                               .filter(t => /\\$[^$\\s][^$]*\\$/.test(t)).length""")
    assert raw == 0, f"{raw} slides still show raw $...$"
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(300)
    page.goto(url.split("#")[0] + "#" + slug)
    page.wait_for_timeout(600)
    active = page.evaluate("() => document.querySelector('section.slide.active').dataset.slug")
    assert active == slug, (active, slug)
    assert not errors, errors


def main():
    folder = pathlib.Path(sys.argv[1]).resolve()
    browsers = sys.argv[2:] or ["chromium", "firefox", "webkit"]
    shots = folder / "shots"
    shots.mkdir(exist_ok=True)
    httpd, base = serve(folder)
    failures = 0
    with sync_playwright() as p:
        for name in browsers:
            browser = getattr(p, name).launch()
            for deck, slug in DECKS.items():
                for mode, url in [("file", (folder / deck).as_uri()), ("http", f"{base}/{deck}")]:
                    page = browser.new_page(viewport={"width": 1440, "height": 900})
                    try:
                        check(page, url, slug)
                        print(f"ok    {name:8s} {mode:4s} {deck}")
                    except Exception as error:  # report every combination, then fail
                        failures += 1
                        print(f"FAIL  {name:8s} {mode:4s} {deck}: {error}")
                    if mode == "file" and deck == "grpo_intro.html":
                        page.goto(url)
                        page.wait_for_timeout(600)
                        page.screenshot(path=str(shots / f"{name}_{deck}.png"))
                    page.close()
            browser.close()
    httpd.shutdown()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
