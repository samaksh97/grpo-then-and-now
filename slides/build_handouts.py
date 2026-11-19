#!/usr/bin/env python3
"""Print each self-contained deck to a PDF handout: one 16:9 page per slide, every fragment shown,
interactive slides in their default state, no navigation controls (see the print rules in
slides/assets/deck_common.css).

Needs Chrome, Chromium or Edge. Set CHROME=/path/to/browser if it is not found automatically; on WSL the
Windows Chrome or Edge is used.

Usage:
    python3 slides/build_handouts.py [OUTPUT_DIR]      (default: slides/handouts)
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_standalone import DECKS, build  # noqa: E402

BROWSERS = [os.environ.get("CHROME"), "google-chrome", "chromium", "chromium-browser",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
            "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]


def find_browser():
    for candidate in BROWSERS:
        if candidate and (shutil.which(candidate) or os.path.exists(candidate)):
            return shutil.which(candidate) or candidate
    sys.exit("Chrome/Chromium not found; set CHROME=/path/to/chrome")


def windows_staging():
    """A Windows browser cannot read or write \\\\wsl.localhost paths: stage under the Windows temp folder."""
    local = subprocess.check_output(["cmd.exe", "/c", "echo %LOCALAPPDATA%"], text=True,
                                    stderr=subprocess.DEVNULL).strip()
    root = pathlib.Path(subprocess.check_output(["wslpath", "-u", local], text=True).strip()) / "Temp"
    return tempfile.TemporaryDirectory(dir=root), (
        lambda p: subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip())


def main():
    out = (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "handouts").resolve()
    out.mkdir(parents=True, exist_ok=True)
    browser = find_browser()
    staging, host = windows_staging() if browser.endswith(".exe") else (tempfile.TemporaryDirectory(), str)
    with staging as tmp:
        tmp = pathlib.Path(tmp)
        build(tmp / "decks")
        for name in DECKS:
            staged = tmp / name.replace("_intro.html", "_handout.pdf")
            url = "file:///" + host(tmp / "decks" / name).replace("\\", "/").lstrip("/")
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-pdf-header-footer",
                            f"--user-data-dir={host(tmp / 'profile')}", "--virtual-time-budget=5000",
                            f"--print-to-pdf={host(staged)}", url], check=True, capture_output=True)
            shutil.copyfile(staged, out / staged.name)
            print(f"  {staged.name}: {staged.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
