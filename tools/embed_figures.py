"""Embed the figures that notebook markdown links to as cell attachments.

A markdown link such as ![alt](../figures/set/F05_clip_headroom.png) only renders when the notebook sits
next to figures/; on Colab, on GitHub, or in a single downloaded .ipynb it shows a broken image. This
script stores each linked figure inside its cell (downscaled to 1600 px wide) and rewrites the link to
attachment:F05_clip_headroom.png. Run it again after re-importing the figure set: attachments are matched
to figures/set/ and figures/ by file name and refreshed.

Usage:
    python tools/embed_figures.py                 # every notebook in notebook/
    python tools/embed_figures.py A.ipynb B.ipynb
"""
import base64
import io
import pathlib
import re
import sys

import nbformat
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
LINK = re.compile(r"!\[([^\]]*)\]\(\.\./figures/((?:set/)?[\w.-]+\.png)\)")
ATTACHED = re.compile(r"!\[[^\]]*\]\(attachment:([\w.-]+\.png)\)")
WIDTH = 1600


def find(name):
    for folder in (FIGURES / "set", FIGURES):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"figure {name} not found in figures/set/ or figures/")


def encode(path):
    image = Image.open(path).convert("RGB")
    if image.width > WIDTH:
        image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def embed(path):
    nb = nbformat.read(path, as_version=4)
    count = 0
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for alt, rel in LINK.findall(cell.source):
            name = pathlib.PurePath(rel).name
            cell.source = cell.source.replace(f"![{alt}](../figures/{rel})", f"![{alt}](attachment:{name})")
        names = ATTACHED.findall(cell.source)
        if names:
            cell["attachments"] = {name: {"image/png": encode(find(name))} for name in names}
            count += len(names)
        elif "attachments" in cell:
            del cell["attachments"]
    nbformat.validate(nb)
    nbformat.write(nb, path)
    return count


if __name__ == "__main__":
    paths = [pathlib.Path(p) for p in sys.argv[1:]] or sorted((ROOT / "notebook").glob("*.ipynb"))
    for path in paths:
        print(f"{path.name}: {embed(path)} figures embedded")
