"""Import the exported GRPO figure set (F1-F14) into figures/set/ with stable ASCII file names.

The figure cards are designed outside this repository and exported as one ZIP of PNGs named like
"F5 · Clip headroom.png". This script matches each PNG to its entry in figures/set/figures.json by the
figure number (F5, F14a, ...), so a card can be retitled without breaking links, writes it under the file
name recorded there (e.g. figures/set/F05_clip_headroom.png), drops the alpha channel when it is fully
opaque, and fails if a listed figure is missing. Re-run it whenever a new export arrives.

Usage:  python figures/import_figure_set.py "figures/GRPO Figure Set-png.zip"
"""
import io
import json
import pathlib
import re
import sys
import zipfile

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "set"
NAME = re.compile(r"F(\d+)([ab]?)\s*·\s*(.+)\.png$")


def main(zip_path: str) -> None:
    manifest = json.loads((OUT / "figures.json").read_text(encoding="utf-8"))
    files = {fig["id"]: fig["file"] for fig in manifest["figures"]}
    written = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            match = NAME.search(pathlib.PurePath(info.filename).name)
            if not match:
                continue
            fig_id = f"F{int(match.group(1))}{match.group(2)}"
            if fig_id not in files:
                raise SystemExit(f"{info.filename}: {fig_id} is not in figures.json; add an entry for it first")
            target = OUT / files[fig_id]
            image = Image.open(io.BytesIO(archive.read(info)))
            if image.mode == "RGBA" and image.getextrema()[3][0] == 255:
                image = image.convert("RGB")
            image.save(target, optimize=True)
            written.append(fig_id)
            print(f"  {info.filename} -> {target.relative_to(HERE.parent)} {image.size}")

    missing = sorted(set(files) - set(written))
    if missing:
        raise SystemExit(f"figures listed in figures.json but missing from the export: {missing}")
    print(f"Imported {len(written)} figures into {OUT.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
