#!/usr/bin/env python3
"""Render the paper's original PDF figures for the SGIP project page.

Usage: python scripts/prepare_figures.py /path/to/manuscript.zip
Requires: PyMuPDF and Pillow. Does not change the manuscript archive.
"""
import argparse
import io
import zipfile
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def render(archive, name, width=2400, clip=None):
    with pymupdf.open(stream=archive.read(f"figures/{name}.pdf"), filetype="pdf") as doc:
        page = doc[0]
        rect = page.rect
        if clip:
            rect = pymupdf.Rect(rect.width * clip[0], rect.height * clip[1],
                                rect.width * clip[2], rect.height * clip[3])
        pix = page.get_pixmap(matrix=pymupdf.Matrix(width / rect.width, width / rect.width),
                             clip=rect, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    out = ROOT / "docs/assets/images"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        for source, dest in [("final_teaser", "teaser"), ("final_main", "framework"),
                             ("final_qual", "qualitative"), ("final_mult", "alternative-futures")]:
            im = render(archive, source)
            im.save(out / f"{dest}.webp", quality=93, method=6)
            im.save(out / f"{dest}.png", optimize=True)
        for scene, left, right in [("kitchen", .006, .330), ("living-room", .341, .662),
                                    ("seminar-room", .676, .998)]:
            for method, top, bottom in [("sgip", .048, .472), ("sif3d", .480, .905)]:
                im = render(archive, "final_qual", 1200, (left, top, right, bottom))
                im.save(out / f"{scene}-{method}.webp", quality=94, method=6)
        # Original student branch, retained as a paper figure (no synthesized results).
        render(archive, "final_main", 2100, (0, .505, .665, 1)).save(
            out / "inference.webp", quality=94, method=6)
        render(archive, "final_teaser", 1200, (.666, 0, 1, 1)).save(
            out / "predicted-motion.webp", quality=92, method=6)
    print(f"Rendered original paper figures to {out}")


if __name__ == "__main__":
    main()
