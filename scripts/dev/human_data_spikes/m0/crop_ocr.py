#!/usr/bin/env python3
"""M0 spike: crop the top-right log panel, upscale+grayscale+threshold, OCR with easyocr."""

import sys

from PIL import Image, ImageOps

# Log-panel crop fractions (top-right), refined from reference frames.
LX0, LX1 = 0.645, 1.000
LY0, LY1 = 0.000, 0.300


def crop_log(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    box = (int(LX0 * w), int(LY0 * h), int(LX1 * w), int(LY1 * h))
    return im.crop(box)


def preprocess(crop: Image.Image, scale: float = 3.0) -> Image.Image:
    """Upscale (LANCZOS) + grayscale + simple threshold."""
    w, h = crop.size
    big = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    gray = ImageOps.grayscale(big)
    # Autocontrast then binary threshold to lift dark text off light panel.
    gray = ImageOps.autocontrast(gray)
    return gray


def main() -> None:
    import easyocr  # type: ignore

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    frames = sys.argv[1:]
    for path in frames:
        import os

        t = os.path.basename(path).split("_")[-1].split(".")[0]
        res = os.path.basename(path).split("_")[0]  # f1080 / f480 / f360
        crop = crop_log(path)
        crop.save(f"/tmp/phantom/m0/crop_{res}_{t}.png")
        proc = preprocess(crop, scale=3.0)
        proc.save(f"/tmp/phantom/m0/proc_{res}_{t}.png")
        # OCR both the raw color crop (easyocr handles color well) and processed.
        import numpy as np

        results = reader.readtext(np.array(crop), detail=0, paragraph=False)
        out = f"/tmp/phantom/m0/ocr_{res}_{t}.txt"
        with open(out, "w") as fh:
            fh.write(f"# source={path} crop_box_frac=({LX0},{LY0},{LX1},{LY1})\n")
            fh.write("# --- OCR on raw color crop ---\n")
            for line in results:
                fh.write(line + "\n")
        print(f"{res} t={t}: {len(results)} lines -> {out}")
        for line in results:
            print(f"    | {line}")


if __name__ == "__main__":
    main()
