#!/usr/bin/env python3
"""Test whether preprocessing (upscale+gray+autocontrast) helps low-res, and OCR the proc images."""

import os
import sys

import easyocr  # type: ignore
import numpy as np
from crop_ocr import crop_log, preprocess

reader = easyocr.Reader(["en"], gpu=False, verbose=False)
for path in sys.argv[1:]:
    t = os.path.basename(path).split("_")[-1].split(".")[0]
    res = os.path.basename(path).split("_")[0]
    crop = crop_log(path)
    proc = preprocess(crop, scale=4.0)
    proc.save(f"/tmp/phantom/m0/proc4_{res}_{t}.png")
    results = reader.readtext(np.array(proc), detail=0, paragraph=False)
    print(f"PROC(4x,gray) {res} t={t}:")
    for line in results:
        print(f"    | {line}")
