"""Fit affine engine->frame for a given frame, reusing board_cv/spike.py logic."""

import json
import sys

import cv2

sys.path.insert(0, "/tmp/phantom/board_cv")
from spike import ENG_HEX, ENG_VTX, apply_affine, detect_tokens, fit_affine


def run(path, tag):
    img = cv2.imread(path)
    toks = detect_tokens(img)
    A, resid0, fin, pairs = fit_affine(toks)
    hex_px = apply_affine(A, ENG_HEX)
    vtx_px = apply_affine(A, ENG_VTX)
    out = {
        "A": A.tolist(),
        "hex_px": hex_px.tolist(),
        "vtx_px": vtx_px.tolist(),
        "n_tokens": len(toks),
        "resid_mean": float(fin.mean()),
        "resid_max": float(fin.max()),
    }
    json.dump(out, open(f"fit_{tag}.json", "w"))
    print(f"[{tag}] tokens={len(toks)} affine resid mean={fin.mean():.2f}px max={fin.max():.2f}px")
    return out


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
