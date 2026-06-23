"""BLOCKER 2 — deterministic orientation lock v2 (screen-anchored + content-verified).

Two-part lock:

PART A (geometric, deterministic): pick the unique D6 orientation by a SCREEN-SPACE
rule that has nothing to do with content, so it is identical every frame:
    engine hex 8  must land at the TOP-CENTER hex on screen (min y, near center x)
    engine hex 11 must land to the RIGHT of hex 8 (max x)
These two screen landmarks fix rotation AND reflection uniquely (they are an
ordered, non-collinear pair). This alone guarantees frame-stability because the
board render is screen-static.

PART B (content cross-check, redundant safety): independently OCR >=2 ground-truth
anchors (a NUMBER token AND its hex RESOURCE color) at fixed screen pixels, and
require the engine-id the affine predicts there to carry exactly that content.
If the geometric lock and the content anchors disagree, REJECT (don't emit a
silently-flipped board). This is the consistency check that catches a flip in
production and rejects a deliberately mis-oriented fit.
"""

import math

import cv2
import numpy as np
import orient_lock as base  # reuse detect_tokens, classify, ocr, geometry

WORK = "/tmp/phantom/blockers"
TPL = base.TPL
ENG_HEX = base.ENG_HEX
ENG_VTX = base.ENG_VTX
ENG_C = base.ENG_C
HCV = base.HCV
EDGE_V = base.EDGE_V


def all_orientation_affines(toks):
    """Return list of (refl, rotdeg, A) for ALL 12 D6 orientations that fit the
    detected tokens (each has identical residual; we keep them all)."""
    tok = np.array([[x, y] for x, y, _ in toks], float)
    fr_c = tok.mean(0)
    s0 = np.linalg.norm(tok - fr_c, axis=1).mean() / np.linalg.norm(ENG_HEX - ENG_C, axis=1).mean()
    out = []
    for refl in (1, -1):
        for k in range(6):
            ang = math.radians(60 * k)
            R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            Rf = R @ np.array([[refl, 0], [0, 1]], float)
            eng_t = (ENG_HEX - ENG_C) @ Rf.T * s0 + fr_c
            pairs = []
            used = set()
            for ti, p in enumerate(tok):
                d = np.linalg.norm(eng_t - p, axis=1)
                for hi in np.argsort(d):
                    if hi not in used:
                        used.add(hi)
                        pairs.append((ti, hi, d[hi]))
                        break
            src = np.array([ENG_HEX[hi] for _, hi, _ in pairs])
            dst = np.array([tok[ti] for ti, _, _ in pairs])
            A, _ = cv2.estimateAffine2D(
                src.astype(np.float32),
                dst.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=12,
            )
            proj = (A[:, :2] @ src.T).T + A[:, 2]
            resid = float(np.linalg.norm(proj - dst, axis=1).mean())
            out.append((refl, 60 * k, A, resid))
    return out


def apply(A, pts):
    return (A[:, :2] @ pts.T).T + A[:, 2]


def pick_screen_orientation(toks):
    """PART A: pick the unique orientation by screen-space rule.
    Rule: engine hex 8 -> the hex with the SMALLEST screen y whose x is closest
    to the board's horizontal center (top-center); engine hex 11 -> the hex with
    the LARGEST screen x (rightmost). The orientation whose projected H8 and H11
    best satisfy BOTH is chosen. Deterministic, content-free."""
    cands = all_orientation_affines(toks)
    tok = np.array([[x, y] for x, y, _ in toks], float)
    cx_board = tok[:, 0].mean()
    scored = []
    for refl, deg, A, resid in cands:
        hx = apply(A, ENG_HEX)
        # screen landmark scores (lower = better fit to the rule)
        h8 = hx[8]
        h11 = hx[11]
        # H8 should be top-center: small y, x near board center
        top_center_pen = (h8[1] - hx[:, 1].min()) + 0.5 * abs(h8[0] - cx_board)
        # H11 should be rightmost
        right_pen = hx[:, 0].max() - h11[0]
        scored.append((top_center_pen + right_pen, refl, deg, A, resid))
    scored.sort(key=lambda z: z[0])
    return scored  # best first


def read_screen_at(img, A, eng_id):
    """Read (resource, number) for a given engine id under affine A."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hx = apply(A, ENG_HEX)
    vx = apply(A, ENG_VTX)
    corners = [vx[HCV[eng_id][c]] for c in range(6)]
    h, s, v = base.sample_fill(hsv, hx[eng_id, 0], hx[eng_id, 1], corners)
    res = base.classify_one(h, s, v)
    num = base.ocr_number(img, hx[eng_id, 0], hx[eng_id, 1]) if res != "DESERT" else None
    return res, num, tuple(hx[eng_id])


def read_full_board(img, A):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hx = apply(A, ENG_HEX)
    vx = apply(A, ENG_VTX)
    hexes = []
    for e in range(19):
        corners = [vx[HCV[e][c]] for c in range(6)]
        h, s, v = base.sample_fill(hsv, hx[e, 0], hx[e, 1], corners)
        res = base.classify_one(h, s, v)
        num = base.ocr_number(img, hx[e, 0], hx[e, 1]) if res != "DESERT" else None
        hexes.append({"hex_id": e, "resource": res, "number": num})
    return hexes, hx, vx


def lock_and_read(img, content_anchors, verbose=False):
    """content_anchors: list of {name, engine_id, resource, number} — known
    ground truth for specific ENGINE ids (established once, then reused as an
    invariant). Returns (hexes, A, diag)."""
    toks = base.detect_tokens(img)
    scored = pick_screen_orientation(toks)
    pen, refl, deg, A, resid = scored[0]
    diag = {
        "n_tokens": len(toks),
        "screen_rule_penalty": float(pen),
        "second_penalty": float(scored[1][0]),
        "refl": refl,
        "rot": deg,
        "residual_mean": resid,
    }

    # PART B: content cross-check on the chosen orientation.
    checks = []
    all_ok = True
    for a in content_anchors:
        res, num, px = read_screen_at(img, A, a["engine_id"])
        ok_r = (a["resource"] is None) or (res == a["resource"])
        ok_n = (a["number"] is None) or (num == a["number"])
        checks.append(
            {
                "name": a["name"],
                "engine_id": a["engine_id"],
                "want": (a["resource"], a["number"]),
                "got": (res, num),
                "ok": ok_r and ok_n,
            }
        )
        all_ok = all_ok and ok_r and ok_n
    diag["content_checks"] = checks
    diag["content_consistent"] = all_ok

    if not all_ok:
        diag["REJECTED"] = "content anchors disagree with screen-locked orientation"
        return None, A, diag

    hexes, hx, vx = read_full_board(img, A)
    return hexes, A, diag
