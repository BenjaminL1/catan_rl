#!/usr/bin/env python3
"""Glyph-classifier validation harness (the user-approved pass bar, 2026-07-04).

Builds the labelled check-set that decides whether the glyph classifier — the
reader side of the orientation-independent joint-flip firewall — is trustworthy
enough to unblock the full harvest. Three subcommands, run in order:

  extract  Sample 'high' videos from data/human/strength_manifest.json, locate
           "received starting resources" log lines, detect the granted card-icon
           boxes, and save labelable crops + per-game palettes + metadata under
           data/human/glyph_valset/.
  sheet    Render contact sheets (grids of upscaled glyph crops with ids) for
           hand-labelling.
  score    Combine hand labels (labels.json: {"<crop_id>": "ORE" | ... }) with
           the extracted metadata into LabeledGrantFrames and run
           validate_glyph_classifier — the module enforces the approved bar
           (>=0.98 exact-multiset accuracy over >=8 frames, ZERO ORE<->BRICK
           confusions, fail-closed coverage reported). Writes
           data/human/glyph_validation.json + a markdown report. The artifact is
           HARDENED (expert SHOULD-FIXes 2026-07-05): it carries the
           validation_fingerprint (live palette/threshold constants + detector
           source — load_glyph_validation refuses a stale PASS), the git rev,
           per-box margin diagnostics (saturation distance to the [75, 95)
           dead band + hue margins), and — with --folds 2 (the default) — a
           video-disjoint 2-fold cross-check (hue medians + ore ceiling fitted
           on one fold, scored on the other).

Local + deterministic (yt-dlp / ffmpeg / easyocr / opencv — no API calls).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from catan_rl.human_data.board_cv import board_hsv_samples, read_board
from catan_rl.human_data.glyph_anchor import (
    CARD_PALETTE,
    GRANT_RE,
    HUE_MIN_SATURATION_ABOVE_ORE,
    HUE_RESOURCES_BY_RANK,
    GlyphPalette,
    LabeledGrantFrame,
    _glyph_median_hsv,
    calibrate_glyph_palette,
    detect_glyph_boxes,
    validate_glyph_classifier,
    validation_fingerprint,
)
from catan_rl.human_data.logparse import LOG_CROP_FRAC, _normalise

MANIFEST = REPO / "data" / "human" / "strength_manifest.json"
OUT_DIR = REPO / "data" / "human" / "glyph_valset"
META = OUT_DIR / "meta.jsonl"
LABELS = OUT_DIR / "labels.json"
VALIDATION_OUT = REPO / "data" / "human" / "glyph_validation.json"
REPORT_OUT = REPO / "data" / "human" / "glyph_validation.md"

# The detector (detect_glyph_boxes) + grant-line matcher (GRANT_RE) were promoted
# into catan_rl.human_data.glyph_anchor (expert review 2026-07-05: production must
# ship the exact detector the 24/24 PASS validated); this harness imports them back.
LABEL_UPSCALE = 8  # per-crop upscale factor for the labelling sheet


def _run(cmd: list[str], timeout: int = 120) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout).stdout


def stream_and_duration(video_id: str) -> tuple[str, float] | None:
    """One yt-dlp call -> (direct media url, duration_seconds)."""
    try:
        out = _run(
            [
                "yt-dlp",
                "-f",
                "bestvideo[height<=1080]/best",
                "--print",
                "%(duration)s|%(url)s",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    line = out.splitlines()[0] if out else ""
    if "|" not in line:
        return None
    dur_s, url = line.split("|", 1)
    try:
        dur = float(dur_s)
    except ValueError:
        return None
    return (url, dur) if url.startswith("http") and dur > 0 else None


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def grab_frame_rgb(url: str, t: float) -> npt.NDArray[np.uint8] | None:
    """Grab one frame at ``t`` seconds as an RGB uint8 array, or None."""
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "f.png"
        try:
            subprocess.run(
                [
                    _ffmpeg(),
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(max(0.0, t)),
                    "-i",
                    url,
                    "-frames:v",
                    "1",
                    "-y",
                    str(png),
                ],
                capture_output=True,
                check=True,
                timeout=90,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        bgr = cv2.imread(str(png))
        if bgr is None:
            return None
        return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


_READER: Any = None


def _reader() -> Any:
    global _READER
    if _READER is None:
        import easyocr

        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def crop_log_rgb(frame_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    h, w = frame_rgb.shape[:2]
    x0f, y0f, x1f, y1f = LOG_CROP_FRAC
    return frame_rgb[int(y0f * h) : int(y1f * h), int(x0f * w) : int(x1f * w)]


def ocr_lines_with_boxes(
    crop_rgb: npt.NDArray[np.uint8],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """OCR the log crop -> [(text, (x0,y0,x1,y1))] per detected text token/line."""
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for bbox, text, _conf in _reader().readtext(crop_rgb):
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        out.append((str(text), (min(xs), min(ys), max(xs), max(ys))))
    return out


def find_grant_events(url: str, dur: float) -> list[tuple[float, str]]:
    """Locate up to 2 grant events: coarse scan for setup activity, then fine scan.

    Returns [(t, normalised_grant_line)]. Distinct events are >=10s apart.
    """
    hits: list[tuple[float, str]] = []
    t_active: float | None = None
    coarse_end = min(dur, 900.0)
    t = 40.0
    while t < coarse_end:
        frame = grab_frame_rgb(url, t)
        if frame is not None:
            toks = ocr_lines_with_boxes(crop_log_rgb(frame))
            joined = " | ".join(_normalise(txt).lower() for txt, _ in toks)
            if GRANT_RE.search(joined):
                t_active = t
                break
            if "placed a" in joined or "happy settling" in joined or "rolled" in joined:
                t_active = t
                break
        t += 15.0
    if t_active is None:
        return []
    fine_lo = max(20.0, t_active - 100.0)
    fine_hi = min(dur, t_active + 160.0)
    t = fine_lo
    while t < fine_hi and len(hits) < 2:
        frame = grab_frame_rgb(url, t)
        if frame is not None:
            for txt, _ in ocr_lines_with_boxes(crop_log_rgb(frame)):
                low = _normalise(txt).lower()
                if GRANT_RE.search(low):
                    if not hits or t - hits[-1][0] >= 10.0:
                        hits.append((t, low))
                    break
        t += 2.5
    return hits


def cmd_extract(n_videos: int, seed: int) -> int:
    import random

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    high = [v["video_id"] for v in manifest["videos"] if v["strength"] == "high"]
    rng = random.Random(seed)
    picks = rng.sample(high, min(n_videos, len(high)))

    done_videos = set()
    if META.exists():
        for line in META.read_text().splitlines():
            done_videos.add(json.loads(line)["video_id"])

    for idx, vid in enumerate(picks, 1):
        if vid in done_videos:
            print(f"[{idx}/{len(picks)}] {vid} already extracted — skip", flush=True)
            continue
        sd = stream_and_duration(vid)
        if sd is None:
            print(f"[{idx}/{len(picks)}] {vid} NO STREAM — skip", flush=True)
            continue
        url, dur = sd
        events = find_grant_events(url, dur)
        if not events:
            print(f"[{idx}/{len(picks)}] {vid} no grant line found — skip", flush=True)
            continue
        for ev_i, (t, line_text) in enumerate(events):
            frame = grab_frame_rgb(url, t)
            if frame is None:
                continue
            # palette needs an accepted board read; the board can be mid-render at
            # grant time, so fall back to later frames of the SAME game (the tile
            # palette is constant within a game).
            board = read_board(frame)
            board_frame = frame
            for dt in (90.0, 180.0):
                if board is not None:
                    break
                fb = grab_frame_rgb(url, min(t + dt, dur - 2.0))
                if fb is not None:
                    board = read_board(fb)
                    board_frame = fb
            if board is None:
                print(f"  {vid}@{t:.0f}s board read rejected — palette unavailable, skip event")
                continue
            palette = calibrate_glyph_palette(
                board_hsv_samples(board_frame, board), board.desert_hex
            )
            crop = crop_log_rgb(frame)
            toks = ocr_lines_with_boxes(crop)
            grant_boxes = [b for txt, b in toks if GRANT_RE.search(_normalise(txt).lower())]
            if not grant_boxes:
                continue
            text_boxes = [b for _, b in toks]
            boxes = detect_glyph_boxes(crop, grant_boxes[0], text_boxes)
            if not boxes:
                print(f"  {vid}@{t:.0f}s grant line found but 0 icon boxes — skip event")
                continue
            crop_id = f"{vid}_t{int(t)}"
            cv2.imwrite(str(OUT_DIR / f"{crop_id}_log.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            for b_i, (bx0, by0, bx1, by1) in enumerate(boxes):
                sw = crop[by0:by1, bx0:bx1]
                big = cv2.resize(
                    sw,
                    (sw.shape[1] * LABEL_UPSCALE, sw.shape[0] * LABEL_UPSCALE),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imwrite(
                    str(OUT_DIR / f"{crop_id}_b{b_i}.png"),
                    cv2.cvtColor(big, cv2.COLOR_RGB2BGR),
                )
            row = {
                "crop_id": crop_id,
                "video_id": vid,
                "t": round(t, 1),
                "event_index": ev_i,
                "line_text": line_text,
                "boxes": [list(b) for b in boxes],
                "palette": {
                    "hue_centres": palette.hue_centres,
                    "ore_max_saturation": palette.ore_max_saturation,
                },
                "board": {
                    "desert_hex": board.desert_hex,
                    "residual_px": round(board.residual_px, 3),
                },
            }
            with META.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"[{idx}/{len(picks)}] {vid}@{t:.0f}s: {len(boxes)} icon boxes saved",
                flush=True,
            )
    print(f"\nextract done. metadata: {META}")
    return 0


def cmd_sheet(per_row: int = 8) -> int:
    """Contact sheets: grids of the upscaled per-box crops, ids printed above."""
    rows = [json.loads(line) for line in META.read_text().splitlines()]
    tiles: list[tuple[str, npt.NDArray[np.uint8]]] = []
    for row in rows:
        for b_i in range(len(row["boxes"])):
            p = OUT_DIR / f"{row['crop_id']}_b{b_i}.png"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    tiles.append((f"{row['crop_id']}_b{b_i}", np.asarray(img, dtype=np.uint8)))
    if not tiles:
        print("no crops found — run extract first")
        return 1
    cell_w = max(t.shape[1] for _, t in tiles) + 20
    cell_h = max(t.shape[0] for _, t in tiles) + 44
    n_rows = (len(tiles) + per_row - 1) // per_row
    per_sheet_rows = 6
    n_sheets = (n_rows + per_sheet_rows - 1) // per_sheet_rows
    for s in range(n_sheets):
        sheet = np.full((per_sheet_rows * cell_h, per_row * cell_w, 3), 255, np.uint8)
        for j in range(per_sheet_rows * per_row):
            k = s * per_sheet_rows * per_row + j
            if k >= len(tiles):
                break
            name, img = tiles[k]
            r, c = divmod(j, per_row)
            y, x = r * cell_h, c * cell_w
            cv2.putText(
                sheet,
                name[-26:],
                (x + 4, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 0, 0),
                1,
            )
            sheet[y + 24 : y + 24 + img.shape[0], x + 4 : x + 4 + img.shape[1]] = img
        out = OUT_DIR / f"sheet_{s}.png"
        cv2.imwrite(str(out), sheet)
        print(f"wrote {out}")
    print(f"{len(tiles)} crops across {n_sheets} sheet(s). Label into {LABELS}")
    return 0


def _dedupe_events(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop near-duplicate grant events (the SAME log line sampled twice ~10s apart
    — it persists on screen ~30s). Two events of one video within 20s with the same
    box count are the same physical line; keeping both double-counts a correlated
    frame in the score. Keeps the first of each pair."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        dup = any(
            k["video_id"] == row["video_id"]
            and abs(float(k["t"]) - float(row["t"])) <= 20.0
            and len(k["boxes"]) == len(row["boxes"])
            for k in kept
        )
        if dup:
            dropped += 1
        else:
            kept.append(row)
    return kept, dropped


def _git_rev() -> str:
    """The repo HEAD at score time (provenance for the artifact)."""
    try:
        return _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _summary(values: list[float]) -> dict[str, Any]:
    """min/median summary stats (the slice's per-box diagnostics contract)."""
    if not values:
        return {"n": 0, "min": None, "median": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "min": round(float(arr.min()), 2),
        "median": round(float(np.median(arr)), 2),
    }


def _margin_summary(frames: list[LabeledGrantFrame]) -> dict[str, Any]:
    """Per-box margin diagnostics over every scored box (expert SHOULD-FIX d).

    For each labelled box: the body-median saturation's distance to the edges of
    the fail-closed dead band [ore_ceiling, ore_ceiling + margin) = [75, 95) —
    reported per side (ORE boxes sit BELOW 75; hue boxes ABOVE 95) — plus the
    hue margin (runner-up minus nearest card-centre distance) for hue-side
    boxes. Small minima here mean the PASS is living near a threshold edge.
    """
    band_lo = CARD_PALETTE.ore_max_saturation
    band_hi = band_lo + HUE_MIN_SATURATION_ABOVE_ORE
    sat_gap_ore: list[float] = []
    sat_gap_hue: list[float] = []
    hue_margins: list[float] = []
    n_in_dead_band = 0
    n_no_body = 0
    for f in frames:
        assert f.expected_by_box is not None
        for x0, y0, x1, y1 in f.glyph_boxes:
            st = _glyph_median_hsv(np.asarray(f.log_crop_rgb[y0:y1, x0:x1], np.uint8))
            if not st.ok:
                n_no_body += 1
                continue
            if st.sat < band_lo:
                sat_gap_ore.append(band_lo - st.sat)
            elif st.sat >= band_hi:
                sat_gap_hue.append(st.sat - band_hi)
                dists = sorted(
                    min(abs(st.hue - c) % 180.0, 180.0 - abs(st.hue - c) % 180.0)
                    for c in CARD_PALETTE.hue_centres.values()
                )
                hue_margins.append(dists[1] - dists[0])
            else:
                n_in_dead_band += 1
    return {
        "dead_band_sat": [band_lo, band_hi],
        "sat_gap_ore_side": _summary(sat_gap_ore),
        "sat_gap_hue_side": _summary(sat_gap_hue),
        "hue_margin": _summary(hue_margins),
        "n_in_dead_band": n_in_dead_band,
        "n_no_body": n_no_body,
    }


def _fit_fold_palette(frames: list[LabeledGrantFrame]) -> GlyphPalette | None:
    """Fit hue medians + an ORE saturation ceiling from labelled boxes only.

    Mirrors how the canonical constants were measured: per-class body-median
    hues -> class medians; the ORE ceiling is the midpoint of the gap between
    the most-saturated labelled ORE body and the least-saturated coloured body.
    Returns None when a class is unrepresented (the fold cannot fit a palette).
    """
    ore_sats: list[float] = []
    coloured_sats: list[float] = []
    hues: dict[str, list[float]] = {r: [] for r in HUE_RESOURCES_BY_RANK}
    for f in frames:
        assert f.expected_by_box is not None
        for (x0, y0, x1, y1), lab in zip(f.glyph_boxes, f.expected_by_box, strict=True):
            st = _glyph_median_hsv(np.asarray(f.log_crop_rgb[y0:y1, x0:x1], np.uint8))
            if not st.ok:
                continue
            if lab == "ORE":
                ore_sats.append(st.sat)
            else:
                hues[lab].append(st.hue)
                coloured_sats.append(st.sat)
    if not ore_sats or not coloured_sats or any(not v for v in hues.values()):
        return None
    return GlyphPalette(
        hue_centres={r: float(np.median(np.asarray(v))) for r, v in hues.items()},
        ore_max_saturation=(max(ore_sats) + min(coloured_sats)) / 2.0,
    )


def _fold_results(
    entries: list[tuple[str, LabeledGrantFrame]], n_folds: int
) -> list[dict[str, Any]]:
    """Video-disjoint k-fold cross-check (expert SHOULD-FIX c).

    Videos (not frames — two frames of one video share a skin, so a frame-level
    split would leak) are assigned round-robin to folds; each fold is scored by
    validate_glyph_classifier under a palette fitted on the OTHER fold(s), so
    the per-fold PASS is evidence the palette generalises across videos rather
    than memorising the very frames it was measured on.
    """
    vids = sorted({vid for vid, _ in entries})
    assign = {vid: i % n_folds for i, vid in enumerate(vids)}
    out: list[dict[str, Any]] = []
    for fold in range(n_folds):
        fit = [f for vid, f in entries if assign[vid] != fold]
        score = [f for vid, f in entries if assign[vid] == fold]
        palette = _fit_fold_palette(fit)
        if palette is None or not score:
            out.append({"fold": fold, "passed": False, "reason": "insufficient fit/score data"})
            continue
        v = validate_glyph_classifier([replace(f, palette=palette) for f in score])
        out.append(
            {
                "fold": fold,
                "score_videos": sorted({vid for vid, _ in entries if assign[vid] == fold}),
                "fitted_palette": {
                    "hue_centres": {r: round(h, 2) for r, h in palette.hue_centres.items()},
                    "ore_max_saturation": round(palette.ore_max_saturation, 2),
                },
                "n_frames": v.n_frames,
                "n_correct": v.n_correct,
                "accuracy": round(v.accuracy, 4),
                "passed": v.passed,
                "reason": v.reason,
            }
        )
    return out


def cmd_score(folds: int) -> int:
    if not LABELS.exists():
        print(f"no labels at {LABELS} — label the sheets first")
        return 1
    labels: dict[str, str] = json.loads(LABELS.read_text())
    all_rows = [json.loads(line) for line in META.read_text().splitlines()]
    rows, n_dupes = _dedupe_events(all_rows)
    if n_dupes:
        print(f"deduped {n_dupes} near-duplicate grant event(s) (same line re-sampled)")
    entries: list[tuple[str, LabeledGrantFrame]] = []
    scored_ids: list[str] = []
    skipped: list[str] = []
    for row in rows:
        cid = row["crop_id"]
        per_box: list[str] = []
        ok = True
        for b_i in range(len(row["boxes"])):
            lab = labels.get(f"{cid}_b{b_i}")
            if lab is None or lab in ("SKIP", "NOT_A_CARD", "UNKNOWN"):
                ok = False
                break
            per_box.append(lab)
        if not ok:
            skipped.append(cid)
            continue
        log_png = OUT_DIR / f"{cid}_log.png"
        bgr = cv2.imread(str(log_png))
        if bgr is None:
            skipped.append(cid)
            continue
        crop = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        # score under the CANONICAL card palette — the palette production uses.
        # (The board-derived per-game palette stored in meta is provenance only;
        # the valset measured it mis-sitting the fixed card-icon hues.)
        entries.append(
            (
                str(row["video_id"]),
                LabeledGrantFrame(
                    log_crop_rgb=crop,
                    glyph_boxes=[
                        (int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in row["boxes"]
                    ],
                    expected=Counter(per_box),
                    palette=CARD_PALETTE,
                    expected_by_box=tuple(per_box),
                ),
            )
        )
        scored_ids.append(cid)
    frames = [f for _, f in entries]
    v = validate_glyph_classifier(frames)
    fold_results = _fold_results(entries, folds) if folds >= 2 else []
    result = {
        "passed": v.passed,
        "n_frames": v.n_frames,
        "n_correct": v.n_correct,
        "accuracy": round(v.accuracy, 4),
        "n_boxes": v.n_boxes,
        "n_unread_boxes": v.n_unread_boxes,
        "confusion": [list(c) for c in v.confusion],
        "reason": v.reason,
        # Binds this PASS to the exact classifier scored (expert SHOULD-FIX
        # 2026-07-05): glyph_classifier_is_validated re-verifies it at gate time,
        # so a stale artifact (classifier edited since) or a hand-edited JSON
        # cannot unblock the harvest. Re-run `score` after any glyph_anchor edit.
        "classifier_fingerprint": v.classifier_fingerprint,
        # Binds the artifact to the LIVE palette/threshold constants + detector
        # source (expert SHOULD-FIX a/b): load_glyph_validation recomputes this
        # at load time and refuses a stale PASS with the reason logged.
        "validation_fingerprint": validation_fingerprint(),
        "git_rev": _git_rev(),
        "scored_frames": scored_ids,
        "skipped_frames": skipped,
        "bar": {
            "min_frames": 8,
            "min_accuracy": 0.98,
            "zero_ore_brick": True,
        },
        "margins": _margin_summary(frames),
        "folds": fold_results,
    }
    VALIDATION_OUT.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Glyph-classifier validation",
        "",
        f"**Verdict: {'PASSED' if v.passed else 'NOT PASSED'}**"
        + (f" — {v.reason}" if v.reason else ""),
        "",
        f"- frames: {v.n_frames} labelled grant events, {v.n_correct} exact"
        f" ({v.accuracy:.1%} accuracy; bar >= 98%)",
        f"- boxes: {v.n_boxes} labelled icons, {v.n_unread_boxes} fail-closed unread (coverage)",
        f"- skipped (unlabelled/unreadable ground truth): {len(skipped)}",
        f"- classifier fingerprint: `{v.classifier_fingerprint}`",
        f"- validation fingerprint: `{result['validation_fingerprint']}`"
        f" (git `{result['git_rev']}`)",
        "",
        "| true \\ predicted | count |",
        "|---|---|",
    ]
    lines += [f"| {t} -> {p} | {c} |" for t, p, c in v.confusion]
    for fr in fold_results:
        lines.append(
            f"- fold {fr['fold']}: "
            + (
                f"{fr['n_correct']}/{fr['n_frames']} ({fr['accuracy']:.1%}) "
                f"{'PASSED' if fr['passed'] else 'NOT PASSED'}"
                if "n_frames" in fr
                else f"NOT PASSED — {fr['reason']}"
            )
        )
    REPORT_OUT.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {VALIDATION_OUT} and {REPORT_OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--videos", type=int, default=30)
    ex.add_argument("--seed", type=int, default=0)
    sub.add_parser("sheet")
    sc = sub.add_parser("score")
    sc.add_argument(
        "--folds",
        type=int,
        default=2,
        help="video-disjoint cross-check folds written into the artifact (0 disables)",
    )
    args = ap.parse_args()
    if args.cmd == "extract":
        return cmd_extract(args.videos, args.seed)
    if args.cmd == "sheet":
        return cmd_sheet()
    return cmd_score(args.folds)


if __name__ == "__main__":
    raise SystemExit(main())
