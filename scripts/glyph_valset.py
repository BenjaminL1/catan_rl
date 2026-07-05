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
           data/human/glyph_validation.json + a markdown report.

Local + deterministic (yt-dlp / ffmpeg / easyocr / opencv — no API calls).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
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
    LabeledGrantFrame,
    calibrate_glyph_palette,
    validate_glyph_classifier,
)
from catan_rl.human_data.logparse import LOG_CROP_FRAC, _normalise

MANIFEST = REPO / "data" / "human" / "strength_manifest.json"
OUT_DIR = REPO / "data" / "human" / "glyph_valset"
META = OUT_DIR / "meta.jsonl"
LABELS = OUT_DIR / "labels.json"
VALIDATION_OUT = REPO / "data" / "human" / "glyph_validation.json"
REPORT_OUT = REPO / "data" / "human" / "glyph_validation.md"

#: OCR-tolerant grant-line matcher ("received" often reads "receivea"/"recelved").
GRANT_RE = re.compile(r"rece\w{0,4} starting resources")

# --- glyph-box detector tuning (calibrated on real 1080p log crops) -------------
BOX_MIN_AREA = 55
BOX_MIN_H, BOX_MAX_H = 9, 30
BOX_AR_LO, BOX_AR_HI = 0.35, 1.8
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


def detect_glyph_boxes(
    crop_rgb: npt.NDArray[np.uint8],
    line_box: tuple[int, int, int, int],
    text_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Detect the granted card-icon boxes for one grant line.

    The mask is COLOUR-DISTANCE FROM THE PANEL BACKGROUND (per-crop median — the
    panel skin varies: light-grey on some UIs, warm cream on others, and the cream
    overlaps any fixed grey-stone S/V band). Candidates are icon-sized components
    that sit either (a) ON the grant line's row, RIGHT of its text (excludes the
    line-leading avatar glyph), or (b) on the WRAP row just below — but never
    inside another log line's y-band (the neighbouring "placed a Road/Settlement"
    lines carry their own piece glyphs, which are NOT granted cards).
    """
    h, _w = crop_rgb.shape[:2]
    _x0, y0, x1, y1 = line_box
    line_h = max(y1 - y0, 12)
    band_y0 = max(0, y0 - 3)
    band_y1 = min(h, y1 + int(1.3 * line_h))

    band_px = crop_rgb[band_y0:band_y1, :].reshape(-1, 3)
    panel = np.median(band_px, axis=0)  # background dominates the band area
    dist = np.abs(crop_rgb.astype(np.int32) - panel.astype(np.int32)).max(axis=2)
    mask = (dist > 30).astype(np.uint8)
    band = np.zeros_like(mask)
    band[band_y0:band_y1, :] = 1
    mask &= band
    mask = np.asarray(
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), dtype=np.uint8
    )

    # y-bands of OTHER text lines (their piece glyphs must never become candidates).
    other_bands = [
        (ty0 - 2, ty1 + 2)
        for tx0, ty0, tx1, ty1 in text_boxes
        if not (abs(ty0 - y0) < line_h // 2 and abs(ty1 - y1) < line_h // 2)
    ]

    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        bx, by, bw, bh, area = (int(stats[i, k]) for k in range(5))
        if area < BOX_MIN_AREA or not (BOX_MIN_H <= bh <= BOX_MAX_H):
            continue
        cx, cy = bx + bw / 2, by + bh / 2
        if any(tx0 <= cx <= tx1 and ty0 <= cy <= ty1 for tx0, ty0, tx1, ty1 in text_boxes):
            continue  # inside a text bbox (letters/avatar merged into OCR box)
        on_grant_row = y0 - 2 <= cy <= y1 + 2
        if on_grant_row:
            if bx < x1 - 4:
                continue  # left of / inside the grant text — the avatar glyph zone
        else:
            if any(oy0 <= cy <= oy1 for oy0, oy1 in other_bands):
                continue  # another log line's own piece glyph, not a granted card
        ar = bw / bh
        if BOX_AR_LO <= ar <= BOX_AR_HI:
            pieces = [(bx, bw)]
        elif ar > BOX_AR_HI:
            # tightly-packed icons merge into one wide run — split by the icon
            # PITCH derived from the line height (the CC's own height is smeared
            # by anti-aliasing, so width/height under-counts the icons).
            pitch = max(10.0, 0.72 * line_h)
            k = round(bw / pitch)
            if not 2 <= k <= 6:
                continue
            cell = bw / k
            pieces = [(bx + int(j * cell), int(cell)) for j in range(k)]
        else:
            continue
        for px, pw in pieces:
            # shrink 2px inward so the box samples the card body, not the border/panel
            boxes.append((px + 2, by + 2, px + pw - 2, by + bh - 2))
    boxes.sort(key=lambda b: (b[1] // line_h, b[0]))  # row-major, left to right
    return boxes


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


def cmd_score() -> int:
    if not LABELS.exists():
        print(f"no labels at {LABELS} — label the sheets first")
        return 1
    labels: dict[str, str] = json.loads(LABELS.read_text())
    all_rows = [json.loads(line) for line in META.read_text().splitlines()]
    rows, n_dupes = _dedupe_events(all_rows)
    if n_dupes:
        print(f"deduped {n_dupes} near-duplicate grant event(s) (same line re-sampled)")
    frames: list[LabeledGrantFrame] = []
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
        frames.append(
            LabeledGrantFrame(
                log_crop_rgb=crop,
                glyph_boxes=[(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in row["boxes"]],
                expected=Counter(per_box),
                palette=CARD_PALETTE,
                expected_by_box=tuple(per_box),
            )
        )
    v = validate_glyph_classifier(frames)
    result = {
        "passed": v.passed,
        "n_frames": v.n_frames,
        "n_correct": v.n_correct,
        "accuracy": round(v.accuracy, 4),
        "n_boxes": v.n_boxes,
        "n_unread_boxes": v.n_unread_boxes,
        "confusion": [list(c) for c in v.confusion],
        "reason": v.reason,
        "skipped_frames": skipped,
        "bar": {
            "min_frames": 8,
            "min_accuracy": 0.98,
            "zero_ore_brick": True,
        },
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
        "",
        "| true \\ predicted | count |",
        "|---|---|",
    ]
    lines += [f"| {t} -> {p} | {c} |" for t, p, c in v.confusion]
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
    sub.add_parser("score")
    args = ap.parse_args()
    if args.cmd == "extract":
        return cmd_extract(args.videos, args.seed)
    if args.cmd == "sheet":
        return cmd_sheet()
    return cmd_score()


if __name__ == "__main__":
    raise SystemExit(main())
