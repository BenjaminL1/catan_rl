#!/usr/bin/env python3
"""Measure the real Colonist 1v1 player-colour palette from ThePhantom footage.

The Tier-5 harvest was a NO-GO because ``catan_rl.human_data.openings`` only had
GREEN + BLACK calibrated, so every game with a red/blue/orange/… opponent failed
``read_hud_seat_colors`` (``hud_unreadable``). This tool MEASURES the colours that
actually appear — it invents no HSV values. Two subcommands:

  measure  Sample ``high``-tier manifest videos (reusing the setup-window
           timestamps already banked in ``data/human/glyph_valset/meta.jsonl``),
           grab full 1080p frames, and per seat measure BOTH
             (i) the HUD seat-avatar RING dominant saturated HSV (what
                 ``read_hud_seat_colors`` keys on), and
             (ii) placed PIECE (settlement/road/city) body HSV on the board.
           Also samples the board's saturated TILE hues (for the collision flag).
           Writes the raw per-video / per-frame / per-seat measurements to JSON.

  derive   Cluster the raw ring measurements into colour identities, derive
           per-identity NON-OVERLAPPING HSV ranges for ring AND piece (each
           justified by its measured sample spread), flag colours whose piece
           hue collides with a same-hued board tile (``tile_subtract``), and flag
           colours seen in too few games to calibrate (harvest-exclusion). Writes
           ``data/human/color_survey.json`` + ``data/human/color_survey.md``.

Frame acquisition mirrors ``scripts/glyph_valset.py`` (yt-dlp ``--print`` stream
URL + a single ffmpeg seek) — fast, CPU-only, download-then-discard, no PNG
accumulation. Never imports ``gui/`` (or ``board_cv``, which pulls pygame) or the
training path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "human" / "strength_manifest.json"
META = REPO / "data" / "human" / "glyph_valset" / "meta.jsonl"
RAW_OUT = REPO / "data" / "human" / "color_survey_raw.json"
SURVEY_JSON = REPO / "data" / "human" / "color_survey.json"
SURVEY_MD = REPO / "data" / "human" / "color_survey.md"

# --- HUD avatar geometry (fixed Colonist overlay at 1080p) --------------------
# The two seat panels stack bottom-right; the avatar disk sits at the left of each
# panel. Centres are frame-fixed at 1080p (verified across videos): top seat =
# opponent, bottom seat = POV/recorder. Ring sampling at r=42 with a [0.35,0.88]·r
# annulus is forgiving of a few-px centre drift and skips the central pawn +
# outer white border.
AVATAR_TOP: tuple[int, int] = (1561, 885)
AVATAR_BOT: tuple[int, int] = (1561, 1003)
AVATAR_R = 42

# Board search region for placed pieces (exclude the right-column HUD/log overlay
# and the very top log band). Pieces are the dominant saturated blobs of the
# player's specific hue inside this box.
BOARD_X = (140, 1435)
BOARD_Y = (110, 1075)


# --- frame acquisition (stream + ffmpeg seek, like glyph_valset) --------------
def _ffmpeg() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def stream_and_duration(video_id: str) -> tuple[str, float] | None:
    try:
        out = subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestvideo[height<=1080]/best",
                "--print",
                "%(duration)s|%(url)s",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
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


def grab_frame_rgb(url: str, t: float) -> npt.NDArray[np.uint8] | None:
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


# --- measurement primitives ---------------------------------------------------
def circular_hue_stats(hue: npt.NDArray[np.floating]) -> tuple[float, float]:
    """(circular-mean hue in 0..179, concentration R in 0..1) — handles the red
    wraparound at 0/180 that a plain median mangles."""
    ang = hue.astype(float) * 2.0 * np.pi / 180.0
    c = float(np.cos(ang).mean())
    s = float(np.sin(ang).mean())
    mean = (np.degrees(np.arctan2(s, c)) % 360.0) / 2.0
    return mean, float(np.hypot(c, s))


@dataclass
class RingSample:
    kind: str  # "colored" | "dark" | "white"
    colored_frac: float
    hue: float | None
    hue_conc: float | None
    sat: float
    val: float
    sat_lo: float
    sat_hi: float
    val_lo: float
    val_hi: float


def measure_ring(rgb: npt.NDArray[np.uint8], cx: int, cy: int, r: int = AVATAR_R) -> RingSample:
    box = rgb[cy - r : cy + r, cx - r : cx + r]
    hsv = cv2.cvtColor(box, cv2.COLOR_RGB2HSV)
    yy, xx = np.mgrid[0 : box.shape[0], 0 : box.shape[1]]
    d = np.hypot(xx - r, yy - r)
    ring = (d > 0.35 * r) & (d < 0.88 * r)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colored = ring & (ss > 90) & (vv > 70)
    cf = float(colored.sum()) / float(ring.sum())
    if cf > 0.15:
        hue, conc = circular_hue_stats(hh[colored])
        return RingSample(
            kind="colored",
            colored_frac=round(cf, 3),
            hue=round(hue, 1),
            hue_conc=round(conc, 3),
            sat=float(np.median(ss[colored])),
            val=float(np.median(vv[colored])),
            sat_lo=float(np.percentile(ss[colored], 10)),
            sat_hi=float(np.percentile(ss[colored], 90)),
            val_lo=float(np.percentile(vv[colored], 10)),
            val_hi=float(np.percentile(vv[colored], 90)),
        )
    vmed = float(np.median(vv[ring]))
    return RingSample(
        kind="white" if vmed > 150 else "dark",
        colored_frac=round(cf, 3),
        hue=None,
        hue_conc=None,
        sat=float(np.median(ss[ring])),
        val=vmed,
        sat_lo=float(np.percentile(ss[ring], 10)),
        sat_hi=float(np.percentile(ss[ring], 90)),
        val_lo=float(np.percentile(vv[ring], 10)),
        val_hi=float(np.percentile(vv[ring], 90)),
    )


def _board_slices() -> tuple[slice, slice]:
    return slice(BOARD_Y[0], BOARD_Y[1]), slice(BOARD_X[0], BOARD_X[1])


def measure_pieces_colored(
    rgb: npt.NDArray[np.uint8], ring_hue: float, hue_pm: float = 12.0
) -> dict | None:
    """Measure placed piece body HSV for a colour whose ring hue is ``ring_hue``.

    Seeds a circular hue window around the measured ring hue (pieces share the
    player's hue identity but render at higher saturation), high saturation, on
    the board region; aggregates the pixels of the strong compact blobs. Returns
    None if too little piece mass is found."""
    ys, xs = _board_slices()
    board = rgb[ys, xs]
    hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)
    hh = hsv[:, :, 0].astype(int)
    ss, vv = hsv[:, :, 1], hsv[:, :, 2]
    dhue = np.minimum(np.abs(hh - ring_hue), 180 - np.abs(hh - ring_hue))
    mask = ((dhue <= hue_pm) & (ss > 130) & (vv > 110)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(mask.shape, bool)
    n_blobs = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 70 or area > 9000:
            continue
        keep |= lab == i
        n_blobs += 1
    if n_blobs == 0 or int(keep.sum()) < 200:
        return None
    hsel = hh[keep].astype(float)
    hue, conc = circular_hue_stats(hsel)
    return {
        "n_blobs": n_blobs,
        "n_px": int(keep.sum()),
        "hue": round(hue, 1),
        "hue_conc": round(conc, 3),
        "sat": float(np.median(ss[keep])),
        "val": float(np.median(vv[keep])),
        "sat_lo": float(np.percentile(ss[keep], 5)),
        "sat_hi": float(np.percentile(ss[keep], 95)),
        "val_lo": float(np.percentile(vv[keep], 5)),
        "val_hi": float(np.percentile(vv[keep], 95)),
        "hue_lo": float(np.percentile(hsel, 5)),
        "hue_hi": float(np.percentile(hsel, 95)),
    }


def measure_pieces_dark(rgb: npt.NDArray[np.uint8]) -> dict | None:
    """BLACK pieces: dark, low-saturation compact blobs on the board (excluding
    the number tokens on hex centres is impossible without the lattice, so this is
    a coarse body-HSV read — reported with its spread, not a segmentation)."""
    ys, xs = _board_slices()
    board = rgb[ys, xs]
    hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)
    ss, vv = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((vv < 70) & (ss < 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(mask.shape, bool)
    n_blobs = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 150 or area > 5000:
            continue
        if w == 0 or h == 0 or not (0.3 < w / h < 3.5):
            continue
        keep |= lab == i
        n_blobs += 1
    if n_blobs == 0 or int(keep.sum()) < 200:
        return None
    return {
        "n_blobs": n_blobs,
        "n_px": int(keep.sum()),
        "sat": float(np.median(ss[keep])),
        "val": float(np.median(vv[keep])),
        "sat_lo": float(np.percentile(ss[keep], 5)),
        "sat_hi": float(np.percentile(ss[keep], 95)),
        "val_lo": float(np.percentile(vv[keep], 5)),
        "val_hi": float(np.percentile(vv[keep], 95)),
    }


def measure_pieces_white(rgb: npt.NDArray[np.uint8]) -> dict | None:
    """WHITE pieces: bright, near-zero-saturation compact blobs on the board. Like
    the dark path this is a coarse body-HSV read (white number tokens / borders /
    port glyphs share the signature — discriminated downstream by the lattice snap,
    not here); reported with its spread."""
    ys, xs = _board_slices()
    hsv = cv2.cvtColor(rgb[ys, xs], cv2.COLOR_RGB2HSV)
    ss, vv = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((vv > 200) & (ss < 45)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(mask.shape, bool)
    n_blobs = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 120 or area > 5000:
            continue
        if w == 0 or h == 0 or not (0.3 < w / h < 3.5):
            continue
        keep |= lab == i
        n_blobs += 1
    if n_blobs == 0 or int(keep.sum()) < 150:
        return None
    return {
        "n_blobs": n_blobs,
        "n_px": int(keep.sum()),
        "sat": float(np.median(ss[keep])),
        "val": float(np.median(vv[keep])),
        "sat_lo": float(np.percentile(ss[keep], 5)),
        "sat_hi": float(np.percentile(ss[keep], 95)),
        "val_lo": float(np.percentile(vv[keep], 5)),
        "val_hi": float(np.percentile(vv[keep], 95)),
    }


def measure_tile_hues(rgb: npt.NDArray[np.uint8]) -> dict:
    """Histogram of the board's saturated (tile) hues — the collision reference
    for the ``tile_subtract`` flag. Reported as the fraction of saturated board
    pixels in each 10-wide hue bin."""
    ys, xs = _board_slices()
    hsv = cv2.cvtColor(rgb[ys, xs], cv2.COLOR_RGB2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sat = (ss > 60) & (vv > 60)
    hues = hh[sat].astype(int)
    if hues.size == 0:
        return {}
    hist, _ = np.histogram(hues, bins=18, range=(0, 180))
    frac = hist / hist.sum()
    return {
        f"{b * 10}-{b * 10 + 10}": round(float(frac[b]), 3) for b in range(18) if frac[b] > 0.01
    }


# --- measure subcommand -------------------------------------------------------
def load_setup_timestamps() -> dict[str, float]:
    """{video_id: setup-window timestamp} from the banked glyph valset meta, for
    manifest ``high`` videos only. These t are 'received starting resources'
    events — HUD visible, opening pieces on the board."""
    videos = json.loads(MANIFEST.read_text())["videos"]
    high = {v["video_id"] for v in videos if v["strength"] == "high"}
    ts: dict[str, float] = {}
    for line in META.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["video_id"] in high:
            ts.setdefault(d["video_id"], float(d["t"]))
    return ts


def cmd_measure(video_ids: list[str], offsets: list[float], merge: bool = False) -> int:
    ts_map = load_setup_timestamps()
    picks = [(v, ts_map.get(v, 150.0)) for v in video_ids] if video_ids else sorted(ts_map.items())
    records: list[dict] = []
    # --merge keeps prior records for videos NOT re-measured (network grabs are
    # flaky, so re-measuring a single dropped video should not discard the corpus).
    prior: dict[str, list[dict]] = defaultdict(list)
    if merge and RAW_OUT.exists():
        for rec in json.loads(RAW_OUT.read_text()):
            prior[rec["video_id"]].append(rec)
    picked_ids = {v for v, _ in picks}
    for vid, base_t in picks:
        s = stream_and_duration(vid)
        if s is None:
            print(f"{vid}\tSTREAM_FAIL", flush=True)
            records.append({"video_id": vid, "error": "stream_fail"})
            continue
        url, dur = s
        for off in offsets:
            t = base_t + off
            rgb = grab_frame_rgb(url, t)
            if rgb is None or rgb.shape[:2] != (1080, 1920):
                print(f"{vid}\tt={t}\tNOFRAME", flush=True)
                continue
            rec: dict = {"video_id": vid, "t": t, "seats": {}}
            for seat, (cx, cy) in (("top", AVATAR_TOP), ("bot", AVATAR_BOT)):
                ring = measure_ring(rgb, cx, cy)
                seat_rec: dict = {"ring": ring.__dict__}
                if ring.kind == "colored" and ring.hue is not None:
                    p = measure_pieces_colored(rgb, ring.hue)
                    if p is not None:
                        seat_rec["piece"] = p
                elif ring.kind == "dark":
                    p = measure_pieces_dark(rgb)
                    if p is not None:
                        seat_rec["piece_dark"] = p
                elif ring.kind == "white":
                    p = measure_pieces_white(rgb)
                    if p is not None:
                        seat_rec["piece_white"] = p
                rec["seats"][seat] = seat_rec
            rec["tile_hues"] = measure_tile_hues(rgb)
            records.append(rec)

            def _fmt(r: dict) -> str:
                return f"{r['kind']} h={r['hue']} s={r['sat']:.0f} v={r['val']:.0f}"

            tp = rec["seats"]["top"]["ring"]
            bt = rec["seats"]["bot"]["ring"]
            print(f"{vid}\tt={int(t)}\tTOP {_fmt(tp)}\tBOT {_fmt(bt)}", flush=True)
    if merge:
        measured_ids = {r["video_id"] for r in records}
        for vid, recs in prior.items():
            if vid not in measured_ids and vid not in picked_ids:
                records.extend(recs)
        records.sort(key=lambda r: (r["video_id"], r.get("t", 0.0)))
    RAW_OUT.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {RAW_OUT} ({len(records)} records)")
    return 0


# --- derive subcommand: cluster measurements -> non-overlapping ranges --------
# Hue bins used to name a colored ring sample (OpenCV hue 0..179). Boundaries sit
# in the empty gaps BETWEEN the measured clusters (see color_survey.md), so a
# sample lands in exactly one identity. RED wraps the 0/180 seam.
_CALIBRATED_MIN_VIDEOS = 4  # >= this many distinct videos -> harvestable; else low-sample
_HUE_MARGIN = 5  # widen a measured hue cluster by this, then clip to the neighbour midpoint
_SV_MARGIN = 20  # widen a measured sat/val span by this (clamped to 0..255)
# Firm saturation split so an achromatic BLACK/WHITE box never overlaps a chromatic
# one (chromatic identities are separated from each other by hue, but from BLACK/
# WHITE only by saturation): achromatic S ceilings sit strictly below the chromatic
# S floor. Measured: BLACK ring sat med ~9, WHITE ~12; chromatic ring sat p10 >= ~100.
_CHROMATIC_SAT_FLOOR = 70
_BLACK_SAT_MAX = 60
_WHITE_SAT_MAX = 55


def classify_ring(kind: str, hue: float | None) -> str | None:
    if kind == "dark":
        return "BLACK"
    if kind == "white":
        return "WHITE"
    if hue is None:
        return None
    if hue >= 163 or hue <= 8:
        return "RED"
    if 8 < hue <= 22:
        return "ORANGE"
    if 45 <= hue <= 85:
        return "GREEN"
    if 85 < hue <= 100:
        return "LIGHTGREEN"
    if 100 < hue <= 120:
        return "BLUE"
    if 125 <= hue <= 150:
        return "PURPLE"
    return f"OTHER_{round(hue)}"


def _circ_mean(hues: list[float]) -> float:
    a = np.array(hues) * 2.0 * np.pi / 180.0
    return float((np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0) / 2.0)


def _clamp(x: float, lo: int = 0, hi: int = 255) -> int:
    return int(max(lo, min(hi, round(x))))


def _hue_wrap_span(hues: list[float]) -> tuple[float, float]:
    """Tight [lo, hi] hue interval covering ``hues`` on the 0..180 circle; hi may be
    < lo (wraparound, only RED). Chooses the representation with the smaller arc."""
    hs = sorted(h % 180 for h in hues)
    best = (hs[0], hs[-1], hs[-1] - hs[0])
    for i in range(len(hs) - 1):
        gap = hs[i + 1] - hs[i]
        arc = 180 - gap
        if arc < best[2]:
            best = (hs[i + 1], hs[i], arc)  # wrap: lo=after-gap, hi=before-gap
    return best[0], best[1]


# Piece-vs-tile hue collisions the on-board reader must handle. Derived from the
# measured board tile-hue histogram (color_survey.md): which same-hued tile (or the
# blue sea background) a colour's PIECE hue overlaps, and whether it needs the
# empty-baseline tile subtraction. The histogram measures tile HUE only (not tile
# saturation), so a "saturation floor separates piece from tile" claim cannot be
# verified from this survey -> fail-closed to tile_subtract=True whenever the piece
# hue overlaps a same-hued tile band.
_TILE_COLLISION: dict[str, dict] = {
    "RED": {
        "collides": True,
        "tiles": ["BRICK"],
        "separable_by_sat": False,
        "tile_subtract": True,
        "note": "piece hue wraps the 0/180 seam and overlaps the brick/red-hills tile "
        "band (measured hue 0-20 = ~11% of saturated board pixels). Red pieces are highly "
        "saturated (measured piece sat median ~248, p5 ~176), but this survey measures "
        "tile HUE only, not tile saturation, so a saturation-floor separation from brick "
        "cannot be verified from the data; apply the empty-baseline tile subtraction "
        "conservatively (fail-closed).",
    },
    "ORANGE": {
        "collides": True,
        "tiles": ["BRICK", "WHEAT"],
        "separable_by_sat": False,
        "tile_subtract": True,
        "note": "piece hue (16-19) sits inside the brick+wheat tile band (hue 10-30, "
        "~20% of saturated board pixels) with overlapping saturation; needs the "
        "empty-baseline tile subtraction.",
    },
    "GREEN": {
        "collides": True,
        "tiles": ["WOOD", "SHEEP"],
        "separable_by_sat": False,
        "tile_subtract": True,
        "note": "the known green_tile_suppressed collision: piece hue (62-65) matches the "
        "forest/pasture tiles (hue 60-70); needs the empty-baseline subtraction.",
    },
    "LIGHTGREEN": {
        "collides": True,
        "tiles": ["WOOD"],
        "separable_by_sat": False,
        "tile_subtract": True,
        "note": "piece hue (~96) overlaps the upper-green tile band (hue 90-100).",
    },
    "BLUE": {
        "collides": True,
        "tiles": ["SEA_BACKGROUND"],
        "separable_by_sat": False,
        "tile_subtract": True,
        "note": "CRITICAL: the board sits on a blue sea background (hue 100-110 = 58% of "
        "saturated board pixels), the same hue as blue pieces; needs the sea/tile "
        "subtraction or a tight sat+val gate.",
    },
    "PURPLE": {
        "collides": False,
        "tiles": [],
        "separable_by_sat": True,
        "tile_subtract": False,
        "note": "piece hue (~141) has no same-hued board tile (hue 130-150 is empty).",
    },
    "BLACK": {
        "collides": True,
        "tiles": ["ORE", "SHADOW"],
        "separable_by_sat": True,
        "tile_subtract": False,
        "note": "dark pieces (val <75) vs grey ore tiles / the robber; the existing "
        "hex-centre exclusion handles the robber, no baseline subtraction.",
    },
    "WHITE": {
        "collides": True,
        "tiles": ["NUMBER_TOKENS", "BORDERS"],
        "separable_by_sat": True,
        "tile_subtract": False,
        "note": "white pieces (val >200, sat <45) share their signature with white number "
        "tokens / vertex borders / port glyphs; the hex-centre + lattice-snap "
        "guards discriminate, not a hue subtraction.",
    },
}


def derive() -> int:
    records = json.loads(RAW_OUT.read_text())
    ring: dict[str, list[dict]] = defaultdict(list)
    ring_vids: dict[str, set[str]] = defaultdict(set)
    piece: dict[str, list[dict]] = defaultdict(list)
    piece_vids: dict[str, set[str]] = defaultdict(set)
    dark_piece: list[dict] = []
    white_piece: list[dict] = []
    tile_bins: dict[str, list[float]] = defaultdict(list)
    raw_by_video: dict[str, list[dict]] = defaultdict(list)
    n_frames = 0
    for r in records:
        if "seats" not in r:
            continue
        n_frames += 1
        for k, v in r.get("tile_hues", {}).items():
            tile_bins[k].append(v)
        for seat, sd in r["seats"].items():
            rg = sd["ring"]
            cid = classify_ring(rg["kind"], rg.get("hue"))
            raw_by_video[r["video_id"]].append(
                {
                    "t": r["t"],
                    "seat": seat,
                    "identity": cid,
                    "ring": rg,
                    "piece": sd.get("piece"),
                    "piece_dark": sd.get("piece_dark"),
                    "piece_white": sd.get("piece_white"),
                }
            )
            if cid is None:
                continue
            ring[cid].append(rg)
            ring_vids[cid].add(r["video_id"])
            if "piece" in sd:
                piece[cid].append(sd["piece"])
                piece_vids[cid].add(r["video_id"])
            if "piece_dark" in sd and cid == "BLACK":
                dark_piece.append(sd["piece_dark"])
            if "piece_white" in sd and cid == "WHITE":
                white_piece.append(sd["piece_white"])

    # chromatic identities in hue order -> Voronoi midpoints guarantee disjoint hue
    chroma = [c for c in ring if c not in ("BLACK", "WHITE") and not c.startswith("OTHER")]

    def ring_hue_mean(c: str) -> float:
        return _circ_mean([s["hue"] for s in ring[c] if s.get("hue") is not None])

    def piece_hue_mean(c: str) -> float:
        return _circ_mean([p["hue"] for p in piece[c]]) if piece.get(c) else ring_hue_mean(c)

    def build_hue_ranges(means: dict[str, float], spans: dict[str, tuple[float, float]]) -> dict:
        """Per chromatic identity, a hue range = the widened measured cluster clipped
        to its Voronoi cell (bounded by the midpoints to the two hue neighbours). All
        arithmetic is done in signed-offset space from the identity's mean, so the
        0/180 seam (RED) is handled uniformly. Each cell is shrunk 0.5 on both sides
        so two adjacent cells never round to a shared integer boundary -> the derived
        ranges are pairwise hue-disjoint by construction."""
        order = sorted(means, key=lambda c: means[c])
        out: dict[str, list[int]] = {}
        n = len(order)
        for i, c in enumerate(order):
            mu = means[c]
            gap_prev = ((mu - means[order[(i - 1) % n]]) % 180) / 2.0
            gap_next = ((means[order[(i + 1) % n]] - mu) % 180) / 2.0
            cell_lo, cell_hi = -gap_prev + 0.5, gap_next - 0.5

            def off(h: float, mu: float = mu) -> float:  # signed hue offset from mu in (-90, 90]
                d = (h - mu) % 180
                return d - 180 if d > 90 else d

            lo_m, hi_m = spans[c]
            lo_off = max(cell_lo, off(lo_m) - _HUE_MARGIN)
            hi_off = min(cell_hi, off(hi_m) + _HUE_MARGIN)
            out[c] = [round(mu + lo_off) % 180, round(mu + hi_off) % 180]
        return out

    def sv_range(
        samples: list[dict], klo: str, khi: str, fallback: tuple[int, int], floor: int = 0
    ) -> list[int]:
        if not samples:
            return [max(fallback[0], floor), fallback[1]]
        lo = min(s[klo] for s in samples) - _SV_MARGIN
        hi = max(s[khi] for s in samples) + _SV_MARGIN
        return [max(_clamp(lo), floor), _clamp(hi)]

    ring_means = {c: ring_hue_mean(c) for c in chroma}
    ring_spans = {
        c: _hue_wrap_span([s["hue"] for s in ring[c] if s.get("hue") is not None]) for c in chroma
    }
    ring_hue = build_hue_ranges(ring_means, ring_spans)
    piece_means = {c: piece_hue_mean(c) for c in chroma}
    piece_spans = {
        c: _hue_wrap_span([p["hue"] for p in piece[c]]) if piece.get(c) else ring_spans[c]
        for c in chroma
    }
    piece_hue = build_hue_ranges(piece_means, piece_spans)

    identities: dict[str, dict] = {}
    for cid in sorted(set(ring) | {"BLACK", "WHITE"}):
        if cid.startswith("OTHER"):
            continue
        nv = len(ring_vids.get(cid, set()))
        low = nv < _CALIBRATED_MIN_VIDEOS
        entry: dict = {
            "n_videos": nv,
            "n_ring_samples": len(ring.get(cid, [])),
            "n_piece_samples": len(dark_piece) if cid == "BLACK" else len(piece.get(cid, [])),
            "kind": "achromatic" if cid in ("BLACK", "WHITE") else "chromatic",
            "low_sample": low,
            "harvest_exclude": low,
        }
        if cid in ("BLACK", "WHITE"):
            rsv = ring[cid]
            if cid == "BLACK":
                entry["ring"] = {
                    "hue": [0, 179],
                    "sat": [0, _BLACK_SAT_MAX],
                    "val": sv_range(rsv, "val_lo", "val_hi", (40, 120)),
                }
                entry["ring"]["val"][1] = min(entry["ring"]["val"][1], 130)
                entry["piece"] = {
                    "hue": [0, 179],
                    "sat": [0, _BLACK_SAT_MAX],
                    "val": sv_range(dark_piece, "val_lo", "val_hi", (20, 75)),
                }
            else:  # WHITE
                entry["ring"] = {
                    "hue": [0, 179],
                    "sat": [0, _WHITE_SAT_MAX],
                    "val": sv_range(rsv, "val_lo", "val_hi", (150, 255)),
                }
                entry["ring"]["val"][0] = max(entry["ring"]["val"][0], 145)
                wv = sv_range(white_piece, "val_lo", "val_hi", (200, 255))
                entry["piece"] = {
                    "hue": [0, 179],
                    "sat": [0, _WHITE_SAT_MAX],
                    "val": [max(wv[0], 150), wv[1]],
                }
                entry["n_piece_samples"] = len(white_piece)
            entry["measured"] = {
                "ring_sat_med": _med([s["sat"] for s in rsv]),
                "ring_val_med": _med([s["val"] for s in rsv]),
            }
        else:
            floor = _CHROMATIC_SAT_FLOOR
            pcs = piece.get(cid, [])
            entry["ring"] = {
                "hue": ring_hue[cid],
                "sat": sv_range(ring[cid], "sat_lo", "sat_hi", (90, 255), floor),
                "val": sv_range(ring[cid], "val_lo", "val_hi", (110, 255)),
            }
            entry["piece"] = {
                "hue": piece_hue[cid],
                "sat": sv_range(pcs, "sat_lo", "sat_hi", (130, 255), floor),
                "val": sv_range(pcs, "val_lo", "val_hi", (110, 255)),
            }
            ring_hues = [s["hue"] for s in ring[cid] if s.get("hue") is not None]
            entry["measured"] = {
                "ring_hue_cmean": round(ring_means[cid], 1),
                "ring_hue_conc": round(_conc(ring_hues), 3),
                "piece_hue_cmean": round(piece_means[cid], 1),
                "ring_sat_med": _med([s["sat"] for s in ring[cid]]),
                "ring_val_med": _med([s["val"] for s in ring[cid]]),
            }
        entry["tile_collision"] = _TILE_COLLISION.get(
            cid, {"collides": None, "tile_subtract": False}
        )
        identities[cid] = entry

    tile_hist = {
        k: round(float(np.mean(v)), 3)
        for k, v in sorted(tile_bins.items(), key=lambda kv: int(kv[0].split("-")[0]))
    }

    survey = {
        "generated_by": "scripts/color_survey.py derive",
        "source": {
            "manifest": "data/human/strength_manifest.json",
            "setup_ts_from": "data/human/glyph_valset/meta.jsonl",
            "videos_sampled": len({r["video_id"] for r in records}),
            "frames_measured": n_frames,
            "raw_measurements": "data/human/color_survey_raw.json",
        },
        "calibrated_min_videos": _CALIBRATED_MIN_VIDEOS,
        "hue_convention": "OpenCV RGB->HSV hue 0..179; a hue range [lo, hi] with lo > hi wraps "
        "the 0/180 seam (RED only)",
        "identities": identities,
        "tile_hue_histogram": tile_hist,
        "raw_measurements_by_video": raw_by_video,
    }
    SURVEY_JSON.write_text(json.dumps(survey, indent=2))
    SURVEY_MD.write_text(_render_md(survey))
    _assert_non_overlap(survey)  # fail loudly if a derived range pair overlaps
    print(f"wrote {SURVEY_JSON} and {SURVEY_MD}")
    print(f"identities: {', '.join(sorted(identities))}")
    return 0


def _med(xs: list[float]) -> float:
    return round(float(np.median(xs)), 1) if xs else 0.0


def _conc(hues: list[float]) -> float:
    if not hues:
        return 0.0
    a = np.array(hues) * 2.0 * np.pi / 180.0
    return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))


def _hue_overlap(a: list[int], b: list[int]) -> bool:
    """Do two OpenCV-hue ranges (lo may exceed hi = wraparound) overlap?"""

    def norm(r: list[int]) -> list[tuple[int, int]]:
        lo, hi = r
        return [(lo, hi)] if lo <= hi else [(lo, 179), (0, hi)]

    for lo1, hi1 in norm(a):
        for lo2, hi2 in norm(b):
            if lo1 <= hi2 and lo2 <= hi1:
                return True
    return False


def _box_overlap(x: dict, y: dict) -> bool:
    """Two HSV boxes overlap iff they overlap in H AND S AND V."""
    return (
        _hue_overlap(x["hue"], y["hue"])
        and x["sat"][0] <= y["sat"][1]
        and y["sat"][0] <= x["sat"][1]
        and x["val"][0] <= y["val"][1]
        and y["val"][0] <= x["val"][1]
    )


def _assert_non_overlap(survey: dict) -> None:
    ids = survey["identities"]
    for field in ("ring", "piece"):
        names = sorted(ids)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = ids[names[i]][field], ids[names[j]][field]
                if _box_overlap(a, b):
                    raise AssertionError(
                        f"{field} ranges overlap: {names[i]} {a} vs {names[j]} {b}"
                    )


def _render_md(survey: dict) -> str:
    ids = survey["identities"]
    order = sorted(ids, key=lambda c: (-ids[c]["n_videos"], c))
    lines = [
        "# Colonist 1v1 player-colour survey",
        "",
        "Measured from real ThePhantom `high`-tier footage by "
        "`scripts/color_survey.py` — **no HSV value here is invented**; every range "
        "is derived from the on-frame measurements in `color_survey_raw.json` "
        "(HUD seat-avatar rings + on-board pieces). This is the calibration source "
        "for widening `openings.PALETTE` / `_HUD_RING` beyond GREEN+BLACK (the "
        "Tier-5 `hud_unreadable` NO-GO).",
        "",
        f"- **Videos sampled:** {survey['source']['videos_sampled']} "
        f"(`high` tier, setup-window frames) — **{survey['source']['frames_measured']} frames**.",
        f"- **Calibrated threshold:** a colour seen in **>= {survey['calibrated_min_videos']} "
        "distinct videos** is harvestable; fewer -> `low_sample` (harvest-excluded until "
        "more games, not a guessed range).",
        f"- **Hue convention:** {survey['hue_convention']}.",
        "",
        "## Colour histogram (how often each identity was seen)",
        "",
        "| identity | kind | videos | ring n | piece n | status |",
        "|---|---|---|---|---|---|",
    ]
    for c in order:
        e = ids[c]
        status = "LOW-SAMPLE (excluded)" if e["harvest_exclude"] else "calibrated"
        lines.append(
            f"| {c} | {e['kind']} | {e['n_videos']} | {e['n_ring_samples']} | "
            f"{e['n_piece_samples']} | {status} |"
        )
    lines += ["", "## Per-colour derived HSV ranges (OpenCV H 0..179, S/V 0..255)", ""]
    lines.append(
        "| identity | ring H | ring S | ring V | piece H | piece S | piece V | tile_subtract |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in order:
        e = ids[c]
        r, p = e["ring"], e["piece"]
        ts = e["tile_collision"].get("tile_subtract")
        lines.append(
            f"| {c} | {r['hue']} | {r['sat']} | {r['val']} | {p['hue']} | {p['sat']} | "
            f"{p['val']} | {ts} |"
        )
    lines += [
        "",
        "## Measured spread (non-overlap proof)",
        "",
        "Chromatic identities are separated by **hue** (boundaries placed at the "
        "Voronoi midpoints between adjacent measured cluster means, so no two hue "
        "ranges share a value); achromatic BLACK/WHITE are separated from every "
        "chromatic colour by **saturation** (S <= ~90 vs chromatic S floors >= ~90) "
        "and from each other by **value** (BLACK V <= 130 vs WHITE V >= 145). The "
        "unit test `test_color_survey.py` enforces pairwise non-overlap of the 3-D "
        "HSV boxes for both ring and piece.",
        "",
    ]
    for c in order:
        e = ids[c]
        m = e.get("measured", {})
        if e["kind"] == "chromatic":
            lines.append(
                f"- **{c}** ({e['n_videos']} vids, {e['n_ring_samples']} ring samples): "
                f"ring hue circular-mean **{m.get('ring_hue_cmean')}** "
                f"(concentration {m.get('ring_hue_conc')}), piece hue circular-mean "
                f"**{m.get('piece_hue_cmean')}**; ring sat/val med "
                f"{m.get('ring_sat_med')}/{m.get('ring_val_med')}."
            )
        else:
            lines.append(
                f"- **{c}** ({e['n_videos']} vids, {e['n_ring_samples']} ring samples): "
                f"achromatic; ring sat/val med {m.get('ring_sat_med')}/{m.get('ring_val_med')}."
            )
    lines += [
        "",
        "## Tile / background hue collisions (`tile_subtract`)",
        "",
        "Board saturated-pixel hue histogram (mean fraction per 10-wide bin, across "
        "all sampled frames) — the reference for which piece colours collide with a "
        "same-hued tile or the sea background:",
        "",
        "| hue bin | fraction | |",
        "|---|---|---|",
    ]
    labels = {
        "0-10": "brick / red-hills",
        "10-20": "brick / wheat",
        "20-30": "wheat",
        "30-40": "sheep / pasture",
        "50-60": "sheep",
        "60-70": "wood / forest",
        "70-80": "wood",
        "90-100": "upper-green",
        "100-110": "**SEA BACKGROUND**",
    }
    for k, v in survey["tile_hue_histogram"].items():
        lines.append(f"| {k} | {v} | {labels.get(k, '')} |")
    lines += ["", "Resulting flags:", ""]
    for c in order:
        tc = ids[c]["tile_collision"]
        lines.append(f"- **{c}**: `tile_subtract={tc.get('tile_subtract')}` — {tc.get('note', '')}")
    lines += [
        "",
        "## Excluded / low-sample colours",
        "",
        "These appeared in too few distinct games to calibrate a trustworthy range; "
        "their measured ranges are recorded (and kept pairwise non-overlapping so a "
        "fail-closed reader never *mislabels* them) but they are **harvest-excluded** "
        "until more games are gathered — a game whose seat is one of these should still "
        "emit a typed `hud_unreadable`/`player_colors_invalid` rejection, never a guess:",
        "",
    ]
    for c in order:
        if ids[c]["harvest_exclude"]:
            lines.append(f"- **{c}** — only {ids[c]['n_videos']} video(s).")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="measure ring+piece HSV across videos")
    m.add_argument("--videos", nargs="*", default=[], help="explicit ids (default: all high)")
    m.add_argument(
        "--offsets",
        nargs="*",
        type=float,
        default=[0.0, 40.0],
        help="seconds past the setup ts to sample",
    )
    m.add_argument(
        "--merge", action="store_true", help="keep prior records for videos not re-measured"
    )
    sub.add_parser("derive", help="cluster raw measurements -> color_survey.json + .md")
    args = ap.parse_args()
    if args.cmd == "measure":
        return cmd_measure(args.videos, args.offsets, args.merge)
    if args.cmd == "derive":
        return derive()
    return 1


if __name__ == "__main__":
    sys.exit(main())
