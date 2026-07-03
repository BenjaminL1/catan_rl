#!/usr/bin/env python3
"""Build ThePhantom strength manifest — label each channel video high/excluded/unknown.

Two deterministic, local signals (NO API / vision-model):

  1. tournament : title keyword match                       -> high  (source=tournament)
  2. ranked     : read ThePhantom's world rank off the 1v1  -> high  if N<=200
                  Global-leaderboard frame (scanned in the      excluded if N>200
                  last ~110s then first ~50s of the video)      unknown if no rank read

Output:
  data/human/strength_manifest.json   (machine source of truth, written incrementally)
  data/human/frames/<id>_rank.png      (winning leaderboard frame, kept for audit)

Resumable: video_ids already present in the JSON are skipped. Safe to Ctrl-C and rerun.

Usage:
  python scripts/build_strength_manifest.py                       # full channel
  python scripts/build_strength_manifest.py --sample ID1,ID2,...  # only these ids
  python scripts/build_strength_manifest.py --limit 20            # first N videos
  python scripts/build_strength_manifest.py --rebuild             # ignore existing JSON
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import cv2

CHANNEL = "https://www.youtube.com/@ThePhantomcatan/videos"
REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "human"
MANIFEST = OUT_DIR / "strength_manifest.json"
FRAMES_DIR = OUT_DIR / "frames"

# --- tournament signal ---------------------------------------------------------
# Conservative: EVERY one of ThePhantom's 15 real tournament uploads contains one of
# these strong series tokens, so we require one of them rather than bare tokens like
# "titans"/"playoff"/"semifinal" that collide with hype/metaphor titles. A false-high
# here is unrecoverable (no frame verification on the tournament path).
TOURNAMENT_RE = re.compile(r"\b(tournament|world\s*cup|invitational)\b", re.IGNORECASE)
# ...but never on a commentary/reaction/guide upload (that board is not his own game).
NON_OWN_GAME_RE = re.compile(
    r"react|reacting|reaction|watching|analy|guide|how\s*to|tips|explained|tier\s*list",
    re.IGNORECASE,
)

# --- rank signal ---------------------------------------------------------------
RANK_HIGH_MAX = 200  # <= this world rank counts as high
CANONICAL_HANDLE = "thephantom"  # the channel owner's exact leaderboard handle
RANK_CONF_MIN = 0.5  # drop rank tokens easyocr is unsure about
TAB_LABELS = {"friends", "global", "oceania", "australia", "victoria"}
# frame sample offsets: end-of-video first (leaderboard usually shown post-game), then
# two mid-video probes (between-games montages), then the start.
END_OFFSETS = [4, 12, 25, 45, 75, 110]
START_TIMES = [3, 9, 18, 30, 48]
MID_FRACTIONS = [0.5]  # one between-games probe (recall vs scan-cost balance)


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


def list_videos() -> list[tuple[str, str]]:
    """Return [(video_id, title)] for the whole channel (flat, fast)."""
    out = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", CHANNEL],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    vids = []
    for line in out.splitlines():
        if "\t" in line:
            vid, title = line.split("\t", 1)
            vids.append((vid.strip(), title.strip()))
    return vids


def is_tournament(title: str) -> bool:
    """A real 1v1 tournament upload: a strong series keyword AND not a reaction/guide."""
    return bool(TOURNAMENT_RE.search(title)) and not bool(NON_OWN_GAME_RE.search(title))


def _stream_and_duration(video_id: str) -> tuple[str, float] | None:
    """One yt-dlp call -> (direct media url, duration_seconds)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        out = subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bestvideo[height<=1080]/best",
                "--print",
                "%(duration)s|%(url)s",
                url,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    line = out.splitlines()[0] if out else ""
    if "|" not in line:
        return None
    dur_s, media_url = line.split("|", 1)
    try:
        dur = float(dur_s)
    except ValueError:
        return None
    if not media_url.startswith("http") or dur <= 0:
        return None
    return media_url, dur


def _grab_frame(media_url: str, t: float, out_path: Path) -> bool:
    try:
        subprocess.run(
            [
                _ffmpeg(),
                "-nostdin",
                "-loglevel",
                "error",
                "-ss",
                str(max(0, t)),
                "-i",
                media_url,
                "-frames:v",
                "1",
                "-y",
                str(out_path),
            ],
            capture_output=True,
            check=True,
            timeout=90,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return out_path.exists() and out_path.stat().st_size > 0


def _is_phantom(text: str) -> bool:
    """Match ONLY the channel owner's exact handle (edit-distance <=1 for OCR slips) —
    NOT a bare 'phantom' substring, so an opponent/copycat like 'PhantomKing' or
    'PhantomCatan2' on the same leaderboard can never be read as ThePhantom.
    """
    c = re.sub(r"[^a-z]", "", text.lower())
    if not c:
        return False
    if c == CANONICAL_HANDLE:
        return True
    # edit-distance <= 1 (SequenceMatcher ratio is a cheap proxy; guard the length band)
    if abs(len(c) - len(CANONICAL_HANDLE)) <= 1 and c.endswith("phantom"):
        return difflib.SequenceMatcher(None, c, CANONICAL_HANDLE).ratio() >= 0.9
    return False


def _has_1v1_cue(toks: list[dict[str, Any]]) -> bool:
    """A Colonist 1v1 leaderboard always shows a '1v1' mode token (easyocr reads it as
    '1v1'/'Iv1'/'lv1'). Require it as a cheap gate that the frame is a Catan leaderboard.
    NOTE: this checks 1v1-context PRESENCE, not active-mode. ThePhantom's channel is
    exclusively 1v1, so a 4-player leaderboard appearing in a 1v1 video is effectively
    impossible; the residual (4p-active with a 1v1 button visible) is accepted as
    negligible for this channel and is covered by the hand-check validation set.
    """
    for t in toks:
        low = t["text"].lower()
        if "divi" in low:  # 'division' column header contains 'ivi' — exclude
            continue
        if re.search(r"[1il]v[1il]", re.sub(r"\s+", "", low)):
            return True
    return False


def _tab_blues(toks: list[dict[str, Any]], img: Any) -> dict[str, float]:
    """Mean blue-channel value of each detected tab-bar label patch."""
    out: dict[str, float] = {}
    for t in toks:
        lab = t["text"].strip().lower()
        if lab in TAB_LABELS:
            xs = [p[0] for p in t["bbox"]]
            ys = [p[1] for p in t["bbox"]]
            cx = (min(xs) + max(xs)) / 2
            patch = img[max(0, int(min(ys))) : int(max(ys)), max(0, int(cx - 25)) : int(cx + 25)]
            if patch.size:
                out[lab] = float(patch.reshape(-1, 3)[:, 0].mean())
    return out


def _is_global_active(tab_blues: dict[str, float]) -> bool:
    """Only the GLOBAL leaderboard is a world rank. The active tab is the highlighted
    one = clearly lowest blue-channel; require Global present, >=1 other tab visible,
    and Global the distinct minimum. A regional tab (Oceania/Australia/...) as the
    minimum, or an un-confirmable tab bar => reject (never mislabel a regional rank).
    """
    if "global" not in tab_blues:
        return False
    others = [v for k, v in tab_blues.items() if k != "global"]
    if len(others) < 1:
        return False
    return tab_blues["global"] < min(others) - 5


def read_leaderboard_rank(frame_path: Path, reader: Any) -> int | None:
    """If frame is the GLOBAL 1v1 leaderboard AND ThePhantom's row is readable, return
    his world rank N. Layout-agnostic (works across both Colonist UI eras): takes the
    leftmost '#N' token on his row. Returns None unless the Global tab is confirmed
    active (so regional-tab ranks can never be mistaken for a world rank).
    """
    try:
        res = reader.readtext(str(frame_path))
    except Exception:
        return None

    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    height, width = img.shape[:2]

    toks = []
    for bbox, text, conf in res:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        toks.append(
            {
                "text": text,
                "conf": float(conf),
                "bbox": bbox,
                "y": (min(ys) + max(ys)) / 2.0,
                "x0": min(xs),
                "fx0": min(xs) / width,
                "fy": ((min(ys) + max(ys)) / 2.0) / height,
            }
        )
    lows = [t["text"].lower() for t in toks]
    # leaderboard header cues: need >=3 of these column/tab labels present
    cues = ["rank", "rating", "division", "win", "games", "player", "leaderboard"]
    if sum(any(cue in lo for lo in lows) for cue in cues) < 3:
        return None

    # GLOBAL-tab gate: a regional or unconfirmable tab is not a world rank.
    if not _is_global_active(_tab_blues(toks, img)):
        return None
    # 1v1-context gate: never read a rank off a non-1v1 leaderboard.
    if not _has_1v1_cue(toks):
        return None

    return _phantom_rank_from_toks(toks)


def _phantom_rank_from_toks(toks: list[dict[str, Any]]) -> int | None:
    """Given OCR tokens of a confirmed Global 1v1 leaderboard frame, return ThePhantom's
    world rank, or None. Pure (unit-testable): (1) exactly one leaderboard-BODY row
    matches his handle (fy>0.13 drops the profile name; >1 -> ambiguous -> None);
    (2) the rank is the leftmost same-row token that is a literal '#N' with conf above
    the floor; (3) 3-digit reads demand higher confidence (digit-clip guard).
    """
    phantom_toks = [
        t for t in toks if _is_phantom(t["text"]) and 0.20 <= t["fx0"] <= 0.60 and t["fy"] > 0.13
    ]
    if len(phantom_toks) != 1:
        return None  # 0 -> not found; >1 -> ambiguous, refuse to guess (conservative)
    pt = phantom_toks[0]

    cands = []
    for t in toks:
        same_row = abs(t["y"] - pt["y"]) <= 18
        left_of_name = t["x0"] < pt["x0"]
        in_rank_col = 0.10 <= t["fx0"] <= 0.48  # sanity band; the '#' sigil is the real gate
        if same_row and left_of_name and in_rank_col and t["conf"] >= RANK_CONF_MIN:
            # REQUIRE the literal '#': Colonist renders the rank as '#N'; a bare number
            # left of the name is a rating/games fragment, not a rank. This kills the
            # false-high paths (stray small number, digit-clipped rating) at the source.
            m = re.fullmatch(r"#\s*(\d{1,3})", t["text"].strip())
            if m:
                cands.append((t["x0"], int(m.group(1)), t["conf"]))
    if not cands:
        return None
    cands.sort()  # leftmost '#' token in the row = the rank column
    _, n, conf_n = cands[0]
    # 3-digit ranks are where OCR digit-clipping could flip an excluded rank to high;
    # demand higher confidence there (ThePhantom's real ranks are almost always <100).
    if n >= 100 and conf_n < 0.6:
        return None
    if 1 <= n <= 999:
        return n
    return None


def classify_ranked(video_id: str, reader: Any, keep_frame: Path) -> dict[str, Any]:
    """Scan frames for a readable leaderboard rank. Returns a partial manifest row."""
    sd = _stream_and_duration(video_id)
    if sd is None:
        return {
            "strength": "unknown",
            "source": "none",
            "evidence": {"reason": "no stream/duration"},
        }
    media_url, dur = sd

    times: list[tuple[str, float]] = [("end", dur - off) for off in END_OFFSETS]
    times += [("mid", dur * f) for f in MID_FRACTIONS]
    times += [("start", t) for t in START_TIMES]

    grabbed_any = False
    consecutive_fail = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "f.png"
        for where, t in times:
            if t < 0 or t > dur:
                continue
            if not _grab_frame(media_url, t, tmp):
                consecutive_fail += 1
                # a signed media URL can expire mid-run; re-resolve once and retry.
                if consecutive_fail == 2:
                    sd2 = _stream_and_duration(video_id)
                    if sd2 is not None:
                        media_url, dur = sd2
                    if not _grab_frame(media_url, t, tmp):
                        continue
                else:
                    continue
            consecutive_fail = 0
            grabbed_any = True
            n = read_leaderboard_rank(tmp, reader)
            if n is not None:
                FRAMES_DIR.mkdir(parents=True, exist_ok=True)
                keep_frame.write_bytes(tmp.read_bytes())
                strength = "high" if n <= RANK_HIGH_MAX else "excluded"
                return {
                    "strength": strength,
                    "source": "ranked_rank",
                    "evidence": {
                        "rank": n,
                        "frame_s": round(t, 1),
                        "where": where,
                        "frame": keep_frame.name,
                    },
                }
    reason = (
        "no leaderboard frame with readable rank"
        if grabbed_any
        else "frames unreadable (stream/seek failure)"
    )
    return {"strength": "unknown", "source": "none", "evidence": {"reason": reason}}


def load_manifest(rebuild: bool) -> dict[str, Any]:
    if MANIFEST.exists() and not rebuild:
        return cast("dict[str, Any]", json.loads(MANIFEST.read_text()))
    return {
        "schema_version": 1,
        "generated_from": CHANNEL,
        "rank_high_max": RANK_HIGH_MAX,
        "videos": [],
    }


def save_manifest(m: dict[str, Any]) -> None:
    """Atomic write so a crash mid-save can't corrupt the resume file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    os.replace(tmp, MANIFEST)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample",
        type=str,
        default=None,
        help="comma-separated video ids to process (else whole channel)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args.rebuild)
    done = {v["video_id"] for v in manifest["videos"]}

    if args.sample:
        # sample ids may not be in the flat list order; fetch titles for them
        all_titles = dict(list_videos())
        ids = [s.strip() for s in args.sample.split(",") if s.strip()]
        videos = [(i, all_titles.get(i, "")) for i in ids]
    else:
        videos = list_videos()
        if args.limit:
            videos = videos[: args.limit]

    print(
        f"channel videos to consider: {len(videos)} "
        f"(already done: {len(done & {v for v, _ in videos})})",
        flush=True,
    )

    reader = None  # lazy: only load easyocr if a ranked video needs OCR

    for idx, (vid, title) in enumerate(videos, 1):
        if vid in done:
            continue
        tmatch = TOURNAMENT_RE.search(title)
        if is_tournament(title) and tmatch is not None:
            row = {
                "video_id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "title": title,
                "strength": "high",
                "source": "tournament",
                "evidence": {"keyword": tmatch.group(0)},
            }
            print(f"[{idx}/{len(videos)}] TOURNAMENT high  :: {title}", flush=True)
        else:
            if reader is None:
                import easyocr

                print("loading easyocr reader (CPU) ...", flush=True)
                reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            part = classify_ranked(vid, reader, FRAMES_DIR / f"{vid}_rank.png")
            row = {
                "video_id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "title": title,
                **part,
            }
            ev = row["evidence"]
            tag = f"rank #{ev.get('rank')}" if "rank" in ev else ev.get("reason", "")
            print(
                f"[{idx}/{len(videos)}] {row['strength'].upper():8s} {tag}  :: {title}", flush=True
            )

        manifest["videos"].append(row)
        save_manifest(manifest)  # incremental: crash-safe

    # summary
    counts: dict[str, int] = {}
    for v in manifest["videos"]:
        counts[v["strength"]] = counts.get(v["strength"], 0) + 1
    print(f"\nDONE. total={len(manifest['videos'])} counts={counts}", flush=True)
    print(f"manifest: {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
