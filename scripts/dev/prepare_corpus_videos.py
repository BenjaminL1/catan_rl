#!/usr/bin/env python3
"""Prepare frames for N high-tier videos, sequentially, for the corpus build.

ONE ingest per video serves BOTH deliverables:
  * the YIELD numbers (games seen / clean opening / both grants readable), and
  * the persisted ``post_setup.png`` + ``meta.json`` the VLM localization step needs.

Running the yield sweep and ``prepare-frames`` separately would ingest every video
TWICE (~50 min each, OCR-bound) — this does it once.

A game is LOCALIZABLE iff the fail-closed router produced a clean 8-pieces-down frame
AND both players' grant glyphs were read (the glyph anchor is the only joint-flip
defence, so a game missing either grant is rejected no matter how good the frame is).

Writes a per-video tally to ``runs/human/corpus_prep.json`` after EACH video, so an
interrupted overnight run loses nothing. Skips videos already prepared.

Usage::

    PYTHONPATH=src python3 scripts/dev/prepare_corpus_videos.py --videos 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MANIFEST = REPO / "data/human/strength_manifest.json"
FRAMES = REPO / "data/human/vlm_spike/frames"
OUT = REPO / "runs/human/corpus_prep.json"

#: Stop LAUNCHING new videos once free disk drops below this (GB). v11 training has
#: its own 5 GB guard; we halt well above it so a parallel sweep never crowds the run.
_MIN_FREE_GB = 8.0


def high_tier_videos() -> list[str]:
    d = json.loads(MANIFEST.read_text(encoding="utf-8"))
    vids = d if isinstance(d, list) else d.get("videos", d)
    items = vids.items() if isinstance(vids, dict) else [(v.get("video_id"), v) for v in vids]
    return [k for k, v in items if isinstance(v, dict) and v.get("strength") == "high"]


def tally(video: str) -> dict:
    """Per-game tally from the frame dirs prepare-frames persisted for this video.

    A game is LOCALIZABLE only when ALL THREE independent gates pass:
      * the fail-closed router produced a clean 8-pieces-down frame (the dir exists);
      * the board CV produced a cross-frame-stable board read (``board_hexes``) — a
        game can have a perfect opening frame and STILL fail here (``board_unreadable``),
        and the snap has no hex layout to snap to without it;
      * BOTH players' grant glyphs were read — the glyph anchor is the only joint-flip
        defence, so a game missing either grant is rejected however good the frame is.
    Counting grants alone OVERCOUNTS (measured: 9Sm86ml04aI had 3 games with both grants
    but only 1 with a readable board).
    """
    games = []
    for d in sorted(FRAMES.glob(f"{video}__g*")):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        grants = meta.get("granted_resources", {})
        players = list(meta.get("players", {}).values())
        both = len(players) == 2 and all(p in grants for p in players)
        board_ok = bool(meta.get("board_hexes"))
        games.append(
            {
                "game": d.name,
                "winner": meta.get("winner"),
                "board_readable": board_ok,
                "grants_readable": both,
                "localizable": bool(board_ok and both),
                "players": players,
            }
        )
    return {"video": video, "games": games, "n_games": len(games)}


def _free_gb(path: Path = REPO) -> float:
    """Free disk (GB) on the volume holding ``path``."""
    return shutil.disk_usage(str(path)).free / 1e9


def _process_video(video: str, i: int, n: int, env_extra: dict[str, str]) -> dict:
    """Ingest ONE video via the ``prepare-frames`` subprocess and return its tally.

    ``env_extra`` carries the parallel-sweep pins (single-thread OCR + the download
    gate); it is EMPTY for ``--workers 1`` so that path stays byte-identical. Any
    per-video failure (yt-dlp flake, timeout) is caught and returned as an ``error``
    row — never raised — so one bad video cannot kill the sweep. The child's
    ``[ocr-cache]`` / ``[ocr-threads]`` lines are surfaced to the sweep log.
    """
    print(f"[prep] {i}/{n} {video}: ingesting…", flush=True)
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), **env_extra}
    try:
        spike = str(REPO / "scripts/vlm_spike.py")
        cmd = [sys.executable, spike, "prepare-frames", "--video", video]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=5400,
        )
        for line in proc.stdout.splitlines():
            if line.startswith(("[ocr-cache]", "[ocr-threads]")):
                print(f"[prep] {i}/{n} {video} :: {line}", flush=True)
        r = tally(video)
        loc = sum(g["localizable"] for g in r["games"])
        print(f"[prep] {i}/{n} {video}: games={r['n_games']} localizable={loc}", flush=True)
        return r
    except Exception as exc:  # one bad video must not kill the night
        print(f"[prep] {i}/{n} {video}: FAILED {exc}", flush=True)
        traceback.print_exc()
        return {"video": video, "error": f"{type(exc).__name__}: {exc}"}


def _run_sequential(todo: list[str], results: dict[str, dict]) -> None:
    """Today's exact one-at-a-time path (``--workers 1``): no thread pins, no gate."""
    for i, video in enumerate(todo, 1):
        results[video] = _process_video(video, i, len(todo), {})
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _run_parallel(todo: list[str], results: dict[str, dict], workers: int, net_conc: int) -> None:
    """Fan videos across ``workers`` single-thread-pinned OCR subprocesses.

    Downloads are capped at ``net_conc`` via a cross-process file semaphore (each
    worker's ``vlm_spike`` builds it from ``CATAN_DOWNLOAD_LOCK_DIR``). A new video is
    launched only when a slot frees AND free disk is >= ``_MIN_FREE_GB``; otherwise the
    sweep stops launching (in-flight videos finish) and logs loudly.
    """
    lock_dir = Path(tempfile.mkdtemp(prefix="corpus_prep_dl_"))
    env_extra = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CATAN_OCR_THREADS": "1",
        "CATAN_DOWNLOAD_LOCK_DIR": str(lock_dir),
        "CATAN_NET_CONCURRENCY": str(net_conc),
    }
    save_lock = threading.Lock()

    def worker(video: str, i: int) -> None:
        r = _process_video(video, i, len(todo), env_extra)
        with save_lock:
            results[video] = r
            OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    todo_iter = iter(enumerate(todo, 1))
    running: dict[concurrent.futures.Future, str] = {}
    stopped = False
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:

            def fill() -> None:
                nonlocal stopped
                while len(running) < workers and not stopped:
                    try:
                        i, video = next(todo_iter)
                    except StopIteration:
                        return
                    free = _free_gb()
                    if free < _MIN_FREE_GB:
                        print(
                            f"[prep] DISK GUARD: {free:.1f} GB free < {_MIN_FREE_GB} GB — "
                            f"NOT launching more videos (in-flight will finish)",
                            flush=True,
                        )
                        stopped = True
                        return
                    running[pool.submit(worker, video, i)] = video

            fill()
            while running:
                done, _ = concurrent.futures.wait(
                    running, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    running.pop(fut)
                fill()
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel OCR worker processes (1 = today's sequential path, byte-identical)",
    )
    ap.add_argument(
        "--net-concurrency", type=int, default=2, help="max concurrent yt-dlp downloads"
    )
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))

    todo = [v for v in high_tier_videos() if v not in results and v not in args.skip]
    todo = todo[: args.videos]
    print(f"[prep] {len(todo)} videos (workers={args.workers}): {todo}", flush=True)

    if args.workers <= 1:
        _run_sequential(todo, results)
    else:
        _run_parallel(todo, results, args.workers, args.net_concurrency)

    ok = [r for r in results.values() if "error" not in r]
    games = [g for r in ok for g in r["games"]]
    if games and ok:
        loc = sum(g["localizable"] for g in games)
        print("\n=== CORPUS PREP TALLY ===")
        print(f"videos prepared : {len(ok)} (failed {len(results) - len(ok)})")
        print(f"games (clean)   : {len(games)}   games/video: {len(games) / len(ok):.2f}")
        print(f"LOCALIZABLE     : {loc}/{len(games)} = {loc / len(games):.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
