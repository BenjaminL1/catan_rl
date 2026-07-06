#!/usr/bin/env python3
"""TIER-5 integration driver: run the e2e harvest on a few REAL videos + capture
per-game consensus-supply telemetry the standard run does not persist.

This is a thin wrapper around :func:`catan_rl.human_data.harvest.run_harvest` used
only for the TIER-5 integration slice (``docs/plans/tier5_report.md``). It adds ONE
thing the production harvest deliberately does not: per-game, per-player
**grant-frame supply** counters, so the report can state the actual number of
readable grant frames each granting player contributed to the multi-frame CONSENSUS
read (brief §5.2 — the anchor demands ≥2 frames agree byte-identical).

It captures these by two *thin* monkeypatch wrappers (no body duplication, so they
cannot drift from the code they instrument):

- ``harvest._read_game_inputs`` → tags the current ``(video_id, segment_index)`` so
  the grant counters below are grouped per game.
- ``harvest._consensus_grant`` → records ``len(grant_frames)`` (candidate grant-line
  frames routed to this game for this player) and whether the consensus read
  succeeded (returned a multiset vs ``None``).
- ``harvest.consensus_granted_glyphs`` → records the exact number of *readable*
  frames (``len(frames_boxes)``) that reached the consensus vote — this is only
  called when ≥2 frames are readable, so a success here is a direct witness that the
  ≥2-readable-grant-frames consensus-supply requirement was met.

Everything else — the download-then-delete ingest, the OCR/CV chain, the glyph HARD
GATE, the resumable ledger — is the exact production path. CPU-only; no ``gui/``.

Usage:
    PYTHONPATH=src python3 scripts/tier5_harvest.py OUT_DIR VIDEO_ID [VIDEO_ID ...] \
        [--work-dir DIR] [--max-workers N] [--net-concurrency N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from catan_rl.human_data import harvest
from catan_rl.human_data.glyph_anchor import load_glyph_validation

MANIFEST = REPO / "data" / "human" / "strength_manifest.json"
GLYPH_VALIDATION = REPO / "data" / "human" / "glyph_validation.json"

# --- shared capture state (serial run: max_workers=1) ------------------------
_CTX: dict[str, Any] = {"video_id": None, "segment_index": None, "handle": None}
#: per (video_id, segment_index, handle) -> {"candidate": int, "readable": int|None, "ok": bool}
_GRANT_SUPPLY: dict[tuple[str, int, str], dict[str, Any]] = {}
#: per (video_id, segment_index, handle) wall-clock is not tracked; per-video is.
_VIDEO_WALL: dict[str, float] = {}


def _install_instrumentation() -> None:
    orig_read = harvest._read_game_inputs
    orig_consensus_grant = harvest._consensus_grant
    orig_consensus_glyphs = harvest.consensus_granted_glyphs

    def read_game_inputs(video_id: str, segment_index: int, *args: Any, **kwargs: Any) -> Any:
        _CTX["video_id"] = video_id
        _CTX["segment_index"] = segment_index
        return orig_read(video_id, segment_index, *args, **kwargs)

    def consensus_grant(handle: str, grant_frames: Any) -> Any:
        _CTX["handle"] = handle
        key = (str(_CTX["video_id"]), int(_CTX["segment_index"]), handle)
        entry = _GRANT_SUPPLY.setdefault(key, {"candidate": 0, "readable": None, "ok": False})
        entry["candidate"] = len(grant_frames)
        result = orig_consensus_grant(handle, grant_frames)
        entry["ok"] = result is not None
        return result

    def consensus_glyphs(frames_boxes: Any) -> Any:
        key = (
            str(_CTX["video_id"]),
            int(_CTX["segment_index"]),
            str(_CTX["handle"]),
        )
        entry = _GRANT_SUPPLY.setdefault(key, {"candidate": 0, "readable": None, "ok": False})
        entry["readable"] = len(frames_boxes)
        return orig_consensus_glyphs(frames_boxes)

    harvest._read_game_inputs = read_game_inputs
    harvest._consensus_grant = consensus_grant
    harvest.consensus_granted_glyphs = consensus_glyphs  # type: ignore[assignment]


def _grant_supply_report() -> dict[str, Any]:
    """Aggregate the captured per-player grant-frame supply into a JSON payload."""
    per_game: dict[str, dict[str, Any]] = {}
    for (video_id, seg_idx, handle), entry in _GRANT_SUPPLY.items():
        gk = f"{video_id}:{seg_idx}"
        per_game.setdefault(gk, {})[handle] = dict(entry)
    # consensus-supply headline: of players who reached the grant read, how many had
    # >=2 readable frames (the consensus requirement).
    reached = [e for e in _GRANT_SUPPLY.values() if e["readable"] is not None or e["candidate"] > 0]
    ok_players = [e for e in _GRANT_SUPPLY.values() if e["ok"]]
    readable_ge2 = [e for e in ok_players if (e["readable"] or 0) >= 2]
    readable_counts = Counter(
        (e["readable"] if e["readable"] is not None else -1) for e in _GRANT_SUPPLY.values()
    )
    return {
        "per_game": per_game,
        "players_reached_grant_read": len(reached),
        "players_consensus_ok": len(ok_players),
        "players_ok_with_ge2_readable": len(readable_ge2),
        "readable_frame_count_histogram": {
            (str(k) if k >= 0 else "unknown(<2)"): v for k, v in sorted(readable_counts.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="destination for corpus/rejected/telemetry")
    parser.add_argument("video_ids", nargs="+", help="explicit video ids to harvest")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--net-concurrency", type=int, default=1)
    args = parser.parse_args(argv)

    validation = load_glyph_validation(GLYPH_VALIDATION)
    if validation is None:
        print("[tier5] BLOCKED — glyph validation missing/failed; harvest gate will refuse")
        return 2

    _install_instrumentation()
    print(f"[tier5] harvesting {len(args.video_ids)} videos -> {args.out_dir}", flush=True)
    t0 = time.time()
    result, telemetry = harvest.run_harvest(
        manifest_path=MANIFEST,
        out_dir=args.out_dir,
        video_ids=args.video_ids,
        max_workers=args.max_workers,
        net_concurrency=args.net_concurrency,
        work_dir=args.work_dir,
        glyph_validation=validation,
    )
    wall = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "telemetry.json").write_text(
        json.dumps(telemetry.to_dict(), indent=1, sort_keys=True), encoding="utf-8"
    )
    supply = _grant_supply_report()
    supply["wall_clock_s_total"] = wall
    supply["wall_clock_s_per_video"] = wall / max(1, len(args.video_ids))
    (out_dir / "grant_supply.json").write_text(json.dumps(supply, indent=2), encoding="utf-8")

    print(telemetry.render(), flush=True)
    print(
        f"[tier5] wall={wall:.0f}s ({wall / len(args.video_ids):.0f}s/video); "
        f"consensus-ok players={supply['players_consensus_ok']} "
        f"(>=2 readable: {supply['players_ok_with_ge2_readable']})",
        flush=True,
    )
    print(f"[tier5] wrote telemetry.json + grant_supply.json to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
