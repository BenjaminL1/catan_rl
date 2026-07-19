#!/usr/bin/env python3
"""Collect every ACCEPTED game record into the provisional human-opening corpus.

Reads ``data/human/vlm_spike/records/*.json`` (written by ``vlm_spike.py localize``)
and appends each ``accepted: true`` record to
``data/human/corpus/provisional_openings.jsonl``.

PROVISIONAL / SEED-ELIGIBLE by DEFAULT. Every real-video record carries
``placement_order_established: False`` under the DEFAULT two-signal regime: the openings
ARE correctly ordered (the glyph anchor pins each player's granting/2nd settlement — an
independent check that held on every hand-verified game), but that regime ALSO demands
LOG corroboration of the order, and the log's re-OCR duplication makes it unavailable on
real footage.

GLYPH-ANCHOR-ONLY ordering (audit Decision 1) is the OPT-IN: with
``--no-require-log-ordinal`` this collector re-localizes every prepared game with
``require_log_ordinal=False`` (glyph anchor alone keeps order established; grant
collisions/ambiguity still fail closed), rewrites its ``records/<game>.json``, then
collects — promoting the winner-bearing games to SCOREBOARD eligibility. The DEFAULT
(``--require-log-ordinal``) is byte-identical to the pre-opt-in behaviour: it does NOT
re-localize and does NOT relax the flag.

Idempotent: de-dups on ``(video_id, game_index)``, writes in stable order, so it can be
re-run after each new localization without duplicating rows.

Usage::

    # DEFAULT (two-signal, seed-only — byte-identical to the pre-opt-in collector):
    PYTHONPATH=src python3 scripts/dev/collect_corpus.py

    # OPT-IN (glyph-anchor-only ordering — the audit-Decision-1 recompute the owner
    # runs post-merge in the MAIN checkout; expected ~15 winner-bearing eligible games):
    PYTHONPATH=src python3 scripts/dev/collect_corpus.py --no-require-log-ordinal
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from catan_rl.human_data.record import GameRecord

FRAMES = REPO / "data/human/vlm_spike/frames"
RECORDS = REPO / "data/human/vlm_spike/records"
OUT = REPO / "data/human/corpus/provisional_openings.jsonl"


def _load_vlm_spike() -> Any:
    """Import ``scripts/vlm_spike.py`` as a module (it is a script, not a package).

    Mirrors the ``tests/unit/human_data/test_vlm_spike.py`` importlib shim so the
    recompute path reuses the SAME ``localize_game`` the ``vlm_spike.py localize`` CLI
    runs — the order-establishing logic lives there, never re-implemented here.
    """
    spec = importlib.util.spec_from_file_location(
        "vlm_spike_mod", REPO / "scripts" / "vlm_spike.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so slots=True dataclasses can resolve their own module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _relocalize_with_glyph_only() -> None:
    """Re-run localize for every prepared game with ``require_log_ordinal=False`` and
    rewrite its ``records/<game>.json`` in the ``vlm_spike.py localize`` payload shape.

    This is the audit-Decision-1 glyph-anchor-only recompute: it opts the placement-order
    flag in for games whose grant anchor established order but whose LOG ordinal is
    unreadable (re-OCR duplication). Only games with a prepared ``meta.json`` are
    re-localized; the fail-closed grant-collision/ambiguous cases stay unestablished.
    """
    vlm = _load_vlm_spike()
    RECORDS.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped = 0
    for meta_path in sorted(FRAMES.glob("*/meta.json")):
        game_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("board_hexes"):
            # Prepared but board-unreadable — localize_game cannot even build the
            # board; skip instead of aborting the whole recompute. (A missing VLM
            # localization is already a graceful typed reject inside localize_game.)
            print(f"[recompute] skip {game_dir.name}: board unreadable")
            skipped += 1
            continue
        result = vlm.localize_game(game_dir, require_log_ordinal=False)
        payload: dict[str, Any] = {
            "game": game_dir.name,
            "accepted": result.accepted,
            "rejection_reason": result.rejection_reason,
            "record": result.record.to_dict(),
        }
        (RECORDS / f"{game_dir.name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        n += 1
    print(
        f"=== GLYPH-ANCHOR-ONLY RECOMPUTE (require_log_ordinal=False) "
        f"-> re-localized {n} game(s), skipped {skipped} non-localizable ==="
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-log-ordinal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "require the LOG setup-event ordinal to keep placement order established "
            "(DEFAULT — byte-identical to the pre-opt-in collector; does NOT re-localize). "
            "--no-require-log-ordinal opts into glyph-anchor-only ordering (audit "
            "Decision 1): re-localize every prepared game with require_log_ordinal=False "
            "(grant anchor alone keeps order established) before collecting."
        ),
    )
    args = parser.parse_args(argv)

    if not args.require_log_ordinal:
        _relocalize_with_glyph_only()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, int], GameRecord] = {}

    for path in sorted(RECORDS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("accepted"):
            continue
        rec = GameRecord.from_dict(payload["record"])
        by_key[(rec.video_id, rec.game_index)] = rec

    rows = [by_key[k] for k in sorted(by_key)]
    OUT.write_text(
        "".join(r.to_json_line() + "\n" for r in rows),
        encoding="utf-8",
    )

    print(f"=== PROVISIONAL HUMAN-OPENING CORPUS -> {OUT} ===")
    print(f"accepted games: {len(rows)}")
    n_scoreboard = sum(1 for r in rows if r.is_scoreboard_eligible())
    for r in rows:
        opens = {h: (o.settlements, o.roads) for h, o in r.openings.items()}
        src = r.provenance.get("order_source")
        eligible = "SCOREBOARD" if r.is_scoreboard_eligible() else "seed-only"
        print(
            f"  {r.video_id} g{r.game_index}  winner={r.winner}  "
            f"order_source={src}  {eligible}  {opens}"
        )
    n_rejected = sum(
        1
        for p in RECORDS.glob("*.json")
        if not json.loads(p.read_text(encoding="utf-8")).get("accepted")
    )
    print(f"(rejected records on disk, not collected: {n_rejected})")
    if args.require_log_ordinal:
        print(
            "ALL rows are PROVISIONAL + seed-eligible "
            "(placement_order_established=False; run --no-require-log-ordinal to opt in)."
        )
    else:
        print(
            f"GLYPH-ANCHOR-ONLY ordering (audit Decision 1): "
            f"{n_scoreboard} of {len(rows)} accepted games are SCOREBOARD-eligible."
        )


if __name__ == "__main__":
    main()
