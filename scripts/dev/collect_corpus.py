#!/usr/bin/env python3
"""Collect every ACCEPTED game record into the provisional human-opening corpus.

Reads ``data/human/vlm_spike/records/*.json`` (written by ``vlm_spike.py localize``)
and appends each ``accepted: true`` record to
``data/human/corpus/provisional_openings.jsonl``.

PROVISIONAL / SEED-ELIGIBLE ONLY. Every real-video record carries
``placement_order_established: False``: the openings ARE correctly ordered (the glyph
anchor pins each player's granting/2nd settlement — an independent check that held on
every hand-verified game), but the record contract ALSO demands LOG corroboration of the
order, and the log's re-OCR duplication makes that unavailable on real footage. Whether
glyph-anchor-only ordering suffices for SCOREBOARD eligibility is a PENDING USER
DECISION — this collector does not presume it, and does not relax the flag.

Idempotent: de-dups on ``(video_id, game_index)``, writes in stable order, so it can be
re-run after each new localization without duplicating rows.

Usage::

    PYTHONPATH=src python3 scripts/dev/collect_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from catan_rl.human_data.record import GameRecord

RECORDS = REPO / "data/human/vlm_spike/records"
OUT = REPO / "data/human/corpus/provisional_openings.jsonl"


def main() -> None:
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
    for r in rows:
        opens = {h: (o.settlements, o.roads) for h, o in r.openings.items()}
        print(f"  {r.video_id} g{r.game_index}  winner={r.winner}  {opens}")
    n_rejected = sum(
        1
        for p in RECORDS.glob("*.json")
        if not json.loads(p.read_text(encoding="utf-8")).get("accepted")
    )
    print(f"(rejected records on disk, not collected: {n_rejected})")
    print("ALL rows are PROVISIONAL + seed-eligible (placement_order_established=False).")


if __name__ == "__main__":
    main()
