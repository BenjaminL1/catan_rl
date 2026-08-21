"""Atomic JSONL persistence for labeling rows (plan §D).

The labeling tool's durable artefact is `scenarios.jsonl`. Every row is
a single line of JSON; the file is append-only; recovery from a
mid-write crash is handled by `repair_jsonl` truncating any malformed
trailing line. The on-disk file is **never rewritten** by migrations —
schema_version is read per-row and missing fields populated with
defaults at read time.

Atomicity guarantee: on POSIX, `write(2)` to an `O_APPEND` fd is atomic
for payloads ≤ `PIPE_BUF` (typically 4096 bytes). JSONL rows for the
labeling schema fit comfortably under this limit (~250-500 bytes per
row). For paranoia, `repair_jsonl()` runs on every session start.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from catan_rl.env.ruleset import RULESET_R0

SCHEMA_VERSION: int = 3
"""Current JSONL schema version. Bump on backward-incompatible changes.

v2 (spec ``setup-labeling-and-champion-finetune`` D3) adds two OPTIONAL
provenance fields — ``source`` (``"tool"`` | ``"game"``) and ``ruleset`` —
populated at READ time for v1 rows by :func:`_migrate_row`.
``_REQUIRED_FIELDS`` is unchanged, so a v1 row written by the labeling tool
keeps loading untouched and the file on disk is never rewritten.

v3 (spec ``setup-scorer-and-blind-reveal`` D0 + D3) adds seven more OPTIONAL
fields — ``replay_of`` (the self-consistency link), the five blind-then-reveal
fields (all defaulting to ``None``), and ``pick_clarity`` (the owner's
"clear best" / "close call" tag, defaulting to :data:`PICK_CLARITY_CLOSE`).
``_REQUIRED_FIELDS`` is again unchanged and the file is again never rewritten,
so every v1/v2 row on disk keeps loading byte-for-byte as it was written."""

_V2_DEFAULTS: dict[str, Any] = {"source": "tool", "ruleset": RULESET_R0}
"""Read-time defaults for the fields v2 added.

``source="tool"`` is the correct value for every v1 row: the records→labels
adapter did not exist when they were written, so the labeling UI was the only
writer. The ruleset default is :data:`~catan_rl.env.ruleset.RULESET_R0`
IMPORTED, not the literal ``"R0"`` — it must mean the same epoch
``ScenarioGenerator`` pins (it builds every mask with ``RULESET_R0``), and a
literal would keep saying "R0" if that constant were ever renamed."""

PICK_CLARITY_CLEAR: str = "clear"
PICK_CLARITY_CLOSE: str = "close"
PICK_CLARITIES: tuple[str, ...] = (PICK_CLARITY_CLEAR, PICK_CLARITY_CLOSE)
"""D3's two submit keys: was this the obvious best vertex, or a close call?

The tag is the OWNER's, not the scorer's, so it is written in BOTH D3 arms —
a ``--no-reveal`` row carries ``pick_clarity`` even though it carries none of
the five scorer fields. D4's strictness bar (top-1 must match on ``clear``
picks) is read on the control arm too, and a tag that only existed in the
reveal arm would put the bar and the anchoring control in conflict."""

PICK_CLARITY_FIELD: str = "pick_clarity"

_V3_DEFAULTS: dict[str, Any] = {
    "replay_of": None,
    "scorer_version": None,
    "scorer_top1": None,
    "scorer_rank_of_pick": None,
    "agree": None,
    "reveal_mode": None,
    PICK_CLARITY_FIELD: PICK_CLARITY_CLOSE,
}
"""Read-time defaults for the fields v3 added — ``None`` for all but
``pick_clarity``, whose legacy reading the spec fixes as ``close`` (below).

``None`` is load-bearing, not a stylistic choice. The five reveal fields mean
"this pick was never graded" on a pre-v3 row and on a ``--no-reveal`` row, and
that is NOT the same statement as ``agree=False`` — defaulting ``agree`` to
``False`` would inject a phantom disagreement for every one of the pre-existing
labels into the D4 gate. Likewise ``replay_of=None`` means "an original pick",
which is what the fit's replay exclusion tests for; ``False`` or ``""`` would
be a second falsy spelling of the same thing and invite an ``is not None``
check to drift into a truthiness check.

``pick_clarity`` is the ONE v3 default that is not ``None``, because the spec
fixes its legacy reading explicitly: "untagged legacy rows read as ``close``".
That is the conservative direction — ``close`` picks are graded by top-3
containment, so a legacy row can never be pulled into the >=70% top-1
strictness bar on the strength of a tag its labeler never gave.
"""

REVEAL_MODE_REVEAL: str = "reveal"
REVEAL_MODE_NO_REVEAL: str = "no_reveal"
"""The two D3 session modes. ``no_reveal`` is the anchoring CONTROL arm: the
owner labels with no scorer overlay at all, and those rows carry none of the
five reveal fields (the session manifest still records the mode and the live
scorer version, so the gate can find the arm by ``session_id`` join)."""

REVEAL_MODES: tuple[str, ...] = (REVEAL_MODE_REVEAL, REVEAL_MODE_NO_REVEAL)

SCORER_ROW_FIELDS: tuple[str, ...] = (
    "scorer_version",
    "scorer_top1",
    "scorer_rank_of_pick",
    "agree",
    "reveal_mode",
)
"""The five D3 reveal fields, named once so the writer, the reader and the
"``--no-reveal`` rows carry none of them" invariant test cannot drift apart."""


_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "scenario_id",
    "session_id",
    "labeled_at",
    "labeler_id",
    "game_seed",
    "draft_position",
    "acting_player",
    "prior_picks",
    "settlement_vertex",
    "road_edge",
)
"""Fields every row must carry to be appendable.

``archetype`` was removed from this list when the strategy-archetype categories
were dropped from the labeling flow. Rows already on disk still carry the key and
keep loading untouched — the file is never rewritten (see the module docstring),
and :func:`load_scenarios` passes unknown keys through verbatim."""


def append_scenario(scenario: dict[str, Any], path: Path) -> None:
    """Append a single scenario as one JSONL line.

    Atomicity: serialised to bytes, then a single `write()` to an
    `O_APPEND` fd. The OS guarantees this is atomic for sub-PIPE_BUF
    payloads. Parent directories are created if missing.

    Raises:
        ValueError: if a required field is missing.
        TypeError: if the row contains a non-JSON-serialisable value.
    """
    for field in _REQUIRED_FIELDS:
        if field not in scenario:
            raise ValueError(f"scenario row missing required field: {field!r}")

    # Validate JSON-serialisability before opening the file.
    line = json.dumps(scenario, separators=(",", ":"), ensure_ascii=False) + "\n"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = line.encode("utf-8")
    if len(payload) > 4000:  # PIPE_BUF safety margin
        # Fall back to a temp-file + os.replace pattern. Slower but safe.
        _atomic_append_via_rename(payload, path)
    else:
        # Single atomic write to an O_APPEND fd.
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)


def _atomic_append_via_rename(payload: bytes, path: Path) -> None:
    """Fallback for oversized rows: read-all + write-via-tempfile."""
    existing = path.read_bytes() if path.exists() else b""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(existing + payload)
    os.replace(tmp, path)


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Read every row. Returns [] for missing or empty file.

    Future-proofing: a `schema_version` field on every row lets the
    loader populate defaults for fields added in later schemas without
    rewriting the file.
    """
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(_migrate_row(row))
    return out


def _migrate_row(row: dict[str, Any]) -> dict[str, Any]:
    """In-memory migration of older schema versions to the current.

    v1 → v2 fills the OPTIONAL provenance fields from :data:`_V2_DEFAULTS`;
    v2 → v3 fills the replay/reveal fields from :data:`_V3_DEFAULTS` (``None``
    for all but ``pick_clarity``, which reads as ``close``).
    The returned dict keeps its original ``schema_version`` — the value records
    what was WRITTEN, and the file on disk is never rewritten (see the module
    docstring); only the in-memory view is completed.

    Defaults are applied to any row missing them, not just to rows stamped v1,
    so a hand-edited or partially-written row cannot hand a consumer a
    ``KeyError`` on a field the schema calls optional.
    """
    for key, default in _V2_DEFAULTS.items():
        row.setdefault(key, default)
    for key, default in _V3_DEFAULTS.items():
        row.setdefault(key, default)
    return row


def count_scenarios(path: Path) -> int:
    """Fast row-count without JSON parsing."""
    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def repair_jsonl(path: Path) -> int:
    """Remove a malformed trailing line (crash recovery).

    Returns the number of bytes truncated. No-op if the file is missing
    or every line parses.

    Strategy: walk the file from the end, find the last newline. If
    everything after it is non-empty, attempt to parse it; if it doesn't
    parse, truncate the file at that newline.
    """
    path = Path(path)
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data:
        return 0
    last_newline = data.rfind(b"\n")
    if last_newline == -1:
        # No newline at all → the entire content is a partial line.
        path.write_bytes(b"")
        return len(data)
    trailing = data[last_newline + 1 :].strip()
    if not trailing:
        return 0
    # There IS content after the last newline. Try to parse it.
    try:
        json.loads(trailing)
        # Parses cleanly but is missing a terminating newline; we treat
        # this as recoverable rather than truncated (the next write will
        # append correctly because we always write "<row>\n").
        return 0
    except json.JSONDecodeError:
        truncate_to = last_newline + 1
        with path.open("rb+") as f:
            f.truncate(truncate_to)
        return len(data) - truncate_to


def held_out_split(
    labels: list[dict[str, Any]], *, frac: float = 0.2, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split label rows into ``(train, held_out)`` BY ``game_seed``.

    Splitting by seed, not by row, is what stops a draft's position-4 label
    landing in the held-out set while the position-1 label it is conditioned on
    was trained on — which would make the D7 gate-1 agreement number optimistic.

    This lives HERE, next to the label store, because it has exactly one
    definition and two consumers that must never disagree: the converter
    (:func:`catan_rl.labeling.to_shard.convert`, which EXCLUDES the held-out
    seeds from the shard) and the gate CLI (``scripts/eval_setup_agreement.py``,
    which measures on them). A second copy of this arithmetic anywhere is how a
    gate silently starts reporting memorisation.
    """
    import numpy as np

    seeds = sorted({int(row["game_seed"]) for row in labels})
    rng = np.random.default_rng(seed)
    rng.shuffle(seeds)
    n_held = max(1, round(len(seeds) * frac)) if seeds else 0
    held_seeds = set(seeds[:n_held])
    held = [r for r in labels if int(r["game_seed"]) in held_seeds]
    train = [r for r in labels if int(r["game_seed"]) not in held_seeds]
    return train, held
