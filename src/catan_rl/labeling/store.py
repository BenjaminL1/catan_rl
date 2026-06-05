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

SCHEMA_VERSION: int = 1
"""Current JSONL schema version. Bump on backward-incompatible changes."""

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
    "archetype",
    "settlement_vertex",
    "road_edge",
)


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

    v1 is the current schema; no migrations active. Adding fields in a
    future schema_version=2 would patch defaults here.
    """
    version = row.get("schema_version", 1)
    if version == SCHEMA_VERSION:
        return row
    # Placeholder: future migrations go here.
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
