"""Report + reclaim disk from a checkpoint tree (dedup + slim anchors).

Two independent wins over a ``runs/`` tree that predates the league-sidecar
cutover:

1. **Byte-identical duplicates.** Some checkpoints are exact copies of each
   other (e.g. ``v10_cand`` / ``v10_chain`` snapshots banked twice). Grouped by
   a sha256 of their file bytes; ``--execute`` replaces the extra copies with
   hardlinks to a single kept file (a no-op-safe report-only fallback is used
   when the files live on different filesystems, where ``os.link`` can't span).

2. **Fat anchors.** ``<root>/anchors/*.pt`` written before this feature embed
   the ~560 MB league pool and/or optimizer state they never need (an anchor is
   only ever a frozen reference opponent / eval baseline). ``--execute`` re-saves
   each via :func:`catan_rl.checkpoint.bank_anchor` as a ~5.6 MB policy-only
   slim file.

**Safety.** Dry-run is the DEFAULT — it prints a report and mutates NOTHING.
Every path acted on is confined to ``root``; the tool never writes outside it.
This module is unit-tested ONLY against synthetic tmp trees; the owner runs it
against the real ``runs/`` tree by hand, post-merge.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """Return the sha256 hex digest of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _is_under(child: Path, parent: Path) -> bool:
    """True if ``child`` is inside ``parent`` (both resolved)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_anchor(path: Path) -> bool:
    """True if any path component up to ``root`` is a directory named ``anchors``."""
    return "anchors" in path.parts


def _is_fat_checkpoint(path: Path) -> bool:
    """True if the checkpoint embeds league and/or optimizer state (reclaimable).

    Reads the raw torch payload (map_location cpu) and inspects the top-level
    keys — cheaper than a full :func:`load_checkpoint` and tolerant of any
    schema. Returns ``False`` on any load error (leave unknown files alone)."""
    import torch

    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # unreadable → treat as non-fat, don't touch
        return False
    if not isinstance(raw, dict):
        return False
    if raw.get("optimizer_state_dict") is not None:
        return True
    league = raw.get("league")
    return bool(isinstance(league, dict) and league.get("snapshots"))


@dataclass
class ReclaimReport:
    """Structured result of a scan (what dry-run prints, what execute acts on)."""

    root: Path
    total_files: int = 0
    total_bytes: int = 0
    # Each group: list of byte-identical paths (len >= 2). First is the "keep".
    duplicate_groups: list[list[Path]] = field(default_factory=list)
    # Fat anchor files eligible for a slim re-save.
    fat_anchors: list[Path] = field(default_factory=list)
    dup_reclaim_bytes: int = 0

    def estimated_reclaim_bytes(self) -> int:
        """Duplicate-copy bytes reclaimable now (anchor slim savings are only
        known precisely after the re-save, so they're excluded here)."""
        return self.dup_reclaim_bytes


def scan(root: Path, *, inspect_fat: bool = True) -> ReclaimReport:
    """Scan ``root`` for ``*.pt`` files, grouping byte-identical duplicates and
    flagging fat anchors. Pure/read-only."""
    report = ReclaimReport(root=root)
    by_hash: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.pt")):
        if not path.is_file():
            continue
        report.total_files += 1
        size = path.stat().st_size
        report.total_bytes += size
        digest = _sha256_file(path)
        by_hash.setdefault(digest, []).append(path)
        if inspect_fat and _is_anchor(path) and _is_fat_checkpoint(path):
            report.fat_anchors.append(path)
    for paths in by_hash.values():
        if len(paths) < 2:
            continue
        report.duplicate_groups.append(paths)
        # Every copy past the first is reclaimable (they're byte-identical).
        report.dup_reclaim_bytes += paths[0].stat().st_size * (len(paths) - 1)
    return report


def _format_bytes(n: int) -> str:
    """Human-readable byte count."""
    val = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024.0 or unit == "TB":
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{n} B"


def render_report(report: ReclaimReport) -> str:
    """Render a human-readable dry-run report."""
    lines: list[str] = []
    lines.append(
        f"scanned {report.total_files} .pt files under {report.root} "
        f"({_format_bytes(report.total_bytes)})"
    )
    lines.append("")
    lines.append(f"byte-identical duplicate groups: {len(report.duplicate_groups)}")
    for group in report.duplicate_groups:
        keep = group[0]
        lines.append(f"  keep  {keep}  ({_format_bytes(keep.stat().st_size)})")
        for dup in group[1:]:
            lines.append(f"  dup   {dup}  -> hardlink to keep")
    lines.append("")
    lines.append(f"fat anchors (embed league/optimizer): {len(report.fat_anchors)}")
    for anchor in report.fat_anchors:
        lines.append(f"  slim  {anchor}  ({_format_bytes(anchor.stat().st_size)}) -> policy-only")
    lines.append("")
    lines.append(
        f"estimated duplicate reclaim: {_format_bytes(report.estimated_reclaim_bytes())} "
        "(anchor slim savings additional, measured on --execute)"
    )
    return "\n".join(lines)


def _dedup_group(group: list[Path]) -> int:
    """Replace copies past the first with hardlinks to it. Returns bytes freed.

    Cross-filesystem groups (``os.link`` raises ``OSError``/EXDEV) are left
    untouched — the report already surfaced them; forcing a copy-delete would
    risk data loss for zero space win."""
    import os

    keep = group[0]
    freed = 0
    for dup in group[1:]:
        size = dup.stat().st_size
        tmp = dup.with_suffix(dup.suffix + ".relink.tmp")
        try:
            os.link(keep, tmp)
        except OSError:
            # Different filesystem (or link unsupported) — skip this copy.
            continue
        os.replace(tmp, dup)
        freed += size
    return freed


def execute(report: ReclaimReport) -> tuple[int, int]:
    """Apply the report: dedup duplicates + slim fat anchors.

    Returns ``(dup_bytes_freed, anchor_bytes_freed)``. Confined to ``report.root``
    (every path came from a scan of that tree)."""
    from catan_rl.checkpoint import bank_anchor

    dup_freed = 0
    for group in report.duplicate_groups:
        # Defensive: refuse to touch anything that escaped the root.
        if not all(_is_under(p, report.root) for p in group):
            continue
        dup_freed += _dedup_group(group)

    anchor_freed = 0
    for anchor in report.fat_anchors:
        if not _is_under(anchor, report.root):
            continue
        before = anchor.stat().st_size
        bank_anchor(anchor)  # in-place slim rewrite
        after = anchor.stat().st_size
        anchor_freed += max(0, before - after)
    return dup_freed, anchor_freed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (free function so tests can introspect it)."""
    p = argparse.ArgumentParser(
        prog="reclaim_disk.py",
        description="Report + reclaim disk from a checkpoint tree "
        "(dedup byte-identical files, slim fat anchors). Dry-run by default.",
    )
    p.add_argument("root", type=Path, help="Root directory to scan (e.g. runs/).")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually dedup + slim. WITHOUT this flag the tool only reports (the safe default).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code (0 on success)."""
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: root is not a directory: {root}")
        return 2
    report = scan(root)
    print(render_report(report))
    if args.execute:
        dup_freed, anchor_freed = execute(report)
        print("")
        print(
            f"EXECUTED: freed {_format_bytes(dup_freed)} (duplicates) + "
            f"{_format_bytes(anchor_freed)} (slim anchors) = "
            f"{_format_bytes(dup_freed + anchor_freed)} total"
        )
    else:
        print("")
        print("dry-run (default): nothing was modified. Re-run with --execute to act.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
