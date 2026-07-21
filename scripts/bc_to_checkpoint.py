#!/usr/bin/env python3
"""Bridge a bare BC checkpoint into a schema'd, ``load_checkpoint``-compatible one.

``catan_rl.bc.train._save_checkpoint`` writes a BARE payload
``{"policy_state_dict", "step", "val_nll"}`` with NO ``schema_version``. The
canonical checkpoint loader (:func:`catan_rl.checkpoint.load_checkpoint`) — used
by the PPO learner warm-start (``init_policy_checkpoint``), ``build_actor``, and
the eval harness — REFUSES a payload without ``schema_version``
(``migrations.py``: "refusing to guess"). So a freshly BC-trained ``best.pt``
cannot warm-start the heuristic-bootstrap PPO stage as-is.

This one-shot converter wraps that bare policy state dict into a
``SCHEMA_VERSION`` POLICY-ONLY checkpoint via
:func:`catan_rl.checkpoint.save_policy_only`, which the canonical loader reads.
It mirrors the in-tree precedent in ``expert_iteration/distill.py`` (load raw BC
-> strict-load into a ``CatanPolicy`` -> re-save schema'd), so the bridged file
is byte-for-byte the same policy, now loadable through the standard path with a
fresh optimizer (a warm-start, not a resume).

Usage::

    # writes runs/bc/pointer_arch_30k/best_ckpt.pt (schema'd, policy-only)
    python scripts/bc_to_checkpoint.py --in runs/bc/pointer_arch_30k/best.pt

    # explicit destination
    python scripts/bc_to_checkpoint.py --in runs/bc/.../best.pt --out /path/out.pt

Exit codes:
* 0 — bridged checkpoint written.
* 1 — input missing, not a bare BC payload, or load/validation failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def default_out_path(src: Path) -> Path:
    """Return the sibling ``<stem>_ckpt<suffix>`` path (e.g. ``best.pt`` ->
    ``best_ckpt.pt``) used when ``--out`` is omitted."""
    return src.with_name(f"{src.stem}_ckpt{src.suffix}")


def convert_bc_checkpoint(src: Path, dest: Path) -> Path:
    """Load a bare BC checkpoint at ``src`` and write a schema'd policy-only
    checkpoint at ``dest``; return the written path.

    Validates the payload shape and that its ``policy_state_dict`` strict-loads
    into a fresh (default, pointer-arch) :class:`CatanPolicy`, then re-saves via
    :func:`save_policy_only`. Raises ``ValueError`` if ``src`` is not a bare BC
    payload (or already carries ``schema_version``), and ``RuntimeError`` if the
    state dict does not match the current policy architecture.
    """
    import torch

    from catan_rl.checkpoint import save_policy_only
    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy

    raw: Any = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "policy_state_dict" not in raw:
        got = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(
            f"{src} is not a bare BC checkpoint (expected a dict with a "
            f"'policy_state_dict' key); got {got}"
        )
    if "schema_version" in raw:
        raise ValueError(
            f"{src} already carries 'schema_version'={raw['schema_version']!r}; it is "
            "already loadable via load_checkpoint — no BC bridge is needed. Use "
            "scripts/migrate_checkpoint.py if a schema upgrade is what you want."
        )

    # Validate + normalise by strict-loading into a fresh pointer-arch policy
    # (mirrors expert_iteration/distill.py). Geometry buffers are set first, then
    # overwritten by the checkpoint's identical geometry on the strict load.
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy.load_state_dict(raw["policy_state_dict"], strict=True)

    val_nll = raw.get("val_nll")
    return save_policy_only(
        dest,
        config={"source": "bc", "bc_src": str(src)},
        policy=policy,
        update_idx=0,
        global_step=int(raw.get("step", 0)),
        metadata={
            "lineage": "bc->heuristic-bootstrap",
            "bc_src": str(src),
            "bc_val_nll": float(val_nll) if val_nll is not None else None,
        },
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        type=Path,
        help="Path to the bare BC checkpoint (e.g. runs/bc/<run>/best.pt).",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=None,
        help="Destination for the bridged checkpoint. Defaults to a sibling "
        "'<stem>_ckpt.pt' next to the input.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if not args.in_path.exists():
        print(f"error: input not found: {args.in_path}", file=sys.stderr)
        return 1

    dest: Path = args.out_path if args.out_path is not None else default_out_path(args.in_path)
    try:
        written = convert_bc_checkpoint(args.in_path, dest)
    except Exception as e:  # CLI: surface any load/validation error as a clean exit 1
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"bridged BC {args.in_path} -> {written} (schema'd policy-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
