"""One-shot migration: transplant a legacy v2 checkpoint into the pointer-arch.

This is the D5 migration utility for the pointer-arch fork. It loads any
existing v2 ``CatanPolicy`` state dict and produces a new-architecture policy:

  * **Transplant verbatim** every block whose schema is unchanged by the fork —
    the tile encoder, the tripartite GNN, the dev-card encoders, the opponent-id
    embedding, the value / belief heads, the type / resource heads.
  * **Zero-pad** the new INPUT columns of the two player encoders and the fusion
    Linear. The fork appends its new signals (per-player scalars + reserved
    slots; the POV-neutral global block) at the TAIL of each vector, so the
    legacy columns keep their positions and only the appended columns need
    zeroing (a zero column contributes nothing until the fresh weights train).
  * **Fresh-initialise** the three location pointer readouts (corner/edge/tile)
    and the auxiliary value head — brand-new submodules whose parameters have no
    legacy counterpart. Name collisions (e.g. ``corner_head.norm`` exists in
    both the old FiLM head and the new pointer head but with a DIFFERENT shape /
    meaning) are handled by fresh-init prefixes, never by a blind copy.

Lineage: the ratified seeding path is a FULL RE-BOOTSTRAP; this transplant keeps
the alternative (contingency) seeding path open per rule 3.
"""

from __future__ import annotations

from typing import Any

import torch

from catan_rl.policy.network import CatanPolicy

#: Parameter-name prefixes that are FRESH-INITIALISED (never transplanted). The
#: three location heads were replaced by pointer readouts and the aux value head
#: is new; any legacy tensor sharing one of these names is a different tensor.
FRESH_INIT_PREFIXES: tuple[str, ...] = (
    "action_heads.corner_head.",
    "action_heads.edge_head.",
    "action_heads.tile_head.",
    "aux_value_head.",
)

#: Parameters whose input dim grew by a pure TAIL-append; copy the leading
#: overlap from the legacy weight, leave the appended columns at zero.
ZERO_PAD_KEYS: tuple[str, ...] = (
    "curr_player_enc.net.0.weight",
    "opp_player_enc.net.0.weight",
    "fusion.0.weight",
)


class MigrationReport(dict[str, list[str]]):
    """Per-parameter disposition: ``transplanted`` / ``zero_padded`` /
    ``fresh_init`` (each a sorted list of parameter names)."""


def migrate_state_dict_to_pointer_arch(
    old_state: dict[str, torch.Tensor],
    *,
    new_policy: CatanPolicy | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[CatanPolicy, MigrationReport]:
    """Transplant ``old_state`` (a legacy ``CatanPolicy.state_dict()``) into a
    fresh pointer-arch ``CatanPolicy``.

    Returns the loaded new policy and a :class:`MigrationReport`.
    """
    if new_policy is None:
        new_policy = CatanPolicy(**(config or {}))
    new_state = new_policy.state_dict()
    report = MigrationReport(transplanted=[], zero_padded=[], fresh_init=[])

    for key, new_tensor in new_state.items():
        if any(key.startswith(p) for p in FRESH_INIT_PREFIXES):
            report["fresh_init"].append(key)
            continue
        if key in ZERO_PAD_KEYS and key in old_state:
            old_tensor = old_state[key]
            merged = torch.zeros_like(new_tensor)
            slices = tuple(
                slice(0, min(o, n)) for o, n in zip(old_tensor.shape, new_tensor.shape, strict=True)
            )
            merged[slices] = old_tensor[slices]
            new_state[key] = merged
            report["zero_padded"].append(key)
            continue
        if key in old_state and tuple(old_state[key].shape) == tuple(new_tensor.shape):
            new_state[key] = old_state[key].clone()
            report["transplanted"].append(key)
        else:
            report["fresh_init"].append(key)

    new_policy.load_state_dict(new_state, strict=True)
    for k in report:
        report[k].sort()
    return new_policy, report


def migrate_checkpoint_payload(
    payload: dict[str, Any], *, config: dict[str, Any] | None = None
) -> tuple[dict[str, Any], MigrationReport]:
    """Return a shallow-copied checkpoint payload whose ``policy_state_dict`` has
    been transplanted to the pointer-arch, plus the report. The optimizer state
    (shape-coupled to the old parameters) is DROPPED — a transplant restarts the
    optimizer — so callers re-bootstrap or warm-start cleanly."""
    if "policy_state_dict" not in payload:
        raise KeyError("payload missing 'policy_state_dict'")
    new_policy, report = migrate_state_dict_to_pointer_arch(
        payload["policy_state_dict"], config=config
    )
    out = dict(payload)
    out["policy_state_dict"] = {k: v.cpu().clone() for k, v in new_policy.state_dict().items()}
    out["optimizer_state_dict"] = None
    out["pointer_arch_migrated"] = True
    return out, report
