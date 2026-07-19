"""Acceptance pins for the pointer-arch fork (spec .claude/veriloop/specs/pointer-arch-fork.md).

Covers:
  * AC-2 honest-derivation pin: dev-deck feature = public-reveal formula; no
    engine-truth deck read reachable from the encoder.
  * AC-3 global-block seat-swap invariance + reserved-slot strict-0.0.
  * AC-4 (GNN branch) pointer-head shape / masking consistency + the BINDING
    D6 pin: GNN-branch equivariance + pointer-head action-remap consistency
    under all 12 dihedral elements.
  * AC-5 migration pin: real-anchor test (skips gracefully when the historical
    checkpoint is absent) + a portable synthetic-legacy pin on the disposition
    counts (123 transplanted / 3 zero-padded / 23 fresh-init) that runs in CI.
  * AC-6 schema single-source: encoder in-dims derive from obs_schema constants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from catan_rl.policy import obs_schema as S
from catan_rl.policy.network import CatanPolicy
from catan_rl.policy.obs_encoder import EnvObsState

# ---------------------------------------------------------------------------
# AC-6 — schema single-source
# ---------------------------------------------------------------------------


def test_corner_context_dim_is_three() -> None:
    assert S.CORNER_CONTEXT_DIM == 3  # settlement, city, is_setup (D2)


def test_player_and_global_dims_derive_from_constants() -> None:
    assert S.CURR_PLAYER_DIM == S.PLAYER_BASE_DIM + S.CURR_EXTRA_DIM + S.RESERVED_PLAYER_SLOTS
    assert S.NEXT_PLAYER_DIM == S.PLAYER_BASE_DIM + S.OPP_EXTRA_DIM + S.RESERVED_PLAYER_SLOTS
    assert S.GLOBAL_DIM == S.GLOBAL_BANK_DIM + S.GLOBAL_DEVDECK_DIM + S.GLOBAL_RESERVED_SLOTS
    assert S.GLOBAL_BANK_DIM == S.N_RESOURCES
    assert S.GLOBAL_DEVDECK_DIM == S.N_DEV_TYPES


def test_encoder_in_dims_match_schema() -> None:
    """Every player-encoder in-dim derives from obs_schema (no hardcoded dims)."""
    policy = CatanPolicy()
    assert policy.curr_player_enc.net[0].in_features == S.CURR_PLAYER_DIM
    assert policy.opp_player_enc.net[0].in_features == S.NEXT_PLAYER_DIM
    # Fusion input includes the appended global block.
    assert policy.fusion[0].in_features >= S.GLOBAL_DIM


# ---------------------------------------------------------------------------
# Helpers — a live env + obs encoder
# ---------------------------------------------------------------------------


def _fresh_env():  # type: ignore[no-untyped-def]
    from catan_rl.env.catan_env import CatanEnv

    env = CatanEnv()
    env.reset(seed=0)
    return env


def _env_state(env, *, setup: bool = False) -> EnvObsState:  # type: ignore[no-untyped-def]
    return EnvObsState(initial_placement_phase=setup)


# ---------------------------------------------------------------------------
# AC-3 — global block seat-swap invariance + reserved strict-0.0
# ---------------------------------------------------------------------------


def test_global_bank_seatswap_invariant_and_reserved_zero() -> None:
    env = _fresh_env()
    enc = env._obs_encoder
    assert enc is not None
    a, o = env.agent_player, env.opponent_player
    # Give the two players different dev hands so the POV-relative dev-deck view
    # differs, isolating the truly-global bank subvector for the invariance test.
    a.devCards = {"KNIGHT": 1, "MONOPOLY": 1, "VP": 0, "ROADBUILDER": 0, "YEAROFPLENTY": 0}
    o.devCards = {"KNIGHT": 0, "MONOPOLY": 0, "VP": 1, "ROADBUILDER": 0, "YEAROFPLENTY": 0}

    g_agent = enc._build_global_features(env.game, a, o)
    g_opp = enc._build_global_features(env.game, o, a)

    # Bank subvector is truly global — identical from either seat.
    assert np.array_equal(g_agent[: S.GLOBAL_BANK_DIM], g_opp[: S.GLOBAL_BANK_DIM])
    # Reserved slots are exactly 0.0 from either seat (a non-zero constant would
    # train a bias and shift the distribution when a slot is repurposed).
    assert np.all(g_agent[-S.GLOBAL_RESERVED_SLOTS :] == 0.0)
    assert np.all(g_opp[-S.GLOBAL_RESERVED_SLOTS :] == 0.0)


def test_player_reserved_slots_strict_zero() -> None:
    env = _fresh_env()
    obs = env._get_obs()
    cur = obs["current_player_main"]
    nxt = obs["next_player_main"]
    assert np.all(cur[-S.RESERVED_PLAYER_SLOTS :] == 0.0)
    assert np.all(nxt[-S.RESERVED_PLAYER_SLOTS :] == 0.0)


# ---------------------------------------------------------------------------
# AC-2 — honest dev-deck derivation; no engine-truth deck read
# ---------------------------------------------------------------------------


def test_devdeck_matches_public_reveal_formula() -> None:
    env = _fresh_env()
    enc = env._obs_encoder
    assert enc is not None
    a, o = env.agent_player, env.opponent_player
    a.devCards = {"KNIGHT": 2, "VP": 0, "ROADBUILDER": 0, "YEAROFPLENTY": 1, "MONOPOLY": 0}
    a.knightsPlayed = 1  # own played (public)
    o.knightsPlayed = 3  # opponent played (public)
    o.monopolyPlayed = 1

    g = enc._build_global_features(env.game, a, o)
    devdeck = g[S.GLOBAL_BANK_DIM : S.GLOBAL_BANK_DIM + S.GLOBAL_DEVDECK_DIM]

    # Manual public-reveal formula: initial - own_held - agent_played - opp_played,
    # clipped >=0, per-type-normalised. KNIGHT (idx 0): 14 - 2 - 1 - 3 = 8 → 8/14.
    initial = S.DEV_DECK_INITIAL
    expected_knight = max(0.0, initial[0] - 2 - 1 - 3) / initial[0]
    assert devdeck[0] == pytest.approx(expected_knight)
    # YEAROFPLENTY (idx 3): 2 - 1(own held) = 1 → 1/2.
    assert devdeck[3] == pytest.approx((initial[3] - 1) / initial[3])


def test_devdeck_does_not_read_engine_deck_truth() -> None:
    """Mutating the engine's real dev-card stack (deck ground truth) must NOT
    change the dev-deck feature — it is derived from public info only."""
    env = _fresh_env()
    enc = env._obs_encoder
    assert enc is not None
    a, o = env.agent_player, env.opponent_player
    g_before = enc._build_global_features(env.game, a, o).copy()

    # Repartition the engine deck arbitrarily (were the encoder to read it, the
    # per-type feature would move).
    stack = getattr(env.game.board, "devCardStack", None)
    if stack is None:
        pytest.skip("engine exposes no devCardStack to perturb")
    for k in list(stack.keys()):
        stack[k] = 0
    g_after = enc._build_global_features(env.game, a, o)
    assert np.array_equal(g_before, g_after)


# ---------------------------------------------------------------------------
# AC-4 (GNN branch) — pointer-head shapes + masking consistency
# ---------------------------------------------------------------------------


def _synthetic_obs(batch: int) -> dict[str, torch.Tensor]:
    return {
        "tile_representations": torch.rand(batch, S.N_TILES, S.TILE_DIM),
        "current_player_main": torch.rand(batch, S.CURR_PLAYER_DIM),
        "next_player_main": torch.rand(batch, S.NEXT_PLAYER_DIM),
        "current_dev_counts": torch.rand(batch, S.N_DEV_TYPES),
        "next_played_dev_counts": torch.rand(batch, S.N_DEV_TYPES),
        "global_features": torch.rand(batch, S.GLOBAL_DIM),
        "hex_features": torch.rand(batch, S.N_TILES, 19),
        "vertex_features": torch.rand(batch, S.N_VERTICES, 16),
        "edge_features": torch.rand(batch, S.N_EDGES, 16),
        "opponent_kind": torch.zeros(batch, dtype=torch.long),
        "opponent_policy_id": torch.zeros(batch, dtype=torch.long),
        "is_setup": torch.ones(batch, 1),
    }


def _full_masks(batch: int) -> dict[str, torch.Tensor]:
    return {
        "type": torch.ones(batch, S.N_ACTION_TYPES, dtype=torch.bool),
        "corner_settlement": torch.ones(batch, S.N_VERTICES, dtype=torch.bool),
        "corner_city": torch.ones(batch, S.N_VERTICES, dtype=torch.bool),
        "edge": torch.ones(batch, S.N_EDGES, dtype=torch.bool),
        "tile": torch.ones(batch, S.N_TILES, dtype=torch.bool),
        "resource1_trade": torch.ones(batch, S.N_RESOURCES, dtype=torch.bool),
        "resource1_discard": torch.ones(batch, S.N_RESOURCES, dtype=torch.bool),
        "resource1_default": torch.ones(batch, S.N_RESOURCES, dtype=torch.bool),
        "resource2_default": torch.ones(batch, S.N_RESOURCES, dtype=torch.bool),
    }


def test_pointer_heads_emit_per_node_logits_and_respect_masks() -> None:
    policy = CatanPolicy()
    obs = _synthetic_obs(2)
    masks = _full_masks(2)
    out = policy.forward(obs)
    nodes = {"v": out["_node_v"], "e": out["_node_e"], "h": out["_node_h"]}
    # Per-node state widths match the board geometry (the pointer-readout sizes).
    assert nodes["v"].shape == (2, S.N_VERTICES, policy.action_heads.node_dim)
    assert nodes["e"].shape == (2, S.N_EDGES, policy.action_heads.node_dim)
    assert nodes["h"].shape == (2, S.N_TILES, policy.action_heads.node_dim)

    ev = policy.evaluate_actions(obs, policy.sample(obs, masks)["action"], masks)
    assert ev["log_prob"].shape == (2,)

    # Masking still works: forbid all but corner index 3 on a settlement action.
    masks["corner_settlement"] = torch.zeros(2, S.N_VERTICES, dtype=torch.bool)
    masks["corner_settlement"][:, 3] = True
    action = torch.zeros(2, 6, dtype=torch.long)  # type 0 = BuildSettlement
    action[:, 1] = 3
    ev2 = policy.evaluate_actions(obs, action, masks)
    assert torch.isfinite(ev2["log_prob"]).all()


def _fwd_perm(x: torch.Tensor, perm: np.ndarray) -> torch.Tensor:
    """Reindex node axis of ``x`` (B, N, ...) by ``g``.

    ``symmetry_tables`` perms use the convention ``perm[old] = new`` (old node
    ``i`` lands on ``perm[i]``). The array whose new-index ``new`` carries old
    node ``old``'s row is therefore ``x[:, argsort(perm)]``.
    """
    inv = torch.as_tensor(np.argsort(perm), dtype=torch.long)
    return x[:, inv]


def test_gnn_branch_and_pointer_heads_are_d6_equivariant() -> None:
    """AC-4 (BINDING pin): GNN-branch equivariance + pointer-head action-remap
    consistency under all 12 D6 elements.

    The fork's D1 pointer readouts sit on the GNN's per-node states, so the
    equivariance claim they rest on is: applying any of the 12 dihedral board
    symmetries to the GNN's per-node INPUT features permutes (i) the GNN's
    per-node OUTPUT states and (ii) the corner / edge / tile pointer LOGITS by
    the matching vertex / edge / tile permutation — i.e. the WHERE-head scores
    remap consistently with the board relabelling. This exercises the real
    ``augmentation.symmetry_tables`` D6 perms (``D6_GROUP_SIZE == 12``) against
    the real board geometry, not just tensor shapes.
    """
    from catan_rl.augmentation import symmetry_tables as st
    from catan_rl.policy.board_geometry import build_geometry

    assert st.D6_GROUP_SIZE == 12

    torch.manual_seed(0)
    policy = CatanPolicy(use_belief_head=False, use_aux_value_head=False)
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy.eval()
    ge = policy.graph_encoder
    heads = policy.action_heads

    b = 2
    hexf = torch.rand(b, S.N_TILES, 19)
    vf = torch.rand(b, S.N_VERTICES, 16)
    ef = torch.rand(b, S.N_EDGES, 16)
    # A fixed, arbitrary trunk: the pointer readout broadcasts it per-node, so
    # remap consistency must hold for ANY trunk (it is a per-graph context, not
    # a per-node signal). Corner FiLM context held fixed at a setup settlement.
    trunk = torch.rand(b, policy.trunk_dim)
    type_idx = torch.zeros(b, dtype=torch.long)  # BUILD_SETTLEMENT
    corner_ctx = heads._corner_context(type_idx, torch.ones(b, 1))

    tol = 1e-4
    with torch.inference_mode():
        _, v, e, h = ge(hexf, vf, ef)
        base_corner = heads.corner_head(trunk, v, corner_ctx)
        base_edge = heads.edge_head(trunk, e)
        base_tile = heads.tile_head(trunk, h)

        for g in range(st.D6_GROUP_SIZE):
            tp, cp, ep = st.tile_perm(g), st.corner_perm(g), st.edge_perm(g)
            # Apply g to the GNN's per-node INPUTS.
            _, vg, eg, hg = ge(_fwd_perm(hexf, tp), _fwd_perm(vf, cp), _fwd_perm(ef, ep))

            # (i) GNN-branch equivariance: output node states permute by g.
            assert torch.allclose(vg, _fwd_perm(v, cp), atol=tol), f"vertex GNN g={g}"
            assert torch.allclose(eg, _fwd_perm(e, ep), atol=tol), f"edge GNN g={g}"
            assert torch.allclose(hg, _fwd_perm(h, tp), atol=tol), f"hex GNN g={g}"

            # (ii) pointer-head action-remap: WHERE logits permute by g.
            corner_g = heads.corner_head(trunk, vg, corner_ctx)
            edge_g = heads.edge_head(trunk, eg)
            tile_g = heads.tile_head(trunk, hg)
            assert torch.allclose(corner_g, _fwd_perm(base_corner, cp), atol=tol), f"corner g={g}"
            assert torch.allclose(edge_g, _fwd_perm(base_edge, ep), atol=tol), f"edge g={g}"
            assert torch.allclose(tile_g, _fwd_perm(base_tile, tp), atol=tol), f"tile g={g}"


# ---------------------------------------------------------------------------
# AC-5 — migration pin
# ---------------------------------------------------------------------------


def _find_anchor() -> Path | None:
    for cand in (
        Path("runs/anchors/v11_cand_u724.pt"),
        Path(__file__).resolve().parents[3] / "runs/anchors/v11_cand_u724.pt",
    ):
        if cand.exists():
            return cand
    return None


def test_migration_transplants_and_zero_pads() -> None:
    anchor = _find_anchor()
    if anchor is None:
        pytest.skip("no historical anchor checkpoint available")
    from catan_rl.checkpoint.pointer_arch_migration import migrate_state_dict_to_pointer_arch

    payload = torch.load(anchor, map_location="cpu", weights_only=False)
    old = payload["policy_state_dict"]
    policy, report = migrate_state_dict_to_pointer_arch(old)
    new = policy.state_dict()

    # Transplanted blocks are byte-equal to the legacy weights.
    for key in new:
        if key.startswith(("tile_encoder.", "graph_encoder.")) and key in old:
            assert torch.equal(new[key], old[key]), key

    # New input columns are zero (pure tail-append).
    assert torch.all(new["curr_player_enc.net.0.weight"][:, S.PLAYER_BASE_DIM :] == 0.0)
    fusion_old_cols = old["fusion.0.weight"].shape[1]
    fw = new["fusion.0.weight"]
    assert torch.equal(fw[:, :fusion_old_cols], old["fusion.0.weight"])
    assert torch.all(fw[:, fusion_old_cols:] == 0.0)

    # Pointer heads + aux head are fresh-init (not transplanted).
    assert any("corner_head" in k for k in report["fresh_init"])
    assert any("aux_value_head" in k for k in report["fresh_init"])

    # Forward runs after migration.
    out = policy.forward(_synthetic_obs(1))
    assert out["value"].shape == (1,)


def _synthetic_legacy_state() -> dict[str, torch.Tensor]:
    """A pre-fork-shaped ``CatanPolicy`` state dict, derived from the current arch.

    The fork only ADDS parameters (pointer/aux heads = fresh-init prefixes) and
    tail-appends input columns to the three zero-pad keys, so a faithful legacy
    dict is the current state dict minus the fresh-init keys, with the zero-pad
    keys trimmed back to their legacy input widths. This lets AC-5's migration
    mechanics + documented disposition counts be pinned WITHOUT the real
    historical anchor (which is absent in CI / isolated worktrees).
    """
    from catan_rl.checkpoint.pointer_arch_migration import FRESH_INIT_PREFIXES

    new = CatanPolicy().state_dict()
    # Legacy input widths (pre-fork): current = base only; opponent = base + the
    # legacy opp extras; fusion loses the appended global-block columns.
    legacy_cols = {
        "curr_player_enc.net.0.weight": S.PLAYER_BASE_DIM,
        "opp_player_enc.net.0.weight": S.PLAYER_BASE_DIM + S.OPP_EXTRA_DIM,
    }
    legacy: dict[str, torch.Tensor] = {}
    for k, v in new.items():
        if any(k.startswith(p) for p in FRESH_INIT_PREFIXES):
            continue  # legacy had no pointer heads / aux value head
        if k in legacy_cols:
            legacy[k] = v[:, : legacy_cols[k]].clone()
        elif k == "fusion.0.weight":
            legacy[k] = v[:, : v.shape[1] - S.GLOBAL_DIM].clone()
        else:
            legacy[k] = v.clone()
    return legacy


def test_migration_disposition_counts_are_portable() -> None:
    """Portable AC-5 pin: the transplant/zero-pad/fresh-init dispositions match
    the counts asserted in docs/plans/v2/pointer_arch_lineage.md (123 / 3 / 23),
    independent of the historical anchor being on disk."""
    from catan_rl.checkpoint.pointer_arch_migration import migrate_state_dict_to_pointer_arch

    legacy = _synthetic_legacy_state()
    policy, report = migrate_state_dict_to_pointer_arch(legacy)
    new = policy.state_dict()

    assert len(report["transplanted"]) == 123
    assert len(report["zero_padded"]) == 3
    assert len(report["fresh_init"]) == 23

    # Transplanted blocks are byte-equal to the legacy weights.
    for key in report["transplanted"]:
        assert torch.equal(new[key], legacy[key]), key

    # Zero-pad keys copy the legacy overlap and zero the appended tail.
    for key in report["zero_padded"]:
        old_cols = legacy[key].shape[1]
        assert torch.equal(new[key][:, :old_cols], legacy[key]), key
        assert torch.all(new[key][:, old_cols:] == 0.0), key

    # Pointer heads + aux head are fresh-init.
    assert any("corner_head" in k for k in report["fresh_init"])
    assert any("aux_value_head" in k for k in report["fresh_init"])

    # Forward runs after migration.
    out = policy.forward(_synthetic_obs(1))
    assert out["value"].shape == (1,)
