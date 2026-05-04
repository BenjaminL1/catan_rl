"""Regression test for the Nash-pruning architecture-mismatch crash.

A live run died at step 1,617,920 because ``_build_payoff_matrix``
constructed inference policies via a bare ``build_agent_model(device=...)``
call (no model_kwargs), producing the legacy 1.54M baseline architecture.
That policy could not load phase4_full state_dicts (which include opp-id
embedding, belief head, opp-action head, GNN encoder, recurrent value
tower, decoupled value tower, FiLM action heads, axial pos-emb, count
dev cards, compact obs schema, etc.).

This test catches the issue cheaply: build a phase-flagged trainer, fill
the league with state_dicts from that trainer's policy, then invoke
``_build_payoff_matrix`` directly. Without the fix, this raises
``RuntimeError: Error(s) in loading state_dict``. With the fix, it
returns a valid payoff matrix.
"""

from __future__ import annotations

import copy

from catan_rl.algorithms.ppo.trainer import CatanPPO


def _phase4_min_config() -> dict:
    """Smallest viable phase4_full-style config for the test."""
    return {
        # ── Compact obs (Phase 1.3) — would crash with 79-dim baseline encoder
        "use_thermometer_encoding": False,
        "curr_player_main_in_dim": 54,
        "other_player_main_in_dim": 61,
        # ── Phase 1.4 count dev cards — replaces MHA path
        "use_devcard_mha": False,
        # ── Phase 2 architecture
        "use_axial_pos_emb": True,
        "axial_pos_dim": 24,
        "transformer_dropout": 0.05,
        "transformer_activation": "gelu",
        "action_head_film": True,
        "value_head_mode": "decoupled",
        # ── Phase 2.5b/c aux heads
        "use_belief_head": True,
        "use_opponent_action_head": True,
        # ── Phase 3.6 opponent-id embedding
        "use_opponent_id_emb": True,
        # ── Phase 4.2 recurrent value
        "use_recurrent_value": True,
        "gru_hidden_dim": 64,
        # ── Phase 2.3 GNN encoder
        "use_graph_encoder": True,
        # ── Smallest viable PPO config (we just need trainer construction +
        # a single Nash pruning round-robin).
        "obs_output_dim": 64,
        "tile_in_dim": 79,
        "tile_model_dim": 32,
        "dev_card_embed_dim": 16,
        "dev_card_model_dim": 16,
        "tile_model_num_heads": 2,
        "proj_dev_card_dim": 8,
        "dev_card_model_num_heads": 2,
        "tile_encoder_num_layers": 1,
        "proj_tile_dim": 8,
        "action_head_hidden_dim": 32,
        "value_hidden_dims": (32,),
        "dropout": 0.0,
        "max_dev_seq": 16,
        "dev_card_vocab_excl_pad": 5,
        "belief_head_hidden_dim": 32,
        "opponent_action_head_hidden_dim": 32,
        "opp_id_emb_dim": 16,
        "n_opp_kinds": 6,
        "league_maxlen": 4,
        "graph_hidden_dim": 16,
        "graph_n_rounds": 1,
        "graph_out_dim": 16,
        # PPO basics
        "n_steps": 64,
        "n_envs": 1,
        "batch_size": 32,
        "n_epochs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "learning_rate": 1e-3,
        "lr_final": 1e-4,
        "weight_decay": 0.0,
        "use_linear_lr_decay": False,
        "entropy_coef": 0.01,
        "entropy_coef_start": 0.01,
        "entropy_coef_final": 0.005,
        "entropy_coef_anneal_start": 100,
        "entropy_coef_anneal_end": 200,
        "total_timesteps": 64,
        "checkpoint_freq": 10**9,
        "add_policy_every": 4,
        "league_random_weight": 0.0,
        "heuristic_opponent_weight": 0.0,
        "max_turns": 100,
        "log_dir": "/tmp/nash_arch_test_logs",
        "checkpoint_dir": "/tmp/nash_arch_test_ckpts",
        "device": "cpu",
        # Nash pruning + ratings
        "prune_strategy": "nash",
        "prune_every": 1,
        "nash_top_k": 4,
        "nash_prune_round_games": 2,  # very small, just to exercise the path
    }


def test_nash_pruning_loads_phase4_state_dicts() -> None:
    """``_build_payoff_matrix`` must build inference policies with the
    same architecture as the league entries.

    Without the fix, this raises ``RuntimeError`` on the first
    ``pol_a.load_state_dict`` call due to size/key mismatches.
    """
    trainer = CatanPPO(_phase4_min_config())
    # Fill the league with copies of the live policy state — these
    # carry phase4-shape weights that a baseline build cannot load.
    sd = copy.deepcopy(trainer._raw_policy_state_dict())
    for _ in range(3):  # league already has 1; add 3 more for top-k=4
        trainer.league.add(sd)
    ids = list(trainer.league._policy_ids)
    state_dicts = dict(zip(ids, list(trainer.league.policies), strict=True))

    # Direct call to the function that crashed in production.
    payoff = trainer._build_payoff_matrix(ids, state_dicts)
    assert payoff.shape == (len(ids), len(ids))
    # Diagonal stays at 0.5 (self-vs-self not played).
    for i in range(len(ids)):
        assert payoff[i, i] == 0.5


def test_nash_pruning_fails_loud_when_build_fn_missing() -> None:
    """Defensive check: the fix raises a clear error if the league has
    no ``build_policy_fn``, instead of falling back to a broken default.
    """
    import pytest

    trainer = CatanPPO(_phase4_min_config())
    sd = copy.deepcopy(trainer._raw_policy_state_dict())
    trainer.league.add(sd)
    trainer.league._build_policy_fn = None  # simulate a misconfigured trainer
    ids = list(trainer.league._policy_ids)
    state_dicts = dict(zip(ids, list(trainer.league.policies), strict=True))
    with pytest.raises(RuntimeError, match="build_policy_fn"):
        trainer._build_payoff_matrix(ids, state_dicts)
