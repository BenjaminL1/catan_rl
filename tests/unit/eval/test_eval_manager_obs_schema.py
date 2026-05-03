"""Regression test for the eval-manager obs-schema mismatch crash.

A live training run died twice at the first eval (step 100k) because
``EvaluationManager`` constructed its `CatanEnv` without the Phase 3.6
``use_opponent_id_emb`` and Phase 2.5b ``use_belief_head`` flags. The
env then emitted obs dicts missing ``opponent_kind`` /
``opponent_policy_id`` / ``belief_target``, but the policy (built with
those flags on) raised ``KeyError`` during ``policy.act``.

This test catches it cheaply: build a phase-flagged policy + matching
EvaluationManager, run a single deterministic eval game, assert no
KeyError. The test won't catch all schema drift, but it covers the
specific failure mode that took down the live run.
"""

from __future__ import annotations

from catan_rl.eval.evaluation_manager import EvaluationManager
from catan_rl.models.build_agent_model import build_agent_model


def test_eval_manager_with_opp_id_emb_does_not_crash() -> None:
    """``use_opponent_id_emb=True`` policy + matching eval env runs cleanly."""
    policy = build_agent_model(
        device="cpu",
        use_opponent_id_emb=True,
        opp_id_emb_dim=16,
        n_opp_kinds=6,
        league_maxlen=100,
    )
    policy.eval()
    em = EvaluationManager(
        n_games=1,
        opponent_type="random",
        max_turns=50,  # short — we're testing the schema, not WR
        use_thermometer_encoding=True,  # legacy schema; pairs with the default policy
        use_opponent_id_emb=True,  # MUST match the policy
        league_maxlen=100,
    )
    # Single seeded game — completes deterministically.
    stats = em.evaluate(policy, device="cpu", seeds=[0], deterministic=True)
    assert "win_rate" in stats


def test_eval_manager_with_belief_head_does_not_crash() -> None:
    """``use_belief_head=True`` policy + matching eval env runs cleanly."""
    policy = build_agent_model(
        device="cpu",
        use_belief_head=True,
    )
    policy.eval()
    em = EvaluationManager(
        n_games=1,
        opponent_type="random",
        max_turns=50,
        use_thermometer_encoding=True,
        use_belief_head=True,
    )
    stats = em.evaluate(policy, device="cpu", seeds=[0], deterministic=True)
    assert "win_rate" in stats


def test_eval_manager_with_full_phase4_does_not_crash() -> None:
    """phase4_full-style policy + fully-matching eval env runs cleanly.

    This is the *exact* config that crashed the live training run twice
    at step 100k.
    """
    policy = build_agent_model(
        device="cpu",
        # Phase 1.3
        use_thermometer_encoding=False,
        curr_player_main_in_dim=54,
        other_player_main_in_dim=61,
        # Phase 1.4
        use_devcard_mha=False,
        # Phase 2 architecture
        use_axial_pos_emb=True,
        action_head_film=True,
        value_head_mode="decoupled",
        # Phase 2.5b/c aux heads
        use_belief_head=True,
        use_opponent_action_head=True,
        # Phase 3.6 opp id emb
        use_opponent_id_emb=True,
        # Phase 4.2 recurrent value
        use_recurrent_value=True,
        gru_hidden_dim=64,
    )
    policy.eval()
    em = EvaluationManager(
        n_games=1,
        opponent_type="random",
        max_turns=50,
        use_thermometer_encoding=False,
        use_opponent_id_emb=True,
        use_belief_head=True,
        league_maxlen=100,
    )
    stats = em.evaluate(policy, device="cpu", seeds=[0], deterministic=True)
    assert "win_rate" in stats


def test_default_eval_manager_with_baseline_policy() -> None:
    """Back-compat: legacy policy + legacy EvaluationManager still works."""
    policy = build_agent_model(device="cpu")  # all flags off
    policy.eval()
    em = EvaluationManager(n_games=1, opponent_type="random", max_turns=50)
    stats = em.evaluate(policy, device="cpu", seeds=[0], deterministic=True)
    assert "win_rate" in stats
