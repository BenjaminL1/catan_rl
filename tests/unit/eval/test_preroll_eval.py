"""Eval-side coverage for spec ``preroll-dev-cards-r1`` — D8 + D11.

D8: a head-to-head between policies from different ruleset epochs is REFUSED,
not annotated (one shared env cannot give the two seats different rules).
D11: at most one non-VP dev card per player per turn — the invariant that would
have caught the D2 turn-boundary bug.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from catan_rl.env.ruleset import RULESET_R0, RULESET_R1
from catan_rl.eval.harness import CrossRulesetEvalError, EvalHarness
from catan_rl.eval.rules_invariants import (
    RulesInvariantViolation,
    check_one_dev_card_per_turn,
    run_all_invariants,
)

# ---------------------------------------------------------------------------
# D8 — cross-ruleset refusal
# ---------------------------------------------------------------------------


def test_harness_defaults_to_r0_and_validates() -> None:
    """R0 — the epoch every banked checkpoint was trained under — is the eval
    default, so upgrading cannot silently re-rule an existing checkpoint."""
    assert EvalHarness().ruleset == RULESET_R0
    assert EvalHarness(ruleset=RULESET_R1).ruleset == RULESET_R1
    with pytest.raises(ValueError):
        EvalHarness(ruleset="R2")


def test_env_defaults_to_r0_and_mirrors_the_seat_rulesets() -> None:
    from catan_rl.env.catan_env import CatanEnv

    env = CatanEnv(opponent_type="random", max_turns=5)
    assert env.ruleset == RULESET_R0
    assert env.opponent_ruleset == RULESET_R0
    # Both seats follow ``ruleset`` unless the opponent seat is pinned apart.
    env_r1 = CatanEnv(opponent_type="random", max_turns=5, ruleset=RULESET_R1)
    assert (env_r1.ruleset, env_r1.opponent_ruleset) == (RULESET_R1, RULESET_R1)
    mixed = CatanEnv(
        opponent_type="random", max_turns=5, ruleset=RULESET_R1, opponent_ruleset=RULESET_R0
    )
    assert (mixed.ruleset, mixed.opponent_ruleset) == (RULESET_R1, RULESET_R0)


def test_training_config_stamps_the_epoch_into_the_checkpoint_config() -> None:
    """``checkpoint_ruleset`` reads this key back; an absent key means R0."""
    from catan_rl.eval.harness import checkpoint_ruleset
    from catan_rl.ppo.arguments import TrainConfig

    cfg = TrainConfig()
    assert cfg.to_dict()["rollout"]["ruleset"] == cfg.rollout.ruleset

    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as d:
        stamped = f"{d}/stamped.pt"
        torch.save({"config": {"rollout": {"ruleset": RULESET_R1}}}, stamped)
        assert checkpoint_ruleset(stamped) == RULESET_R1
        # A pre-slice checkpoint carries no stamp at all -> R0 by construction.
        legacy = f"{d}/legacy.pt"
        torch.save({"config": {"rollout": {"max_turns": 400}}}, legacy)
        assert checkpoint_ruleset(legacy) == RULESET_R0


def test_cross_ruleset_h2h_refuses() -> None:
    harness = EvalHarness(opponent_types=("snapshot",), n_games_per_seat=1, ruleset=RULESET_R1)
    with pytest.raises(CrossRulesetEvalError, match="cross-ruleset h2h refused"):
        harness.evaluate_vs_policy(
            champion=object(),
            opponent=object(),
            opponent_ref="ptr_v1_u500.pt",
            opponent_ruleset=RULESET_R0,
        )


def test_matching_ruleset_is_not_refused_before_any_game_runs() -> None:
    """The guard must fire on epoch mismatch ONLY — a matching epoch has to
    fall through into the normal (here: stub-driven, so failing later) path."""
    harness = EvalHarness(opponent_types=("snapshot",), n_games_per_seat=1, ruleset=RULESET_R1)
    with pytest.raises(Exception) as exc:
        harness.evaluate_vs_policy(
            champion=object(),
            opponent=object(),
            opponent_ref="same_epoch.pt",
            opponent_ruleset=RULESET_R1,
        )
    assert not isinstance(exc.value, CrossRulesetEvalError)


def test_d9_flag_experiment_is_runnable_with_per_seat_rules() -> None:
    """Acceptance #10's second clause: the refusal must not FORECLOSE D9.

    D9's replacement accept gate is "R1 policy (pre-roll ON) vs an R0 policy
    (pre-roll OFF for its seat)". With ``allow_mixed_ruleset=True`` the harness
    builds a per-seat env instead of raising — each policy plays its own rules.
    Driven here to the point the env is constructed (the stub policies then
    fail), which is all that has to be proven: the matchup is expressible.
    """
    from catan_rl.env.catan_env import CatanEnv

    built: list[CatanEnv] = []
    real_init = CatanEnv.__init__

    def _spy(self: CatanEnv, *a: object, **kw: object) -> None:
        real_init(self, *a, **kw)  # type: ignore[arg-type]
        built.append(self)

    harness = EvalHarness(opponent_types=("snapshot",), n_games_per_seat=1, ruleset=RULESET_R1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CatanEnv, "__init__", _spy)
        with pytest.raises(Exception) as exc:
            harness.evaluate_vs_policy(
                champion=object(),
                opponent=object(),
                opponent_ref="ptr_v1_u500.pt",
                opponent_ruleset=RULESET_R0,
                allow_mixed_ruleset=True,
            )
    assert not isinstance(exc.value, CrossRulesetEvalError)
    assert built, "no env was constructed — the matchup was refused, not run"
    assert (built[0].ruleset, built[0].opponent_ruleset) == (RULESET_R1, RULESET_R0)


def test_cross_arch_reads_the_opponents_epoch_off_the_checkpoint() -> None:
    """D8's refusal has to be reachable WITHOUT the caller knowing to look.

    ``cross_arch_h2h`` is the accept-gate clause-(a) entry point and no
    production caller passes a ruleset, so the epoch is read off the checkpoint
    (absent stamp ⇒ R0) and an R1 champion vs a banked R0 snapshot refuses on
    its own.
    """
    import tempfile

    import torch

    from catan_rl.eval.cross_arch import cross_arch_h2h

    with tempfile.TemporaryDirectory() as d:
        old = f"{d}/banked_r0.pt"
        torch.save({"config": {"rollout": {"max_turns": 400}}}, old)
        with pytest.raises(CrossRulesetEvalError, match="cross-ruleset h2h refused"):
            cross_arch_h2h(
                new_ckpt=f"{d}/new.pt",
                old_ckpt=old,
                n_games=2,
                ruleset=RULESET_R1,
                strict_engine_parity=False,
            )


def test_cross_arch_reads_the_NEW_champions_epoch_too() -> None:
    """Regression: the refusal was once wired on the opponent side ONLY.

    ``ruleset`` was a plain ``R0`` constant, so ``--new <R1 ckpt> --old <banked
    R0 ckpt>`` with default flags compared ``R0 == R0``, refused nothing, and
    ran the whole accept-gate matchup with the R1 champion silently denied the
    pre-roll window it was trained with. Both seats must default to their own
    checkpoint's stamp.
    """
    import tempfile

    import torch

    from catan_rl.eval.cross_arch import cross_arch_h2h

    with tempfile.TemporaryDirectory() as d:
        new_ckpt = f"{d}/new_r1.pt"
        old_ckpt = f"{d}/banked_r0.pt"
        torch.save({"config": {"rollout": {"ruleset": RULESET_R1}}}, new_ckpt)
        torch.save({"config": {"rollout": {"max_turns": 400}}}, old_ckpt)
        with pytest.raises(CrossRulesetEvalError, match="cross-ruleset h2h refused"):
            cross_arch_h2h(  # NO ruleset= passed: exactly the production call
                new_ckpt=new_ckpt,
                old_ckpt=old_ckpt,
                n_games=2,
                strict_engine_parity=False,
            )


def test_search_eval_refuses_a_cross_epoch_matchup() -> None:
    """Same contract one tier down — the banked +54.6 Elo search rung is an R0
    number and must not silently absorb an R1 checkpoint."""
    import tempfile

    import torch

    from catan_rl.search.config import SearchConfig
    from catan_rl.search.eval_search import evaluate_search_vs_policy

    with tempfile.TemporaryDirectory() as d:
        r1 = f"{d}/search_r1.pt"
        r0 = f"{d}/opp_r0.pt"
        torch.save({"config": {"rollout": {"ruleset": RULESET_R1}}}, r1)
        torch.save({"config": {"rollout": {"max_turns": 400}}}, r0)
        with pytest.raises(CrossRulesetEvalError, match="cross-ruleset search h2h refused"):
            evaluate_search_vs_policy(SearchConfig(sims_per_move=2), r1, r0, n_games=2)


# ---------------------------------------------------------------------------
# Training-config epoch hygiene — the OTHER half of D8's "no silent re-ruling"
# ---------------------------------------------------------------------------


def test_rollout_ruleset_defaults_to_r0_so_banked_lineages_do_not_flip() -> None:
    """Eleven of the twelve train configs carry no ``ruleset:`` key, and
    several exist to RESUME a banked R0 lineage. An ``R1`` dataclass default
    would flip every one of them on a merge — and stamp the next checkpoint
    ``R1``, poisoning the field the whole refusal machinery reads back."""
    from pathlib import Path

    from catan_rl.ppo.arguments import RolloutConfig, TrainConfig

    assert RolloutConfig().ruleset == RULESET_R0
    configs = Path(__file__).resolve().parents[3] / "configs"
    resume = TrainConfig.from_yaml(configs / "selfplay_v8_cont_resume.yaml")
    assert resume.rollout.ruleset == RULESET_R0
    # ppo_default.yaml is the ONE deliberate opt-in (new lineages).
    assert TrainConfig.from_yaml(configs / "ppo_default.yaml").rollout.ruleset == RULESET_R1


def test_ruleset_is_resume_critical() -> None:
    """A resume takes weights from the payload but config from the LIVE yaml,
    so an unnoticed epoch flip continues a lineage under a different legal
    action set at every roll node — strictly more corrupting than ``gamma``,
    which was already on the list."""
    from catan_rl.ppo.training_loop import _RESUME_CRITICAL_FIELDS, _diff_resume_config

    assert ("rollout", "ruleset") in _RESUME_CRITICAL_FIELDS
    diff = _diff_resume_config(
        {"rollout": {"ruleset": RULESET_R0}}, {"rollout": {"ruleset": RULESET_R1}}
    )
    assert any("rollout.ruleset" in line for line in diff)


# ---------------------------------------------------------------------------
# D11 — one non-VP dev card per turn
# ---------------------------------------------------------------------------


def _stub_game(events: list[dict], players: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        broadcast=SimpleNamespace(events=events),
        playerQueue=SimpleNamespace(queue=players),
    )


def _stub_player(name: str, **counters: int) -> SimpleNamespace:
    base = {
        "knightsPlayed": 0,
        "yopPlayed": 0,
        "monopolyPlayed": 0,
        "roadBuilderPlayed": 0,
    }
    base.update(counters)
    return SimpleNamespace(name=name, **base)


def test_two_monopolies_in_one_turn_is_a_violation() -> None:
    events = [
        {"type": "DICE_ROLL", "player": "P1", "value": 8},
        {"type": "MONOPOLY", "player": "P1", "resource": "ORE", "count": 3},
        {"type": "MONOPOLY", "player": "P1", "resource": "WOOD", "count": 1},
    ]
    game = _stub_game(events, [_stub_player("P1", monopolyPlayed=2), _stub_player("P2")])
    with pytest.raises(RulesInvariantViolation, match="in one turn"):
        check_one_dev_card_per_turn(game)


def test_preroll_play_then_own_roll_then_main_play_is_a_violation() -> None:
    """THE D2 bug, in its exact shape — the one this invariant exists for.

    Under R1 a turn is *pre-roll window -> own roll -> main phase*, so the
    player's own DICE_ROLL sits in the MIDDLE of their turn. A segmentation
    that cleared every counter on every roll would put the pre-roll Monopoly
    and the main-phase YoP in different segments and see nothing; the aggregate
    pigeonhole cannot rescue it either, because a single straddle gives
    ``total == turns + 1``, inside the deliberate slack.
    """
    events = [
        {"type": "DICE_ROLL", "player": "P2", "value": 6},  # opponent's turn ends
        {"type": "MONOPOLY", "player": "P1", "resource": "ORE", "count": 3},  # pre-roll
        {"type": "DICE_ROLL", "player": "P1", "value": 8},  # P1's own roll, mid-turn
        {"type": "YOP", "player": "P1", "resources": ["WOOD", "BRICK"]},  # main phase
    ]
    game = _stub_game(
        events,
        [_stub_player("P1", monopolyPlayed=1, yopPlayed=1), _stub_player("P2")],
    )
    # Pigeonhole alone would pass this: total=2, turns=1, slack allows 2.
    with pytest.raises(RulesInvariantViolation, match="in one turn"):
        check_one_dev_card_per_turn(game)


def test_preroll_play_after_the_opponents_roll_belongs_to_the_next_turn() -> None:
    """The mirror of the case above: a main-phase play, then the OPPONENT's
    roll, then a pre-roll play is two different turns and must NOT fire."""
    events = [
        {"type": "DICE_ROLL", "player": "P1", "value": 8},
        {"type": "MONOPOLY", "player": "P1", "resource": "ORE", "count": 3},  # turn N main
        {"type": "DICE_ROLL", "player": "P2", "value": 6},  # P1's turn N is over
        {"type": "YOP", "player": "P1", "resources": ["WOOD", "BRICK"]},  # turn N+1 pre-roll
        {"type": "DICE_ROLL", "player": "P1", "value": 4},
    ]
    game = _stub_game(
        events,
        [_stub_player("P1", monopolyPlayed=1, yopPlayed=1), _stub_player("P2")],
    )
    check_one_dev_card_per_turn(game)


def test_one_play_per_turn_across_turns_is_fine() -> None:
    events = [
        {"type": "DICE_ROLL", "player": "P1", "value": 8},
        {"type": "MONOPOLY", "player": "P1", "resource": "ORE", "count": 3},
        {"type": "DICE_ROLL", "player": "P2", "value": 6},
        {"type": "YOP", "player": "P2", "resources": ["ORE", "WOOD"]},
        {"type": "DICE_ROLL", "player": "P1", "value": 5},
        {"type": "YOP", "player": "P1", "resources": ["WOOD", "WOOD"]},
    ]
    game = _stub_game(
        events,
        [
            _stub_player("P1", monopolyPlayed=1, yopPlayed=1),
            _stub_player("P2", yopPlayed=1),
        ],
    )
    check_one_dev_card_per_turn(game)


def test_aggregate_pigeonhole_catches_silent_knights() -> None:
    """KNIGHT emits no unambiguous event, so only the counter check can see it:
    4 knights across 2 rolled turns exceeds the ``turns + 1`` allowance."""
    events = [
        {"type": "DICE_ROLL", "player": "P1", "value": 8},
        {"type": "DICE_ROLL", "player": "P1", "value": 4},
    ]
    game = _stub_game(events, [_stub_player("P1", knightsPlayed=4), _stub_player("P2")])
    with pytest.raises(RulesInvariantViolation, match="non-VP dev cards across"):
        check_one_dev_card_per_turn(game)


def test_one_card_of_slack_is_allowed_for_a_game_ending_preroll_play() -> None:
    """A pre-roll Knight can win via Largest Army BEFORE that turn's roll is
    ever emitted, so ``plays == turns + 1`` must not be a violation."""
    events = [
        {"type": "DICE_ROLL", "player": "P1", "value": 8},
        {"type": "DICE_ROLL", "player": "P1", "value": 4},
    ]
    game = _stub_game(events, [_stub_player("P1", knightsPlayed=3), _stub_player("P2")])
    check_one_dev_card_per_turn(game)


def test_two_knights_in_one_turn_around_the_roll_is_a_violation() -> None:
    """Regression: mechanism 1 once attributed only YOP/MONOPOLY, so a single
    straddling double-KNIGHT — the reward-positive case D2 names, and 14 of the
    25 dev cards — escaped it entirely and sat inside mechanism 2's ``+1``
    slack. Knight is inferred from the robber instead."""
    events = [
        {"type": "DICE_ROLL", "player": "P2", "value": 5},
        {"type": "MOVE_ROBBER", "player": "P1", "hex_idx": 3},  # pre-roll knight
        {"type": "DICE_ROLL", "player": "P1", "value": 8},
        {"type": "MOVE_ROBBER", "player": "P1", "hex_idx": 4},  # 2nd knight, same turn
        {"type": "DICE_ROLL", "player": "P2", "value": 4},
        {"type": "DICE_ROLL", "player": "P1", "value": 6},
    ]
    game = _stub_game(events, [_stub_player("P1", knightsPlayed=2), _stub_player("P2")])
    with pytest.raises(RulesInvariantViolation, match="in one turn"):
        check_one_dev_card_per_turn(game)


def test_a_rolled_seven_entitles_one_robber_move_that_is_not_a_knight() -> None:
    """The Knight inference must not fire on the ordinary 7-robber move, nor on
    a legal pre-roll Knight followed by the mover's own rolled 7."""
    seven_only = [
        {"type": "DICE_ROLL", "player": "P1", "value": 7},
        {"type": "MOVE_ROBBER", "player": "P1", "hex_idx": 3},
        {"type": "DICE_ROLL", "player": "P2", "value": 5},
    ]
    check_one_dev_card_per_turn(_stub_game(seven_only, [_stub_player("P1"), _stub_player("P2")]))

    preroll_knight_then_own_seven = [
        {"type": "DICE_ROLL", "player": "P2", "value": 5},
        {"type": "MOVE_ROBBER", "player": "P1", "hex_idx": 3},
        {"type": "DICE_ROLL", "player": "P1", "value": 7},
        {"type": "MOVE_ROBBER", "player": "P1", "hex_idx": 4},
        {"type": "DICE_ROLL", "player": "P2", "value": 3},
    ]
    check_one_dev_card_per_turn(
        _stub_game(
            preroll_knight_then_own_seven,
            [_stub_player("P1", knightsPlayed=1), _stub_player("P2")],
        )
    )


def test_a_knight_plus_a_named_card_in_one_turn_is_a_violation() -> None:
    events = [
        {"type": "DICE_ROLL", "player": "P2", "value": 5},
        {"type": "MOVE_ROBBER", "player": "P1", "hex_idx": 3},  # pre-roll knight
        {"type": "DICE_ROLL", "player": "P1", "value": 5},
        {"type": "MONOPOLY", "player": "P1", "resource": "ORE", "count": 1},
        {"type": "DICE_ROLL", "player": "P2", "value": 5},
    ]
    game = _stub_game(
        events,
        [_stub_player("P1", knightsPlayed=1, monopolyPlayed=1), _stub_player("P2")],
    )
    with pytest.raises(RulesInvariantViolation, match="in one turn"):
        check_one_dev_card_per_turn(game)


def test_missing_event_log_is_not_a_violation() -> None:
    check_one_dev_card_per_turn(SimpleNamespace())


def test_a_real_game_has_no_events_attribute_without_the_opt_in() -> None:
    """The engine's ``GameBroadcast`` has ``last_event`` and nothing else.

    This is what made the first cut of the invariant a total no-op on every
    real game: it guarded on ``hasattr(broadcast, "events")`` against a shape
    only the test stubs ever produced. Pin the guard's premise so the two can
    never silently diverge again.
    """
    from catan_rl.engine.broadcast import GameBroadcast
    from catan_rl.env.event_log import attach_event_log

    b = GameBroadcast()
    assert not hasattr(b, "events")
    game = SimpleNamespace(broadcast=b)
    attach_event_log(game)
    assert b.events == []  # type: ignore[attr-defined]
    b.dice_roll("P1", 8)
    assert b.events == [{"type": "DICE_ROLL", "player": "P1", "value": 8}]  # type: ignore[attr-defined]
    # Idempotent — a second attach must not double-subscribe.
    attach_event_log(game)
    b.dice_roll("P2", 6)
    assert len(b.events) == 2  # type: ignore[attr-defined]


def test_audited_env_actually_arms_the_event_stream_checks() -> None:
    """``audit_events=True`` must make the D11 check LIVE, not vacuously green.

    Without a retained stream ``check_one_dev_card_per_turn`` returns silently,
    so ``run_all_invariants(...) == []`` proves nothing. Here the log is real,
    so a forged counter (as a runaway PPO exploit would produce) is caught.
    """
    from catan_rl.env.catan_env import CatanEnv

    env = CatanEnv(opponent_type="heuristic", max_turns=20, ruleset=RULESET_R1, audit_events=True)
    env.reset(seed=1)
    assert env.game is not None
    # The log is armed from game creation (reset stops at the agent's first
    # setup placement, so it is still empty here — which is the point: the
    # check is now LIVE rather than short-circuiting on a missing attribute).
    assert isinstance(env.game.broadcast.events, list)  # type: ignore[attr-defined]
    # Zero rolled turns so far for the agent -> the slack allows exactly 1.
    assert env.agent_player is not None
    env.agent_player.knightsPlayed = 99
    with pytest.raises(RulesInvariantViolation):
        check_one_dev_card_per_turn(env.game)


def test_real_game_passes_the_new_invariant() -> None:
    import random

    from catan_rl.env.catan_env import CatanEnv

    random.seed(0)
    np.random.seed(0)
    env = CatanEnv(opponent_type="heuristic", max_turns=60, ruleset=RULESET_R1, audit_events=True)
    env.reset(seed=0)
    from catan_rl.policy.obs_schema import ActionType

    for _ in range(4000):
        masks = env.get_action_masks()
        legal = np.flatnonzero(masks["type"])
        atype = int(legal[0])
        corner = (
            int(np.flatnonzero(masks["corner_settlement"])[0])
            if masks["corner_settlement"].any()
            else 0
        )
        edge = int(np.flatnonzero(masks["edge"])[0]) if masks["edge"].any() else 0
        tile = int(np.flatnonzero(masks["tile"])[0]) if masks["tile"].any() else 0
        if atype == ActionType.BUILD_CITY and masks["corner_city"].any():
            corner = int(np.flatnonzero(masks["corner_city"])[0])
        res1 = 0
        for key in ("resource1_discard", "resource1_yop", "resource1_default"):
            if masks[key].any():
                res1 = int(np.flatnonzero(masks[key])[0])
                break
        action = np.array([atype, corner, edge, tile, res1, (res1 + 1) % 5], dtype=np.int64)
        _obs, _r, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    # The event log must be non-trivial, or the assertion below is vacuous.
    assert len(env.game.broadcast.events) > 10  # type: ignore[attr-defined]
    assert run_all_invariants(env.game, truncated=True) == []
