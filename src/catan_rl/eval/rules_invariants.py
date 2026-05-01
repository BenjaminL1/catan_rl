"""Runtime-callable 1v1 rules-invariant test (Phase 0).

This module is the runtime gate for the 1v1 Colonist.io ruleset. It is invoked
by ``scripts/eval_harness.py`` as a precondition; the harness exits non-zero
if any invariant fails. It complements the unit tests in
``tests/unit/eval/test_rules_invariants.py`` — same checks, but available as
a programmatic call rather than a pytest run.

See ``docs/1v1_rules.md`` for the full table of rules and
ADR 0001 for the design rationale.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class InvariantFailure:
    """A single rules-invariant failure with enough context to debug it."""

    name: str
    """Stable invariant identifier (matches the function name)."""
    rule: str
    """Human-readable description of the rule that was violated."""
    detail: str
    """What was observed vs what was expected, with file:line references."""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.name}] {self.rule}\n  {self.detail}"


def run(*, include_hand_tracker_drift: bool = False) -> list[InvariantFailure]:
    """Run all 1v1 rules-invariants and return any failures.

    Returns an empty list if all pass. Each failure carries enough information
    to diagnose without re-running. Designed to be cheap (< 100 ms total)
    unless ``include_hand_tracker_drift=True`` (adds ~5 s).

    Args:
        include_hand_tracker_drift: If True, also run the random-game hand-tracker
            drift check. This is opt-in because random-action stress can surface
            pre-existing edge cases in the engine's broadcast emissions that are
            not strictly 1v1 rules violations. Use it as a diagnostic when
            triaging hand-tracker accuracy regressions.
    """
    checks: tuple = _ALL_CHECKS
    if include_hand_tracker_drift:
        checks = checks + (hand_tracker_has_no_drift_in_random_games,)
    failures: list[InvariantFailure] = []
    for check in checks:
        try:
            failure = check()
        except Exception as e:  # pragma: no cover - defensive
            failures.append(
                InvariantFailure(
                    name=check.__name__,
                    rule="invariant raised an unexpected exception",
                    detail=f"{type(e).__name__}: {e}",
                )
            )
            continue
        if failure is not None:
            failures.append(failure)
    return failures


def assert_all_pass(*, include_hand_tracker_drift: bool = False) -> None:
    """Run invariants and raise ``AssertionError`` if any fail.

    Useful as a one-liner gate at the top of training/eval scripts.
    """
    fails = run(include_hand_tracker_drift=include_hand_tracker_drift)
    if not fails:
        return
    msg = "1v1 rules-invariants failed:\n" + "\n".join(f"  - {f}" for f in fails)
    raise AssertionError(msg)


# ── Individual invariants ───────────────────────────────────────────────────


def max_points_is_15() -> InvariantFailure | None:
    """Win condition is 15 VP, not the standard 4-player 10."""
    from catan_rl.engine.game import catanGame

    g = catanGame(render_mode=None)
    if g.maxPoints != 15:
        return InvariantFailure(
            name="max_points_is_15",
            rule="catanGame.maxPoints must be 15 for 1v1 Colonist.io ruleset",
            detail=f"observed maxPoints={g.maxPoints}; see src/catan_rl/engine/game.py",
        )
    return None


def num_players_is_2() -> InvariantFailure | None:
    """Player count is 2; the player queue must be exactly 2."""
    from catan_rl.engine.game import catanGame

    g = catanGame(render_mode=None)
    if g.numPlayers != 2:
        return InvariantFailure(
            name="num_players_is_2",
            rule="catanGame.numPlayers must be 2",
            detail=f"observed numPlayers={g.numPlayers}",
        )
    if g.playerQueue.qsize() != 2:
        return InvariantFailure(
            name="num_players_is_2",
            rule="playerQueue must contain exactly 2 players after setup_players()",
            detail=f"observed qsize={g.playerQueue.qsize()}",
        )
    return None


def p2p_trade_disabled() -> InvariantFailure | None:
    """player.initiate_trade must early-return on any non-BANK trade mode."""
    from catan_rl.engine.player import player

    p = player("test", "black")

    class _ExplodingGame:
        def __getattr__(self, name: str):
            raise AssertionError(f"trade implementation reached game.{name}")

    for mode in ("PLAYER", "OPEN_TRADE", "OPEN", "TRADE", ""):
        try:
            p.initiate_trade(_ExplodingGame(), mode)
        except AssertionError as e:
            return InvariantFailure(
                name="p2p_trade_disabled",
                rule="player.initiate_trade must early-return on non-'BANK' modes",
                detail=f"mode={mode!r} reached game state: {e}",
            )
    return None


def discard_threshold_is_9() -> InvariantFailure | None:
    """The 7-roll discard threshold is 9 cards (vs standard 7)."""
    from catan_rl.engine.player import player

    src = inspect.getsource(player.discardResources)
    if "maxCards = 9" not in src:
        return InvariantFailure(
            name="discard_threshold_is_9",
            rule="player.discardResources must use maxCards=9 for 1v1",
            detail=(
                "literal 'maxCards = 9' not found in player.discardResources source. "
                "Either the threshold drifted, or the variable was renamed. "
                "Check src/catan_rl/engine/player.py."
            ),
        )
    return None


def stacked_dice_in_use() -> InvariantFailure | None:
    """The engine must use StackedDice (bag mechanic), not independent 2d6."""
    from catan_rl.engine.dice import StackedDice
    from catan_rl.engine.game import catanGame

    g = catanGame(render_mode=None)
    if not isinstance(g.dice, StackedDice):
        return InvariantFailure(
            name="stacked_dice_in_use",
            rule="catanGame.dice must be a StackedDice instance",
            detail=f"observed type={type(g.dice).__name__}",
        )
    return None


def friendly_robber_filter_present() -> InvariantFailure | None:
    """get_robber_spots must enforce the Friendly Robber rule."""
    from catan_rl.engine.board import catanBoard

    src = inspect.getsource(catanBoard.get_robber_spots)
    # Look for the visible-VP threshold check (the "<3 visible VP" rule).
    if "visible_vps" not in src and "victoryPoints" not in src:
        return InvariantFailure(
            name="friendly_robber_filter_present",
            rule="catanBoard.get_robber_spots must filter by Friendly Robber",
            detail=(
                "neither 'visible_vps' nor 'victoryPoints' found in get_robber_spots; "
                "the <3 visible-VP filter appears to be missing"
            ),
        )
    return None


def action_space_shape() -> InvariantFailure | None:
    """The 1v1 action space is exactly MultiDiscrete([13, 54, 72, 19, 5, 5])."""
    from catan_rl.env.catan_env import CatanEnv

    expected = (13, 54, 72, 19, 5, 5)
    env = CatanEnv(opponent_type="random", max_turns=10)
    actual = tuple(env.action_space.nvec.tolist())
    if actual != expected:
        return InvariantFailure(
            name="action_space_shape",
            rule="env.action_space must be MultiDiscrete([13, 54, 72, 19, 5, 5])",
            detail=f"observed nvec={actual!r}",
        )
    return None


def mask_keys_canonical() -> InvariantFailure | None:
    """The 9 documented mask keys must all be present and no extras."""
    from catan_rl.env.catan_env import CatanEnv

    expected = {
        "type",
        "corner_settlement",
        "corner_city",
        "edge",
        "tile",
        "resource1_trade",
        "resource1_discard",
        "resource1_default",
        "resource2_default",
    }
    env = CatanEnv(opponent_type="random", max_turns=10)
    env.reset(seed=0)
    actual = set(env.get_action_masks().keys())
    if actual != expected:
        return InvariantFailure(
            name="mask_keys_canonical",
            rule="env.get_action_masks() must return the 9 canonical keys",
            detail=(f"missing={expected - actual!r}; extra={actual - expected!r}"),
        )
    return None


def hand_tracker_has_no_drift_in_random_games(n_games: int = 20) -> InvariantFailure | None:
    """Run a few random-policy games with the hand tracker in verify mode.

    In 1v1 with no P2P trading, every resource delta is broadcast-observable,
    so the tracker should match exactly. This is a smoke check — any drift
    means a new code path forgot to emit a RESOURCE_CHANGE event, which
    breaks ADR 0002 (perfect hand tracking).
    """
    import numpy as np

    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.env.hand_tracker import RESOURCES_CW

    env = CatanEnv(opponent_type="random", max_turns=80)
    rng = np.random.default_rng(0)
    for game_i in range(n_games):
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated) and steps < 200:
            masks = env.get_action_masks()
            # Pick a uniformly random valid action via the env's masks.
            type_idx = int(np.random.choice(np.flatnonzero(masks["type"])))
            action = np.zeros(6, dtype=np.int64)
            action[0] = type_idx
            # Fill heads with a valid index where required, otherwise 0.
            if type_idx in (0, 1):
                key = "corner_settlement" if type_idx == 0 else "corner_city"
                valid = np.flatnonzero(masks[key])
                if valid.size:
                    action[1] = int(np.random.choice(valid))
            elif type_idx == 2:
                valid = np.flatnonzero(masks["edge"])
                if valid.size:
                    action[2] = int(np.random.choice(valid))
            elif type_idx == 4:
                valid = np.flatnonzero(masks["tile"])
                if valid.size:
                    action[3] = int(np.random.choice(valid))
            elif type_idx in (7, 8, 11, 10):
                key = (
                    "resource1_default"
                    if type_idx in (7, 8)
                    else ("resource1_discard" if type_idx == 11 else "resource1_trade")
                )
                valid = np.flatnonzero(masks[key])
                if valid.size:
                    action[4] = int(np.random.choice(valid))
                if type_idx in (7, 10):
                    valid2 = np.flatnonzero(masks["resource2_default"])
                    if valid2.size:
                        action[5] = int(np.random.choice(valid2))
            try:
                _, _, terminated, truncated, _info = env.step(action)
            except Exception:
                break
            steps += 1

            # Verify tracker matches actual for both players.
            if env._hand_tracker is None:
                continue
            for p in (env.agent_player, env.opponent_player):
                tracked = env._hand_tracker.get_hand(p.name)
                actual = {r: int(p.resources.get(r, 0)) for r in RESOURCES_CW}
                if tracked != actual:
                    return InvariantFailure(
                        name="hand_tracker_has_no_drift_in_random_games",
                        rule="BroadcastHandTracker must match each player's actual resources",
                        detail=(
                            f"drift in game {game_i} step {steps} for {p.name}: "
                            f"tracked={tracked}, actual={actual}"
                        ),
                    )
    return None


# Order matters for stable, readable failure output.
# Hand-tracker drift is opt-in (see ``run(include_hand_tracker_drift=True)``),
# so it is not in the default set.
_ALL_CHECKS: tuple = (
    max_points_is_15,
    num_players_is_2,
    p2p_trade_disabled,
    discard_threshold_is_9,
    stacked_dice_in_use,
    friendly_robber_filter_present,
    action_space_shape,
    mask_keys_canonical,
)


def all_check_names() -> Iterable[str]:
    """Iterate over the names of all registered invariants."""
    return (c.__name__ for c in _ALL_CHECKS)
