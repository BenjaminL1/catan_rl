"""Post-game audit of the 1v1 Colonist.io ruleset.

Run after each eval game (or at training-time intervals) to verify the
engine + env did not silently drift from the 1v1 rules pinned in
``CLAUDE.md``. Any violation surfaces as a raised
:class:`RulesInvariantViolation` so the operator sees it loudly
rather than diagnosing a months-into-training plateau.

Coverage (the 1v1 ruleset checks that have a cheap post-game proxy):

* ``check_max_points_15`` — engine's win condition is 15 VP, not 10.
* ``check_two_players`` — the engine's player queue has exactly 2.
* ``check_winner_actually_at_15_or_truncated`` — terminated games
  end with a winner at >= 15 VP; truncated games end at turn cap.
* ``check_friendly_robber_enabled`` — the board's
  ``get_robber_spots`` filters out hexes adjacent to <3-VP players.
* ``check_no_p2p_trade_events`` — broadcast event stream contains no
  P2P trade messages (only BANK trades).
* ``check_stacked_dice_in_use`` — engine's dice instance is
  ``StackedDice`` (not a vanilla pair-of-dice fallback).

Each check is a free function that takes the ``catanGame`` instance
post-termination and either returns silently or raises
:class:`RulesInvariantViolation`. :func:`run_all_invariants` aggregates
them and reports the full set of violations rather than short-
circuiting on the first.
"""

from __future__ import annotations

from typing import Any


class RulesInvariantViolation(AssertionError):
    """Raised when a 1v1 Colonist.io rule pin fails post-game."""


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_max_points_15(game: Any) -> None:
    """``catanGame.maxPoints`` must be 15 (1v1 Colonist.io ruleset, not
    standard 4-player's 10)."""
    if getattr(game, "maxPoints", None) != 15:
        raise RulesInvariantViolation(
            f"maxPoints expected 15 (1v1 ruleset); got {getattr(game, 'maxPoints', None)}"
        )


def check_two_players(game: Any) -> None:
    """The engine must be configured for exactly 2 players."""
    n = getattr(game, "numPlayers", None)
    if n != 2:
        raise RulesInvariantViolation(f"numPlayers expected 2 (1v1 ruleset); got {n}")
    # Also verify the queue's current snapshot matches.
    queue_size = len(list(game.playerQueue.queue)) if hasattr(game, "playerQueue") else None
    if queue_size is not None and queue_size != 2:
        raise RulesInvariantViolation(f"playerQueue.size expected 2; got {queue_size}")


def check_terminal_state(game: Any, *, truncated: bool) -> None:
    """For terminated games (a real win), the winning player must be at
    >= 15 VP. For truncated games (time cap), no such requirement, but
    neither player should be at < 0 VP (which would mean negative VP
    state leaked in)."""
    players = list(game.playerQueue.queue)
    if truncated:
        # Just check VP non-negative.
        for p in players:
            vp = getattr(p, "victoryPoints", 0)
            if vp < 0:
                raise RulesInvariantViolation(
                    f"truncated game has negative-VP player: {p.name} VP={vp}"
                )
        return
    # Terminated → winner is at >= 15.
    max_vp = max(getattr(p, "victoryPoints", 0) for p in players)
    if max_vp < 15:
        raise RulesInvariantViolation(
            f"terminated game but no player reached 15 VP; max VP={max_vp}"
        )


def check_friendly_robber(game: Any) -> None:
    """The board must filter out robber-spots adjacent to <3-VP players.

    Two-layer probe:

    1. Structural: ``board.get_robber_spots()`` exists and returns
       a container with at most 19 entries.
    2. Behavioural: for every player whose *visible* VP (``victoryPoints``
       minus hidden VP dev cards) is < 3, walk their settlements + cities
       and assert none of the adjacent hexes are in ``get_robber_spots()``.
       A regression that silently removed the filter would let those
       hexes leak through.
    """
    board = getattr(game, "board", None)
    if board is None or not hasattr(board, "get_robber_spots"):
        raise RulesInvariantViolation(
            "board.get_robber_spots missing — Friendly Robber rule not implemented"
        )
    spots = board.get_robber_spots()
    if not isinstance(spots, list | tuple | set | dict):
        raise RulesInvariantViolation(
            f"get_robber_spots returned unexpected type: {type(spots).__name__}"
        )
    if len(spots) > 19:
        raise RulesInvariantViolation(f"get_robber_spots returned >{19} entries; got {len(spots)}")

    spots_idx: set[Any] = set(spots.keys()) if isinstance(spots, dict) else set(spots)

    queue = getattr(game, "playerQueue", None)
    if queue is None or not hasattr(queue, "queue"):
        # Engine missing the standard queue field — structural check
        # already gated us above; nothing else we can probe here.
        return

    for p in list(queue.queue):
        vp = int(getattr(p, "victoryPoints", 0))
        hidden_vp = (
            int(p.devCards.get("VP", 0))
            if hasattr(p, "devCards") and isinstance(p.devCards, dict)
            else 0
        )
        visible_vp = vp - hidden_vp
        if visible_vp >= 3:
            continue
        # Protected player. Their settlement and city hexes must NOT
        # appear in the robber-spot list.
        build_graph = getattr(p, "buildGraph", None)
        if not isinstance(build_graph, dict):
            continue
        protected_vertices = list(build_graph.get("SETTLEMENTS", [])) + list(
            build_graph.get("CITIES", [])
        )
        board_graph = getattr(board, "boardGraph", None)
        if not isinstance(board_graph, dict):
            continue
        for v in protected_vertices:
            v_obj = board_graph.get(v)
            if v_obj is None:
                continue
            adj_hexes = getattr(v_obj, "adjacent_hex_indices", []) or []
            for h_idx in adj_hexes:
                if h_idx in spots_idx:
                    raise RulesInvariantViolation(
                        f"Friendly Robber filter leak: hex {h_idx} "
                        f"adjacent to player {getattr(p, 'name', '?')} "
                        f"(visible VP={visible_vp} < 3) is in "
                        "get_robber_spots()"
                    )


def check_no_p2p_trade(game: Any) -> None:
    """Inspect the broadcast event stream for any P2P trade event.

    1v1 Colonist.io disables player-to-player trading
    (``initiate_trade`` early-returns on non-BANK in
    ``catan/engine/player.py``). A P2P trade event in the stream means
    either a leak in the player API or a third-party patch.
    """
    broadcast = getattr(game, "broadcast", None)
    if broadcast is None or not hasattr(broadcast, "events"):
        # No event log → cannot verify. Treated as a warning, not a
        # violation — older test envs may not retain the log.
        return
    for event in broadcast.events:
        kind = event.get("type", "") if isinstance(event, dict) else ""
        if "P2P" in kind or "PLAYER_TRADE" in kind:
            raise RulesInvariantViolation(f"P2P trade event in broadcast log: {event}")


def check_stacked_dice(game: Any) -> None:
    """The engine must be using ``StackedDice`` (the Colonist.io
    36-outcome bag + 1 noise swap + 20% Karma-7), not a vanilla 2d6
    pair-of-dice fallback."""
    from catan_rl.engine.dice import StackedDice

    if not isinstance(getattr(game, "dice", None), StackedDice):
        raise RulesInvariantViolation(
            f"game.dice is not StackedDice — Colonist.io ruleset broken; "
            f"got {type(getattr(game, 'dice', None)).__name__}"
        )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def run_all_invariants(game: Any, *, truncated: bool = False) -> list[str]:
    """Run every check and return a list of violation messages.

    Empty list = all invariants passed. Callers may either re-raise via
    ``raise RulesInvariantViolation("\n".join(msgs))`` or log them.

    Aggregating rather than short-circuiting lets the operator see the
    full failure surface (e.g. if maxPoints AND p2p trade both leak in
    the same buggy refactor).
    """
    violations: list[str] = []

    def _check(fn: Any, *args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except RulesInvariantViolation as e:
            violations.append(str(e))

    _check(check_max_points_15, game)
    _check(check_two_players, game)
    _check(check_terminal_state, game, truncated=truncated)
    _check(check_friendly_robber, game)
    _check(check_no_p2p_trade, game)
    _check(check_stacked_dice, game)
    return violations
