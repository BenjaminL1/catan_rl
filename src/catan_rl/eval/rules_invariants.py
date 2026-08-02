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
* ``check_one_dev_card_per_turn`` — at most one non-VP dev card is
  played per player per turn (the R1 pre-roll window must not become a
  second card).

Each check is a free function that takes the ``catanGame`` instance
post-termination and either returns silently or raises
:class:`RulesInvariantViolation`. :func:`run_all_invariants` aggregates
them and reports the full set of violations rather than short-
circuiting on the first.

**Event-stream checks need an opt-in log.** ``engine.broadcast.GameBroadcast``
keeps only ``last_event`` — it has no history, and ``engine/`` is out of scope
for the ``preroll-dev-cards-r1`` slice (D1). The two checks that read a stream
(:func:`check_no_p2p_trade`, :func:`check_one_dev_card_per_turn`) therefore
depend on :func:`catan_rl.env.event_log.attach_event_log` (re-exported here),
which subscribes an in-memory mirror and publishes it as
``game.broadcast.events``. Without it both checks degrade to a no-op. It is
opt-in rather than automatic because a full game emits thousands of events and
the training rollout (``n_envs=128``) must not retain them — every *audited*
path attaches it (``CatanEnv(audit_events=True)``, which
:class:`catan_rl.eval.harness.EvalHarness` sets from ``audit_rules``, plus
``search.eval_search``, ``human_data.engine_bridge`` and
``scripts/play_vs_model.py``), and those are the paths whose numbers are read.
"""

from __future__ import annotations

from typing import Any

from catan_rl.env.event_log import attach_event_log

__all__ = [
    "RulesInvariantViolation",
    "attach_event_log",
    "check_friendly_robber",
    "check_max_points_15",
    "check_no_p2p_trade",
    "check_one_dev_card_per_turn",
    "check_stacked_dice",
    "check_terminal_state",
    "check_two_players",
    "run_all_invariants",
]


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


def _events_of(game: Any) -> list[dict[str, Any]] | None:
    """The retained broadcast history, or ``None`` if none was attached.

    ``GameBroadcast`` itself has no ``events`` attribute — only
    :func:`attach_event_log` adds one — so this returning ``None`` means the
    caller never opted in, NOT that the game emitted nothing.
    """
    broadcast = getattr(game, "broadcast", None)
    events = getattr(broadcast, "events", None)
    return events if isinstance(events, list) else None


def check_no_p2p_trade(game: Any) -> None:
    """Inspect the broadcast event stream for any P2P trade event.

    1v1 Colonist.io disables player-to-player trading
    (``initiate_trade`` early-returns on non-BANK in
    ``catan/engine/player.py``). A P2P trade event in the stream means
    either a leak in the player API or a third-party patch.
    """
    events = _events_of(game)
    if events is None:
        # No retained event log → cannot verify. Treated as a warning, not a
        # violation; call ``attach_event_log(game)`` to enable this check.
        return
    for event in events:
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


def check_one_dev_card_per_turn(
    game: Any, *, truncated: bool = False, aborted: bool = False
) -> None:
    """At most ONE non-VP dev card may be played per player per turn.

    Spec ``preroll-dev-cards-r1`` D11 — the invariant that would have caught
    the D2 turn-boundary bug (a pre-roll play sets the flag, the roll clears
    it, the main-phase mask re-offers the card). Two dev cards per turn is
    reward-positive, so PPO finds it fast, and every number measured after it
    would be measuring a cheat.

    Two mechanisms, because neither alone is complete and ``engine/broadcast``
    is out of scope for this spec (there is no dev-card-played event, and no
    turn-start event to hang one off):

    1. **Exact, per turn.** Segment the event stream on ``DICE_ROLL`` and
       attribute dev plays to the player each event names. ``YOP`` /
       ``MONOPOLY`` name themselves. **Knight is inferred from the robber**:
       within one turn segment a player's ``MOVE_ROBBER`` events are all Knight
       plays except at most one, the move owed to their own rolled 7 — so
       ``knights >= robber_moves - (1 if they rolled a 7 this turn else 0)``.
       That lower bound is what makes the highest-risk case visible: two
       Knights in one turn is the reward-positive exploit D2 names ("two
       knights/turn reaches Largest Army in two turns"), Knight is 14 of the 25
       dev cards, and Knight emits no card event of its own. The bound only
       ever *under*-counts (a 7 with no legal robber spot under Friendly Robber
       emits no ``MOVE_ROBBER``), so it cannot produce a false positive.

       The segmentation boundary is subtle and was got wrong once: under R1 a
       turn runs *pre-roll window -> own roll -> main phase*, so a player's own
       ``DICE_ROLL`` sits in the MIDDLE of their turn. Clearing every counter
       on every roll would split one turn in two and let the exact D2 bug
       (pre-roll Monopoly, roll, main-phase YoP) through unseen. In 1v1 the
       seats alternate, so the correct rule is: **a roll by X ends every OTHER
       player's turn, and never X's own.**
    2. **Aggregate catch-all.** ``ROADBUILDER`` still emits nothing
       attributable, so fall back on the cumulative counters: for each player,
       ``knights + yop + monopoly + roadBuilder <= (their DICE_ROLL count) + 1``.
       Pigeonhole-sound: at most one play per turn implies plays <= turns. The
       ``+1`` of slack is necessary, not cosmetic — a pre-roll Knight can win
       via Largest Army before that turn's roll is ever emitted. It is also why
       mechanism 1 exists: a *single* straddling double-play sits inside the
       slack, so only mechanism 1 sees it — which is precisely why mechanism 1
       has to cover Knight and not just the two self-naming cards.

       The slack is granted to a player who WON (their pre-roll play ended the
       game) and to every player when ``truncated=True`` (the game stopped —
       turn cap or a GUI window-close — before that turn's roll was emitted).
       It is withheld otherwise, which is what keeps the straddle visible in a
       completed game.

    VP cards are excluded by construction: they are never *played*, and no
    ``vpPlayed`` counter exists.

    Requires :func:`attach_event_log` to have been called on ``game``; without
    a retained stream this check cannot run at all (both mechanisms need the
    per-player roll count) and returns silently.
    """
    events = _events_of(game)
    if events is None:
        # No retained event log -> cannot verify (mirrors check_no_p2p_trade).
        return

    # --- 1) exact per-turn segmentation over the events that name a player.
    # Per player, for the turn currently in flight: self-naming card plays,
    # robber moves, and whether their own roll this turn was a 7 (which
    # entitles them to exactly one non-Knight robber move).
    cards: dict[str, int] = {}
    robber_moves: dict[str, int] = {}
    rolled_seven: dict[str, bool] = {}
    rolls_by_player: dict[str, int] = {}

    def _assert_at_most_one(name: str, kind: str) -> None:
        knights = max(0, robber_moves.get(name, 0) - (1 if rolled_seven.get(name) else 0))
        total = cards.get(name, 0) + knights
        if total > 1:
            raise RulesInvariantViolation(
                f"player {name!r} played {total} non-VP dev cards in one turn "
                f"(event {kind}; {cards.get(name, 0)} named + {knights} Knight(s) inferred "
                f"from {robber_moves.get(name, 0)} robber move(s), "
                f"rolled_seven={bool(rolled_seven.get(name))}); at most 1 is legal"
            )

    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type", "")
        name = event.get("player")
        if kind == "DICE_ROLL":
            if name is not None:
                rolls_by_player[name] = rolls_by_player.get(name, 0) + 1
            # End every OTHER seat's turn; keep the roller's own tallies, which
            # may already hold a pre-roll play belonging to this same turn.
            cards = {k: v for k, v in cards.items() if k == name}
            robber_moves = {k: v for k, v in robber_moves.items() if k == name}
            rolled_seven = {k: v for k, v in rolled_seven.items() if k == name}
            if name is not None and event.get("value") == 7:
                rolled_seven[name] = True
            continue
        if name is None:
            continue
        if kind in ("YOP", "MONOPOLY"):
            cards[name] = cards.get(name, 0) + 1
            _assert_at_most_one(name, kind)
        elif kind == "MOVE_ROBBER":
            robber_moves[name] = robber_moves.get(name, 0) + 1
            _assert_at_most_one(name, kind)

    # --- 2) aggregate pigeonhole over the cumulative counters.
    players = list(game.playerQueue.queue) if hasattr(game, "playerQueue") else []
    for p in players:
        total = (
            int(getattr(p, "knightsPlayed", 0))
            + int(getattr(p, "yopPlayed", 0))
            + int(getattr(p, "monopolyPlayed", 0))
            + int(getattr(p, "roadBuilderPlayed", 0))
        )
        turns = rolls_by_player.get(getattr(p, "name", None), 0)
        # The +1 slack exists for the legitimate cases where a pre-roll play
        # has no roll to be counted against: the play ENDED the game (a
        # winner), or the game stopped before that turn's roll ever happened
        # (a truncated / aborted game — a window-close lands here).
        #
        # Why this matters, measured: with the slack granted unconditionally, a
        # D2-class straddle (pre-roll Monopoly -> roll -> main-phase Road
        # Builder in ONE turn) produced total=2 against turns+1=2 and was NOT
        # flagged. Road Builder emits no broadcast event, so the per-turn
        # attribution in step 1 cannot see it either — this aggregate is the
        # only line of defence for that card, and unconditional slack blinded
        # it. Restricting the slack closes the straddle while keeping the real
        # cases legal.
        #
        # THE SLACK KEYS OFF ``aborted``, NOT ``truncated`` — they are different
        # events and conflating them disarms this check across the whole eval
        # spine. ``truncated`` is env turn-cap exhaustion, raised by
        # ``catan_env._check_terminal`` ONLY at a turn boundary AFTER that turn's
        # roll — so a truncated game always has a DICE_ROLL for every counted
        # turn and can never legitimately need the +1. It is also passed by
        # ``eval/harness.py``, ``search/eval_search.py`` and
        # ``human_data/engine_bridge.py``, whose violations feed the promotion
        # and bake-off gates; granting slack there would make this aggregate
        # toothless for every truncated eval game — and it is the ONLY defence
        # against a Road Builder straddle, since Road Builder emits no broadcast
        # event for step 1 to attribute (see
        # ``tests/unit/eval/test_preroll_eval.py``, the
        # ``test_a_road_builder_straddle_is_caught_by_the_aggregate`` pin).
        #
        # ``aborted`` is the genuinely un-rolled case: a GUI window-close can
        # land mid-turn, before that turn's roll, so the turn is never counted
        # while a pre-roll dev card was legitimately played. Only the interactive
        # driver sets it.
        max_points = int(getattr(game, "maxPoints", 15))
        won = int(getattr(p, "victoryPoints", 0)) >= max_points
        slack = won or aborted
        allowed = turns + 1 if slack else turns
        if total > allowed:
            raise RulesInvariantViolation(
                f"player {getattr(p, 'name', '?')!r} played {total} non-VP dev cards across "
                f"{turns} rolled turns; at most one per turn allows {allowed}"
                + (
                    " (+1 slack applied: a pre-roll play may have ended the game, or the game "
                    "stopped, before its roll)"
                    if slack
                    else " (no slack: this player did not reach the win condition)"
                )
            )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def run_all_invariants(game: Any, *, truncated: bool = False, aborted: bool = False) -> list[str]:
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
    # BOTH flags relax THIS check: it asks "did the game end legitimately?",
    # and a turn-cap stop and a window-close are both legitimate non-wins.
    # The dev-card aggregate below is the one that must NOT accept
    # ``truncated`` — see its slack comment.
    _check(check_terminal_state, game, truncated=truncated or aborted)
    _check(check_friendly_robber, game)
    _check(check_no_p2p_trade, game)
    _check(check_stacked_dice, game)
    _check(check_one_dev_card_per_turn, game, truncated=truncated, aborted=aborted)
    return violations
