"""Opt-in retained history for a game's broadcast stream.

``engine.broadcast.GameBroadcast`` keeps only ``last_event`` — it has no
history — and ``src/catan_rl/engine/`` is out of scope for the
``preroll-dev-cards-r1`` slice (D1; it is also the tree pinned by
``eval/engine_parity.py``, so even adding a file there would be a re-pin). The
post-game rule audit (:mod:`catan_rl.eval.rules_invariants`) needs a stream to
check ``check_no_p2p_trade`` and ``check_one_dev_card_per_turn``, so the log is
bolted on from OUTSIDE the engine: subscribe a list-appending callback and
publish the backing list as ``game.broadcast.events``.

It lives in ``env/`` rather than ``eval/`` so :class:`catan_rl.env.CatanEnv`
can attach it at game-creation time without an ``env -> eval`` import edge.

**Opt-in on purpose.** A full game emits thousands of events; retaining them
across ``n_envs=128`` training envs is pure waste. Only audited paths turn it
on (``CatanEnv(audit_events=True)``, which ``EvalHarness`` sets from its own
``audit_rules`` flag, plus the search / human-data / playtest harnesses).
"""

from __future__ import annotations

from typing import Any


class _EventLog:
    """Append-only mirror of a broadcast stream, usable as a subscriber."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def __deepcopy__(self, memo: dict[int, Any]) -> _EventLog:
        """A clone starts with an EMPTY history. Load-bearing for search.

        ``search/mcts.py`` ``clone_env`` deepcopies the whole env on every
        determinization root AND every expansion, so a retained log would be
        copied thousands of times per move — and the cost GROWS WITHIN a game as
        the log fills. Measured on this branch before this override: 1.99 ms ->
        5.10 ms at 1053 events (2.6x), and 2.02 / 4.40 / 5.17 / 7.40 ms at
        0 / 500 / 1500 / 2600 events (a full R1 game emits ~2.6k). Under
        ``SearchConfig.time_budget_s`` that silently buys fewer simulations
        late-game than early — a deepening regression that would sail past a
        sims/s gate, on the exact surface the banked search uplift was measured.

        Dropping the history is CORRECT, not merely cheap: the audit is a
        post-game check on the real game's stream. A simulated line inside the
        tree was never played, so its events are not audit input, and nothing
        reads ``broadcast.events`` during search.
        """
        return _EventLog()


def attach_event_log(game: Any) -> list[dict[str, Any]]:
    """Give ``game`` a retained broadcast history and return it.

    Subscribes an :class:`_EventLog` to ``game.broadcast`` and publishes the
    backing list as ``game.broadcast.events`` — the attribute the event-stream
    invariants read. Idempotent: a second call on the same game returns the
    list already attached rather than double-subscribing (which would duplicate
    every event and corrupt the ``DICE_ROLL`` turn segmentation).

    Attach as early as possible — before any dice are rolled. The per-turn
    dev-card invariant counts a player's ``DICE_ROLL`` events as their turn
    count, so a log started mid-game undercounts turns and can false-positive.

    Returns the (live) event list, or an empty throwaway list when ``game`` has
    no broadcast at all.
    """
    broadcast = getattr(game, "broadcast", None)
    if broadcast is None:
        return []
    existing = getattr(broadcast, "events", None)
    if isinstance(existing, list):
        return existing
    log = _EventLog()
    broadcast.subscribe(log)
    broadcast.events = log.events
    return log.events
