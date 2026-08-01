"""BC generator turn-boundary ordering — spec ``preroll-dev-cards-r1`` D2.

``updateDevCards()`` / ``devCardPlayedThisTurn = False`` used to run BELOW the
``"roll"`` ``_record_decision``, so every recorded roll row carried a STALE dev
hand: a card bought on the previous turn was still sitting in ``newDevCards``
(invisible to the obs' dev counts) and the previous turn's "already played"
flag was still set.
"""

from __future__ import annotations

from typing import Any

import catan_rl.bc.dataset as dataset_mod


def test_roll_rows_are_recorded_after_the_turn_boundary(monkeypatch: Any) -> None:
    observed: list[tuple[int, bool]] = []
    original = dataset_mod._record_decision

    def _spy(ctx: Any, action: Any, phase: str, mask_overrides: Any = None) -> None:
        if phase == "roll":
            p = ctx.agent_player
            observed.append((len(p.newDevCards), bool(p.devCardPlayedThisTurn)))
        return original(ctx, action, phase, mask_overrides)

    monkeypatch.setattr(dataset_mod, "_record_decision", _spy)
    dataset_mod.play_game(game_id=0, seed=3, max_turns=80)

    assert observed, "no roll decisions were recorded"
    assert all(n_new == 0 for n_new, _ in observed), (
        "a roll row was recorded while cards were still pending in newDevCards "
        "— updateDevCards() has drifted back below the roll record"
    )
    assert all(played is False for _, played in observed), (
        "a roll row was recorded with last turn's devCardPlayedThisTurn still set"
    )
