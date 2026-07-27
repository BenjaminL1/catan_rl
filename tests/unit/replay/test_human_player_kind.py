"""``"human"`` is a legal RECORD value but never a constructible actor.

Labelling the person ``"heuristic"`` to squeeze past the old enum would poison
every downstream consumer, so both construction paths must fail loudly instead.
"""

from __future__ import annotations

import pytest

from catan_rl.replay.player_factory import (
    HumanPlayerNotActorError,
    PlayerSpec,
    build_actor,
)


def test_build_actor_rejects_human_with_a_typed_error() -> None:
    with pytest.raises(HumanPlayerNotActorError, match="no actor to build"):
        build_actor(PlayerSpec(kind="human"), seed=0)


def test_the_error_is_a_valueerror_subclass() -> None:
    # Existing callers catch ValueError from build_actor; keep them working.
    assert issubclass(HumanPlayerNotActorError, ValueError)


@pytest.mark.parametrize("seat", [0, 1])
def test_record_game_refuses_to_host_a_human_seat(seat: int) -> None:
    from catan_rl.replay.recorder_loop import _resolve_seat_and_opp

    specs = [PlayerSpec(kind="policy", ckpt_path="x"), PlayerSpec(kind="heuristic")]
    specs[seat] = PlayerSpec(kind="human")
    with pytest.raises(HumanPlayerNotActorError, match="cannot host a human seat"):
        _resolve_seat_and_opp(specs[0], specs[1])


def test_the_other_kinds_still_build() -> None:
    for kind in ("random", "heuristic"):
        actor = build_actor(PlayerSpec(kind=kind), seed=0)  # type: ignore[arg-type]
        assert actor.kind == kind
