"""End-to-end smoke for the human-game recorder in ``scripts/play_vs_model.py``.

Headless: the harness's ``_auto_human`` path (the same one ``--self-test``
uses) plays the human seat with legal moves, so a whole game runs with no
display. Asserts the emitted ``Replay`` loads strictly, records BOTH seats, and
carries the policy's per-decision internals on the bot's steps.

Run for BOTH seatings. ``bot_seat=1`` (the DEFAULT interactive configuration,
since ``--human-seat`` defaults to 0) is the fragile path: it depends on
``setup_steps_seat_1`` reconstructing the human's reset-time placement and on
the ``split_at_setup_complete`` residual block recovering the human's first main
turn from inside ``env.step`` #3.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "play_vs_model.py"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("play_vs_model_smoke", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["play_vs_model_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", params=[0, 1], ids=["bot_seat0", "bot_seat1"])
def recorded(request, tmp_path_factory):  # type: ignore[no-untyped-def]
    import torch

    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.replay.io import load_replay, save_replay
    from catan_rl.replay.player_factory import _PolicyActor

    mod = _load_module()
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy.eval()
    agent = mod._RawPolicyAgent(
        _PolicyActor(kind="policy", ckpt_path="<fresh>", policy=policy, device=torch.device("cpu"))
    )

    env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=40)
    env._auto_human = True  # no GUI: the human seat auto-plays legal moves
    bot_seat = int(request.param)
    env.reset(seed=11, options={"agent_seat": bot_seat})
    recorder = mod._HumanGameRecorder(env, bot_seat=bot_seat)

    terminated = truncated = False
    steps = 0
    while not terminated and not truncated and steps < env.max_turns * 20:
        action = agent.choose_action(env)
        _obs, _r, terminated, truncated, _info = env.step(action)
        recorder.after_env_step(
            action, agent.last_internals, terminated=terminated, truncated=truncated
        )
        steps += 1

    replay = recorder.finish(
        ckpt="<fresh>",
        seed=11,
        mode="raw_policy",
        sims=None,
        clairvoyant=False,
        reveal_bot=False,
    )
    dest = tmp_path_factory.mktemp("replays") / "game.json"
    save_replay(replay, dest)
    return load_replay(dest, strict=True), bot_seat, recorder


def test_replay_loads_strictly_and_has_setup_plus_main(recorded) -> None:  # type: ignore[no-untyped-def]
    replay, _bot_seat, _rec = recorded
    kinds = {s.kind for s in replay.steps}
    assert "setup" in kinds and "main" in kinds
    assert len([s for s in replay.steps if s.kind == "setup"]) == 4
    assert replay.metadata.total_steps == len(replay.steps)
    assert [s.step_idx for s in replay.steps] == list(range(len(replay.steps)))


def test_both_seats_appear_in_the_record(recorded) -> None:  # type: ignore[no-untyped-def]
    replay, _bot_seat, _rec = recorded
    actors = {s.actor for s in replay.steps}
    assert actors == {"player_a", "player_b"}


def test_the_human_seat_is_recorded_as_human(recorded) -> None:  # type: ignore[no-untyped-def]
    replay, bot_seat, _rec = recorded
    kinds = {replay.metadata.player_a.kind, replay.metadata.player_b.kind}
    assert kinds == {"policy", "human"}
    bot_spec = replay.metadata.player_a if bot_seat == 0 else replay.metadata.player_b
    assert bot_spec.kind == "policy"


def test_policy_internals_land_on_bot_steps_only(recorded) -> None:  # type: ignore[no-untyped-def]
    replay, bot_seat, _rec = recorded
    bot_actor = "player_a" if bot_seat == 0 else "player_b"
    with_internals = [s for s in replay.steps if s.policy_internals]
    assert with_internals, "no step carried policy internals"
    for step in with_internals:
        assert step.actor == bot_actor
        for internals in step.policy_internals:
            assert len(internals.type_probs) == 13
            assert internals.type_mask[internals.chosen_action[0]] is True


def test_every_bot_decision_keeps_its_internals(recorded) -> None:  # type: ignore[no-untyped-def]
    """No decision class may be silently dropped.

    Before the decision-only step landed, every ``END_TURN`` (and any other
    action emitting no broadcast events) lost its internals — ~37-40% of the
    main phase plus all four setup decisions."""
    replay, _bot_seat, rec = recorded
    recorded_internals = sum(len(s.policy_internals) for s in replay.steps)
    assert rec.n_bot_decisions > 0
    assert recorded_internals == rec.n_bot_decisions == rec.n_internals_recorded
    # The opening is the phase under suspicion: all four setup decisions kept.
    setup_internals = sum(len(s.policy_internals) for s in replay.steps if s.kind == "setup")
    assert setup_internals == 4
    # END_TURN (type 3) is the class that used to vanish entirely.
    chosen = [i.chosen_action[0] for s in replay.steps for i in s.policy_internals]
    assert 3 in chosen, "no END_TURN decision survived into the record"


def test_provenance_flags_are_recorded(recorded) -> None:  # type: ignore[no-untyped-def]
    meta = recorded[0].metadata
    assert meta.mode == "raw_policy"
    assert meta.sims is None
    assert meta.clairvoyant is False
    assert meta.reveal_bot is False
