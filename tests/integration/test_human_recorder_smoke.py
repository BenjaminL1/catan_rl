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


def _play_recorded_game(mod, *, hud_log):  # type: ignore[no-untyped-def]
    """Play one deterministic headless game, optionally with a HUD log attached.

    ``hud_log`` is the harness-owned bounded deque the on-screen move log uses.
    It is reached exactly the way the live game reaches it — through
    ``view.move_log`` — and is appended to on every step, so if the log had any
    contention with the recorder's ``EventCollector`` this run would diverge."""
    import types

    import torch

    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.replay.player_factory import _PolicyActor

    torch.manual_seed(1234)
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy.eval()
    agent = mod._RawPolicyAgent(
        _PolicyActor(kind="policy", ckpt_path="<fresh>", policy=policy, device=torch.device("cpu"))
    )

    env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=25)
    env._auto_human = True  # no GUI: the human seat auto-plays legal moves
    env.reset(seed=17, options={"agent_seat": 0})
    if hud_log is not None:
        # The log lives on the VIEW, never on the env (see _log_move).
        env._human_view = types.SimpleNamespace(move_log=hud_log)
    recorder = mod._HumanGameRecorder(env, bot_seat=0)

    terminated = truncated = False
    steps = 0
    torch.manual_seed(99)
    while not terminated and not truncated and steps < env.max_turns * 20:
        action = agent.choose_action(env)
        _obs, _r, terminated, truncated, _info = env.step(action)
        if hud_log is not None:
            hud_log.append(f"Bot: {mod._describe_bot_move(action, with_location=True)}")
            env._log_move("You did something")
        recorder.after_env_step(
            action, agent.last_internals, terminated=terminated, truncated=truncated
        )
        steps += 1

    return recorder.finish(
        ckpt="<fresh>",
        seed=17,
        mode="raw_policy",
        sims=None,
        clairvoyant=False,
        reveal_bot=False,
    )


def test_the_hud_log_does_not_steal_the_recorders_events() -> None:
    """D5 / AC7: the on-screen log must not drain the recorder's collector.

    ``EventCollector.drain()`` replaces its buffer with an empty one and is
    single-consumer. Had the HUD reused it, some frames would drain first and the
    recorder would bank an empty list — the Replay losing a scatter of events
    nobody notices, since the replay is what you consult AFTER you stop trusting
    your memory. Same seed, same policy, log on vs log off: identical streams."""
    from collections import deque

    mod = _load_module()
    hud_log: deque[str] = deque(maxlen=6)
    with_log = _play_recorded_game(mod, hud_log=hud_log)
    without_log = _play_recorded_game(mod, hud_log=None)

    assert hud_log, "the HUD log recorded nothing — the test would prove nothing"
    assert any(s.events for s in with_log.steps), "no events at all — nothing to lose"
    assert len(with_log.steps) == len(without_log.steps)
    for a, b in zip(with_log.steps, without_log.steps, strict=True):
        assert a.actor == b.actor and a.kind == b.kind
        assert a.events == b.events
    assert with_log.metadata.total_steps == without_log.metadata.total_steps
