"""Unit tests for scripts/eval_wr_non_inferiority.py — the D7 gate-2 CLI.

The arithmetic is ``bc.gates.paired_wr_non_inferiority``'s and is tested there;
what is only testable HERE is the orchestration the CLI owns and that a caller
depends on:

  * the R0 refusal (D6) — an R1-stamped checkpoint must fail CLOSED, not be
    annotated, because R0 and R1 win rates are not comparable;
  * the exit-code contract — ``0`` on pass, ``1`` on fail, because this is
    invoked from a shell gate that reads ``$?`` and nothing else;
  * the ``by_opponent`` None guard, which is the difference between a clear
    refusal and an ``AttributeError`` on ``None``; and
  * the harness wiring — both rounds must be configured IDENTICALLY, since that
    identity is the only thing making the two runs seed-paired.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_CLI = Path(__file__).resolve().parents[3] / "scripts" / "eval_wr_non_inferiority.py"
_spec = importlib.util.spec_from_file_location("eval_wr_non_inferiority_module", _CLI)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
sys.modules["eval_wr_non_inferiority_module"] = cli
_spec.loader.exec_module(cli)


# ---------------------------------------------------------------------------
# Fakes standing in for the harness. The CLI's contract with it is narrow:
# ``EvalHarness(...).run(policy).by_opponent(label)`` -> something with ``.games``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Game:
    seed: int
    agent_seat: int
    won: bool


class _Report:
    def __init__(self, result: Any) -> None:
        self._result = result

    def by_opponent(self, label: str) -> Any:
        return self._result


class _Result:
    def __init__(self, games: list[_Game]) -> None:
        self.games = games


class _FakePolicy:
    """Stands in for the loaded champion. Records that the CLI put it in eval
    mode — the harness samples from it, and a train()-mode net is not the
    checkpoint anyone banked."""

    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


def _games(wins: int, n: int = 8) -> _Result:
    """``n`` paired games over a fixed (seed, seat) plan, ``wins`` of them won."""
    return _Result(
        [_Game(seed=i // 2, agent_seat=i % 2, won=i < wins) for i in range(n)],
    )


def _install_fake_harness(
    monkeypatch: pytest.MonkeyPatch, results: list[Any], *, stub_ruleset: bool = True
) -> list[dict]:
    """Patch out the policy loader and the harness; record harness kwargs.

    ``results`` is consumed in call order: the CLI runs the CANDIDATE first and
    the BASELINE second. ``stub_ruleset=False`` leaves the real stamp reader in
    place, for the tests that hand in genuinely-stamped checkpoints.
    """
    from catan_rl.bc import finetune as ft
    from catan_rl.eval import harness as hn

    calls: list[dict] = []
    pending = list(results)

    monkeypatch.setattr(ft, "build_policy", lambda path, device: _FakePolicy())
    if stub_ruleset:
        monkeypatch.setattr(hn, "checkpoint_ruleset", lambda path: "R0")

    class _FakeHarness:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def run(self, policy: Any) -> _Report:
            return _Report(pending.pop(0))

    monkeypatch.setattr(hn, "EvalHarness", _FakeHarness)
    return calls


def _argv(tmp_path: Path, **extra: str) -> list[str]:
    cand = tmp_path / "candidate.pt"
    base = tmp_path / "baseline.pt"
    cand.touch()
    base.touch()
    argv = ["--candidate", str(cand), "--baseline", str(base), "--n-games-per-seat", "4"]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", value]
    return argv


# ---------------------------------------------------------------------------
# D6 — an R1-stamped checkpoint fails CLOSED
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stamped(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One R0-stamped and one R1-stamped policy-only checkpoint, really written.

    Real files, not a patched ``checkpoint_ruleset``: the refusal is only worth
    testing if it reads an actual saved stamp the way production does.
    """
    from catan_rl.checkpoint import save_policy_only
    from catan_rl.policy import CatanPolicy
    from catan_rl.policy.board_geometry import build_geometry

    root = tmp_path_factory.mktemp("stamped")
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    out: dict[str, Path] = {}
    for epoch in ("R0", "R1"):
        path = root / f"{epoch}.pt"
        save_policy_only(
            path,
            config={"rollout": {"ruleset": epoch}},
            policy=policy,
            update_idx=0,
            global_step=0,
        )
        out[epoch] = path
    return out


@pytest.mark.parametrize("r1_side", ["candidate", "baseline"])
def test_it_refuses_an_r1_stamped_checkpoint(stamped: dict[str, Path], r1_side: str) -> None:
    """Both seats are checked, not just the candidate: a baseline from the other
    epoch makes the comparison just as meaningless."""
    paths = {"candidate": stamped["R0"], "baseline": stamped["R0"]}
    paths[r1_side] = stamped["R1"]

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--candidate",
                str(paths["candidate"]),
                "--baseline",
                str(paths["baseline"]),
                "--n-games-per-seat",
                "1",
            ]
        )
    # ``SystemExit(str)`` exits non-zero (the interpreter prints it and uses 1).
    assert excinfo.value.code != 0
    message = str(excinfo.value)
    assert r1_side in message
    assert "R1" in message and "R0" in message


def test_two_r0_checkpoints_are_not_refused(
    stamped: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for the refusal above: the SAME invocation with both seats on
    R0 reaches the harness instead of exiting at the epoch check.

    ``checkpoint_ruleset`` is deliberately NOT stubbed — the real reader must
    return R0 for a real R0 stamp, or the refusal above proves nothing."""
    _install_fake_harness(monkeypatch, [_games(wins=4), _games(wins=4)], stub_ruleset=False)
    assert (
        cli.main(
            [
                "--candidate",
                str(stamped["R0"]),
                "--baseline",
                str(stamped["R0"]),
                "--n-games-per-seat",
                "1",
            ]
        )
        == 0
    )


# ---------------------------------------------------------------------------
# The exit-code contract: 0 on pass, 1 on fail
# ---------------------------------------------------------------------------


def test_a_non_inferior_candidate_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Identical outcomes on every paired (seed, seat) -> delta 0, CI degenerate
    # at 0, and 0 > -0.05. The gate asks "provably not worse", not "better".
    _install_fake_harness(monkeypatch, [_games(wins=4), _games(wins=4)])
    assert cli.main(_argv(tmp_path)) == 0

    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["passes"] is True
    assert payload["ruleset"] == "R0"
    assert payload["n_pairs"] == 8


def test_a_regressed_candidate_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-vacuity for the test above. The candidate loses every game the
    baseline won, so the CI lower bound sits far below ``-margin``."""
    _install_fake_harness(monkeypatch, [_games(wins=0), _games(wins=8)])
    assert cli.main(_argv(tmp_path)) == 1

    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["passes"] is False
    assert payload["delta"] == pytest.approx(-1.0)
    assert payload["ci_lower"] < -payload["margin"]


def test_the_margin_is_a_non_inferiority_bound_not_a_superiority_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is ``lower > -margin``. A candidate that is slightly WORSE than
    the baseline therefore PASSES, which is the whole point of a non-inferiority
    test and the thing the ``--margin`` help text has to describe correctly."""
    # 3/8 vs 4/8: delta = -0.125 on one pair only, so the CI lower bound clears
    # -0.5 comfortably while the candidate is genuinely behind on the point
    # estimate. A superiority reading of the same margin would fail this.
    _install_fake_harness(monkeypatch, [_games(wins=3), _games(wins=4)])
    assert cli.main(_argv(tmp_path, margin="0.5")) == 0


# ---------------------------------------------------------------------------
# The by_opponent guard
# ---------------------------------------------------------------------------


def test_a_missing_matchup_is_refused_not_dereferenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``by_opponent`` returns ``None`` for a label the report does not carry.
    Without the guard the next line raises ``AttributeError: 'NoneType' has no
    attribute 'games'`` from inside the gate arithmetic, which names neither the
    checkpoint nor the missing opponent."""
    _install_fake_harness(monkeypatch, [None, _games(wins=4)])
    with pytest.raises(SystemExit) as excinfo:
        cli.main(_argv(tmp_path))
    assert excinfo.value.code != 0
    assert cli.HEURISTIC in str(excinfo.value)
    assert "candidate.pt" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Harness wiring — the two rounds must be configured identically
# ---------------------------------------------------------------------------


def test_both_rounds_build_an_identically_configured_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pairing is by construction: ``EvalHarness`` derives its seed plan from
    ``seed`` / ``opponent_types`` / ``n_games_per_seat`` alone, so the candidate
    and baseline rounds only play the same ``(seed, agent_seat)`` keys while
    those arguments match. If they ever diverge, the gate is comparing two
    different game sets and calling the difference a paired delta."""
    from catan_rl.env.ruleset import RULESET_R0

    calls = _install_fake_harness(monkeypatch, [_games(wins=4), _games(wins=4)])
    cli.main(_argv(tmp_path, seed="17", max_turns="123"))

    assert len(calls) == 2, "one fresh harness per checkpoint"
    assert calls[0] == calls[1]
    assert calls[0]["opponent_types"] == (cli.HEURISTIC,)
    assert calls[0]["n_games_per_seat"] == 4
    assert calls[0]["seed"] == 17
    assert calls[0]["max_turns"] == 123
    # D6 again, on the play side this time: the games themselves are R0.
    assert calls[0]["ruleset"] == RULESET_R0
