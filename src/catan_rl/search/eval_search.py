"""Search-aware evaluation loop (contract C4).

The eval harness drives the agent seat with ``policy.sample(obs, masks)`` — but a
search agent needs the LIVE env to clone + simulate (the obs is a lossy encoding
insufficient to reconstruct the game). So this module owns a search-aware game
loop that hands ``SearchAgent.choose_action(env)`` the live env, while REUSING
the harness's seat-symmetrization seed scheme, ``GameOutcome`` /
``EvalMatchupResult`` types, ``wilson_interval`` CI, and ``run_all_invariants``
legality audit (no parallel implementation, no semantics drift).

The opponent seat is driven by the env's existing snapshot-opponent driver (a
``FrozenSnapshotOpponent``), exactly as ``evaluate_policy_vs_policy`` does — the
engine and rules are untouched; only legal actions are played. CPU-pinned;
global numpy / stdlib / torch RNG is snapshotted + restored around the run so it
is reproducible and safe to call from anywhere.
"""

from __future__ import annotations

import random
import zlib
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from catan_rl.env.ruleset import RULESET_R0, validate_ruleset
from catan_rl.eval.harness import EvalMatchupResult, GameOutcome
from catan_rl.eval.rules_invariants import run_all_invariants
from catan_rl.eval.wilson import wilson_interval

if TYPE_CHECKING:
    from typing import Protocol

    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig
    from catan_rl.selfplay.snapshot_opponent import SnapshotOpponent

    class _ActionChooser(Protocol):
        """The decision surface ``_play_search_game`` needs (SearchAgent + the
        spec-008 oracle-root agent both satisfy it structurally)."""

        def choose_action(self, env: CatanEnv) -> np.ndarray: ...


def _play_search_game(
    env: CatanEnv,
    agent: _ActionChooser,
    *,
    seed: int,
    agent_seat: int,
    audit_rules: bool,
    opponent_seed: int | None = None,
) -> GameOutcome:
    """Play one game with the agent seat driven by search; agent-POV outcome.

    ``opponent_seed`` (optional) pins the snapshot opponent's per-game RNG seed
    explicitly instead of letting the env draw one from its own RNG — the SPRT
    differential driver uses it so A's and B's game at the same seed face an
    identical reference stream. ``None`` preserves the env's default draw exactly
    (byte-identical for every existing caller).
    """
    options: dict[str, object] = {"agent_seat": agent_seat}
    if opponent_seed is not None:
        options["opponent_seed"] = int(opponent_seed)
    env.reset(seed=seed, options=options)
    terminated = False
    truncated = False
    n_steps = 0
    safety_cap = env.max_turns * 50
    hit_safety = False
    while not terminated and not truncated:
        action = agent.choose_action(env)
        _obs, _r, terminated, truncated, _info = env.step(action)
        n_steps += 1
        if n_steps > safety_cap:
            truncated = True
            hit_safety = True
            break

    assert env.game is not None and env.agent_player is not None and env.opponent_player is not None
    agent_vp = int(env.agent_player.victoryPoints)
    opp_vp = int(env.opponent_player.victoryPoints)
    won = (not hit_safety) and agent_vp >= env.game.maxPoints and agent_vp > opp_vp
    violations: tuple[str, ...] = ()
    if audit_rules:
        violations = tuple(run_all_invariants(env.game, truncated=truncated))
    return GameOutcome(
        seed=seed,
        agent_seat=agent_seat,
        won=won,
        truncated=truncated,
        final_vp_agent=agent_vp,
        final_vp_opp=opp_vp,
        n_turns=n_steps,
        rules_violations=violations,
    )


def run_search_matchup(
    agent: SearchAgent,
    *,
    opponent_type: str,
    opponent: SnapshotOpponent | None,
    n_games: int,
    seed: int,
    max_turns: int = 400,
    opponent_ref: str,
    audit_rules: bool = True,
    alpha: float = 0.05,
    ruleset: str = RULESET_R0,
    opponent_ruleset: str | None = None,
) -> EvalMatchupResult:
    """Seat-symmetrized search-vs-opponent matchup; returns WR + Wilson CI.

    ``opponent`` is a frozen snapshot driving the opponent seat (None ->
    ``opponent_type``'s engine body, e.g. the heuristic). Mirrors
    ``EvalHarness._run_matchup_games`` seeding so two runs at one seed match.

    ``ruleset`` is the epoch the SEARCH agent's seat plays; ``opponent_ruleset``
    (default: same) is the other seat's. Both default to ``R0`` — the epoch the
    banked +54.6 Elo search rung was measured under — so an R1 policy is never
    seated under R0 rules by omission (spec ``preroll-dev-cards-r1`` D8). The
    caller resolves the epochs; :func:`evaluate_search_vs_policy` reads them off
    the checkpoints and refuses a mismatch.
    """
    from catan_rl.env.catan_env import CatanEnv

    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    try:
        # ``audit_events`` retains the broadcast stream so the post-game
        # ``run_all_invariants`` call below can actually run its event-stream
        # checks — with no retained log they silently no-op. Tied to
        # ``audit_rules``: with the audit off nothing reads the log, and every
        # MCTS ``clone_env`` deep-copies it (a full game emits thousands of
        # events), so retaining it unread would slow the search path down.
        env = CatanEnv(
            opponent_type=opponent_type,
            max_turns=max_turns,
            audit_events=audit_rules,
            ruleset=validate_ruleset(ruleset),
            opponent_ruleset=(
                None if opponent_ruleset is None else validate_ruleset(opponent_ruleset)
            ),
        )
        if opponent is not None:
            env.set_snapshot_opponent(opponent)
        try:
            n_per_seat = max(1, n_games // 2)
            opp_hash = zlib.crc32(opponent_ref.encode("utf-8"))
            base_seed = (seed * 1_000_003 + opp_hash) % (2**31 - 1)
            games: list[GameOutcome] = []
            for i in range(n_per_seat):
                seed0 = (base_seed + i * 2) % (2**31 - 1)
                seed1 = (base_seed + i * 2 + 1) % (2**31 - 1)
                games.append(
                    _play_search_game(env, agent, seed=seed0, agent_seat=0, audit_rules=audit_rules)
                )
                games.append(
                    _play_search_game(env, agent, seed=seed1, agent_seat=1, audit_rules=audit_rules)
                )
        finally:
            env.close()
        wins = sum(1 for g in games if g.won)
        ci = wilson_interval(wins=wins, n=len(games), alpha=alpha)
        return EvalMatchupResult(
            opponent_type="snapshot" if opponent is not None else opponent_type,
            games=tuple(games),
            wins=wins,
            n=len(games),
            ci=ci,
            opponent_ref=opponent_ref,
        )
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)


def evaluate_search_vs_policy(
    search_cfg: SearchConfig,
    search_ckpt: str,
    opponent_ckpt: str,
    *,
    n_games: int,
    seed: int = 0,
    device: str = "cpu",
    max_turns: int = 400,
    audit_rules: bool = True,
    ruleset: str | None = None,
    opponent_ruleset: str | None = None,
    allow_mixed_ruleset: bool = False,
) -> EvalMatchupResult:
    """Evaluate a search agent (wrapping ``search_ckpt``) vs a raw ``opponent_ckpt``.

    Both checkpoints load through the existing ``build_actor`` (no new loader, no
    state-dict change). The search models its opponent as its OWN wrapped policy
    (exact when ``search_ckpt == opponent_ckpt`` — the bake-off case).

    Each seat's ruleset epoch defaults to the one stamped on its OWN checkpoint
    (:func:`~catan_rl.eval.harness.checkpoint_ruleset`; absent ⇒ ``R0``), and a
    mismatch is REFUSED with
    :class:`~catan_rl.eval.harness.CrossRulesetEvalError` unless
    ``allow_mixed_ruleset=True`` — same contract as
    :func:`~catan_rl.eval.harness.evaluate_policy_vs_policy` (D8). Search is a
    place this matters twice over: ``search/mcts.py`` short-circuits forced
    nodes, so under R0 a roll node costs zero simulations and under R1 it costs
    a full search — the two epochs are not the same experiment.
    """
    from catan_rl.eval.harness import CrossRulesetEvalError, checkpoint_ruleset
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    search_epoch = checkpoint_ruleset(search_ckpt) if ruleset is None else validate_ruleset(ruleset)
    opp_epoch = (
        checkpoint_ruleset(opponent_ckpt)
        if opponent_ruleset is None
        else validate_ruleset(opponent_ruleset)
    )
    if search_epoch != opp_epoch and not allow_mixed_ruleset:
        raise CrossRulesetEvalError(
            f"cross-ruleset search h2h refused: {search_ckpt!r} was trained under "
            f"{search_epoch!r} but {opponent_ckpt!r} under {opp_epoch!r}. Numbers "
            "from the two epochs are not comparable. Either evaluate both under "
            "one epoch, or pass allow_mixed_ruleset=True to give each seat its own."
        )

    search_actor = cast(
        _PolicyActor,
        build_actor(
            PlayerSpec(kind="policy", ckpt_path=str(search_ckpt)), seed=seed, device=device
        ),
    )
    agent = SearchAgent(search_actor.policy, search_cfg, device=search_actor.device)

    opp_actor = cast(
        _PolicyActor,
        build_actor(
            PlayerSpec(kind="policy", ckpt_path=str(opponent_ckpt)), seed=seed, device=device
        ),
    )
    opponent = FrozenSnapshotOpponent(opp_actor.policy, device=opp_actor.device, seed=seed)

    return run_search_matchup(
        agent,
        opponent_type="snapshot",
        opponent=opponent,
        n_games=n_games,
        seed=seed,
        max_turns=max_turns,
        opponent_ref=str(opponent_ckpt),
        audit_rules=audit_rules,
        ruleset=search_epoch,
        opponent_ruleset=opp_epoch,
    )
