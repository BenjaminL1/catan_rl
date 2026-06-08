# Phase 1 Data Model: self-play snapshot opponent

No persistent schema change. These are in-memory/runtime entities the feature
introduces or threads through existing code.

## LeagueSnapshot (exists — `selfplay/league.py`)

A frozen past policy in the league pool.

- `snapshot_id: int` — stable, monotonic identity (already assigned by
  `add_snapshot`).
- `state_dict` — CPU clone of the policy parameters at capture.
- `update_idx: int` — training update at which it was captured.

Validation: `state_dict` must match the current policy architecture (shape-correct
by construction, since snapshots are clones of the live policy).

## OpponentAssignment (new — threaded into env reset)

Per-environment description of who the opponent is for the next episode.

- `kind: {"heuristic", "random", "snapshot"}`
- `snapshot_id: int | None` — set iff `kind == "snapshot"`.

State transition: produced by `league.build_env_opponent_mix(...)` once per
rollout, applied via `vec_env.set_opponents(...)`, consumed at the next
`env.reset`. Immutable for the duration of an episode.

## FrozenOpponentPolicy (new — runtime, inference-only)

A `CatanPolicy` loaded from a `LeagueSnapshot`, used only to produce opponent
moves.

- Built once per distinct in-play `snapshot_id`, `eval()` mode, on the learner
  device, cached for the rollout.
- Never added to an optimizer; only invoked under `torch.no_grad`.
- Interface used: `sample(obs, masks) -> action` (the existing policy surface).

## EvalMatchupResult (EXTENDS the existing `EvalResult` — policy-vs-policy output)

Outcome of champion vs a loaded opponent over N seat-symmetrized games. Extends
`eval/harness.py:EvalResult` (adds `opponent_ref`); does NOT fork a parallel type.

- `opponent_ref: str` — checkpoint path or `snapshot_id`.
- `wins: int`, `n: int`
- `wr: float`, `ci: WilsonInterval` (reuse `eval/wilson.py`).
- `seat0_wr / seat1_wr` — symmetrization breakdown (mirror `EvalResult`).

Determinism: identical across runs at a fixed seed **on CPU**.
