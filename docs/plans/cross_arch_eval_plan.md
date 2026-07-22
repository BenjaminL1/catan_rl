# Cross-Architecture Eval Harness — new pointer-arch policy vs v11 champion (head-to-head, 100 games)

**Date:** 2026-07-21 · **Purpose:** the one instrument that answers "is the new architecture
actually better than v11." Feeds the ratified accept gate (dual-gate clause a) and is the progress
meter for self-play (in-lineage WR-vs-heuristic is saturated at ~0.97, so it can't tell strong
models apart).

## The problem

The pointer-arch fork changed BOTH:
- the **network shape** (corner/edge/tile pointer heads — new state-dict keys), and
- the **obs schema** (`CURR_PLAYER_DIM` 54→67, new `global_features` block, `is_setup`).

So the current code cannot instantiate v11: `build_actor('runs/anchors/v11_cand_u724.pt')` → shape
mismatch. `evaluate_policy_vs_policy` (`src/catan_rl/eval/harness.py`) therefore cannot pit
new-arch vs v11 in one process. Migrating v11 into the new arch gives it fresh untrained pointer
heads → plays terribly → a meaningless comparison. We need a *true* head-to-head.

## The key enabler

**The game engine is byte-identical across the fork** — we never touched engine rules. Only the
policy network and the obs encoder differ. Since an observation is a **pure function of game
state**, both architectures can play the *same* game: at each decision, the acting player's own
codebase builds its own-schema obs from the shared engine state and returns an action. This is what
makes a faithful cross-version tournament possible.

## Approach (recommended: cross-version subprocess bridge)

Run each policy under the code version that matches it, driving one shared game:

- **v11 side** runs under a **git worktree pinned at the last pre-fork commit** (the commit just
  before the pointer-arch merge `e34b011` — i.e. the tree where `build_actor(v11_cand)` still
  succeeds; the harness resolves/creates this worktree automatically). It hosts v11 with the *old*
  obs encoder.
- **new-arch side** runs under the current checkout with the new obs encoder.
- A **driver** owns the single authoritative `catanGame` instance and steps it. On each turn it asks
  the seat's policy for an action: the new-arch policy is called in-process; v11 is queried across a
  thin **subprocess/IPC boundary** that ships the serialized game state (or the minimal fields the
  obs encoder needs) to the pinned-worktree worker, which builds the old-schema obs, runs v11, and
  returns the action. The driver applies it to the shared game.

Because the engine is identical, both workers observe the same state and the game evolves
deterministically for a fixed seed.

**Fallback (documented, NOT primary):** if the subprocess bridge proves infeasible in the time
budget, an in-process "legacy obs + legacy arch" module could let v11 be instantiated alongside the
new arch, with the harness building both obs schemas per state. Weaker option — common-reference
(both play the heuristic, compare WR) — is explicitly rejected as the primary: both score ~0.97, so
it has no resolving power at the top.

## Design details

- **N = 100 games** (per the request), **seat-symmetrized**: 50 with new-arch as player 0 / v11 as
  player 1, 50 swapped, so first-move advantage cancels. Report combined WR + per-seat WR.
- **Statistics:** Wilson 95% CI on the combined WR (reuse `src/catan_rl/eval/wilson.py`). At n=100
  the half-width is ≈ ±0.10 — enough for a directional read (is new-arch clearly above/below 0.50),
  though the formal accept gate specifies n=600 for a tight bound; this 100-game tool is the fast
  progress check, and the same CLI scales to 600 with `--n-games 600` when gating.
- **Determinism:** fixed `--seed`; both sides consume the same board+dice stream from the shared
  engine. A given seed must reproduce the same games/result bit-for-bit.
- **Both policies play their deployed form.** Default = raw-policy argmax/sample as configured;
  optionally wrap each in the deployed PUCT search later (out of scope for v1 — measure the raw
  policies first).
- **1v1 ruleset preserved** — the shared engine is the standard 1v1 Colonist engine; no rule drift.

## CLI

```
scripts/eval_cross_arch.py \
  --new  <new-arch checkpoint, e.g. runs/train/selfplay_pointer_arch_*/checkpoints/ckpt_*.pt> \
  --old  runs/anchors/v11_cand_u724.pt \
  --n-games 100 --seed 0
# → combined WR, per-seat WR, Wilson 95% CI, n
```
The pre-fork worktree for the v11 side is created/reused automatically (documented one-time setup if
manual creation is needed).

## Verification (correctness before trust)

1. **New-vs-new equivalence:** run a new-arch checkpoint against ITSELF through the bridge and assert
   the result matches the in-process `evaluate_policy_vs_policy` for the same seed — proves the
   bridge doesn't distort play.
2. **Determinism pin:** same `--seed` twice → identical WR.
3. **Engine-parity guard:** stamp the engine git-SHA on both sides and refuse to run if they differ
   (the whole method rests on the engine being identical across the fork).

## How it plugs into the accept gate

The ratified dual gate requires new-arch to beat v11 head-to-head (Wilson-LB > 0.50 at n=600). This
harness IS that measurement — the 100-game default is the frequent self-play progress check; the
gate runs the same tool at `--n-games 600`. Until a self-play checkpoint clears it, "surpasses v11"
stays unproven.

## Implementation notes (2026-07-21 — what actually shipped)

**Chosen approach: the IN-PROCESS variant (the plan's documented fallback), NOT the subprocess
bridge.** Recon confirmed the enabler is airtight — `src/catan_rl/engine/` and
`src/catan_rl/policy/board_geometry.py` are byte-identical between the pre-fork commit and HEAD (the
engine tree SHA is literally equal) — so a subprocess/IPC boundary buys nothing but fragility
(serialising the unpicklable `catanGame` per decision, two venvs). One process, one shared game; each
policy reads its own-schema obs from the same live state. The common-reference fallback stays
rejected.

Shipped:
- `src/catan_rl/eval/legacy_arch/` — the v11-era policy arch (obs_schema/encoders/heads/network/
  obs_encoder) vendored as **byte-faithful copies** of the pre-fork commit `501b3d2` (= `9692a79~1`),
  modulo three declared import rewrites. Generated by `scripts/vendor_legacy_arch.py`; faithfulness
  pinned by `_provenance.py` + a provenance test.
- `src/catan_rl/eval/cross_arch.py` — `CrossArchEnv` (subclasses `CatanEnv`, overrides ONLY the
  opponent's obs-build seam `_sample_snapshot_action` to build a legacy-schema obs; every other path
  is the unmodified parent) + `cross_arch_h2h(...)` returning a seat-symmetrized `EvalMatchupResult`.
- `src/catan_rl/eval/engine_parity.py` — guard (c): refuses to run if the live engine/board-geometry
  drifted from the pinned pre-fork tree (keys off HEAD's tree SHA so it works in shallow CI).
- `scripts/eval_cross_arch.py` — the CLI (`--new --old --n-games 100 --seed 0`;
  `--old-arch new` for the equivalence self-check; `--skip-engine-parity-check` to override).

Correctness guards, all in `tests/unit/eval/test_cross_arch.py` (14 tests, CI-runnable on fresh
random-init policies — no `runs/` dependency): (1) new-vs-new through `CrossArchEnv` reproduces
in-process `evaluate_policy_vs_policy` bit-for-bit; (2) determinism at a fixed seed (+ RNG
containment + a different-seed negative control); (3) engine-parity guard fires on drift; plus
provenance byte-identity, the legacy strict-load + v11-schema check, and the "current arch can't load
v11" gap proof.

Command (identical to the plan's CLI, default now n=100):
```
python scripts/eval_cross_arch.py --new <new-arch ckpt> --old runs/anchors/v11_cand_u724.pt \
  --n-games 100 --seed 0          # gate: --n-games 600
```
No worktree setup is needed (the subprocess bridge was not built).
