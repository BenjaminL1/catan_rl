# Spec: opening-scoreboard (measurement) — pointer-arch accept-gate clause (b)

**Status: RATIFIED BINDING (owner, 2026-07-22).** Split decision: **commit the by-video
EVAL-A/HOLDOUT-B split now, measure on EVAL-A** (HOLDOUT-B reserved virgin for the eventual formal
gate) — D3.4 first option.

**Feature in one line.** Build `src/catan_rl/human_data/opening_scoreboard.py` — a read-only
measurement that scores how well a policy checkpoint *judges ThePhantom's parsed winning openings*,
computed **paired** for the new pointer-arch model vs v11, as accept-gate clause (b) and the ongoing
"is it still opening-weak?" probe. Measurement-first (defer the seed-injection and the heavy
inferential apparatus); grounded in the ratified step6 (`docs/plans/v2/step6_human_corpus.md`).

**Provenance.** 2026-07-22 /dev-plan: recon + interview (owner chose measurement-first + paired
new-vs-v11) + expert council (baseline-reviewer, drift-sentinel), which **converged** on the metric
choice below. Corpus reality measured: **88 eligible games** (`is_scoreboard_eligible()`), 100%
tournament-sourced, all below the step6 100-game floor.

## Decisions (binding once ratified)

### D1 — The metric is rank-based ΔAUC, NOT a raw value comparison (the load-bearing decision)
Comparing raw value-head outputs `v̂` across the two architectures is **invalid** — the new-arch and
v11 value heads are separately trained with their own value-normalizers, and the only V→common-scale
map (`σ(3.22·v̂−1.14)`, `search/value.py:34`) was fit to a *third* policy (v8-era). Both experts
independently reached this.
- **Primary clause-(b) metric = paired ΔAUC** = AUC(new-arch `v̂`, realized outcome) −
  AUC(v11 `v̂`, realized outcome), one observation per eligible game. AUC is a rank statistic
  (Mann–Whitney) → **scale-invariant → no squash → the cross-arch scale problem disappears**. This
  answers the real question — "does new-arch judge ThePhantom's *winning* openings at least as well
  as v11?" (discrimination), not "does it print a bigger number" (level). Reuses step6's GATE-A
  "Secondary door" ΔAUC machinery (`pregate0.py` already implements `auc`).
- **REJECTED:** raw-`v̂` level comparison, and any "assigns higher `p̂`" comparison even with refits —
  a globally-more-optimistic model would win without judging openings better (confounded).
- **The M2 partial-Spearman residual metric is OPTIONAL/secondary and deferred.** It needs a
  calibrated `p̂`, so IF later included it requires a **per-policy squash refit** (spec-003 procedure)
  for *both* new-arch and v11 — never the shared v8-fit constants. Not built in this slice.

### D2 — Paired-identical-state via ONE rebuild (reuse cross_arch/legacy_arch)
Per eligible game: build a net-new `GameRecord → BridgeState` adapter → `engine_bridge.rebuild_env`
→ **one shared engine state** → new-schema obs (native `ObsEncoder`) AND legacy-schema obs
(`LegacyObsEncoder`, exactly as `cross_arch.CrossArchEnv._sample_snapshot_action`) → two
`value_from_obs` calls. Gate on `assert_engine_parity` (the v11 leg needs the vendored legacy arch;
v11 cannot load in-process post-fork). The adapter must synthesize the 5 fields a `GameRecord` lacks
and both legs must use identical values: **ports** (the K=8 port-marginal — see D4), **robber** =
desert hex, **hands** derived from `settlements[1]` to match the bridge's grant assertion,
**opp_kind/opp_policy_id** pinned to one fixed sentinel (they enter the obs → affect `v̂` → must be
identical across both legs). Route through the bridge's spiral-consistency gate + invariant asserts
(finite-bank, conservation, hand-tracker) — do NOT add a `v̂` path that bypasses them.

### D3 — Pre-registration seam (measurement-first defers ENFORCEMENT, not RECORDING)
Even below the floor, the tool MUST, or a below-floor number gets laundered into a verdict later:
1. **Self-stamp a banner** in its own output: n per cell, `BELOW FLOOR (<100)`, `MEASUREMENT-ONLY`,
   `NO VERDICT`. The verdict/NO-GO path is suppressed under 100 (step6:60); the measurement still returns.
2. **Record (not yet refuse-to-run-enforce) input hashes**: `opening_scoreboard.py`, the adapter,
   `engine_bridge.py` + the `engine/board.py` injection surface, the corpus JSONL snapshot, BOTH
   checkpoints (new-lineage + `v11_cand_u724.pt`), and the **v11 pre-fork worktree commit**. Recording
   is one `sha256`; enforcement is deferred.
3. **Pre-register clause (b)'s sign + tolerance BEFORE computing**: fail only if the paired
   video-cluster bootstrap CI on ΔAUC excludes a small tolerance (≈0.02 AUC) in the regression
   direction (borrow step6 B3(4)) — a bare "≥ 0" point check fails an unchanged model ~20% of the time
   at this n.
4. **Commit the deterministic BY-VIDEO EVAL-A/HOLDOUT-B split file NOW** (seeded, per step6 §1) and
   measure on EVAL-A; HOLDOUT-B stays virgin for the later formal gate. (Alternative only if the owner
   prefers: stamp that the number is POOLED and HOLDOUT-B is thereby *burned* — but you cannot compute
   pooled now and later pretend a clean holdout exists.) Recommended: commit the split now.

### D4 — K=8 port-marginal cache from day one; resolve the "D6" naming
The 8 layouts marginalize the **unrecorded port assignment** over the fixed 9 port slots (NOT D6
board symmetries — ports are unrecorded; step6 §4 "port-marginal `p̂` = mean over 8 layouts"). Cache
`v̂` per `(record, k)`, k∈0..7, with the pinned sha256 per-record-per-layout seed — computed once per
game, never re-bridged inside any later bootstrap/permutation loop. Structuring the cache per
`(record,k)` now is required so the deferred apparatus just reads it (else it's a rework). State the
port-marginal reading explicitly in the module docstring.

### D5 — Read-only safety w.r.t. the LIVE self-play run
The run is training on MPS at `runs/train/selfplay_pointer_arch` right now. The scoreboard:
- reads only **numbered, atomic, immutable** checkpoints (`ckpt_000000NNN.pt`) — never a `latest`/tmp
  handle; **writes artifacts to `data/human/**`**, never inside `runs/train/**`;
- **CPU-pinned** (`--device cpu`); its only cost is CPU contention with the in-rollout snapshot
  opponents → schedule during low-contention windows / run on each promotion checkpoint, not continuously;
- additive TB scalar names only; snapshot/restore global RNG if any part runs in-process.

### D6 — Population honesty + the record.py same-commit docstring rewrite
- Print **ThePhantom's eligible-game WR with a cluster CI beside the metric** (winners-heavy channel;
  the absolute level is uninterpretable without it). Split/report by `opponent_strength.source`
  (all 88 are tournament today — moot but don't hardcode "all tournament"; the tournament sign-guard
  ≥10 is satisfied). Key eligibility off `record.is_scoreboard_eligible()` (the SoT predicate), never
  re-implement the filter.
- Building/freezing `opening_scoreboard.py` obligates rewriting the `record.py` guidance docstrings
  (module header AND `is_strong_opponent_scoreboard_eligible`) to the paired-cross-arch-consumer
  contract **in the same commit** (step6 §0.4).

## Non-goals (each with its no-restart path)
- **Seed-injection into self-play** (Workstream B / `human_seed` episodes) — deferred; conditional on
  this measurement confirming an opening weakness.
- **Full step6 GATE-A verdict apparatus** — the freeze-manifest *enforcement* (refuse-to-run), the
  B=10,000 within-stratum permutation, and the formal GO/NO-GO — deferred until the corpus grows past
  the 100-game floor (Intel sweep ongoing). The measurement-first core is a clean *subset* (D3.4 + D4
  keep it additive).
- The M2 partial-Spearman residual metric (D1) — optional secondary, not this slice.
- No engine-rule / obs-schema / glyph_anchor / training-config change; 1v1 ruleset + checkpoint
  lineage preserved.

## Acceptance criteria (the /dev-loop gate is the authority on checks)
1. Full gate green (typecheck / lint / rust fmt / unit tests).
2. **Paired-identical-state pin**: the new-schema and legacy-schema obs for a game are built from the
   SAME `engine_state_hash` (one rebuild feeds both value heads) — the core correctness guard.
3. Determinism pin (same seed → identical layouts, `v̂`, ΔAUC bit-for-bit) + different-seed control.
4. Adapter round-trip pin: a real `serialize_post_setup` state → record-shaped adapter → `rebuild_env`
   → `engine_state_hash` + `v̂` parity.
5. K=8 caching pin: `v̂` computed exactly 8× per game; no re-bridge inside resample loops.
6. Below-floor pin: at n=88 the tool emits the metric + CI but SUPPRESSES any NO-GO verdict.
7. Squash-isolation pin (only if a `p̂` metric is kept): v11 uses v11's own refit, not `(3.22,−1.14)`.
8. Read-only pins: reads numbered ckpts only; writes only under `data/human/**`; provenance hashes
   recorded (incl. the v11 pre-fork commit); base-rate WR printed.
9. `record.py` docstring rewrite in the same commit; by-video split file committed before first metric.

## Ratified open item (owner, 2026-07-22)
- **Split handling: COMMIT THE SPLIT NOW.** Emit a deterministic seeded by-video EVAL-A/HOLDOUT-B
  split file (per step6 §1) as a committed artifact in this build, and compute the metric on **EVAL-A
  only**; HOLDOUT-B stays untouched for the later formal GATE-A verdict once the corpus passes the
  100-game floor. (This makes the by-video split file a required deliverable of this slice even though
  the diversity-seed workstream that consumes EVAL-A is deferred.)
