# Round 4 — Resolution

Per-disagreement classification: RESOLVED / COMPROMISE / DEADLOCK. No fake consensus.

---

## D1 — MCTS earning its compute

**Status**: COMPROMISE

**The hybrid both sides accept**:
- v2 ships Step 5 as planned with the AlphaZero-style PUCT + learned prior + learned value + belief determinization (B's position retained — the search architecture is right).
- BUT: add a **value-calibration probe** to Step-4 milestone evals (`value/calibration_l1_vs_heuristic_wr`) — the gap between `value_head(s)` and empirical late-game WR against the heuristic on a held-out 1024-state batch. Threshold: ≤ 0.10 absolute. Both B and D agree this is the right diagnostic; they disagree about what its value will be.
- Step-5 preflight 0.4 gates on belief KL; **add a new preflight 0.5** that gates on this value-calibration probe.
- Step-5's existing "first MCTS-vs-policy-alone A/B" stays as the decisive gate (`v2_step5_mcts.md` STOP/RESUME §9).

**Why this is a real compromise, not capitulation**: B retains the algorithm choice; D retains the right to be vindicated by a measurement. The plan doesn't pre-commit to a winner.

---

## D2 — Reward shape `±1 + (vp_diff)/15` vs pure `±1`

**Status**: DEADLOCK with decisive experiment

**Both positions**:
- **C**: keep `±1 + (vp_diff)/15` — margin distinguishes blowouts from squeakers, is bounded so cannot dominate WR signal, and per-step ΔVP is already ruled out by Collazo.
- **D**: drop the margin term — `(vp_diff)/15` couples gradients to VP-leading, which Friendly Robber explicitly penalises; same mechanism Catanatron's depth-3 paradox revealed.

**The crux of disagreement**: whether the value head learns to *correct* the reward's VP coupling (C) or is *trained into* the coupling (D). Both arguments are theoretically defensible.

**Decisive experiment (Step-4 ablation A1)**:
- Same Step-4 config, single-feature-flag toggle on reward function.
- Two arms: `reward_terminal_only=True, vp_margin_coef=0.0` vs `vp_margin_coef=1/15`.
- N=3 seeds × 5M PPO steps each (half of full Gate-1 budget; the 5M is enough to discriminate ≥ 0.03 WR difference at our seed count).
- Metric: PPO-BR-gap after 5M steps (NOT in-league WR, which has the same reward dependency).
- Decision rule:
  - `margined_BR_gap − binary_BR_gap > 0.05` → D was right; v2.1 drops the margin term.
  - `|margined − binary| ≤ 0.02` → wash; C wins on Occam (simpler reward).
  - `binary − margined > 0.05` → unlikely but possible; C strongly confirmed.

**Cost**: 1 extra seed-equivalent (~3-7 days M1 Pro CPU per Step-4 plan §7), runs in parallel with the canonical seed.

**Status held until experiment runs**. Until then: ship the faculty-corrected `±1 + (vp_diff)/15` as the default; flag A1 as launch-blocking for any v2.1 reward decision.

---

## D3 — piKL decay window (2M vs ≥ 10M)

**Status**: RESOLVED (instrumentation already in plan, decision moves to data)

**Resolution**: ship 2M decay (the panel-revised window, current default in `v2_step4_ppo.md:78`). Already-planned `bc/v_drift_l1` probe is the discriminator. New gating rule explicitly added:
- **After 1M PPO steps of seed #1**, evaluate the probe:
  - If `v_drift_l1` is trending *down* or flat → 2M decay is adequate; launch seeds #2-3 with the same window.
  - If `v_drift_l1` is trending *up* (slope > 0 over the last 500k steps) → D's variance horizon argument is confirmed; extend decay to 5M and re-launch.
- This is a STOP/RESUME point added to the Step-4 plan §9 between "First 1M PPO steps" and "First eval at 100k steps."

D conceded that the *measurement* settles this. B conceded that the gating rule should fire on the measurement, not be tuned by hand. Both accepted.

---

## D4 — Obs schema completeness (bag state)

> **Erratum (2026-05-15)**: two corrections.
> 1. Original Round-3 debate used `opp_last_rolled_7` (one-roll signal). Karma is persistent state — see `docs/1v1_rules.md` "Karma mechanic note."
> 2. The Karma part of the proposed extension is **already in the obs** (`src/catan_rl/policy/obs_encoder.py:545-560`). The A2 ablation reduces to `bag_remaining` only.
>
> Experiment design and decision rule are otherwise unchanged.

**Status**: DEADLOCK with decisive experiment

**Both positions**:
- **A**: the env's dice mechanism is part of the MDP, not the agent's input. PPO converges to the policy that's optimal for the *induced* MDP, dice noise included. Self-play symmetry cancels first-order bias.
- **D**: at eval time against asymmetric opponents (heuristic, AlphaBeta, v1 champion), the symmetry argument fails. `V(obs)` over-trusts late-game states with bag-residual information the obs doesn't contain.

**The crux**: whether the value-head bias is symmetric (cancels) or asymmetric at eval (doesn't).

**Decisive experiment (Step-3 BC ablation A2)**:
- Two BC training runs:
  - Arm 1 — canonical obs (10 keys, current schema).
  - Arm 2 — extended obs adding `bag_remaining: (11,) float32` (normalised count per sum). *(Karma state was originally proposed as a second new field but is already encoded — see erratum.)*
- N=3 seeds each × 30k-game dataset × 10 epochs.
- Metric: WR vs heuristic at 600-game eval (200 × 3 seats × 1 seed for budget; expand if marginal).
- Decision rule:
  - `extended_WR − canonical_WR > 0.02` → D was right; v2.1 adopts extended obs (re-running Step-3 first, then proceeding).
  - `|extended − canonical| ≤ 0.01` → wash; A wins on Occam (smaller obs, fewer params).
  - `canonical − extended > 0.01` → unlikely; would indicate noise from the extra inputs.

**Cost**: ~2× Step-3 BC training (10-15 hours each), parallelizable. Runs before any Step-4 launch.

**Status**: launch-blocking experiment. The BC-stage cost is cheap; getting Step-4 right matters more.

---

## D5 — Opp-id embedding generalisation

**Status**: RESOLVED-WITH-INSTRUMENTATION

**Resolution**: keep the 8-d embedding with 40% mask probability. Add a milestone-eval probe:
- At every Gate-1 / champion / AlphaBeta milestone eval, run two parallel 200-game evals against the same opponent:
  - (a) `opp_kind` + `opp_policy_id` provided correctly (when known) or `unknown`.
  - (b) Both forced to `unknown` regardless.
- Report `Δ_WR_emb_on_vs_off`.
- Decision rule: `Δ_WR > 2pp` → embedding has memorised; consider dropping or reducing dim for v2.1. `Δ_WR ≤ 2pp` → generalisation is fine; keep.

All four agents accept. D's concern is real but cheap to falsify; the other three's "keep" is conditional on this probe.

**Cost**: ~200 extra games per milestone eval. Negligible.

---

## Summary table

| ID | Topic | Status | Cost to resolve | Launch-blocking? |
|---|---|---|---|---|
| D1 | MCTS vs depth-3 pathology | COMPROMISE (add value-calibration probe) | Probe only; no extra training | No — Step 5 still gated on its existing A/B |
| D2 | Reward `vp_diff/15` vs pure ±1 | DEADLOCK + experiment | 1 extra 5M-step Step-4 seed | Soft — required for v2.1 reward decision; v2.0 ships with margined |
| D3 | piKL decay 2M vs ≥ 10M | RESOLVED (gate on `v_drift_l1` at 1M of seed #1) | Free (probe already planned) | No — gating rule added to STOP/RESUME |
| D4 | Obs schema + bag (Karma already encoded) | DEADLOCK + experiment | 2× Step-3 BC runs (~30 hrs) | **Yes** — fixes baseline schema before Step 4 |
| D5 | Opp-id embedding generalisation | RESOLVED (add `Δ_WR_emb_on_vs_off` probe) | ~200 games per milestone | No — probe runs at existing checkpoints |

**Two RESOLVEDs, one COMPROMISE, two DEADLOCKs with decisive experiments.**

The two DEADLOCKs are not weaknesses of the analysis — they're genuine empirical questions the literature doesn't settle for 1v1 Catan. D2 is cheap; D4 is cheaper. Both should run before the headline Step-4 training begins.
