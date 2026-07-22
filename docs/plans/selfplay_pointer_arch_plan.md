# Self-play launch plan — pointer-arch lineage (stage 4/4)

**Date:** 2026-07-21 · **Author:** Fable (plan) · **Executor:** launchd (I launch), config by Opus
**Status:** REVIEWED (2026-07-21, senior-RL, verdict LAUNCH-AFTER-FIXES) — resolutions in flight (see
"Review resolutions" at end). Launch HELD until every BLOCKER **and** SHOULD-FIX is resolved + re-smoked
+ the cross-arch harness confirms the seed is viable vs v11 (owner: self-play errors compound across
every future generation, so nothing marginal is deferred).

## Objective

Take the bootstrapped new-architecture policy (which crushed the heuristic, WR ~0.97) and improve it
by **lowered-bar league self-play**, the recipe that produced the v8→v11 champions. The open question
this stage answers: **does the pointer-arch lineage, grown by self-play, surpass v11?** — measured by
the cross-arch eval harness (`docs/plans/cross_arch_eval_plan.md`), NOT by in-lineage WR (which is
saturated/self-referential).

## Seed

- **Warm-start + initial anchor:** `runs/train/bootstrap_pointer_arch_20260721_020848/checkpoints/ckpt_000000649.pt`
  — a proper schema'd training checkpoint (schema_version=2, policy+optimizer+league refs), directly
  `load_checkpoint`-compatible (NO bridge needed). Both `init_policy_checkpoint` (the learner) and
  `anchor_checkpoint_path` (the frozen anchor) point at this same checkpoint. The bootstrap plateaued
  at ~update 160; ckpt_649 is deep in that plateau, so it is a representative strong seed.

## Recipe (copy the v8/v11 winning lowered-bar ratchet verbatim — do NOT retune)

Model the config on `configs/selfplay_v11.yaml` (the latest validated self-play config). Binding
requirements — verify exact field names against `src/catan_rl/ppo/arguments.py`:
- **Self-play ON:** snapshot/league opponent enabled (`opponent_type: snapshot` / `snapshot_weight > 0`
  as v11 sets it), NOT heuristic.
- **Auto-reanchor ratchet:** bar **0.63**, sustained_checks **2**, check cadence **25**,
  min_games **150**, cooldown **75** (the exact v8 recipe that broke the plateau; NOT the mis-set 0.85
  that stalled v7).
- **auto_stop:** enabled, hard **400** updates-since-promotion / soft **250** + window median **0.55** /
  checks **8** (the plateau terminator shipped in v11).
- **Disk guard:** `min_free_disk_gb: 5.0` (the guard that — combined with the league-sidecar fix —
  now protects long runs instead of killing them).
- **keep_last_n 6**, **aux_value_coef 0.05** (pointer-arch D4 — verify carried), **device mps**,
  fresh **seed**, **run_name selfplay_pointer_arch**, **total_steps** a large runaway ceiling
  (auto_stop is the real terminator, not the step budget).
- **1v1 ruleset preserved** — engine unchanged; obs schema unchanged (this is the new-arch schema).

## Launch mechanism (the load-bearing lesson from earlier stages)

- Launch via **launchd** (`~/Library/LaunchAgents/com.catan.selfplay.plist`, already staged) —
  `python scripts/train.py --config configs/selfplay_pointer_arch.yaml`, WorkingDirectory = repo root,
  PYTHONPATH=src, Standard QoS (P-cores — NOT Background, which throttled BC to E-cores), MPS,
  log `runs/logs/selfplay.log`. NEVER launch via the Bash tool — the harness reaps its descendants.
- Preflight before `launchctl bootstrap`: config parses; ckpt_649 + its `.refs.json` + `league_store`
  exist; MPS free (no other training proc); free disk ≥ 10 GB; smoke (below) green.
- Verify at update 0: `warm-started learner from …ckpt_000000649.pt` AND `installed frozen anchor`
  both log; KL/entropy/losses in-band.

## Monitoring (CORRECTED per review S6 — the plan's first draft read TB-only metrics from the wrong file)

The bootstrap plateaued and I didn't catch it because the monitor grepped `wr_heuristic` but the eval
logs as `wr`. **Verified metric locations** (config-agent + review, `training_loop.py:635-636,1036-1041`):
- `metrics.jsonl` carries: `kind=train` (PPO metrics), `kind=eval` (has `wr` = WR vs **heuristic** only),
  **`kind=reanchor`** (fields `n_promotions`, `new_anchor_id`), **`kind=auto_stop`** (terminator).
- **TB-ONLY** (NOT in metrics.jsonl): `selfplay/anchor_wr_window` (the promotion-decision statistic),
  `selfplay/updates_since_promotion`, `selfplay/anchor_window_games`, `selfplay/reanchor_streak`.

So the monitor must:
- **grep `"kind": "reanchor"` and `"kind": "auto_stop"` and `"kind": "eval"` from `metrics.jsonl`** —
  promotions (the health signal: v8→v10 promoted 1→2→3), the terminator, and WR-vs-heuristic
  (a ≥0.85 forgetting tripwire, NOT a progress signal — it's saturated ~0.97).
- **read `selfplay/anchor_wr_window` + `selfplay/updates_since_promotion` from TensorBoard** — the
  real graduation gauge (window WR should climb toward >0.63 to fire reanchors).
- **Failure signatures:** NaN/Inf, entropy collapse toward 0, disk-abort marker, launchd non-restart.
- **The real yardstick vs v11: run the cross-arch eval (n=100) on each promotion-era checkpoint.**
  This is the ONLY signal that answers "surpassing v11" (in-lineage WR is self-referential). After
  each reanchor promotion, eval the latest promo ckpt vs `v11_cand_u724`; when one clears Wilson-LB
  > 0.50 at n=100, re-run at n=600 for the formal gate. Schedule these during lower-contention windows
  (NIT: the CPU-pinned eval competes with the CPU-pinned in-rollout snapshot opponents).

## Success / graduation criteria

- **Primary (the whole point):** a self-play checkpoint beats **v11** head-to-head, Wilson-LB > 0.50
  at n=600 (the ratified dual-gate clause a) AND the human-scoreboard opening metric ≥ v11's (clause b,
  now backed by the growing corpus, 107 rows). Neither is measurable until the cross-arch eval harness
  lands — building in parallel.
- **auto_stop** ends the run when it plateaus; the best promotion-era checkpoint is the candidate.

## Kill / invalidation criteria (kill + relaunch, don't limp)

- Warm-start or anchor line missing/wrong at update 0 → misconfigured seed → kill.
- Entropy collapses toward 0 in the first ~20 updates, or KL sustained ≫ 0.05, or NaN/Inf → unstable →
  kill, investigate LR/entropy_coef.
- **Zero reanchor promotions after ~150 updates with anchor-window WR stuck < 0.63** → the learner
  isn't beating its own anchor → the seed may already be near the arch's self-play ceiling → stop and
  reassess (this would itself be an informative result about the arch).
- Disk guard trips (should not, post-sidecar-fix, but the slim-fallback must produce a usable ckpt).

## Risks / open questions (for the review to pressure-test)

1. **Is warm+anchor-from-the-same-checkpoint right?** v11's config did this; but a learner anchored to
   itself must *beat itself* to promote — confirm the recipe expects the first promotions off the
   warm-started learner (the anchor is frozen; the learner moves). Reviewer: sanity-check.
2. **aux_value_coef / belief_coef in self-play** — these aux heads were validated in bootstrap; confirm
   they don't destabilize league self-play (belief target is opponent hidden dev types — valid vs
   snapshot opponents?).
3. **Runaway ceiling vs auto_stop** — confirm auto_stop actually fires for this lineage (it never got
   to fire on v11, which died of disk). The soft/hard thresholds are copied but unvalidated on the new
   arch.
4. **Compute reality** — self-play is many hours/generation on a contended M1; a full ratchet climb is
   days. This plan does not promise a v11-beating checkpoint on any timeline; it commits to *measuring*
   progress vs v11 honestly at each promotion.

## Review resolutions (2026-07-21)

Senior-RL review verdict **LAUNCH-AFTER-FIXES**. Every BLOCKER + SHOULD-FIX is being resolved (Opus);
launch is gated on all of them + a green re-smoke + the cross-arch seed sanity. Core recipe (warm+anchor,
ratchet bars, LR/entropy) was judged sound and faithfully copied.

| ID | Finding | Resolution |
|----|---------|-----------|
| **B1** (blocker) | ~2-week run has NO crash recovery: CLI always timestamps a fresh dir, no `--resume`, plist has no restart → a mid-run death silently discards league pool + optimizer + promotion clock + all promotions | Make it resumable: add `--run-dir` (stable, non-timestamped) so `maybe_resume_from_checkpoint` picks up `checkpoints/` on relaunch (restores policy+optimizer+league+RNG+clock); verified by kill+relaunch test. THEN add launchd `KeepAlive` crash-only (order matters — KeepAlive without resume is a restart-from-seed loop). |
| **S1/S2** | Seed ckpt_649's opening entropy collapsed (2.40→0.12); `setup_entropy_coef:0.0` won't re-widen; earlier higher-entropy checkpoints were **pruned** (only u449–649 remain) so reseeding isn't available | Enable `setup_entropy_coef > 0` (principled value from step6) to actively maintain opening diversity during self-play — a justified lineage-specific deviation. Plus a **pre-launch cross-arch sanity**: ckpt_649 vs v11 at n=100; if hopelessly behind, reassess before spending days. |
| **S3** | `auto_stop soft=250` < v10's first-promo latency (~275) can stop a healthy learner pre-promotion; dead band [0.55,0.63] → possible hard-400 stop with ZERO promotion ckpts | Raise `soft_updates_since_promotion` ≥300; VERIFY candidate-selection falls back to best rolling ckpt when promotions==0 (clean exit, usable candidate); optionally narrow dead band (soft bar ~0.58). |
| **S4** | `heuristic_weight ~0.25` spends 25% of rollouts on an opponent the seed beats 0.97 (degenerate advantages dilute self-play + train aux/belief on trivial states) | Lower to ~0.1 (keep >0 floor), reallocate to snapshot/anchor. |
| **S5** | `eval_every_updates:40` vs saturated heuristic is info-free but burns ~1000 serial CPU games/40 upd | Raise to ≥100 (or coarse forgetting-tripwire only). |
| **S6** | Monitoring section read anchor-window WR + updates_since_promotion from `metrics.jsonl` — but they are **TB-only**; the only `wr` in the jsonl is vs-heuristic | Monitoring section CORRECTED above (real keys: `kind=reanchor`/`kind=auto_stop` from jsonl; `anchor_wr_window` from TB). |
| NITs | `opp_action_coef` inert (no opp_action head); stale `keep_last_n` disk comment (16.7MB not 577MB); relative ckpt paths (cwd-dependent → launchd WorkingDirectory covers it) | Drop opp_action_coef; fix comment; rely on launchd WorkingDirectory. |
