# v11 — crown v10, then train UNBOUNDED-until-plateau (audited stop logic)

**Date:** 2026-07-14 · **Author:** Fable (audit + plan) · **Executor:** Opus, unattended
**User intent:** strongest possible model; no fixed update budget; run until the plateau check
says the lever is exhausted.

## Audit verdict on the existing plateau/promotion logic

**The ratchet is SOUND — do not touch its constants or logic.** `league.maybe_promote_anchor`
(windowed mean over `min_games=150` anchor outcomes, strictly > 0.63, `sustained_checks=2` with
streak reset on any sub-bar check, cooldown 75, window cleared on promotion) produced 1 → 2 → 3
promotions across v8→v9→v10 with zero false fires, and the 2026-07 audit already replaced the
noisy EMA statistic. Promotion cadence observed in v10: every ~250 updates (275/525/775), with the
final window still hovering 0.61–0.68 — v10 ended by BUDGET, not plateau, which is exactly why
this plan exists.

**Four real gaps for an unbounded run (all fixed below):**
1. **The kill rule is prose, not code** — nothing stops a plateaued run automatically.
2. **LR schedule breaks silently**: `lr_anneal_total_updates: 0` = "infer from total_steps", so a
   huge budget stretches the 2.0e-4 → 1.5e-4 anneal over the whole (near-infinite) run — a
   silent recipe change vs the validated v10 schedule.
3. **Checkpoint retention vs candidate selection**: the rolling `keep_last_n` window can PRUNE a
   promotion-era checkpoint on a long run — and v9's candidate WAS a promotion-era ckpt, not the
   final one. Also 16 × 577 MB ≈ 9.2 GB does not fit in the 8.6 GB currently free.
4. **No disk guard**: June precedent — a full disk TRUNCATED a `torch.save` mid-write and killed
   a run. An unbounded run must stop cleanly BEFORE that, never corrupt a checkpoint.

## STEP 0 — crown v10 + reclaim disk (~15 min)

1. `cp runs/anchors/v10_cand_u899.pt runs/anchors/v10_chain_u899.pt` (the champion file; keep the
   `_cand_` file too — other JSONs reference it).
2. Add the v10 rung to `scripts/elo_ladder.py` `RUNGS` exactly as v9 was added (`("v10_chain_u899",
   "policy", "runs/anchors/v10_chain_u899.pt")` at the top, one-line comment). Run the elo_ladder
   unit tests.
3. **Disk reclaim (explicit protect-list; verify each survivor exists AFTER):** in
   `runs/train/selfplay_v10_20260711_210552/checkpoints/`, DELETE all rolling checkpoints EXCEPT
   `ckpt_000000899.pt` (the champion source) and `ckpt_000000774.pt` (last-promotion era). NEVER
   touch `runs/anchors/*` or any `scripts/elo_ladder.py` RUNGS path (v6/v5/v3/bootstrap/ckpt_524
   files). Expected reclaim ≈ 7 GB (target ≥ 14 GB free before launch; abort launch if < 10 GB).
4. Commit: `feat(eval): crown v10 (v10_chain_u899) + add its elo-ladder rung`.

## STEP 1 — automated plateau stop (`auto_stop`; additive, default OFF) (~2 h)

New optional config block (in `src/catan_rl/ppo/arguments.py`, e.g. `AutoStopConfig` on
`TrainConfig`, ALL defaults preserving today's behaviour exactly — `enabled: false` means the
training loop is byte-identical to now; every existing config keeps working unchanged):

```yaml
auto_stop:
  enabled: true
  hard_updates_since_promotion: 400   # promotions came every ~250 in v10; 400 = 1.6x that
  soft_updates_since_promotion: 250   # earlier stop needs BOTH: no promotion for 250+ updates...
  soft_window_bar: 0.55               # ...AND the anchor window median clearly below the bar
  soft_window_checks: 8               # median over the last 8 promotion-cadence checks
```

Semantics (implement in `training_loop.py`, evaluated at the SAME cadence as the reanchor check,
immediately after `maybe_promote_anchor`):
- Maintain `updates_since_promotion` (reset to 0 on every promotion; starts at 0 at run start —
  the warm-started learner gets a full grace period). Emit it as a NEW TensorBoard scalar
  `selfplay/updates_since_promotion` (additive — never rename existing scalars).
- Keep the last `soft_window_checks` values of the anchor-window mean (the same
  `anchor_window_stats()` number the ratchet uses; record only when the window is FULL).
- **HARD stop**: `updates_since_promotion >= hard_updates_since_promotion`.
- **SOFT stop**: `updates_since_promotion >= soft_updates_since_promotion` AND the recorded
  window means number >= `soft_window_checks` AND `median(last soft_window_checks) < soft_window_bar`.
- On either: log `AUTO-STOP (<hard|soft>): <numbers>`, save the terminal checkpoint via the
  existing terminal-save path, exit the update loop cleanly (identical exit path to budget
  exhaustion — TB writer closed, league state saved).
Rationale for the numbers: v10's promotion gaps were ~250 updates; 400 = 1.6× the observed gap so
a healthy-but-slow learner is not cut early (a near-bar hover like v10's final stretch KEEPS
training — that is the "strongest model" bias the user asked for), while the soft clause kills a
clearly-dead run (median < 0.55 ≈ v9's terminal 0.507-falling pattern) ~150 updates sooner.

**Tests (write first, `tests/unit/ppo/` or wherever the training-loop units live — follow the
existing monkeypatched-stage style):** hard fires at exactly 400 and not at 399; soft fires only
when BOTH clauses hold (247 updates + low median → no; 251 + median 0.56 → no; 251 + median 0.54 →
yes); promotion resets the counter and clears the recorded medians (a promotion mid-hover must
restart the clock); `enabled: false` → the evaluation function is never invoked (assert via
monkeypatch counter); the stop uses the SAME terminal-save path (assert the checkpoint exists).

Commit: `feat(ppo): auto_stop — automated plateau termination for unbounded runs`.

## STEP 2 — promotion-time permanent checkpoints + disk guard (~1 h)

1. **Promo checkpoints**: when `maybe_promote_anchor` returns a new anchor id, save the learner to
   `checkpoints/promo_ckpt_{update:09d}.pt` via the existing save machinery, EXEMPT from the
   rolling `keep_last_n` pruning (the pruner must match only the `ckpt_` prefix — verify its glob
   and add a test: promo files survive a prune that evicts older rolling ckpts). These are the
   candidate-selection insurance for a run whose length nobody knows in advance.
2. **Disk guard**: immediately BEFORE every checkpoint save (rolling, promo, terminal), check free
   space on the run dir's filesystem (`shutil.disk_usage`); if < `min_free_disk_gb: 5.0` (new
   optional field, default 0 = disabled), log CRITICAL and trigger the SAME clean auto-stop exit
   (terminal save is attempted ONLY if space allows a full write, else skipped with a loud log —
   never risk the June torch.save truncation). Test with a monkeypatched `disk_usage`.

Commit: `feat(ppo): promotion checkpoints + free-disk guard for long runs`.

## STEP 3 — v11 config + launch (~30 min)

`configs/selfplay_v11.yaml` = v10's config verbatim (same league block: bar 0.63 / sustained 2 /
check 25 / min_games 150 / cooldown 75; same ppo/gae/loss/optimizer) with EXACTLY these changes,
each with a one-line comment:
- `anchor_checkpoint_path: runs/anchors/v10_chain_u899.pt` and
  `init_policy_checkpoint: runs/anchors/v10_chain_u899.pt` (warm + anchor on the crowned v10).
- `total_steps: 196608000`  # 6000 updates x 128 x 256 — a RUNAWAY CEILING (~2 weeks), not a
  budget; auto_stop is the real terminator.
- `lr_anneal_total_updates: 900`  # pin v10's validated schedule: 2.0e-4 -> 1.5e-4 over the first
  900 updates, then HOLD at lr_end (verify the trainer holds at lr_end past the horizon — read the
  anneal code; if it extrapolates below lr_end, clamp it and add a test).
- `checkpoint.keep_last_n: 6`  # 6 x 577 MB ≈ 3.5 GB rolling + promo ckpts; fits the disk budget
- `auto_stop: {enabled: true, hard: 400, soft: 250, bar: 0.55, checks: 8}` (real field names)
- `min_free_disk_gb: 5.0`, `seed: 10`, `run_name: selfplay_v11`.

Preflight (same as v10's): anchor file exists; config parses; NO other training process alive;
free disk ≥ 10 GB. Launch detached: `nohup caffeinate -is python scripts/train.py --config
configs/selfplay_v11.yaml --run-name selfplay_v11 > runs/logs/selfplay_v11_launch.log 2>&1 &`.
Verify update 0 lands and `warm-started learner from .../v10_chain_u899.pt` + `installed frozen
anchor` both appear in the log, KL/entropy in the usual bands, then report.

Commit the config: `feat(selfplay): v11 unbounded run — auto_stop plateau termination`.

## House rules + stop conditions

Fail-closed / additive: `auto_stop` and `min_free_disk_gb` default OFF — every existing config
byte-identical (test-pinned). Never rename TB scalars. Test-first; ruff + mypy --strict + full
pytest green per commit; stage by explicit path (other sessions' scratch is in the tree); no AI
trailers; push origin/main. The deleted v10 rolling ckpts are IRREVERSIBLE — triple-check the
protect-list (`ckpt_000000899.pt`, `ckpt_000000774.pt`) before `rm`, and `ls` the survivors after.
STOP and report instead of improvising if: the LR-hold semantics are not what the plan assumes,
the pruner cannot exempt promo files cleanly, or preflight disk < 10 GB even after the reclaim.

## Explicitly out of scope (do not do)

- No ratchet-constant tuning (bar/cooldown/window are validated across 3 generations).
- No n=2000 v6 re-verify launch (open gate debt, separately decided).
- No grant/corpus work (a different executor owns it).

## Implementation notes (2026-07-14, shipped)

Executed as planned; commits `39f0bb7` (STEP 0), `185522b` (STEP 1), `5247eed` (STEP 2),
`98311bb` (STEP 3 config). No divergences from the design; a few concretions worth recording:

- **LR-hold: no code change needed.** `linear_lr_schedule` already clamps
  `progress = min(1.0, update_idx/(total_updates-1))`, so it HOLDS at `lr_end` past the
  horizon (verified: updates 900/901/2000/5999 all resolve to exactly 1.5e-4 with
  `lr_anneal_total_updates=900`). No clamp added; the plan's "if it extrapolates" branch
  did not fire.
- **Pruner exemption by construction.** The rolling pruner's glob is `^ckpt_(\d{9})\.pt$`;
  promotion checkpoints are written as `promo_ckpt_NNNNNNNNN.pt` (new
  `CheckpointManager.save_promotion` / `promotion_checkpoint_filename`), so they never match
  `list_checkpoints` / `prune_checkpoints` and survive `keep_last_n` with no exemption
  bookkeeping. Pinned by `test_promotion_checkpoint_survives_prune`.
- **auto_stop = additive `AutoStopConfig` block on `TrainConfig`** (`enabled: false` default =
  byte-identical). Evaluated at the reanchor-check cadence immediately after
  `maybe_promote_anchor` via `AutoStopTracker` (counter ticked per completed update, reset on
  promotion; window mean recorded only when the ratchet window is full). A stop sets a
  `stop_reason` that the while-condition honours, exiting into the SAME terminal-save path as
  budget exhaustion. Added a coherence guard: `auto_stop.enabled` requires
  `league.auto_reanchor_enabled` (else the counter could never reset).
- **Disk guard = top-level `min_free_disk_gb` field** (default `0.0` = disabled), checked
  before every save (rolling/promo/terminal) via `shutil.disk_usage`; a trip logs CRITICAL,
  SKIPS the write, and requests the same clean auto-stop exit.
- **New additive TB scalars** (existing names untouched): `selfplay/updates_since_promotion`
  (every update) and `selfplay/auto_stop_event` (on a stop).
- **v11 launched** from `configs/selfplay_v11.yaml` (6000-update runaway ceiling, warm+anchor
  on `v10_chain_u899`, seed 10) under `nohup caffeinate -is`; disk after reclaim = 14–15 GB
  free (target ≥14, abort <10 both satisfied). auto_stop is the real terminator.
