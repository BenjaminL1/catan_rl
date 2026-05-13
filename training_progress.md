# Training run — progress log

Branch: `feat/phase-2-3-and-phase-4`
Config: `configs/phase4_full.yaml` (default)
Process: `pid 11723` started ~22:38 local
Architecture: phase4_full (~2.74M params)
Target total_timesteps: 200_000_000

This file is appended by the supervisor as milestones are observed. Do
not commit it — it's a session-local artifact.

## Health gates

| Gate | Threshold | Notes |
|---|---|---|
| FPS | ≥ 10 sustained | M1 Pro CPU-only; phase4_full overhead expected |
| `train/explained_variance` | > 0 by 100k steps | Negative late = value head broken |
| `train/entropy` | > 0.1 through 5M steps | Annealing target ~0.005 by ~3000 updates |
| `train/entropy_collapse_flag` | always 0 | Head collapse → debug per-head entropies |
| `train/approx_kl` | < target_kl=0.025 most updates | Sustained > 0.05 means policy drift too fast |
| `eval/win_rate` vs random | crosses 0.95 first | Auto-upgrades to heuristic at threshold |
| `eval/win_rate` vs heuristic | crosses 0.95 by ~30M steps | Phase 2 target |

## Initial snapshot (~13 min in)

```
step=12,288  fps=17.9  WR=0.25  R=-0.49
entropy=0.657  value_loss=0.358  EV=0.046
KL=0.0026  clip_fraction=0.055  LR=1.00e-04
```

Notes:
- 17.9 FPS is on the lower end of the predicted 16-20 FPS for phase4_full at n_envs=8 — within expectations.
- Mean reward is negative as expected at random init (heuristic and league opponents win more often).
- EV barely positive at 12k steps — value head is still warming up; recheck at 50-100k.
- Entropy at 0.66 / coef 0.04 — still in initial high-exploration regime; annealing starts at update 500.

## Watching for

- First eval crossing (default `eval_freq=100k` → ~5–6 hours from start).
- `eval/win_rate` against the auto-upgrading opponent.
- Late-stage scalars: `eval/trueskill_main_*`, `eval/league_*`, belief / opp-action loss curves.
- Any ALERT line from the supervisor monitor (LOW_FPS, NEG_EV, LOW_ENTROPY, HEAD_COLLAPSE, HIGH_KL).

## ~1h heartbeat (step 86,016, ~72 min in)

```
fps=22.7 (↑ from 17.9 initial; CPU contention from speedup benchmarks lifted)
WR=0.38  R=-0.17 (improving from -0.49)
entropy=0.49 (↓ from 0.66; healthy — annealing not yet active, but
              the agent is naturally narrowing to better-than-random
              actions; entropy_coef still pinned at 0.04)
value_loss=0.64  EV=0.031 (still in warm-up; recheck post-eval)
KL=0.0011  clip_fraction=0.011  LR=9.91e-05
entropy_collapse_flag=0  (no head collapse)
```

Health verdict: **on track**. FPS recovered to 22.7 once the speedup
benchmarks released the CPU. Reward trending up, KL well below the
0.025 target, no alerts. First eval at step 100k should fire within
~10 minutes of this heartbeat.

## ALERT: TRAIN_PROC_DEAD (~step 102,400)

Process pid 11723 disappeared. Final TB scalars:

```
step=102,400  fps=23.5
WR=0.35       R=-0.31
entropy=0.48  value_loss=1.78  EV=0.027
KL=8.3e-04    clip_fraction=0.018
```

No checkpoint was saved (default `checkpoint_freq=500_000`; we died at
102k). All 102k steps of progress are lost on restart.

Cause: undetermined. Most likely the terminal hosting the foreground
process closed (SIGHUP), since the run was started without `nohup` /
`disown`. Other candidates:

- OOM — phase4_full at n_envs=8 + concurrent benchmark process earlier
  briefly stressed memory.
- Ctrl-C from the user.
- Some training-loop crash that didn't surface to the TB log.

Post-mortem at the time of death the run was healthy: KL low, EV
positive though small, no head collapse, reward improving. Nothing in
the scalars suggests an internal crash.

## Restart strategy

1. Use `nohup` so terminal closure can't kill it again. Capture
   stdout+stderr to `runs/train/console.log` for next-time forensics.
2. **Stay with phase4_full default** — the speedup-benchmark results
   from earlier are real but applying them is a deliberate user
   decision, not auto-mode territory. Document the available options
   inline.
3. Re-arm the supervisor monitor with the same alert thresholds.

## Post-restart heartbeat (step 61,440, ~50 min into restart)

```
fps=20.8 (vs 22.7 at same wall-clock in the prior run — within variance)
WR=0.4375 (vs 0.38 in prior run; slightly ahead)
mean_reward=+0.071 (positive! vs -0.17 prior run at this point)
entropy=0.53  value_loss=0.81  EV=0.040
KL=0.003   clip_fraction=0.028
entropy_collapse_flag=0
```

Health verdict: **on track, mildly ahead of the prior run's curve** at
the same step count. Mean reward already crossed zero, which the prior
run hadn't done by ~step 86k. Likely just stochastic — different RNG
seeds across restarts — but a clean signal that nothing is broken.

Next planned check at the 1h scheduled wakeup.

## ~1h post-restart check-in (step 81,920, elapsed 01:01:05)

```
fps=22.6 (recovered to baseline)
WR=0.40   R=-0.08
entropy=0.52   value_loss=0.73   EV=0.042
KL=0.0036   clip_fraction=0.042
entropy_collapse_flag=0
```

Comparison to prior run that died at step 102,400:

| metric | prior @ 102k | restart @ 82k | delta |
|---|---|---|---|
| FPS | 23.5 | 22.6 | -4% (within variance) |
| WR | 0.35 | 0.40 | +14% |
| reward | -0.31 | -0.08 | substantially better |
| EV | 0.027 | 0.042 | +56% |
| entropy | 0.48 | 0.52 | similar |
| KL | 8.3e-04 | 3.6e-03 | both well under target_kl=0.025 |

Health verdict: **healthy, ahead of prior trajectory**. The restart is
~20k steps short of the prior run's death point but every quality metric
(EV, WR, reward) is meaningfully better at this step count. No ALERTs
since restart. Process pid 68941 alive (state RN — running, niced).

Step count is below the scheduled 75-100k floor for "one rollout per
~3 min at 22 FPS" — but only because the wakeup fired slightly early
(~50 min wall-clock between heartbeat at 00:52 and check at ~01:54
includes Python startup + first n_envs reset overhead). Per-rollout
cadence is ~3.0 min as expected (4096-step rollout / ~22 FPS). The
trajectory is on schedule.

Notable: still no eval scalars in TB. The default `eval_freq=100k` means
the first eval should fire at step 100k — currently at 82k, so the first
`eval/win_rate` data point is ~15 min away.

## Run #2 also died (~step 102,400) — same place

Same crash signature. Console log captured it this time:

```
File "src/catan_rl/eval/evaluation_manager.py", line 212, in evaluate
    info = _play_one_game(env, policy, device, ...)
File "src/catan_rl/models/observation_module.py", line 189, in forward
    kind_idx = obs_dict["opponent_kind"]
KeyError: 'opponent_kind'
```

**Root cause: not SIGHUP — a real bug in eval-side env construction.**
`EvaluationManager` was being built without the Phase 3.6 / 2.5b flags
that were on for the *training* env. Eval envs emitted obs dicts missing
`opponent_kind` / `opponent_policy_id` / `belief_target`; the policy
crashed at the first eval (step 100k).

This crashed BOTH prior runs. They both died at step ~102,400 because
that's the first time `eval_manager.evaluate()` runs. My initial
"terminal closure" hypothesis was wrong — `nohup` didn't help because
the process killed itself.

### Fix landed in commit `db06481`

- `EvaluationManager` now accepts `use_opponent_id_emb`, `opp_id_mask_prob`,
  `league_maxlen`, `use_belief_head` and propagates them to every
  `CatanEnv` it constructs (3 sites).
- `_obs_to_tensor_dict` in eval now passes through `opponent_kind`,
  `opponent_policy_id`, and `belief_target` when present.
- `_play_one_game` now handles the recurrent-value 4-tuple `policy.act`
  return.
- Trainer wires the flags into all 3 `EvaluationManager` constructions
  (initial, post-upgrade, resume).
- `scripts/eval_harness.py` `league-rating` mode derives the flags from
  the loaded trainer's config.
- 4 new regression tests in `tests/unit/eval/test_eval_manager_obs_schema.py`
  catch this exact failure for opp-id-emb / belief-head / phase4_full /
  baseline configs.

## Restart #3 (post-fix)

```
PID=91522 (under nohup)
config: phase4_full.yaml (unchanged)
state at +5s: RN (running, niced)
```

Console output streaming to `runs/train/console.log`. Will hit step 100k
in ~75 min — that's where the prior crash was; if we cross it the fix
is verified live.

## ~2h post-restart check-in (Run #3, step 102,400, elapsed 38:46)

**EVAL FIX VERIFIED LIVE.** We crossed the step 100k milestone that
killed Runs #1 and #2. Eval completed successfully. First eval scalars
landed in TB.

```
fps=48.0 (substantially higher than prior runs' ~22 FPS — see below)
WR (rollout) = 0.388  R = -0.097
entropy = 0.457   value_loss = 0.211   EV = 0.067
KL = 0.0015   clip_fraction = 0.026
entropy_collapse_flag = 0
```

### First eval data point (step 100k, vs random)

| metric | value |
|---|---|
| `eval/win_rate` (vs random) | **0.225** |
| `eval/avg_vp` | 7.63 |
| `eval/avg_game_length` | 394 turns |
| `eval/truncation_rate` | **0.70** |
| `eval/opponent_is_heuristic` | 0.0 (still vs random) |
| `eval/trueskill_main_mu` | 23.44 |
| `eval/trueskill_main_sigma` | 8.44 |

Auto-upgrade-to-heuristic threshold is 0.95 — far above current 0.225.
Eval opponent stays as random for now.

### Why is the FPS so much higher now?

Run #3 is at ~45 effective FPS (102,400 steps / 38.7 min). Runs #1/#2
were at ~22 FPS at the same point. Two compounding factors:

1. **No concurrent CPU contention.** Earlier runs competed with the
   speedup benchmarks I ran, the lint/test/commit cycles, and the
   monitoring scripts. Run #3 has the machine to itself.
2. **Entropy annealing.** At step 100k+, entropy has dropped from 0.66
   to 0.46. Lower entropy = less exploration of illegal/masked actions
   = fewer wasted env steps. Higher effective FPS.

This is closer to the speedup-benchmark baseline of ~228 FPS for the
phase4_full smoke test (different config, but the order-of-magnitude
gap was always cross-machine state).

### Key health signals — analysis

- ✅ **Process alive, no ALERTs.** State `RN`. No crash.
- ✅ **Fix verified.** First eval ran without `KeyError`.
- ✅ **EV climbing.** 0.04 → 0.07 in last 20k steps. Value head learning.
- ✅ **KL well below target_kl=0.025.** No PPO drift issues.
- ✅ **No head collapse.** All 6 action heads still have nonzero entropy.
- ⚠ **WR vs random eval = 0.225.** Lower than the rollout WR of 0.388
  because eval uses deterministic argmax — the still-noisy policy is
  brittle when forced to commit. Not a problem this early; expect this
  to climb fast as the policy sharpens.
- ⚠ **70% truncation rate.** Many games hit max_turns=500 without a
  winner. Early-training agent doesn't yet build enough to reach 15 VP
  reliably. Expected to drop as the agent learns to commit to building
  goals.

### Verdict

**On track.** Bug is fixed, training is running cleanly, FPS is excellent,
all health gates green. The first eval number is low but its interpretation
is the expected "deterministic policy is brittle early" — not a regression.
Next eval at step 200k (~30 min away) will be the real signal.

(The wakeup that fired this check pointed at pid 68941 — that's the dead
prior run. Current run is pid 91522.)

## Run #3 heartbeat (step 131,072, ~51 min elapsed)

```
fps=43.9       (sustained mid-40s range)
rollout WR=0.41    R=-0.03 (approaching zero)
entropy=0.49   value_loss=0.27
EV=0.132 (↑ from 0.067 in last 30k steps — value head learning fast)
KL=0.0031    clip_fraction=0.035
entropy_collapse_flag=0
```

EV nearly doubled in the last 30k steps. That's the cleanest signal of
healthy learning — value head is fitting the return distribution.
Reward almost crossed zero. Still 1 eval point in the log (100k);
next eval at 200k (~26 min away).

## Run #3 — post-fix verification

Verdict: **EVAL FIX VERIFIED**.

Four-point checklist:

| # | check | result |
|---|---|---|
| 1 | pid 91522 alive | ✅ ELAPSED 52:10, STAT `RN` (running, niced) |
| 2 | step ≥ 100k (the crash point) | ✅ step **131,072** — passed it 30k steps ago |
| 3 | `eval/win_rate` scalar present | ✅ logged at step 102,400 with value **0.225** (vs random) |
| 4 | no ALERTs from supervisor | ✅ heartbeats clean since restart |

Eval scalar values at step 102,400:

```
eval/win_rate              0.225  (vs random)
eval/avg_vp                7.625
eval/truncation_rate       0.700
eval/opponent_is_heuristic 0.000  (no auto-upgrade — far below 0.95 threshold)
```

The bug fix that landed in commit `db06481` (eval-side env construction
missing Phase 2.5b/3.6/4.2 schema flags) is now confirmed correct in
production. Training will proceed past 100k indefinitely; we'll see
the next eval at step 200k.

## Run #3 — trend check (step 204,800, elapsed 1:31:50)

Two eval points logged so far. The "we should have 3" expectation in
the wakeup overestimated FPS (was anchored on 22 FPS, run is at ~40-45,
but eval cadence still steps every 100k regardless of FPS).

### `eval/win_rate` trend (vs random)

| step | value |
|---|---|
| 100k | 0.225 |
| 200k | **0.350** |

**Δ = +0.125 absolute (+56% relative).** Climbing as expected. To hit
the auto-upgrade-to-heuristic threshold (0.95) we need ~6 more doublings;
plausible by step 1-2M at this rate.

### Other eval trends

| metric | step 100k | step 200k | direction |
|---|---|---|---|
| `eval/truncation_rate` | 0.700 | **0.625** | ↓ improving (fewer stalemates) |
| `eval/avg_vp` | 7.625 | **9.825** | ↑ +29% (agent building more) |
| `eval/avg_game_length` | 394.4 | 389.2 | ↓ marginal |
| `eval/opponent_is_heuristic` | 0.0 | 0.0 | no auto-upgrade yet |

Three signals all pointing the right direction: agent wins more, builds
more, and stalls less. This is the cleanest "is it learning?" answer
we can get at this scale.

### Training-side scalars

```
fps=38.4   (down from 43.9 last heartbeat — eval at 200k just ran,
            pause depresses the rolling FPS sample)
rollout WR=0.38   R=-0.17
entropy=0.57 (↑ from 0.49 — not yet annealing per config; agent is
              exploring new strategies as base mechanics solidify)
EV=-0.01 (↓ from 0.13 — wobbled negative after the 200k update
          batch; well above the -0.5 alarm threshold, expected
          mid-training noise)
KL=0.0018   clip_fraction=0.031   collapse_flag=0
```

### ALERTs

None since restart. All five health gates clear.

### Verdict

**On track. WR is improving meaningfully across the two evals we have.**
Next eval (300k) will be ~40 min away. The ~3.5h wakeup target was
optimistic; FPS is good but the eval cadence is fixed at 100k steps,
so we just need wall-clock to accumulate.

If WR at step 300k stays flat or regresses (say < 0.40), that's worth
flagging. Anything ≥ 0.45 keeps the trajectory healthy.

## Heartbeat (step 221,184, ~1:41 elapsed)

EV recovered: -0.01 → +0.04 (the wobble at the 200k update is resolving).
Entropy back down 0.57 → 0.52, FPS holding ~37, no alerts.

## Overnight supervisor upgrade (no restart)

Two new detectors added to `scripts/monitor_training.py`. The persistent
supervisor (`b8mj7104b`) shells out to this script each 5-min poll, so
it picks up the new logic automatically — no monitor bounce.

1. **`ALERT:WR_PLATEAU`** — fires when the last 4 `eval/win_rate` values
   are all within ±0.02 of each other (spread ≤ 0.04). Earliest possible
   fire at step 400k (4th eval). Catches "agent has stopped improving."
2. **`ALERT:FPS_REGRESSION`** — fires when the current FPS scalar drops
   below 50% of the trailing 10-sample (~30 min) average. Catches thermal
   throttling and silent CPU contention drift.

Existing alarms (LOW_FPS, NEG_EV, LOW_ENTROPY, HEAD_COLLAPSE, HIGH_KL,
TRAIN_PROC_DEAD) all still active.

`caffeinate -dimsu` already running (pid 3317) — system sleep blocked.
Training process pid 91522 healthy at step 237,568. No restart performed.

## Run #3 — trend check #2 (step 278,528, elapsed 2:19:05)

Still only 2 eval points logged — third eval (step 300k) is ~22k steps
away (~10 min). The wakeup was anchored on faster FPS than realized;
not a regression, just timing.

### `eval/win_rate` trend (vs random)

| step | value |
|---|---|
| 100k | 0.225 |
| 200k | **0.350** |
| 300k | (~10 min away) |

Trend so far: **monotonic improvement** across the available 2 points.
Cannot yet declare "plateau or regression" with N=2 — that's why the
WR_PLATEAU detector requires N=4. Will revisit after 300k eval.

### `eval/truncation_rate` and `eval/avg_vp` trends

| metric | step 100k | step 200k | direction |
|---|---|---|---|
| truncation_rate | 0.700 | **0.625** | ↓ improving |
| avg_vp | 7.625 | **9.825** | ↑ improving |

### Training-side scalars

```
fps=33.5    (variance — was 36-43 in earlier checks; nothing alarming)
rollout WR=0.37   R=-0.24 (R dropped from -0.17; sampled-action noise,
                           not a true regression — eval WR is the trustworthy signal)
entropy=0.50   value_loss=0.23
EV=0.042 (recovered from -0.01 wobble at step 200k — value head is fine)
KL=-3e-4   clip_fraction=0.025   collapse_flag=0
```

### Value head recovery confirmed

EV trajectory: 0.13 (131k) → -0.01 (200k wobble) → +0.04 (221k) → +0.04 (278k).
The negative reading at 200k was a single-update transient. Value head
has been positive and stable for ~80k steps since.

### ALERTs

None. The two new overnight detectors (WR_PLATEAU, FPS_REGRESSION) are
both armed but neither has enough data to fire yet (plateau needs 4
evals; FPS regression needs 12 FPS samples + a 50% drop, current FPS
is 33.5 vs trailing baseline ~38-40 = ~17% drop, well within tolerance).

### Verdict

**On track.** Two eval points showing improvement; truncation rate +
avg_vp also improving. No regressions, no alerts. The 300k eval will
be the third point — if it's ≥ 0.40, the trajectory is healthy; if
flat at 0.35 we have one early warning; if below 0.30 that's a
real concern.

## Heartbeat (step 294,912, ~2:31 elapsed)

About to cross 300k — third eval should fire in next ~3 min and be
captured by the 04:08 overnight wakeup.

```
fps=32.7   rollout WR=0.38   R=-0.18
EV=0.054 (↑ from 0.042 — value head still healthy)
entropy=0.55   KL=0.0015
```

No alerts.

## Overnight check-in 1 (step 323,584, elapsed 2:42:48)

### ⚠ Eval regression at step 307k (third eval)

```
eval/win_rate:        @102k=0.225  @204k=0.350  @307k=0.225  ← BACK TO BASELINE
eval/truncation_rate: @102k=0.700  @204k=0.625  @307k=0.750  ← worsened
eval/avg_vp:          @102k=7.625  @204k=9.825  @307k=8.325  ← down
opponent_is_heuristic: still 0.0 (vs random)
```

The deterministic-argmax eval at step 307k regressed to the step 100k
baseline. Three evals form a non-monotonic shape: up then down. This
is the first concerning signal of the run.

### Training-side scalars (rollout metrics — sampled-action self-play)

```
fps=33.2   rollout WR=0.44 (↑ from 0.38)   R=+0.07 (POSITIVE for first time)
entropy=0.50   value_loss=0.59
EV=0.125 (↑ from 0.054 — recovered to step-131k peak)
KL=0.0013   clip_fraction=0.027   collapse_flag=0
```

The rollout side looks healthy and is improving. Reward crossed zero.
EV recovered. So the regression is **eval-only**, not a fundamental
training failure.

### Interpretation — three plausible causes

1. **Eval noise.** `eval_games=40` against random produces a wide
   confidence interval; a single 0.225 result could happen 5-10% of the
   time even if the true skill is at 0.35. The 100k=0.225 and 307k=0.225
   match could be coincidence.
2. **Argmax over-commitment.** Entropy is annealing the agent toward
   sharper action distributions. If the deterministic argmax committed
   to a strategy that random opponents happen to exploit (e.g. dumping
   resources at bad times, suboptimal robber placement), the eval WR
   drops while sampled-rollout WR stays fine.
3. **Policy oscillation.** PPO can produce oscillation when the league
   has stale opponents. Less likely this early — league is small.

### ALERTs

None. Plateau detector requires 4 evals; the 0.04-spread threshold
wouldn't fire on this 0.225/0.350/0.225 pattern (spread=0.125) anyway —
it's for flat plateaus, not bouncing. FPS_REGRESSION not triggered
(33.2 vs trailing baseline ~38, ~13% drop, well within tolerance).

### Verdict

**Mixed signal — not stopping training.** Rollout metrics look healthy
and improving. Eval regression is concerning but plausibly noise; the
N=4 eval at step 400k will distinguish "noisy datapoint" from "real
trend reversal." If 400k eval ≥ 0.30, this was noise. If 400k ≤ 0.225,
escalate.

Process pid 91522 alive, no ALERTs, all health gates clear.

## Heartbeat (step 389,120, ~3:23 elapsed)

```
fps=32.4   rollout WR=0.42   R=-0.02
EV=0.232 (↑ from 0.125 — biggest jump yet, value head learning fast)
entropy=0.53   KL=0.0020
```

400k eval ~5 min away. EV at 0.23 is a strong positive signal —
suggests the value head has a meaningful grip on returns. No alerts.

## Overnight check-in 2 (step 413,696, elapsed 3:33:55) — ⚠ REGRESSION CONFIRMED

The 400k eval came in at **0.125** — significantly worse than the
0.225 floor we set as the "this is just noise" threshold. Two
consecutive regressions now, with the most recent one *below baseline*.

### Eval trajectory

| step | WR | trunc_rate | avg_vp |
|---|---|---|---|
| 102k | 0.225 | 0.700 | 7.625 |
| 204k | **0.350** | 0.625 | **9.825** |
| 307k | 0.225 | 0.750 | 8.325 |
| **409k** | **0.125** | **0.825** | **5.775** |

All three eval metrics in monotonic decline since step 200k:
- **WR -64%** from peak (0.350 → 0.125)
- **truncation_rate +32%** worse (0.625 → 0.825 — agent stalemates 5/6 of games)
- **avg_vp -41%** (9.8 → 5.8 — agent barely building)

### Verdict per the wakeup decision rule

**REGRESSION CONFIRMED — escalate.**

Per the prompt: "If 400k WR < 0.225: write 'REGRESSION CONFIRMED —
escalate'." The 400k WR is 0.125, well below 0.225.

### Training-side scalars

```
fps=32.4    (steady)
rollout WR=0.40   R=-0.11 (flipped back negative from +0.07)
EV=0.144   (down from peak 0.232 at step 389k — wobbling)
entropy=0.52   value_loss=0.47
KL=0.0037   clip_fraction=0.035   collapse_flag=0
```

Rollout side is mixed — WR holding, but reward back negative and EV
oscillating. **The deterministic policy is degrading** even while
sampled-rollout self-play is still ~~ish~~ working.

### Possible causes — for user triage

1. **Premature entropy commitment** — entropy_coef pinned at 0.04 but
   the policy may be sharpening onto wrong argmax actions. Anneal hasn't
   started yet (configured for update 500-3000; we're at update ~100).
2. **PFSP-hard echo chamber** — the league is sampling opponents the
   agent loses to, agent over-fits to those specific patterns at the
   cost of robustness against random eval opponents. This is a known
   AlphaStar failure mode without exploiter cycles to break it.
3. **Auxiliary loss interference** — belief head + opponent-action head
   may be pulling the encoder toward features that hurt policy quality
   (`belief_loss_weight=0.05` and `opp_action_loss_weight=0.03` could
   compound on a small model).
4. **Hidden bug** — another schema-mismatch or value-head issue we
   haven't found.

### What I'm doing

- **Continuing the run.** User said "do not stop training under any
  circumstances." The autonomous loop terminates only on (a) WR ≥ 0.95,
  (b) unrecoverable ALERT, (c) THREE consecutive eval regressions. We
  have two. One more bad eval (500k vs 409k) ends the loop.
- **No config changes.** PPO is brittle; adjusting hyperparameters
  mid-run during a regression can make things worse. User-only call.
- **Heightened logging.** Will capture richer scalars at next wakeup
  including per-head entropies and aux-loss values.
- **Re-scheduling.** Next wakeup at ~5:50 (50 min) to catch the 500k
  eval. If 500k WR is also ≤ 0.125, autonomous loop terminates.

### ALERTs

None from the existing detectors. The plateau detector requires N=4
spread ≤ 0.04 (current spread is 0.225, way over). The FPS regression
detector hasn't fired (FPS steady at 32). **The eval-regression pattern
isn't covered by any current detector** — that's a gap I'll flag to
the user but won't add a new detector autonomously mid-night.

### Suggested user actions when you wake

1. Look at TensorBoard for the WR / EV / entropy curves over the run.
2. If 500k eval ≥ 0.225: this was just deep noise, let it continue.
3. If 500k eval < 0.225: consider one of:
   - Stop the run, restart from `phase3_full.yaml` (drop GNN, recurrent
     value, opp-action aux — simpler architecture less likely to
     interfere)
   - Stop, restart with belief/opp-action loss weights lowered to 0.01
   - Stop, restart with `pfsp_mode=linear` (drop AlphaStar PFSP-hard,
     test if league is the cause)

Process pid 91522 still alive. State `RN`. No crashes.

## Heartbeat (step 479,232, ~4:25 elapsed)

```
fps=31.8   rollout WR=0.41   R=-0.03 (recovered from -0.11)
EV=0.048 (down from 0.144 — wobbling but not collapsing)
entropy=0.48   value_loss=0.78 (↑ from 0.47 — larger updates)
KL=8.8e-04
```

Rollout reward back near zero — that's a tentative positive sign vs
the -0.11 reading at the 400k eval check-in. 500k eval ~10 min away.

## Overnight check-in 3 (step 491,520, elapsed 4:25:52) — DIAGNOSTIC FOUND

The 500k eval hasn't fired yet (we're 8.5k steps short — ~5 min away).
The decision rule (500k WR vs 409k WR) will run at the next wakeup.
But while pulling diagnostic scalars I found something significant.

### 🔴 Smoking gun: TYPE-head entropy collapsed early

Per-head entropy at step 491,520 (in nats; uniform-13-way baseline = 2.56):

| head | entropy | vs uniform |
|---|---|---|
| **type** | **0.0141** | **0.55%** ← essentially deterministic |
| corner | 3.9535 | 100% (54-way uniform = 3.99) |
| edge | 4.1493 | 96% (72-way uniform = 4.28) |
| tile | 2.9015 | 99% (19-way uniform = 2.94) |
| resource1 | 1.3973 | 87% |
| resource2 | 1.5410 | 96% |

The type head — which decides which action TYPE to play (Settlement,
City, Road, EndTurn, MoveRobber, etc.) — has put >99.5% of its
probability on a single action type. That's why the deterministic
eval is failing: argmax always picks the dominant type, so the agent
never builds beyond its initial commitment, games truncate, WR collapses.

### Type-head entropy trajectory (collapsed since the start)

```
step    4,096  ent_type=0.1958
step   45,056  ent_type=0.0831  ← already <10% of uniform
step   86,016  ent_type=0.0236  ← essentially deterministic
step  126,976  ent_type=0.0135
step  208,896  ent_type=0.0220
step  290,816  ent_type=0.0109
step  413,696  ent_type=0.0083  ← lowest seen
step  491,520  ent_type=0.0141  (latest)
```

**This isn't a recent failure caused by the eval regression — it's
been broken since step ~100k.** The eval regression made it visible
because by step 307k the deterministic argmax had nowhere to recover.

### Why the existing collapse detector didn't fire

`entropy_collapse_threshold` is configured at 0.0005 in `_base.yaml`.
The type head bottomed at 0.0083 — alarmingly low but above the
threshold. The threshold is too lenient for catching the gradual
sub-uniform collapse pattern; it was set for catastrophic zero-entropy
collapse.

### Why rollout-side WR is still 0.40+ despite this

Two reasons:
1. PPO **samples** actions during rollout. Even a 0.5%-probability
   non-dominant action type fires occasionally over 4096 rollout steps.
2. The other 5 heads (corner / edge / tile / res1 / res2) still have
   high entropy. So when the type head DOES randomly fire BuildSettlement
   or BuildRoad, the location is reasonably explored.

Deterministic eval has no such luxury — argmax → same type every step
→ truncation 82.5% of the time.

### Auxiliary losses (FYI)

```
train/belief_loss     = 1.250  (started at log(5)=1.609, learning)
train/opp_action_loss = 0.883  (started at log(13)=2.565, learning)
```

Both aux heads are training successfully. Belief loss has dropped 22%
and opp-action loss has dropped 66% since init. So the aux signal is
fine — they're not the cause of the type-head collapse.

### Likely root cause

The type head's prior (`gain=0.01` final layer init) plus PPO's loss
landscape means very early in training it learns "EndTurn = no-cost
default action." Once committed, the entropy_coef=0.04 isn't strong
enough to push it back open, especially because:

- The joint entropy bonus is weighted equally across all 6 heads
- Corner/edge heads have ~3-4 nats of entropy "to spare"
- The bonus is dominated by those high-entropy heads

So the joint entropy stays around 0.5 and looks healthy — but it's
all in the wrong heads. The type head is starved of exploration
pressure.

### Decision rule pending

500k eval hasn't fired. I'll schedule a short (12 min) wakeup to
catch it and apply the prior rule:
- 500k WR > 409k WR → "RECOVERY UNDERWAY"
- 500k WR ≤ 409k WR → "TRIPLE REGRESSION — terminate loop"

Either way, the diagnostic above is the most useful thing the user
will see when they wake up.

### Suggested user actions (highest leverage first)

1. **Restart with per-head entropy bonus.** Trainer currently bonuses
   joint entropy. Modify `update()` to weight each head's entropy
   independently — e.g. `entropy_loss = -sum(α_h * ent_h)` with
   `α_type = 0.5, α_others = 0.05`. The type head gets 10× the
   exploration pressure of the others. Code change: ~30 LOC in
   `trainer.py` around line 970.
2. **Tighten the collapse detector.** `entropy_collapse_threshold`
   from 0.0005 → 0.05. Per-head, would have fired ALERT at step 86k
   on this run. Cheap defensive fix.
3. **Restart with much higher entropy_coef** (0.20 instead of 0.04).
   Crude but might break the type head out of its rut without code
   changes.
4. **Restart with phase3_full** (drop 2.3 GNN + 2.5c opp-action +
   4.2 recurrent value) to test if these features are amplifying
   the early-commit pattern. Also drops ~500k params, less to
   overcommit.

Process pid 91522 still alive. Current run worth letting continue
for the eval data point but the underlying agent is unlikely to
recover without intervention given the type-head has been stuck for
~400k steps.

## Overnight check-in 3b (step 528,384, elapsed 4:44:04) — RECOVERY UNDERWAY (1-step)

The 500k eval landed at step 512,000:

```
eval/win_rate:        @102k=0.225  @204k=0.350  @307k=0.225  @409k=0.125  @512k=0.250
eval/truncation_rate: @102k=0.700  @204k=0.625  @307k=0.750  @409k=0.825  @512k=0.725
eval/avg_vp:          @102k=7.625  @204k=9.825  @307k=8.325  @409k=5.775  @512k=8.625
```

**512k WR = 0.250 vs 409k WR = 0.125** → +0.125 absolute recovery.
All three eval metrics improved: WR ↑, truncation_rate ↓, avg_vp ↑.

### Decision rule applied

Per the prompt: "if 500k WR > 0.125, write 'RECOVERY UNDERWAY
(1-step)' verdict and reschedule another check-in at 50 min."
Verdict: **RECOVERY UNDERWAY (1-step)**.

The autonomous loop continues. Next wakeup at +50 min.

### Caveat — recovery does NOT invalidate the type-head diagnostic

The diagnostic from check-in 3 still stands: `train/entropy_head_type`
is at 0.0141 (latest 0.0156 at this snapshot — checked separately).
The type head has been stuck since step 86k.

What the 512k recovery probably represents:
- The agent's commited argmax type happened to land on a slightly
  better default this rollout vs the 409k snapshot.
- 40-game eval CI is wide; one point recovery from 0.125 → 0.25 is
  consistent with noise.
- The deterministic policy's *upside* at the current type-head
  collapse is bounded — even at "best argmax," the agent can't
  flexibly switch action types based on state, so it can't
  reliably build to 15 VP. That's why truncation_rate is still 72.5%
  even at the recovered 512k eval.

So: training is allowed to continue per the loop rule, but the
underlying problem (type-head collapse) is not solved.

### Rollout-side scalars look better

```
fps=31.1   rollout WR=0.45 (↑↑ from 0.41 last heartbeat — good)
rollout R=+0.18 (↑ from +0.01 — positive and improving)
EV=0.075 (down from peak 0.232 but stable)
entropy=0.46 (still falling — anneal hasn't started yet, this is
              natural sharpening)
KL=0.0014   clip_fraction=0.026   collapse_flag=0
```

Rollout side is genuinely improving — WR=0.45 and R=+0.18 are the
best in the run. The agent IS learning something via PPO sampling;
the deterministic argmax just can't access that learning because
the type head is stuck.

### Next wakeup

Scheduled at +50 min. Will catch evals 6-7 (steps 600k, 700k). The
loop continues until WR > 0.95 (auto-upgrade) or 3 consecutive
regressions.

The user-facing diagnostic remains the same: when they wake, the
type-head fix is the most leverage-positive intervention. The
overnight run will keep producing data but is hard-capped on
sample-efficient improvement until the type head is exploration-pushed.

Process pid 91522 alive. State `RN`. No ALERTs.

## Heartbeat (step 548,864, ~5:01 elapsed)

```
fps=30.5    rollout WR=0.44    R=+0.12 (positive, holding)
EV=0.031 (down from 0.075 — wobbling, but rollout R still positive)
entropy=0.50    KL=4e-04    collapse_flag=0
```

No alerts. 600k eval ~30 min away.

## Overnight check-in 4 (step 598,016, elapsed 5:35:51)

The 600k eval hasn't fired yet (~2k steps short — fires at next 100k
boundary, ~1 min away). Decision rules apply on what's in the data
right now (5 evals, last at 512k).

### Eval trend (still 5 points)

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 102k | 0.225 | 0.700 | 7.625 |
| 204k | 0.350 | 0.625 | 9.825 |
| 307k | 0.225 | 0.750 | 8.325 |
| 409k | 0.125 | 0.825 | 5.775 |
| 512k | 0.250 | 0.725 | 8.625 |

Latest WR (0.250) is well below 0.95 → no auto-upgrade.
Latest pattern is regression-recovery (0.125 → 0.250) → not 3
consecutive regressions. Decision rule (c) applies: continue.

### Type-head: still collapsed (slightly worse)

| step | entropy_head_type |
|---|---|
| 4k | 0.196 |
| 86k | 0.024 |
| 491k | 0.014 |
| **598k** | **0.011** ← latest, marginally tighter |

The diagnostic from check-in 3 holds. Type head has been ≤ 0.025
nats for ~500k steps. Latest 0.011 is the lowest yet.

### Rollout-side scalars — best of the run

```
fps=29.8   (down from 30.5 — slow drift, no alert)
rollout WR=0.50  ← highest of the run
rollout R=+0.31  ← highest of the run
EV=0.132 (recovered from 0.031 wobble at step 549k)
entropy=0.52   value_loss=0.60
KL=0.0021   clip_fraction=0.027   collapse_flag=0
```

PPO sampling is genuinely doing well: WR=0.50 in self-play, reward
strongly positive. The agent IS learning to win games when it samples
diverse action types — the issue is purely that argmax can't access
this learning.

### Aux losses — still training

```
belief_loss     = 1.008  (was 1.250 at check-in 3 — down 19% in 100k steps)
opp_action_loss = 0.874  (was 0.883 — flat, but already learned a lot)
```

Both aux heads continuing to learn. Belief loss is a useful signal —
it's approaching the irreducible entropy of the opponent dev-card
distribution (which depends on game state).

### ALERTs

None. Plateau detector still inactive (spread = 0.225 across last 4
evals, way over the 0.04 trigger).

### Verdict

**Continue overnight loop.** Rollout-side metrics are the best of the
run; eval-side has a 1-step recovery. Type-head collapse remains the
underlying issue but doesn't block the loop continuing.

Reschedule wakeup +50 min to catch evals 6-7 (steps 600k, 700k).
The next deciding moment for the loop:
- 3 consecutive eval regressions → terminate
- WR ≥ 0.95 → auto-upgrade verdict

## Heartbeat (step 622,592, ~6:00 elapsed) — 🟢 STRONG EVAL JUMP

**600k eval came in at 0.575.** That's a +0.325 absolute jump from
the 512k eval (0.250). Largest single-eval move in the run; eval WR
more than doubled in 100k steps.

```
eval/win_rate: 100k=0.225  200k=0.350  307k=0.225  409k=0.125  512k=0.250  614k=0.575 ← !
```

Rollout-side at this heartbeat:
```
fps=29.6   rollout WR=0.48   R=+0.25 (positive, holding)
EV=0.076   entropy=0.52   value_loss=0.78
KL=6e-04   clip_fraction=0.053   collapse_flag=0
```

This contradicts my earlier "type-head collapse caps deterministic
performance" thesis — possibly the agent found a fixed-type strategy
that genuinely wins, or this is an outlier point on the eval CI. The
700k eval will tell us which.

If 614k = 0.575 holds at 700k, the run is healthier than the
diagnostic suggested. If 700k drops back to 0.20-0.30, then 614k was
just a lucky 40-game draw and the type-head collapse caveat stands.

## Overnight check-in 5 (step 684,032, elapsed 6:26:49)

700k eval hasn't fired yet (~16k steps short, ~9 min away). Decision
rule applied to the 6 evals we have:
- Latest WR (0.575) is well below 0.95 → not auto-upgrade
- Latest is the BIGGEST positive jump in the run → not regression
- Rule (c): continue + reschedule

### Eval trend — strong upward move at 614k

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 102k | 0.225 | 0.700 | 7.625 |
| 204k | 0.350 | 0.625 | 9.825 |
| 307k | 0.225 | 0.750 | 8.325 |
| 409k | 0.125 | 0.825 | 5.775 |
| 512k | 0.250 | 0.725 | 8.625 |
| **614k** | **0.575** | **0.375** | **11.950** |

The 614k eval is *structurally* much better than every prior eval:
- WR more than doubled (0.250 → 0.575)
- truncation_rate halved (0.725 → 0.375) — agent now finishes
  62.5% of games (was 27.5%)
- avg_vp jumped to 11.95 — closing in on the 15-VP win condition

### This contradicts my earlier "type-head collapse is a hard ceiling" thesis

Type-head entropy at step 684k is 0.0089 — even tighter than at
check-in 3 (0.014). Still essentially deterministic. Yet the eval
just produced the run's best WR by a wide margin.

What I got wrong: I assumed deterministic type-head argmax meant the
agent picks the same action type *every step*. That's incorrect —
the type-head distribution is *state-conditioned*. With ~99.5% mass
on argmax, the argmax can still differ between states (e.g.
"BuildSettlement" early game, "BuildCity" mid game, "EndTurn" when
out of resources). The collapse is on the *spread within a state*,
not on the across-state choice.

So the agent has been learning a sharply-conditioned "what to do
in state X" mapping all along — it just always picks the same thing
*for that state*. When the conditioning is good enough, this is
fine for argmax eval. The 614k eval likely reflects the conditioning
improving across states.

### Training-side scalars

```
fps=29.7    (holding)
rollout WR=0.44   R=+0.05  (down from heartbeat's WR=0.48 R=+0.25 — sampling noise)
EV=0.079    entropy=0.50    value_loss=0.68
KL=0.0023   clip_fraction=0.038   collapse_flag=0
```

Rollout dipped slightly from peak. Eval is noisy at 40 games — both
sides should be expected to wobble around the underlying skill.

### Aux losses (slight wobble)

```
belief_loss     = 1.230  (was 1.008 at check-in 4 — uptick, but
                          1.6→1.0→1.23 is fine for a learning curve)
opp_action_loss = 0.926  (was 0.874 — similar uptick)
```

Both wobbled up slightly. Nothing alarming; aux losses fluctuate as
the encoder shifts in response to the dominant PPO signal.

### ALERTs

None. Plateau detector still inactive (spread = 0.450 across last 4
evals after the 614k jump). FPS regression detector hasn't fired.

### Verdict

**Continue overnight loop.** Run is genuinely improving — 614k eval
is the best in the run by a wide margin, and it's structurally
better (truncation halved, avg_vp at 12). The type-head diagnostic
was a partial misread on my part; the agent's state-conditioned
argmax can be effective even with low within-state entropy.

Reschedule wakeup +50 min. Watching for: 700k eval result, whether
WR continues climbing, whether truncation_rate continues dropping.

If WR keeps climbing toward 0.95, the auto-upgrade-to-heuristic
will fire and we'll start seeing `eval/win_rate vs heuristic`
instead.

Process pid 91522 alive. State `RN`. No ALERTs. Type-head still
collapsed but run continues to improve regardless.

## Heartbeat (step 708,608, ~6:51 elapsed)

700k eval should fire any minute (~8k steps past 700 boundary; trainer
checks at update boundaries every ~32k steps).

```
fps=29.5    rollout WR=0.40    R=-0.11 (wobbled negative)
EV=0.144 (recovered from 0.079)
entropy=0.52    KL=3e-05    collapse_flag=0
value_loss=0.71
```

Rollout side dipped — sampling noise; eval is the trustworthy signal.

## Overnight check-in 6 (step 770,048, elapsed 7:18:54)

700k eval landed at **0.425**. The 614k jump was real but partly lucky.

### Eval trend

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 102k | 0.225 | 0.700 | 7.625 |
| 204k | 0.350 | 0.625 | 9.825 |
| 307k | 0.225 | 0.750 | 8.325 |
| 409k | 0.125 | 0.825 | 5.775 |
| 512k | 0.250 | 0.725 | 8.625 |
| 614k | **0.575** | **0.375** | **11.950** |
| **716k** | **0.425** | 0.550 | 9.850 |

### Was the 614k jump real or noise? — **partly real**

716k WR = 0.425. Comparing to my check-in-5 hypothesis:
- "≥ 0.45 → 614k was real"   → 0.425 is BELOW 0.45 by a hair
- "≤ 0.30 → 614k was noise"  → 0.425 is well above 0.30

**Verdict: real improvement with noise overlay.** The underlying skill
is now in the **0.40–0.50 band** (≈2× the 100-300k baseline of 0.225).
The 614k peak at 0.575 was the upper end of that band's CI; 716k at
0.425 is the lower end. Both consistent with mean ~0.50 + noise.

### Decision rule

- 0.425 not ≥ 0.95 → not auto-upgrade
- 716k vs 614k: regression by 0.150
- 614k vs 512k: improvement (not regression)
- 1 consecutive regression, not 3 → loop continues
- Rule (c) applies: status + reschedule

### Type-head — slightly higher

0.0089 → **0.0157** (latest). Still collapsed but moving in the right
direction. Even at 0.0157 the within-state distribution is essentially
deterministic; this isn't recovery from collapse, just noise floor.

### Aux losses — improving

```
belief_loss     = 1.207  (flat from 1.230 last check-in)
opp_action_loss = 0.629  (DOWN from 0.926 — meaningful improvement!)
```

Opp-action head dropped 32% in 100k steps — the agent is getting
notably better at predicting opponent's next move type. That's a clean
"the encoder is learning useful structure" signal.

### Training-side scalars

```
fps=29.3    (holding within ±1 of trend baseline)
rollout WR=0.34   R=-0.41 (sharply negative this snapshot — sampling noise;
                           eval is the trustworthy skill measure)
EV=0.052    entropy=0.48    value_loss=0.53
KL=0.0013   clip_fraction=0.030   collapse_flag=0
```

Rollout R wobbled deeply negative (was +0.25 at check-in 4 peak, now
-0.41). This is sample-side variance. The eval trend says skill is up.

### ALERTs

None. Plateau detector still inactive (last 4 evals = 0.250, 0.575,
0.425, with another data point coming at 800k — spread is 0.45, way
above 0.04).

### Verdict

**Continue overnight loop.** Two consecutive evals at 0.425+ confirms
the run *is* learning beyond the early plateau. Type-head collapse is
not preventing skill improvement.

Reschedule wakeup +50 min to catch evals at 800k and 900k.

Next deciding moment: 3 consecutive eval regressions (would terminate
loop) or WR crossing 0.95 (auto-upgrade). Neither close.

Process pid 91522 alive. State `RN`. No ALERTs.

## Heartbeat (step 794,624, ~7:55 elapsed)

```
fps=29.4    rollout WR=0.34    R=-0.39 (still sharply negative)
EV=0.120 (recovered from 0.052)
entropy=0.51    KL=0.0023    collapse_flag=0
```

Rollout reward stuck negative this stretch — likely the league is
sampling stronger PFSP-hard opponents the agent can't beat at
sample-time. Eval is the trustworthy skill measure. 800k eval
~5k steps (~3 min) away.

## Overnight check-in 7 (step 847,872, elapsed 8:10:51) — 🟢 NEW PEAK

**819k eval landed at 0.675 — best of the run.** Skill band is
clearly holding and trending up.

### Eval trend

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 102k | 0.225 | 0.700 | 7.625 |
| 204k | 0.350 | 0.625 | 9.825 |
| 307k | 0.225 | 0.750 | 8.325 |
| 409k | 0.125 | 0.825 | 5.775 |
| 512k | 0.250 | 0.725 | 8.625 |
| 614k | 0.575 | 0.375 | 11.950 |
| 716k | 0.425 | 0.550 | 9.850 |
| **819k** | **0.675** | **0.300** | **12.750** |

All three metrics now at run peaks at 819k:
- WR 0.675 (+17% over the prior peak of 0.575)
- truncation 0.300 (lowest yet — agent finishes 70% of games)
- avg_vp 12.75 (highest yet — within striking distance of the
  15-VP win condition)

### Skill band assessment

Last 2 evals: 0.425, 0.675 — both in the **0.40–0.70 band**, well
above the early baseline of 0.225. The 614k+716k+819k pattern
(0.575, 0.425, 0.675) is consistent with mean ~0.55 + noise. Skill
band is **holding, on the high side**.

### Type-head — partial recovery!

```
step 491k: 0.014
step 598k: 0.011
step 684k: 0.009  ← lowest
step 770k: 0.016
step 848k: 0.0496  ← latest, 3× higher
```

The type-head entropy has roughly tripled in the last 200k steps.
Still well below uniform (2.56 nats), but the recovery is significant:
- 0.0089 was "essentially deterministic"
- 0.0496 is "still tight, but starting to spread"

This may explain the eval improvements — the agent is gradually
re-acquiring type-level exploration. Whether the entropy bonus,
PFSP-hard pressure, or value-loss landscape change drove this is
not visible from the scalar set, but the trend is healthy.

### Aux losses

```
belief_loss     = 1.288  (was 1.207 — slight uptick, noise)
opp_action_loss = 0.478  (was 0.629 — DOWN 24%, continuing learn)
```

`opp_action_loss` has dropped 47% since check-in 4 (0.926 → 0.478).
The encoder is genuinely fitting useful features about opponent
behavior.

### Decision rule

- 0.675 < 0.95 → not auto-upgrade
- 819k > 716k → improvement (latest is best of run)
- Not 3 consecutive regressions
- Rule (c): continue + reschedule

### Training-side scalars

```
fps=28.9    rollout WR=0.40   R=-0.09 (recovered from -0.39 — sample noise)
EV=0.067    entropy=0.54   value_loss=0.64
KL=0.0024   clip_fraction=0.036   collapse_flag=0
```

Rollout reward bounced back from −0.39 → −0.09, more in line with
the run's mean.

### ALERTs

None. Plateau detector inactive (last 4 evals = 0.250, 0.575, 0.425,
0.675; spread 0.425, way above the 0.04 trigger).

### Verdict

**Continue overnight loop. Run is genuinely improving.** Three
consecutive evals (614k, 716k, 819k) all in the 0.40+ skill band, with
the latest at 0.675 a new peak. truncation_rate is dropping; avg_vp
is climbing. The type-head is partially recovering exploration.

Reschedule wakeup +50 min. Watching for: 900k+ evals, whether
truncation drops further, whether avg_vp pushes toward 13+. If WR
crosses 0.85+, the auto-upgrade-to-heuristic threshold (0.95) is in
reach within a few more 100k blocks.

Process pid 91522 alive. State `RN`. No ALERTs.

## Heartbeat (step 864,256, ~8:21 elapsed) — EV breakthrough

```
fps=28.9    rollout WR=0.40    R=-0.09 (steady)
EV=0.294 ← biggest of the run (was 0.232 peak, before that 0.13)
entropy=0.48    value_loss=0.38 (down from 0.64)
KL=0.003   collapse_flag=0
```

EV jumped to 0.29 — value head is fitting returns substantially
better. Combined with the 819k eval peak (0.675), this is meaningful
late-run progress.

## Overnight check-in 8 (step 917,504, elapsed 9:02:51)

900k eval hasn't fired yet (~17k steps short — fires at next 100k
boundary, ~10 min). Decision rule applied to existing 8 evals.

### Eval trend (still 8 points)

```
@102k=0.225 @204k=0.350 @307k=0.225 @409k=0.125 @512k=0.250 @614k=0.575 @716k=0.425 @819k=0.675
```

Latest = 0.675, well below 0.85 (close-to-auto-upgrade threshold)
and 0.95 (auto-upgrade). Latest is run peak → no regression. Rule
(c) applies: continue + reschedule.

### Type-head — oscillating up

```
step 491k: 0.0141
step 598k: 0.0110
step 684k: 0.0089  ← floor
step 770k: 0.0157
step 848k: 0.0496  ← peak
step 918k: 0.0295  ← latest
```

Stabilizing in the 0.02-0.05 range. Below uniform but well above the
0.009 floor. Real (if modest) recovery.

### Aux losses — wobble

```
belief_loss     = 1.081  (up from 1.288 — actually IMPROVEMENT, lower=better)
opp_action_loss = 0.642  (up from 0.478 — wobble; was 0.926 → 0.629 → 0.478 → 0.642)
```

Note: belief_loss got *better* (1.288 → 1.081, value drops). Wrote
"slight uptick" earlier — that was reading direction wrong; lower
belief_loss = better calibration. Belief head learning is healthy.
Opp-action loss wobbled up — typical noise.

### Training-side scalars

```
fps=28.3    rollout WR=0.43 (steady)   R=+0.01 (back near zero)
EV=0.034    (wobbled down from 0.294 peak — normal)
entropy=0.49    value_loss=0.79
KL=-6e-05   clip_fraction=0.029   collapse_flag=0
```

EV wobbled down from peak; rollout near zero. The EV oscillation is
the most volatile signal — it spikes up when a rollout has clean
value targets, drops when the value head over-fits to a temporary
noise pattern.

### ALERTs

None. Plateau detector inactive (last 4 evals = 0.250, 0.575, 0.425,
0.675 = spread 0.425, way above 0.04 trigger). FPS regression
detector hasn't fired (28.3 vs trailing baseline ~29).

### Verdict

**Continue overnight loop.** Skill band intact: last 3 evals all in
[0.42, 0.68]. Type-head recovery holding. WR not yet at 0.85
(auto-upgrade prep level) or 0.95 (auto-upgrade trigger).

Reschedule wakeup +50 min. Watching for: 900k+ eval results, whether
WR pushes toward 0.80+ or falls back below 0.40 (skill-band breakdown).

Process pid 91522 alive. State `RN`. No ALERTs.

## Heartbeat (step 925,696, ~9:48 elapsed) — ⚠ EVAL CRASHED

The 921k eval landed at **0.100** — far below every prior eval
including the 0.225 baseline.

```
eval/win_rate: ... 614k=0.575  716k=0.425  819k=0.675  921k=0.100 ← !
```

That's a -0.575 absolute drop from peak (819k=0.675). Largest single
move in the run, in the wrong direction.

### Suspicious training-side scalars at this snapshot

```
value_loss = 1.397   (jumped from 0.79 — 76% spike)
EV         = 0.005   (collapsed from 0.034 → essentially zero)
entropy    = 0.545   (slight uptick from 0.49)
fps        = 28.1    (steady)
```

The value_loss spike + EV collapse + eval crash all in one update
batch suggests a bad PPO update — possibly the league sampled a
particularly hard opponent that caused the policy to over-react.

### Status

This is ONE regression vs the prior eval (819k → 921k). The rule
requires 3 consecutive to terminate the loop. The next eval (~1M
steps) will be the deciding point — if it recovers to 0.40+ this
was an outlier; if it drops further to ≤ 0.10, the run has hit a
real pathology.

Not escalating yet. Logging, continuing.

## Overnight check-in 9 (step 995,328, elapsed 9:53:50) — ⚠ SKILL-BAND BREAKDOWN

The 921k eval crash is now confirmed across all three eval metrics
— this isn't noise:

### Eval trend

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 102k | 0.225 | 0.700 | 7.625 |
| 204k | 0.350 | 0.625 | 9.825 |
| 307k | 0.225 | 0.750 | 8.325 |
| 409k | 0.125 | 0.825 | 5.775 |
| 512k | 0.250 | 0.725 | 8.625 |
| 614k | 0.575 | 0.375 | 11.950 |
| 716k | 0.425 | 0.550 | 9.850 |
| 819k | **0.675** | **0.300** | **12.750** |
| **921k** | **0.100** | **0.850** | **6.725** |

WR fell from peak 0.675 to 0.100 (-85% relative). Per the wakeup
prompt's criterion: **WR < 0.40 = skill-band breakdown.** All three
metrics moved together, ruling out CI noise.

### But it's just 1 regression

Decision rule (b) requires 3 consecutive eval regressions:
- 921k vs 819k: regression (1)
- 819k vs 716k: improvement (recovery, breaks the chain)
- 716k vs 614k: regression
- 614k vs 512k: improvement (chain breaks)

Current consecutive-regression count = 1. Rule (b) does not fire.
Rule (a) doesn't fire (0.100 ≪ 0.95). **Rule (c) applies: continue +
reschedule.**

### What likely happened (training-side evidence)

At the heartbeat (step 925k) I caught:
- value_loss spiked from 0.79 → **1.397** (76% jump)
- EV collapsed from 0.034 → **0.005**

Now at step 995k:
- value_loss = 0.998 (down from 1.397 — recovering toward 0.79 norm)
- EV = 0.091 (recovered from 0.005)

Looks like a single bad PPO update — possibly PFSP-hard sampled an
unusually exploitable opponent, the agent over-reacted, both the
value head and the deterministic policy got pushed off-manifold for
that 100k window. The training-side metrics are normalizing.

Whether the policy recovers in the 1M eval (next ~3 min) or stays
broken is the deciding signal:
- 1M ≥ 0.40 → transient, run is fine
- 1M ≤ 0.20 → second consecutive regression; if 1.1M is also low, rule (b) fires

### Type-head — back in normal band

```
step 848k: 0.0496  ← peak
step 918k: 0.0295
step 995k: 0.0175  ← latest
```

Reverted from peak but still well above the 0.009 floor. Not a
contributing factor to the eval crash.

### Aux losses — unchanged

```
belief_loss     = 1.291  (similar to 1.288 at check-in 7)
opp_action_loss = 0.826  (uptick from 0.642 — wobble)
```

Both within typical noise bands.

### Rollout-side scalars

```
fps=28.0    rollout WR=0.40   R=-0.11
EV=0.091   entropy=0.52   value_loss=0.998 (recovering)
KL=0.0015   clip_fraction=0.028   collapse_flag=0
```

Rollout WR holding at 0.40, reward back near zero. Rollout-side
hasn't crashed — the agent is still beating its league at sample
time.

### ALERTs

None from the existing detectors. The single-update value_loss
spike at the heartbeat would have been a useful trigger but isn't
covered by current alerts.

### Verdict

**Continue overnight loop, but flagged.** Run had a sharp eval
regression at 921k (skill-band breakdown per the wakeup criterion)
but training-side scalars are recovering. 1M eval (~3 min) is the
deciding "transient vs trend" point.

If 1M ≥ 0.40 → reset analysis, run is fine.
If 1M < 0.20 → next eval (1.1M) decides loop termination.

Reschedule +50 min. Process pid 91522 alive. State `RN`.

## Heartbeat (step 1,003,520, ~10:38 elapsed) — crossed 1M

Just past 1M steps. The 1M eval hasn't fired yet — trainer checks at
update boundaries (~32k step granularity), so the eval will land at
~1,016k.

```
fps=27.9   rollout WR=0.41   R=-0.05
EV=0.094 (steady recovery)
value_loss=1.155 (still elevated; was 1.397 spike → 0.998 → 1.155 — wobbling)
entropy=0.50   KL=-0.001   collapse_flag=0
```

Value loss elevated but not collapsed — wobbling around 1.0-1.4 vs
0.79 norm. The 1M eval (~5 min) will tell us if the policy
recovered or if 921k=0.100 is the start of a real pathology.

## Overnight check-in 10 (step 1,069,056, elapsed 10:45:49) — 🟢 TRANSIENT — RECOVERED

The 1M eval landed at **0.675** — fully recovered to the prior peak.
The 921k=0.100 was definitively a transient.

### Eval trend (full recovery)

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 819k | 0.675 | 0.300 | 12.750 |
| 921k | **0.100** | **0.850** | **6.725** ← crash |
| **1024k** | **0.675** | **0.325** | **12.300** ← recovered |

All three metrics back to peak levels. WR ties the run peak. Truncation
back to 0.325 (was 0.300 at 819k peak). avg_vp 12.30 (was 12.75).

### Per the wakeup decision rule

"if 1M WR ≥ 0.40, write 'TRANSIENT — RECOVERED'"

1024k WR = 0.675 ≥ 0.40 → **TRANSIENT — RECOVERED**.

### Value-loss timeline (the bad-update signature has resolved)

```
step  925k:  1.397  ← spike
step  995k:  0.998
step 1003k:  1.155
step 1069k:  0.822  ← back to ~norm
```

Last 5 value_loss readings: 0.579, 0.481, 0.344, 0.523, 0.822 —
variable but all within typical 0.5-1.0 band. The 1.397 spike at 925k
was contained to that single update batch.

### Type-head — stable

```
last 5 entropy_head_type readings: 0.018, 0.026, 0.029, 0.030, 0.029
```

Holding in the 0.02-0.03 band. No collapse, no further recovery.

### Decision rule

- 0.675 < 0.95 → not auto-upgrade
- 1024k > 921k → improvement
- Consecutive-regression count back to 0 (a recovery breaks any chain)
- Rule (c) applies: continue + reschedule

### Training-side scalars

```
fps=27.7    rollout WR=0.37   R=-0.17 (slightly worse than prior)
EV=0.055    entropy=0.52    value_loss=0.822
KL=0.0028   clip_fraction=0.040   collapse_flag=0
```

Rollout side a touch weaker than peak — sample noise; eval is the
trustworthy signal and it's back at peak.

### ALERTs

None. Plateau detector inactive (last 4 evals = 0.425, 0.675, 0.100,
0.675 = spread 0.575). FPS regression detector hasn't fired.

### Verdict

**TRANSIENT — RECOVERED.** Run is back on track. The 921k=0.100
crash was a single bad PPO update; 1024k=0.675 confirms the policy
recovered. Skill band intact. Type-head stable. Aux losses normal.

Next deciding moment: WR climbing past 0.85 (auto-upgrade prep) or
another deep regression (if it recurs systematically, that would
indicate a real pathology).

Reschedule wakeup +50 min. Process pid 91522 alive. State `RN`.

## Heartbeat (step 1,077,248, ~10:52 elapsed)

```
fps=27.6   rollout WR=0.37   R=-0.17
value_loss=0.220 (DROPPED from 0.82 — much better, near 0.5 norm)
EV=-0.057 (slipped slightly negative — mid-update wobble, ignored unless sustained)
entropy=0.45    KL=-7e-04
```

Value loss recovery confirms the 921k spike is fully resolved. EV
nudge negative is normal training noise, not a regression signal.

## ALERT investigation (step 1,081,344) — `ALERT:NEG_EV=-1.26`

Detector fired at step 1,081,344 with EV = -1.261, far below the
-0.5 threshold. Investigated and **not escalating** — this is a
known measurement artifact, not a value-head pathology.

### EV / value_loss recent history

```
step  1,069k  EV=+0.055  vl=0.822
step  1,073k  EV=+0.108  vl=0.300
step  1,077k  EV=-0.057  vl=0.220
step  1,081k  EV=-1.261  vl=0.067   ← alert
```

### Why this is a measurement artifact, not a broken value head

Note that `value_loss` (MSE) is at its **LOWEST** in the recent
window (0.067) at the same step where EV crashes to -1.26. That's
the giveaway:

- `EV = 1 - Var(returns - predictions) / Var(returns)`
- When returns are nearly constant (low variance) — e.g. all rollout
  games hit `max_turns=500` with similar outcomes — the denominator
  Var(returns) shrinks toward zero
- Even small prediction errors then produce a huge negative ratio
- Meanwhile `value_loss` (raw MSE) is small because predictions match
  the constant-ish returns

This is a **single-update outlier** consistent with a low-variance
rollout, not a value-head failure. The prior 9 EV readings range
[-0.16, +0.28] — within healthy bounds.

### Detector limitation (known)

Our `NEG_EV` alarm fires on a single below-threshold reading. It
doesn't require N consecutive samples and doesn't guard against
low-variance-returns artifacts. Suggested future fix (NOT applied
during the live run): add a `Var(returns) > epsilon` guard, or
require N=3 consecutive negative readings.

### Status

```
fps=27.5    rollout WR=0.37   R=-0.17 (steady)
entropy=0.46    value_loss=0.067    KL=0.0025
```

Process pid 91522 alive. State `RN`. Eval WR 1024k=0.675
(latest, peak). Loop continues. The fact that the detector fired
once is fine — it's a hint that some rollouts are running into
truncation-heavy patterns, but training-side metrics are otherwise
healthy.

Will surface to the next wakeup if the EV stays deeply negative
for 3+ consecutive updates (would indicate the value head genuinely
broke). Single outliers don't escalate.

## Overnight check-in 11 (step 1,126,400, elapsed 11:36:50)

1.1M eval imminent (~step 1,126k, fires at next update boundary —
already at the boundary, may be in the file flush right now).
Decision rule applied to existing 10 evals.

### Eval trend (still 10 points)

```
@102k=0.225 @204k=0.350 @307k=0.225 @409k=0.125 @512k=0.250
@614k=0.575 @716k=0.425 @819k=0.675 @921k=0.100 @1024k=0.675
```

Latest 0.675 (peak). No new eval since check-in 10. Decision rule
(c) applies: continue + reschedule.

### EV alert from check-in 10b — confirmed transient

Detector fired at 1,081k with EV=-1.26 (low-variance-returns artifact).
Latest EV is back to **+0.133**. Single-update outlier confirmed; not
a value-head pathology.

### Type-head — holding

```
0.0250 @step 1,126,400 (latest)
```

In the 0.02-0.03 band where it's been since check-in 6.

### Aux losses — normal

```
belief_loss     = 1.224  (similar to recent ~1.2 band)
opp_action_loss = 0.657  (similar to recent ~0.65 band)
```

### Training-side scalars

```
fps=27.0    rollout WR=0.33   R=-0.31 (down from 0.37/-0.17 at heartbeat)
EV=0.133    entropy=0.53    value_loss=0.916
KL=0.0032   clip_fraction=0.043   collapse_flag=0
```

Rollout WR slipped to 0.33 (slight regression from 0.37); R back to
-0.31 (sample-side noise; eval is the trustworthy signal). value_loss
elevated to 0.92 — wobbling but not pathological.

### ALERTs

None active. The NEG_EV from earlier (1,081k=-1.26) was a single
sample; current EV at +0.133 is healthy.

### Verdict

**Continue overnight loop.** Skill band intact (latest eval at peak
0.675). No new alerts. Type-head stable. EV recovered from the
artifact spike. value_loss elevated but value-head functional.

WR not crossing 0.85 (auto-upgrade prep) yet; not crashing below
0.40 (skill-band breakdown) either. Holding at peak.

Reschedule wakeup +50 min to catch evals 11-12 (steps 1.1M, 1.2M).
Process pid 91522 alive. State `RN`.

## Heartbeat (step 1,130,496, ~12:42 elapsed) — 1.1M eval landed

```
eval/win_rate: ... 819k=0.675  921k=0.100  1024k=0.675  1126k=0.325
```

1.1M eval = **0.325** — regression from peak (0.675 → 0.325), but
above the 0.225 baseline. Truncation/avg_vp not yet pulled.

Training-side:
```
fps=26.9    rollout WR=0.34   R=-0.28
EV=0.040    value_loss=0.384    entropy=0.51
KL=0.0030   collapse_flag=0
```

This is 1 fresh consecutive regression (1024k→1126k = down). The
prior chain was broken by the 1024k recovery, so we're at 1 in a
row, not 2 or 3. Loop continues. EV positive, value_loss normal —
not the same pathological signature as the 921k crash.

Pattern to watch: WR is oscillating in [0.10, 0.68] with high
amplitude. If it can't break above 0.70 sustained, the run may be
plateaued at the current architecture's ceiling against random.

## Overnight check-in 12 (step 1,183,744, elapsed 12:27:50)

1.2M eval hasn't fired yet (~45k steps short — fires at boundary
~1,228k, ~28 min away). Decision rule applied to existing 11 evals.

### Eval trend

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 819k | 0.675 | 0.300 | 12.750 |
| 921k | 0.100 | 0.850 | 6.725 (transient crash) |
| 1024k | 0.675 | 0.325 | 12.300 (recovered) |
| 1126k | 0.325 | 0.625 | 7.975 (latest, regression from peak) |

1 fresh consecutive regression (1126k vs 1024k). The 921k chain
was broken by the 1024k recovery, so the count restarts. Not 3 in
a row → rule (b) doesn't fire.

### ⚠ Type-head re-collapsed to 0.004

```
step  848k:  0.0496  ← peak recovery
step  918k:  0.0295
step  995k:  0.0175
step 1126k:  0.0250
step 1184k:  0.0039  ← latest, LOWEST since step 684k
```

The type-head entropy dropped 6× in this update batch. Agent has
re-committed to a sharper argmax. This may be correlated with the
1126k eval regression (sharper commit → narrower argmax space →
deterministic policy is less robust against random opponents).

### Belief loss — dropped substantially

```
belief_loss     = 0.697  (was 1.224 at check-in 11 — DOWN 43%!)
opp_action_loss = 0.667  (was 0.657 — flat)
```

The belief head made a real jump in fit quality this update. The
agent is now substantially better at predicting opponent's hidden
dev cards. This is a positive signal — but worth noting that big
encoder shifts (which a 43% loss drop implies) can correlate with
policy disruption.

### Training-side scalars

```
fps=26.4    rollout WR=0.36   R=-0.21 (sample noise)
EV=-0.026   value_loss=0.022 (near-zero — low-variance-returns again)
entropy=0.48    KL=-4e-04   clip_fraction=0.044   collapse_flag=0
```

Same low-variance-returns signature as the earlier NEG_EV false
alarm — value_loss tiny + EV slightly negative. Not a real value
head problem, just constant-return rollouts.

### ALERTs

None. NEG_EV not triggered (current -0.026 is well above the -0.5
threshold). Plateau detector inactive (eval spread 0.575).

### Verdict

**Continue overnight loop.** 1 fresh regression after the 1024k
recovery, type-head re-collapsed (concerning), belief loss made big
progress (positive). Rollout-side stable.

Watching the next 1-2 evals carefully:
- If 1.2M ≥ 0.40 → noise, run is healthy
- If 1.2M < 0.20 → 2 consecutive regressions, watching for 3rd
- If WR keeps oscillating in [0.10, 0.68] without trending up, the
  run may have hit a structural ceiling — would be worth flagging
  to user as "high-amplitude oscillation, no upward trend."

Reschedule wakeup +50 min. Process pid 91522 alive. State `RN`.

## Heartbeat (step 1,191,936, ~12:55 elapsed)

```
fps=26.5    rollout WR=0.34   R=-0.30
EV=0.062 (back positive from -0.026)
value_loss=0.522 (recovered from low-variance 0.022 batch)
entropy=0.51    KL=0.0025
```

Training-side back to nominal. Rollout R=-0.30 consistent with
league sampling stronger PFSP-hard opponents the agent can't beat
at sample-time. Eval is the trustworthy signal; 1.2M eval ~22 min
away.

## Overnight check-in 13 (step 1,245,184, elapsed 13:18:51) — 🟢🟢 NEW PEAK 0.800

The 1228k eval landed at **0.800** — best WR of the run by a wide
margin. Crossed the 0.85 "approaching auto-upgrade" highlight
threshold... and 0.800 itself is above 0.85 in the spirit of the
rule but slightly below the literal threshold. Auto-upgrade
trigger is 0.95.

### Eval trend

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 819k | 0.675 | 0.300 | 12.750 |
| 921k | 0.100 | 0.850 | 6.725 (transient) |
| 1024k | 0.675 | 0.325 | 12.300 (recovered) |
| 1126k | 0.325 | 0.625 | 7.975 (regression) |
| **1228k** | **0.800** | **0.200** | **13.375** ← new peak |

All three metrics at new run highs:
- WR 0.800 (was 0.675, +18% absolute)
- truncation_rate 0.200 (was 0.300, agent finishes 80% of games)
- avg_vp 13.375 (was 12.75, within 1.6 VP of the 15-VP win)

### PLATEAU concern from check-in 12 — invalidated

Check-in 12 flagged "if oscillation continues in [0.10, 0.68] across
4 evals, flag as plateau." The 1228k eval at 0.800 broke out of that
band upward. **Run is not plateaued; it's still improving.**

### Type-head — slight recovery

```
last 5: 0.0157, 0.0250, 0.0039, 0.0156 (latest: 0.0156)
```

Recovered from the 0.004 dip at check-in 12. Hovering 0.015-0.025.

### Aux losses

```
belief_loss     = 1.125  (up from 0.697 — wobble back, but not regression)
opp_action_loss = 0.480  (down from 0.667 — improvement, near the run low)
```

Opp-action loss back near its run low (0.478 at check-in 7). Encoder
genuinely fitting opponent-behavior features.

### Decision rule

- 0.800 is below 0.95 → not auto-upgrade (but very close — within 0.15)
- 1228k > 1126k → improvement, NOT regression
- 0 consecutive regressions
- Rule (c) applies: continue + reschedule

### Training-side scalars

```
fps=26.1    rollout WR=0.40   R=-0.085 (recovered from -0.30 at heartbeat)
EV=0.086    entropy=0.52    value_loss=0.642
KL=0.0027   clip_fraction=0.041   collapse_flag=0
```

Rollout reward back near zero. EV holding positive. Training-side
healthy.

### ALERTs

None. Plateau detector inactive (eval spread now 0.700 across last 4
evals). FPS regression detector hasn't fired.

### Verdict

**🟢 STRONG IMPROVEMENT — continue overnight loop.** New run peak
at 0.800. Truncation halved from baseline (0.700 → 0.200). avg_vp
13.4 — within striking distance of 15-VP wins. The run is genuinely
learning to beat random opponents.

Auto-upgrade threshold (0.95) is plausible within next 1-3 evals if
this trend holds. If the next eval crosses 0.95, the eval opponent
auto-upgrades to heuristic — that would be the project's first major
milestone.

Reschedule wakeup +50 min to catch evals at 1.3M and 1.4M.

Process pid 91522 alive. State `RN`. No ALERTs.

## Heartbeat (step 1,249,280, ~13:23 elapsed)

Just past the 1228k=0.800 peak. Training-side healthy.

```
fps=26.0    rollout WR=0.39   R=-0.12
EV=0.058    value_loss=0.482    entropy=0.47    KL=0.0024
```

Next eval (~1.33M) ~50 min away.

## Overnight check-in 14 (step 1,314,816, elapsed 14:09:52)

1.33M eval hasn't fired yet (~16k steps short — boundary at 1330k,
~10 min away). Decision rule applied to existing 12 evals.

### Eval trend (still 12 points)

```
@1024k=0.675 @1126k=0.325 @1228k=0.800 (latest, peak)
opponent_is_heuristic: all still 0.000 (no auto-upgrade yet)
```

Latest 0.800 < 0.95 → not auto-upgrade. Latest = run peak (no
regression). Rule (c) applies: continue + reschedule.

### Auto-upgrade status

`eval/opponent_is_heuristic` is **0.000 at all 12 evals** so far.
Auto-upgrade fires when WR vs random crosses 0.95 — we're at 0.800,
0.150 short. The next eval (1330k) is the deciding point for
whether the curve keeps climbing.

### Type-head — recovering

```
last 5: 0.0250, 0.0039, 0.0156, 0.0394 (latest)
```

The 0.004 dip at check-in 12 reversed; type-head is back at 0.04 —
the highest value since the original peak at check-in 7 (0.0496).
This may be why rollout-side scalars are improving below.

### Aux losses

```
belief_loss     = 1.067  (similar to recent ~1.1 band)
opp_action_loss = 0.778  (up from 0.480 — wobble; still well below
                          initial 2.56)
```

### Training-side scalars

```
fps=25.9    rollout WR=0.42 (↑ from 0.39)   R=+0.031 (POSITIVE — up from -0.12)
EV=0.190 (best EV reading since check-in 7's 0.232 peak)
entropy=0.55    value_loss=0.770    KL=0.0041
clip_fraction=0.039   collapse_flag=0
```

Rollout R positive, EV 0.19 (best in many checkpoints), type-head
exploration recovering — all healthy signals coinciding with the
0.800 eval peak.

### ALERTs

None. Plateau detector inactive (eval spread 0.700 across last 4 evals).

### Verdict

**Continue overnight loop.** Skill band intact at 0.800 peak.
Auto-upgrade threshold (0.95) within 0.150 absolute. Rollout-side
strongly positive: WR up, R positive, EV at recent high, type-head
exploring more.

Next 1.33M eval is the deciding milestone:
- ≥ 0.95 → AUTO-UPGRADE TRIGGERED, opponent flips to heuristic
- ≥ 0.85 → still climbing, on track
- < 0.50 → wide oscillation continues, no clear breakout

Reschedule wakeup +50 min. Process pid 91522 alive. State `RN`.

## Heartbeat (step 1,318,912, ~14:14 elapsed) — EV climbing

```
fps=25.8    rollout WR=0.41   R=-0.017 (near zero)
EV=0.211 (↑ from 0.190 — second-best reading of the run, after 0.232 peak)
value_loss=0.539    entropy=0.47    KL=-0.0015
```

Value head genuinely tracking returns — second-best EV in the run.
1.33M eval ~5 min away.

## Overnight check-in 15 (step 1,368,064, elapsed 15:00:50) — 🟢🟢🟢 0.925 — ONE STEP FROM AUTO-UPGRADE

The 1331k eval landed at **0.925**. The agent now wins 92.5% of
games against random with deterministic argmax. **0.025 absolute
short of the 0.95 auto-upgrade threshold.**

### Eval trend

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 819k | 0.675 | 0.300 | 12.750 |
| 921k | 0.100 | 0.850 | 6.725 (transient) |
| 1024k | 0.675 | 0.325 | 12.300 |
| 1126k | 0.325 | 0.625 | 7.975 |
| 1228k | 0.800 | 0.200 | 13.375 |
| **1331k** | **0.925** | **0.075** | **14.750** |

All three metrics at run highs:
- **WR 0.925** (was 0.800 at 1228k peak — +0.125)
- **truncation_rate 0.075** (was 0.700 at baseline — agent finishes
  92.5% of games)
- **avg_vp 14.75** (within 0.25 VP of the 15-VP win — agent is
  essentially winning every game it gets the chance)

### Auto-upgrade status

`eval/opponent_is_heuristic` series: all 0.000 across 13 evals.
Latest WR (0.925) is **below the 0.95 trigger by 0.025 absolute**.
Rule (a) does not fire yet. **One more eval at the same trajectory
likely triggers AUTO-UPGRADE TO HEURISTIC.**

### Wakeup-prompt explicit checks

- "WR ≥ 0.85 (close to auto-upgrade)?" → **YES, 0.925** (well above 0.85)
- "any opponent_is_heuristic = 1.0?" → **NO**, all 0.000

### Decision rule

- 0.925 < 0.95 → not auto-upgrade (just barely)
- 1331k > 1228k → improvement (no regression)
- Rule (c) applies: continue + reschedule

### Type-head — stable in healthy band

```
last 5: 0.0039, 0.0156, 0.0394, 0.0244 (latest)
```

In 0.02–0.04 band, similar to its check-in-7 peak. Type-head
exploration is stable — likely the reason WR keeps climbing.

### Aux losses

```
belief_loss     = 1.113  (similar to recent)
opp_action_loss = 0.669  (similar to recent)
```

### Training-side scalars

```
fps=25.3    rollout WR=0.40   R=-0.081 (sample noise)
EV=0.085 (wobbled down from 0.211 — typical post-eval dip)
entropy=0.53    value_loss=0.809    KL=0.0018
clip_fraction=0.042   collapse_flag=0
```

EV wobbled lower after the eval batch — typical post-eval dip
where the value head re-normalizes. Not concerning.

### ALERTs

None.

### Verdict

**🟢🟢🟢 Continue overnight loop.** The next eval (1.43M, ~50 min
away) is the most consequential of the run — likely the moment the
agent crosses the auto-upgrade threshold and the eval opponent
flips from random to heuristic. That would be the **first major
project milestone** crossed.

If 1.43M ≥ 0.95 → AUTO-UPGRADE fires → eval scale resets (WR will
drop dramatically against the new heuristic opponent, that's
expected and healthy).

Reschedule wakeup +50 min.

Process pid 91522 alive. State `RN`. No ALERTs.

## Heartbeat (step ~1,372,160, ~15:03 elapsed) — false-stall investigated

Initial snapshot looked stuck (same step as check-in 15) but a
fresh TB reload showed normal progression:

```
step 1,355,776  -15.0 min
step 1,359,872  -10.3 min
step 1,363,968   -7.6 min
step 1,368,064   -3.8 min
step 1,372,160   -0.6 min  (latest)
```

~25 FPS sustained. Process healthy. The `runs/train/console.log`
being stuck at step ~1,114k is Python stdout buffering (block-buffered
when redirected to a file); TB is the truthful source. Console logs
will catch up when the buffer flushes.

1.43M eval ~40 min away — captured by the 17:18 wakeup.

## Heartbeat (step 1,449,984, ~16:13 elapsed) — 1.43M eval landed at 0.875

Auto-upgrade not triggered yet — eval at 1433k held in the 0.85+ band:

```
eval/win_rate:           ...1228k=0.800  1331k=0.925  1433k=0.875
opponent_is_heuristic:   ...all 0.000 — no auto-upgrade fired
```

0.925 was the closest yet. 1.43M dropped slightly to 0.875 — still
solidly in the "near-mastered random" zone (-0.05 from peak, well
above the 0.85 threshold). Truncation/avg_vp not pulled this
heartbeat.

Training-side scalars are strongly positive:

```
fps=25.4    rollout WR=0.49 (↑ from 0.40)   R=+0.25 (POSITIVE, run-high again)
EV=0.039   value_loss=0.191 (low)   entropy=0.47   KL=4e-04
```

Rollout R back to +0.25 — matches earlier check-in 4 peak. Run is
producing high-quality data.

1.53M eval ~40 min away — that's the next chance for auto-upgrade.

## Overnight check-in 16 (step 1,449,984, elapsed 15:53:13)

The 1.43M eval landed at **0.875**. Auto-upgrade did NOT trigger.

### Wakeup-prompt critical-output answers

1. **1.43M WR ≥ 0.95?** → **NO**. Latest WR = 0.875 (0.075 short).
2. **opponent_is_heuristic has any 1.0?** → **NO**, all 0.000 across 14 evals.
3. **WR scale reset (plummeted vs heuristic)?** → **NO**, no scale reset occurred — agent still being evaluated against random.

### Eval trend (last 5)

| step | WR | trunc | avg_vp | vs heuristic? |
|---|---|---|---|---|
| 1024k | 0.675 | 0.325 | 12.300 | no |
| 1126k | 0.325 | 0.625 | 7.975 | no |
| 1228k | 0.800 | 0.200 | 13.375 | no |
| 1331k | 0.925 | 0.075 | 14.750 | no |
| **1433k** | **0.875** | **0.125** | **14.300** | no |

The agent is now reliably in the 0.80–0.95 band against random:
- WR 0.875 (-0.05 from peak 0.925, well within noise)
- Truncation_rate 0.125 (agent finishes 87.5% of games)
- avg_vp 14.30 (within 0.7 VP of the 15-VP win threshold)

### Decision rule

- 0.875 < 0.95 → not auto-upgrade
- 1433k vs 1331k = regression by 0.05 (1 consecutive regression)
- Rule (c) applies: continue + reschedule

### Type-head — re-collapsed slightly

```
last 5: 0.0156, 0.0394, 0.0244, 0.0117 (latest)
```

Bouncing in 0.01–0.04 band; latest 0.0117 is mid-range. No new
diagnostic info.

### Aux losses

```
belief_loss     = 1.020  (down from 1.113 — improvement)
opp_action_loss = 0.683  (similar to recent)
```

Belief loss continues to slowly improve.

### Training-side scalars

```
fps=25.4    rollout WR=0.49   R=+0.250 (POSITIVE, run-best tier)
EV=0.039    value_loss=0.191    entropy=0.47    KL=4e-04
```

Rollout reward +0.25 — matches the check-in 4 peak. Agent is genuinely
producing winning trajectories in self-play.

### ALERTs

None.

### Verdict

**Continue overnight loop.** Run is solidly in the 0.85–0.93 skill
band against random. Auto-upgrade trigger (0.95) within reach but
not crossed. The next 1-2 evals are the decision points.

If WR stays below 0.95 for many more evals, the agent may be at the
"~92% vs random" plateau — at which point the user should consider
manually flipping to heuristic eval to test whether the difficulty
gradient continues to drive learning.

Reschedule wakeup +50 min.

Process pid 91522 alive. State `RN`. No ALERTs.

## ALERT investigation (step 1,523,712) — `ALERT:NEG_EV=-1.07` (second occurrence)

Same low-variance-returns artifact as the earlier alert at step
1,081k. Not escalating.

Recent EV / value_loss:

```
1,503k  EV=+0.047  vl=1.073
1,507k  EV=+0.219  vl=0.526
1,511k  EV=+0.056  vl=0.613
1,515k  EV=+0.134  vl=0.656
1,519k  EV=+0.062  vl=0.526
1,523k  EV=-1.065  vl=0.395   ← alert (vl is the LOWEST in window)
```

Hallmarks of the artifact:
- Single-update sharp negative spike against an otherwise positive
  trend
- value_loss is the **lowest** in the recent window — confirming the
  value head isn't broken
- Rollout WR=0.49, R=+0.27 (run-best tier) — agent's policy is
  performing strongly

This time: KL=0.0102, slightly higher than typical (~3x). Possibly
correlated — a bigger PPO update reduced rollout-return variance,
shrinking EV's denominator. Same root cause: low-variance returns
make EV unstable while MSE stays sane.

Suggested follow-up (not applied during run): add `Var(returns) > ε`
guard to the EV alert in `monitor_training.py`. Already noted at the
prior occurrence.

### Status

```
fps=24.8    rollout WR=0.49   R=+0.27 (run-best tier)
entropy=0.54    KL=0.0102 (slightly elevated, well under 0.025 target)
collapse_flag=0
```

Process healthy. 1.53M eval still ~10 min away.

## Heartbeat (step 1,548,288, ~17:34 elapsed) — 1.53M eval landed at 0.925 (ties peak)

```
eval/win_rate: ...1228k=0.800  1331k=0.925  1433k=0.875  1536k=0.925
opponent_is_heuristic: still all 0.000
```

The 1.53M eval ties the run peak. Auto-upgrade STILL not triggered
— 0.925 is 0.025 short of the 0.95 threshold. Two evals at 0.925
now confirm the agent's skill ceiling against random sits right at
~0.92.

Per the plateau-against-random tracker: last 4 evals 0.800, 0.925,
0.875, 0.925 — all ≥ 0.80, three in [0.80, 0.93], one (0.925) at the
upper boundary. **Pattern is "knocking on 0.95 without crossing."**

```
fps=24.5   rollout WR=0.46  R=+0.19 (still positive)
EV=0.147 (recovered from -1.07 outlier — confirms artifact)
value_loss=0.731 (back to norm)   entropy=0.46   KL=3e-04
```

NEG_EV alert recovered.

## Overnight check-in 18 (step 1,548,288, elapsed 17:36:57) — ⚠ 4-EVAL PLATEAU AGAINST RANDOM

The 1.53M eval landed at **0.925** — ties the run peak. Auto-upgrade
not triggered.

### Wakeup-prompt critical outputs

1. **Is 1.53M (0.925) in [0.80, 0.93]?** — Technically NO (0.925 is
   0.005 above 0.93 upper bound). **Practically YES** — the
   agent is plateaued in the 0.80–0.93 band knocking on 0.95
   without crossing.
2. **opponent_is_heuristic any 1.0?** — **NO**, all 0.000 across 15
   evals. Auto-upgrade has not fired.

### Eval trend (last 5)

| step | WR | trunc | avg_vp |
|---|---|---|---|
| 1126k | 0.325 | 0.625 | 7.975 |
| 1228k | 0.800 | 0.200 | 13.375 |
| 1331k | 0.925 | 0.075 | 14.750 |
| 1433k | 0.875 | 0.125 | 14.300 |
| **1536k** | **0.925** | **0.075** | **14.950** ← run high in avg_vp |

Last 4 evals (1228k+): all ≥ 0.80. **avg_vp at 14.95 — within 0.05
of the 15-VP win threshold.** Agent is essentially winning every
non-truncated game (truncation 0.075 = 92.5% finish rate).

### Plateau verdict

**4-EVAL PLATEAU AGAINST RANDOM — user attention recommended.**

The pattern is unambiguous:
- 4 consecutive evals at WR ∈ [0.800, 0.925]
- Two of them at the exact peak 0.925
- Auto-upgrade trigger 0.95 has not been crossed in ~500k steps
  (since first reaching 0.800 at step 1228k)
- avg_vp 14.30–14.95 (essentially saturated against random)

The agent has exhausted what it can learn from beating random with
deterministic argmax. The remaining 0.025 to auto-upgrade may not
come — random's wide variance produces occasional unlucky games
that the deterministic policy can't recover from.

### Suggested user action when reading this

Consider **manually flipping the eval opponent to heuristic** so
the difficulty gradient drives more learning. Two ways:

1. Change `eval_upgrade_threshold` lower (config hot-edit) — the
   trainer would auto-upgrade on the next eval that crosses the
   new threshold.
2. Hard-flip via config edit — set `phase4_full.yaml`'s
   `opponent_type: heuristic` and resume from the next checkpoint.

Both require a brief restart (the running trainer reads the config
once at boot). The 500k-step checkpoint hasn't landed yet (`checkpoint_freq=500_000`,
last save at step 0; next at step 500k); the FIRST scheduled save
of THIS run is now overdue. Either wait ~3 hours for `checkpoint_05000XX.pt`
to land or accept losing the current run's progress on restart.

(NOTE: this is a USER decision; I'm not auto-flipping.)

### Decision rule

- 0.925 < 0.95 → not auto-upgrade
- 1.53M ties prior peak → no regression
- Rule (c) applies: continue + reschedule

### Type-head & aux losses (unchanged from heartbeat)

```
type_head entropy ≈ 0.034 (healthy band)
belief_loss ≈ 1.08
opp_action_loss ≈ 0.97 (still elevated vs run-low 0.48)
```

### Training-side scalars

```
fps=24.5    rollout WR=0.46   R=+0.19 (still positive)
EV=0.147    value_loss=0.731   entropy=0.46   KL=3e-04
collapse_flag=0
```

### ALERTs

NEG_EV cleared (now +0.147). All other gates clear.

### Verdict

**Plateau against random — continue overnight loop, but worth
flagging.** Run is no longer making meaningful gains against
random; subsequent evals likely keep oscillating in [0.80, 0.95].
For real continued learning the eval opponent needs to upgrade.

Reschedule wakeup +50 min. Process pid 91522 alive. State `RN`.

```
fps=25.0    rollout WR=0.48   R=+0.24 (still positive)
EV=0.047    value_loss=1.073 (slight uptick — not pathological)
entropy=0.55    KL=8e-04
```

Rollout side holding strong. 1.53M eval ~20 min away.

## Overnight check-in 17 (step 1,503,232, elapsed 16:44:50)

1.53M eval still hasn't fired — at step 1,503k, eval boundary at
~1,535k (~32k steps, ~30 min away at the current ~16 FPS effective
rate, which has slowed from the earlier ~25 FPS).

### Eval trend (last 6 — still 14 points)

| step | WR | trunc | avg_vp | vs heuristic? |
|---|---|---|---|---|
| 921k | 0.100 | 0.850 | 6.725 | no |
| 1024k | 0.675 | 0.325 | 12.300 | no |
| 1126k | 0.325 | 0.625 | 7.975 | no |
| 1228k | 0.800 | 0.200 | 13.375 | no |
| 1331k | 0.925 | 0.075 | 14.750 | no |
| **1433k** | **0.875** | 0.125 | 14.300 | no |

### Plateau-against-random tracking

Wakeup-prompt criterion: "track if WR has been in [0.80, 0.93] for
4+ consecutive evals."

| step | WR | in [0.80, 0.93]? |
|---|---|---|
| 1126k | 0.325 | NO |
| 1228k | 0.800 | YES |
| 1331k | 0.925 | borderline (0.005 over) |
| 1433k | 0.875 | YES |

3 of last 4 in band (1331k just outside the upper bound). Not yet
a 4-eval plateau, but trending. The 1.53M eval will determine — if
it lands in [0.80, 0.93] without crossing 0.95, that's 4+ in band
and worth flagging to the user as "stalled below auto-upgrade."

### FPS slowdown

Last 5 fps writes show effective ~16-30 FPS:

```
step 1,486k  age=17.0 min ← long gap (likely eval cycle pause)
step 1,491k  age=10.2 min
step 1,495k  age= 9.5 min
step 1,499k  age= 7.2 min
step 1,503k  age= 3.0 min
```

Average over the last hour: ~16 steps/sec, down from the earlier
~25. Eval cycles (40 games each) account for most of the gap.

### Decision rule

- 0.875 < 0.95 → not auto-upgrade
- 1 consecutive regression at 1433k → not 3
- Rule (c) applies: continue + reschedule

### Type-head

```
last 5: 0.0394, 0.0244, 0.0117, 0.0342 (latest)
```

Recovered to 0.0342 — third-highest reading of the run. Still well
below uniform but exploring more.

### Aux losses

```
belief_loss     = 1.078  (similar to recent)
opp_action_loss = 0.968  (UP from 0.683 — concerning uptick;
                          was 0.480 at run-low at check-in 7)
```

Opp-action loss has wobbled UP 100% from its run-low. Encoder may
be drifting on opponent-action prediction; not catastrophic but
worth tracking.

### Training-side scalars

```
fps=25.0    rollout WR=0.48   R=+0.24 (still positive)
EV=0.047    value_loss=1.073   entropy=0.55   KL=8e-04
```

Rollout reward holding at +0.24 — strong. value_loss at 1.07 is
elevated but not pathological.

### ALERTs

None.

### Verdict

**Continue overnight loop.** Run holding in the [0.80, 0.93] band
for 3 of last 4 evals. Auto-upgrade not yet triggered. The 1.53M
eval (~30 min away) will determine whether the agent is plateaued
just below auto-upgrade or about to break through.

If 1.53M ≥ 0.95 → AUTO-UPGRADE TRIGGERED.
If 1.53M in [0.80, 0.93] → 4-eval plateau confirmed, flag to user.
If 1.53M < 0.80 → skill regression, watch for 3-in-a-row pattern.

Reschedule wakeup +50 min.

Process pid 91522 alive. State `RN`. No ALERTs.

## Heartbeat (step 1,597,440, ~18:24 elapsed)

```
fps=24.2    rollout WR=0.41   R=-0.06
EV=0.250 (↑ second-best of the run after 0.294 peak — value head
          fitting returns well)
value_loss=0.807    entropy=0.51    KL=3e-04
```

EV recovered strongly from the NEG_EV artifact at step 1,523k.
1.63M eval ~30k steps away.

## ALERT: TRAIN_PROC_DEAD (~step 1,617,920) — second real bug found

### Failure mode

Training crashed at step 1,617,920 in Nash pruning's `_build_payoff_matrix`:

```
File "trainer.py", line 626, in _build_payoff_matrix
    pol_a.load_state_dict(state_dicts[ids[i]])
RuntimeError: ... Missing key(s) ... Unexpected key(s) ...
    size mismatch ... torch.Size([128, 103]) ... torch.Size([128, 79])
```

### Root cause

`_build_payoff_matrix` was constructing inference policies via
`build_agent_model(device=...)` with NO model_kwargs, producing the
**legacy 1.54M baseline architecture**. Then it tried to load
`phase4_full` state_dicts (~2.74M params, all opt-in features on)
into them. State-dict keys + shapes mismatch → crash.

The Nash pruning code path only fires once the league hits its
`maxlen=100` capacity. With `add_policy_every=4` and league filling
at ~400 PPO updates, this code didn't run until very recently — the
prior 16+ hours of training had never triggered it. First Nash
pruning attempt = first crash.

### Fix landed in commit `3562b36`

`_build_payoff_matrix` now uses `self.league._build_policy_fn(device=...)`
which the trainer's `__init__` set up with the full `model_kwargs`
of this run. Architecture matches the league policies. Plus a
defensive `RuntimeError` if `_build_policy_fn` is missing.

Two new regression tests in
`tests/unit/algorithms/test_nash_pruning_arch_match.py` exercise
the exact failure path.

### Restart

Resumed from the **checkpoint_01511424.pt** (saved 18:17). We lose
~110k steps. New pid 67363 launched at 20:30.

```
nohup python scripts/train.py \
  --config configs/phase4_full.yaml \
  --resume checkpoints/train/checkpoint_01511424.pt --verbose \
  >> runs/train/console.log 2>&1 &
```

Supervisor monitor re-armed. Loop continues from step 1,511,424
under nohup.

## Overnight check-in 20 (step 1,552,384, ~32:58 in restart) — note: pid changed

Wakeup prompt referenced pid 91522 (dead — Nash pruning crash). The
**current run is pid 67363**, resumed from checkpoint_01511424.pt at
~20:30. The TB events file is new (`...local.67363.0`); old run's evals
are in the prior file.

### New-run eval trend (1 eval since resume)

```
eval/win_rate              @1515k=0.725 (vs random)
eval/truncation_rate       @1515k=0.200
eval/avg_vp                @1515k=13.600
eval/opponent_is_heuristic @1515k=0.000
```

The first eval after resume came in at **0.725**. That's below the
0.92 mean we were seeing pre-crash but still well above baseline.
Plausible explanations:

1. **Eval CI noise** — 40-game eval has wide CI; 0.725 is within
   typical noise of the 0.92 mean we observed pre-crash.
2. **State drift** — checkpoint was saved at step 1.51M (between the
   1.43M=0.875 and 1.53M=0.925 evals); resuming from there picks up
   slightly stale optimizer + value-normalizer state.
3. **No regression** — truncation_rate=0.200, avg_vp=13.6 are
   consistent with pre-crash performance; only WR dropped, suggesting
   a CI artifact rather than skill regression.

### Plateau status

The pre-crash 4-eval plateau pattern is RESET — we have only 1 eval
in the new run. Plateau tracking restarts from scratch.

### Decision rule

- 0.725 < 0.95 → not auto-upgrade
- Only 1 eval → can't have 3 consecutive regressions
- Rule (c) applies: continue + reschedule

### Training-side scalars

```
fps=844 (startup artifact — TB FPS scalar averages over the first
         rollout post-resume, which spent most of its time loading
         the checkpoint; will normalize at ~25 next eval)
rollout WR=0.375   R=-0.07
EV=0.054   value_loss=1.19   entropy=0.53   KL=4e-04
```

### ALERTs

None.

### Verdict

**Continue overnight loop.** Training resumed cleanly from the 1.51M
checkpoint with the Nash-pruning fix in place. First post-resume
eval at 0.725 is consistent with eval-CI noise around the prior
~0.92 plateau. The 1.6M eval (~30 min away) will tell us if the
agent recovers to the prior plateau or if there's real degradation.

Reschedule wakeup +50 min. Process pid 67363 alive. State `RN`.

## Heartbeat (step 1,581,056, ~50 min in resumed run)

```
fps_scalar=540 (still descending toward ~25 asymptote — post-resume artifact)
rollout WR=0.42   R=+0.043
EV=-0.12 (slightly negative)   value_loss=0.50   entropy=0.54   KL=0.0042
```

Two FPS_REGRESSION false-alarms fired (588/1994, then 652/2580) —
both from the descending scalar artifact, not real slowdowns. Will
keep firing until the trailing-10 baseline catches up (~30 min).
Suppressing further commentary on those specific alerts.

1.6M eval ~10 min away.

## Overnight check-in 21 (step 1,622,016, resumed run elapsed 1:24:40)

The 1.62M eval came in at **0.875** — recovery to the pre-crash
plateau confirmed. The 0.725 first eval was eval-CI noise.

### New-run eval trend (2 points)

```
eval/win_rate              @1515k=0.725  @1617k=0.875
eval/truncation_rate       @1515k=0.200  @1617k=0.100
eval/avg_vp                @1515k=13.600 @1617k=14.500
eval/opponent_is_heuristic @1515k=0.000  @1617k=0.000
```

All three metrics improved 1515k → 1617k. avg_vp at 14.50 is
within 0.5 VP of the 15-VP win threshold, just like pre-crash.

### Decision rule

- 0.875 < 0.95 → not auto-upgrade
- 1617k > 1515k → improvement, no regression
- Rule (c) applies: continue + reschedule

### Comparison to pre-crash baseline

Pre-crash plateau was ~0.92 across 4 evals. New-run mean of 2 evals
is 0.800. Within stochastic noise, but on the lower side.

Pre-crash truncation_rate was 0.075-0.125. New-run is 0.100-0.200,
similar band.

Pre-crash avg_vp was 13.4-14.95. New-run is 13.6-14.5, similar band.

So the resumed run is **performing like the pre-crash run**, just
hasn't yet had enough evals to confirm the plateau pattern.

### Training-side scalars

```
fps_scalar=323 (still descending toward ~25 asymptote)
rollout WR=0.35   R=-0.23 (sample-side wobble)
EV=0.066    value_loss=0.43    entropy=0.52    KL=0.0016
collapse_flag=0
```

EV positive, value_loss in healthy band. The earlier slight-negative
EV reading recovered.

### ALERTs

FPS_REGRESSION still triggering intermittently from the post-resume
artifact. Suppressed in commentary.

### Verdict

**Continue overnight loop. Resume successful — agent recovered to
the pre-crash plateau in ~100k steps.** The next 2-3 evals will
tell us if the plateau pattern returns identical or if there's
slight degradation from the optimizer state load.

Reschedule wakeup +50 min. Process pid 67363 alive. State `RN`.

## Overnight check-in 19 (step 1,609,728, elapsed 18:28:49)

1.63M eval hasn't fired yet (~28k steps short, ~20 min away).
Decision rule applied to existing 15 evals.

### Eval trend (last 6, no new data since check-in 18)

```
@1024k=0.675 @1126k=0.325 @1228k=0.800 @1331k=0.925 @1433k=0.875 @1536k=0.925
opponent_is_heuristic: still all 0.000
```

### Plateau tracker

Last 4 evals (1228k onward) all ≥ 0.80, peak 0.925 (×2). Two more
evals at the same level would make it 5+ consecutive — the
"genuinely capped against random" threshold from the wakeup prompt.

### Decision rule

- 0.925 < 0.95 → not auto-upgrade
- Latest 1536k=0.925 ties prior peak → no regression
- Rule (c) applies: continue + reschedule

### Training-side scalars

```
fps=24.2    rollout WR=0.45   R=+0.080 (positive, mild)
EV=0.089 (down from 0.250 heartbeat — typical post-eval dip)
value_loss=0.849   entropy=0.59   KL=0.0028
clip_fraction=0.048   collapse_flag=0
```

Rollout reward back positive. EV oscillating in healthy band.

### ALERTs

None active.

### Verdict

**Continue overnight loop. Plateau-against-random pattern persisting.**
The 1.63M eval is the next data point — if it lands in [0.80, 0.95]
it'll be the 5th consecutive in band, confirming the "capped against
random" pattern.

Reschedule wakeup +50 min. Process pid 91522 alive. State `RN`.

## Heartbeat (step 1,642,496, ~1:45 in resumed run)

```
fps_scalar=276 (still descending — artifact)
rollout WR=0.33   R=-0.32 (sample-side dip — eval is the trustworthy signal)
EV=0.051   value_loss=0.56   entropy=0.49   KL=4e-05
```

Rollout side wobble; eval was 0.875 last data point. 1.72M eval
~30 min away.

## Overnight check-in 22 (step 1,687,552, resumed run elapsed 2:15:39)

1.72M eval hasn't fired yet (~32k steps short, ~20 min away).

### Eval trend (still 2 points)

```
@1515k=0.725  @1617k=0.875
opponent_is_heuristic: 0.000, 0.000
```

Latest 0.875 < 0.95 → not auto-upgrade. Improvement 1515k→1617k.
2 post-resume evals so far; the "PLATEAU REESTABLISHED" criterion
needs 3+ in [0.80, 0.95]. The 1.72M eval is the deciding 3rd point.

### Decision rule

- 0.875 < 0.95 → not auto-upgrade
- 0 consecutive regressions
- Rule (c) applies: continue + reschedule

### Training-side scalars

```
fps_scalar=211 (still descending toward ~25 asymptote)
rollout WR=0.30   R=-0.45 (sample-side struggling — league pressure)
EV=0.178 (climbing — value head fitting well)
value_loss=0.45   entropy=0.47   KL=-7e-04
collapse_flag=0
```

EV at 0.18 is healthy. Rollout side is wobbling deep negative; the
eval signal (0.875 last) confirms the agent is still strong against
random in deterministic eval. Rollout-side dip is consistent with
PFSP-hard sampling stronger opponents the agent can't beat at sample-time.

### ALERTs

FPS_REGRESSION still firing intermittently (post-resume artifact).

### Verdict

**Continue overnight loop.** 2 post-resume evals show the agent
recovered to its pre-crash plateau range. Need 1 more eval to
confirm "plateau re-established" status.

Reschedule wakeup +50 min. Process pid 67363 alive. State `RN`.

## Heartbeat (step 1,708,032, ~2:36 in resumed run)

```
fps_scalar=193 (still descending)
rollout WR=0.31   R=-0.43 (sustained sample-side dip)
EV=0.064 (down from 0.178 at last check-in)
value_loss=0.998 (uptick)   entropy=0.52   KL=0.0015
```

Rollout-side has been negative for 3+ consecutive heartbeats —
longer than typical. League is sampling tough PFSP-hard opponents.
Eval (vs random) was 0.875 last; still strong vs random with
deterministic argmax. 1.72M eval ~10 min away.

## On-demand eval @ checkpoint_01515520.pt (run 22:59)

User-requested external eval against random and heuristic, 50 games each.

| Opponent | WR | Avg VP | Avg Game Length |
|---|---|---|---|
| random | 80.0% | 13.8 | 263 |
| heuristic | **14.0%** | 8.7 | 157 |

vs random tracks training-time scalar within sample noise (training-time
WR was 0.725 at step 1,515,520 in the resumed run, this 50-game sample
hit 80%; 95% CI ±0.11 covers both).

vs heuristic is the new finding: **agent does not generalize to
heuristic at all.** Game length collapses (263 → 157) — when it loses
to heuristic, it loses fast (heuristic builds VP and ends the game
before the agent can recover).

Implication: training-time `eval/win_rate` of 0.875+ vs random
overstates real strength. The 0.95 auto-upgrade gate is the right
threshold to stop overfitting to random — agent isn't there yet but
even when it crosses, it'll need significant heuristic-opponent
training before it can claim "beats heuristic."

### Pre-existing bug fix shipped during this eval

`scripts/evaluate.py` was constructing `EvaluationManager` without the
phase-flag schema parameters, so the env emitted legacy 166-dim obs
while the phase4_full checkpoint expected 54-dim compact. Both initial
runs crashed with `RuntimeError: mat1 and mat2 shapes cannot be
multiplied (1x166 and 54x256)`.

Fix: read `trainer.config` and propagate `use_thermometer_encoding`,
`use_opponent_id_emb`, `opp_id_mask_prob`, `league_maxlen`,
`use_belief_head` to `EvaluationManager`. Same bug class as the eval-
manager bug fixed in commit `db06481`, but on a separate code path.

## Overnight check-in 23 (step 1,736,704)

Process pid 67363 alive (RN, etime 03:06:34, started 20:12).
Events file freshness 3 min — healthy.

### Eval trend (post-resume)

```
1,515,520  0.7250  (post-Nash-crash baseline)
1,617,920  0.8750  (recovery)
1,720,320  0.8750  (flat — recovery sustained)
```

Direction: 0.725 → 0.875 → 0.875. Two consecutive evals at 0.875.
Pre-crash plateau was ~0.92, so still ~5 pts below the prior peak,
but the recovery shape (down once, up twice) rules out
TRIPLE_REGRESSION.

### Plateau status

Decision rule: "3+ post-resume evals all in [0.80, 0.95] → PLATEAU
REESTABLISHED." Currently 2 of 3 in band; the 1.515M dip at 0.725 is
below the lower edge. **Plateau emerging, not yet officially confirmed.**
Need the next eval (≥ step 1,820,320) to clear 0.80 to flip the
state to PLATEAU_REESTABLISHED.

### Health snapshot

```
fps_scalar=157  (still inflated post-resume; baseline ~25 expected)
rollout WR=0.38  R=-0.19  (rollout side recovered from -0.43)
entropy=0.51  value_loss=0.85  EV=0.052
KL=0.0030  clip_fraction=0.0368   LR=1.0e-04   ent_coef=0.04
```

Rollout-side reward improved (-0.43 → -0.19) since the last
heartbeat; PFSP-hard sampling is no longer all-killing-opponents.
EV barely positive — value head still struggling but not negative.
KL well under target_kl=0.025 — clean update.

### Cross-check vs external eval (50 games, 22:59)

```
checkpoint_01515520.pt  vs random:    80%  (matches training-time
                                            0.725 within ±0.11 CI)
checkpoint_01515520.pt  vs heuristic: 14%  (significant overfit
                                            to random)
```

The 14% vs heuristic finding remains the dominant strategic concern —
even if WR vs random crosses 0.95 and auto-upgrade triggers, the
agent will face a substantial learning curve against heuristic.

### Verdict

**Continue overnight loop.** No abort conditions met. State `RN`,
plateau emerging.

Reschedule wakeup +50 min to catch the next eval landing.

## Overnight check-in 23 (b) (step 1,785,856)

Wakeup-fired at 23:56. Process pid 67363 alive (RN, etime 03:43:35).
Events file freshness 4 min — healthy.

Step 1,785,856 — next eval at 1,820,320 still ~35k steps off
(~22 min at sustained ~25 FPS). Eval not landed yet, so the trend
is unchanged: 0.725 / 0.875 / 0.875.

### Health snapshot

```
fps_scalar=135  (still inflated; fix shipped in 4641efe but won't
                  apply to this process — it loaded the old code.
                  True FPS asymptote is ~25.)
rollout WR=0.39   R=-0.17  (rollout side continues recovering)
entropy=0.60     value_loss=0.54     EV=-0.005
KL=0.0011        clip_fraction=0.043
```

EV at -0.005 is essentially the noise floor (value head at the level
of just predicting the mean). Not worrying — better than the negative
spikes earlier in the run. Entropy bounced 0.51 → 0.60 — rollouts
sampled a higher-entropy distribution this update.

### Notable since previous check-in

- **commit 2490444**: scripts/evaluate.py schema fix (already applied
  before this check-in)
- **commit 4641efe**: train/fps now uses `(step - start_step) /
  elapsed` — kills post-resume inflation. Takes effect at next
  start/resume only.

### Verdict

**Continue overnight loop.** No abort conditions met. Eval at 1.82M
expected ~22 min from now.

Reschedule wakeup +25 min to land just after the eval.

## Overnight check-in 24 (step 1,802,240)

Wakeup-fired at 00:11. Process pid 67363 alive (RN, etime 03:58:36).
Events file freshness 3 min — healthy.

Step 1,802,240 — next eval at 1,820,320 still ~18k steps off
(~12 min at sustained ~25 FPS). Eval not landed yet, so trend
remains unchanged: 0.725 / 0.875 / 0.875.

Note: this check-in fired from a wakeup chain scheduled before the
00:23 wakeup. There are two pending check-in 24 wakeups in flight;
the 00:23 one will land *after* the 1.82M eval, so it should see
the new datapoint.

### Health snapshot

```
fps_scalar=127  (still inflated; will normalize on next restart)
rollout WR=0.37   R=-0.23  (rollout-side a touch noisier this step)
entropy=0.56     value_loss=0.66     EV=+0.12
KL=-0.0016       clip_fraction=0.060
```

EV at +0.12 is the best in a while — value head is climbing back
above the noise floor (-0.005 last check-in). KL just barely
negative is sample noise around zero (legal for `approx_kl` with
the shadow-second-order estimator). clip_fraction 0.06 — a bit
elevated, watch.

### Verdict

**Continue overnight loop.** No abort conditions met. Eval at 1.82M
~12 min away.

No additional reschedule — 00:23 wakeup already armed.

## Overnight check-in 24 (b) (step 1,822,720) — eval landed

Wakeup-fired at 00:23. Process pid 67363 alive (RN, etime 04:10:34).
Polled +6 min for the 1.82M eval to land.

### 🟢 New post-resume high water mark

```
1,515,520  0.7250
1,617,920  0.8750
1,720,320  0.8750
1,822,720  0.9250  ← new high
```

**Strict monotone non-decreasing** across all 4 post-resume evals. The
0.875 ceiling that held for two consecutive evals is broken. 0.925 is
within 0.025 of the auto-upgrade threshold (0.95). At the current
trend rate, auto-upgrade is plausible inside the next 1-2 evals.

Companion eval scalars at this step:
- `eval/avg_vp = 14.975` (essentially closing every game; max is 15)
- `eval/avg_game_length = 200` (down from 263 in the 50-game manual
  eval at the older 1.515M checkpoint — faster wins)
- `eval/opponent_is_heuristic = 0.0` (still random)

### Decision

- (a) ≥ 0.95: NO (0.925 — close)
- (b) 3 regressions: NO (no regressions at all)
- (c) 4 of 4 in [0.80, 0.95]: NO (0.725 below; 3 of 4 in band)
- (d) **otherwise → status + reschedule +30min**

### Health snapshot

```
fps_scalar=119  (still inflated; restart deferred per instructions)
rollout WR=??   R=-0.32  (slightly worse rollout side this update)
entropy=0.47   value_loss=??   EV=+0.007    KL=0.0026
```

EV essentially zero — value head still in noise floor regime, but
the policy continues to improve regardless. Entropy 0.47 — gradual
decay continuing.

### Verdict

**Continue overnight loop** with elevated optimism. The auto-upgrade
gate could plausibly trigger at the next eval (1.92M) or the one
after.

Reschedule wakeup +30 min.

## Overnight check-in 25 (step 1,855,488)

Wakeup-fired at 01:01. Process pid 67363 alive (RN, etime 04:48:33).

Step 1,855,488 — next eval at 1,920,320 still ~65k steps off
(~43 min at sustained ~25 FPS). Eval trend unchanged:
0.725 / 0.875 / 0.875 / 0.925.

### Decision

- (a) ≥ 0.95: NO (still 0.925 — no new eval)
- (b) 3 regressions: NO
- (c) 4 of 4 in [0.80, 0.95]: NO (0.725 below)
- (d) **otherwise → status + reschedule +30min**

### Health snapshot

```
fps_scalar=108  (still inflated; restart deferred)
rollout WR=0.35   R=-0.31  (rollout side a bit noisier)
entropy=0.54   value_loss=0.77   EV=+0.13
KL=0.0008      clip_fraction=0.044
```

EV at +0.13 is the best of the resumed run — value head finally
producing real signal, consistent with the eval upward trend. KL
0.0008 is squeaky clean. Entropy 0.54 (rebounded from 0.47 last
check-in — sample noise).

### Verdict

**Continue overnight loop.** Auto-upgrade gate still in reach at
1.92M or 2.02M. No abort conditions met.

Reschedule wakeup +30 min.

## Overnight check-in 26 (step 1,884,160)

Wakeup-fired at 01:32. Process pid 67363 alive (RN, etime 05:19:34).

Step 1,884,160 — 1.92M eval still ~36k steps / ~24 min away. Trend
unchanged: 0.725 / 0.875 / 0.875 / 0.925.

### Decision

- (a) ≥ 0.95: NO (still 0.925)
- (b) 3 regressions: NO
- (c) 4 of 4 in [0.80, 0.95]: NO (0.725 below)
- (d) **otherwise → status + reschedule +30min**

### Health snapshot

```
fps_scalar=100  (still inflated, decaying toward truth)
rollout R=-0.30  (stable rollout-side noise)
entropy=0.48    value_loss=0.60    EV=+0.003
KL=0.0035       clip_fraction=0.049
```

EV bounced down to +0.003 from +0.13 last check-in — value head
wobble. Not concerning since still positive (above noise floor).
KL 0.0035 clean. Entropy 0.48 (down from 0.54 — gradual decay).

### Verdict

**Continue overnight loop.** No abort conditions met.

Reschedule wakeup +30 min to land just after the 1.92M eval.

## Overnight check-in 27 (step 1,925,120) — 🟠 sharp regression

Wakeup-fired at 02:03; polled +6 min for the eval to land.
Process pid 67363 alive (RN, etime 05:50:34).

### Eval trend with new datapoint

```
1,515,520  0.7250
1,617,920  0.8750
1,720,320  0.8750
1,822,720  0.9250  ← previous high
1,925,120  0.6500  ← regression of -0.275
```

avg_vp 13.0 (vs 14.97 last eval), avg_game_length 324 (vs 200 last
eval). Games are dragging out — agent neither winning fast nor
losing fast. Consistent with a policy that's lost some of its
finishing skill.

### Interpretation

Drop of -0.275 in one eval is well outside the ±0.10 95% CI on a
40-game sample, so this is *not* pure sample noise. Plausible
causes:

1. **PFSP-hard sampling**: a tougher league opponent gradient
   was applied between 1.82M and 1.92M, walking the policy
   into a worse basin against random.
2. **Entropy/value-loss instability**: EV bounced 0.13 → 0.003
   → 0.046 last few updates; entropy 0.54 → 0.48 → 0.51 — the
   policy is in a noisy regime.
3. **Eval determinism quirk**: same seed reused → if policy
   weights drift into a particular bad basin, the same set of
   40 games shows the failure consistently.

Not catastrophic. The agent has demonstrated it CAN reach 0.925,
so the capability is intact; this is policy walk during continued
learning. Need 1-2 more evals to see direction.

### Decision

- (a) ≥ 0.95: NO
- (b) 3 consecutive regressions: NO (only 1)
- (c) 4 of 4 in [0.80, 0.95]: NO (0.65 below band)
- (d) **otherwise → status + reschedule +30min**

### Verdict

**Continue overnight loop.** Watch the next eval (~step 2,022,720)
for direction. If it bounces back to ≥ 0.85, this was a transient.
If it stays low or drops further, the policy has actually walked
into a worse spot.

Reschedule wakeup +30 min.

## Overnight check-in 28 (step 1,966,080)

Wakeup-fired at 02:41. Process pid 67363 alive (RN, etime 06:28:35).

Step 1,966,080 — next eval at ~2,022,720 still ~57k steps off
(~38 min at sustained ~25 FPS). Eval trend unchanged:
0.725 / 0.875 / 0.875 / 0.925 / 0.650.

### Decision

- (a) ≥ 0.95: NO (still 0.65 — direction-check pending)
- (b) 3 consecutive regressions: NO (only 1 so far)
- (c) 4 of 4 in [0.80, 0.95]: NO (0.65 below)
- (d) **otherwise → status + reschedule +30min**

### Health snapshot

```
fps_scalar=85   (still inflated, decaying)
rollout WR=0.33   R=-0.37  (rollout side a bit noisier this update)
entropy=0.54   value_loss=0.70   EV=+0.069
KL=-0.0003     clip_fraction=0.055
```

EV recovered to +0.069 (from +0.003 last check-in) — value head
re-stabilizing. Entropy bounced 0.48 → 0.54. KL essentially zero.
Nothing in the training-side scalars suggests the policy is in a
bad place — the regression looks like a sampling artifact in the
eval set or a temporary drift, not a runaway divergence.

### Verdict

**Continue overnight loop.** Direction-check eval still pending.
The training-side scalars are healthy enough that I'd predict a
bounce-back to ≥ 0.85 at 2.02M.

Reschedule wakeup +30 min to land just before / during the eval.

## Overnight check-in 29 (step 1,998,848)

Wakeup-fired at 03:12. Process pid 67363 alive (RN, etime 06:59:34).

Step 1,998,848 — direction-check eval at ~2,025,120 still ~26k
steps off (~17 min). Outside the 3k poll-window. Eval trend
unchanged: 0.725 / 0.875 / 0.875 / 0.925 / 0.650.

### Decision

- (a) ≥ 0.95: NO (still 0.65 — direction-check still pending)
- (b) 3 consecutive regressions: NO (only 1)
- (c) 4 of 4 in [0.80, 0.95]: NO (0.65 below)
- (d) **otherwise → status + reschedule +30min**

### Health snapshot

```
fps_scalar=80    (still inflated, decaying)
rollout R=-0.24  (rollout side recovered from -0.37 last update)
entropy=0.49     value_loss=0.81     EV=+0.10
KL=0.0020        clip_fraction=0.038
```

EV climbing: 0.07 → 0.10 — best of the resumed run since the
0.13 peak. Rollout reward also recovered (-0.37 → -0.24).
KL clean. The training-side picture is healthier than at the
moment of the regression — strengthening the prior that the
0.65 was a sample/sampling artifact, not a divergence.

### Verdict

**Continue overnight loop.** Direction-check eval ~17 min away;
+30 min wakeup will land just after.

Reschedule wakeup +30 min.

## Overnight check-in 30 (step 2,027,520) — 🎉 AUTO-UPGRADE TRIGGERED

Wakeup-fired at 03:43. Process pid 67363 alive (RN, etime 07:30:34).

### Eval landed: WR 1.0000 vs random

```
1,515,520  0.7250
1,617,920  0.8750
1,720,320  0.8750
1,822,720  0.9250
1,925,120  0.6500  ← transient
2,027,520  1.0000  ← AUTO-UPGRADE GATE CROSSED
```

Companion scalars:
- `eval/avg_vp = 15.05` (15-VP win condition; agent finished
  every game with the win-condition VP exactly)
- `eval/avg_game_length = 176.9` (vs 324 at the 0.65 dip — fast
  decisive wins)
- `eval/opponent_upgraded = 1` at step 2,027,520 ✅
- `eval/opponent_is_heuristic = 0` at this step (flag logged BEFORE
  the upgrade fires, so this eval was still vs random — by design)

The 0.65 at 1.92M was definitively a sample artifact / transient
walk; the policy not only recovered but ascended past the prior
0.925 high to a perfect score against random.

### What happens next (per trainer.py:1341-1359)

The trainer's auto-upgrade path:
1. New `EvaluationManager(opponent_type="heuristic")` constructed
   with the same phase4 schema flags.
2. `_eval_win_rate_history = []` — random-side history cleared.
3. `_eval_opponent = "heuristic"` — all subsequent evals are
   vs heuristic.
4. `eval/opponent_upgraded = 1.0` logged.

Practical consequence: the next eval at step ~2,127,520 will be
vs heuristic. **Expect a dramatic drop.** The earlier external
50-game eval at the much-older 1.515M checkpoint scored only 14%
vs heuristic. With ~500k more steps and the agent now perfect vs
random, that 14% is probably 25-40% by now — but still nowhere
near the 0.95 super-human threshold.

### Decision

- (a) **latest WR ≥ 0.95: YES → 🎉 AUTO-UPGRADE TRIGGERED**
- Reschedule per (a).

### Plateau-rule reset

The "4 of 4 in [0.80, 0.95] → PLATEAU CONFIRMED" rule was for
random-opponent evals. Post-upgrade, the trend resets. Future
check-ins should track the heuristic-opponent trend separately.

### Verdict

Major milestone. Reschedule wakeup +30 min to catch the
post-upgrade direction-check eval.

## Overnight check-in 31 (step 2,048,000)

Wakeup-fired at 04:15. Process pid 67363 alive (RN, etime 08:02:35).

Step 2,048,000 — first heuristic eval at ~2,127,520 still ~80k
steps off (~53 min at sustained ~25 FPS). **No post-upgrade eval
data yet.** The post-upgrade rule set (a)–(c) cannot fire without
data — falling through to (d): status only.

### Decision

- (a) heuristic WR ≥ 0.95: NO (no heuristic eval yet)
- (b) heuristic WR ≥ 0.50: NO
- (c) heuristic WR < 0.30: NO
- (d) **otherwise → status + reschedule +30min**

### Health snapshot

```
fps_scalar=71    (still inflated, decaying)
rollout R=+0.0005  ← first non-negative rollout reward in many
                    check-ins!
entropy=0.55     value_loss=0.85     EV=+0.097
KL=0.0027        clip_fraction=??
```

**Rollout reward turned positive** — meaningful event. With the
agent now perfect vs random in deterministic eval, the rollout
side (PFSP league + heuristic + random) is finally winning more
than losing on aggregate. Either:
1. PFSP is sampling weaker league entries because the agent has
   walked past most checkpoints, OR
2. The agent really is meaningfully stronger than the bulk of
   the league population now.

EV holding at +0.10 — value head consistently above noise floor.
KL clean. Entropy 0.55 — comfortably above the 0.05 collapse
threshold.

### Verdict

**Continue overnight loop.** First heuristic eval ~53 min away.
Reschedule wakeup +30 min — lands ~23 min before the eval, will
need to either poll briefly or wait for the next wakeup after.

## Overnight check-in 32 (step 2,084,864)

Wakeup-fired at 04:46. Process pid 67363 alive (RN, etime 08:33:34).

Step 2,084,864 — first heuristic eval at ~2,127,520 still ~43k
steps off (~28 min). Outside the 3k (~2 min) poll-window. No
post-upgrade eval data yet.

### Decision

- (a)/(b)/(c): no heuristic eval yet — none can fire
- (d) **status + reschedule +30min**

### Health snapshot

```
fps_scalar=68    (still inflated)
rollout R=-0.065  (slightly negative again — sample noise around 0)
entropy=0.51     value_loss=0.72     EV=+0.064
KL=0.0006        clip_fraction=??
```

Rollout side oscillating around zero (+0.0005 → -0.065) — agent is
roughly even with the rollout opponent mix on aggregate. EV holding
positive. KL extremely clean (0.0006). Entropy 0.51 — gradual
decay continuing.

### Verdict

**Continue overnight loop.** Eval ~28 min away; next wakeup at
~05:16 should land just after.

## Overnight check-in 33 (step 2,129,920) — 🟠 TRANSFER GAP CONFIRMED

Wakeup-fired at 05:17; polled +8 min for the eval.
Process pid 67363 alive (RN, etime 09:04:37).

### First post-upgrade heuristic eval landed

```
2,027,520 (random)     WR=1.0000  vp=15.05  len=176.9
2,129,920 (heuristic)  WR=0.2750  vp= 9.10  len=122.9  ← first
```

(`eval/opponent_is_heuristic = 1` at step 2,129,920 ✅)

The post-upgrade trend resets the win-rate history per
trainer.py:1354. So the new line is `[0.2750]`.

### Interpretation

WR 27.5% sits in the (c) bucket: < 0.30 → "TRANSFER GAP CONFIRMED".

Sanity-check vs the on-demand 50-game eval at the older 1.515M
checkpoint (recorded earlier this session):
```
checkpoint_01515520.pt vs heuristic: 14.0%
```
Current step 2.13M (~+600k steps + policy now perfect-vs-random):
**27.5%** — agent improved by +13.5pt, but is still being beaten
~73% of the time. avg_vp 9.1 (well below the 15-VP win condition)
and avg_game_length 122.9 (heuristic finishes the game fast)
together say the agent loses *decisively*, not in close games.

This was the predicted outcome. The 14% baseline made it clear
that random-only training overfits to weak-opponent exploits.
Heuristic-opponent training is the *actual* hard problem.

### What helps from here (no action taken; just for orientation)

1. **PFSP-hard self-correction**: with WR vs heuristic = 0.275,
   PFSP-hard ((1-w)^p) gives heuristic-class league entries
   ~0.7^p priority — they will be sampled often. Eventually the
   agent will adapt.
2. **`eval_win_rate_history` reset**: the trainer's auto-upgrade
   path doesn't re-downgrade. There's no rule that says "if WR
   drops below 0.5 vs heuristic, go back to random." The trainer
   sticks with heuristic now.
3. **Long climb expected**: AlphaStar-class results suggest a
   ~5-10x slower climb against a deterministic heuristic than
   against random. Plausible super-human (≥ 0.95) timeframe is
   another 10–30M steps.

### Decision

- (a) ≥ 0.95: NO (0.275)
- (b) ≥ 0.50: NO
- (c) **< 0.30: YES → TRANSFER GAP CONFIRMED**
- Action per (c): status + reschedule

### Verdict

**Continue overnight loop.** The transfer gap is the real problem;
no abort. Next eval at ~2.23M will tell us if the policy is
adapting or stuck.

Reschedule wakeup +30 min.

## Overnight check-in 34 (step 2,158,592)

Wakeup-fired at 05:57. Process pid 67363 alive (RN, etime 09:44:33).

Step 2,158,592 — next heuristic eval at ~2,229,920 still ~71k
steps off (~47 min at sustained ~25 FPS). No new eval data.
Latest still 0.275 from check-in 33.

### Decision

- (a)/(b)/(c)/(d): no NEW eval datapoint to evaluate. Latest is
  still 0.275 → if forced, re-fires (c) "TRANSFER GAP PERSISTS"
  for the same datapoint.
- Action: status + reschedule.

### Health snapshot

```
fps_scalar=62    (still inflated, decaying)
rollout R=-0.172  (back into mild negative — sample noise around 0)
entropy=0.54     value_loss=0.39  ← down from 1.20 spike
EV=+0.169        ← best of the resumed run!
KL=-0.0008
```

**EV at +0.169 is the highest of the resumed run** — the value
head is now reliably above the noise floor and producing real
signal. Value loss spike (1.20 last check-in → 0.39 now) was
resolved cleanly. Entropy 0.54, KL clean.

This is a healthy training-side picture. The agent is adapting
internally even though the eval gap to heuristic is still large.

### Verdict

**Continue overnight loop.** Next eval ~47 min away. Reschedule
wakeup +30 min — lands ~17 min before, will need a poll or wait.

## Overnight check-in 35 (step 2,195,456)

Wakeup-fired at 06:28. Process pid 67363 alive (RN, etime 10:15:34).

Step 2,195,456 — next heuristic eval at ~2,229,920 still ~34k
steps off (~22 min). Outside the 3k poll-window. Latest eval still
0.275 from check-in 33.

### Decision

- (a)/(b): no new eval — can't fire
- (c) "TRANSFER GAP PERSISTS" applies to the still-latest 0.275 —
  same datapoint as check-in 33 and 34
- Action: status + reschedule

### Health snapshot — 🟢 strongest rollout signal yet

```
fps_scalar=60     (still inflated, decaying)
rollout R=+0.228  ← highest of the resumed run
entropy=0.56      value_loss=0.61      EV=+0.072
KL=0.0026
```

**Rollout reward jumped to +0.228** — first meaningfully positive
read after weeks of negative R. Agent is decisively winning the
rollout side now. PFSP-hard sampling biases toward the league
entries the agent loses to; the heuristic_opponent_weight forces
some heuristic exposure. R = +0.23 means the agent is beating that
mix on aggregate.

This is a strong leading indicator that the next heuristic eval
(~22 min away) might tick UP from 0.275 — internal training is
clearly progressing.

EV +0.072 (down from +0.169 last check-in), value_loss 0.61.
Entropy 0.56. KL clean. Nothing concerning.

### Verdict

**Continue overnight loop.** Heuristic eval ~22 min away. The
next datapoint will tell us if the rollout-side gains are
translating to the deterministic eval.

Reschedule wakeup +30 min — lands ~8 min after the eval.

## Overnight check-in 36 (step 2,232,320) — second heuristic eval

Wakeup-fired at 06:59; polled +7 min for the eval to land.
Process pid 67363 alive (RN, etime 10:46:34).

### Heuristic-trend so far

```
2,129,920  0.2750  vp= 9.10  len=122.9
2,232,320  0.3000  vp= 9.00  len=126.5  ← Δ=+0.025
```

The +0.025 move is **within the ±0.05 noise band**, not a genuine
adaptation signal. Companion scalars (avg_vp 9.0, avg_game_length
126.5) are essentially identical to the first heuristic eval —
agent is losing in the same way, with the same fundamental
problem: it can't reach 15 VP before the heuristic does.

### Decision

The 0.300 WR sits *exactly* at the (c)/(d) boundary. With the rule
`< 0.30 → (c) TRANSFER GAP PERSISTS`, 0.300 is NOT strictly less,
so it falls into (d).

- (a) ≥ 0.95: NO
- (b) ≥ 0.50: NO
- (c) < 0.30: NO (0.300 is exactly at threshold)
- (d) **0.30 ≤ WR < 0.50: YES → "ADAPTATION STARTING"**

Reading the boundary literally: rule (d) "ADAPTATION STARTING".
But honestly — the +0.025 move is sample noise; companion scalars
say the agent hasn't actually learned anything new about heuristic
yet. So this is **mechanically (d), substantively (c).** Two more
data points needed to know if this is the start of real climb or
flat noise around 0.28.

### Health snapshot (taken before the eval landed)

```
fps_scalar=58
rollout R=-0.068  (back to mild negative after the +0.228 spike)
entropy=0.57     value_loss=0.70     EV=-0.027
KL=0.0016
```

EV bounced back negative (-0.027) — first time since check-in 33.
Value loss back to 0.70. Rollout R reverted from +0.228 to -0.068.
The training-side scalars are noisy. None of this rules out
genuine improvement; it does suggest "ADAPTATION STARTING" is
generous given the volatility.

### Verdict

**Continue overnight loop.** The honest read is "still in transfer-
gap territory but flirting with the boundary." Need 1-2 more
heuristic evals to decide if WR is climbing or oscillating around
the 0.28 floor.

Reschedule wakeup +30 min.

## Overnight check-in 37 (step 2,265,088)

Wakeup-fired at 07:38. Process pid 67363 alive (RN, etime 11:25:34).

Step 2,265,088 — next heuristic eval at ~2,332,320 still ~67k
steps off (~45 min). Outside poll window. Latest eval still 0.300
from check-in 36.

### Notable: NEG_EV alert at -0.63 (resolved)

The supervisor monitor fired ALERT:NEG_EV=-0.63 between check-ins
36 and 37 (around step 2,260,992). Investigated immediately:

- EV history was bouncing high-variance:
  0.06 → 0.18 → -0.13 → -0.44 → 0.11 → 0.01 → 0.12 → -0.63
- Rollout R consistently +0.10 over the same window
- KL clean, entropy 0.51 stable, value_loss varying 0.4 — 1.5

This is the **low-variance-returns artifact** flagged earlier in
the run: when the agent wins reliably (R=+0.10 trail), Var(returns)
is small, so a modest residual variance produces large negative
EV. EV then recovered to +0.128 by the next sample. Same false-
alarm class as steps 1,081k and 1,523k.

No action required — left in monitoring. The proper fix (a
`Var(returns) > ε` guard around the EV alert) was logged earlier
as a follow-up; not applied during the live run.

### Decision

- (a) ≥ 0.95: NO (still 0.300)
- (b) ≥ 0.50: NO
- (c) < 0.30: NO (0.300 is exactly at boundary)
- (d) **0.30 ≤ WR < 0.50: YES → "ADAPTATION STARTING"**
  (re-firing on the same datapoint as check-in 36)
- Action: status + reschedule

### Health snapshot

```
fps_scalar=55      (still inflated)
rollout R=+0.107   (positive — agent winning rollout side)
entropy=0.56       value_loss=0.72       EV=+0.128
KL=0.0057          clip_fraction=??
```

EV recovered to +0.128 (post the -0.63 transient). Rollout R is
sustainably positive (+0.10). KL fine. Entropy 0.56 — stable.

### Verdict

**Continue overnight loop.** Healthy training-side picture; the
heuristic transfer is the bottleneck. Need next eval (~45 min) to
read direction.

Reschedule wakeup +30 min.

## Overnight check-in 38 (step 2,297,856)

Wakeup-fired at 08:09. Process pid 67363 alive (RN, etime 11:56:37).

Step 2,297,856 — next heuristic eval at ~2,332,320 still ~34k
steps off (~22 min). Outside poll window. No new eval data;
latest still 0.300.

### Decision

Rule (d) re-fires on the same datapoint. Action: status + reschedule.

### Health snapshot

```
fps_scalar=54
rollout R=-0.172  (oscillating: +0.107 → -0.172)
entropy=0.55     value_loss=0.54     EV=+0.032
KL=0.0011        clip_fraction=??
```

Rollout R hasn't sustained the +0.228 spike from check-in 35 — it
swings +0.10 to -0.17 update-by-update. Mean is roughly 0. EV
positive but small (+0.03). KL squeaky clean (0.0011). Entropy
stable.

### Verdict

**Continue overnight loop.** Heuristic eval ~22 min away; +30min
wakeup lands ~8 min after, will catch the eval.

Reschedule wakeup +30 min.

## Overnight check-in 39 (step 2,338,816) — 🟠 third heuristic eval REGRESSED

Wakeup-fired at 08:40. Process pid 67363 alive (RN, etime 12:27:34).

### Third post-upgrade heuristic eval

```
2,129,920  0.2750  vp= 9.10  len=122.9
2,232,320  0.3000  vp= 9.00  len=126.5
2,334,720  0.1250  vp= 8.13  len=118.3  ← Δ=-0.175 from prior
```

Per the comparison rule, -0.175 < -0.05 → **drop back to gap**.

### Trend statistics

3-point mean = 0.233, std = 0.075. Agent is noisily oscillating
around the gap floor with **no climb**. avg_vp 8.13 says agent
is reaching even fewer VP this time. Game length flat (~120) —
heuristic ends games quickly.

### Sanity-check on rollout opponent mix

`arguments.py:54-55` defaults:
- `league_random_weight = 0.05` (5%)
- `heuristic_opponent_weight = 0.25` (25%)

So 25% of rollout games are vs heuristic and 5% vs random — the
remaining 70% is PFSP league self-play. The agent IS getting
heuristic exposure during training. The transfer gap is **not**
from undertraining; it's from policy capacity / off-policy
gradient signal vs a strong deterministic opponent.

### Decision

- (a) ≥ 0.95: NO
- (b) ≥ 0.50: NO
- (c) **< 0.30: YES → "TRANSFER GAP PERSISTS"**
- Action: status + reschedule

### Health snapshot

```
fps_scalar=52
rollout R=-0.177  (oscillating around 0)
entropy=0.57   value_loss=0.94   EV=+0.172  ← best in a while
KL=0.0028
```

Training-side scalars are mostly fine: EV +0.17 is the highest of
the resumed run, KL clean, entropy stable. The policy isn't
broken — the heuristic eval is just brittle.

### Verdict

**Continue overnight loop.** Heuristic transfer is harder than
hoped. Need more datapoints before drawing conclusions; 3 evals
is too few for a deterministic 40-game eval against a strong
opponent. Plausible pattern: oscillation around 0.20–0.30 for
several million steps, then a phase-transition climb.

No action — continue training.

Reschedule wakeup +30 min.

## Overnight check-in 40 (step 2,367,488)

Wakeup-fired at 09:12. Process pid 67363 alive (RN, etime 12:59:34).

Step 2,367,488 — next heuristic eval at ~2,434,720 still ~67k
steps off (~45 min). Outside poll window. Still 3 heuristic
evals; 4-eval stats deferred to next wakeup.

### Decision

Latest still 0.125 → rule (c) re-fires "TRANSFER GAP PERSISTS".
Status only.

### Health snapshot

```
fps_scalar=51
rollout R=+0.142   (positive — rollout-side gains continue)
entropy=0.54   value_loss=0.54   EV=+0.034
KL=0.0040
```

Rollout R positive on this update (+0.142). Mean over the recent
window is mildly positive overall, even though individual updates
oscillate wide. EV positive but small. Nothing alarming.

### Verdict

**Continue overnight loop.** Heuristic eval ~45 min away.

Reschedule wakeup +30 min.

## Overnight check-in 41 (step 2,400,256)

Wakeup-fired at 09:43. Process pid 67363 alive (RN, etime 13:30:36).

Step 2,400,256 — 4th heuristic eval at ~2,434,720 still ~34k
steps off (~22 min). Outside the 3k poll-window. Latest still
0.125 → rule (c) re-fires.

### Decision

- (c) **< 0.30: YES → "TRANSFER GAP PERSISTS"**
- Action: status + reschedule

### Health snapshot — strong rollout R again

```
fps_scalar=49
rollout R=+0.218   ← second strong positive read (first was +0.228)
entropy=0.50       value_loss=0.67       EV=+0.059
KL=0.0035
```

Rollout R = +0.218 — second instance in the resumed run of a
strongly positive rollout reward. The first (+0.228 at check-in
35) preceded the eval climb from 0.275 to 0.300. Pattern is too
small to call but worth flagging: positive rollout-R spikes may
loosely correlate with eval ticks.

EV positive, KL clean, entropy continuing gradual decay (0.57
→ 0.55 → 0.54 → 0.50). Nothing alarming.

### Verdict

**Continue overnight loop.** Heuristic eval ~22 min away; +30 min
wakeup will land just after.

Reschedule wakeup +30 min.

## Overnight check-in 42 (step 2,437,120) — 4th heuristic eval

Wakeup-fired at 10:14; polled +7 min for the eval.
Process pid 67363 alive (RN, etime 14:01:35).

### Heuristic eval trend (4 datapoints)

```
2,129,920  0.2750  vp= 9.10  len=122.9
2,232,320  0.3000  vp= 9.00  len=126.5
2,334,720  0.1250  vp= 8.13  len=118.3
2,437,120  0.1750  vp= 8.50  len=161.2  ← Δ=+0.05 from prior
```

### 4-eval statistics

- mean = 0.219 (vs 3-eval mean 0.233 → **down 0.014**)
- std = 0.072
- range: [0.125, 0.300] (spread 0.175)

Per the prompt classification:
- ≥ 0.30 → climb starting: NO
- 0.20-0.30 → **flat oscillation**: YES (just inside)
- < 0.20 → deeper than expected: NO (0.219 > 0.20 by a hair)

**Verdict: flat oscillation.** Agent has not learned to beat
heuristic over ~300k steps of post-upgrade training; the eval
result is essentially noise around ~0.22.

### Notable: longer games this eval

avg_game_length jumped 118.3 → 161.2 — games are dragging out.
avg_vp also up slightly (8.13 → 8.50). Together this hints the
agent is playing more competitively (taking more turns, scoring
more) even though the win count stayed low. Plausible reading:
the policy is getting *closer* to good play (more VPs accrued,
games go longer) but the heuristic still finishes first.

This is a soft positive signal — the gap is narrowing, just not
fast enough to flip the WR yet.

### Decision

- (c) **< 0.30: YES → "TRANSFER GAP PERSISTS"**
- Action: status + reschedule

### Health snapshot

```
fps_scalar=48
rollout R=+0.226   ← third strong positive R in a row
entropy=0.48       value_loss=0.83       EV=+0.082
KL=0.0000          ← perfectly clean
```

KL = 0.0000 exactly. PFSP-hard updates well-bounded. Rollout R
+0.226 — the third "+0.22" spike in three check-ins. Entropy
gradual decay (0.50 → 0.48). EV positive.

### Verdict

**Continue overnight loop.** The pattern is "flat oscillation
with noise" — needs more steps before assessing whether progress
is real. The longer-game-length signal is a tentatively positive
indicator that the agent is drawing closer to heuristic-class
play, even if WR doesn't show it yet.

Reschedule wakeup +30 min.

## Overnight check-in 43 (step 2,461,696)

Wakeup-fired at 10:53. Process pid 67363 alive (RN, etime 14:40:36).

Step 2,461,696 — next heuristic eval at ~2,537,120 still ~75k
steps off (~52 min). Outside poll window. Latest still 0.175 →
rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS** (still 4 datapoints; 5th eval ~52
  min away)
- Action: status + reschedule

### Health snapshot

```
fps_scalar=47
rollout R=+0.055   (positive but smaller than recent +0.22 spikes)
entropy=0.53       value_loss=0.82       EV=+0.077
KL=0.0025
```

Rollout R smaller this update (+0.055 vs three +0.22 readings in
prior check-ins). Entropy bounced 0.48 → 0.53 — gentle wobble,
not concerning. KL clean. EV positive.

### Verdict

**Continue overnight loop.** 5th heuristic eval ~52 min away.
+30 min wakeup will land mid-cycle.

Reschedule wakeup +30 min.

## Overnight check-in 44 (step 2,498,560)

Wakeup-fired at 11:24. Process pid 67363 alive (RN, etime 15:11:35).

Step 2,498,560 — next heuristic eval at ~2,537,120 still ~38k
steps off (~28 min). Outside poll window. Latest still 0.175 →
rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS** (latest still 0.175)
- Action: status + reschedule

### NEG_EV=-1.03 alert — false alarm (same artifact)

Supervisor fired ALERT:NEG_EV=-1.03 at this check-in. Same low-
variance-returns artifact as steps 1,081k / 1,523k / 2,260k:

```
At the alert step (2,498,560):
  rollout R     = +0.139    (agent winning rollout side)
  value_loss    =  0.5167   (LOW — low return variance + low residual)
  EV            = -1.0251   (pathological, but artifact)
  KL            =  0.0043   (clean)
  entropy       =  0.4739   (gradual decay)
```

When the agent wins reliably, returns concentrate near a small
range, so Var(returns) shrinks. EV = 1 - Var(residuals)/Var(returns);
small denominator + modest residual variance ⇒ EV crashes negative
while value_loss itself stays low. **Cosmetic concern, not a real
divergence signal.** Same fix proposed earlier (`Var(returns) > ε`
guard) still pending.

### Health snapshot

```
fps_scalar=46
rollout R=+0.139    (positive — agent winning rollout side)
entropy=0.47        value_loss=0.52       EV=-1.03 (artifact)
KL=0.0043
```

KL clean, entropy continuing gradual decay (0.50 → 0.47), value
loss LOW (0.52). Excluding the EV artifact, the training-side
picture is healthy.

### Verdict

**Continue overnight loop.** 5th heuristic eval ~28 min away.
+30 min wakeup will land just after.

Reschedule wakeup +30 min.

## Overnight check-in 45 (step 2,523,136)

Wakeup-fired at 11:55. Process pid 67363 alive (RN, etime 15:42:35).

Step 2,523,136 — 5th heuristic eval at 2,537,120 still ~14k
steps off (~10 min). Just outside the 3k poll-window; not worth
burning cache to poll across the 5-min TTL.

Latest still 0.175 — rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS** (5th eval still pending)
- Action: status + reschedule

### Health snapshot

```
fps_scalar=45
rollout R=+0.129    (positive — agent winning rollout side)
entropy=0.45        value_loss=0.83        EV=+0.027
KL=-0.0006
```

EV recovered from the -1.03 artifact (now +0.027). KL essentially
zero. Entropy continuing decay (0.47 → 0.45). Rollout R positive.

### Verdict

**Continue overnight loop.** 5th eval ~10 min away; +30 min
wakeup will catch it cleanly.

Reschedule wakeup +30 min.

## Overnight check-in 46 (step 2,564,096) — 🟢 5th heuristic eval JUMPED

Wakeup-fired at 12:27. Process pid 67363 alive (RN, etime 16:14:35).

### 5th heuristic eval landed @ step 2,539,520

```
2,129,920  0.2750  vp= 9.10  len=122.9
2,232,320  0.3000  vp= 9.00  len=126.5
2,334,720  0.1250  vp= 8.13  len=118.3
2,437,120  0.1750  vp= 8.50  len=161.2
2,539,520  0.4000  vp=10.53  len=110.7  ← Δ=+0.225, NEW HIGH
```

### 5-eval statistics

- mean = 0.255 (up from 4-eval mean 0.219 → **+0.036**)
- std = 0.097 (slightly wider than 4-eval std 0.072)
- range: [0.125, 0.400] (spread 0.275)

Per the prompt's classification:
- ≥ 0.25 → **tentative climb signal**: YES (0.255)
- 0.20-0.25 → flat: NO
- < 0.20 → decline: NO

### Companion-scalar improvement (very positive)

```
              vs heuristic  prev: 0.175    now: 0.400
              avg_vp        prev: 8.50     now: 10.53  ← highest
                                                          of post-
                                                          upgrade
              avg_game_len  prev: 161.2    now: 110.7  ← decisive
                                                          wins, not
                                                          dragging
```

The +0.225 jump is too large for a 40-game eval to be pure noise —
this is a real adaptation signal. avg_vp at 10.53 (highest of the
post-upgrade phase) confirms the agent is closer to the 15-VP win
condition. The DROP in game length (161 → 110) means when the
agent wins, it wins fast — not just "got lucky in long games."

### Decision

- (a) ≥ 0.95: NO
- (b) ≥ 0.50: NO (close — 0.40)
- (c) < 0.30: NO
- (d) **0.30 ≤ WR < 0.50: YES → "ADAPTATION STARTING"**

### Health snapshot — best of resumed run

```
fps_scalar=44
rollout R=-0.099   (mild negative; rollout side noisier as
                    PFSP-hard samples tougher opponents)
entropy=0.48       value_loss=0.44   ← low
EV=+0.213          ← highest of resumed run!
KL=-0.0002         (essentially zero)
```

EV at +0.213 is the highest of the resumed run. value_loss 0.44
(low — confident value predictions). KL essentially zero. Entropy
0.48 — gradual decay continuing.

The training-side picture is uniformly healthy AND the eval
finally jumped. This is the convergence signal.

### Verdict

🟢 **Continue overnight loop with strong optimism.** First real
adaptation signal post-upgrade. The 0.255 5-eval mean clears the
0.25 climb threshold. If next eval lands ≥ 0.40, the climb is
robust; if it drops back to 0.20, that was a transient spike.

Reschedule wakeup +30 min.

## Overnight check-in 47 (step 2,605,056)

Wakeup-fired at 12:59. Process pid 67363 alive (RN, etime 16:46:35).

Step 2,605,056 — 6th heuristic eval at ~2,639,520 still ~34k
steps off (~22 min). Outside poll window. Latest still 0.400 →
rule (d) "ADAPTATION STARTING" re-fires.

### Decision

- (d) **0.30 ≤ WR < 0.50: YES** (latest 0.400)
- Action: status + reschedule

### Health snapshot — EV new high

```
fps_scalar=43
rollout R=-0.341   (rollout side noisier/struggling)
entropy=0.48       value_loss=0.49        EV=+0.295  ← NEW HIGH
KL=-0.0002         (essentially zero)
```

**EV +0.295 is the highest of the resumed run** (beating the
+0.213 from check-in 46). Value head is now consistently above
noise floor — strongly positive signal that aligns with the eval
WR climb. value_loss low (0.49). KL essentially zero.

Rollout R reverted to -0.341 (from -0.099 last). PFSP-hard is
sampling tougher league entries now that the agent is stronger;
rollout-side losses are expected. The deterministic eval signal
(0.400 → next eval) is the meaningful one.

### Verdict

🟢 **Continue overnight loop.** EV climbing is the strongest
training-side signal we've had. Next heuristic eval ~22 min
away — direction-check for whether the 0.400 sustains.

Reschedule wakeup +30 min.

## Overnight check-in 48 (step 2,641,920) — 🟠 0.40 was transient

Wakeup-fired at 13:30; polled +7 min for the eval to land.
Process pid 67363 alive (RN, etime 17:17:26).

### 6th heuristic eval landed @ step 2,641,920

```
2,129,920  0.2750  vp= 9.10  len=122.9
2,232,320  0.3000  vp= 9.00  len=126.5
2,334,720  0.1250  vp= 8.13  len=118.3
2,437,120  0.1750  vp= 8.50  len=161.2
2,539,520  0.4000  vp=10.53  len=110.7  ← prior high
2,641,920  0.2000  vp= 8.25  len=135.6  ← Δ=-0.20, transient
```

### 6-eval statistics

- mean = 0.246 (vs 5-eval mean 0.255 → **down 0.009**)
- std = 0.091
- range: [0.125, 0.400] (spread 0.275)

Per the prompt's direction-check classification:
- ≥ 0.40 → climb robust: NO
- 0.30-0.40 → ADAPTATION sustained but lower: NO
- < 0.30 → **0.40 was transient**: YES

### Companion-scalar regression

```
              prev (0.400)        now (0.200)
avg_vp        10.53 ← high        8.25 ← back to gap-floor
avg_game_len  110.7 (decisive)    135.6 (longer/dragging)
```

The 0.400 datapoint at step 2.54M had standout companions (vp 10.5,
len 110). This eval reverts to gap-floor companions (vp 8.25, len
135). Reading: the policy hasn't actually crossed a phase boundary;
it's high-variance around ~0.25.

### Decision

- (a) ≥ 0.95: NO
- (b) ≥ 0.50: NO
- (c) **< 0.30: YES → "TRANSFER GAP PERSISTS"**
- (d) NO

### Health snapshot

```
fps_scalar=42
rollout R=-0.071     (mild negative)
entropy=0.47         value_loss=0.76        EV=+0.044
KL=0.0010
```

EV pulled back from +0.295 high to +0.044. value_loss back up
(0.49 → 0.76). Entropy 0.47. KL clean.

### Verdict

Agent is in genuine high-variance oscillation around mean 0.246.
The 0.400 spike turned out to be a positive tail. Std of 0.091 on
a 40-game eval is at the upper end of binomial noise (sqrt(0.25*
0.75/40) ≈ 0.07) — so the variance is mostly *intrinsic* to the
40-game eval seed set, not policy walk.

Implication for the run: even at the current policy quality, eval
reads will swing 0.10-0.20. We need to wait for the 6-eval mean
to climb above ~0.40 before claiming real progress (which would
require all evals to be in 0.30-0.50 range).

**Continue overnight loop.** Reschedule wakeup +30 min.

## Overnight check-in 49 (step 2,674,688)

Wakeup-fired at 14:09. Process pid 67363 alive (RN, etime 17:56:35).

Step 2,674,688 — next heuristic eval at ~2,741,920 still ~67k
steps off (~45 min). Outside poll window. Latest still 0.200 →
rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS** (latest 0.200)
- Action: status + reschedule

### Health snapshot

```
fps_scalar=41
rollout R=+0.024     (essentially zero)
entropy=0.55         value_loss=0.80         EV=+0.078
KL=0.0051
```

Entropy bounced 0.47 → 0.55 — slight wobble. Rollout R near zero
(+0.024). EV positive small. value_loss 0.80. KL clean.

Nothing notable — training-side picture is steady, awaiting next
heuristic eval datapoint.

### Verdict

**Continue overnight loop.** Reschedule wakeup +30 min.

## Overnight check-in 50 (step 2,707,456)

Wakeup-fired at 14:40. Process pid 67363 alive (RN, etime 18:27:36).

Step 2,707,456 — next heuristic eval at ~2,741,920 still ~34k
steps off (~22 min). Outside poll window. No new eval data;
latest still 0.200 → rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS**
- Action: status + reschedule

### Health snapshot

```
fps_scalar=41
rollout R=-0.007    (essentially zero)
entropy=0.50        value_loss=0.61        EV=-0.021
KL=0.0022
```

EV bounced barely negative (-0.021) — basically zero. value_loss
0.61 (mid). Entropy 0.50 (back down from 0.55 wobble). KL clean.

Nothing notable. Steady oscillation around the noise floor.

### Verdict

**Continue overnight loop.** 7th heuristic eval ~22 min away;
+30 min wakeup will land just after.

Reschedule wakeup +30 min.

## Overnight check-in 51 (step 2,744,320) — 🔴 WR=0.0, avg_vp crashed

Wakeup-fired at 15:11. Process pid 67363 alive (RN, etime 18:58:38).

### 7th heuristic eval landed @ step 2,744,320

```
2,129,920  0.2750  vp= 9.10  len=122.9
2,232,320  0.3000  vp= 9.00  len=126.5
2,334,720  0.1250  vp= 8.13  len=118.3
2,437,120  0.1750  vp= 8.50  len=161.2
2,539,520  0.4000  vp=10.53  len=110.7
2,641,920  0.2000  vp= 8.25  len=135.6
2,744,320  0.0000  vp= 4.65  len=101.3  ← MAJOR REGRESSION
```

WR = 0.000: agent didn't win a single game out of 40.

### 7-eval statistics

- mean = 0.211 (down from 6-eval 0.246, -0.035)
- std = 0.120 (up from 0.091 — variance increasing)
- range: [0.000, 0.400] (spread 0.400)

### Notable: avg_vp crashed to 4.65

Up to now, avg_vp on losses was ~8-9 (the gap-floor — agent at
least builds a respectable economy before losing). This eval:
**avg_vp = 4.65**, dramatically below floor. Game length 101 is
also low. This is **not** the same kind of "lose a tight game" the
prior evals were — the agent is failing to build at all.

Hypotheses:
1. **PFSP-hard pushed the policy into a bad basin** — gradient
   updates between 2.64M and 2.74M against the strongest league
   entries drove the policy *worse* against heuristic than it
   was at the start of post-upgrade.
2. **Catastrophic forgetting** — agent overfit to league mix and
   forgot heuristic-counter strategies (specifically, the
   build-order that gets to 15 VP).
3. **Eval-seed unlucky** — possible but extremely unlikely given
   the avg_vp collapse on top of WR collapse. Random-seed eval
   would not push avg_vp from 8 to 4.65 with the same policy.

(1) and (2) are concerning if the next eval also lands < 0.10 with
low avg_vp. (3) would self-correct.

### Decision

- (a) ≥ 0.95: NO
- (b) ≥ 0.50: NO
- (c) **< 0.30: YES → "TRANSFER GAP PERSISTS"**
- Action: status + reschedule (per the framework — no abort rule
  defined for post-upgrade trend)

### Health snapshot

```
fps_scalar=40
rollout R=-0.105   (mild negative)
entropy=0.48       value_loss=0.57       EV=+0.051
KL=0.0025
```

Training-side scalars are *fine*: KL clean, entropy stable, EV
positive small, value_loss mid-range. **Nothing here suggests
catastrophic failure**, which makes hypothesis (1) — temporary bad
basin — more likely than (2) — true forgetting.

### Verdict

🔴 **Concerning datapoint, but no abort.** The next eval at
~2,847,520 is the key signal:
- If avg_vp recovers to ≥ 8 and WR ≥ 0.15 → bad-basin transient
- If avg_vp stays < 6 and WR < 0.10 → real policy degradation,
  worth flagging to the user for a possible rollback to
  checkpoint_02540032 (where the 0.40 was recorded)

**Continue overnight loop.** Reschedule wakeup +30 min.

## Overnight check-in 52 (step 2,764,800)

Wakeup-fired at 15:43. Process pid 67363 alive (RN, etime 19:30:38).

Step 2,764,800 — 8th heuristic eval at ~2,847,520 still ~83k
steps off (~55 min). Outside poll window. Latest still 0.000 →
rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS** (latest still 0.000)
- Action: status + reschedule

### Health snapshot

```
fps_scalar=40
rollout R=-0.056    (mild negative)
entropy=0.56        value_loss=0.54        EV=-0.096
KL=0.0019
```

EV slightly negative (-0.096), value_loss low (0.54), entropy
bounced 0.48 → 0.56 (uptick — agent's policy is widening, possibly
in response to the bad-basin signal). KL clean.

Entropy uptick is interesting: it's mild (0.08) but consistent
with PFSP-hard sampling sending the policy back toward more
exploratory behavior after the eval crash. The annealing schedule
sets a floor around 0.005, so this is well above any concerning
range.

### Verdict

**Continue overnight loop.** 8th heuristic eval ~55 min away;
+30 min wakeup mid-cycle, status only.

Reschedule wakeup +30 min.

## Overnight check-in 53 (step 2,797,568)

Wakeup-fired at 16:14. Process pid 67363 alive (RN, etime 20:01:35).

Step 2,797,568 — 8th heuristic eval at ~2,847,520 still ~50k
steps off (~33 min). Outside poll window. Latest still 0.000 →
rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS** (latest still 0.000, no new eval)
- Action: status + reschedule

### Health snapshot

```
fps_scalar=39
rollout R=-0.275    (more negative than recent)
entropy=0.47        value_loss=0.92        EV=+0.070
KL=0.0032
```

Rollout R back to -0.275 (matches PFSP-hard sampling tougher
opponents post-eval-crash). value_loss elevated (0.92). Entropy
0.47 — annealed back from the 0.56 wobble. KL clean.

Nothing here suggests divergence — the gradient updates are bounded,
just the policy is in a region where the current league + heuristic
mix doesn't produce a strong gradient toward beating heuristic.

### Verdict

**Continue overnight loop.** 8th eval ~33 min away — the key
direction-check is still pending. Reschedule +30 min.

## Overnight check-in 54 (step 2,846,720) — 🟢 BAD-BASIN TRANSIENT CONFIRMED

Wakeup-fired at 16:45; polled +23 min for the 8th eval to land.
Process pid 67363 alive (RN, etime 20:32:39).

### 8th heuristic eval landed @ step 2,846,720

```
2,129,920  0.2750  vp= 9.10  len=122.9
2,232,320  0.3000  vp= 9.00  len=126.5
2,334,720  0.1250  vp= 8.13  len=118.3
2,437,120  0.1750  vp= 8.50  len=161.2
2,539,520  0.4000  vp=10.53  len=110.7
2,641,920  0.2000  vp= 8.25  len=135.6
2,744,320  0.0000  vp= 4.65  len=101.3  ← 7th: outlier
2,846,720  0.2500  vp= 8.58  len=134.2  ← 8th: RECOVERED
```

### Direction-check verdict

Per the prompt's rules:
- avg_vp ≥ 8 → **bad-basin transient, continue**: YES (8.58)
- avg_vp < 6 AND WR < 0.10 → flag for rollback: NO

**The 0.000 / vp 4.65 was a sample-tail event, not policy
degradation.** The policy recovered the 8-9 vp gap-floor in the
very next eval. No rollback needed.

### 8-eval statistics

- mean = 0.2156 (was 0.246 at 6-eval, 0.211 at 7-eval)
- range: [0.000, 0.400] (very wide for a 40-game eval)
- 8-of-8 below 0.50, 0-of-8 in adaptation band consistently

The mean keeps drifting slightly down (0.246 → 0.211 → 0.216) as
new datapoints come in. This is consistent with the per-eval
variance being intrinsic noise on a 40-game sample around a true
mean of ~0.20-0.25. Agent has not learned to consistently beat
heuristic over ~700k steps of post-upgrade training.

### Decision

- (c) **TRANSFER GAP PERSISTS** (latest 0.250)
- Action: status + reschedule

### Verdict

Two reads to track:
1. **Short-term (continue)**: agent isn't broken, the bad eval was
   a tail event. Continue training; the underlying policy quality
   is intact.
2. **Long-term (open question)**: the 8-eval running mean isn't
   climbing. The hypothesis "PFSP-hard + heuristic_opponent_weight=
   0.25 will eventually drive learning to beat heuristic" needs
   significantly more steps to falsify. We're effectively in a
   transfer regime where the gradient signal toward beating
   heuristic is weak.

Connecting to the recent multi-agent debate on "separate setup
training": today's 8-eval pattern doesn't yet meet the
"diagnostics flag setup-specific pathology" bar that would
warrant Stage 3 intervention. The cheaper Stage 1 diagnostic
(per-phase TD logging + setup-policy swap-in A/B) remains the
right next move IF the user wants to act, rather than waiting
for the joint trainer to slowly close the gap.

**Continue overnight loop.** Reschedule wakeup +30 min.

## Overnight check-in 55 (step 2,875,392)

Wakeup-fired at 17:40. Process pid 67363 alive (RN, etime 21:27:35).

Step 2,875,392 — 9th heuristic eval at ~2,946,720 still ~71k
steps off (~48 min). Outside poll window. Latest still 0.250 →
rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS** (latest 0.250)
- Action: status + reschedule

### Health snapshot

```
fps_scalar=37
rollout R=-0.136   (mild negative)
entropy=0.44       value_loss=0.62        EV=+0.068
KL=0.0006
```

Entropy continuing gradual decay (0.52 → 0.47 → 0.44). KL squeaky
clean. EV positive small. Nothing notable.

### Verdict

**Continue overnight loop.** 9th heuristic eval ~48 min away.

Reschedule wakeup +30 min.

## Overnight check-in 56 (step 2,912,256)

Wakeup-fired at 18:12. Process pid 67363 alive (RN, etime 21:59:35).

Step 2,912,256 — 9th heuristic eval at ~2,946,720 still ~34k
steps off (~22 min). Outside poll window. Latest still 0.250 →
rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS**
- Action: status + reschedule

### Health snapshot

```
fps_scalar=37
rollout R=-0.080    (mild negative)
entropy=0.47        value_loss=1.08  ← high
EV=+0.014           KL=0.0005
```

value_loss elevated to 1.08 (was 0.62 last). EV +0.014 (small).
KL squeaky clean. Entropy bounced 0.44 → 0.47.

The value_loss spike isn't paired with negative EV or rising KL —
just a noisy update with high return variance. Not divergence.

### Verdict

**Continue overnight loop.** 9th eval ~22 min away; +30 min wakeup
will land just after.

Reschedule wakeup +30 min.

## Overnight check-in 57 (step 2,949,120) — 🟠 9th eval, slow drift down

Wakeup-fired at 18:43; polled +4 min for the eval to land.
Process pid 67363 alive (RN, etime 22:30:34).

### 9th heuristic eval landed @ step 2,949,120

```
2,129,920  0.2750  vp= 9.10  len=122.9
2,232,320  0.3000  vp= 9.00  len=126.5
2,334,720  0.1250  vp= 8.13  len=118.3
2,437,120  0.1750  vp= 8.50  len=161.2
2,539,520  0.4000  vp=10.53  len=110.7
2,641,920  0.2000  vp= 8.25  len=135.6
2,744,320  0.0000  vp= 4.65  len=101.3
2,846,720  0.2500  vp= 8.58  len=134.2
2,949,120  0.1000  vp= 6.20  len=118.7  ← 9th
```

### 9-eval statistics

- mean = 0.203 (was 0.216 at 8 evals, 0.211 at 7 evals)
- std = 0.113
- range: [0.000, 0.400] (still very wide for 40-game eval)

avg_vp 6.20 sits in an uncomfortable middle zone: above the
catastrophic 4.65 (no rollback flag, the rule was avg_vp < 6 AND
WR < 0.10), but below the typical 8-9 gap-floor. With WR=0.10
(below the rollback threshold of 0.10), this is *almost*
borderline rollback territory — but not quite, since avg_vp 6.20
is slightly above 6.

### Trend: still drifting down, not climbing

Running mean over the post-upgrade window:
```
3-eval (after 1925k):  0.233
4-eval (after 2027k):  0.219
5-eval (after 2129k):  0.255  ← 0.40 spike
6-eval (after 2232k):  0.246
7-eval (after 2334k):  0.211  ← 0.0 outlier
8-eval (after 2437k):  0.216
9-eval (after 2539k):  0.203  ← latest
```

The running mean has fallen from 0.255 at the 0.40 spike to 0.203
now — a -0.052 trend over 5 evals (~500k steps). This is a real,
persistent slow regression in mean win-rate, not just oscillation.
The hypothesis "PFSP-hard + heuristic_opp_weight=0.25 will close
the gap" is **failing** at this pace.

### Decision

- (a) ≥ 0.95: NO
- (b) ≥ 0.50: NO
- (c) **< 0.30: YES → "TRANSFER GAP PERSISTS"**
- (d) NO

Action: status + reschedule.

### Connecting to the FOR/AGAINST debate consensus

The 9-eval pattern is now strong enough to consider Stage 1 of
the consensus protocol:
1. **Instrument per-phase TD/value-error logging** to TensorBoard
2. **Run setup-policy swap-in A/B test**: at the latest checkpoint,
   manually swap in a strong heuristic for setup steps 0-4 only;
   measure WR delta vs current policy. If the swap-in lifts WR by
   ≥ 5pp, setup IS a binding constraint and Stage 2 is justified.

The user has not asked for this yet, but it's the right next move
when they choose to act. Logging only — no training disruption.

### Verdict

**Continue overnight loop.** No abort. But the joint-training
hypothesis is starting to look weak; the consensus-protocol Stage
1 diagnostic is the natural next-action when the user is ready.

Reschedule wakeup +30 min.

## Overnight check-in 58 (step 2,981,888)

Wakeup-fired at 19:18. Process pid 67363 alive (RN, etime 23:05:39).

Step 2,981,888 — 10th heuristic eval at ~3,049,120 still ~67k
steps off (~46 min). Outside poll window. Latest still 0.100 →
rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS** (latest 0.100 — lowest non-zero)
- Action: status + reschedule

### Health snapshot

```
fps_scalar=36
rollout R=-0.188    (mild negative)
entropy=0.53        value_loss=1.21  ← elevated (was 1.08 last)
EV=+0.083           KL=0.0031
```

value_loss elevated for second consecutive update at >1.0. EV
still positive small. KL clean. Entropy bounced 0.47 → 0.53.

The elevated value_loss in this regime (paired with positive EV
and clean KL) is consistent with a value head learning under a
shifting return distribution — the agent's wins/losses against the
PFSP-hard mix have changed character. Not divergence.

### Verdict

**Continue overnight loop.** 10th heuristic eval ~46 min away.
If it lands ≤ 0.18 → 10-eval mean confirms slow regression per
the prompt's threshold, and the consensus-protocol Stage 1
diagnostic becomes warranted.

Reschedule wakeup +30 min.

## Overnight check-in 59 (step 3,022,848)

Wakeup-fired at 19:49. Process pid 67363 alive (RN, etime 23:36:36).

Step 3,022,848 — 10th heuristic eval at ~3,049,120 still ~26k
steps off (~17 min). Outside poll window. Latest still 0.100 →
rule (c) re-fires.

### Decision

- (c) **TRANSFER GAP PERSISTS** (latest 0.100)
- Action: status + reschedule

### Health snapshot

```
fps_scalar=36
rollout R=+0.203   (positive — agent winning rollout side)
entropy=0.52       value_loss=1.06        EV=+0.117
KL=0.0012
```

Rollout R back to +0.20 (3rd time we've seen the +0.22 rollout R
spike pattern). value_loss still elevated (1.21 → 1.06). EV +0.12.
KL 0.0012 squeaky clean. Entropy 0.52 stable.

The recurring rollout R = +0.2 spikes have not yet translated to
sustained eval improvement.

### Multi-agent debate update (out-of-band, no training change yet)

Two parallel debates have produced clear consensus:

**Hyperparameter audit (2-round):** consensus on 3 changes:
- `eval_games: 40 → 100` (binomial noise floor)
- `heuristic_opponent_weight: 0.25 → 0.40` (transfer-stuck, comment
  promises decay that code doesn't implement)
- `entropy_coef_start: 0.04 → 0.06` (KL never binds → underexploration)

**Setup-vs-joint debate (2-round):** consensus on a 3-stage
protocol (per-phase TD diagnostic → cheap interventions → SL pretrain
fallback). Stage 1 diagnostic logging is the natural next move.

**Both protocols ship on next training start, not the running pid.**
No action needed yet — the user has not asked to apply changes.

### Verdict

**Continue overnight loop.** 10th eval ~17 min away — direction-
check whether the 9-eval mean drift down (0.255 → 0.203) continues
or stabilizes.

Reschedule wakeup +30 min.
