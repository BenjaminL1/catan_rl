#!/usr/bin/env python3
"""Throwaway diagnostic: per-phase wall-clock of the PPO update loop.

Times the *exact* operations run by
``catan_rl.ppo.training_loop.run_training_loop`` — rollout collect, GAE,
SGD update, and one eval round — for whichever engine backend
``CATAN_ENGINE_BACKEND`` selects (rust|py). Forces device=cpu to match
the production run (resolve_device would otherwise pick mps on M1).

rollout + SGD timings are policy-independent (a rollout always collects
exactly n_envs*n_steps engine steps and SGD always runs the same fixed
number of minibatch steps), so a fresh-init policy measures them
faithfully. Only eval is policy-dependent (games run to completion), so
its number is an upper bound here (random policy => longer games).

Env knobs:
  CATAN_ENGINE_BACKEND  rust|py   (default: python, reverted from "rust"
                                   on 2026-06-06 — no production code
                                   under src/catan_rl/ dispatches on
                                   this selector yet; see
                                   docs/plans/rust_engine_actual_state.md.
                                   This script reads it because it
                                   declares operator intent, not because
                                   it changes the engine the training
                                   loop actually constructs.)
  DIAG_N_ENVS           int       (override n_envs; for a fast smoke)
  DIAG_N_WARMUP         int=1
  DIAG_N_TIMED          int=4
  DIAG_EVAL             0|1=1
  DIAG_EVAL_GAMES       int=20    (override eval_games for timing)
  DIAG_DEVICE           str=cpu
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

from catan_rl.engine.backend import current_backend
from catan_rl.ppo.arguments import TrainConfig
from catan_rl.ppo.training_loop import build_training_state

logging.getLogger("catan_rl.train").setLevel(logging.ERROR)

CFG_PATH = "configs/ppo_default.yaml"
N_WARMUP = int(os.environ.get("DIAG_N_WARMUP", "1"))
N_TIMED = int(os.environ.get("DIAG_N_TIMED", "4"))
DO_EVAL = os.environ.get("DIAG_EVAL", "1") == "1"
EVAL_GAMES = int(os.environ.get("DIAG_EVAL_GAMES", "20"))
DEVICE = os.environ.get("DIAG_DEVICE", "cpu")
N_ENVS_OVR = os.environ.get("DIAG_N_ENVS")

cfg = TrainConfig.load(yaml_path=CFG_PATH)
cfg = replace(cfg, eval=replace(cfg.eval, eval_games=EVAL_GAMES))
if N_ENVS_OVR is not None:
    cfg = replace(cfg, rollout=replace(cfg.rollout, n_envs=int(N_ENVS_OVR)))

n_envs = cfg.rollout.n_envs
n_steps = cfg.rollout.n_steps
transitions = n_envs * n_steps
n_mb = transitions // cfg.ppo.batch_size

print("=== phase-timing diagnostic ===")
print(
    f"backend={current_backend()} device={DEVICE} n_envs={n_envs} "
    f"n_steps={n_steps} transitions/rollout={transitions}"
)
print(
    f"warmup={N_WARMUP} timed={N_TIMED} eval={'on' if DO_EVAL else 'off'} "
    f"eval_games={EVAL_GAMES} n_epochs={cfg.ppo.n_epochs} batch={cfg.ppo.batch_size} "
    f"(=> {n_mb} mb/epoch, {n_mb * cfg.ppo.n_epochs} sgd steps)"
)

run_dir = Path("runs/_diag_tmp")
state = build_training_state(cfg, run_dir=run_dir, device_label=DEVICE)
print(f"policy params: {state.policy.num_parameters():,}\n")

base = (cfg.seed * 1_000_003) & 0x7FFFFFFF
seeds = [(base + i) & 0x7FFFFFFF for i in range(n_envs)]
obs, masks = state.vec_env.reset_all(seeds=seeds)

roll: list[float] = []
gae: list[float] = []
sgd: list[float] = []
for i in range(N_WARMUP + N_TIMED):
    timed = i >= N_WARMUP
    t0 = time.perf_counter()
    obs, masks = state.collector.collect(obs, masks)
    t1 = time.perf_counter()
    state.buffer.compute_returns_and_advantages(
        last_values=state.collector.last_values,
        gamma=cfg.gae.gamma,
        gae_lambda=cfg.gae.gae_lambda,
        advantage_norm=cfg.ppo.advantage_norm,
    )
    t2 = time.perf_counter()
    state.trainer.update(state.buffer, update_idx=i, rng=state.rng)
    t3 = time.perf_counter()
    tag = "timed " if timed else "warmup"
    print(
        f"  [{tag}] upd {i}: rollout={t1 - t0:6.1f}s  gae={t2 - t1:5.2f}s  "
        f"sgd={t3 - t2:6.1f}s  total={t3 - t0:6.1f}s"
    )
    if timed:
        roll.append(t1 - t0)
        gae.append(t2 - t1)
        sgd.append(t3 - t2)

eval_t = None
if DO_EVAL:
    te = time.perf_counter()
    state.eval_harness.run(state.policy)
    eval_t = time.perf_counter() - te
    print(f"  eval: {eval_t:6.1f}s for {EVAL_GAMES} games x {len(cfg.eval.eval_opponents)} opp")

state.vec_env.close()

med = statistics.median
mr, mg, ms = med(roll), med(gae), med(sgd)
per_update = mr + mg + ms
print("\n=== medians (s) ===")
print(f"  rollout (engine + policy/opp inference): {mr:7.2f}")
print(f"  gae:                                     {mg:7.3f}")
print(f"  sgd ({cfg.ppo.n_epochs} epochs x {n_mb} mb):                {ms:7.2f}")
print(f"  per-update (no eval):                    {per_update:7.2f}")
print(f"  -> rollout share {100 * mr / per_update:.0f}%   sgd share {100 * ms / per_update:.0f}%")

# Full-run extrapolation at PRODUCTION shape (128 x 256), valid only when
# run without DIAG_N_ENVS override.
PROD_TRANS = 128 * 256
n_upd = cfg.total_steps // PROD_TRANS
eval_every = cfg.eval.eval_every_updates
n_eval = (n_upd // eval_every) if eval_every else 0
prod_eval_games = 200
eval_per_round = (eval_t / EVAL_GAMES * prod_eval_games) if eval_t else 0.0
train_h = per_update * n_upd / 3600.0
eval_h = eval_per_round * n_eval / 3600.0
print(f"\n=== full-run extrapolation (n_envs=128, {n_upd} updates) ===")
note = "" if N_ENVS_OVR is None else "  [INVALID: n_envs overridden — smoke only]"
print(f"  train: {per_update:.1f}s x {n_upd} = {train_h:.1f}h{note}")
if eval_t:
    print(
        f"  eval:  ~{eval_per_round:.0f}s/round (scaled {EVAL_GAMES}->{prod_eval_games} games) "
        f"x {n_eval} rounds = {eval_h:.1f}h  [fresh-policy upper bound]"
    )
print(f"  TOTAL: {train_h + eval_h:.1f}h  ({(train_h + eval_h) / 24:.2f} days){note}")
