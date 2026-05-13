# Part B: AlphaGo-style Setup Pretrain — Implementation Plan

**Status**: PENDING — execute only if Part A trend analysis returns
``unsatisfactory`` at step ≥ 18,700,000.

**Owner**: Re-read this plan when the Part A verdict comes back.

**Created**: 2026-05-11 (step ~13.7M), to be reviewed when results land
~step 18.7M (~5-7 days wall time at current ~25 FPS).

**Prerequisite**: Part A trend-analysis harness has shipped and is running.
See ``scripts/analyze_trend.py`` and ``src/catan_rl/eval/trend_analysis.py``.

---

## Why this plan exists

By step ~13.7M the agent's win rate vs the engine heuristic plateaued at
~0.30 for ~6-8 weeks of training. We landed a plateau-breaking bundle at
step 13,135,872 (heuristic_opp_weight=0.60, duo exploiter, symmetry_aug=1.0,
target_kl=0.008, per-head grad clip, plus the prior speed-pack). If the
bundle is judged "unsatisfactory" at step 18.7M, the next intervention is
**AlphaGo-style supervised pretrain of just the corner and edge action heads**,
trained on Monte-Carlo-evaluated setup positions, then surgically merged
back into the live policy, with optional KL-regularization to keep the
fine-tune from immediately undoing the pretrain.

The hypothesis: the agent reliably reaches avg_vp ≈ 10-11 but the heuristic
reaches 15 first because its settlement placements lead to better resource
income. Teaching the settlement-picking heads directly — with high-SNR
supervised labels from search/rollouts — gives the model a stronger setup
prior than 13M+ steps of joint PPO have managed.

## Hard constraints (carried over from the original plan)

1. **Checkpoint compatibility preserved.** ``CatanPPO.load()`` must
   continue to load the merged checkpoint. Only ``policy.action_heads.corner``
   and ``policy.action_heads.edge`` parameters are replaced; everything else
   (encoder, value tower, type/tile/resource heads, GRU, opp-id embedding,
   league, optimizer state, value normalizer, RNG state) is bit-identical.
2. **No disruption to a live training process during data collection.**
   The MC rollout collector runs as a separate process with ``nice`` /
   limited workers so the live trainer's FPS doesn't drop more than ~20%.
3. **Falsifiable success criterion.** After merging the pretrained
   weights, the joint run must lift its 10-eval rolling-mean WR-vs-heuristic
   by ≥0.05 within 2M further steps. Otherwise the pretrain is rejected
   and we revert to the pre-merge checkpoint.
4. **Revertable failure.** A snapshot of the pre-merge checkpoint is
   preserved before any merge happens.

## Phase-by-phase implementation

### Phase 3 — MC rollout data collection

**Goal**: produce a shard dataset of (setup-decision obs, taken action,
MC win rate proxy) tuples for ~10K positions × ~50 rollouts.

**Files (new)**:
- ``src/catan_rl/setup_phase/mc_rollout_collector.py`` — ``MCRolloutCollector``
  class. Spawns N worker processes; each loads ``CatanPPO.load()`` once,
  then in a loop:
  1. Resets env to setup phase. Samples a full snake-draft setup via the
     current policy with action sampling at T=1.0 (this is the "position").
  2. Captures the obs dict at each of the 4 setup decisions (settle1,
     road1, settle2, road2), plus the action taken at each.
  3. Clones the post-setup state (see Phase 3 risk #1 — engine ``copy()``).
  4. Runs ``n_rollouts_per_position`` independent rollouts from there.
     Each rollout plays to termination with the current policy as both
     players (or with the heuristic as opponent — configurable). Records
     the agent-side win/loss outcome.
  5. Writes a shard ``shard_<worker>.npz`` with columns:
     ``obs_*``, ``action_corner``, ``action_edge``, ``decision_index``
     (0-3), ``value`` (win rate proxy = mean of ``n`` rollouts).
- ``scripts/collect_setup_mc.py`` — CLI driver. Required args:
  ``--policy-ckpt``, ``--n-positions``, ``--n-rollouts-per-position``.
  Optional: ``--n-workers 4`` (cap at 4 of 10 cores on M1 Pro),
  ``--nice 10``, ``--output-dir data/setup_mc/<run_id>/``,
  ``--rollout-opponent {self|heuristic}``.
- ``tests/integration/setup_phase/test_mc_collector_smoke.py`` —
  end-to-end run with ``n_positions=10, n_rollouts_per_position=2,
  n_workers=2``. Asserts shard schema, no NaN, value ∈ [0, 1].

**Risks**:
- **HIGH**: ``CatanGame`` may lack a ``copy()`` method. ``CLAUDE.md`` Phase
  4 ISMCTS note says exactly this: "would require ``CatanGame.copy()``
  which doesn't exist yet." **Mitigation**: in a 10-position smoke test,
  measure how fast we can replay from ``(board_seed, action_sequence)``.
  If it's >50 ms per replay, add a ``copy()`` method as a SEPARATE PR
  before starting the full collection.
- **HIGH**: MC variance at 50 rollouts/position may be too noisy for
  learnable supervised targets. Stderr ≈ √(0.25/50) ≈ 0.07.
  **Mitigation**: calibration smoke at ``n_positions=100,
  n_rollouts_per_position=50``; if observed stderr > 0.1, bump to 100
  rollouts or switch from soft to hard (top-1 win-rate-weighted) targets.
- **MEDIUM**: CPU contention with a still-live trainer. **Mitigation**:
  ``nice 10`` + ``--n-workers 4`` cap. Monitor live FPS during the first
  10 min; abort if it drops >20%.

**Compute estimate**:
- 10,000 positions × 50 rollouts × ~150 main-game steps ≈ 75M game-steps.
- Single-thread game-step rate measured at ~5500 step/sec.
- 4 parallel workers: 75M / (4 × 5500) = ~3,400 sec ≈ **~1 hour**.
- Per-rollout policy forward pass (batch=1) is ~5 ms → adds to the cost.
- Realistic estimate: **3-6 hours wall time** for the full dataset.

**Decision gate after smoke**: if extrapolated wall time > 12 hours,
reduce to ``n_positions=5000`` per the user's original cost concern.

### Phase 4 — Supervised pretrain (corner + edge heads only)

**Goal**: train ONLY ``policy.action_heads.corner`` and
``policy.action_heads.edge`` against the MC dataset; freeze everything
else.

**Files (new)**:
- ``src/catan_rl/setup_phase/head_pretrainer.py`` — ``HeadPretrainer`` class.
  Loads the current best checkpoint via ``CatanPPO.load()``. Sets
  ``requires_grad=False`` on every parameter except the two target heads.
  Adds a checksum assertion: after one epoch, the frozen parameters'
  ``hash(p.detach().cpu().numpy().tobytes())`` is unchanged.

  Loss design (ship the simpler one first; escalate if it doesn't work):
  - **v1 (simple)**: cross-entropy on the actually-taken action, weighted
    by ``2 * mc_value - 1`` (centered advantage). Skip rows with fewer
    than 10 rollouts to avoid tiny-sample noise.
  - **v2 (if v1 plateaus)**: soft cross-entropy on a softmax-temperature-
    shaped target over all candidate (corner, edge) actions at each
    decision, computed by grouping rows by ``(board_seed, decision_index)``
    and pooling their MC values.

  Optimizer: AdamW, lr=1e-4, weight_decay=1e-5. Cosine schedule to 1e-6
  over ``n_epochs=10``. Batch size 256.

  Critical detail: the corner head uses FiLM conditioning
  (``action_head_film=True`` in phase4). For setup-phase decisions only
  settlements get placed, so the FiLM context = ``[1, 0]`` (settle, not
  city). The pretrainer must invoke the head with this context — verify
  with a shape/value check on a single batch before kicking off the full
  training loop.

  Output: ``checkpoints/setup_pretrain/<run_id>/pretrain_final.pt``,
  containing ``{'corner_state_dict': {...}, 'edge_state_dict': {...},
  'metadata': {'n_positions': ..., 'n_rollouts': ..., 'data_run_id': ...,
  'src_checkpoint': ..., 'epochs': ..., 'final_train_loss': ...,
  'final_val_loss': ...}}``.

  TB logs: writes under ``runs/setup_pretrain/<run_id>/`` (NOT
  ``runs/train/``, to keep the main TB tree clean).

- ``scripts/pretrain_setup_heads.py`` — CLI. Required:
  ``--data-dir``, ``--policy-ckpt``, ``--output-dir``. Optional:
  ``--epochs 10``, ``--batch-size 256``, ``--lr 1e-4``.

- ``tests/unit/setup_phase/test_head_pretrainer_freezes_encoder.py`` —
  asserts non-target parameter checksums are unchanged after one epoch.
  Uses a tiny synthetic dataset and a 1-step optimizer to keep the test
  fast (~5 seconds).

**Risks**:
- **MEDIUM**: FiLM context for corner head wrong → silently train the wrong
  pathway. **Mitigation**: explicit shape/value test, plus log the FiLM γ
  and β at decision_index=0 for inspection.
- **LOW**: data leakage between train and val splits. **Mitigation**: split
  by ``board_seed`` so no setup position appears in both.

**Compute estimate**: ~1-3 hours on the dataset above.

### Phase 5 — Surgical merge + KL-regularized resume

**Goal**: write a NEW checkpoint that's bit-identical to the current
best except the two pretrained heads, then resume training with an
optional KL-reg term against the pretrained snapshot.

**Files (new)**:
- ``scripts/merge_setup_heads.py`` — CLI:
  ``--main-ckpt --pretrain-ckpt --output [--allow-overwrite-different-name-ok]``.
  Logic:
  1. ``torch.load(main_ckpt)`` (eagerly; need optimizer state etc).
  2. Enumerate keys matching ``policy.action_heads.corner.*`` and
     ``policy.action_heads.edge.*`` in the main state_dict. **Don't
     hardcode** — introspect, fail loud if names changed.
  3. Pull the corresponding keys from pretrain_ckpt. Shape-equality assert
     on every key.
  4. Build the output dict: copy of main_ckpt with those keys replaced.
  5. **Verification pass**: load output via ``CatanPPO.load()``; run one
     env step end-to-end without error; for every key that should NOT have
     been changed, assert ``torch.equal(main_ckpt[k], output_ckpt[k])``.
  6. Write output to ``checkpoints/train/<input_name>.with_setup_pretrain.pt``.
     **Never overwrite the input**; always a new filename.
  7. Also write a side-by-side diff log to
     ``checkpoints/train/<input_name>.merge_diff.json`` listing keys
     changed + their old/new norms (for audit).

- ``configs/phase4_full_setup_pretrain.yaml`` — inherits ``phase4_full.yaml``;
  adds:
  ```yaml
  ppo:
    setup_kl_regularization: true
    setup_kl_weight_start: 0.05
    setup_kl_weight_end: 0.0
    setup_kl_decay_steps: 2_000_000
    setup_pretrain_snapshot_path: checkpoints/setup_pretrain/<run_id>/pretrain_final.pt
  ```

**Files (modified)**:
- ``src/catan_rl/algorithms/ppo/trainer.py`` — adds optional KL-reg path
  gated on ``config["setup_kl_regularization"]``:
  - At ``__init__``: if flag is on, load the pretrain snapshot as a frozen
    second policy (``requires_grad=False``, ``eval()``).
  - During PPO update: on transitions where ``info["is_setup_phase"]`` is
    true, compute soft KL between live ``(corner_logits, edge_logits)`` and
    frozen snapshot's. Add ``setup_kl_weight * mean_kl`` to total loss.
    Decay: ``setup_kl_weight = setup_kl_weight_start *
      max(0, 1 - (global_step - merge_step) / setup_kl_decay_steps)``.
  - Logs ``train/setup_kl``, ``train/setup_kl_weight``,
    ``train/setup_kl_n_states``.
- ``src/catan_rl/env/catan_env.py`` — add ``info["is_setup_phase"]: bool``
  to the info dict returned by ``step()`` and ``reset()``. **Don't change
  the obs dict** — that would force a checkpoint-compat migration.
- ``src/catan_rl/algorithms/ppo/arguments.py`` — add new config keys with
  defaults that match the OFF behavior (``setup_kl_regularization: False``,
  others as quoted above).

**Risks**:
- **HIGH**: silent checkpoint corruption. **Mitigation**: never overwrite
  the input; bit-equality verification pass on non-target keys before
  writing.
- **MEDIUM**: KL-reg code path adds a hot-loop cost. **Mitigation**: skip
  the KL computation when ``info["is_setup_phase"]`` is false for every
  transition in the batch (early exit). Gate every line behind the config
  flag so the default-off path is byte-identical to current.

**Files (tests)**:
- ``tests/unit/algorithms/test_merge_setup_heads.py`` — bit-identical
  preservation of non-target keys across merge. Run with a real
  ``checkpoint_07390040.pt`` and a synthetic pretrain checkpoint that has
  matching corner/edge shapes.
- ``tests/integration/algorithms/test_setup_kl_regularization.py`` — one-
  step PPO update with flag on vs off; assert (a) total loss differs only
  by the KL term, (b) gradients on the encoder are unchanged when no
  ``is_setup_phase`` transitions are in the batch.

### Phase 6 — A/B verdict + revert path

**Goal**: after 2M post-merge steps, decide whether the pretrain helped.

**Files (new)**:
- ``scripts/revert_setup_pretrain.py`` — CLI driver for the revert path:
  1. Stop the current training process (SIGTERM → wait → SIGKILL if needed).
  2. Resume from the pre-merge snapshot checkpoint using
     ``configs/phase4_full.yaml`` (no setup_kl flag).
  3. Append a "Setup pretrain rejected" entry to ``MEMORY.md`` documenting
     the observed lift (or lack thereof).

**Workflow** (no code change, just steps):
1. Bookmark the merge step as ``merge_step``.
2. Run training for 2M steps from ``merge_step``.
3. Run ``python scripts/analyze_trend.py --tb-dir runs/train/<new run> --window 10``.
4. Compare ``stats.mean_last5`` to the value in
   ``runs/trend_analysis/baseline_at_bundle_landing.json`` (or a freshly
   captured pre-merge baseline).
5. Decision:
   - ``Δ ≥ +0.05``: KEEP. Update ``CLAUDE.md`` with the success entry.
   - ``Δ < +0.05``: REVERT via ``scripts/revert_setup_pretrain.py``.

**Risks**:
- **MEDIUM**: stopping the post-merge run cleanly. The trainer handles
  ``KeyboardInterrupt`` (SIGINT) but ignores ``SIGTERM`` per earlier
  session observation. Use ``kill -INT``; if it hangs >5 min, escalate to
  ``kill -9`` (rollout state will be lost but the latest periodic
  checkpoint is safe). Always confirm the pre-merge snapshot exists before
  stopping anything.

## Final-deliverable success criteria

- [ ] If verdict at step 18.7M is satisfactory: protocol terminates;
      no Part B code runs.
- [ ] If verdict is unsatisfactory:
      - [ ] MC rollout collection completes without slowing live training
            more than 20%.
      - [ ] Pretrain produces a head-only checkpoint with non-trivial
            train/val loss reduction.
      - [ ] Merge produces an output checkpoint that passes the
            bit-equality verification on non-target keys.
      - [ ] 2M post-merge steps complete; A/B Δ ≥ +0.05 → KEEP, else
            REVERT.
- [ ] If kept: ``CLAUDE.md`` updated, run continues; if reverted:
      ``MEMORY.md`` updated with the negative result, run continues from
      the pre-merge snapshot.

## Files Touched (absolute paths, all NEW unless marked)

**New**:
- ``/Users/benjaminli/my_projects/catan_rl/src/catan_rl/setup_phase/mc_rollout_collector.py``
- ``/Users/benjaminli/my_projects/catan_rl/src/catan_rl/setup_phase/head_pretrainer.py``
- ``/Users/benjaminli/my_projects/catan_rl/scripts/collect_setup_mc.py``
- ``/Users/benjaminli/my_projects/catan_rl/scripts/pretrain_setup_heads.py``
- ``/Users/benjaminli/my_projects/catan_rl/scripts/merge_setup_heads.py``
- ``/Users/benjaminli/my_projects/catan_rl/scripts/revert_setup_pretrain.py``
- ``/Users/benjaminli/my_projects/catan_rl/configs/phase4_full_setup_pretrain.yaml``
- ``/Users/benjaminli/my_projects/catan_rl/tests/unit/setup_phase/test_head_pretrainer_freezes_encoder.py``
- ``/Users/benjaminli/my_projects/catan_rl/tests/unit/algorithms/test_merge_setup_heads.py``
- ``/Users/benjaminli/my_projects/catan_rl/tests/integration/setup_phase/test_mc_collector_smoke.py``
- ``/Users/benjaminli/my_projects/catan_rl/tests/integration/algorithms/test_setup_kl_regularization.py``

**Modified**:
- ``/Users/benjaminli/my_projects/catan_rl/src/catan_rl/algorithms/ppo/trainer.py`` — KL-reg loss term (additive, flag-gated default-off)
- ``/Users/benjaminli/my_projects/catan_rl/src/catan_rl/algorithms/ppo/arguments.py`` — new config flags (default-off)
- ``/Users/benjaminli/my_projects/catan_rl/src/catan_rl/env/catan_env.py`` — ``info["is_setup_phase"]`` flag (no obs schema change)

**Outcome doc** (per CLAUDE.md rule 14 — update existing only):
- ``/Users/benjaminli/my_projects/catan_rl/CLAUDE.md`` — final-outcome
  note after either keep or revert lands.
- ``/Users/benjaminli/my_projects/catan_rl/.claude/projects/-Users-benjaminli-my-projects-catan-rl/memory/MEMORY.md`` —
  same.

## Estimated effort

| Phase | LOC | Implementation hours | Wall-clock for compute |
|---|---|---|---|
| Phase 3 (MC collector) | ~450 | 8 | 3-6h (1-hour smoke first) |
| Phase 4 (head pretrainer) | ~300 | 5 | 1-3h |
| Phase 5 (merge + KL-reg) | ~250 | 4 | n/a |
| Phase 6 (verdict + revert) | ~80 | 1 | 2M steps ≈ 22h at 25 FPS |
| **Total** | **~1080** | **~18h** | **~30-35h compute** |

End-to-end: ~2-3 days wall time once Part A says go.

## Open questions to resolve at execution time

1. Does ``CatanGame`` have a fast ``copy()`` method by the time we get here,
   or do we need to add one? Check first.
2. Is the FiLM context-vector format for ``corner_head`` stable across
   phase3 → phase4? Re-read the code at execution time.
3. Should the MC rollouts use the live policy or the heuristic as the
   opponent for setup-position labeling? Heuristic is more deterministic;
   live policy is what we'll face in eval. Try the heuristic first
   (faster, lower variance).
4. Do we want to KL-regularize the merge for >2M steps or fewer? Start
   with 2M; ablate later if the post-merge plateau is at the new level.

## Cross-references

- Part A trend-analysis: ``src/catan_rl/eval/trend_analysis.py``,
  ``scripts/analyze_trend.py``.
- Baseline snapshot (captured 2026-05-11):
  ``runs/trend_analysis/baseline_at_bundle_landing.json``.
- Existing setup-phase scaffolding to potentially reuse:
  ``src/catan_rl/setup_phase/`` and ``scripts/train_setup.py``.
- Original superhuman roadmap (Phase 3 setup pretrain is the AlphaGo
  analog of what's being elaborated here):
  ``docs/plans/superhuman_roadmap.md``.
