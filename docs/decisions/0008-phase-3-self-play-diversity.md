# ADR 0008: Phase 3 Self-Play Diversity Upgrades

**Status:** Accepted (with one item deferred to a follow-up)
**Date:** 2026-05-02

## Context

Phase 2 fixed the architecture (axial pos-emb, GELU/dropout, AdaLN heads,
decoupled value tower). The next bottleneck is opponent distribution. The
legacy league sampled with linear-recency bias and a flat `w·(1-w)` PFSP —
both encourage the policy to keep beating slightly-weaker historical
versions of itself rather than learning to beat opponents that are
strategically distinct. AlphaStar showed that diverse populations are the
single biggest late-game lever.

This ADR records what shipped in Phase 3, what's deferred, and the design
choices that fell out.

## Decisions

### 3.1 PFSP-hard sampling

`League` accepts `pfsp_mode ∈ {linear, hard, var}` and a tunable exponent
`pfsp_p`. `'hard'` uses `(1-w)^p` (AlphaStar) so opponents we currently
lose to dominate the sampling distribution. Per-opponent win rates use a
sliding `pfsp_window=32` so the priority reacts to *current* form, not
career stats.

- **Why a window:** without it, an opponent we used to lose to but now
  dominate keeps drawing PFSP-hard mass forever (their cumulative WR
  remains low). The window forgets old outcomes and lets PFSP follow the
  policy's current weakness frontier.
- **Why ε > 0 priority floor:** `(1-w)^p = 0` when `w=1.0`. Without ε the
  numpy choice would crash (or starve a fully-dominated opponent of any
  resampling probability, so we never re-validate that we still dominate).

### 3.2 Latest-policy regularization

`League.sample()` returns the special `('current_self', None, -2)` tuple
with probability `latest_policy_weight`. The trainer passes a
`current_policy_state_fn` callable into `GameManager` so this opponent is
populated with the **live** policy snapshot — not a stale league entry.

- **Why a sentinel ID (-2):** keeps the existing `(-1 = random)` convention
  intact and ensures `update_result(-2, win)` is a no-op (we don't pollute
  PFSP win-rate tracking with self-vs-self matches).
- **Why "current self" instead of just "most recent league entry":**
  league entries are added every `add_every=4` updates; between adds the
  most recent entry is several updates stale, large enough for measurable
  policy drift on small-batch CPU runs.

### 3.3 Duo exploiter cycle

**Status (2026-05-02):** landed as a follow-up PR after the rest of Phase 3
merged. Config keys (`exploiter_mode`, `exploiter_cycle_steps`,
`exploiter_n_updates`, `exploiter_priority_multiplier`, `exploiter_priority_games`)
all wired through `arguments.MODEL_CONFIG`, the trainer, and YAML.

The implementation reuses the existing `CatanPPO` machinery rather than
spinning up a sub-trainer. Inside `_maybe_run_exploiter_cycle`:

1. `snap = self._snapshot_for_exploiter()` deep-copies policy weights,
   optimizer state, value-normalizer running stats, episode counters, and
   the league + GameManager wiring.
2. `self.policy.load_state_dict(snap['policy'])` — exploiter starts from
   *current main weights* (not from scratch); fresh `AdamW` so Adam
   moments are zeroed for the exploiter's gradient direction.
3. `self.league = League.frozen_main_for_exploiter(snap['policy'])` —
   single-policy league, no random/heuristic/current_self mass; the
   exploiter's only opponent is the frozen main snapshot.
4. `_train_exploiter_inner` runs `exploiter_n_updates` PPO updates with
   TB scalars under `exploiter/<key>`. `self.global_step` is rolled back
   after each rollout so main's step counter doesn't drift across cycles.
5. `self._restore_from_snapshot(snap)` (in a `finally:` block — even on
   error, main is restored).
6. `self.league.add_with_boost(exploiter_state, multiplier=…, boost_games=…)`
   injects the exploiter into the *restored* main league with an amplified
   PFSP priority for its first `exploiter_priority_games` matches.

Design choices and the alternatives we rejected:

- **Why no sub-trainer instance:** would require duplicating buffer / writer
  / GameManager / League — every piece of Phase-1-2-3 state. The
  in-place snapshot/restore pattern is ~50 LOC vs ~500 for a separate
  class, and round-trip coverage in `test_exploiter_snapshot.py` confirms
  no state leak across the swap.
- **Why fresh AdamW (instead of preserving main's optimizer):** the
  exploiter's gradient direction is by design adversarial to main's;
  inheriting main's Adam first moment would bias the early exploiter
  updates toward main's recent direction, defeating the point. Welford
  stats *are* preserved (they're just running estimates of value scale,
  not directional).
- **Why `_train_exploiter_inner` rolls back `self.global_step`:** main's
  step counter governs eval / checkpoint frequency. If it advanced during
  an exploiter cycle we'd evaluate the exploiter against the heuristic
  (wrong opponent) and save exploiter weights to main's checkpoint dir.
- **Why amplified PFSP priority instead of forced sampling:** AlphaStar's
  recipe boosts but doesn't pin the exploiter — boosting preserves PFSP's
  natural curriculum (the exploiter still drops out of the rotation once
  main learns to beat it).
- **Why disable `record_match_fn` and `current_policy_state_fn` during
  the cycle:** TrueSkill ratings are commensurable only across the *main*
  rating timeline; exploiter matches against frozen main shouldn't update
  the same μ/σ. And `current_self` would mean "exploiter plays current
  exploiter," which is a degenerate self-play mode incompatible with the
  cycle's goal of beating frozen main.

### 3.4 TrueSkill league rating

The `RatingTable` (built in Phase 0) is now wired into the trainer.
`GameManager._report_match` is the single hook for every match outcome,
calling both `League.update_result` (for PFSP) and the trainer's
`_record_rating_match` (for TrueSkill). Random/heuristic/current_self
opponents are skipped — they have no stable policy ID and would corrupt
the table.

- **σ-decay (default 1.001 per PPO update):** TrueSkill's natural σ
  shrinkage assumes stationary skill. Self-play is non-stationary, so we
  re-inflate σ per update; without this the main's σ collapses to ~1.0
  within a few hundred updates and the rating system becomes overconfident.

### 3.5 Nash-weighted checkpoint pruning

When league hits capacity and `prune_strategy='nash'`, the trainer kicks
off an internal round-robin among the most recent `nash_top_k` entries
(via `EvaluationManager.evaluate_h2h`), builds a (k, k) win-rate matrix,
centers it to antisymmetric zero-sum form, runs 100 multiplicative-weights
replicator iterations from a uniform initialization, and evicts the entry
with the lowest mixture mass.

- **Why MW replicator (not LP):** the math is simple, deterministic, and
  works fine for the well-conditioned 2-player zero-sum case. LP would
  require scipy; MW is one numpy expression.
- **Why "most recent K" instead of a global league-wide round-robin:** a
  league-wide N×N round-robin at maxlen=100 with 50 games per pair is
  ~250k games per prune, repeated every 20 adds. Capping at top-K=32
  bounds the cost to ~25k games and is "good enough" because old entries
  the league has been keeping are likely already strategically distinct
  (they survived prior prune rounds).
- **Why every 20 league adds instead of every add:** the round-robin is
  the most expensive thing in the entire trainer; running it less often
  makes the wall-clock cost tractable on CPU.

### 3.6 Opponent ID embedding

`ObservationModule` optionally adds two embeddings (`opp_kind ×
n_opp_kinds`, `opp_policy_id × (league_maxlen + 1)`) which are split in
halves, concatenated to make a single `opp_id_emb_dim` vector, and added
to the fusion input. Env emits the corresponding two int scalars via
`_opponent_id_obs`, which applies random masking with prob
`opp_id_mask_prob=0.40` to set both to UNKNOWN — so the policy learns to
play well even when opponent identity is not revealed.

- **Why a separate dict key (not extending `current_player_main`):**
  cleaner versioning (Phase 2 checkpoints stay loadable since they don't
  see the new keys) and avoids reshuffling the existing 54/61-dim layout.
- **Why higher mask prob (0.40) than the typical 0.25:** at eval time
  (champion-bench, exploitability) we deliberately use the UNKNOWN
  opponent kind so eval is a clean estimate of skill against an unseen
  opponent. Pre-training-time masking matches that distribution and
  prevents the policy from over-exploiting opponent identity.
- **Why modulo `league_maxlen` for the policy_id:** the league's
  monotonic ID is unbounded across a long training run, but the embedding
  has a fixed-size lookup. Modulo folds them into a stable bounded slot.

## Consequences

- Param count: ~2.22M (phase2_full) → ~2.24M (phase3_full). The only
  delta is the small embedding pair.
- FPS impact: Nash pruning adds bursts of round-robin h2h every ~80 PPO
  updates. With `phase3_full`'s tightened defaults (top-K=16, 16 games/pair)
  the burst is bounded to ~120 games per prune. PFSP/ratings/embeddings
  are all O(1) per-update.
- Phase 3 lineage stays incompatible with `checkpoint_07390040.pt` (Phase
  1 already broke it; the new opponent-id embedding makes it doubly so).
- All five landed sub-features have config-driven leave-one-outs for
  ablation.

## Alternatives considered

- **PFSP-var (`w * (1-w)^p`):** included as a sampling mode for
  completeness but not the default. Rarely useful at p=2.
- **Synchronous exploiter training (single trainer instance toggling
  between main and exploiter):** prone to optimizer-state corruption when
  switching, and the wall-clock cost is identical to a sub-trainer
  approach. Sub-trainer scoped for the follow-up PR.
- **Full league-wide round-robin (no top-K):** quadratic in maxlen,
  prohibitive on CPU.

## Follow-up work

- **Champion-bench integration of TrueSkill:** wire `use_trueskill=True`
  output into the eval-harness JSON so league ratings can be tracked across
  runs.
- **Exploiter eval harness:** the cycle currently has no built-in
  measurement of "did this exploiter actually beat frozen main." Add a
  short H2H eval at end-of-cycle and log
  `exploiter/cycle_<n>_win_rate_vs_main`.
