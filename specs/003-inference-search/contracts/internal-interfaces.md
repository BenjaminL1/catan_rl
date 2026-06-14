# Phase 1 Contracts: Inference-Time Search (internal interfaces)

No external/network API — this is an internal library + CLI feature. Contracts are the module's Python surfaces. All new code in `src/catan_rl/search/`; nothing in the training path changes.

## C1 — Value squash (leaf evaluator)

```text
search/value.py
  squash_value(v: float | Tensor, a=3.22, b=-1.14) -> float | Tensor   # sigmoid(a*v+b) in (0,1)
  leaf_value(policy, env, *, perspective_seat) -> float                 # forward -> squash; sign from perspective
```
**Contract**: returns a bounded win-probability in (0,1) from the to-move seat's perspective. NEVER returns raw V. For a terminal state, the caller uses the true outcome (1/0), not `leaf_value`.

## C2 — Priors over the autoregressive action space

```text
search/priors.py
  action_priors(policy, env) -> dict[action_tuple, float]   # legal actions only, normalized
```
**Contract**: keys are legal 6-tuple actions consistent with `env.get_action_masks()`; probabilities sum to 1 over the expanded set; uses the type-head distribution × conditional sub-head priors (D3). MUST agree with the mask (no illegal action ever has nonzero prior).

## C3 — SearchAgent (the decision surface)

```text
search/agent.py
  class SearchAgent:
      def __init__(self, policy, cfg: SearchConfig): ...
      def choose_action(self, env: CatanEnv) -> np.ndarray   # legal 6-tuple
      last_diagnostics: dict                                  # value est, root visits, budget used
```
**Contract**: deterministic given `(cfg.seed, budget, env state)` (FR-006); returned action is always legal (FR-007); on a forced move (1 legal action) short-circuits without spending budget; anytime — respects `time_budget_s` and returns best-so-far. MUST NOT mutate the passed `env` (clones internally).

## C4 — Search-aware eval loop (the env-access bake-off harness)

```text
search/eval_search.py
  evaluate_search_vs_policy(
      search_cfg: SearchConfig, search_ckpt: str, opponent_ckpt: str,
      *, n_games: int, seed: int, device="cpu", max_turns=400,
  ) -> SearchEvalResult                                       # wr + WilsonInterval, seat-symmetrized
```
**Contract**: mirrors `evaluate_policy_vs_policy` semantics (seat-symmetrized, Wilson CI, CPU-pinned, RNG saved/restored) BUT drives a `SearchAgent.choose_action(env)` for the agent seat (it owns the live env), with the opponent seat a `FrozenSnapshotOpponent(opponent_ckpt)`. The engine + rules are untouched; only legal actions are played. Reproducible at a fixed seed.

## C5 — CLI entry (offline use)

```text
cli/search_eval.py  (console script catan-rl-search-eval)
  --ckpt PATH --opponent {policy:PATH|heuristic|random} --sims N | --time-budget S
  --n-games N --seed N --determinizations N --out PATH
```
**Contract**: runs C4, prints WR + Wilson CI (+ optional Elo delta), writes JSON. Off the training path; no GUI import; CPU by default.

## C6 — Bake-off gate (acceptance harness)

```text
search/bakeoff.py  (or a quickstart script)
  run_bakeoff(ckpt) -> {passed: bool, wr, ci, ladder: {0.25s,1s,5s: wr}}
```
**Contract** (encodes SC-001/002): minimal search vs the raw `ckpt`; PASS iff Wilson **lower bound > 0.50** at n≥200 then re-confirmed at n≥500; ALSO runs the 0.25/1/5s time-budget ladder and reports whether WR is monotone in budget. On FAIL, returns the failure mode for the documented pivot.

## Invariant contracts (all surfaces)

- **Additive/inert**: importing/using `search/` does not change any existing module's behavior; `tests/` for existing eval/training stay byte-identical (FR-009, SC-005).
- **Legality**: every action emitted by C1–C5 is in the env's current legal-action mask (FR-007, SC-006).
- **No checkpoint/obs/action change**: loads existing v2-lineage checkpoints via the existing `build_actor`; no state-dict migration (FR-010).
- **Determinism**: C3/C4 reproducible under fixed seed+budget (FR-006, SC-003).
