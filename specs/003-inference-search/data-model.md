# Phase 1 Data Model: Inference-Time Search

Entities are in-memory only (no persistence; offline inference). All live under the new `src/catan_rl/search/` module unless noted.

## SearchConfig

Immutable config for a search run. Source of truth = a dataclass (mirrors `arguments.py` conventions; **not** added to `TrainConfig` — search is isolated).

| Field | Type | Default | Notes |
|---|---|---|---|
| `sims_per_move` | int \| None | 100 | budget; mutually exclusive with `time_budget_s` |
| `time_budget_s` | float \| None | None | anytime budget; mutually exclusive with `sims_per_move`. NOT bit-reproducible (sim count varies with load) — only `sims_per_move` mode satisfies FR-006 |
| `n_determinizations` | int | 1 | independent trees aggregated per move; start 1, tune via variance probe |
| `c_puct` | float | 1.5 | PUCT exploration constant (tuned for the squashed [0,1] value) |
| `value_squash_a` | float | 3.22 | leaf squash `sigmoid(a·V + b)` (Platt fit, D4) |
| `value_squash_b` | float | -1.14 | " |
| `progressive_widening` | bool | False | OFF by default — the type head's 2-6 branching doesn't need taming, and the US1 gate (WR 0.578) measured the all-types path; a tunable for ablations |
| `pw_c`, `pw_alpha` | float | 2.0, 0.5 | when widening is on, expose top-`max(2, ⌈c·N^alpha⌉)` types by prior |
| `max_depth` | int \| None | None | optional depth cut (leaf-eval beyond it) |
| `seed` | int | 0 | reproducibility |

**Validation**: exactly one of `sims_per_move`/`time_budget_s` set; `c_puct>0`; `value_squash_a>0`; `n_determinizations≥1`; `pw_c>0`; `0<pw_alpha≤1`; `max_depth≥1` or None.

## SearchNode

A node in the search tree = a clonable game state at a decision point.

| Field | Type | Notes |
|---|---|---|
| `env` | CatanEnv (cloned) | the live state; `deepcopy`'d on expansion |
| `legal_types` | list[int] | legal action-types from the mask (the branching set) |
| `priors` | dict[action → float] | from the policy heads (type prior × conditional sub-priors) |
| `children` | dict[action → SearchNode] | expanded children (progressive widening) |
| `N`, `W` | dict[action → int/float] | visit counts, summed squashed-value (PUCT stats) |
| `is_terminal`, `outcome` | bool, float \| None | true game outcome used instead of leaf eval |

**State transition**: select (PUCT) → expand (one new child via PW on type, sub-action from priors) → evaluate (squashed leaf value, or true outcome if terminal) → backup (propagate value up the path; sign flips per to-move seat — see contracts).

## SearchAgent

Wraps a frozen policy + a SearchConfig; the decision surface.

| Member | Type | Notes |
|---|---|---|
| `policy` | CatanPolicy (frozen, CPU) | priors + value head |
| `cfg` | SearchConfig | |
| `choose_action(env) -> np.ndarray` | method | runs search from the **live env**, returns a legal 6-tuple action |
| `last_diagnostics` | dict | chosen action's value est, root visit distribution, budget used (FR-011) |

Note: takes the **env**, not just `(obs, masks)` — the one interface deviation from the raw policy (D6).

## Determinization

Not a stored object — a *procedure*: `deepcopy(node.env)` yields an env whose `StackedDice` bag (and dev-deck) fixes one concrete future. Multiple clones = multiple determinizations; values averaged.

## BakeoffResult / SearchEvalResult

Output of the search-aware eval loop (reuses `eval/wilson.py` + the Elo fitter).

| Field | Type | Notes |
|---|---|---|
| `wr`, `ci` | float, WilsonInterval | search agent's WR vs the opponent (seat-symmetrized) |
| `n`, `n_truncated` | int | game counts |
| `budget` | SearchConfig | the budget used (for the ladder rows) |
| `elo_delta` | float \| None | vs the raw policy on the ladder (US3) |

## Relationships

`SearchAgent` owns a `SearchConfig` and a frozen `policy`; per move it builds a transient tree of `SearchNode`s (rooted at a clone of the live `CatanEnv`), each expansion cloning the env (a Determinization). The search-aware eval loop drives `SearchAgent.choose_action(env)` against a `FrozenSnapshotOpponent` (the raw policy) and aggregates `BakeoffResult`s; results feed the existing Elo ladder.
