<!--
Sync Impact Report
- Version change: 1.0.0 → 1.1.0 (2026-06-08): Development Workflow + Principle IV
  amended for a SOLO project — no pull requests; commit/push directly to
  origin/main; CI on push is a safety net, not a merge gate.
- Version change: (none) → 1.0.0  (initial ratification)
- Modified principles: n/a (first ratification)
- Added sections: Core Principles (I–V), Additional Constraints, Development
  Workflow, Governance
- Removed sections: none
- Templates reviewed (no structural change required — generic "Constitution
  Check" gates remain compatible):
  - .specify/templates/plan-template.md   ✅
  - .specify/templates/spec-template.md    ✅
  - .specify/templates/tasks-template.md   ✅ (test-first categorization aligns
    with Principle IV)
- Runtime guidance file: CLAUDE.md (referenced in Governance)
- Deferred TODOs: none
-->

# Catan RL (1v1 Superhuman Agent) Constitution

## Core Principles

### I. The 1v1 Ruleset Is Sacred (NON-NEGOTIABLE)

The agent targets 1v1 Settlers of Catan under the Colonist.io ruleset and MUST
NOT be generalized to 4-player. These invariants MUST hold in every change:
15 VP win condition; exactly 2 players; player-to-player trading DISABLED
(bank/port only); 9-card discard threshold on a 7; Friendly Robber (no robber
on a hex adjacent to a player with `< 3` visible VP); StackedDice (shuffled bag
of 36 outcomes + 1 noise swap + 20% Karma forced-7). Any PR touching game-rule
constants, the action space, the observation schema, or the trading API MUST
explicitly state how it preserves this ruleset, or be rejected.

**Rationale**: Core design choices — perfect hand-tracking, single-opponent
obs, the 13-type action space — only hold under 1v1; silent 4-player
assumptions break correctness and evaluation comparability.

### II. Engine Integrity

The game engine MUST match Colonist.io exactly. No change may alter game rules
without explicit, surfaced justification in the PR.

**Rationale**: Engine drift silently invalidates all historical evaluation and
league comparisons.

### III. Backward-Compatible, Additive Artifacts

Policy state-dict shape changes MUST ship a one-shot migration script with a
documented checkpoint lineage, and SHOULD prefer keeping existing checkpoints
loadable. TensorBoard scalar names are append-only — new diagnostics are new
scalars, never renames.

**Rationale**: In-flight and archived checkpoints, and the dashboards reading
them, are long-lived assets; breaking them discards real compute and history.

### IV. Test-First & Green CI (NON-NEGOTIABLE)

Every behavioral change MUST ship with tests, written before or alongside the
implementation. CI runs on push to main — ruff, mypy (strict), and pytest on
Python 3.11+ (GUI pixel-rendering tests skip off-darwin) — and MUST be kept
green. When checking CI status, verify the per-check conclusions explicitly; a
passing watch exit code is NOT sufficient evidence of green CI.

**Rationale**: RL correctness bugs are subtle and silent; the test + CI gate is
the only reliable signal, and a masked exit code has already caused a bad merge.

### V. Self-Play Is 2-Player Zero-Sum

All self-play machinery — PFSP, league rating, Nash pruning, exploitability, and
the perfect 1v1 hand-tracker — is defined ONLY for the symmetric 2-player
zero-sum game and MUST NOT assume more players or hidden-trade dynamics.

**Rationale**: These methods' guarantees and the hand-tracker's observability
assumption break outside 1v1.

## Additional Constraints

- `arguments.py` is the single source of truth for hyperparameters; README and
  MEMORY may lag and MUST be verified against it before being quoted.
- Device policy: training resolves auto→MPS on Apple Silicon (batched SGD is
  ~3× faster at batch 512); evaluation is pinned to CPU (batch=1 inference is
  faster there); CUDA is opt-in.
- Observation and action-head shapes are stable by default: a change that
  resizes them MUST be justified against Principle III and account for every
  in-flight and archived checkpoint.
- Two resource orderings exist (engine `RESOURCES` vs RL `RESOURCES_CW`); code
  MUST import the correct one.
- Long-running training jobs MUST be launched detached from the editor session
  (e.g. `nohup`) so a session restart cannot kill them; rely on periodic
  checkpoints for crash recovery.

## Development Workflow

- Commits follow Conventional Commits: lowercase, type-prefixed, under 72 chars.
- **Solo project — NO pull requests.** Commit and push directly to
  `origin/main`. Short-lived *local* branches are allowed for keeping
  risky/in-progress work off main until green, then merge to main and push.
- Commits MUST NOT include `Co-Authored-By` trailers referencing any AI account.
- CI runs on push to main (ruff + mypy + pytest) as a safety net — keep it
  green, but it is not a merge gate. A change touching game rules, the action
  space, the obs schema, or trading MUST still state how it preserves the 1v1
  ruleset in its commit message.
- Documentation describing touched code (README, `docs/`, `CLAUDE.md`,
  migrations) MUST be updated in the same commit/push.

## Governance

This constitution supersedes ad-hoc practice. Amendments MUST be versioned
(semantic versioning: MAJOR for principle removals/redefinitions, MINOR for
added/expanded principles, PATCH for clarifications), dated, and — where they
change shapes or rules — accompanied by a migration plan. All PRs and reviews
MUST verify compliance with these principles; added complexity MUST be
justified. `CLAUDE.md` provides runtime, project-specific guidance that
operationalizes this constitution; where the two conflict, this constitution
governs.

**Version**: 1.1.0 | **Ratified**: 2026-06-07 | **Last Amended**: 2026-06-08
