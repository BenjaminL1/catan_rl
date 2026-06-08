# Spec Kit Playbook — driving the self-play build

Ready-to-paste inputs for each Spec Kit step, plus the keystone spec. Spec Kit
runs inside Claude Code as `/speckit-*` skills (note the **hyphen**). One-time
setup is committed (`.specify/`, `.claude/skills/`, `plansDirectory`).

> **Before you start:** restart Claude Code once so the `/speckit-*` skills
> register. Run the constitution **once** for the project; run `specify →
> clarify → plan → tasks → analyze → implement` **per feature**.

## Workflow order

```
/speckit-constitution     (once, project-wide guardrails)
        │
/speckit-specify          → specs/NNN-<feature>/spec.md     (what, not how)
        │
/speckit-clarify          → resolves ≤5 ambiguities into spec.md
        │
/speckit-plan             → plan.md (the how: seams, constraints)
        │
/speckit-checklist        (optional: requirements-quality gate)
        │
/speckit-tasks            → tasks.md (ordered, dependency-aware)
        │
/speckit-analyze          → spec↔plan↔tasks consistency report
        │
/speckit-implement        → executes tasks; one PR per feature/phase
```

---

## 1. `/speckit-constitution` — run once

Paste this as the input. It encodes the non-negotiables every later spec/plan
must respect (seeded from `CLAUDE.md`).

```
Create the project constitution for a superhuman 1v1 Settlers of Catan RL agent
(Colonist.io ruleset), trained with custom PPO + league self-play. Encode these
as binding principles:

1. THE 1v1 RULESET IS SACRED. 15 VP win, exactly 2 players, NO player-to-player
   trading (bank/port only), 9-card discard threshold, Friendly Robber (no
   robber on a hex adjacent to a player with <3 visible VP), StackedDice
   (shuffled bag of 36 + 1 noise swap + 20% Karma forced-7). Never generalize to
   4-player. Any change touching game-rule constants, the action space, or the
   obs schema must state how it preserves the 1v1 ruleset or be rejected.

2. ENGINE INTEGRITY. Never edit the engine to change game rules without
   explicitly flagging it — the engine matches Colonist.io and any drift breaks
   evaluation comparability.

3. CHECKPOINT BACK-COMPAT. Policy state-dict shape changes require a one-shot
   migration script and a documented lineage. Prefer changes that keep existing
   checkpoints loadable.

4. OBSERVABILITY IS ADDITIVE. TensorBoard scalar names are append-only; new
   diagnostics are new scalars, never renames.

5. CONFIG SOURCE OF TRUTH. arguments.py is authoritative for hyperparameters;
   README/MEMORY may lag. Verify against arguments.py.

6. DEVICE POLICY. Training device resolves auto→MPS on M1 (batched SGD is
   ~3x faster); eval is pinned to CPU (batch=1 is faster there). CUDA is opt-in.

7. ENGINEERING DISCIPLINE. Conventional commits (<72 chars, lowercase). One PR
   per phase — no big-bang merges spanning phases. No `Co-Authored-By` AI
   trailers. Every behavioral change ships with tests. CI must be green on the
   merge SHA (ruff + mypy strict + pytest, Python 3.11+, GUI pixel tests skip
   off-darwin). Verify `gh pr checks` conclusions line-by-line before merging.

8. SELF-PLAY IS 2-PLAYER ZERO-SUM. PFSP, Nash pruning, exploitability, and
   perfect 1v1 hand-tracking are all defined only for the symmetric 2-player
   game.
```

After it runs, review `.specify/memory/constitution.md` and commit it.

---

## 2. `/speckit-specify` — the keystone feature

This is the keystone spec input. Paste it verbatim.

```
Feature: Frozen-policy self-play opponent + policy-vs-policy evaluation.

WHY: Training currently runs only against a fixed, weak heuristic (random
opening placement, never plays dev cards). The agent plateaus at "beats a weak
bot," and once WR-vs-heuristic saturates there is no way to measure further
progress. The only mechanism to climb past the heuristic — playing against
frozen snapshots of the agent's own improving past selves — is scaffolded but
unwired (two NotImplementedError guards), and there is no policy-vs-policy eval.
This feature wires the single primitive that unblocks self-play, best-response /
exploitability probing, AND human-vs-policy play.

WHAT IT MUST DO (behavior, not implementation):
1. A league SNAPSHOT can act as an in-env opponent: when an environment's
   opponent is a league snapshot, the opponent's moves are produced by a FROZEN
   policy loaded from the league pool — not the heuristic.
2. Turning self-play on no longer errors: with a non-empty league and
   snapshot_weight > 0, a rollout runs to completion with no NotImplementedError.
3. Fresh snapshots can enter play: after a new snapshot is added mid-training,
   the vec env can be told the new opponent assignment so subsequent rollouts
   use it (today the opponent mix is frozen at construction).
4. Policy-vs-policy evaluation exists: the champion can be evaluated head-to-head
   against any loaded opponent policy (a league snapshot or a saved checkpoint),
   returning a symmetrized win rate with a Wilson confidence interval.

ACCEPTANCE CRITERIA (testable):
- With snapshot_weight=0.5 and a one-entry league whose snapshot is a stub
  policy that always ends its turn, a short rollout completes and the opponent's
  observed actions come from that stub policy (not the heuristic).
- The two NotImplementedError guards are removed; their guarding tests are
  replaced with tests of the live snapshot path.
- After add_snapshot + a vec-env opponent refresh, the next rollout's opponent
  is the newly added snapshot.
- Opponent NN inference for the snapshot opponent is batched in the main process
  across envs (not per-env batch=1).
- Policy-vs-policy eval returns a finite WR + Wilson CI for champion-vs-loaded-
  opponent over N seat-symmetrized games.
- Determinism: same seed + same snapshot id → reproducible opponent behavior.

CONSTRAINTS / INVARIANTS:
- Preserve the 1v1 ruleset exactly (no rule, action-space, or reward changes).
- NO observation or action-head SHAPE change — the in-flight bootstrap
  checkpoint must remain loadable as the first self-play seed. (The opponent-id
  embedding already exists in the trunk; feed it real values, do not resize it.)
- Maintain checkpoint back-compat.
- Eval runs on CPU; snapshot-opponent inference during rollout follows the
  learner device but stays batched.

NON-GOALS (explicitly out of scope — separate later phases):
- PFSP / TrueSkill-Glicko rating / Nash-weighted pruning / exploiter cycles.
- Curriculum-weight tuning beyond a simple configurable heuristic-vs-snapshot mix.
- BC warm-start, MCTS/PUCT search, piKL anchor.

DONE WHEN: a self-play run trains without error, the agent's WR against its own
recent snapshots hovers near 50% (healthy zero-sum equilibrium) while its WR
against a FROZEN early baseline rises, and policy-vs-policy eval can produce a
champion-vs-snapshot strength curve.
```

---

## 3. `/speckit-clarify` — answer the likely questions

Run it; it asks up to 5 targeted questions and writes answers back into the
spec. Expect (and answer) roughly:

- **Snapshot opponent inference:** batched main-process inference across envs
  (mirrors the v1 "deferred opponent" design; the measured bottleneck is the
  small-batch forward pass, so batching matters).
- **Mid-rollout swap mechanism:** rebuild/refresh per-env opponent assignment
  each rollout via a `vec_env.set_opponents(...)` call threading kind/policy_id
  into the next `reset`.
- **Curriculum:** include a single configurable `heuristic:snapshot` mix
  (e.g. start 60/40) but DEFER any schedule/annealing to a later phase.
- **Opponent device:** follows the learner (MPS), batched; eval stays CPU.
- **Snapshot selection when pool > 1:** uniform-random for now (PFSP is a
  non-goal here).

---

## 4. `/speckit-plan` — the technical how

Paste this so the plan targets the real seams.

```
Plan the implementation against these code seams (verify line numbers, they
drift):

- src/catan_rl/env/catan_env.py — `_run_opponent_main_turn` (~:699) currently
  calls `opp.move()`; the NotImplementedError for snapshot opponents is ~:181.
  Add a path: on opponent_type='snapshot', drive the opponent turn by encoding
  its POV obs + action masks and sampling from a frozen CatanPolicy.
- src/catan_rl/selfplay/league.py — `add_snapshot` (producer, already wired),
  `peek_by_id`, and the raise at ~:245. Wire snapshot SAMPLING into
  `build_env_opponent_mix` so it returns 'snapshot' + a concrete snapshot_id.
- src/catan_rl/ppo/vec_env.py + game_manager.py — where each env's opponent is
  constructed; add `set_opponents()` + batched main-process opponent inference.
- src/catan_rl/ppo/training_loop.py — `_build_env_kwargs_list`/opponent mix
  (~:159), the construction-time mix lock (~:200), the add_snapshot call (~:646).
- src/catan_rl/replay/player_factory.py — `PolicyPlayer` already loads a
  state_dict and runs `policy.sample`; REUSE it for policy-vs-policy eval rather
  than writing a new loader.
- configs/ppo_default.yaml — snapshot_weight (~:65), and a new
  heuristic:snapshot mix knob.

Hard constraints: no obs/action shape change; batched opponent inference;
deterministic eval seeding; 1v1 ruleset preserved; checkpoint back-compat.
Follow TDD — tests for the snapshot-opponent primitive and the eval path BEFORE
implementation. Surface the design as: (1) the frozen in-env opponent primitive,
(2) the league consumer + delete the two guards, (3) mid-rollout swap, (4)
policy-vs-policy eval + a champion-WR scalar.
```

---

## 5. `/speckit-checklist` *(optional)* — requirements-quality gate

```
Generate a checklist verifying the keystone spec is complete and unambiguous:
every acceptance criterion is independently testable; the no-shape-change
invariant is explicit; non-goals are listed; the determinism and batched-
inference requirements are measurable; and the phase advancement gate
(WR thresholds) is stated.
```

## 6. `/speckit-tasks` — generate the task list

Run it (no input needed). Then verify the ORDER respects dependencies:
frozen-opponent primitive → league consumer + delete guards → mid-rollout swap →
policy-vs-policy eval → curriculum mix. Each task should map to a test.

## 7. `/speckit-analyze` — consistency gate (run after tasks, before implement)

Run it. Look for: any acceptance criterion in `spec.md` with no covering task;
any task that contradicts a constraint (e.g. proposes an obs-shape change);
plan/spec terminology drift. Fix drift before implementing.

## 8. `/speckit-implement` — execute

Run it. While it works:
- Keep TDD: tests first, then minimal code to pass.
- **Do not change obs/action shapes** — the bootstrap run is producing the seed
  checkpoint in parallel; a shape change orphans it.
- Verify each acceptance criterion as its task closes.
- Land it as ONE PR (the keystone = one phase). Run `/code-review` and the
  `verify` skill before pushing; confirm `gh pr checks` are green line-by-line.

---

## Phase gate — when the keystone is "done enough" to advance

From `docs/plans/v2/step4_ppo.md §6` (seat-symmetrized, N≥3 seeds):
- **G1:** WR ≥ 0.90 vs heuristic — the bootstrap / graduate-to-self-play bar (equivalently, once the WR-vs-heuristic curve flattens). Subsumes "beat v1" (v1 peaked ~0.55) with no v1 policy loaded.
- **G2:** PPO best-response gap ≤ 0.65 (a fresh 1M-step adversary can't exceed
  this WR against the champion).
- **G3:** WR ≥ 0.60 vs a frozen early-**v2** baseline (the bootstrap checkpoint or an early self-play snapshot — **no v1 champions**).

G2/G3 use the *same* frozen-opponent primitive this keystone builds — which is
why it unblocks measurement too.

## Tips to fully utilize Spec Kit here

- **Seed, don't start blank.** The constitution and specify inputs above are
  pre-seeded from `CLAUDE.md` and the gap audit — paste and refine, don't
  retype.
- **One feature = one `specs/NNN-*/` dir = one PR.** Commit `spec.md`,
  `plan.md`, `tasks.md` together; they're the durable record the everything-
  claude-code planner couldn't give you.
- **Plan-mode plans persist** to `docs/plans/claude/` (via `plansDirectory`) —
  separate from these curated specs.
- **The bootstrap checkpoint is the first league snapshot.** Pick the strongest
  checkpoint produced by `bootstrap_v1` when the keystone lands; seed self-play
  from it.
- **After the keystone:** the next specs are A.3 (PFSP + rating), the
  AlphaBeta/exploitability bench, then BC warm-start, then MCTS/piKL — each its
  own `/speckit-specify`.
