# Catan RL Docs

This is the **v2** codebase (`src/catan_rl/`). v1 (`catan/`) is deprecated.

## Reference

- [`1v1_rules.md`](1v1_rules.md) — Colonist.io 1v1 rule table (single source of truth).
- [`architecture.md`](architecture.md) — one-pager on observation, action space, training loop.
- [`io_schema.md`](io_schema.md) — canonical observation keys + 6-head action space + mask keys (merged obs/action schemas).

## Plans

- [`plans/v2/`](plans/v2/) — **the active roadmap**: `design.md` (locked design)
  plus `step3_bc.md`, `step4_ppo.md`, `step5_mcts.md`,
  `setup_strength_roadmap.md`, `setup_labeling.md`, `speckit-playbook.md`.
- [`plans/superhuman_roadmap.md`](plans/superhuman_roadmap.md) — one-page
  north-star vision (defers to `plans/v2/` for detail).
- [`plans/rust_engine.md`](plans/rust_engine.md) — Rust engine status
  (scaffolding only, not the default backend) + migration notes.

## Research

- [`research/SYNTHESIS.md`](research/SYNTHESIS.md) — design synthesis (the
  conclusion of the planning-phase debate).
- [`research/00_briefing/`](research/00_briefing/) — competitor/prior-art
  reference (Catanatron, QSettlers, glossary, this agent).

## Decisions (ADRs)

Permanent records of design choices. See [`decisions/`](decisions/).

## Templates

Reusable prompts. See [`templates/`](templates/).
