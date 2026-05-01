---
description: Set up an ablation comparing a candidate change against the current champion
---

Set up an ablation for `$ARGUMENTS` (a feature name from the roadmap, e.g. `value-clipping`, `symmetry-aug`).

Steps:
1. Read `docs/plans/superhuman_roadmap.md` and locate the phase that introduces `$ARGUMENTS`. If not found, ask the user which phase.
2. Identify the relevant config key(s) from `catan/rl/ppo/arguments.py`.
3. Propose two runs:
   - **baseline**: feature disabled (current code or config flag off)
   - **candidate**: feature enabled
4. Both runs use 5M steps, same seed, same opponent mix, distinct `--log-dir` and `--checkpoint-dir`.
5. Print the exact two `python scripts/train.py ...` commands and the planned comparison metric (e.g. WR vs heuristic at 5M, or H2H vs frozen champion).
6. Do NOT launch automatically — output the plan and wait for approval.
