---
description: Re-audit the codebase against the documented design
---

Verify the current code against `README.md`, `CLAUDE.md`, and `docs/plans/`. Report drift.

Specifically check:
1. Hyperparameters in `catan/rl/ppo/arguments.py` vs claimed values in `README.md` and `~/.claude/projects/-Users-benjaminli-my-projects-catan-rl/memory/MEMORY.md`.
2. Observation dimensions: `OBS_TILE_DIM`, `CURR_PLAYER_DIM`, `NEXT_PLAYER_DIM` constants vs documentation.
3. Action space: 13 types, 6 heads, mask keys.
4. Resource ordering: `RESOURCES_CW` vs `RESOURCES` usage at each call site.
5. League sampling mode (PFSP vs linear-recency).
6. Checkpoint format compatibility with `checkpoint_07390040.pt`.

Output: bulleted list of drift points, each with file:line and the doc that disagrees.
