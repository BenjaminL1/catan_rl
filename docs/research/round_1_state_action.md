# Round 1 — State & Action Representation (Agent A)

## Thesis

For a 1v1 Colonist.io target, v2's current factored representation is the right baseline, but two pieces — the explicit hex-grid prior and the hidden-information head — are decisively more valuable than either Catanatron's or QSettlers' designs. The places to keep humble are masking granularity and symmetry augmentation, both of which carry real costs.

## 1. Action space — factor before you flatten

v2 uses `MultiDiscrete([13, 54, 72, 19, 5, 5])` decomposed into 6 autoregressive heads (`my_agent.md` §2, citing `docs/action_schema.md:7`, `src/catan_rl/policy/heads.py:129-180`). Catanatron flattens to **289 discrete actions** after compressing a 5000+ original space (`catanatron.md` §"Action space", citing `catanatron_env.py:238` and the Medium postmortem). QSettlers fragments into a **2-way trade network** and an unbuilt **37-way settlement network** (`qsettlers.md` §"Action space + masking").

The Catanatron postmortem itself names the failure mode the flat space invites: "5000+ → 289 action-space compression was necessary; the original space had so many no-op-like actions that learning collapsed" (`catanatron.md` §"Citable failure modes" #1). The compressed 289 still cannot share parameters across "which corner" decisions because each corner-build action is its own integer. v2's factored heads share the corner head's weights across `BUILD_SETTLEMENT` (type 0) and `BUILD_CITY` (type 1) (`my_agent.md` §2 table). For a superhuman 1v1 target where corner-placement quality compounds across 30+ decisions per game, parameter sharing is not optional. QSettlers' fragmentation is the worst of both — separate networks with no shared value head, which the briefing notes as "fragmentation, no unified value head, no transfer" (`qsettlers.md` §"What this means").

**Keep v2's 6-head autoregressive design.**

## 2. Action masking — granularity earns its keep

v2 ships a 9-key mask dict that splits `corner_settlement` / `corner_city`, three `resource1_*` contexts, and `resource2_default` (`my_agent.md` §3, citing `src/catan_rl/env/masks.py:55-101`). Catanatron returns a single flat bool list — `[action_int in valid for action_int in range(self.action_space_size)]` (`catanatron.md`, citing `catanatron_env.py:106-112`). QSettlers describes "no explicit masking strategy" — the network emits 37 tile scores regardless of legality (`qsettlers.md` §"Action space + masking").

The granularity matters because masks differ by context — a settlement corner must have no adjacent settlement, a city corner must already hold the agent's settlement, a trade resource must satisfy port-ratio thresholds (`my_agent.md` §3 table). Collapsing them onto one flat list, as Catanatron does, forces the policy to learn that the same corner integer means different things in different game phases. With hard masking via `masked_log_softmax` setting illegal slots to -∞ (`my_agent.md` §3, citing `src/catan_rl/policy/heads.py:51-61`), v2 gets zero invalid-action probability mass for free. QSettlers' no-masking design partially explains its inability to leave 2nd-place average ("very infrequently winning" per `qsettlers.md` §"Reported results").

**Keep v2's 9-key contextual masking.** The split costs little; the learning-stability win is large.

## 3. Hex-grid encoding — three prior strengths, ranked

| Approach | Inductive bias | Cost |
|---|---|---|
| v2: TileEncoder transformer + axial pos emb + tripartite GNN | Hex adjacency + axial coords + explicit hex/vertex/edge graph | 2-layer transformer + 30k GNN params |
| Catanatron: 2D-CNN board tensor `(channels, 21, 11)` | Cartesian grid prior — misaligned with hex topology | Channels per `catanatron_env.py:62-66` |
| QSettlers: 37-tile one-hot input | No prior — flat MLP | 83 input neurons total |

v2's GNN encodes 19 hex + 54 vertex + 72 edge nodes with 2 rounds of mean-pool message passing (`my_agent.md` §1, citing `src/catan_rl/policy/encoders.py:134-180`). That is the right shape for a board where a settlement vertex earns from up to three hexes; adjacency is the load-bearing relation, and explicit graph message-passing encodes it. Catanatron's CNN forces a rectangular embedding (`catanatron.md`, citing `catanatron_env.py:62-66`); the brick-pattern offset means a "diagonal" hex neighbor is not a CNN-kernel neighbor without padding tricks. QSettlers' one-hot has no spatial prior at all.

**Keep v2's GNN + axial-pos-emb stack.** It is strictly stronger than either alternative for hex topology.

## 4. Hidden information — why 1v1 makes belief modelling decisive

v2 maintains *perfect* opponent resource counts via `BroadcastHandTracker` (`my_agent.md` §4, citing `src/catan_rl/env/hand_tracker.py:53-100`) and a 5-way **belief head** over hidden dev-card types trained with soft CE at weight 0.05 (`my_agent.md` §4, citing `src/catan_rl/policy/heads.py:407-426`). This is correct only because 1v1 with no P2P trading makes every resource mutation observable (`my_agent.md` §4, citing `CLAUDE.md` lines 28-29). Catanatron's value function "reads the *true* opp resources" — it cheats (`catanatron.md` §"What's NOT in the repo"). QSettlers does no opp modelling.

In 4-player with P2P trade, perfect tracking is impossible; Catanatron's shortcut breaks self-play comparability but works pragmatically. In 1v1 the tracker is rule-legal and gives v2 a representation advantage neither baseline can match. The belief head is the only *unobservable* component (dev-card type), so the policy gets one cleanly-shaped uncertainty signal rather than a joint-distribution mess.

**Keep both. They are 1v1's free lunch.**

## 5. Symmetry — not free, but cheap

v2 applies D6 dihedral augmentation (12 group elements) with `symmetry_aug_prob=0.5` (`my_agent.md` §8, citing `src/catan_rl/augmentation/symmetry_tables.py`). Neither baseline uses board symmetry. The cost is real: the axial positional embedding "breaks the encoder's permutation-equivariance over tiles" (CLAUDE.md Phase 2.1), so the augmented samples cannot share an encoder pass — they require fresh forward passes through permuted obs. At prob 0.5 this is a ~1.5× sample-multiplier cost.

**Keep symmetry aug at 0.5**; defer raising it to 1.0 until preflight E0.3/E0.4 confirms the equivariance probe passes (per `my_agent.md` §8).

## Recommendation

**Keep:** (a) 6-head autoregressive action space; (b) 9-key contextual masking; (c) TileEncoder + axial pos emb + tripartite GNN; (d) `BroadcastHandTracker` perfect tracking + 5-way belief head; (e) D6 symmetry aug at 0.5.

**Borrow from Catanatron:** nothing in this scope. Its flat 289-action space is a downgrade; its CNN board tensor is misaligned with hex topology; its perfect-info value function is unavailable to us only because we *already have* the legal alternative.

**Borrow from QSettlers:** nothing. Its fragmented two-network design and unmasked 37-way head are documented anti-patterns (`qsettlers.md` §"What this means").

**Drop:** the opponent-action head's weight in PPO is already correctly disabled because "supervision shifts under self-play" (`my_agent.md` §7); freezing its parameters at BC value is the right call. No further drops in this scope.
