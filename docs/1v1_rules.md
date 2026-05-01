# 1v1 Colonist.io Ruleset

This is the **single source of truth** for the rules invariants. Every line below is enforced by code and verified by the rules-invariant test (Phase 0 of the roadmap).

| Rule | 1v1 value | Standard 4p | Code enforcement |
|---|---|---|---|
| Win condition | **15 VP** | 10 VP | `catanGame.maxPoints = 15` (`src/catan_rl/engine/game.py`) |
| Player count | **2** | 3-4 | `catanGame.numPlayers = 2` |
| Player-to-player trading | **DISABLED** | enabled | `player.initiate_trade` early-returns on non-`'BANK'` |
| Discard threshold on 7 | **9 cards** | 7 | `player.discardResources` `maxCards=9` |
| Friendly Robber | cannot place on hex adjacent to a player with `<3` visible VP | none | `catanBoard.get_robber_spots` |
| Dice mechanic | `StackedDice` (36-bag + noise + 20% Karma forced-7) | independent 2d6 | `src/catan_rl/engine/dice.py` |
| Setup | snake draft (1→2→2→1); 2nd settlement yields starting resources | same | `catanGame.build_initial_settlements` + env `_grant_setup_resources` |
| Largest Army threshold | 3 knights | 3 | `catanGame.check_largest_army` |
| Longest Road threshold | 5 roads | 5 | `catanGame.check_longest_road` |

## Invariants enforced by code

These are baked-in and must not be undone:

- **No P2P trade actions.** The 13-type action space has BankTrade only.
- **Single-opponent observation.** `next_player_main` models exactly one opponent.
- **Perfect opponent hand tracking.** Valid only because no P2P trade ⇒ every resource delta is broadcast-observable. See [ADR 0002](decisions/0002-perfect-hand-tracking.md).
- **2-player symmetric zero-sum self-play.** PFSP, Nash pruning, exploitability metric all rest on this.
- **Reward sign-flip symmetry.** Used in Phase 1.5 Z_2 player-swap data augmentation (1v1-only optimization).

Any PR that touches game-rule constants, the action space, the obs schema, or the trading API must explicitly preserve these invariants or be rejected. See [ADR 0001](decisions/0001-1v1-rules-invariant.md).
