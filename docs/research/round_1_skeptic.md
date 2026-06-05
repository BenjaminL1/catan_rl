# Round 1 — Skeptic / Red Team

I have read the briefing pack. I do not believe v2's design is sound. Every claim below is an attack; if any attack lacks a citation, treat it as withdrawn.

---

## Attack 1 — Catanatron's depth-3 regression is a value-function pathology, not a bug. v2 inherits it.

The Catanatron postmortem (`catanatron.md:84`) reports depth-3 *underperforms* depth-2, and the author's "high-risk-high-reward" explanation is mechanistically explainable from the weights themselves: `catanatron.md:27-41` shows `public_vps: 3e14` dominating `production: 1e8` by **six orders of magnitude**. Depth-3 expectimax (full 11-way 2d6, `catanatron.md:48`) does not "fail to estimate" — it finds *exactly* the VP-maximizing path the value function points to. Those paths systematically trade production for low-probability VP, and the dice expectation flattens the win rate.

**This is a property of any miscalibrated scalar value function under deep stochastic search.** v2 inherits it directly. The Step-4 reward `±1 + (vp_diff)/15` (`glossary.md:57`) couples policy gradients to a VP-leaning shaping term. The value head bootstraps from this signal. Step-5 MCTS (`my_agent.md:101-108`) does PUCT search using `value head at leaf` — same loop, deeper search of a VP-biased value.

Crucially, `my_agent.md:139` gates Step-5 on belief KL ≤ 0.35. **There is no gate on value-head calibration.** The value head could be 15% biased toward VP-race states and the gate would not catch it. Catanatron's measured outcome is the prior: more search of a biased value is *worse*. v2 has produced no evidence to override that prior.

## Attack 2 — QSettlers is statistical noise.

`qsettlers.md:56-60` admits **35% of games discarded** "due to unexpected game resets or other internal errors." Discards are not random — long games crash JSettlers more, and edge-case board states are over-represented. The reported "average reward ~6 ≈ 2nd place" (`qsettlers.md:64`) is therefore an estimator on the *surviving 65%*, biased toward short games where the DQN had early-game luck on initial settlement placement (a one-shot the trade network never sees).

"Loss spiking every 4 episodes" (`qsettlers.md:66`) is framework restart cadence, not learning dynamics. Replay-buffer-of-100 (`qsettlers.md:42`) is below any reasonable off-policy threshold; that result is *predictable from theory*, not learned from data.

**Implication**: `qsettlers.md:81` cites QSettlers as a "cautionary tale for sparse-reward DQN-on-Catan." Invalid — you cannot extract a cautionary tale from a 35%-discarded sample. The sparse-reward warning is derivable from DQN theory alone; QSettlers adds zero signal. Any future v2 doc that cites QSettlers without acknowledging the 35% discard is hand-waving.

## Attack 3 — Five concrete failure modes v2 does not anticipate.

1. **Friendly Robber penalises early VP leaders.** `my_agent.md:64` and `CLAUDE.md:26`: robber cannot land adjacent to a player with `< 3 visible VP`. The value head must estimate state value in a regime where being *behind* is robber-safe and being *ahead* is robber-attracting. No asymmetry term exists in the obs schema (`my_agent.md:5-19`). The reward `±1 + (vp_diff)/15` (`glossary.md:57`) *positively* rewards visible-VP leads even when they invite the robber. Head-on coupling problem.

2. **StackedDice breaks Markov.** `CLAUDE.md` line 184: dice are a 36-bag + 1 noise swap + 20% Karma forced-7 buff. Next-dice distribution depends on (a) bag remaining, (b) **persistent** Karma buff state — `last_player_to_roll_7` is updated only when a 7 rolls (never reset on turn change), and any roll by a player who is not the `last_player_to_roll_7` is forced to 7 with 20% probability. The buff therefore covers however many turns it takes the buffed player to actually roll a 7, not just the next single turn. **Neither bag state nor `karma_buff_active(current_player)` is in the obs dict** (`my_agent.md:5-19`). `value_head(obs)` is therefore not a state-value function in the MDP sense. GAE bootstrap (`my_agent.md:91`) is biased by an unknown amount. MCTS chance nodes (`my_agent.md:105`) need the bag — the plan never states whether the search has access.

> **Erratum (2026-05-15)**: two corrections to this attack.
> 1. Original phrasing treated Karma as a one-roll signal. The mechanic is persistent (`last_player_to_roll_7` updates only on a 7).
> 2. **Karma is already in the obs.** `src/catan_rl/policy/obs_encoder.py:545-560` ships two persistent Karma flags inside `current_player_main`. The "missing from obs" portion of this attack applies only to `bag_remaining`. The persistent-state framing makes the bag-state gap (the one that *is* missing) *larger* than originally described — but the Karma half is moot.

3. **D6 augmentation likely breaks port chirality.** `my_agent.md:149` permutes tiles + corner/edge action axes. Ports are on specific *edges* with *resource* identity (5×2:1 specific per `CLAUDE.md:17`). Reflections flip port chirality. The briefing does not state whether `symmetry_tables.py` permutes the five 2:1 resource-specific ports under reflection. If not, augmented obs are not valid Catan boards — the policy trains on OOD states.

4. **Opp-id embedding will memorize, not generalize.** `my_agent.md:17-18, 99` configure `opponent_kind∈[0,5]` and `opponent_policy_id∈[0,100]` with 8-d embedding + 40% mask. League maxlen is 100 (`my_agent.md:95`) — embedding capacity exceeds the support set. The mask is *training-time only* (`CLAUDE.md:160`); no eval-time test confirms generalization. Against held-out opponents the embedding is forced to "unknown" and the policy silently loses learned conditioning.

5. **2M-step piKL decay is shorter than the learning horizon.** `my_agent.md:94`: piKL decays to 0 over 2M steps. Reward is **terminal-only** (`glossary.md:57`). 1v1 15-VP games are ~250 steps each → 2M steps ≈ 8000 games. With terminal-only signal and self-play variance, 8000 games is the *variance horizon*, not the learning horizon. piKL decays before the value head has a chance to estimate returns from the policy's own trajectories — PPO then collapses onto an undertrained value head, the exact pathology Attack 1 describes.

---

## What I would change my mind on

- **Attack 1**: if v2's Step-5 100-game A/B (`my_agent.md:104`) shows policy+MCTS beats policy-alone by **≥3% absolute symmetrised WR at N≥3 seeds**, I withdraw. A 3pp gain inverts the depth-3 prior.
- **Attack 2**: if any QSettlers number is re-derivable after filtering the 35% discards (impossible — raw logs unpublished), I retract.
- **Attack 3.1**: if value-head calibration on "leading-by-3-VP" states is measured within ±5% of empirical WR, retract.
- **Attack 3.2**: if the obs schema adds bag-remaining counts + previous-roll-was-7 flag, retract.
- **Attack 3.3**: open `src/catan_rl/augmentation/symmetry_tables.py` and grep for `port`; if reflections permute port-resource identity, retract.
- **Attack 3.4**: if eval against a held-out opponent (kind=UNKNOWN at eval) hits WR within 2pp of training-distribution opponents, retract.
- **Attack 3.5**: if reward moves from terminal-only to per-turn shaping, *or* piKL extends to ≥10M steps, retract.

Until then: every claim stands. v2 may ship; it should ship knowing the depth-3 prior is unbeaten.
