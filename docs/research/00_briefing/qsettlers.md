# Briefing — QSettlers

Source: https://akrishna77.github.io/QSettlers/ (Krishna et al. — class project, fetched 2026-05-14).

## What it is

A DQN-based agent for Catan built on top of the JSettlers Java framework. The project is **two partial Q-networks** rather than an end-to-end agent:

- **Trade network** — accept/reject binary classifier on incoming P2P trade offers. Implemented and trained.
- **Settlement-placement network** — value head over 37 tiles for settlement priority. **Unimplemented** ("time and infrastructure constraints").

This is a partial system; the published results cover only the trade module.

## State representations

- **Trade network**: feature vector of player resource counts + opponent resources + trade offer details. Dimensions not exact in the writeup.
- **Settlement network** (proposed-only): **83 input neurons** = 37 tile types + 5 ports + 1 robber + 20 settlement slots + 20 city slots.

The authors **explicitly excluded** exhaustive state features in favor of "domain knowledge selected" inputs to improve training efficiency.

## Action space + masking

- Trade network: **2-way binary** (accept / reject).
- Settlement network: **37-way value head over tiles**, output as placement scores. **No explicit masking strategy** described — the network would emit scores for all 37 tiles regardless of legality.
- This is a substantially **smaller** decision surface than the full Catan action space (Catanatron's 289, our v2's 13-head autoregressive). The narrow scope is what makes the trade module tractable.

## Reward function

**Sparse, end-of-game ranked**:
- 1st place: +10
- 2nd place: +7
- 3rd place: +4
- 4th place: 0

No intermediate shaping. The authors note this "makes learning harder for long-term rewards" but did not address it.

## Learning algorithm + hyperparameters

- DQN with experience replay
- **Replay buffer: 100 experiences** (very small)
- **Batch size: 16**
- **2 hidden layers**: 256 (trade) or 60/50 (settlement)
- **Activation**: ReLU
- **Loss**: MSE with Adam (LR 1e-3 trade, 5e-4 settlement)
- **Exploration**: ε-greedy with ε=1.0, decay rate 0.975

## Training regime

- **Self-play exclusively** — agent vs 3 rule-based JSettlers bots.
- No human benchmarking.
- ~10.5 hours for ~200 games → ~19 games/hour throughput.

## Framework instability (the load-bearing caveat)

**~35% of total games discarded** due to "unexpected game resets or other internal errors" in the JSettlers framework when running AI-only sessions.

This is a serious confound:
- Discards are likely *non-random* (long games, edge-case states, specific configurations more likely to crash).
- Reported reward curves are computed over the surviving 65% — a biased sample.
- Loss-metric spikes "every 4 episodes" suggest framework restarts, not real learning dynamics.

## Reported results

- **Average reward ~6** = roughly 2nd place average (3rd-place threshold is 4, 2nd-place is 7).
- "Very infrequently winning" — no win-rate %, no head-to-head comparison data.
- Loss "general downward trend, spiking after every 4 episodes."

The authors' own qualitative summary is: "perfect trading cannot overcome poor building decisions."

## What this means for our analysis

- **QSettlers is not a strong baseline.** The trade module trained, but settlement module is missing, and the framework discarded 35% of games.
- **Their data is unreliable** for relative-strength claims.
- **The reward shape** (`+10/+7/+4/0` ranking) is a 4-player design and isn't directly applicable to 1v1 (only 2 buckets exist).
- **Useful negative signals from QSettlers**:
  - Sparse end-game reward → slow learning (matches the Phase-A piKL + curriculum decision).
  - Action-space partition (separate networks per decision-type) → fragmentation, no unified value head, no transfer. Our 6-head autoregressive composite avoids this.
  - Replay buffer of 100 is laughably small — they could not train an off-policy method effectively. Sample efficiency was the hidden constraint.
  - Framework dependence → if the env is unstable, learning curves are uninterpretable. v2's env is in-repo Python, fully controlled.
- **What we should NOT borrow**: the writeup as a strength benchmark. QSettlers underperforms Catanatron's `WeightedRandomPlayer` for non-trade decisions because those decisions weren't learned at all.
- **What we should consider**: the writeup as a cautionary tale for sparse-reward, small-replay-buffer DQN-on-Catan. We're not using DQN — but the failure mode (slow learning under sparse reward) is the reason we have piKL + curriculum + BC warm-start.
