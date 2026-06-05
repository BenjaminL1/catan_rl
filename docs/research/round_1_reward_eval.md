# Round 1 — Reward, Evaluation, and Training Regime (Agent C)

## 1. Reward shape: terminal-only with a VP-margin coefficient

Four candidates are on the table. Catanatron Gym's `simple_reward()` (`catanatron_env.py:24-30`) returns `+1 / -1 / 0` — pure terminal. QSettlers uses ranked `+10 / +7 / +4 / 0` (briefing/qsettlers.md:29-35), which collapses to a binary 1v1 since only two buckets exist. The Collazo Medium postmortem reports that `VP_DISCOUNTED_RETURN = 0.9999^turns × VP` was tried in supervised experiments and **still lost to RandomPlayer** (catanatron.md:78) — a strong negative datum: time-discounting VP does not, on its own, fix learnability. v2's pre-revision design used per-step ΔVP shaping; the faculty review flipped this to terminal-only `±1 + (vp_diff)/15` (glossary.md:57).

The faculty review is correct, and the Medium postmortem tells us why: per-step ΔVP shaping silently pays out for *any* action that increases VP — but in Catan, the vast majority of turn states have no productive action, so the dominant gradient signal flows from END_TURN (catanatron.md:75, "~30% of game states give the bot no productive action; classifier collapses to 'always END_TURN'"). Dense ΔVP shaping does not solve this; it *amplifies* it, because policy learns that holding actions stack ΔVP-positive future state and END_TURN is the cheapest way to reach those states. The faculty-corrected `±1 + (vp_diff)/15` keeps the signal dense enough to distinguish *blowout wins from squeaker wins* (a non-trivial credit-assignment signal in a 15-VP game where most matches end 15–8 to 15–14), while never paying for in-turn micro-decisions. The `/15` normalisation keeps the dense term in `[-1, 1]` so it cannot dominate the win signal — a softer hedge than QSettlers' unscaled `+10/+7/+4/0` ladder, which over-rewards rank without bounding margin.

**Verdict**: keep terminal-only `±1 + (vp_diff)/15`. Do **not** reintroduce per-step ΔVP. Do **not** copy Catanatron's flat binary — the VP-margin term carries useful gradient at the same compute cost.

## 2. Training regime: BC warm-start + curriculum + PFSP-hard is load-bearing, not overengineered

QSettlers ran "agent vs 3 JSettlers bots" with no warm-start, no league, and **a 100-sample replay buffer** (qsettlers.md:42, 48-49); the result was "very infrequently winning" and an average reward ~6 (= 2nd place avg in a 4-bucket reward — qsettlers.md:64). Catanatron's TensorForce DQN plateaued at 80% WR vs RandomPlayer after 8 hours and never broke the ceiling against the handcrafted value function (catanatron.md:77). Both negative results stem from the same root cause: sparse terminal reward + no warm-start + no curriculum = exploration that never finds the productive subspace of action sequences.

v2's BC warm-start (my_agent.md:82-87) plus 60→25% heuristic curriculum (my_agent.md:96) is the *minimum-viable* response to that failure mode, not overengineering. PFSP-hard with a 32-game sliding window (my_agent.md:95) is also load-bearing: in a 1v1 zero-sum game, FSP-uniform sampling wastes rollouts on stale weak opponents; PFSP-hard biases toward currently-losing matchups, which is exactly what we need for an exploitability-bounded result.

## 3. Self-play diversity: duo exploiter + opp-id + TrueSkill are *cheap* relative to PPO step cost

Duo exploiter cycles (my_agent.md:97), opponent-id embedding (my_agent.md:99), and TrueSkill (my_agent.md:98) add no measurable wall-clock to a PPO step — embeddings are ~k params, TrueSkill is an O(1) update, exploiter cycles run every 1M main steps. The question is not "is it justified" but "would dropping any of them recover meaningful compute". The answer is no. Keep all three.

## 4. Evaluation protocol: v2 over-specifies, but cuts are easy if compute bites

Catanatron evaluates by leaderboard wins (catanatron.md:11-20); QSettlers has no benchmark at all (qsettlers.md:69-71). v2's protocol — symmetrised WR with N≥3 seeds, PPO-BR-gap, Δ_BR sensitivity, AlphaBeta-d4, champion regression (my_agent.md:165-172) — is heavyweight by comparison, but every component answers a distinct question:

| Component | Question answered | Drop priority |
|---|---|---|
| Symmetrised WR vs heuristic (200 games × N seeds) | "Does it beat the BC target?" | **Keep** (cheapest, most informative) |
| Champion regression (v1 0.56 WR ckpt) | "Did we regress on a known-good test?" | **Keep** (one-shot) |
| AlphaBeta-d2 every 500k | "Does it beat Catanatron's best 1v1 baseline?" | **Keep** (the only credible "superhuman" anchor) |
| PPO-BR-gap 1M + Δ_BR | "What's our exploitability lower bound?" | **Keep** for final claim |
| AlphaBeta-d4 every 5M | "Does deeper search beat us?" | **Drop first** if budget binds — d2 is the published top (catanatron.md:14) |
| TrueSkill in-league | "Is the policy still improving?" | Cheap; keep |
| Nash-pruning round-robin | "Is league diverse?" | Already deferred (FIFO substitute) |

**Minimum-viable "superhuman" claim**: WR ≥ 0.55 (symmetrised, 95% CI lower bound > 0.50 via bootstrap) vs `AlphaBetaPlayer(n=2)` on 1v1-15VP rules + PPO-BR-gap ≤ 0.05. Drop d4, drop ablation seeds beyond N=3 if needed.

## 5. Sample size: 1200 games is right, not overkill

Collazo's MCTS judgement "subhuman with no sample-size context" (catanatron.md:76) is the exact failure to avoid. A single-seed 100-game claim has SE ≈ √(0.25/100) = 0.05, so a 0.55 observed WR has 95% CI ≈ [0.45, 0.65] — *cannot distinguish from 0.50*. v2's 200 games × 2 seats × N=3 seeds = 1200 games gives SE ≈ 0.014; 0.55 WR has bootstrap 95% CI ≈ [0.52, 0.58] — significant at p < 0.05. Below this we get the QSettlers situation (qsettlers.md:55-60): 35% discard rate over ~200 games yielded zero defensible relative-strength claims. **Keep 1200 minimum for gate eval.** Reduce to 400 (single seed) only for *intra-training* probe runs that don't gate anything.

## Recommendation

- **Reward**: terminal-only `±1 + (vp_diff)/15`. Do not revisit until v2.0 ships.
- **Regime**: BC → PPO + piKL anchor → curriculum 60→25% heuristic → PFSP-hard with 32-game window → duo exploiter + opp-id + TrueSkill. All of it. None of these line items is the bottleneck.
- **Eval (full)**: heuristic + champion + AlphaBeta-d2 + PPO-BR-gap + Δ_BR + d4-spot-check. Bootstrap 95% CI on every WR claim. 1200-game gate floor.
- **Eval (compute-bound fallback)**: drop AlphaBeta-d4, drop in-training ablation seeds beyond N=3, keep everything else. Do **not** cut sample size below 1200 for the headline claim.
