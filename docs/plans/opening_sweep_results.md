# Opening-quality sweep — champion policy, setup phase

**Numbers only.** No conclusions, diagnoses or recommendations appear in this
document by design; it is an input to a separate review process.

## Run metadata

| field | value |
|---|---|
| checkpoint | `/Users/benjaminli/my_projects/catan_rl_v2/runs/train/selfplay_pointer_arch_v2/checkpoints/ckpt_000000500.pt` |
| n_boards | `200` |
| seats | `0 (drafts first), 1 (drafts second)` |
| conditions | `greedy (opponent argmax); diverse (opponent setup settlement ~ Uniform(top-8))` |
| decoding | `deterministic argmax under the legal mask` |
| device | `cpu` |
| torch_threads | `2` |
| games | `800` |
| decisions | `3200` |
| wall_clock_seconds | `26.6` |

## Metric definitions

Reproduced verbatim from the script docstring
(`scripts/opening_sweep.py`), which is the authoritative definition.

`dots(h)` is the standard Catan dot count of hex `h` (2/12=1, 3/11=2, 4/10=3,
5/9=4, 6/8=5; desert = 0), imported from
`catan_rl.policy.obs_encoder.DOTS_BY_TOKEN`. `adj(v)` are the hexes touching
vertex `v`; `N(v)` the vertices one road away. *Road-distance* is the number of
board edges on a shortest path.

| metric | definition |
|---|---|
| `pip_sum` | Σ `dots(h)` over `h ∈ adj(v)`. |
| `ore_pips` / `wheat_pips` | Σ `dots(h)` over ORE / WHEAT hexes in `adj(v)`. |
| `has_ore` | 1 iff `ore_pips > 0`. |
| `robber_robustness` | `pip_sum - max(dots(h) for h ∈ adj(v))` — production surviving a robber on the single best hex. |
| `exp_d2/d3/d4` | count of legal future settlement sites at road-distance exactly 2 / 3 / 4. A site `u` is legal iff `u` and every vertex in `N(u)` are unoccupied, evaluated on the live board **plus** the candidate placed at `v`. BFS does not expand *through* an opponent-owned vertex (the engine's road rule). Distance 1 is never legal. |
| `exp_d2/3/4_pip` | the same sets weighted by each site's own `pip_sum`. |
| `centrality_dist` | Euclidean distance from the board centroid (the layout origin `(500,400)`, which is exactly the mean of all 54 vertex positions) in units of one hex edge length (80 px). |
| `pair_*` | pair-level; defined only once both settlements exist, so scored per-candidate at decision 2 and for the realised pair. `E_R` = expected cards of resource `R` per **dice roll** = Σ `dots(h)/36` over both settlements' adjacent `R` hexes. |
| `pair_city_self_sufficient` | 1 iff `E_ORE > 0` **and** `E_WHEAT > 0`, i.e. the pair's own production can eventually cover a city's 3 ore + 2 wheat with no trade. |
| `pair_exp_rolls_to_city` | `max(3/E_ORE, 2/E_WHEAT)` **dice rolls** (both seats roll, ~2 per game round). Deterministic-rate approximation (requirement ÷ rate, then the slower resource); **not** the exact expectation of the max of two hitting times. Ignores 7s, the robber, discards and spending. Undefined (excluded) when `E_ORE` or `E_WHEAT` is 0. |
| `pair_max_ore_lump` | max over number tokens `t` of the count of the pair's settlements adjacent to an ORE hex bearing `t` — the most ore a single dice number pays at once. |
| `pair_robber_robustness` | `pair_pip_sum` minus the largest single-hex loss, where hex `h` costs `dots(h) * (#pair settlements adjacent to h)`. |
| `pair_spread` | distance between the two settlements, in hex edge lengths. |
| `road_2hop_pip_max` / `_sum` | over `u ∈ N(v2)\{v1}` that are legal future sites (`v1` = the settlement just placed, `v2` = the road's far end): max / sum of `pip_sum(u)`. Distance 2 is the nearest legal site along the road. |
| `road_breadth` / `_pip` | legal future sites within road-distance 2 of `v2` with `v1` deleted from the graph (all paths run outward through this road); count / `pip_sum`-weighted. |
| `nonshared_pip_sum` / `nonshared_ore_pips` | pip / ore-pip mass on `adj(v2) - adj(v1)` — the hexes the road newly reaches. |
| `shared_ore_pips` | ore-pip mass on `adj(v1) ∩ adj(v2)`. |

**Percentile rank** of the chosen candidate among the legal set `S`
(chosen included), mid-rank so ties split symmetrically:

```
pct = 100 * ( #{s ∈ S : m(s) < m(chosen)} + 0.5*#{s ∈ S : m(s) = m(chosen)} ) / |S|
```

A uniformly-random chooser averages 50. **Higher percentile always means a
higher RAW value**; the `high percentile =` column in every table below states
what that means for each metric (note `centrality_dist` and
`pair_exp_rolls_to_city` are metrics where high = further out / slower).

Geometric distances (`centrality_dist`, `pair_spread`) are rounded to 3 decimal places before ranking. The engine rounds vertex pixel coordinates to 2 dp, so vertices that are geometrically equidistant from the centroid differ by ~1e-4 edge-lengths in float; unrounded, that noise imposes an arbitrary strict order on an exact tie. The 54 vertices form only 6 distinct radius clusters (sizes 6/6/12/12/6/12), within-cluster spread 8.1e-5, minimum between-cluster gap 0.359 — so 3 dp merges only true ties and separates every real one.

## n per cell

| cell | games | settlement decisions | road decisions |
|---|---:|---:|---:|
| greedy, seat 0 | 200 | 400 | 400 |
| greedy, seat 1 | 200 | 400 | 400 |
| diverse, seat 0 | 200 | 400 | 400 |
| diverse, seat 1 | 200 | 400 | 400 |
| **total** | **800** | **1600** | **1600** |

Legal-candidate-set sizes (the percentile denominators):

| decision | n | mean candidates | min | max |
|---|---:|---:|---:|---:|
| settlement #1 (d0) | 800 | 52.0 | 50 | 54 |
| settlement #2 (d2) | 800 | 44.5 | 42 | 48 |
| road (d1,d3) | 1600 | 3.0 | 2 | 3 |

> Note on design overlap: at `seat 0`, decision 0 is taken on an empty board, so it is *identical* between the greedy and diverse conditions for a given seed (the opponent has not acted yet). The two conditions diverge from decision 2 onward at seat 0, and from decision 0 onward at seat 1.

## Percentile ranks — settlement decisions (pooled)

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pip_sum` | 1600 | more total production | 94.0 | 96.3 | 97.9 | 0.883 | 0.002 |
| `ore_pips` | 1600 | more ore production | 38.0 | 83.3 | 92.6 | 0.331 | 0.396 |
| `has_ore` | 1600 | touches ore | 38.0 | 83.3 | 85.9 | 0.019 | 0.396 |
| `wheat_pips` | 1600 | more wheat production | 33.0 | 79.6 | 89.2 | 0.234 | 0.364 |
| `robber_robustness` | 1600 | more production survives a robber on the best hex | 93.5 | 96.5 | 98.1 | 0.828 | 0.002 |
| `exp_d2` | 1600 | more legal sites 2 roads away | 77.8 | 78.3 | 87.2 | 0.133 | 0.041 |
| `exp_d3` | 1600 | more legal sites 3 roads away | 66.3 | 72.1 | 82.6 | 0.117 | 0.096 |
| `exp_d4` | 1600 | more legal sites 4 roads away | 59.2 | 66.7 | 79.1 | 0.063 | 0.116 |
| `exp_d2_pip` | 1600 | richer sites 2 roads away | 65.7 | 75.0 | 83.7 | 0.105 | 0.037 |
| `exp_d3_pip` | 1600 | richer sites 3 roads away | 60.2 | 71.0 | 79.3 | 0.046 | 0.113 |
| `exp_d4_pip` | 1600 | richer sites 4 roads away | 56.5 | 69.0 | 79.8 | 0.057 | 0.176 |
| `centrality_dist` | 1600 | FURTHER from the board centre | 23.3 | 30.2 | 32.0 | 0.000 | 0.998 |

### Settlement #1 (decision 0)

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pip_sum` | 800 | more total production | 94.4 | 96.3 | 98.0 | 0.891 | 0.000 |
| `ore_pips` | 800 | more ore production | 40.0 | 88.0 | 94.4 | 0.424 | 0.255 |
| `has_ore` | 800 | touches ore | 40.0 | 85.0 | 85.2 | 0.000 | 0.255 |
| `wheat_pips` | 800 | more wheat production | 31.5 | 79.0 | 88.4 | 0.209 | 0.372 |
| `robber_robustness` | 800 | more production survives a robber on the best hex | 94.0 | 97.0 | 98.1 | 0.889 | 0.000 |
| `exp_d2` | 800 | more legal sites 2 roads away | 77.8 | 77.8 | 86.0 | 0.005 | 0.026 |
| `exp_d3` | 800 | more legal sites 3 roads away | 66.7 | 66.7 | 75.0 | 0.046 | 0.125 |
| `exp_d4` | 800 | more legal sites 4 roads away | 66.7 | 66.7 | 74.0 | 0.019 | 0.026 |
| `exp_d2_pip` | 800 | richer sites 2 roads away | 63.9 | 71.0 | 78.0 | 0.026 | 0.025 |
| `exp_d3_pip` | 800 | richer sites 3 roads away | 58.8 | 66.8 | 75.0 | 0.011 | 0.106 |
| `exp_d4_pip` | 800 | richer sites 4 roads away | 60.0 | 68.5 | 76.9 | 0.020 | 0.086 |
| `centrality_dist` | 800 | FURTHER from the board centre | 30.0 | 32.0 | 33.3 | 0.000 | 1.000 |

### Settlement #2 (decision 2) — includes pair-level metrics

Note: at decision 2 the first settlement is already fixed, so `pair_pip_sum`, `pair_ore_pips` and `pair_wheat_pips` are each the corresponding vertex metric plus a per-decision constant. A constant shift is a monotone transform, so their percentile rows are *identical* to `pip_sum` / `ore_pips` / `wheat_pips` by construction — not a duplication bug. `pair_robber_robustness`, `pair_city_self_sufficient`, `pair_exp_rolls_to_city`, `pair_max_ore_lump` and `pair_spread` are not constant shifts and do differ.

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pip_sum` | 800 | more total production | 92.4 | 96.5 | 97.9 | 0.875 | 0.004 |
| `ore_pips` | 800 | more ore production | 38.0 | 41.9 | 89.5 | 0.239 | 0.537 |
| `has_ore` | 800 | touches ore | 38.0 | 41.9 | 86.2 | 0.037 | 0.537 |
| `wheat_pips` | 800 | more wheat production | 33.7 | 79.8 | 90.2 | 0.259 | 0.356 |
| `robber_robustness` | 800 | more production survives a robber on the best hex | 90.4 | 96.4 | 98.8 | 0.767 | 0.004 |
| `exp_d2` | 800 | more legal sites 2 roads away | 75.6 | 83.7 | 90.5 | 0.261 | 0.056 |
| `exp_d3` | 800 | more legal sites 3 roads away | 66.0 | 79.3 | 87.0 | 0.189 | 0.068 |
| `exp_d4` | 800 | more legal sites 4 roads away | 52.4 | 73.9 | 81.4 | 0.107 | 0.206 |
| `exp_d2_pip` | 800 | richer sites 2 roads away | 69.8 | 80.4 | 88.3 | 0.184 | 0.049 |
| `exp_d3_pip` | 800 | richer sites 3 roads away | 62.8 | 75.0 | 82.6 | 0.080 | 0.119 |
| `exp_d4_pip` | 800 | richer sites 4 roads away | 46.8 | 69.6 | 83.0 | 0.094 | 0.265 |
| `centrality_dist` | 800 | FURTHER from the board centre | 18.1 | 26.7 | 30.2 | 0.000 | 0.996 |
| `pair_pip_sum` | 800 | more total production | 92.4 | 96.5 | 97.9 | 0.875 | 0.004 |
| `pair_ore_pips` | 800 | more ore production | 38.0 | 41.9 | 89.5 | 0.239 | 0.537 |
| `pair_wheat_pips` | 800 | more wheat production | 33.7 | 79.8 | 90.2 | 0.259 | 0.356 |
| `pair_city_self_sufficient` | 800 | pair can reach 3 ore + 2 wheat unaided | 50.0 | 50.0 | 81.9 | 0.016 | 0.185 |
| `pair_exp_rolls_to_city` | 650 | SLOWER to the first city | 12.8 | 32.2 | 60.9 | 0.000 | 0.602 |
| `pair_max_ore_lump` | 800 | bigger single-roll ore payout | 46.4 | 46.7 | 48.9 | 0.044 | 0.756 |
| `pair_robber_robustness` | 800 | more production survives a robber on the best hex | 93.0 | 96.7 | 98.8 | 0.901 | 0.004 |
| `pair_spread` | 800 | settlements FURTHER apart | 39.3 | 54.7 | 65.1 | 0.003 | 0.417 |

## Percentile ranks — road decisions (pooled)

Setup roads emanate from the settlement just placed, so there are at most 3 legal candidates; percentiles are correspondingly coarse (with 3 candidates the only attainable mid-ranks are 16.7 / 50.0 / 83.3, before ties). The argmax rates below are the primary road statistic.

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `road_2hop_pip_max` | 1600 | richer best site unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.052 |
| `road_2hop_pip_sum` | 1600 | richer sites unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.091 |
| `road_breadth` | 1600 | opens more legal sites | 50.0 | 66.7 | 66.7 | 0.000 | 0.219 |
| `road_breadth_pip` | 1600 | opens richer sites | 50.0 | 83.3 | 83.3 | 0.000 | 0.041 |
| `nonshared_pip_sum` | 1600 | road reaches toward richer new hexes | 50.0 | 83.3 | 83.3 | 0.000 | 0.044 |
| `nonshared_ore_pips` | 1600 | road reaches toward more ore | 50.0 | 50.0 | 50.0 | 0.000 | 0.130 |
| `shared_pip_sum` | 1600 | road's shared hexes are richer | 16.7 | 50.0 | 66.7 | 0.000 | 0.420 |
| `shared_ore_pips` | 1600 | road's shared hexes hold more ore | 50.0 | 50.0 | 66.7 | 0.000 | 0.193 |

## Per-condition and per-seat breakdown

### Settlement decisions — greedy, seat 0

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pip_sum` | 400 | more total production | 94.0 | 96.3 | 98.0 | 0.902 | 0.000 |
| `ore_pips` | 400 | more ore production | 38.4 | 83.3 | 92.6 | 0.330 | 0.380 |
| `has_ore` | 400 | touches ore | 38.4 | 83.3 | 85.2 | 0.028 | 0.380 |
| `wheat_pips` | 400 | more wheat production | 33.3 | 79.3 | 88.9 | 0.212 | 0.338 |
| `robber_robustness` | 400 | more production survives a robber on the best hex | 93.0 | 96.5 | 98.1 | 0.858 | 0.000 |
| `exp_d2` | 400 | more legal sites 2 roads away | 77.8 | 77.8 | 83.7 | 0.117 | 0.043 |
| `exp_d3` | 400 | more legal sites 3 roads away | 66.7 | 66.7 | 85.8 | 0.120 | 0.033 |
| `exp_d4` | 400 | more legal sites 4 roads away | 66.7 | 66.7 | 80.2 | 0.087 | 0.125 |
| `exp_d2_pip` | 400 | richer sites 2 roads away | 64.8 | 74.1 | 83.7 | 0.090 | 0.028 |
| `exp_d3_pip` | 400 | richer sites 3 roads away | 62.8 | 69.6 | 77.5 | 0.040 | 0.050 |
| `exp_d4_pip` | 400 | richer sites 4 roads away | 59.3 | 68.5 | 80.2 | 0.077 | 0.160 |
| `centrality_dist` | 400 | FURTHER from the board centre | 18.6 | 30.2 | 33.3 | 0.000 | 1.000 |

### Settlement decisions — greedy, seat 1

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pip_sum` | 400 | more total production | 92.4 | 96.0 | 97.8 | 0.833 | 0.005 |
| `ore_pips` | 400 | more ore production | 37.0 | 77.6 | 91.0 | 0.287 | 0.468 |
| `has_ore` | 400 | touches ore | 37.0 | 83.7 | 87.0 | 0.007 | 0.468 |
| `wheat_pips` | 400 | more wheat production | 32.9 | 76.6 | 90.2 | 0.258 | 0.398 |
| `robber_robustness` | 400 | more production survives a robber on the best hex | 90.2 | 95.7 | 98.0 | 0.755 | 0.005 |
| `exp_d2` | 400 | more legal sites 2 roads away | 66.0 | 86.0 | 87.2 | 0.138 | 0.037 |
| `exp_d3` | 400 | more legal sites 3 roads away | 60.9 | 75.0 | 80.9 | 0.117 | 0.175 |
| `exp_d4` | 400 | more legal sites 4 roads away | 59.0 | 71.7 | 76.1 | 0.060 | 0.100 |
| `exp_d2_pip` | 400 | richer sites 2 roads away | 67.0 | 76.1 | 84.0 | 0.105 | 0.037 |
| `exp_d3_pip` | 400 | richer sites 3 roads away | 58.4 | 73.0 | 80.9 | 0.050 | 0.147 |
| `exp_d4_pip` | 400 | richer sites 4 roads away | 55.1 | 69.4 | 79.0 | 0.068 | 0.180 |
| `centrality_dist` | 400 | FURTHER from the board centre | 28.3 | 30.4 | 32.0 | 0.000 | 0.995 |

### Settlement decisions — diverse, seat 0

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pip_sum` | 400 | more total production | 94.4 | 96.4 | 98.8 | 0.915 | 0.000 |
| `ore_pips` | 400 | more ore production | 38.4 | 84.7 | 92.9 | 0.360 | 0.367 |
| `has_ore` | 400 | touches ore | 38.4 | 83.3 | 85.2 | 0.028 | 0.367 |
| `wheat_pips` | 400 | more wheat production | 33.3 | 79.8 | 88.9 | 0.225 | 0.347 |
| `robber_robustness` | 400 | more production survives a robber on the best hex | 93.5 | 96.6 | 98.1 | 0.877 | 0.000 |
| `exp_d2` | 400 | more legal sites 2 roads away | 77.8 | 77.8 | 83.7 | 0.135 | 0.055 |
| `exp_d3` | 400 | more legal sites 3 roads away | 66.7 | 66.7 | 85.7 | 0.113 | 0.055 |
| `exp_d4` | 400 | more legal sites 4 roads away | 66.7 | 66.7 | 80.2 | 0.065 | 0.110 |
| `exp_d2_pip` | 400 | richer sites 2 roads away | 64.7 | 73.1 | 83.4 | 0.125 | 0.048 |
| `exp_d3_pip` | 400 | richer sites 3 roads away | 60.2 | 67.6 | 77.8 | 0.048 | 0.105 |
| `exp_d4_pip` | 400 | richer sites 4 roads away | 59.3 | 68.2 | 78.6 | 0.030 | 0.150 |
| `centrality_dist` | 400 | FURTHER from the board centre | 20.9 | 28.6 | 33.3 | 0.000 | 1.000 |

### Settlement decisions — diverse, seat 1

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pip_sum` | 400 | more total production | 94.6 | 96.7 | 98.0 | 0.882 | 0.003 |
| `ore_pips` | 400 | more ore production | 38.0 | 84.0 | 92.0 | 0.347 | 0.370 |
| `has_ore` | 400 | touches ore | 38.0 | 84.0 | 86.2 | 0.013 | 0.370 |
| `wheat_pips` | 400 | more wheat production | 32.6 | 80.0 | 89.2 | 0.240 | 0.375 |
| `robber_robustness` | 400 | more production survives a robber on the best hex | 94.0 | 96.7 | 98.9 | 0.823 | 0.003 |
| `exp_d2` | 400 | more legal sites 2 roads away | 70.2 | 86.0 | 88.0 | 0.142 | 0.030 |
| `exp_d3` | 400 | more legal sites 3 roads away | 63.0 | 75.0 | 81.5 | 0.120 | 0.122 |
| `exp_d4` | 400 | more legal sites 4 roads away | 59.0 | 74.0 | 78.0 | 0.040 | 0.130 |
| `exp_d2_pip` | 400 | richer sites 2 roads away | 67.0 | 77.0 | 83.7 | 0.100 | 0.035 |
| `exp_d3_pip` | 400 | richer sites 3 roads away | 58.6 | 72.0 | 79.3 | 0.045 | 0.147 |
| `exp_d4_pip` | 400 | richer sites 4 roads away | 52.2 | 70.8 | 81.9 | 0.052 | 0.212 |
| `centrality_dist` | 400 | FURTHER from the board centre | 27.7 | 30.0 | 32.0 | 0.000 | 0.998 |

### Road decisions — greedy, seat 0

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `road_2hop_pip_max` | 400 | richer best site unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.048 |
| `road_2hop_pip_sum` | 400 | richer sites unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.100 |
| `road_breadth` | 400 | opens more legal sites | 50.0 | 66.7 | 66.7 | 0.000 | 0.193 |
| `road_breadth_pip` | 400 | opens richer sites | 50.0 | 83.3 | 83.3 | 0.000 | 0.043 |
| `nonshared_pip_sum` | 400 | road reaches toward richer new hexes | 50.0 | 83.3 | 83.3 | 0.000 | 0.052 |
| `nonshared_ore_pips` | 400 | road reaches toward more ore | 50.0 | 50.0 | 50.0 | 0.000 | 0.122 |
| `shared_pip_sum` | 400 | road's shared hexes are richer | 33.3 | 50.0 | 66.7 | 0.000 | 0.412 |
| `shared_ore_pips` | 400 | road's shared hexes hold more ore | 50.0 | 50.0 | 66.7 | 0.000 | 0.190 |

### Road decisions — greedy, seat 1

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `road_2hop_pip_max` | 400 | richer best site unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.040 |
| `road_2hop_pip_sum` | 400 | richer sites unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.095 |
| `road_breadth` | 400 | opens more legal sites | 50.0 | 66.7 | 83.3 | 0.000 | 0.230 |
| `road_breadth_pip` | 400 | opens richer sites | 50.0 | 83.3 | 83.3 | 0.000 | 0.035 |
| `nonshared_pip_sum` | 400 | road reaches toward richer new hexes | 50.0 | 83.3 | 83.3 | 0.000 | 0.037 |
| `nonshared_ore_pips` | 400 | road reaches toward more ore | 50.0 | 50.0 | 50.0 | 0.000 | 0.150 |
| `shared_pip_sum` | 400 | road's shared hexes are richer | 16.7 | 50.0 | 66.7 | 0.000 | 0.420 |
| `shared_ore_pips` | 400 | road's shared hexes hold more ore | 50.0 | 50.0 | 66.7 | 0.000 | 0.182 |

### Road decisions — diverse, seat 0

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `road_2hop_pip_max` | 400 | richer best site unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.080 |
| `road_2hop_pip_sum` | 400 | richer sites unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.102 |
| `road_breadth` | 400 | opens more legal sites | 50.0 | 66.7 | 66.7 | 0.000 | 0.215 |
| `road_breadth_pip` | 400 | opens richer sites | 50.0 | 83.3 | 83.3 | 0.000 | 0.058 |
| `nonshared_pip_sum` | 400 | road reaches toward richer new hexes | 66.7 | 83.3 | 83.3 | 0.000 | 0.055 |
| `nonshared_ore_pips` | 400 | road reaches toward more ore | 50.0 | 50.0 | 50.0 | 0.000 | 0.120 |
| `shared_pip_sum` | 400 | road's shared hexes are richer | 33.3 | 50.0 | 66.7 | 0.000 | 0.432 |
| `shared_ore_pips` | 400 | road's shared hexes hold more ore | 50.0 | 50.0 | 66.7 | 0.000 | 0.193 |

### Road decisions — diverse, seat 1

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `road_2hop_pip_max` | 400 | richer best site unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.043 |
| `road_2hop_pip_sum` | 400 | richer sites unlocked | 50.0 | 83.3 | 83.3 | 0.000 | 0.065 |
| `road_breadth` | 400 | opens more legal sites | 50.0 | 66.7 | 66.7 | 0.000 | 0.237 |
| `road_breadth_pip` | 400 | opens richer sites | 50.0 | 83.3 | 83.3 | 0.000 | 0.028 |
| `nonshared_pip_sum` | 400 | road reaches toward richer new hexes | 50.0 | 83.3 | 83.3 | 0.000 | 0.030 |
| `nonshared_ore_pips` | 400 | road reaches toward more ore | 50.0 | 50.0 | 50.0 | 0.000 | 0.128 |
| `shared_pip_sum` | 400 | road's shared hexes are richer | 33.3 | 50.0 | 66.7 | 0.000 | 0.415 |
| `shared_ore_pips` | 400 | road's shared hexes hold more ore | 50.0 | 50.0 | 66.7 | 0.000 | 0.205 |

### Pair metrics at decision 2, by condition and seat

**greedy, seat 0**

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pair_pip_sum` | 200 | more total production | 91.9 | 96.5 | 97.7 | 0.880 | 0.000 |
| `pair_ore_pips` | 200 | more ore production | 38.3 | 41.7 | 87.5 | 0.185 | 0.560 |
| `pair_wheat_pips` | 200 | more wheat production | 34.3 | 79.1 | 89.5 | 0.240 | 0.330 |
| `pair_city_self_sufficient` | 200 | pair can reach 3 ore + 2 wheat unaided | 50.0 | 50.0 | 81.0 | 0.010 | 0.160 |
| `pair_exp_rolls_to_city` | 167 | SLOWER to the first city | 12.6 | 33.3 | 61.6 | 0.000 | 0.587 |
| `pair_max_ore_lump` | 200 | bigger single-roll ore payout | 45.4 | 46.5 | 48.8 | 0.025 | 0.815 |
| `pair_robber_robustness` | 200 | more production survives a robber on the best hex | 93.0 | 96.5 | 98.8 | 0.900 | 0.000 |
| `pair_spread` | 200 | settlements FURTHER apart | 36.0 | 54.7 | 67.9 | 0.000 | 0.445 |

**greedy, seat 1**

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pair_pip_sum` | 200 | more total production | 92.2 | 95.7 | 97.9 | 0.865 | 0.010 |
| `pair_ore_pips` | 200 | more ore production | 37.2 | 40.3 | 88.6 | 0.245 | 0.555 |
| `pair_wheat_pips` | 200 | more wheat production | 33.0 | 77.1 | 90.2 | 0.255 | 0.385 |
| `pair_city_self_sufficient` | 200 | pair can reach 3 ore + 2 wheat unaided | 44.6 | 50.0 | 82.6 | 0.030 | 0.280 |
| `pair_exp_rolls_to_city` | 143 | SLOWER to the first city | 13.8 | 38.2 | 60.6 | 0.000 | 0.594 |
| `pair_max_ore_lump` | 200 | bigger single-roll ore payout | 45.7 | 46.8 | 83.0 | 0.040 | 0.700 |
| `pair_robber_robustness` | 200 | more production survives a robber on the best hex | 93.6 | 96.7 | 98.9 | 0.920 | 0.010 |
| `pair_spread` | 200 | settlements FURTHER apart | 39.1 | 52.2 | 65.9 | 0.005 | 0.400 |

**diverse, seat 0**

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pair_pip_sum` | 200 | more total production | 92.9 | 96.5 | 98.8 | 0.905 | 0.000 |
| `pair_ore_pips` | 200 | more ore production | 38.1 | 42.9 | 89.5 | 0.245 | 0.535 |
| `pair_wheat_pips` | 200 | more wheat production | 34.9 | 80.0 | 90.7 | 0.265 | 0.350 |
| `pair_city_self_sufficient` | 200 | pair can reach 3 ore + 2 wheat unaided | 50.0 | 50.0 | 82.1 | 0.010 | 0.145 |
| `pair_exp_rolls_to_city` | 171 | SLOWER to the first city | 11.6 | 26.7 | 59.2 | 0.000 | 0.626 |
| `pair_max_ore_lump` | 200 | bigger single-roll ore payout | 46.4 | 46.5 | 48.8 | 0.055 | 0.765 |
| `pair_robber_robustness` | 200 | more production survives a robber on the best hex | 93.0 | 96.5 | 98.8 | 0.885 | 0.000 |
| `pair_spread` | 200 | settlements FURTHER apart | 36.9 | 52.9 | 64.5 | 0.000 | 0.450 |

**diverse, seat 1**

| metric | n | high percentile = | p25 | median | p75 | frac >90th | frac <50th |
|---|---:|---|---:|---:|---:|---:|---:|
| `pair_pip_sum` | 200 | more total production | 92.4 | 96.7 | 97.9 | 0.850 | 0.005 |
| `pair_ore_pips` | 200 | more ore production | 37.0 | 58.2 | 90.2 | 0.280 | 0.500 |
| `pair_wheat_pips` | 200 | more wheat production | 33.0 | 80.4 | 90.4 | 0.275 | 0.360 |
| `pair_city_self_sufficient` | 200 | pair can reach 3 ore + 2 wheat unaided | 50.0 | 50.0 | 81.5 | 0.015 | 0.155 |
| `pair_exp_rolls_to_city` | 169 | SLOWER to the first city | 13.0 | 31.2 | 60.9 | 0.000 | 0.598 |
| `pair_max_ore_lump` | 200 | bigger single-roll ore payout | 46.7 | 46.7 | 81.0 | 0.055 | 0.745 |
| `pair_robber_robustness` | 200 | more production survives a robber on the best hex | 93.5 | 96.7 | 98.9 | 0.900 | 0.005 |
| `pair_spread` | 200 | settlements FURTHER apart | 40.2 | 56.5 | 64.9 | 0.005 | 0.375 |

## Ore-substitution rate

Fraction of **settlement decisions** where the chosen candidate has `ore_pips ≤ 1` while some legal alternative has `ore_pips ≥ 3` and `|pip_sum(alt) - pip_sum(chosen)| ≤ 2`.

| slice | n | substitutions | rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| all settlement decisions | 1600 | 426 | 0.2662 | [0.2452, 0.2884] |
| decision 0 (settlement #1) | 800 | 161 | 0.2013 | [0.1749, 0.2304] |
| decision 2 (settlement #2) | 800 | 265 | 0.3312 | [0.2995, 0.3646] |
| condition = greedy | 800 | 223 | 0.2787 | [0.2488, 0.3108] |
| condition = diverse | 800 | 203 | 0.2537 | [0.2248, 0.2850] |
| seat 0 | 800 | 201 | 0.2512 | [0.2224, 0.2824] |
| seat 1 | 800 | 225 | 0.2812 | [0.2512, 0.3134] |

Conditional variant — denominator restricted to decisions where a qualifying
ore-rich alternative actually existed:

| slice | n (alternative existed) | substitutions | rate | Wilson 95% CI |
|---|---:|---:|---:|---|
| all settlement decisions | 1320 | 426 | 0.3227 | [0.2981, 0.3484] |

A qualifying ore-rich alternative existed at 1320 / 1600 = 0.8250 of settlement decisions.

## Road argmax rates

A chosen road that **ties** the maximum counts as argmax.

| slice | metric | n | chosen = argmax | rate | Wilson 95% CI | n with unique max |
|---|---|---:|---:|---:|---|---:|
| all road decisions | `road_2hop_pip_max` | 1600 | 1171 | 0.7319 | [0.7096, 0.7530] | 1444 |
| all road decisions | `road_breadth` | 1600 | 1148 | 0.7175 | [0.6949, 0.7390] | 769 |
| road #1 (d1) | `road_2hop_pip_max` | 800 | 614 | 0.7675 | [0.7370, 0.7955] | 743 |
| road #1 (d1) | `road_breadth` | 800 | 705 | 0.8812 | [0.8570, 0.9019] | 204 |
| road #2 (d3) | `road_2hop_pip_max` | 800 | 557 | 0.6963 | [0.6635, 0.7271] | 701 |
| road #2 (d3) | `road_breadth` | 800 | 443 | 0.5537 | [0.5191, 0.5879] | 565 |
| condition = greedy | `road_2hop_pip_max` | 800 | 590 | 0.7375 | [0.7059, 0.7668] | 716 |
| condition = greedy | `road_breadth` | 800 | 581 | 0.7262 | [0.6943, 0.7560] | 399 |
| condition = diverse | `road_2hop_pip_max` | 800 | 581 | 0.7262 | [0.6943, 0.7560] | 728 |
| condition = diverse | `road_breadth` | 800 | 567 | 0.7087 | [0.6763, 0.7392] | 370 |
| seat 0 | `road_2hop_pip_max` | 800 | 580 | 0.7250 | [0.6930, 0.7548] | 724 |
| seat 0 | `road_breadth` | 800 | 585 | 0.7312 | [0.6995, 0.7608] | 328 |
| seat 1 | `road_2hop_pip_max` | 800 | 591 | 0.7388 | [0.7072, 0.7680] | 720 |
| seat 1 | `road_breadth` | 800 | 563 | 0.7037 | [0.6712, 0.7344] | 441 |

## Shared / non-shared hex partition (road decisions)

For each road decision with ≥2 legal candidates, `b` = the best alternative to the chosen road `c` by `road_2hop_pip_max`. The table asks where `b`'s extra pip mass sits: on the **non-shared** hex (`adj(v2) - adj(v1)`, the hex only the road reaches) or on the **shared** hexes (`adj(v1) ∩ adj(v2)`, already touched by the settlement).

The four classes are mutually exclusive and sum to `n`.

| slice | n | richer on NON-SHARED only | richer on SHARED only | richer on both | not richer |
|---|---:|---:|---:|---:|---:|
| all road decisions | 1600 | 292 | 685 | 45 | 578 |
| road #1 (d1) | 800 | 131 | 339 | 15 | 315 |
| road #2 (d3) | 800 | 161 | 346 | 30 | 263 |
| condition = greedy | 800 | 163 | 338 | 29 | 270 |
| condition = diverse | 800 | 129 | 347 | 16 | 308 |
| seat 0 | 800 | 119 | 348 | 20 | 313 |
| seat 1 | 800 | 173 | 337 | 25 | 265 |

Ore subset — same partition, asking where the best alternative's EXTRA ORE
sits (`nonshared_ore_pips` / `shared_ore_pips`, `b` minus `c`):

| slice | n | ore gain on NON-SHARED only | ore gain on SHARED only | ore gain on both | no ore gain |
|---|---:|---:|---:|---:|---:|
| all road decisions | 1600 | 138 | 291 | 31 | 1140 |
| road #1 (d1) | 800 | 67 | 185 | 20 | 528 |
| road #2 (d3) | 800 | 71 | 106 | 11 | 612 |
| condition = greedy | 800 | 69 | 139 | 14 | 578 |
| condition = diverse | 800 | 69 | 152 | 17 | 562 |
| seat 0 | 800 | 58 | 147 | 11 | 584 |
| seat 1 | 800 | 80 | 144 | 20 | 556 |

## Realised opening pairs (the settlement pair the policy actually built)

| slice | n | mean pair_pip_sum | mean pair_ore_pips | frac has ore | frac city-self-sufficient | median exp. rolls to city (defined only) | mean pair_max_ore_lump | mean pair_spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all games | 800 | 22.30 | 5.40 | 0.9163 | 0.8125 | 21.60 (n=650) | 0.96 | 4.14 |
| condition = greedy | 400 | 22.14 | 4.99 | 0.8850 | 0.7750 | 24.00 (n=310) | 0.92 | 4.13 |
| condition = diverse | 400 | 22.45 | 5.82 | 0.9475 | 0.8500 | 21.60 (n=340) | 1.01 | 4.15 |
| seat 0 | 400 | 22.38 | 5.67 | 0.9275 | 0.8450 | 21.60 (n=338) | 0.97 | 4.12 |
| seat 1 | 400 | 22.21 | 5.14 | 0.9050 | 0.7800 | 24.00 (n=312) | 0.95 | 4.16 |

Wilson 95% CIs over all 800 games — pair touches ore: [0.8950, 0.9335]; pair is city-self-sufficient: [0.7840, 0.8380].

Distribution of `pair_ore_pips` over the realised pairs:

| pair_ore_pips | count | fraction |
|---:|---:|---:|
| 0 | 67 | 0.0838 |
| 2 | 39 | 0.0488 |
| 3 | 87 | 0.1087 |
| 4 | 138 | 0.1725 |
| 5 | 137 | 0.1713 |
| 6 | 40 | 0.0500 |
| 7 | 78 | 0.0975 |
| 8 | 90 | 0.1125 |
| 9 | 58 | 0.0725 |
| 10 | 33 | 0.0413 |
| 11 | 18 | 0.0225 |
| 12 | 10 | 0.0125 |
| 13 | 3 | 0.0037 |
| 15 | 1 | 0.0013 |
| 16 | 1 | 0.0013 |

Distribution of `pair_max_ore_lump` (largest single-dice-number ore payout):

| pair_max_ore_lump | count | fraction |
|---:|---:|---:|
| 0 | 67 | 0.0838 |
| 1 | 694 | 0.8675 |
| 2 | 39 | 0.0488 |

## LIMITATIONS — what this sweep does NOT measure

1. **No outcome, no counterfactual win rate.** Every number here scores the opening against hand-defined board metrics. Not one game was played past the setup phase. Nothing here establishes that a candidate with a better percentile would have won more; the metrics are assumptions about what a good opening is, not measurements of value.
2. **No value-head or policy-confidence signal.** Only the argmax identity is recorded; the margin between the chosen and runner-up candidate, the head's entropy and the value head's estimate are not.
3. **Ports are not scored.** No metric references `BoardVertex.port`, so a candidate on a 2:1 ore port and one on open coast are treated identically. Port access changes the real cost of the 4:1 trades referenced in the motivating observation.
4. **Dev cards, robber play, largest army and longest road are out of scope.** The metrics cover production geometry and expansion room only.
5. **`pair_exp_rolls_to_city` is a rate approximation**, not an expectation (see the definition table): it ignores 7s, the robber, the 9-card discard, spending, the finite 19-per-resource bank and the variance of hitting times. It is undefined whenever the pair produces no ore or no wheat, and those cases are excluded from its statistics rather than imputed — so its median is conditioned on a self-sufficient pair and is NOT comparable across slices with different self-sufficiency rates.
6. **Expansion metrics ignore resource costs and tempo.** `exp_d2/3/4` and `road_breadth` count reachable legal sites; they do not model whether the player can afford the roads, nor who arrives first.
7. **The opponent-cut dimension is only partially captured.** BFS refuses to expand through opponent-owned vertices, so present blocking is reflected, but no metric simulates *future* opponent road-building or robber placement.
8. **Territory / contested-middle is proxied by `centrality_dist` and `pair_spread` alone.** Neither measures control of the middle relative to the opponent's settlements.
9. **Percentile ranks are relative to the legal set, not to an absolute standard.** On a board where every legal vertex is poor, a 99th-percentile choice is still poor; the raw per-candidate values in the JSON are needed to separate the two.
10. **Road decisions have ≤3 candidates**, so their percentile distributions are coarse and heavily tied; argmax rates carry the signal.
11. **Single checkpoint, single architecture.** One champion (`ckpt_000000500.pt`) under argmax decoding. No comparison against another checkpoint, a human corpus, a heuristic baseline or a search-augmented agent is made here, so no number in this report is a *relative* strength claim.
12. **The two conditions are not independent samples of the same population.** The diverse condition perturbs only the opponent's setup settlement; at seat 0 decision 0 the two conditions are identical by construction (see the n-per-cell note), so pooled statistics double-count those decisions.
