"""Perturbed heuristic variants for the BC dataset's variant mix.

Per ``v2_step3_bc.md`` §1, the 30,000 BC games are split:
  * 70% canonical heuristic-vs-heuristic
  * 30% perturbed-vs-heuristic, where one side runs a perturbation
    of the heuristic so the dataset sees broader state coverage.

Two perturbation modes, each owning ~half of the perturbed share:

  :class:`EpsilonGreedyHeuristicAIPlayer`
    Sampling at *initial setup* gets ε-greedy: with probability ε, the
    setup settlement is sampled uniformly from the top-K
    highest-scoring candidate vertices instead of the argmax. Main-turn
    moves are unchanged (the heuristic is already uniform-random over
    main-turn candidates per E0.3).

  :class:`WeightNoisedHeuristicAIPlayer`
    Setup-score weights ``diceRoll_expectation`` are jittered by
    ±``noise_std`` (default 0.15) per game; the main-turn dev-card draw
    probability is independently jittered from 1/3 to a value in
    ``[1/3 - noise_std, 1/3 + noise_std]``. These two changes shift
    both the heuristic's opening and its mid-game strategy without
    breaking its game-rule correctness.

Neither variant mutates the parent class — they subclass and override
``initial_setup`` / ``move``, so the canonical ``heuristicAIPlayer``
in :mod:`catan_rl.agents.heuristic` is unaffected and can still be
mixed into the same dataset.

Both variants accept an optional ``rng: np.random.Generator`` so the
data-generation script can seed them deterministically per game.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from catan_rl.agents.heuristic import heuristicAIPlayer

# ---------------------------------------------------------------------------
# Shared scoring helper (same scoring rule as canonical heuristic.initial_setup)
# ---------------------------------------------------------------------------


_DICE_ROLL_EXPECTATION_BASE: dict[int | None, int] = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    8: 5,
    9: 4,
    10: 3,
    11: 2,
    12: 1,
    None: 0,
}


def _score_setup_vertex(
    v_pt: Any,
    board: Any,
    setup_resources: list[str],
    dice_roll_expectation: Mapping[int | None, float],
) -> float:
    """Score a candidate setup vertex (same formula as canonical).

    Same logic as :meth:`heuristicAIPlayer.initial_setup` lines 47-68,
    factored here so both perturbed variants can call it with their own
    ``dice_roll_expectation`` table.
    """
    score = 0.0
    resources_at_vertex: list[str] = []
    for adj_hex in board.boardGraph[v_pt].adjacent_hex_indices:
        tile = board.hexTileDict[adj_hex]
        res = tile.resource_type
        if res not in resources_at_vertex:
            resources_at_vertex.append(res)
        score += dice_roll_expectation[tile.number_token]
    score += len(resources_at_vertex) * 2.0
    for r in resources_at_vertex:
        if r != "DESERT" and r not in setup_resources:
            score += 2.5
    return score


# ---------------------------------------------------------------------------
# ε-greedy variant — setup-only perturbation
# ---------------------------------------------------------------------------


class EpsilonGreedyHeuristicAIPlayer(heuristicAIPlayer):
    """Heuristic that ε-greedy-picks among the top-K setup candidates.

    Args:
        epsilon: probability of *not* taking the argmax — falls back to
            uniform sampling over the top-K scoring vertices.
        top_k: how many highest-scoring vertices to sample from when
            the ε branch fires.
        rng: optional numpy Generator; defaults to ``np.random.default_rng()``.
    """

    def __init__(
        self,
        name: str,
        colour: str,
        *,
        epsilon: float = 0.10,
        top_k: int = 3,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(name, colour)
        self.epsilon = float(epsilon)
        self.top_k = int(top_k)
        self._rng = rng if rng is not None else np.random.default_rng()

    def initial_setup(self, board) -> None:
        possible = board.get_setup_settlements(self)
        if not possible:
            return
        vertices = list(possible.keys())
        scores = [
            _score_setup_vertex(v, board, self.setupResources, _DICE_ROLL_EXPECTATION_BASE)
            for v in vertices
        ]
        if self._rng.random() < self.epsilon and len(vertices) > 1:
            # Pick uniformly among the top-K (or all, if fewer than K).
            k = min(self.top_k, len(vertices))
            order = sorted(range(len(vertices)), key=lambda i: -scores[i])
            chosen_idx = int(self._rng.choice(order[:k]))
        else:
            chosen_idx = int(np.argmax(scores))
        v_to_build = vertices[chosen_idx]

        for adj_hex in board.boardGraph[v_to_build].adjacent_hex_indices:
            res = board.hexTileDict[adj_hex].resource_type
            if res not in self.setupResources and res != "DESERT":
                self.setupResources.append(res)

        self.build_settlement(v_to_build, board, is_free=True)

        roads = board.get_setup_roads(self)
        if roads:
            r = list(roads.keys())[self._rng.integers(0, len(roads))]
            self.build_road(r[0], r[1], board, is_free=True)


# ---------------------------------------------------------------------------
# Weight-noised variant — setup-score + dev-card-prob perturbation
# ---------------------------------------------------------------------------


class WeightNoisedHeuristicAIPlayer(heuristicAIPlayer):
    """Heuristic with noised setup weights + dev-card-draw probability.

    Args:
        noise_std: fractional noise applied to (a) setup score weights
            (each diceRoll_expectation value × ``(1 + N(0, noise_std))``)
            and (b) the per-turn dev-card draw probability (offset by
            ``±noise_std`` from the base 1/3). Sampled once per
            instance so a single game uses one consistent perturbation.
        rng: optional numpy Generator for deterministic sampling.
    """  # noqa: RUF002

    def __init__(
        self,
        name: str,
        colour: str,
        *,
        noise_std: float = 0.15,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(name, colour)
        if not 0.0 <= noise_std < 1.0:
            raise ValueError(f"noise_std must be in [0, 1), got {noise_std}")
        self._rng = rng if rng is not None else np.random.default_rng()
        self.noise_std = float(noise_std)
        # Sample the noised weights ONCE so the whole game is consistent.
        self._noised_dice_weights = self._sample_noised_weights()
        self._noised_dev_card_prob = self._sample_noised_dev_card_prob()

    def _sample_noised_weights(self) -> dict[int | None, float]:
        out: dict[int | None, float] = {}
        for token, base in _DICE_ROLL_EXPECTATION_BASE.items():
            scale = 1.0 + float(self._rng.normal(0.0, self.noise_std))
            # Clamp scale to a safe positive range so weights don't flip sign.
            scale = max(0.1, min(scale, 2.0))
            out[token] = base * scale
        return out

    def _sample_noised_dev_card_prob(self) -> float:
        offset = float(self._rng.uniform(-self.noise_std, self.noise_std))
        return float(np.clip((1.0 / 3.0) + offset, 0.0, 1.0))

    def initial_setup(self, board) -> None:
        possible = board.get_setup_settlements(self)
        if not possible:
            return
        vertices = list(possible.keys())
        scores = [
            _score_setup_vertex(v, board, self.setupResources, self._noised_dice_weights)
            for v in vertices
        ]
        v_to_build = vertices[int(np.argmax(scores))]

        for adj_hex in board.boardGraph[v_to_build].adjacent_hex_indices:
            res = board.hexTileDict[adj_hex].resource_type
            if res not in self.setupResources and res != "DESERT":
                self.setupResources.append(res)

        self.build_settlement(v_to_build, board, is_free=True)

        roads = board.get_setup_roads(self)
        if roads:
            r = list(roads.keys())[self._rng.integers(0, len(roads))]
            self.build_road(r[0], r[1], board, is_free=True)

    def move(self, board) -> None:
        # Run the canonical move() but with our noised dev-card prob.
        # The cheapest way to swap that probability in is to mirror the
        # canonical body — keeps full control over the per-decision rng.
        self.trade()
        possible_v = board.get_potential_settlements(self)
        if possible_v and (
            self.resources["BRICK"] > 0
            and self.resources["WOOD"] > 0
            and self.resources["SHEEP"] > 0
            and self.resources["WHEAT"] > 0
        ):
            v_keys = list(possible_v.keys())
            self.build_settlement(v_keys[int(self._rng.integers(0, len(v_keys)))], board)

        possible_c = board.get_potential_cities(self)
        if possible_c and (self.resources["WHEAT"] >= 2 and self.resources["ORE"] >= 3):
            v_keys = list(possible_c.keys())
            self.build_city(v_keys[int(self._rng.integers(0, len(v_keys)))], board)

        for _ in range(2):
            if self.resources["BRICK"] > 0 and self.resources["WOOD"] > 0:
                roads = board.get_potential_roads(self)
                if not roads:
                    break
                r_keys = list(roads.keys())
                r = r_keys[int(self._rng.integers(0, len(r_keys)))]
                self.build_road(r[0], r[1], board)

        # Noised dev-card draw probability.
        if self._rng.random() < self._noised_dev_card_prob:
            self.draw_devCard(board)
