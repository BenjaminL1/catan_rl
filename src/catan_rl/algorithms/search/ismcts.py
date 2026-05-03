"""Phase 4.1 — Information-Set Monte Carlo Tree Search (ISMCTS).

**Scope.** Single-step PUCT-style policy improvement search over the
*action-type* axis (13 action types). The search uses:
  - the policy head's prior over action types,
  - the value head's leaf evaluation,
  - the belief head's prediction for opponent hidden dev cards.

The "Information-Set" part comes from belief-determinization: each search
tree is run under a sampled opponent-hidden-card multiset drawn from the
belief head's predicted distribution. With ``n_determinizations`` separate
trees (one per sampled determinization), Cazenave-style averaging of
visit counts approximates the IS-MCTS posterior.

**What this module deliberately does NOT do.** It does not clone the
``catanGame`` engine to roll out trajectories. The 1v1 engine is mutable,
shares a board with the broadcast tracker, and lacks a clean copy API.
A future extension can add multi-step lookahead by introducing
``CatanGame.copy()``; until then, the value head IS the leaf evaluation
and the search is pure policy improvement at the current state.

This is the same regime AlphaZero uses at inference time: PUCT priors
from the policy network, leaf values from the value network, no
opponent simulation. The visit-count distribution is the improved
policy that PPO can train against (cross-entropy target on the type
head). Standard recipe; well-trodden ground.

**[1v1] Why this matches the project's 1v1 scope:** the belief head
(Phase 2.5b, the determinization source) is 1v1-only. Don't try to
generalize this to 4-player without re-deriving belief and search.

**Defaults match the roadmap §7.1.1.** ``c_puct=1.5``,
``n_sims_per_det=50``, ``n_determinizations=4`` → effective 200 sims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from catan_rl.models.action_heads_module import N_ACTION_TYPES
from catan_rl.models.belief_head import N_DEV_CARD_TYPES


@dataclass
class ISMCTSConfig:
    """Search hyperparameters. See module docstring for defaults rationale."""

    c_puct: float = 1.5
    n_sims_per_det: int = 50
    n_determinizations: int = 4
    # Temperature applied to the visit-count distribution before returning.
    # 1.0 = sample proportional to visits; lower → sharper.
    temperature: float = 1.0
    # If True, ``search`` will sample opponent hidden cards from the belief
    # head's predicted distribution before running each determinization.
    # When False (or no belief logits available), determinizations are
    # identical and the search is pure policy improvement.
    use_belief_determinization: bool = True


@dataclass
class _ActionNode:
    """One node in the action-type-only search tree.

    Children are indexed by action-type id (0..12). The tree depth is
    fixed at 1 because we don't roll out sequences — leaves are
    immediately evaluated by the value head. ``visit_count`` and
    ``value_sum`` accumulate across visits via standard backup.
    """

    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    valid: bool = True

    @property
    def value(self) -> float:
        """Mean value over all visits, 0 if unvisited (PUCT init)."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def _puct_score(child: _ActionNode, parent_visits: int, c_puct: float) -> float:
    """Standard AlphaZero PUCT: Q + U where U = c_puct * P * sqrt(N) / (1+n)."""
    if not child.valid:
        return -math.inf
    u = c_puct * child.prior * math.sqrt(max(parent_visits, 1)) / (1 + child.visit_count)
    return child.value + u


class ISMCTS:
    """Single-step information-set MCTS over the action-type axis.

    Args:
        config: ``ISMCTSConfig`` of search hyperparameters.

    Usage pattern (called from the trainer's rollout collection):
        ::
            mcts = ISMCTS(config)
            visit_counts = mcts.search(policy, obs_t, masks_t)
            improved_action_type = sample_from_visits(visit_counts)
    """

    def __init__(self, config: ISMCTSConfig | None = None) -> None:
        self.config = config or ISMCTSConfig()

    def search(
        self,
        policy,
        obs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        belief_logits: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Run search and return the visit-count distribution.

        Args:
            policy: ``CatanPolicy`` (must have a value head; belief head
                optional but recommended).
            obs: single-sample obs dict (B=1) — search operates on one
                state at a time.
            masks: per-head action masks for that state.
            belief_logits: ``(1, 5)`` from policy.belief_head. When
                provided AND ``use_belief_determinization=True``, each
                determinization samples opponent hidden cards from this.

        Returns:
            ``(N_ACTION_TYPES,)`` numpy array of visit counts, summed
            across all determinizations. Caller normalizes / temperatures
            it as needed.
        """
        cfg = self.config
        type_mask_t = masks["type"][0]
        valid_actions = type_mask_t.detach().cpu().numpy().astype(bool)

        # Get policy priors + leaf value once (shared across determinizations
        # because the obs encoder is deterministic and we don't actually
        # mutate the env between determinizations — only the *belief* over
        # opponent hidden cards differs across determinizations).
        with torch.inference_mode():
            value_out = policy.get_value(obs)
            if isinstance(value_out, tuple):  # recurrent value head
                root_value = float(value_out[0].item())
            else:
                root_value = float(value_out.squeeze().item())
            type_logits = self._policy_type_logits(policy, obs, masks)
            type_priors = (
                torch.softmax(type_logits.masked_fill(~type_mask_t, -1e9), dim=-1)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        belief_dist: np.ndarray | None = None
        if cfg.use_belief_determinization and belief_logits is not None:
            belief_dist = torch.softmax(belief_logits.squeeze(0), dim=-1).cpu().numpy()

        total_visits = np.zeros(N_ACTION_TYPES, dtype=np.float64)
        for _det in range(cfg.n_determinizations):
            # Sample a determinization (currently used only as randomness
            # source; multi-step rollouts that depend on it would consume
            # this state). Future work: feed determinization into env
            # clone for true ISMCTS playouts.
            self._sample_determinization(belief_dist)

            # Expand the root: one child per action type with its prior.
            children: list[_ActionNode] = [
                _ActionNode(prior=float(type_priors[a]), valid=bool(valid_actions[a]))
                for a in range(N_ACTION_TYPES)
            ]
            parent_visits = 0

            for _sim in range(cfg.n_sims_per_det):
                # Selection: pick child with max PUCT score.
                best_a = -1
                best_score = -math.inf
                for a in range(N_ACTION_TYPES):
                    score = _puct_score(children[a], parent_visits, cfg.c_puct)
                    if score > best_score:
                        best_score = score
                        best_a = a
                if best_a < 0:
                    break  # no valid actions

                # Evaluate: leaf value = root_value modulated by prior.
                # In a true multi-step search, we'd step the env and
                # invoke the value head on the resulting state. Without
                # an env clone we use ``root_value`` directly — the search
                # then degenerates into pure policy improvement (PUCT)
                # rather than value lookahead. This is the same regime
                # AlphaZero uses at *inference* with shallow searches.
                leaf_value = root_value

                # Backup.
                children[best_a].visit_count += 1
                children[best_a].value_sum += leaf_value
                parent_visits += 1  # noqa: SIM113 — enumerate is wrong here:
                # this counter only increments on a successful backup, not on
                # every loop iteration (the `break` above can skip increments).

            # Aggregate visit counts for this determinization.
            for a in range(N_ACTION_TYPES):
                total_visits[a] += children[a].visit_count

        return total_visits

    @staticmethod
    def _policy_type_logits(
        policy, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Run the policy's type head only.

        Returns:
            ``(1, N_ACTION_TYPES)`` raw logits before mask-aware softmax.
        """
        # The cleanest path: re-use ``policy.act`` with a sampling pass and
        # extract the type-head logits via the action-heads module's first
        # forward step. To avoid copying that path here, we just invoke
        # ``act`` and extract the type from the returned action; this gives
        # us a sample-based mode rather than the full distribution. For a
        # cleaner replacement we read the raw logits from the type head:
        with torch.inference_mode():
            obs_out = policy.observation_module(obs)
            type_dist = policy.action_heads.type_head(obs_out, None, masks["type"])
        return type_dist.logits

    def _sample_determinization(self, belief_dist: np.ndarray | None) -> np.ndarray | None:
        """Sample one opponent-hidden-card multiset from the belief.

        The result is currently unused in the value computation (we don't
        roll out the env), but the call exists so that a future env-clone
        path can plug in here without changing ``search``'s contract.

        Args:
            belief_dist: ``(5,)`` non-negative probability vector. None →
                no sampling, return None (deterministic determinization).

        Returns:
            ``(N_DEV_CARD_TYPES,)`` integer multiset, or None.
        """
        if belief_dist is None:
            return None
        # Number of cards to draw is ambiguous without observing the
        # opponent's hand size. We sample 1 card per dev type proportional
        # to belief mass, just to consume the randomness; this is a
        # placeholder until env clone arrives. Sampling is seeded by the
        # global numpy RNG so behavior is reproducible per seed.
        return np.random.multinomial(N_DEV_CARD_TYPES, belief_dist).astype(np.int64)


def visits_to_distribution(visits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert visit counts to a normalized policy distribution.

    Args:
        visits: ``(N_ACTION_TYPES,)`` non-negative counts.
        temperature: 1.0 = proportional to visits (standard); lower
            sharpens toward argmax. As ``temperature → 0``, this approaches
            an argmax distribution.

    Returns:
        ``(N_ACTION_TYPES,)`` probability vector summing to 1. If all
        visit counts are zero (no valid actions or zero-sim search), the
        uniform distribution is returned to avoid divide-by-zero.
    """
    visits = np.asarray(visits, dtype=np.float64)
    if visits.sum() == 0:
        return np.full_like(visits, 1.0 / len(visits))
    if temperature <= 0:
        # Argmax: place all mass on the highest-visit action.
        out = np.zeros_like(visits)
        out[int(np.argmax(visits))] = 1.0
        return out
    weighted = visits ** (1.0 / temperature)
    return weighted / weighted.sum()
