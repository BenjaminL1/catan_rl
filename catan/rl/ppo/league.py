"""
Charlesworth-style league with PFSP (Prioritized Fictitious Self-Play) sampling.

Policies are added every add_every updates. Sampling uses PFSP: opponents
with win rates closest to 50% against the current policy are sampled most
often. This avoids wasting time on opponents that are trivially easy or
impossibly hard.

PFSP priority: p(i) ∝ w_i * (1 - w_i) + ε
  - w_i = 0.0 → priority ≈ ε  (agent always wins, too easy)
  - w_i = 0.5 → priority = 0.25 (balanced, sampled most)
  - w_i = 1.0 → priority ≈ ε  (agent always loses, too hard)

New policies start with w=0.5 (neutral assumption).

League.sample() now returns a 3-tuple: (type, state_dict, policy_id)
  - policy_id is -1 for "random" opponents.
  - Call league.update_result(policy_id, win) after each game to update stats.
"""
import copy
import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


class League:
    """In-memory pool of past policy state dicts with PFSP opponent sampling."""

    def __init__(
        self,
        maxlen: int = 100,
        add_every: int = 4,
        random_weight: float = 0.5,
        heuristic_weight: float = 0.0,
        build_policy_fn=None,
        use_pfsp: bool = True,
        pfsp_epsilon: float = 1e-3,
    ):
        """
        Args:
            maxlen: Max policies to store (FIFO eviction).
            add_every: Add current policy every this many updates.
            random_weight: Probability of sampling a fully random opponent.
            heuristic_weight: Probability of sampling the heuristic opponent.
                Remaining probability (1 - random - heuristic) samples from
                the league pool. Heuristic games provide consistent, diverse
                signal and break self-play echo chambers.
            build_policy_fn: Callable(device) -> CatanPolicy for inference.
            use_pfsp: If True use PFSP weighting; if False use Charlesworth
                      linear-recency bias (original behaviour).
            pfsp_epsilon: Smoothing constant so zero-game policies get sampled.
        """
        self.policies: deque = deque(maxlen=maxlen)
        self.add_every = add_every
        self.random_weight = random_weight
        self.heuristic_weight = heuristic_weight
        self._build_policy_fn = build_policy_fn
        self._policy_cache: Optional[Any] = None

        self.use_pfsp = use_pfsp
        self.pfsp_epsilon = pfsp_epsilon

        # Monotonically increasing policy IDs (survive eviction tracking)
        self._next_id: int = 0
        self._policy_ids: deque = deque(maxlen=maxlen)     # parallel to self.policies
        self._policy_stats: Dict[int, List[int]] = {}       # id → [wins, games]

    # ── Adding policies ──────────────────────────────────────────────────────

    def add(self, state_dict: Dict) -> None:
        """Add a policy snapshot (deep copy). Assigns a new stable ID."""
        # If at capacity, the oldest entry will be evicted — clean up its stats.
        if len(self.policies) == self.policies.maxlen and len(self._policy_ids) > 0:
            evict_id = self._policy_ids[0]
            self._policy_stats.pop(evict_id, None)

        policy_id = self._next_id
        self._next_id += 1

        self.policies.append(copy.deepcopy(state_dict))
        self._policy_ids.append(policy_id)
        self._policy_stats[policy_id] = [0, 0]  # [wins, games]

    def maybe_add(self, update_num: int, state_dict: Dict) -> bool:
        """Add policy if update_num % add_every == 0. Returns True if added."""
        if update_num > 0 and update_num % self.add_every == 0:
            self.add(state_dict)
            return True
        return False

    # ── Sampling ─────────────────────────────────────────────────────────────

    def sample(self) -> Tuple[str, Optional[Dict], int]:
        """Sample an opponent.

        Returns:
            (opponent_type, state_dict, policy_id)
            - opponent_type: 'random' or 'policy'
            - state_dict: None for random opponents
            - policy_id: -1 for random, else the stable int ID of the policy
        """
        r = np.random.random()
        if not self.policies or r < self.random_weight:
            return ("random", None, -1)
        if r < self.random_weight + self.heuristic_weight:
            return ("heuristic", None, -1)

        n = len(self.policies)
        ids = list(self._policy_ids)

        if self.use_pfsp:
            p = self._pfsp_weights(ids)
        else:
            p = self._linear_recency_weights(n)

        idx = np.random.choice(n, p=p)
        return ("policy", self.policies[idx], ids[idx])

    def _pfsp_weights(self, ids: List[int]) -> np.ndarray:
        """PFSP: p(i) ∝ w_i*(1-w_i) + ε, peaks at win_rate=0.5."""
        priorities = []
        for pid in ids:
            wins, games = self._policy_stats.get(pid, [0, 0])
            wr = wins / games if games > 0 else 0.5
            priorities.append(wr * (1.0 - wr) + self.pfsp_epsilon)
        p = np.array(priorities, dtype=np.float64)
        return p / p.sum()

    def _linear_recency_weights(self, n: int) -> np.ndarray:
        """Charlesworth-style: uniform base + linear bias toward recent 25."""
        linear_num = min(25, n)
        linear_prob = 0.5
        p = ((1 - linear_prob) / n) * np.ones(n)
        h = (2 * linear_prob) / (linear_num + 1)
        grad = h / linear_num
        for i in range(linear_num):
            p[n - 1 - i] += i * grad
        return p / p.sum()

    # ── Result reporting ──────────────────────────────────────────────────────

    def update_result(self, policy_id: int, win: int) -> None:
        """Record the outcome of a game against policy_id.

        Args:
            policy_id: The ID returned by sample() (-1 = random, ignored).
            win: 1 if the current (learning) policy won, 0 if it lost.
        """
        if policy_id < 0:
            return
        if policy_id in self._policy_stats:
            self._policy_stats[policy_id][0] += win
            self._policy_stats[policy_id][1] += 1

    # ── Inference helper ─────────────────────────────────────────────────────

    def get_policy_for_inference(self, state_dict: Dict, device: str = "cpu") -> Any:
        """Load state_dict into a reused policy instance for inference."""
        if self._build_policy_fn is None:
            raise RuntimeError("League needs build_policy_fn to use policy opponents")
        if self._policy_cache is None:
            self._policy_cache = self._build_policy_fn(device=device)
        self._policy_cache.load_state_dict(state_dict)
        self._policy_cache.eval()
        return self._policy_cache

    def set_build_policy_fn(self, fn) -> None:
        self._build_policy_fn = fn

    def __len__(self) -> int:
        return len(self.policies)
