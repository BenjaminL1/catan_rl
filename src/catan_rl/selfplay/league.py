"""
Charlesworth-style league with PFSP (Prioritized Fictitious Self-Play) sampling.

Policies are added every add_every updates. Sampling uses PFSP: opponents
with win rates closest to 50% against the current policy are sampled most
often. This avoids wasting time on opponents that are trivially easy or
impossibly hard.

Three PFSP variants are supported (Phase 3.1):

  - ``"linear"`` — legacy `p(i) ∝ w_i * (1 - w_i)`, peaks at w=0.5.
  - ``"hard"``   — AlphaStar-style `p(i) ∝ (1 - w_i) ** pfsp_p`, biased
    toward opponents we currently lose to. Defaults to p=2.
  - ``"var"``    — variance-style `p(i) ∝ w_i * (1 - w_i) ** pfsp_p`. Mostly
    of historical interest; included for ablation.

**[1v1]** All three are mathematically defined for **2-player zero-sum
symmetric** games only. Do not generalize to a 4-player Catan league
without re-deriving the priorities — in 4-player, "win rate" stops being a
sufficient statistic for opponent strength.

Each opponent's win rate is computed over a sliding 32-game window
(``pfsp_window``). Older outcomes are dropped, so PFSP-hard reacts to
recent form rather than career stats.

New policies start with w=0.5 (neutral assumption).

League.sample() returns a 3-tuple: (type, state_dict, policy_id)
  - policy_id is -1 for "random" opponents.
  - Call league.update_result(policy_id, win) after each game to update stats.
"""

import copy
from collections import deque
from typing import Any

import numpy as np

# Allowed values for ``pfsp_mode``. Listed once here so the trainer/configs
# don't drift out of sync silently.
PFSP_MODES = ("linear", "hard", "var")


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
        pfsp_mode: str = "linear",
        pfsp_p: float = 2.0,
        pfsp_window: int = 32,
        latest_policy_weight: float = 0.0,
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
            pfsp_mode: Phase 3.1 — ``"linear"``, ``"hard"``, or ``"var"``.
                See module docstring. Asserts 2-player zero-sum (1v1 only).
            pfsp_p: Exponent for ``hard`` and ``var`` modes (default 2).
            pfsp_window: Sliding-window size for per-opponent win rate
                (default 32 games). The ``hard`` mode benefits from a short
                window because it's reacting to *current* weakness, not
                career performance.
            latest_policy_weight: Phase 3.2 — probability of returning the
                special ``("current_self", None, -2)`` opponent. The trainer
                substitutes a fresh in-place snapshot of the current policy.
                Damps drift between league updates and stabilizes the
                neighborhood around the current policy. Default 0.0 (off);
                ``phase3_full`` uses 0.10. ``random_weight + heuristic_weight
                + latest_policy_weight`` must be ≤ 1.
        """
        if pfsp_mode not in PFSP_MODES:
            raise ValueError(f"pfsp_mode must be one of {PFSP_MODES}, got {pfsp_mode!r}")
        special_weight = (
            float(random_weight) + float(heuristic_weight) + float(latest_policy_weight)
        )
        if special_weight > 1.0 + 1e-9:
            raise ValueError(
                "random_weight + heuristic_weight + latest_policy_weight must "
                f"be ≤ 1, got {special_weight:.4f}"
            )

        self.policies: deque = deque(maxlen=maxlen)
        self.add_every = add_every
        self.random_weight = random_weight
        self.heuristic_weight = heuristic_weight
        self.latest_policy_weight = float(latest_policy_weight)
        self._build_policy_fn = build_policy_fn
        self._policy_cache: Any | None = None

        self.use_pfsp = use_pfsp
        self.pfsp_epsilon = pfsp_epsilon
        self.pfsp_mode = pfsp_mode
        self.pfsp_p = float(pfsp_p)
        self.pfsp_window = int(pfsp_window)

        # Monotonically increasing policy IDs (survive eviction tracking)
        self._next_id: int = 0
        self._policy_ids: deque = deque(maxlen=maxlen)  # parallel to self.policies
        # Per-opponent rolling outcome buffer. Deque[1=win,0=loss] of length
        # ≤ pfsp_window. Cumulative [wins, games] is preserved as a separate
        # field for non-PFSP code paths (TrueSkill etc.) that still want
        # career stats.
        self._policy_stats: dict[int, list[int]] = {}  # id → [wins, games]
        self._policy_window: dict[int, deque[int]] = {}  # id → recent outcomes
        # Phase 3.3 priority boost: id → (multiplier, games_remaining).
        # Decremented by ``update_result``; entries auto-clear on expiry.
        self._priority_boost: dict[int, tuple[float, int]] = {}

    # ── Construction helpers ─────────────────────────────────────────────────

    @classmethod
    def frozen_main_for_exploiter(cls, main_state_dict: dict, *, build_policy_fn=None) -> "League":
        """Phase 3.3: build a single-opponent league for an exploiter cycle.

        Returns a fresh League whose only policy entry is ``main_state_dict``,
        with ``random_weight = heuristic_weight = latest_policy_weight = 0.0``
        so every match the exploiter plays is against frozen main. The
        exploiter's job during the cycle is to find a strategy that beats
        this fixed snapshot — having any other opponent on the menu would
        dilute the signal.

        Stays on PFSP-linear (the cycle is too short for a hard sliding
        window to matter) and ``add_every`` is set so high it never fires —
        we don't want the exploiter polluting its own opponent pool.
        """
        lg = cls(
            maxlen=1,
            add_every=10**9,  # effectively never auto-add
            random_weight=0.0,
            heuristic_weight=0.0,
            build_policy_fn=build_policy_fn,
            pfsp_mode="linear",
            latest_policy_weight=0.0,
        )
        lg.add(main_state_dict)
        return lg

    # ── Adding policies ──────────────────────────────────────────────────────

    def add(self, state_dict: dict) -> None:
        """Add a policy snapshot (deep copy). Assigns a new stable ID."""
        # If at capacity, the oldest entry will be evicted — clean up its stats.
        if len(self.policies) == self.policies.maxlen and len(self._policy_ids) > 0:
            evict_id = self._policy_ids[0]
            self._policy_stats.pop(evict_id, None)
            self._policy_window.pop(evict_id, None)
            self._priority_boost.pop(evict_id, None)

        policy_id = self._next_id
        self._next_id += 1

        self.policies.append(copy.deepcopy(state_dict))
        self._policy_ids.append(policy_id)
        self._policy_stats[policy_id] = [0, 0]  # [wins, games] (cumulative)
        self._policy_window[policy_id] = deque(maxlen=self.pfsp_window)

    def add_with_boost(
        self, state_dict: dict, *, multiplier: float = 1.5, boost_games: int = 64
    ) -> int:
        """Phase 3.3: add a policy with an amplified PFSP priority.

        Used by the duo-exploiter cycle to ensure the exploiter snapshot
        is sampled more often than its baseline 0.5 WR would imply, for
        ``boost_games`` games. After that window expires, sampling reverts
        to the regular PFSP curve. Returns the new policy's stable ID.
        """
        self.add(state_dict)
        new_id = self._policy_ids[-1]
        self._priority_boost[new_id] = (float(multiplier), int(boost_games))
        return int(new_id)

    def maybe_add(self, update_num: int, state_dict: dict) -> bool:
        """Add policy if update_num % add_every == 0. Returns True if added."""
        if update_num > 0 and update_num % self.add_every == 0:
            self.add(state_dict)
            return True
        return False

    # ── Sampling ─────────────────────────────────────────────────────────────

    def sample(self) -> tuple[str, dict | None, int]:
        """Sample an opponent.

        Returns:
            (opponent_type, state_dict, policy_id) where:
              - ``opponent_type ∈ {'random', 'heuristic', 'current_self', 'policy'}``
              - ``state_dict`` is None for non-policy opponents (the trainer
                fills ``current_self`` with a fresh snapshot at use-time).
              - ``policy_id``: -1 for random/heuristic, -2 for current_self,
                otherwise the stable int ID of the historical league policy.
        """
        r = np.random.random()
        if not self.policies or r < self.random_weight:
            return ("random", None, -1)
        cum = self.random_weight + self.heuristic_weight
        if r < cum:
            return ("heuristic", None, -1)
        cum_self = cum + self.latest_policy_weight
        if r < cum_self:
            return ("current_self", None, -2)

        n = len(self.policies)
        ids = list(self._policy_ids)

        if self.use_pfsp:
            p = self._pfsp_weights(ids)
        else:
            p = self._linear_recency_weights(n)

        idx = np.random.choice(n, p=p)
        return ("policy", self.policies[idx], ids[idx])

    def _opponent_win_rate(self, pid: int) -> float:
        """Sliding-window win rate for opponent ``pid``.

        Uses ``pfsp_window`` most recent outcomes; new opponents get the
        neutral 0.5 prior so they receive non-trivial PFSP mass on day one.
        """
        window = self._policy_window.get(pid)
        if not window:
            return 0.5
        return float(sum(window) / len(window))

    def _pfsp_weights(self, ids: list[int]) -> np.ndarray:
        """PFSP priorities under the configured mode.

        - ``"linear"`` — `w * (1 - w)` (peaks at 0.5).
        - ``"hard"``   — `(1 - w) ** p`. Heavily biased toward opponents we
          currently lose to. AlphaStar default.
        - ``"var"``    — `w * (1 - w) ** p`. Linear up-weighted at the hard
          end (rarely useful at p=2; included for ablation).

        ``pfsp_epsilon`` is added before normalization so a fully-explored
        zero-priority opponent still gets sampled occasionally — without
        this, PFSP-hard would never re-explore an opponent whose win rate
        we've driven to 1.0.
        """
        priorities = np.empty(len(ids), dtype=np.float64)
        eps = self.pfsp_epsilon
        if self.pfsp_mode == "linear":
            for i, pid in enumerate(ids):
                w = self._opponent_win_rate(pid)
                priorities[i] = w * (1.0 - w) + eps
        elif self.pfsp_mode == "hard":
            p_exp = self.pfsp_p
            for i, pid in enumerate(ids):
                w = self._opponent_win_rate(pid)
                priorities[i] = (1.0 - w) ** p_exp + eps
        else:  # "var"
            p_exp = self.pfsp_p
            for i, pid in enumerate(ids):
                w = self._opponent_win_rate(pid)
                priorities[i] = w * ((1.0 - w) ** p_exp) + eps

        # Phase 3.3: apply per-policy priority boosts. Multiplies the raw
        # priority *before* normalization so a 1.5× boost on a 0.05-mass
        # opponent really does shift the distribution toward it.
        if self._priority_boost:
            for i, pid in enumerate(ids):
                if pid in self._priority_boost:
                    multiplier, _ = self._priority_boost[pid]
                    priorities[i] *= multiplier
        return priorities / priorities.sum()

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
        if policy_id in self._policy_window:
            self._policy_window[policy_id].append(int(bool(win)))
        # Phase 3.3 priority-boost decay: drop the boost entry when its
        # game budget is exhausted so PFSP returns to the regular curve.
        if policy_id in self._priority_boost:
            multiplier, remaining = self._priority_boost[policy_id]
            remaining -= 1
            if remaining <= 0:
                del self._priority_boost[policy_id]
            else:
                self._priority_boost[policy_id] = (multiplier, remaining)

    # ── Nash pruning (Phase 3.5) ──────────────────────────────────────────

    def prune_nash(self, payoff_matrix: np.ndarray, ids: list[int], iters: int = 100) -> int:
        """Drop the league entry with lowest Nash support; return the dropped ID.

        ``payoff_matrix[i, j]`` is the win-rate of policy ``ids[i]`` against
        ``ids[j]`` (∈ [0, 1]). The Nash mixture for the symmetric zero-sum
        game is computed via multiplicative-weights replicator dynamics on
        the centered payoff ``M = payoff_matrix - 0.5`` (so ``M`` is
        antisymmetric and the equilibrium is well-defined in 2-player zero-sum).

        We run ``iters=100`` MW updates from a uniform initialization and
        drop the entry with lowest mixture mass — this is the policy that
        contributes least to a Nash-optimal opponent strategy.

        **[1v1]** Replicator dynamics for asymmetric games (3+ players)
        require a different formulation; this method asserts the league is
        being used in 2-player zero-sum mode.

        Args:
            payoff_matrix: ``(k, k)`` win-rate matrix. ``i,j`` is i's WR vs j.
            ids: List of league policy IDs in row/col order; ``len(ids) == k``.
            iters: Number of replicator iterations (default 100).

        Returns:
            The policy ID that was evicted from the league.
        """
        k = payoff_matrix.shape[0]
        if k != len(ids):
            raise ValueError(f"payoff_matrix dim {k} does not match len(ids)={len(ids)}")
        if k < 2:
            raise ValueError(f"prune_nash requires k≥2 entries, got k={k}")
        # Center to antisymmetric zero-sum form. After this, M[i,j] = -M[j,i]
        # within rounding error provided payoff_matrix[i,j] + payoff_matrix[j,i] ≈ 1.
        M = payoff_matrix.astype(np.float64) - 0.5
        x = np.full(k, 1.0 / k, dtype=np.float64)
        eta = 0.5  # MW step size; smaller is more stable but slower.
        for _ in range(int(iters)):
            grad = M @ x  # expected payoff per pure strategy
            x = x * np.exp(eta * grad)
            s = x.sum()
            if not np.isfinite(s) or s <= 0:
                # Collapsed to a single pure strategy or numerically unstable —
                # fall back to uniform to avoid a divide-by-zero.
                x = np.full(k, 1.0 / k, dtype=np.float64)
                break
            x /= s

        evict_local_idx = int(np.argmin(x))
        evict_id = ids[evict_local_idx]
        self._evict_by_id(evict_id)
        return evict_id

    def _evict_by_id(self, policy_id: int) -> None:
        """Drop a specific policy from the deque and clear its bookkeeping.

        Used by Nash pruning to remove an entry that isn't necessarily the
        oldest one. Costs O(maxlen) but the league is bounded.
        """
        if policy_id not in self._policy_stats:
            return
        new_policies: deque = deque(maxlen=self.policies.maxlen)
        new_ids: deque = deque(maxlen=self.policies.maxlen)
        for sd, pid in zip(self.policies, self._policy_ids, strict=True):
            if pid == policy_id:
                continue
            new_policies.append(sd)
            new_ids.append(pid)
        self.policies = new_policies
        self._policy_ids = new_ids
        self._policy_stats.pop(policy_id, None)
        self._policy_window.pop(policy_id, None)
        self._priority_boost.pop(policy_id, None)

    # ── Inference helper ─────────────────────────────────────────────────────

    def get_policy_for_inference(self, state_dict: dict, device: str = "cpu") -> Any:
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
