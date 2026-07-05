"""Minimal opponent league for the v2 PPO training loop.

Phase 6 of the v2 training-infra build-out. Two layered jobs:

1. **Opponent diversity at rollout time.** :meth:`League.build_env_opponent_mix`
   returns a list of ``opponent_type`` strings of length ``n_envs``
   based on the configured weights. The vec env wires this into its
   per-env ``env_kwargs_list`` at construction.

2. **Snapshot retention + sampling.** :meth:`League.add_snapshot` stores
   past policy state dicts in a bounded ``deque``;
   :meth:`League.build_env_opponent_assignments` hands out
   :class:`OpponentAssignment`\\ s carrying a stable ``snapshot_id`` so the
   vec env can instantiate a frozen-policy opponent inside ``CatanEnv``
   (US1, T017). An empty/evicted pool falls back to a non-snapshot kind
   (FR-011) rather than erroring.

Why now: opponent diversity is a known lever for PPO self-play
stability, but the Phase-3 features (PFSP-hard / Nash pruning /
TrueSkill) are non-trivial to validate and the Phase C.0 plateau
diagnostic suggests the v2 plateau is probably value-learning, not
opponent diversity. Ship the minimal layer; add complexity only when
the loss curve says it's needed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from catan_rl.ppo.arguments import LeagueConfig

# ---------------------------------------------------------------------------
# Opponent kinds
# ---------------------------------------------------------------------------

OPPONENT_KIND_RANDOM = "random"
OPPONENT_KIND_HEURISTIC = "heuristic"
OPPONENT_KIND_SNAPSHOT = "snapshot"
"""Opponent kind for a frozen past-policy snapshot. The vec env resolves the
assignment's ``snapshot_id`` via :meth:`League.peek_by_id` and injects a
``FrozenSnapshotOpponent`` into ``CatanEnv`` (US1)."""

# PFSP: minimum per-snapshot weight under the "hard" curve, so a fully-beaten
# opponent (p_hat=1) keeps a small non-zero share (no starvation; FR-008).
_PFSP_WEIGHT_FLOOR = 0.05

_KNOWN_KINDS = (OPPONENT_KIND_RANDOM, OPPONENT_KIND_HEURISTIC, OPPONENT_KIND_SNAPSHOT)


@dataclass(frozen=True)
class OpponentAssignment:
    """Per-environment opponent assignment for one rollout (T007).

    ``kind`` is one of ``_KNOWN_KINDS``. ``snapshot_id`` is set iff
    ``kind == OPPONENT_KIND_SNAPSHOT`` — the stable league id (NOT a deque
    index; resolve via :meth:`League.peek_by_id`). The vec env threads this
    into each env's next ``reset`` for self-play.
    """

    kind: str
    snapshot_id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KNOWN_KINDS:
            raise ValueError(f"unknown opponent kind: {self.kind!r}")
        if (self.kind == OPPONENT_KIND_SNAPSHOT) != (self.snapshot_id is not None):
            raise ValueError(
                "snapshot_id must be set iff kind == 'snapshot' "
                f"(kind={self.kind!r}, snapshot_id={self.snapshot_id!r})"
            )


# ---------------------------------------------------------------------------
# Snapshot entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeagueSnapshot:
    """One past-policy snapshot in the league pool.

    ``state_dict`` is the raw torch state dict, cloned **to CPU** on
    insert so (a) the league owns its own tensors and the trainer can
    free the active policy's memory, and (b) the snapshot pool doesn't
    accumulate hundreds of MB on the training device (1.4M params *
    fp32 * maxlen=100 = ~560 MB if left on GPU).
    ``snapshot_id`` is a process-monotonic integer that is **stable
    across FIFO eviction** — the deque's positional indices shift on
    each ``append`` once ``maxlen`` is hit, so any downstream consumer
    holding an id MUST look the snapshot up via
    :meth:`League.peek_by_id` rather than ``peek_snapshot(idx)``.
    ``update_idx`` is the global PPO update at which this snapshot
    was taken. ``metadata`` is a free-form dict for any extra TB
    scalars (e.g., wr-vs-heuristic at the time of snapshot).
    """

    state_dict: Mapping[str, Any]
    update_idx: int
    snapshot_id: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# League config re-export (lives in :mod:`catan_rl.ppo.arguments` so it
# round-trips through YAML alongside the other sub-configs).
# ---------------------------------------------------------------------------

__all__ = [
    "OPPONENT_KIND_HEURISTIC",
    "OPPONENT_KIND_RANDOM",
    "OPPONENT_KIND_SNAPSHOT",
    "League",
    "LeagueConfig",
    "LeagueSnapshot",
]


# ---------------------------------------------------------------------------
# League
# ---------------------------------------------------------------------------


class League:
    """Bounded pool of past-policy snapshots + opponent-kind sampler.

    A single instance lives on the trainer process; the vec env reads
    the opponent mix at construction time only (no mid-rollout opponent
    swap in Phase 6). Snapshot mutations (``add_snapshot``) happen on
    the trainer side after each PPO update; consumers reading via
    :meth:`peek_snapshot` see the most-recently-added pool, so a
    snapshot added on update K is visible by update K+1.
    """

    def __init__(self, cfg: LeagueConfig) -> None:
        self.cfg = cfg
        self._snapshots: deque[LeagueSnapshot] = deque(maxlen=cfg.maxlen)
        # Monotonic id sequence — stays valid across FIFO eviction so
        # vec-env consumers can hold a stable handle to a specific
        # snapshot even after the deque rolls over.
        self._next_snapshot_id: int = 0
        # The permanent, non-evicted anchor (set via set_anchor). It lives
        # OUTSIDE the deque so FIFO eviction never touches it, and carries a
        # stable id resolvable through peek_by_id like any snapshot.
        self._anchor: LeagueSnapshot | None = None
        # PFSP per-opponent win-rate store: snapshot_id -> [p_hat (EMA), games].
        # p_hat is a recency-weighted estimate of the LEARNER's win rate vs that
        # snapshot; games gates cold-start. Pruned when a pool snapshot is
        # evicted; the anchor's entry persists with the anchor.
        self._opp_stats: dict[int, list[float]] = {}
        # Auto-re-anchor promotion bookkeeping (additive; round-trips via the
        # checkpoint's anchor dict). All inert unless cfg.auto_reanchor_enabled.
        self._reanchor_streak: int = 0  # consecutive qualifying checks so far
        self._last_promote_update: int = -1  # update_idx of last promotion (-1=never)
        self._n_promotions: int = 0  # monotonic count (also a TB scalar)
        # Sliding window of the learner's last ``auto_reanchor_min_games``
        # outcomes vs the CURRENT anchor (1.0=win). This — not the PFSP EMA —
        # is the promotion-decision statistic: an alpha=0.1 EMA has effective
        # sample size (2-a)/a ~= 19 games regardless of how many were recorded
        # (stationary SD ~0.11), giving a true-0.55 learner ~24% odds of
        # clearing a 0.63 bar on any single check (audit 2026-07). The full
        # window has N_eff == min_games. Cleared whenever the anchor changes;
        # NOT checkpointed — after a resume the gate simply waits for a fresh
        # window (~min_games anchor games, a few updates at 128 envs), which
        # is conservative and keeps the checkpoint schema unchanged.
        self._anchor_window: deque[float] = deque(maxlen=max(1, cfg.auto_reanchor_min_games))

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def add_snapshot(
        self,
        state_dict: Mapping[str, Any],
        *,
        update_idx: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Clone ``state_dict`` to CPU and append to the pool.

        Returns the assigned ``snapshot_id`` — opaque integer that the
        caller can later pass to :meth:`peek_by_id` to retrieve this
        specific snapshot regardless of how many evictions have
        happened since.

        Why CPU: callers typically pass ``self.policy.state_dict()``
        which returns *device-aliased* tensors. Without a clone the
        league would silently mutate "past" snapshots every PPO
        update. Cloning to CPU also keeps the pool from accumulating
        hundreds of MB on the training device (1.4M params * fp32 *
        maxlen=100 = ~560 MB if left on GPU).
        """
        snapshot_id = self._next_snapshot_id
        self._next_snapshot_id += 1
        # FIFO eviction is silent on a full deque; capture the id that is about
        # to drop so its PFSP win-rate entry is pruned (store stays <= maxlen+1).
        evicted_id: int | None = None
        if len(self._snapshots) == self._snapshots.maxlen and self._snapshots.maxlen:
            evicted_id = self._snapshots[0].snapshot_id
        self._snapshots.append(
            LeagueSnapshot(
                state_dict=self._clone_to_cpu(state_dict),
                update_idx=update_idx,
                snapshot_id=snapshot_id,
                metadata=dict(metadata or {}),
            )
        )
        if evicted_id is not None:
            self._opp_stats.pop(evicted_id, None)
        return snapshot_id

    @staticmethod
    def _clone_to_cpu(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for k, v in state_dict.items():
            if hasattr(v, "detach") and hasattr(v, "to"):
                cloned[k] = v.detach().to("cpu", copy=True)
            else:
                cloned[k] = v
        return cloned

    def set_anchor(
        self,
        state_dict: Mapping[str, Any],
        *,
        update_idx: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Install a PERMANENT, non-evicted reference opponent (the frozen
        anchor) and return its stable ``snapshot_id``.

        The anchor lives outside the FIFO pool so it is never evicted;
        ``build_env_opponent_assignments`` reserves ``cfg.anchor_weight`` of the
        env assignments for it. It resolves through :meth:`peek_by_id` exactly
        like a pool snapshot, so the snapshot-opponent driver needs no special
        casing. Replaces any prior anchor.
        """
        snapshot_id = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._anchor = LeagueSnapshot(
            state_dict=self._clone_to_cpu(state_dict),
            update_idx=update_idx,
            snapshot_id=snapshot_id,
            metadata=dict(metadata or {}),
        )
        # New anchor -> the promotion-decision window starts empty (its
        # outcomes were against the OLD anchor).
        self._anchor_window.clear()
        return snapshot_id

    def anchor_id(self) -> int | None:
        """Stable id of the frozen anchor, or ``None`` if no anchor is set."""
        return None if self._anchor is None else self._anchor.snapshot_id

    def maybe_promote_anchor(
        self, current_state_dict: Mapping[str, Any], *, update_idx: int
    ) -> int | None:
        """Auto-re-anchor IN-PROCESS when the learner has durably outgrown the
        frozen anchor: demote the old anchor into the PFSP pool, then install a
        frozen CPU snapshot of the current learner as the new anchor (fresh id,
        empty EMA). Returns the new anchor id on promotion, else ``None``.

        OFF (``auto_reanchor_enabled=False``) returns ``None`` on the FIRST line
        WITHOUT touching any state, so the call is byte-identical to absent.

        Trigger (all must hold): cooldown elapsed since the last promotion; the
        anchor-outcome window is FULL (``min_games`` outcomes vs the current
        anchor — a window-short check SKIPS without resetting the streak); the
        window's mean WR is STRICTLY above ``winrate_threshold`` (a
        sub-threshold check RESETS the streak); and the qualifying streak has
        reached ``sustained_checks``. The strict threshold above the WR
        oscillation band plus the streak debounce stop a transient up-wobble
        from thrashing promotions.

        The decision statistic is the sliding-window mean (N_eff ==
        ``min_games``; SD ~= sqrt(p(1-p)/min_games)), NOT the PFSP EMA whose
        N_eff is ~19 at alpha=0.1 (audit 2026-07: the EMA gave a true-0.55
        learner roughly coin-flip odds of a spurious promotion over a long
        run). Same bar semantics as the validated v8 recipe (window mean vs
        threshold), just a real sample size behind it.
        """
        cfg = self.cfg
        if not cfg.auto_reanchor_enabled:
            return None
        aid = self.anchor_id()
        if aid is None or self._anchor is None:
            return None
        # Cooldown gate: a freshly installed anchor must have time to start losing.
        if (
            self._last_promote_update >= 0
            and (update_idx - self._last_promote_update) < cfg.auto_reanchor_cooldown_updates
        ):
            return None
        p, g = self.anchor_window_stats()
        # Window-short gate: a "not enough data yet" SKIP — does NOT reset the streak.
        if g < cfg.auto_reanchor_min_games:
            return None
        # Qualifying check (STRICT >). A sub-threshold check RESETS the streak.
        if p > cfg.auto_reanchor_winrate_threshold:
            self._reanchor_streak += 1
        else:
            self._reanchor_streak = 0
            return None
        if self._reanchor_streak < cfg.auto_reanchor_sustained_checks:
            return None
        # FIRE: demote the outgoing anchor into the pool (anti-forgetting — it is
        # the second-strongest reference; PFSP keeps it drawable), then install
        # the current learner as the fresh frozen anchor.
        old = self._anchor
        self.add_snapshot(
            old.state_dict,
            update_idx=old.update_idx,
            metadata={
                **old.metadata,
                "demoted_from_anchor_id": old.snapshot_id,
                "demoted_at_update": update_idx,
            },
        )
        new_id = self.set_anchor(
            current_state_dict,
            update_idx=update_idx,
            metadata={
                "auto_reanchored": True,
                "promotion_index": self._n_promotions + 1,
                "prev_anchor_id": old.snapshot_id,
            },
        )
        # The old anchor id is now neither the anchor nor a pool id (the demoted
        # copy got a fresh id), so prune its orphaned EMA entry — mirrors the
        # FIFO eviction prune in add_snapshot.
        self._opp_stats.pop(old.snapshot_id, None)
        self._reanchor_streak = 0
        self._last_promote_update = update_idx
        self._n_promotions += 1
        return new_id

    # ------------------------------------------------------------------
    # PFSP win-rate tracking + weighting
    # ------------------------------------------------------------------

    def record_outcome(self, snapshot_id: int, *, agent_won: bool) -> None:
        """Record one finished game's outcome (learner win/loss) vs a snapshot,
        updating its recency-weighted (EMA) win-rate estimate. No-op for ids
        that are neither in the pool nor the anchor (defensive against stale
        ids whose snapshot was evicted)."""
        if snapshot_id not in self.snapshot_ids():
            return
        won = 1.0 if agent_won else 0.0
        st = self._opp_stats.get(snapshot_id)
        if st is None:
            self._opp_stats[snapshot_id] = [won, 1.0]  # seed EMA with first outcome
        else:
            a = self.cfg.pfsp_ema_alpha
            st[0] = (1.0 - a) * st[0] + a * won
            st[1] += 1.0
        if self._anchor is not None and snapshot_id == self._anchor.snapshot_id:
            self._anchor_window.append(won)

    def opponent_win_rate(self, snapshot_id: int) -> tuple[float, int]:
        """``(p_hat, games)`` for a snapshot; ``(0.0, 0)`` if unseen."""
        st = self._opp_stats.get(snapshot_id)
        return (0.0, 0) if st is None else (float(st[0]), int(st[1]))

    def anchor_window_stats(self) -> tuple[float, int]:
        """``(mean_wr, n)`` over the sliding window of the learner's last
        ``auto_reanchor_min_games`` outcomes vs the CURRENT anchor —
        the promotion-decision statistic. ``(0.0, 0)`` when empty."""
        n = len(self._anchor_window)
        if n == 0:
            return (0.0, 0)
        return (float(sum(self._anchor_window)) / n, n)

    def opponent_stats_state(self) -> dict[int, tuple[float, int]]:
        """Serialisable PFSP state for the checkpoint, pruned to live ids."""
        live = self.snapshot_ids()
        return {sid: (float(p), int(g)) for sid, (p, g) in self._opp_stats.items() if sid in live}

    def load_opponent_stats(self, state: Mapping[int, tuple[float, int]]) -> None:
        """Restore PFSP win-rate state on resume (replaces any current)."""
        self._opp_stats = {int(sid): [float(p), float(g)] for sid, (p, g) in state.items()}

    def _pfsp_pool_weights(self, ids: list[int]) -> np.ndarray:
        """Per-id sampling weight for the ``"hard"`` curve: the cold-start
        weight if under ``pfsp_min_games``, else ``max(floor, (1 - p_hat)**k)``.
        Equal inputs -> equal outputs (uniform); finite at p in {0, 1}."""
        k = self.cfg.pfsp_k
        min_games = self.cfg.pfsp_min_games
        cold = self.cfg.pfsp_cold_start_weight
        w = np.empty(len(ids), dtype=np.float64)
        for j, sid in enumerate(ids):
            p, g = self.opponent_win_rate(sid)
            w[j] = cold if g < min_games else max(_PFSP_WEIGHT_FLOOR, (1.0 - p) ** k)
        return w

    def selfplay_diagnostics(self) -> dict[str, float]:
        """Lightweight self-play health metrics for TB/jsonl (no extra inference):
        the anchor's win-rate (the strength-vs-frozen-reference signal) and PFSP
        pool concentration (sampling entropy + p_hat spread). Returns only the
        keys that are currently meaningful — empty when nothing to report."""
        out: dict[str, float] = {}
        aid = self.anchor_id()
        if aid is not None:
            p, g = self.opponent_win_rate(aid)
            if g > 0:
                out["anchor_winrate"] = p
                out["anchor_games"] = float(g)
        pool_ids = [s.snapshot_id for s in self._snapshots]
        if self.cfg.pfsp_enabled and self.cfg.pfsp_curve == "hard" and len(pool_ids) >= 2:
            w = self._pfsp_pool_weights(pool_ids)
            probs = w / w.sum()
            nz = probs[probs > 0]
            ent = float(-(nz * np.log(nz)).sum())
            out["pfsp_sampling_entropy"] = ent / float(np.log(len(pool_ids)))  # 1=uniform
            out["pfsp_pool_effective_n"] = float(np.exp(ent))
            phats = [
                self.opponent_win_rate(sid)[0]
                for sid in pool_ids
                if self.opponent_win_rate(sid)[1] > 0
            ]
            if phats:
                out["pfsp_p_hat_min"] = float(min(phats))
                out["pfsp_p_hat_median"] = float(np.median(phats))
                out["pfsp_p_hat_max"] = float(max(phats))
        if self.cfg.auto_reanchor_enabled:
            out["reanchor_streak"] = float(self._reanchor_streak)
            wr, n = self.anchor_window_stats()
            if n > 0:
                # The actual promotion-decision statistic (window mean), next
                # to the legacy EMA ``anchor_winrate`` (kept — TB scalars are
                # append-only).
                out["anchor_wr_window"] = wr
                out["anchor_window_games"] = float(n)
            out["anchor_promotions_total"] = float(self._n_promotions)
        return out

    def n_snapshots(self) -> int:
        return len(self._snapshots)

    def snapshot_ids(self) -> set[int]:
        """Stable ids currently resolvable (pool + anchor) — used by the
        self-play opponent cache to prune evicted snapshots. The anchor is
        always included so its cached policy is never pruned."""
        ids = {s.snapshot_id for s in self._snapshots}
        if self._anchor is not None:
            ids.add(self._anchor.snapshot_id)
        return ids

    def peek_snapshot(self, idx: int) -> LeagueSnapshot:
        """Read the snapshot at *positional* index ``idx`` (0 = oldest,
        -1 = newest).

        WARNING: positional indices shift on FIFO eviction. Use
        :meth:`peek_by_id` for stable lookup across rollouts.
        """
        if not self._snapshots:
            raise IndexError("league has no snapshots")
        try:
            return self._snapshots[idx]
        except IndexError as e:
            raise IndexError(
                f"snapshot index {idx} out of range [0, {len(self._snapshots)})"
            ) from e

    def peek_by_id(self, snapshot_id: int) -> LeagueSnapshot | None:
        """Look up by stable monotonic id; returns ``None`` if evicted.

        This is the consumer-safe path for any code that holds onto a
        snapshot handle across multiple rollouts (Phase 8+ snapshot
        opponent). Positional ``peek_snapshot(idx)`` reshuffles on
        eviction and is reserved for tests + diagnostics. The frozen anchor
        resolves here too (it is never evicted).
        """
        if self._anchor is not None and self._anchor.snapshot_id == snapshot_id:
            return self._anchor
        for snap in self._snapshots:
            if snap.snapshot_id == snapshot_id:
                return snap
        return None

    def should_snapshot_this_update(self, update_idx: int) -> bool:
        """``True`` if this PPO update should add a fresh snapshot.

        Convention: snapshots happen on update indices that are
        *positive* multiples of ``add_snapshot_every_n_updates``. The
        update-0 case is explicitly skipped — at update 0 the policy
        is freshly random-initialised and capturing it would add a
        no-op opponent to the pool that survives until the first FIFO
        eviction (~maxlen * add_every updates later, ~400 updates at
        defaults). Reviewer-caught HIGH.
        """
        return update_idx > 0 and update_idx % self.cfg.add_snapshot_every_n_updates == 0

    # ------------------------------------------------------------------
    # Opponent-kind sampling
    # ------------------------------------------------------------------

    def build_env_opponent_mix(self, *, n_envs: int, rng: np.random.Generator) -> list[str]:
        """Return a list of ``opponent_type`` strings of length ``n_envs``.

        Each entry is one of ``"random"``, ``"heuristic"``, or
        ``"snapshot"``. The vec env wires this into
        ``env_kwargs_list[i]["opponent_type"]`` at construction.

        Sampling is independent per env (no stratification yet). The
        weights are renormalised over the kinds that are currently
        sample-eligible — for example, if ``snapshot_weight > 0`` but
        the pool is empty, snapshots are excluded and the remaining
        weights are renormalised over random + heuristic.
        """
        if n_envs <= 0:
            raise ValueError(f"n_envs must be > 0, got {n_envs}")

        kinds = []
        weights = []
        if self.cfg.random_weight > 0:
            kinds.append(OPPONENT_KIND_RANDOM)
            weights.append(self.cfg.random_weight)
        if self.cfg.heuristic_weight > 0:
            kinds.append(OPPONENT_KIND_HEURISTIC)
            weights.append(self.cfg.heuristic_weight)
        # Snapshots are sample-eligible only when the pool is non-empty (it is
        # empty during self-play warm-up); an evicted/empty pool simply
        # excludes the snapshot kind and renormalises over the rest (FR-011).
        if self.cfg.snapshot_weight > 0 and len(self._snapshots) > 0:
            kinds.append(OPPONENT_KIND_SNAPSHOT)
            weights.append(self.cfg.snapshot_weight)
        if not kinds:
            kinds = [OPPONENT_KIND_HEURISTIC]
            weights = [1.0]

        probs = np.asarray(weights, dtype=np.float64)
        probs = probs / probs.sum()
        idxs = rng.choice(len(kinds), size=n_envs, p=probs)
        return [kinds[int(i)] for i in idxs]

    def build_env_opponent_assignments(
        self, *, n_envs: int, rng: np.random.Generator
    ) -> list[OpponentAssignment]:
        """Like :meth:`build_env_opponent_mix` but returns full
        :class:`OpponentAssignment`\\ s — each ``snapshot`` entry carries a
        concrete, **stable** ``snapshot_id`` sampled uniformly from the current
        pool (resolve via :meth:`peek_by_id`; ``None`` if later evicted →
        caller falls back per FR-011). This is the consumer API for the vec env
        / training loop (T017); the string ``build_env_opponent_mix`` is kept
        for back-compat.
        """
        if n_envs <= 0:
            raise ValueError(f"n_envs must be > 0, got {n_envs}")
        # Self-contained weighted draw over categories. "pool" = a uniform draw
        # from the evicting snapshot pool; "anchor" = the permanent non-evicted
        # reference (kept at a guaranteed weight so the policy can't forget how
        # to beat it). Both map to a kind="snapshot" assignment carrying the
        # resolved id — the driver treats them identically. When anchor_weight
        # is 0 (default) this matches the prior pool/heuristic/random behaviour.
        labels: list[str] = []
        weights: list[float] = []
        if self.cfg.random_weight > 0:
            labels.append(OPPONENT_KIND_RANDOM)
            weights.append(self.cfg.random_weight)
        if self.cfg.heuristic_weight > 0:
            labels.append(OPPONENT_KIND_HEURISTIC)
            weights.append(self.cfg.heuristic_weight)
        if self.cfg.snapshot_weight > 0 and len(self._snapshots) > 0:
            labels.append("pool")
            weights.append(self.cfg.snapshot_weight)
        if self.cfg.anchor_weight > 0 and self._anchor is not None:
            labels.append("anchor")
            weights.append(self.cfg.anchor_weight)
        if not labels:
            labels = [OPPONENT_KIND_HEURISTIC]
            weights = [1.0]

        probs = np.asarray(weights, dtype=np.float64)
        probs = probs / probs.sum()
        draws = rng.choice(len(labels), size=n_envs, p=probs)
        pool_ids = [s.snapshot_id for s in self._snapshots]
        # PFSP: precompute the within-pool probability vector ONCE (constant over
        # the env loop). Only when enabled + "hard"; otherwise the per-env draw
        # stays the exact uniform ``rng.choice(pool_ids)`` call (byte-identical,
        # FR-005). Computed only if the pool category is actually eligible.
        pool_probs: np.ndarray | None = None
        if self.cfg.pfsp_enabled and self.cfg.pfsp_curve == "hard" and pool_ids:
            w = self._pfsp_pool_weights(pool_ids)
            pool_probs = w / w.sum()
        out: list[OpponentAssignment] = []
        for d in draws:
            label = labels[int(d)]
            if label == "pool":
                sid = (
                    int(rng.choice(pool_ids, p=pool_probs))
                    if pool_probs is not None
                    else int(rng.choice(pool_ids))
                )
                out.append(OpponentAssignment(kind=OPPONENT_KIND_SNAPSHOT, snapshot_id=sid))
            elif label == "anchor":
                assert self._anchor is not None  # guarded by the weight check above
                out.append(
                    OpponentAssignment(
                        kind=OPPONENT_KIND_SNAPSHOT, snapshot_id=self._anchor.snapshot_id
                    )
                )
            else:
                out.append(OpponentAssignment(kind=label))
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"League(n_snapshots={self.n_snapshots()}, "
            f"cfg=random={self.cfg.random_weight}, "
            f"heuristic={self.cfg.heuristic_weight}, "
            f"snapshot={self.cfg.snapshot_weight})"
        )
