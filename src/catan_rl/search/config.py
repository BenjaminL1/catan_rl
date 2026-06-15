"""``SearchConfig`` — the isolated config source-of-truth for search.

Deliberately NOT bolted onto ``ppo/arguments.py``'s ``TrainConfig`` (search is an
offline, training-path-independent feature). A frozen dataclass with eager
``__post_init__`` validation so a bad budget/exploration setting fails fast at
construction rather than mid-game.
"""

from __future__ import annotations

from dataclasses import dataclass

from catan_rl.search.value import VALUE_SQUASH_A, VALUE_SQUASH_B


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for a determinized PUCT-MCTS search.

    Exactly one of ``sims_per_move`` / ``time_budget_s`` must be set — they are
    the two mutually-exclusive budget modes (fixed sims vs anytime wall-clock).
    """

    #: Fixed simulation budget per move. Mutually exclusive with ``time_budget_s``.
    sims_per_move: int | None = 100
    #: Anytime wall-clock budget (seconds) per move. Mutually exclusive with sims.
    time_budget_s: float | None = None
    #: Number of determinizations (independent dice/opponent worlds) to average.
    n_determinizations: int = 1
    #: PUCT exploration constant (applied on the SQUASHED value scale, in (0,1)).
    c_puct: float = 1.5
    #: Value-head squash: P(win) = sigmoid(a * V + b). Fitted on peer games
    #: (Brier 0.149, ECE 0.039). Raw V exceeds [-1,1] for ~27% of states, which
    #: is why the leaf MUST be squashed before backup. Defaults are sourced from
    #: ``search.value`` (single SoT for the fitted constants); the MCTS layer
    #: threads these into ``value_from_obs(..., a=, b=)``.
    value_squash_a: float = VALUE_SQUASH_A
    value_squash_b: float = VALUE_SQUASH_B
    #: Progressive widening on the type head: at N visits, expand up to
    #: ceil(pw_c * N**pw_alpha) of the legal types (used in US2 hardening).
    pw_c: float = 1.0
    pw_alpha: float = 0.5
    #: Optional hard depth cut for the simulation (None = no cut).
    max_depth: int | None = None
    #: RNG seed — search + uplift eval are reproducible at a fixed seed (FR-006).
    seed: int = 0

    def __post_init__(self) -> None:
        if (self.sims_per_move is None) == (self.time_budget_s is None):
            raise ValueError(
                "exactly one of sims_per_move / time_budget_s must be set "
                f"(got sims_per_move={self.sims_per_move}, time_budget_s={self.time_budget_s})"
            )
        if self.sims_per_move is not None and self.sims_per_move <= 0:
            raise ValueError(f"sims_per_move must be > 0, got {self.sims_per_move}")
        if self.time_budget_s is not None and self.time_budget_s <= 0:
            raise ValueError(f"time_budget_s must be > 0, got {self.time_budget_s}")
        if self.n_determinizations < 1:
            raise ValueError(f"n_determinizations must be >= 1, got {self.n_determinizations}")
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be > 0, got {self.c_puct}")
        if self.value_squash_a <= 0:
            # A non-positive slope inverts the win-probability map (a confident
            # win -> P~0), silently breaking every backup. Guard it.
            raise ValueError(f"value_squash_a must be > 0, got {self.value_squash_a}")
        if self.pw_c <= 0:
            raise ValueError(f"pw_c must be > 0, got {self.pw_c}")
        if not 0.0 < self.pw_alpha <= 1.0:
            raise ValueError(f"pw_alpha must be in (0, 1], got {self.pw_alpha}")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1 or None, got {self.max_depth}")
