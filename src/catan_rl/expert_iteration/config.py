"""Config dataclasses for expert iteration (isolated — not on TrainConfig)."""

from __future__ import annotations

from dataclasses import dataclass

#: The v6 frontier base policy (the search teacher + the distillation warm-start).
V6_BASE_CKPT = "runs/train/selfplay_v6_20260611_065459/checkpoints/ckpt_000001499.pt"


@dataclass(frozen=True)
class SearchLabelConfig:
    """Generate search-labeled, BcDataset-compatible shards from search games."""

    out_dir: str
    base_ckpt: str = V6_BASE_CKPT
    sims_per_move: int = 50
    opponent: str = "heuristic"  # "heuristic" | "random" | "policy:<ckpt>"
    n_positions: int = 5000  # target non-forced labeled positions
    min_games: int = 2  # play at least this many games (so a by-game val split is valid)
    discount: float = 0.998
    max_turns: int = 400
    seed: int = 0

    def __post_init__(self) -> None:
        if self.sims_per_move <= 0:
            raise ValueError(f"sims_per_move must be > 0, got {self.sims_per_move}")
        if self.n_positions <= 0:
            raise ValueError(f"n_positions must be > 0, got {self.n_positions}")
        if self.min_games < 2:
            raise ValueError(
                f"min_games must be >= 2 (for a valid val split), got {self.min_games}"
            )
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be > 0, got {self.max_turns}")
        if not (self.opponent in ("heuristic", "random") or self.opponent.startswith("policy:")):
            raise ValueError(
                f"opponent must be 'heuristic', 'random', or 'policy:<ckpt>', got {self.opponent!r}"
            )


@dataclass(frozen=True)
class DistillConfig:
    """Distillation = a warm-started BC fine-tune on search-labeled shards."""

    data_dir: str
    out_dir: str
    init_ckpt: str = V6_BASE_CKPT  # warm-start weights (the round base)
    peak_lr: float = 5e-5  # low: nudging a warm-started net, not retraining (research D8)
    max_epochs: int = 5
    batch_size: int = 1024
    value_weight: float = 0.10
    belief_weight: float = 0.05
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.peak_lr <= 0:
            raise ValueError(f"peak_lr must be > 0, got {self.peak_lr}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {self.max_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
