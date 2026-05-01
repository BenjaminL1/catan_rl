"""Shared pytest fixtures for catan_rl tests."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "train"
FROZEN_CHAMPION = CHECKPOINT_DIR / "checkpoint_07390040.pt"


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Reset all RNGs before each test for reproducibility."""
    random.seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        import torch

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    except ImportError:
        pass


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def frozen_champion_path() -> Path:
    """Path to the frozen champion checkpoint, or skip if missing."""
    if not FROZEN_CHAMPION.exists():
        pytest.skip(f"Frozen champion not found at {FROZEN_CHAMPION}")
    return FROZEN_CHAMPION
