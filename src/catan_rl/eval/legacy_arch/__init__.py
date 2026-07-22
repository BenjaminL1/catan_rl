"""Vendored v11-era (pre-pointer-arch-fork) policy architecture.

This package holds FAITHFUL COPIES of the ``catan_rl.policy`` modules as they
existed at the last commit before the pointer-arch fork (see
:mod:`catan_rl.eval.legacy_arch._provenance`). Its sole purpose is to let a
v11-era checkpoint — whose obs schema (``CURR_PLAYER_DIM = 54``, no global
block, no ``is_setup``) and action heads (FiLM corner / MLP edge+tile, no
aux-value head) predate the fork — be instantiated IN-PROCESS alongside the
current pointer-arch ``CatanPolicy`` for cross-architecture head-to-head
evaluation (:mod:`catan_rl.eval.cross_arch`).

The engine + board geometry are byte-identical across the fork, so the legacy
obs encoder built here consumes the SAME live game state as the current encoder
— each side simply projects that shared state into its own obs schema.

The five ``*.py`` modules besides ``_provenance`` are GENERATED (do not
hand-edit); regenerate with ``scripts/vendor_legacy_arch.py``. They are imported
lazily by :mod:`catan_rl.eval.cross_arch`, never at package load, so importing
this package is cheap and side-effect-free.
"""

from __future__ import annotations
