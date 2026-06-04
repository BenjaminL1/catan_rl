"""Frozen anchor-policy wrapper for piKL.

The anchor is a policy network whose parameters are NEVER updated by
the training loop. Two safeguards ensure that:

1. Every parameter has ``requires_grad = False`` after wrapping.
2. ``AnchorPolicy.evaluate_actions`` runs inside ``torch.no_grad()``
   and detaches the returned tensors. A misuse like ``loss = anchor.
   evaluate_actions(...)["log_prob"].mean()`` produces a leaf tensor
   that backward() can't reach the anchor through.

The wrapper presents the same surface as :class:`catan_rl.policy.
network.CatanPolicy.evaluate_actions` — ``log_prob`` (B,),
``per_head_log_prob`` (B, 6), ``relevance`` (B, 6) — so the piKL
loss can consume both anchor and student outputs uniformly.

Wrapping an already-trained network: pass it to ``AnchorPolicy(net)``.
Loading from a Phase 8 checkpoint: use
:func:`catan_rl.algorithms.pikl.loader.load_pikl_anchor`.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn


class AnchorPolicyError(RuntimeError):
    """Raised when the wrapped module doesn't expose the surface piKL
    requires (``evaluate_actions(obs, action, masks) -> dict``)."""


class AnchorPolicy(nn.Module):
    """Wrap a policy module as a frozen piKL anchor.

    The wrapper:

    * Freezes every parameter (``requires_grad=False``).
    * Forces ``eval()`` mode so dropout / BN / LayerNorm running
      stats stay fixed.
    * Runs ``evaluate_actions`` inside ``torch.no_grad()`` and
      detaches outputs so the student's backward pass can't touch
      the anchor's compute graph.

    The wrapper itself is an ``nn.Module`` so it lives on a device
    and can be moved with ``.to(device)``. It is registered as a
    submodule via ``self._inner`` for state-dict round-trips through
    Phase 8's checkpoint loader.
    """

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        # Surface check before freezing — we want a clear error if the
        # caller hands in something that doesn't quack like a CatanPolicy.
        if not callable(getattr(policy, "evaluate_actions", None)):
            raise AnchorPolicyError(
                "AnchorPolicy requires the wrapped module to expose "
                "evaluate_actions(obs, action, masks); got "
                f"{type(policy).__name__}"
            )
        self._inner = policy
        # Freeze all params. New params added to ``_inner`` after this
        # call will NOT be frozen — callers shouldn't mutate the
        # wrapped module post-wrap.
        for p in self._inner.parameters():
            p.requires_grad_(False)
        # Eval mode: deterministic forward, no dropout, no BN running
        # stat updates. Even though ``no_grad`` blocks gradients, the
        # running-stat update path in BN is not gated by no_grad.
        self._inner.eval()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward-only evaluation under the anchor.

        Returns the same dict shape as
        :meth:`CatanPolicy.evaluate_actions` but with every tensor
        detached. Always runs in ``no_grad`` regardless of the
        ambient autograd state.
        """
        with torch.no_grad():
            # nn.Module.__getattr__ is typed `Tensor | Module`, so
            # mypy can't see the inner's evaluate_actions. The
            # AnchorPolicy ctor asserts the method exists.
            inner_eval = cast(Any, self._inner).evaluate_actions
            out = inner_eval(obs, action, masks)
        # Belt-and-suspenders detach so the returned tensors carry no
        # graph reference even if a future evaluate_actions
        # implementation forgets the no_grad ambient.
        return {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}

    # ------------------------------------------------------------------
    # nn.Module overrides — these are intentionally locked
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> AnchorPolicy:
        """Locked to ``eval()``. The frozen anchor must never enter
        train mode (which would flip BN running-stat updates back on).
        Calling ``train(True)`` is a no-op + the wrapped module stays
        in eval. Returning ``self`` matches the base signature."""
        super().train(False)
        self._inner.eval()
        return self

    def forward(self, *_args: Any, **_kwargs: Any) -> None:
        """Calling ``forward`` directly is an API error — the piKL
        consumer should call :meth:`evaluate_actions`. Raises so a
        misuse fails loudly instead of silently producing trunk
        features that the caller doesn't know are detached."""
        raise AnchorPolicyError(
            "Call AnchorPolicy.evaluate_actions(obs, action, masks), "
            "not forward(). The anchor exposes only the action-eval "
            "surface that the piKL loss needs."
        )

    def inner_state_dict(self) -> dict[str, Any]:
        """Return the wrapped policy's state_dict directly — no
        ``_inner.`` prefix.

        Phase 8's checkpoint save calls ``policy.state_dict()``. If a
        caller saves an ``AnchorPolicy`` directly via the default
        ``nn.Module.state_dict``, every key gets prefixed with
        ``_inner.`` and the saved file can't be loaded back into a
        bare ``CatanPolicy`` without a key-renaming dance.
        :meth:`inner_state_dict` is the safe way to grab the
        prefix-free dict for a "save the current anchor" debug
        snapshot::

            save_checkpoint(path, policy=anchor._inner, ...)
            # or equivalently for symmetry:
            torch.save(anchor.inner_state_dict(), path)

        Loading: instantiate a fresh ``CatanPolicy``, load the
        dict, wrap with :class:`AnchorPolicy` again. The Phase 9
        :func:`load_pikl_anchor` does exactly that.
        """
        return cast(Any, self._inner).state_dict()

    def load_inner_state_dict(self, state_dict: Any, strict: bool = True) -> Any:
        """Symmetric to :meth:`inner_state_dict` — load a bare
        ``CatanPolicy`` state-dict into the wrapped inner."""
        return cast(Any, self._inner).load_state_dict(state_dict, strict=strict)
