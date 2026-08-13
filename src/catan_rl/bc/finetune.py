"""Champion fine-tune with self-distillation anchoring (spec D5).

Installs the owner's hand-labeled openings into ``runs/anchors/ptr_v1_u500.pt``
without damaging its midgame.

The objective
-------------
``L = L_BC(human rows) + kl_coef · KL_anchor``

* ``L_BC`` is the ordinary BC loss over the human shard, with the **value loss
  zeroed per row** (``bc_loss(..., value_row_mask=...)``): a hand-labeled
  opening carries no outcome, so its ``z_disc`` is a filler zero and regressing
  toward it would teach the champion that every labeled opening is a dead draw.

* ``KL_anchor`` is the ONLINE self-distillation term the owner chose over
  freezing heads. The fine-tuned policy and a FROZEN copy of ``u500`` are both
  evaluated on freshly sampled NON-setup states, and the divergence between them
  is penalised. Online rather than offline because the offline form needs a
  stored per-head-distribution dataset that does not exist anywhere in ``bc/``
  today — building one is the largest hidden cost in this slice, and its absence
  is exactly what invites the silent simplification to hard-label self-BC that
  the spec calls a violation.

  **What the term actually measures.** The six heads are autoregressive: corner
  is conditioned on the action type, resource2 on the type and the chosen
  resource1. There is no tractable joint distribution to diverge, so this is the
  relevance-weighted sum of the six per-head CONDITIONAL KLs **at a single
  reference action context** — the legal action the anchor sampler walked the
  game with. That is an upper-bound-flavoured surrogate for the joint KL, not
  the joint KL, and it is named as such here and in
  :meth:`catan_rl.policy.heads.CatanActionHeads.masked_log_dists` so no later
  reader mistakes it for one.

  **Where the states come from** is not a detail either: they are rolled out
  from games whose setups are FORCED to the owner's labeled openings (see
  :mod:`catan_rl.bc.anchor_states`), so the anchor covers the distribution the
  fine-tune moves TOWARD rather than the one it leaves.

Epoch (D6)
----------
``ptr_v1_u500.pt`` carries no ruleset stamp, so it is **R0**. The candidate is
stamped R0 explicitly. Any R1 transition is the successor slice's explicit
decision, never a side effect of a default config.

Deliverables (D8)
-----------------
:func:`finetune` writes the candidate checkpoint AND a frozen copy designated
the **human-opening prior**. The successor self-play slice is contractually
required to carry a ready-to-enable KL anchor to that prior at setup nodes —
without it, the installed openings are expected to decay on contact with PPO.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from catan_rl.bc.anchor_states import sample_anchor_states
from catan_rl.bc.loader import BcDataset, bc_collate
from catan_rl.bc.loss import bc_loss
from catan_rl.policy import CatanPolicy
from catan_rl.policy.board_geometry import build_geometry

_HEAD_NAMES: tuple[str, ...] = ("type", "corner", "edge", "tile", "resource1", "resource2")


@dataclass
class FinetuneConfig:
    """Everything the fine-tune needs. No hidden defaults that change behaviour."""

    ckpt_path: Path
    """The champion to fine-tune. Production value: ``runs/anchors/ptr_v1_u500.pt``."""

    shard_dir: Path
    """Human-label BC shard, from ``catan_rl.labeling.to_shard.convert``."""

    labels_path: Path
    """The label JSONL the anchor states' FORCED openings are drawn from."""

    out_dir: Path
    """Receives ``candidate.pt``, ``human_opening_prior.pt`` and ``history.json``."""

    steps: int = 200
    batch_size: int = 32
    lr: float = 1e-5
    """Deliberately small. This is a surgical edit to a banked champion, not a
    training run; the anchor term bounds drift but does not license a large LR."""

    kl_coef: float = 1.0
    """Keep this MODERATE. Measured while building this module: raising it to
    10x-200x does not tighten the anchor, it destabilises the optimisation —
    with ``clip_grad_norm_(0.5)`` a huge KL coefficient consumes the whole
    gradient budget and held-out drift comes out WORSE than an unanchored run.
    At 1.0 the same setup cut held-out drift to ~0.19x the unanchored control.
    ``tests/unit/bc/test_finetune.py`` pins the effect at this default."""

    anchor_batch: int = 16
    n_anchor_states: int = 256
    anchor_refresh_every: int = 50
    """Anchor states are re-sampled this often, so the term is evaluated on
    FRESH states rather than a fixed set the fine-tune can overfit around.

    Not optional in practice: with refresh OFF and a small pool, the fine-tune
    drives the training KL down on those exact states while HELD-OUT drift grows
    — measured at ~3.2x an unanchored run. Setting this to 0 disables the
    refresh and reintroduces that failure."""

    belief_weight: float = 0.05
    seed: int = 0
    device: str = "cpu"
    """CPU by default per the repo device policy (eval is pinned to CPU; this is
    a few hundred small-batch steps, not a training run)."""

    history: dict[str, Any] = field(default_factory=dict)


def _build_policy(ckpt_path: Path, device: torch.device) -> CatanPolicy:
    from catan_rl.checkpoint import load_checkpoint

    policy = CatanPolicy()
    # Geometry BEFORE the device move + state-dict apply, matching
    # ``replay.player_factory``: the setter writes registered buffers that the
    # checkpoint then overwrites.
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy = policy.to(device)
    payload = load_checkpoint(ckpt_path, map_location=device)
    payload.apply_to_policy(policy, strict=True)
    return policy


def anchor_kl(
    trainable: CatanPolicy,
    frozen: CatanPolicy,
    batch: dict[str, Any],
) -> torch.Tensor:
    """Relevance-weighted sum of the six per-head conditional KLs.

    ``KL(fine-tuned ‖ frozen)`` — the fine-tuned policy is the one being held in
    place, so it is the distribution the expectation is taken under. Heads the
    reference action type does not use contribute nothing (their masks are
    all-False and ``masked_log_softmax`` would hand back a uniform placeholder —
    noise wearing the shape of an opinion).
    """
    out_ft = trainable.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    with torch.no_grad():
        out_fz = frozen.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    relevance = out_ft["relevance"]  # (B, 6)

    total = relevance.new_zeros(())
    for h_idx, name in enumerate(_HEAD_NAMES):
        log_p = out_ft[f"log_dist/{name}"]
        log_q = out_fz[f"log_dist/{name}"]
        # p*(log p - log q). Masked slots carry a large finite negative logit
        # (``heads._LOGIT_NEG_INF``), so p ≈ 0 there and the product vanishes
        # without any -inf arithmetic.
        per_row = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        rel = relevance[:, h_idx]
        total = total + (per_row * rel).sum() / rel.sum().clamp_min(1.0)
    return total


def _anchor_batch_tensors(
    states: list[dict[str, Any]], idx: np.ndarray, device: torch.device
) -> dict[str, Any]:
    obs_keys = states[0]["obs"].keys()
    mask_keys = states[0]["mask"].keys()
    obs: dict[str, torch.Tensor] = {}
    for key in obs_keys:
        arr = np.stack([np.asarray(states[i]["obs"][key]) for i in idx])
        obs[key] = torch.as_tensor(arr, device=device)
        if obs[key].dtype not in (torch.int64,):
            obs[key] = obs[key].float()
    mask = {
        key: torch.as_tensor(
            np.stack([np.asarray(states[i]["mask"][key], dtype=bool) for i in idx]),
            device=device,
        )
        for key in mask_keys
    }
    action = torch.as_tensor(
        np.stack([states[i]["action"] for i in idx]).astype(np.int64), device=device
    )
    return {"obs": obs, "mask": mask, "action": action}


def _shard_held_out_seeds(shard_dir: Path) -> tuple[int, ...]:
    """The ``game_seed`` s the converter withheld from ``shard_dir``.

    Empty for a shard built with ``held_out_frac=0`` (and for the pre-split
    manifests), which is the honest reading: nothing was withheld, so nothing
    needs excluding here — ``scripts/eval_setup_agreement.py`` is the place that
    refuses to gate such a shard.
    """
    manifest = Path(shard_dir) / "manifest.json"
    if not manifest.is_file():
        return ()
    payload = json.loads(manifest.read_text())
    return tuple(int(s) for s in payload.get("held_out_game_seeds", ()))


def _human_row_value_mask(batch: dict[str, Any]) -> torch.Tensor:
    """Every row of the human shard is value-less. Explicit, not implied."""
    return torch.zeros_like(batch["z_disc"])


def finetune(config: FinetuneConfig) -> dict[str, Any]:
    """Run the fine-tune and write the D8 deliverables. Returns the history dict."""
    from catan_rl.checkpoint import save_policy_only

    device = torch.device(config.device)
    # Seed EVERY generator the run transitively touches, not just torch: the
    # anchor rollouts go through the engine, whose ``StackedDice`` draws its seed
    # from the stdlib ``random`` and whose heuristic opponent uses the numpy
    # GLOBAL. Leaving those to ambient process state makes a checkpoint-producing
    # routine irreproducible depending on what ran before it.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    trainable = _build_policy(config.ckpt_path, device)
    trainable.train()
    frozen = _build_policy(config.ckpt_path, device)
    frozen.eval()
    frozen.requires_grad_(False)

    # The shard's manifest names the game_seeds the converter WITHHELD for the
    # D7 gate-1 measurement. They are excluded from the anchor rollouts too:
    # forcing a held-out opening into the anchor states would let it shape the
    # candidate through the KL term, and the gate would again be measuring a
    # position the fine-tune had seen.
    held_out_seeds = _shard_held_out_seeds(config.shard_dir)

    dataset = BcDataset(config.shard_dir, aug_prob=0.0, seed=config.seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        collate_fn=bc_collate,
        num_workers=0,
        drop_last=False,
    )
    optimizer = AdamW(trainable.parameters(), lr=config.lr, weight_decay=1e-4)

    anchor = sample_anchor_states(
        config.labels_path,
        n_states=config.n_anchor_states,
        rng=rng,
        exclude_game_seeds=held_out_seeds,
    )
    history: list[dict[str, float]] = []
    step = 0
    while step < config.steps:
        for batch in loader:
            if step >= config.steps:
                break
            if config.anchor_refresh_every and step and step % config.anchor_refresh_every == 0:
                anchor = sample_anchor_states(
                    config.labels_path,
                    n_states=config.n_anchor_states,
                    rng=rng,
                    exclude_game_seeds=held_out_seeds,
                )
            batch = _to_device(batch, device)
            out = trainable.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
            losses = bc_loss(
                policy_out=out,
                batch=batch,
                belief_weight=config.belief_weight,
                value_row_mask=_human_row_value_mask(batch),
            )
            idx = rng.choice(len(anchor), size=min(config.anchor_batch, len(anchor)), replace=False)
            kl = anchor_kl(trainable, frozen, _anchor_batch_tensors(anchor, idx, device))
            total = losses["total"] + config.kl_coef * kl

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), 0.5)
            optimizer.step()

            history.append(
                {
                    "step": float(step),
                    "total": float(total.item()),
                    "policy": float(losses["policy"].item()),
                    "value": float(losses["value"].item()),
                    "anchor_kl": float(kl.item()),
                }
            )
            step += 1

    config.out_dir.mkdir(parents=True, exist_ok=True)
    trainable.eval()
    # D6: stamp the epoch EXPLICITLY. ``checkpoint_ruleset`` reads this back, and
    # an absent stamp would be read as R0 by accident rather than by decision.
    stamped_config = {"rollout": {"ruleset": "R0"}, "source": "bc.finetune"}
    candidate = config.out_dir / "candidate.pt"
    save_policy_only(
        candidate,
        config=stamped_config,
        policy=trainable,
        update_idx=0,
        global_step=config.steps,
        metadata={
            "kind": "policy_only",
            "finetune": "human_openings",
            "base_ckpt": str(config.ckpt_path),
            "kl_coef": config.kl_coef,
        },
    )
    # D8 deliverable (iii): a FROZEN copy designated the human-opening prior. It
    # is a separate file on purpose — the candidate may be superseded, and the
    # successor self-play slice's KL anchor must keep pointing at these weights.
    prior = config.out_dir / "human_opening_prior.pt"
    save_policy_only(
        prior,
        config=stamped_config,
        policy=trainable,
        update_idx=0,
        global_step=config.steps,
        metadata={
            "kind": "policy_only",
            "role": "human_opening_prior",
            "base_ckpt": str(config.ckpt_path),
        },
    )
    out_history = {
        "steps": history,
        "candidate": str(candidate),
        "human_opening_prior": str(prior),
        "n_rows": len(dataset),
        "ruleset": "R0",
        "held_out_game_seeds": sorted(held_out_seeds),
    }
    (config.out_dir / "history.json").write_text(json.dumps(out_history, indent=2))
    return out_history


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = {k: v.to(device) for k, v in value.items()}
        else:
            out[key] = value.to(device)
    return out


def held_out_anchor_kl(
    candidate_path: Path,
    base_path: Path,
    labels_path: Path,
    *,
    n_states: int = 64,
    seed: int = 1234,
    device: str = "cpu",
    exclude_game_seeds: Sequence[int] = (),
) -> float:
    """Mean per-state anchor KL between a candidate and the base champion.

    Evaluated on anchor states sampled with a DIFFERENT seed than training used,
    so the number is a held-out drift measurement rather than the training term
    read back. Pass the shard's ``held_out_game_seeds`` to keep the same
    openings out of this measurement that the fine-tune kept out of training.
    """
    torch_device = torch.device(device)
    candidate = _build_policy(candidate_path, torch_device)
    candidate.eval()
    base = _build_policy(base_path, torch_device)
    base.eval()
    rng = np.random.default_rng(seed)
    states = sample_anchor_states(
        labels_path, n_states=n_states, rng=rng, exclude_game_seeds=exclude_game_seeds
    )
    with torch.no_grad():
        batch = _anchor_batch_tensors(states, np.arange(len(states)), torch_device)
        return float(anchor_kl(candidate, base, batch).item())


def setup_agreement(
    ckpt_path: Path,
    labels: list[dict[str, Any]],
    *,
    device: str = "cpu",
    opponent_kind: int | None = None,
) -> list[bool]:
    """Per-label top-1 settlement agreement for the policy at ``ckpt_path``.

    The position is rebuilt through the SAME converter path the training shard
    uses (:func:`catan_rl.labeling.to_shard.rows_for_label`), so an agreement
    number can never be measured on a differently-encoded state than the one the
    fine-tune trained on.

    **Opponent identity is stamped here, explicitly.** ``rows_for_label``
    returns PRE-duplication rows, whose ``opponent_kind`` is whatever
    :class:`~catan_rl.policy.obs_encoder.EnvObsState` defaults to —
    ``OPP_KIND_UNKNOWN``, which
    :data:`catan_rl.labeling.to_shard.DEFAULT_OPPONENT_KINDS` deliberately
    EXCLUDES from the shard. Reading the gate in that slice would measure the
    one opponent-id conditional the fine-tune never trains, and D4's own
    reasoning (a single-kind stamp "can leave the learned openings unexpressed
    under the other kind") says that number need not match. ``None`` therefore
    means ``OPP_KIND_HEURISTIC`` — a kind the shard DOES carry, and the one the
    D7 gate-2 full-game eval plays.
    """
    from catan_rl.labeling.to_shard import UNKNOWN_POLICY_ID, rows_for_label
    from catan_rl.policy.obs_schema import OPP_KIND_HEURISTIC

    kind = OPP_KIND_HEURISTIC if opponent_kind is None else int(opponent_kind)

    torch_device = torch.device(device)
    policy = _build_policy(ckpt_path, torch_device)
    policy.eval()

    out: list[bool] = []
    for label in labels:
        settle_row, _road_row = rows_for_label(label, game_id=0)
        obs = {
            key: torch.as_tensor(np.asarray(value), device=torch_device).unsqueeze(0)
            for key, value in settle_row.obs.items()
        }
        for key, value in obs.items():
            if value.dtype not in (torch.int64,):
                obs[key] = value.float()
            else:
                obs[key] = value.reshape(-1)
        obs["opponent_kind"] = torch.full((1,), kind, dtype=torch.int64, device=torch_device)
        obs["opponent_policy_id"] = torch.full(
            (1,), UNKNOWN_POLICY_ID, dtype=torch.int64, device=torch_device
        )
        mask = {
            key: torch.as_tensor(np.asarray(value, dtype=bool), device=torch_device).unsqueeze(0)
            for key, value in settle_row.mask.items()
        }
        action = torch.as_tensor(settle_row.action, device=torch_device).unsqueeze(0)
        with torch.no_grad():
            head_out = policy.evaluate_actions(obs, action, mask)
        top1 = head_out["log_dist/corner"].argmax(dim=-1).item()
        out.append(top1 == int(label["settlement_vertex"]))
    return out
