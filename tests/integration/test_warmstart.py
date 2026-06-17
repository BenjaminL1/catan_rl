"""Learner warm-start (`TrainConfig.init_policy_checkpoint`) — Spec 005 / S0.

The exploiter probe warm-starts the *learner* from a v2-lineage checkpoint (with a
FRESH optimizer), distinct from resume. These tests pin:
  1. With `init_policy_checkpoint` set, `build_training_state` loads those weights
     into the learner (a sentinel param matches the source checkpoint).
  2. With it `None` (default), the learner is fresh-init (sentinel does NOT match)
     — the additivity guarantee that the default path is unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from catan_rl.checkpoint.manager import save_checkpoint
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy
from catan_rl.ppo.arguments import TrainConfig
from catan_rl.ppo.training_loop import build_training_state

_SENTINEL = 0.123456


def _tiny_cfg(**over: object) -> TrainConfig:
    cfg = TrainConfig.default()
    cfg = replace(
        cfg,
        rollout=replace(cfg.rollout, n_envs=2, n_steps=8, max_turns=25),
        ppo=replace(cfg.ppo, batch_size=4, n_epochs=1, target_kl=0.0),
        eval=replace(cfg.eval, eval_every_updates=10_000, eval_games=2),
        league=replace(cfg.league, heuristic_weight=1.0, add_snapshot_every_n_updates=1, maxlen=4),
        total_steps=2 * 2 * 8,
    )
    return replace(cfg, **over)  # type: ignore[arg-type]


def _make_source_ckpt(path: Path, cfg: TrainConfig) -> str:
    """Save a v2-lineage checkpoint whose FIRST learnable param is a sentinel constant.

    Returns that param's name so the test can compare it after warm-start.
    """
    pol = CatanPolicy()
    pol.set_board_geometry(build_geometry().as_dict_of_tensors())
    name, param = next(iter(pol.named_parameters()))
    with torch.no_grad():
        param.fill_(_SENTINEL)
    save_checkpoint(
        path,
        config=cfg.to_dict(),
        policy=pol,
        optimizer=None,
        update_idx=0,
        global_step=0,
        capture_rng=False,
    )
    return name


def test_warmstart_loads_learner_weights(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    src = tmp_path / "src.pt"
    name = _make_source_ckpt(src, cfg)

    cfg_ws = replace(cfg, init_policy_checkpoint=str(src))
    state = build_training_state(cfg_ws, run_dir=tmp_path / "ws", device_label="cpu")
    got = dict(state.policy.named_parameters())[name].detach().cpu()
    assert torch.allclose(got, torch.full_like(got, _SENTINEL)), (
        "warm-start did not load the source checkpoint's learner weights"
    )
    state.vec_env.close()


def test_no_warmstart_is_fresh_init(tmp_path: Path) -> None:
    cfg = _tiny_cfg()  # init_policy_checkpoint is None (default)
    src = tmp_path / "src.pt"
    name = _make_source_ckpt(src, cfg)

    assert cfg.init_policy_checkpoint is None
    state = build_training_state(cfg, run_dir=tmp_path / "nows", device_label="cpu")
    got = dict(state.policy.named_parameters())[name].detach().cpu()
    assert not torch.allclose(got, torch.full_like(got, _SENTINEL)), (
        "default (no warm-start) path must NOT load the sentinel checkpoint — additivity broken"
    )
    state.vec_env.close()


def test_init_policy_checkpoint_round_trips(tmp_path: Path) -> None:
    # Config field survives to_dict -> _from_dict and YAML, and defaults to None.
    cfg = TrainConfig.default()
    assert cfg.init_policy_checkpoint is None
    cfg2 = replace(cfg, init_policy_checkpoint="runs/anchors/foo.pt")
    rt = TrainConfig._from_dict(cfg2.to_dict())
    assert rt.init_policy_checkpoint == "runs/anchors/foo.pt"
    yaml_path = tmp_path / "c.yaml"
    cfg2.to_yaml(yaml_path)
    assert TrainConfig.from_yaml(yaml_path).init_policy_checkpoint == "runs/anchors/foo.pt"
