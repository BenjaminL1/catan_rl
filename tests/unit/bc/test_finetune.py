"""D5 — champion fine-tune with self-distillation anchoring.

Everything here runs on a FRESH randomly-initialised policy standing in for
``runs/anchors/ptr_v1_u500.pt`` (which is not in the test tree) and on a
throwaway label corpus. The fine-tune RUN against the real champion is deferred
until >=200 labeled scenarios exist (spec AC7); only the code lands now.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from catan_rl.bc.finetune import FinetuneConfig, finetune, held_out_anchor_kl
from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.to_shard import convert


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict:  # type: ignore[no-untyped-def]
    """A mock champion + a small labeled corpus + its BC shard."""
    from catan_rl.checkpoint import save_policy_only
    from catan_rl.policy import CatanPolicy
    from catan_rl.policy.board_geometry import build_geometry

    root = tmp_path_factory.mktemp("ft")
    session = LabelingSession(data_dir=root / "labels", labeler_id="test", session_seed=99)
    session.start()
    rng = np.random.default_rng(0)
    for _ in range(8):  # two complete drafts
        scenario = session.current_scenario()
        assert scenario is not None
        vertex = int(rng.choice(np.flatnonzero(scenario.legal_settlement_corners)))
        edge = int(rng.choice(np.flatnonzero(scenario.compute_legal_road_edges(vertex))))
        session.submit(vertex, edge)
    session.quit()

    shard_dir = root / "shard"
    convert(session.scenarios_path, shard_dir)

    torch.manual_seed(7)
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    ckpt = root / "champion.pt"
    save_policy_only(ckpt, config={"rollout": {}}, policy=policy, update_idx=500, global_step=0)
    return {"ckpt": ckpt, "labels": session.scenarios_path, "shard": shard_dir, "root": root}


def _config(corpus: dict, out: Path, **kwargs) -> FinetuneConfig:  # type: ignore[no-untyped-def]
    params = dict(
        ckpt_path=corpus["ckpt"],
        shard_dir=corpus["shard"],
        labels_path=corpus["labels"],
        out_dir=out,
        steps=6,
        batch_size=8,
        lr=1e-3,
        kl_coef=1.0,
        anchor_batch=4,
        n_anchor_states=16,
        anchor_refresh_every=0,
        seed=3,
    )
    params.update(kwargs)
    return FinetuneConfig(**params)  # type: ignore[arg-type]


def test_smoke_finetune_runs_end_to_end(corpus: dict, tmp_path: Path) -> None:
    history = finetune(_config(corpus, tmp_path / "run"))
    assert len(history["steps"]) == 6
    assert (tmp_path / "run" / "candidate.pt").is_file()
    assert (tmp_path / "run" / "human_opening_prior.pt").is_file()
    assert (tmp_path / "run" / "history.json").is_file()
    for entry in history["steps"]:
        assert np.isfinite(entry["total"])
        assert entry["anchor_kl"] >= -1e-6


def test_the_candidate_is_stamped_r0(corpus: dict, tmp_path: Path) -> None:
    """D6. ptr_v1_u500.pt carries no ruleset stamp, so the lineage is R0 — and
    the candidate says so explicitly rather than inheriting a default."""
    from catan_rl.eval.harness import checkpoint_ruleset

    finetune(_config(corpus, tmp_path / "run"))
    assert checkpoint_ruleset(tmp_path / "run" / "candidate.pt") == "R0"
    assert checkpoint_ruleset(tmp_path / "run" / "human_opening_prior.pt") == "R0"


def test_value_loss_is_zero_on_human_rows(corpus: dict, tmp_path: Path) -> None:
    """A hand-labeled opening carries NO outcome; ``z_disc`` is a filler zero.
    Regressing the value head toward it would teach the champion that every
    labeled opening is a dead draw."""
    history = finetune(_config(corpus, tmp_path / "run"))
    assert [entry["value"] for entry in history["steps"]] == [0.0] * 6


def test_the_anchor_term_bounds_non_setup_drift(corpus: dict, tmp_path: Path) -> None:
    """The bound test is NOT vacuous: an identically-configured run with
    ``kl_coef=0`` must blow through the same bound."""
    # ``kl_coef`` is the PRODUCTION default (1.0). Only the learning rate is
    # inflated, so the unanchored control has room to move the midgame inside a
    # 60-step test. Anchor refresh is ON: with a fixed pool the fine-tune simply
    # overfits those states, driving the training term down while the held-out
    # drift grows — which is what the refresh exists to prevent, and which this
    # test would otherwise report as the anchor "not working".
    common = dict(lr=1e-3, steps=60, anchor_refresh_every=10, n_anchor_states=64, anchor_batch=16)
    finetune(_config(corpus, tmp_path / "anchored", kl_coef=1.0, **common))
    finetune(_config(corpus, tmp_path / "free", kl_coef=0.0, **common))

    anchored_kl = held_out_anchor_kl(
        tmp_path / "anchored" / "candidate.pt", corpus["ckpt"], corpus["labels"], n_states=32
    )
    free_kl = held_out_anchor_kl(
        tmp_path / "free" / "candidate.pt", corpus["ckpt"], corpus["labels"], n_states=32
    )
    # The bound is CALIBRATED off the control rather than hardcoded: an absolute
    # constant would encode this machine's arithmetic and would go stale the
    # first time the mock champion's init changed. What the anchor claims is a
    # RATIO — held-out drift materially below what the same run does unanchored.
    bound = 0.5 * free_kl
    assert anchored_kl < bound, (
        f"anchored drift {anchored_kl} is not materially below the unanchored "
        f"control {free_kl} (bound {bound})"
    )
    # Absolute sanity so a collapse of BOTH runs to ~0 cannot pass vacuously.
    assert free_kl > 1e-3, f"unanchored drift {free_kl} is too small to test against"


def test_anchor_states_are_never_setup_states(corpus: dict) -> None:
    """Setup contexts are what the human rows TEACH; anchoring them to the
    champion would cancel the fine-tune out."""
    from catan_rl.bc.anchor_states import sample_anchor_states

    states = sample_anchor_states(corpus["labels"], n_states=12, rng=np.random.default_rng(5))
    assert len(states) == 12
    for state in states:
        assert float(np.asarray(state["obs"]["is_setup"]).reshape(-1)[0]) == 0.0


def test_anchor_sampler_refuses_to_fall_back(tmp_path: Path) -> None:
    """No heuristic/random openings, ever — the anchor must cover the state
    distribution the owner's openings lead to."""
    from catan_rl.bc.anchor_states import NoLabeledOpeningsError, sample_anchor_states

    empty = tmp_path / "scenarios.jsonl"
    empty.write_text("")
    with pytest.raises(NoLabeledOpeningsError, match="no COMPLETE labeled opening"):
        sample_anchor_states(empty, n_states=4, rng=np.random.default_rng(0))


def test_setup_agreement_measures_top1_against_the_labels(corpus: dict) -> None:
    """D7 gate 1's measurement half: agreement is read off the CORNER head's
    masked argmax at the same obs the shard trained on."""
    from catan_rl.bc.finetune import setup_agreement
    from catan_rl.labeling.store import held_out_split, load_scenarios

    labels = load_scenarios(corpus["labels"])
    train, held = held_out_split(labels, frac=0.5, seed=0)
    assert train and held
    # Split by game_seed, so no draft straddles the two halves.
    assert not ({int(r["game_seed"]) for r in train} & {int(r["game_seed"]) for r in held})

    got = setup_agreement(corpus["ckpt"], held)
    assert len(got) == len(held)
    assert all(isinstance(x, bool) for x in got)


def test_setup_agreement_is_perfect_against_a_policy_that_labels_itself(
    corpus: dict,
) -> None:
    """Not vacuous: build the labels FROM the policy's own argmax and the
    measurement must read 1.0, which pins the index convention (corner head
    index == label ``settlement_vertex``) rather than merely the plumbing."""
    from catan_rl.bc.finetune import _build_policy, setup_agreement
    from catan_rl.labeling.store import load_scenarios
    from catan_rl.labeling.to_shard import UNKNOWN_POLICY_ID, rows_for_label
    from catan_rl.policy.obs_schema import OPP_KIND_HEURISTIC

    labels = load_scenarios(corpus["labels"])
    policy = _build_policy(corpus["ckpt"], torch.device("cpu"))
    policy.eval()

    self_labeled = []
    for label in labels[:3]:
        settle_row, _ = rows_for_label(label, game_id=0)
        obs = {}
        for key, value in settle_row.obs.items():
            tensor = torch.as_tensor(np.asarray(value))
            obs[key] = (
                tensor.reshape(-1) if tensor.dtype == torch.int64 else tensor.unsqueeze(0).float()
            )
        # Same opponent-id slice ``setup_agreement`` reads in — a self-label
        # taken under a DIFFERENT conditional would not have to be reproduced.
        obs["opponent_kind"] = torch.full((1,), OPP_KIND_HEURISTIC, dtype=torch.int64)
        obs["opponent_policy_id"] = torch.full((1,), UNKNOWN_POLICY_ID, dtype=torch.int64)
        mask = {
            k: torch.as_tensor(np.asarray(v, dtype=bool)).unsqueeze(0)
            for k, v in settle_row.mask.items()
        }
        action = torch.as_tensor(settle_row.action).unsqueeze(0)
        with torch.no_grad():
            out = policy.evaluate_actions(obs, action, mask)
        row = dict(label)
        row["settlement_vertex"] = int(out["log_dist/corner"].argmax(dim=-1).item())
        # The road must stay legal under the substituted settlement.
        edges = np.flatnonzero(_legal_roads(label, row["settlement_vertex"]))
        row["road_edge"] = int(edges[0])
        self_labeled.append(row)

    assert setup_agreement(corpus["ckpt"], self_labeled) == [True] * len(self_labeled)


def _legal_roads(label: dict, vertex: int) -> np.ndarray:
    from catan_rl.labeling.scenario_gen import Pick, ScenarioGenerator

    gen = ScenarioGenerator(seed=int(label["game_seed"]))
    for pick in label["prior_picks"]:
        p = Pick.from_dict(pick)
        gen.apply(p.settlement_vertex, p.road_edge)
    scenario = gen.current()
    assert scenario is not None
    return scenario.compute_legal_road_edges(vertex)


def test_agreement_is_measured_in_a_slice_the_shard_actually_trains(
    corpus: dict, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """D7 gate 1 must not be read under an opponent-id conditional the fine-tune
    never sees.

    ``rows_for_label`` returns PRE-duplication rows, whose ``opponent_kind`` is
    the encoder default ``OPP_KIND_UNKNOWN`` — the one kind
    ``DEFAULT_OPPONENT_KINDS`` deliberately excludes from the shard. Measuring
    there could report a successfully-installed opening as a gate failure whose
    stated remedy is MORE LABELS.
    """
    from catan_rl.bc import finetune as ft
    from catan_rl.labeling.store import load_scenarios
    from catan_rl.labeling.to_shard import (
        DEFAULT_OPPONENT_KINDS,
        UNKNOWN_POLICY_ID,
        rows_for_label,
    )
    from catan_rl.policy.obs_schema import OPP_KIND_HEURISTIC, OPP_KIND_UNKNOWN

    labels = load_scenarios(corpus["labels"])[:2]

    # The premise: the raw converter row really does carry UNKNOWN.
    settle_row, _ = rows_for_label(labels[0], game_id=0)
    assert int(settle_row.obs["opponent_kind"]) == OPP_KIND_UNKNOWN
    assert OPP_KIND_UNKNOWN not in DEFAULT_OPPONENT_KINDS

    seen: list[tuple[int, int]] = []
    real_build = ft._build_policy

    def _spy(ckpt_path, device):  # type: ignore[no-untyped-def]
        policy = real_build(ckpt_path, device)
        inner = policy.evaluate_actions

        def wrapped(obs, action, mask):  # type: ignore[no-untyped-def]
            seen.append(
                (
                    int(obs["opponent_kind"].reshape(-1)[0]),
                    int(obs["opponent_policy_id"].reshape(-1)[0]),
                )
            )
            return inner(obs, action, mask)

        policy.evaluate_actions = wrapped  # type: ignore[method-assign]
        return policy

    monkeypatch.setattr(ft, "_build_policy", _spy)
    ft.setup_agreement(corpus["ckpt"], labels)
    assert seen
    assert set(seen) == {(OPP_KIND_HEURISTIC, UNKNOWN_POLICY_ID)}
    assert all(kind in DEFAULT_OPPONENT_KINDS for kind, _ in seen)


def test_anchor_states_span_every_opponent_kind_the_shard_moves(corpus: dict) -> None:
    """The fine-tune's gradient reaches all three id-conditional slices, so the
    anchor term must be evaluated in all three — otherwise non-setup drift in
    the two the successor self-play runs under is neither measured nor
    penalised."""
    from catan_rl.bc.anchor_states import ANCHOR_OPPONENT_KINDS, sample_anchor_states

    # The kind is fixed per GAME (it is an env-construction argument), so the
    # coverage claim is "over enough games", not "over enough states". Short
    # games here; production's 256 states at 60 steps/game spans ~5.
    states = sample_anchor_states(
        corpus["labels"],
        n_states=120,
        rng=np.random.default_rng(11),
        max_steps_per_game=20,
    )
    kinds = {int(np.asarray(s["obs"]["opponent_kind"]).reshape(-1)[0]) for s in states}
    assert kinds == set(ANCHOR_OPPONENT_KINDS)


def test_held_out_openings_never_reach_the_anchor_rollouts(corpus: dict, tmp_path: Path) -> None:
    """A held-out opening that shapes the candidate through the KL term is not
    held out. The exclusion set comes from the SHARD manifest, so it cannot
    disagree with what the converter withheld."""
    from catan_rl.bc.anchor_states import complete_openings
    from catan_rl.bc.finetune import _shard_held_out_seeds
    from catan_rl.labeling.to_shard import convert

    out = tmp_path / "split_shard"
    manifest = convert(corpus["labels"], out, held_out_frac=0.5, split_seed=0)
    held = set(manifest["held_out_game_seeds"])
    assert held

    assert set(_shard_held_out_seeds(out)) == held
    openings = complete_openings(corpus["labels"], exclude_game_seeds=held)
    assert openings, "the throwaway corpus must retain a trainable opening"
    assert not ({o.game_seed for o in openings} & held)
    # And without the exclusion the held-out seed IS present, so the pin is not
    # vacuous.
    assert {o.game_seed for o in complete_openings(corpus["labels"])} & held
