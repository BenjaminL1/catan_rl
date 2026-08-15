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
    for _ in range(12):  # three complete drafts -> three game_seeds
        scenario = session.current_scenario()
        assert scenario is not None
        vertex = int(rng.choice(np.flatnonzero(scenario.legal_settlement_corners)))
        edge = int(rng.choice(np.flatnonzero(scenario.compute_legal_road_edges(vertex))))
        session.submit(vertex, edge)
    session.quit()

    shard_dir = root / "shard"
    # DEFAULT ``held_out_frac`` — the shard the AC5 drift test fine-tunes on must
    # really withhold something, or ``exclude_game_seeds`` is exercised empty on
    # both sides and "held out" is a claim about no code path at all.
    manifest = convert(session.scenarios_path, shard_dir)
    assert manifest["held_out_game_seeds"], "the AC5 corpus must withhold at least one seed"

    torch.manual_seed(7)
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    ckpt = root / "champion.pt"
    save_policy_only(ckpt, config={"rollout": {}}, policy=policy, update_idx=500, global_step=0)
    return {
        "ckpt": ckpt,
        "labels": session.scenarios_path,
        "shard": shard_dir,
        "root": root,
        "held_out": tuple(int(s) for s in manifest["held_out_game_seeds"]),
    }


def _walker(corpus: dict):  # type: ignore[no-untyped-def]
    """The frozen champion, in the role the anchor walker plays (D5)."""
    from catan_rl.bc.finetune import build_policy

    policy = build_policy(corpus["ckpt"], torch.device("cpu"))
    policy.eval()
    return policy


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
    anchored_run = finetune(_config(corpus, tmp_path / "anchored", kl_coef=1.0, **common))
    finetune(_config(corpus, tmp_path / "free", kl_coef=0.0, **common))

    # AC5's "held-out" is only true if something was actually withheld AND the
    # measurement excludes it. The seeds come from the run's own history, which
    # read them off the shard manifest — so the two sides cannot disagree.
    held = tuple(anchored_run["held_out_game_seeds"])
    assert held, "the fixture shard must withhold a seed for this to mean anything"
    assert set(held) == set(corpus["held_out"])

    anchored = held_out_anchor_kl(
        tmp_path / "anchored" / "candidate.pt",
        corpus["ckpt"],
        corpus["labels"],
        n_states=32,
        exclude_game_seeds=held,
    )
    free = held_out_anchor_kl(
        tmp_path / "free" / "candidate.pt",
        corpus["ckpt"],
        corpus["labels"],
        n_states=32,
        exclude_game_seeds=held,
    )
    # The bound is CALIBRATED off the control rather than hardcoded: an absolute
    # constant would encode this machine's arithmetic and would go stale the
    # first time the mock champion's init changed. What the anchor claims is a
    # RATIO — held-out drift materially below what the same run does unanchored.
    bound = 0.5 * free.kl
    assert anchored.kl < bound, (
        f"anchored drift {anchored.kl} is not materially below the unanchored "
        f"control {free.kl} (bound {bound})"
    )
    # Absolute sanity so a collapse of BOTH runs to ~0 cannot pass vacuously.
    assert free.kl > 1e-3, f"unanchored drift {free.kl} is too small to test against"
    assert anchored.n_states == free.n_states == 32


def test_the_anchor_bounds_value_drift_too(corpus: dict, tmp_path: Path) -> None:
    """D5/D8. The six action heads are not the whole champion: the deployed
    search reads the VALUE head through ``sigma(3.22V - 1.14)``, so unconstrained
    value drift is a deployment-visible regression the KL term never sees."""
    # THREE SEEDS, averaged. Single-seed value drift on this mock corpus swings
    # by an order of magnitude run to run (measured 0.010 / 0.176 / 0.084 for the
    # unguarded control at seeds 3/4/5), so a one-seed ratio would pin the seed,
    # not the guard.
    common = dict(lr=1e-3, steps=60, anchor_refresh_every=10, n_anchor_states=64, anchor_batch=16)
    guarded, unguarded = [], []
    for seed in (3, 4, 5):
        tag = f"s{seed}"
        finetune(_config(corpus, tmp_path / f"guarded_{tag}", seed=seed, **common))
        finetune(
            _config(
                corpus, tmp_path / f"unguarded_{tag}", seed=seed, anchor_value_coef=0.0, **common
            )
        )
        for name, bucket in (("guarded", guarded), ("unguarded", unguarded)):
            bucket.append(
                held_out_anchor_kl(
                    tmp_path / f"{name}_{tag}" / "candidate.pt",
                    corpus["ckpt"],
                    corpus["labels"],
                    n_states=32,
                    exclude_game_seeds=corpus["held_out"],
                ).value_mse
            )

    guarded_mean = float(np.mean(guarded))
    unguarded_mean = float(np.mean(unguarded))
    assert guarded_mean < 0.5 * unguarded_mean, (
        f"guarded value drift {guarded_mean} is not materially below the "
        f"unguarded control {unguarded_mean} (per-seed {guarded} vs {unguarded})"
    )
    # Non-vacuity: a control that never moved would make the ratio meaningless.
    assert unguarded_mean > 1e-3, (
        f"unguarded value drift {unguarded_mean} is too small to test against"
    )


def test_history_records_the_anchor_value_term(corpus: dict, tmp_path: Path) -> None:
    import json

    finetune(_config(corpus, tmp_path / "run"))
    payload = json.loads((tmp_path / "run" / "history.json").read_text())
    assert payload["steps"]
    for entry in payload["steps"]:
        assert entry["anchor_value_mse"] >= 0.0


def test_anchor_states_are_never_setup_states(corpus: dict) -> None:
    """Setup contexts are what the human rows TEACH; anchoring them to the
    champion would cancel the fine-tune out."""
    from catan_rl.bc.anchor_states import sample_anchor_states

    states = sample_anchor_states(
        corpus["labels"], policy=_walker(corpus), n_states=12, rng=np.random.default_rng(5)
    )
    assert len(states) == 12
    for state in states:
        assert float(np.asarray(state["obs"]["is_setup"]).reshape(-1)[0]) == 0.0


def test_anchor_sampler_refuses_to_fall_back(corpus: dict, tmp_path: Path) -> None:
    """No heuristic/random openings, ever — the anchor must cover the state
    distribution the owner's openings lead to."""
    from catan_rl.bc.anchor_states import NoLabeledOpeningsError, sample_anchor_states

    empty = tmp_path / "scenarios.jsonl"
    empty.write_text("")
    with pytest.raises(NoLabeledOpeningsError, match="no COMPLETE labeled opening"):
        sample_anchor_states(
            empty, policy=_walker(corpus), n_states=4, rng=np.random.default_rng(0)
        )


# ---------------------------------------------------------------------------
# D5 — the anchor walker IS the frozen champion, not a random legal walker
# ---------------------------------------------------------------------------


def test_the_anchor_walker_is_the_policy_that_was_handed_in(corpus: dict) -> None:
    """D5 pins the anchor states to the distribution the fine-tune moves toward.
    A random legal walker makes the FORCED opening the only champion-shaped thing
    about the trajectory: every ply after it is off-policy noise. Spy on
    ``policy.sample`` and pin BOTH halves — it is called once per collected
    state, and the emitted action is exactly what it returned."""
    from catan_rl.bc.anchor_states import sample_anchor_states

    policy = _walker(corpus)
    returned: list[list[int]] = []
    inner = policy.sample

    def spy(obs, masks):  # type: ignore[no-untyped-def]
        out = inner(obs, masks)
        returned.append([int(x) for x in out["action"].reshape(-1)])
        return out

    policy.sample = spy  # type: ignore[method-assign]
    states = sample_anchor_states(
        corpus["labels"], policy=policy, n_states=10, rng=np.random.default_rng(2)
    )
    assert len(states) == 10
    # The sampler may over-collect within a game only up to ``n_states``; every
    # call must correspond to a kept state.
    assert len(returned) == 10
    for state, action in zip(states, returned, strict=True):
        assert [int(x) for x in np.asarray(state["action"]).reshape(-1)] == action


def test_the_walkers_choice_is_not_overridden(corpus: dict) -> None:
    """Non-vacuity for the test above: force the walker onto the HIGHEST legal
    action type at every node and the collected actions must all be it. A
    sampler that quietly kept its own random walker would not reproduce that."""
    from catan_rl.bc.anchor_states import sample_anchor_states

    policy = _walker(corpus)
    inner = policy.sample

    def forced(obs, masks):  # type: ignore[no-untyped-def]
        # Narrow the TYPE mask to one entry and let the real heads pick the
        # sub-indices, so the forced action stays legal under the env's masks.
        legal = torch.nonzero(masks["type"].reshape(-1)).reshape(-1)
        narrowed = dict(masks)
        type_mask = torch.zeros_like(masks["type"])
        type_mask[..., int(legal[-1])] = True
        narrowed["type"] = type_mask
        return inner(obs, narrowed)

    policy.sample = forced  # type: ignore[method-assign]
    states = sample_anchor_states(
        corpus["labels"], policy=policy, n_states=8, rng=np.random.default_rng(4)
    )
    for state in states:
        legal_types = np.flatnonzero(np.asarray(state["mask"]["type"], dtype=bool))
        assert int(np.asarray(state["action"]).reshape(-1)[0]) == int(legal_types[-1])


def test_the_anchor_sampler_has_no_policy_default(corpus: dict) -> None:
    """A defaulted walker is how the random fallback the module docstring forbids
    would come back — silently, under a name that says otherwise."""
    from catan_rl.bc.anchor_states import sample_anchor_states

    with pytest.raises(TypeError):
        sample_anchor_states(  # type: ignore[call-arg]
            corpus["labels"], n_states=2, rng=np.random.default_rng(0)
        )


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
    from catan_rl.bc.finetune import build_policy, setup_agreement
    from catan_rl.labeling.store import load_scenarios
    from catan_rl.labeling.to_shard import UNKNOWN_POLICY_ID, rows_for_label
    from catan_rl.policy.obs_schema import OPP_KIND_HEURISTIC

    labels = load_scenarios(corpus["labels"])
    policy = build_policy(corpus["ckpt"], torch.device("cpu"))
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
    real_build = ft.build_policy

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

    monkeypatch.setattr(ft, "build_policy", _spy)
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
        policy=_walker(corpus),
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


# ---------------------------------------------------------------------------
# D7 gate-1 honesty: an ungated shard produces an ungateable candidate
# ---------------------------------------------------------------------------


def test_finetune_refuses_a_shard_that_withheld_nothing(corpus: dict, tmp_path: Path) -> None:
    """``eval_setup_agreement.py`` refusing at GATE time is too late — the
    candidate is already trained on every label, and the only remedy is to
    re-convert and re-run. Refuse where the decision still costs nothing."""
    from catan_rl.bc.finetune import UngatedShardError

    full = tmp_path / "full_shard"
    convert(corpus["labels"], full, held_out_frac=0.0)
    with pytest.raises(UngatedShardError, match="EMPTY held-out set"):
        finetune(_config(corpus, tmp_path / "run", shard_dir=full))


def test_finetune_refuses_a_shard_with_no_manifest(corpus: dict, tmp_path: Path) -> None:
    """Missing-manifest and empty-set are different corpus defects and the
    message says which."""
    from catan_rl.bc.finetune import UngatedShardError

    bare = tmp_path / "bare_shard"
    convert(corpus["labels"], bare, held_out_frac=0.5)
    (bare / "manifest.json").unlink()
    with pytest.raises(UngatedShardError, match=r"no manifest\.json"):
        finetune(_config(corpus, tmp_path / "run", shard_dir=bare))


def test_allow_ungated_is_an_explicit_escape_recorded_in_the_history(
    corpus: dict, tmp_path: Path
) -> None:
    """Deliberately training on the whole corpus is legitimate (there is no gate
    to protect once the labels are spent). It must be a DECISION in the config
    and a fact in the artefact, not a silent default."""
    full = tmp_path / "full_shard"
    convert(corpus["labels"], full, held_out_frac=0.0)
    history = finetune(_config(corpus, tmp_path / "run", shard_dir=full, allow_ungated=True))
    assert history["allow_ungated"] is True
    assert history["held_out_gated"] is False
    assert history["held_out_game_seeds"] == []


def test_a_gated_run_records_that_it_was_gated(corpus: dict, tmp_path: Path) -> None:
    history = finetune(_config(corpus, tmp_path / "run"))
    assert history["allow_ungated"] is False
    assert history["held_out_gated"] is True
    assert history["held_out_game_seeds"] == sorted(corpus["held_out"])
