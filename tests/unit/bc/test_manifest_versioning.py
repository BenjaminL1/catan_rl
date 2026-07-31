"""D5 — the corpus ordering must be ENFORCEABLE, not prose.

``bc.loader`` never re-evaluates ``forced``: the write-time filter in
``bc.dataset`` is the only gate, so a corpus generated under an older
predicate is silently missing whole decision classes and nothing in the arrays
reveals it. The manifest therefore carries ``(forced_rule_version,
ruleset_version)`` and the loader refuses an unstamped or stale directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_rl.bc.dataset import generate_dataset
from catan_rl.bc.loader import BcDataset, StaleCorpusError, load_manifest
from catan_rl.policy.obs_schema import FORCED_RULE_VERSION, RULESET_VERSION


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bc_corpus")
    generate_dataset(
        out_dir=out,
        n_games=2,
        perturb_pct=0.0,
        shard_size=2,
        seed=7,
        max_turns=60,
        progress_every=10**9,
    )
    return out


def test_fresh_corpus_is_stamped_and_loads(corpus: Path) -> None:
    manifest = load_manifest(corpus)
    assert manifest["forced_rule_version"] == FORCED_RULE_VERSION
    assert manifest["ruleset_version"] == RULESET_VERSION
    BcDataset(corpus, aug_prob=0.0)


def _rewrite_manifest(src: Path, dst: Path, mutate) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".npz":
            (dst / f.name).write_bytes(f.read_bytes())
    manifest = json.loads((src / "manifest.json").read_text())
    mutate(manifest)
    (dst / "manifest.json").write_text(json.dumps(manifest))
    return dst


def test_unstamped_manifest_is_refused(corpus: Path, tmp_path: Path) -> None:
    def drop(m: dict) -> None:
        m.pop("forced_rule_version", None)
        m.pop("ruleset_version", None)

    stale = _rewrite_manifest(corpus, tmp_path / "unstamped", drop)
    with pytest.raises(StaleCorpusError, match="forced_rule_version"):
        load_manifest(stale)
    with pytest.raises(StaleCorpusError):
        BcDataset(stale, aug_prob=0.0)


def test_stale_forced_rule_version_is_refused(corpus: Path, tmp_path: Path) -> None:
    stale = _rewrite_manifest(
        corpus, tmp_path / "stale_rule", lambda m: m.update(forced_rule_version=1)
    )
    with pytest.raises(StaleCorpusError, match="forced_rule_version=1"):
        load_manifest(stale)


def test_stale_ruleset_version_is_refused(corpus: Path, tmp_path: Path) -> None:
    stale = _rewrite_manifest(
        corpus, tmp_path / "stale_ruleset", lambda m: m.update(ruleset_version=1)
    )
    with pytest.raises(StaleCorpusError, match="ruleset_version=1"):
        load_manifest(stale)


def test_split_loader_also_enforces_the_stamp(corpus: Path, tmp_path: Path) -> None:
    stale = _rewrite_manifest(
        corpus, tmp_path / "split_stale", lambda m: m.pop("ruleset_version", None)
    )
    with pytest.raises(StaleCorpusError):
        BcDataset.train_val_split(stale, val_pct=0.5)
