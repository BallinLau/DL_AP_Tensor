"""Unit tests for the P-stage eta=1 b-distribution augmentation.

Covers the two design contracts of exp/pv-eta1-baugment:
  1. share=0 / mode=none keeps the existing counterfactual relabel behavior
     bit-identical (sizes, eta share, children identity, parent columns).
  2. mode=uniform appends full source bundles (only parent eta overwritten),
     flattening the b distribution of the eta=1 half; validation stream is
     never augmented; sampling is deterministic under the configured seed.
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config.hyperparams import HyperParams  # noqa: E402
from training.episode import Episode  # noqa: E402

N_BATCHES = 3
ROWS = 64


class _Shim:
    """Minimal Episode host for _balance_p_current_eta_batches."""

    def __init__(self, hyperparams, episode_id=29):
        self.hyperparams = hyperparams
        self.episode_id = episode_id

    _current_eta_counts_from_batches = staticmethod(
        Episode._current_eta_counts_from_batches
    )
    _balance_p_current_eta_batches = Episode._balance_p_current_eta_batches
    _append_eta1_augmented_rows = staticmethod(
        Episode._append_eta1_augmented_rows
    )
    _append_eta1_augmented_rows = staticmethod(
        Episode._append_eta1_augmented_rows
    )


def _make_batches(seed=7):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(N_BATCHES):
        parent = torch.zeros(ROWS, 8)
        n_low = int(ROWS * 0.9)  # 90% low-b concentration (natural caliber)
        parent[:n_low, 0] = torch.rand(n_low, generator=g) * 0.1
        parent[n_low:, 0] = 0.1 + torch.rand(ROWS - n_low, generator=g) * 0.9
        parent[:, 1] = torch.randn(ROWS, generator=g)
        parent[:, 2] = (torch.rand(ROWS, generator=g) < 0.05).float()
        parent[:, 3] = torch.rand(ROWS, generator=g) * 0.5
        children = [
            torch.randn(ROWS, 8, generator=g),
            torch.randn(ROWS, 8, generator=g),
        ]
        batches.append(
            {
                "parent": parent,
                "children": children,
                "child0": children[0],
                "child1": children[1],
                "parent_index": torch.arange(ROWS),
                "parent_source_index": torch.arange(ROWS),
                "source_id": torch.zeros(ROWS, dtype=torch.long),
            }
        )
    return batches


def _hp(**augment):
    hp = HyperParams()
    hp.pv_current_eta_balance_enabled = True
    hp.pv_current_eta1_train_share = 0.50
    hp.pv_current_eta_balance_validation = True
    hp.pv_current_eta_balance_seed = 97531
    hp.pv_eta1_b_augment_mode = augment.get("mode", "none")
    hp.pv_eta1_b_augment_share = augment.get("share", 1.0)
    hp.pv_eta1_b_augment_bins = augment.get("bins", 10)
    hp.pv_eta1_b_augment_seed = augment.get("seed", 20261101)
    return hp


def test_augment_off_preserves_current_behavior():
    batches = _make_batches()
    balanced, summary = _Shim(_hp(mode="none"))._balance_p_current_eta_batches(
        batches, stream="train"
    )
    assert len(balanced) == len(batches)
    for old, new in zip(batches, balanced):
        assert new["parent"].shape == old["parent"].shape
        # children bundles are the very same objects (no copy, no append)
        assert new["children"] is old["children"]
        assert new["child0"] is old["child0"]
        assert new["child1"] is old["child1"]
        # only the eta column may differ
        same_cols = (
            new["parent"][:, [0, 1, 3, 4, 5, 6, 7]]
            == old["parent"][:, [0, 1, 3, 4, 5, 6, 7]]
        ).all()
        assert bool(same_cols)
        eta1 = int((new["parent"][:, 2] > 0.5).sum())
        assert eta1 == int(round(ROWS * 0.5))
    assert summary["p_eta_sampling_method"] == "counterfactual_parent_eta_balance"
    assert summary["augmented_rows"] == 0
    assert summary["pv_eta1_b_augment_mode"] == "none"
    # originals untouched
    for old in batches:
        assert int((old["parent"][:, 2] > 0.5).sum()) <= int(ROWS * 0.05) + 3


def test_augment_on_shapes_eta_and_uniformity():
    share = 0.5
    batches = _make_batches()
    shim = _Shim(_hp(mode="uniform", share=share))
    balanced, summary = shim._balance_p_current_eta_batches(
        batches, stream="train"
    )
    n_aug = int(round(ROWS * 0.5 * share))
    assert n_aug == 16
    total_eta1 = 0
    total_high_b_eta1 = 0
    for new in balanced:
        assert new["parent"].shape[0] == ROWS + n_aug
        appended = new["parent"][ROWS:]
        assert bool((appended[:, 2] > 0.5).all())
        # in-batch half keeps exactly round(ROWS*0.5)-n_aug natural relabels
        assert int((new["parent"][:ROWS, 2] > 0.5).sum()) == int(ROWS * 0.5) - n_aug
        eta1_rows = new["parent"][new["parent"][:, 2] > 0.5]
        total_eta1 += eta1_rows.shape[0]
        total_high_b_eta1 += int((eta1_rows[:, 0] > 0.3).sum())
    assert summary["p_eta_sampling_method"] == "counterfactual_eta_balance_b_uniform"
    assert summary["augmented_rows"] == N_BATCHES * n_aug
    assert summary["pv_eta1_b_augment_mode"] == "uniform"
    assert summary["eta1_b_hist_after"] is not None
    assert sum(summary["eta1_b_hist_after"]) == total_eta1
    # the b distribution of the eta=1 half must be strongly flattened:
    # natural caliber would give ~1-2% of eta=1 rows above b=0.3.
    high_share = total_high_b_eta1 / total_eta1
    assert high_share > 0.15, f"eta=1 high-b share too low: {high_share:.3f}"


def test_augment_bundle_fidelity():
    batches = _make_batches()
    shim = _Shim(_hp(mode="uniform", share=1.0))
    balanced, _ = shim._balance_p_current_eta_batches(batches, stream="train")
    n_aug = int(round(ROWS * 0.5))
    # every appended row must reproduce some source row verbatim except eta
    all_parents = torch.cat([b["parent"] for b in batches], dim=0)
    all_child0 = torch.cat([b["children"][0] for b in batches], dim=0)
    all_child1 = torch.cat([b["children"][1] for b in batches], dim=0)
    for new in balanced:
        appended_parent = new["parent"][ROWS:]
        appended_c0 = new["children"][0][ROWS:]
        appended_c1 = new["children"][1][ROWS:]
        for r in range(appended_parent.shape[0]):
            row = appended_parent[r].clone()
            row[2] = 0.0  # eta was overwritten; mask it out
            match = (
                (all_parents[:, [0, 1]] == row[[0, 1]]).all(dim=1)
                & (all_parents[:, 3:] == row[3:]).all(dim=1)
            ).nonzero(as_tuple=True)[0]
            assert match.numel() >= 1
            idx = int(match[0])
            assert torch.equal(appended_c0[r], all_child0[idx])
            assert torch.equal(appended_c1[r], all_child1[idx])


def test_augment_deterministic():
    balanced_a, sum_a = _Shim(_hp(mode="uniform", share=0.5))._balance_p_current_eta_batches(
        _make_batches(), stream="train"
    )
    balanced_b, sum_b = _Shim(_hp(mode="uniform", share=0.5))._balance_p_current_eta_batches(
        _make_batches(), stream="train"
    )
    for a, b in zip(balanced_a, balanced_b):
        assert torch.equal(a["parent"], b["parent"])
        assert torch.equal(a["children"][0], b["children"][0])
        assert torch.equal(a["children"][1], b["children"][1])
    assert sum_a["augmented_rows"] == sum_b["augmented_rows"]


def test_validation_stream_never_augmented():
    batches = _make_batches()
    balanced, summary = _Shim(_hp(mode="uniform", share=0.5))._balance_p_current_eta_batches(
        batches, stream="validation"
    )
    for old, new in zip(batches, balanced):
        assert new["parent"].shape[0] == old["parent"].shape[0]
    assert summary["augmented_rows"] == 0
    assert summary["p_eta_sampling_method"] == "counterfactual_parent_eta_balance"


def test_augment_requires_balance_enabled():
    hp = _hp(mode="uniform", share=0.5)
    hp.pv_current_eta_balance_enabled = False
    balanced, summary = _Shim(hp)._balance_p_current_eta_batches(
        _make_batches(), stream="train"
    )
    assert summary["enabled"] is False
    assert summary.get("augmented_rows") is None or summary["reassigned_count"] == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
