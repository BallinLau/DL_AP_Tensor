from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from data.tensor_data import TensorTable  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402
from training.episode import Episode  # noqa: E402
from training.pv_mixture import build_fixed_total_mixture_split  # noqa: E402


def _episode(hp: HyperParams | None = None, episode_id: int = 1) -> Episode:
    device = torch.device("cpu")
    Config.DEVICE = device
    model = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return Episode(
        models={"policy_value": model},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=hp or HyperParams(),
        device=device,
        episode_id=episode_id,
    )


def _firm_table(n_parent: int, *, source_offset: int = 0, parent_branch: int = -1) -> TensorTable:
    rows = []
    for idx in range(n_parent):
        path = idx + source_offset
        firm = 10_000 + idx + source_offset
        parent = [
            float(path),
            0.0,
            float(parent_branch),
            0.0,
            0.10 + idx * 0.01,
            -0.20 + idx * 0.02,
            float(idx % 2),
            0.30,
            0.05,
            -2.00 + idx * 0.01,
            4.00 + idx * 0.01,
            1.00,
            1.00,
        ]
        rows.append([parent[0], float(firm), *parent[1:]])
        for branch in [0, 1]:
            child = parent.copy()
            child[1] = 1.0
            child[2] = float(branch)
            child[4] += 0.01 * (branch + 1)
            child[5] += 0.02 * (branch + 1)
            child[11] = 0.98 + 0.01 * branch
            rows.append([child[0], float(firm), *child[1:]])
    columns = [
        "path",
        "firm",
        "t",
        "branch",
        "Entry",
        "b",
        "z",
        "ETA",
        "i",
        "x",
        "Hatcf",
        "LnKF",
        "M",
        "K",
    ]
    return TensorTable(torch.tensor(rows, dtype=torch.float32), columns)


def _hp_mixture() -> HyperParams:
    hp = HyperParams()
    hp.pv_mixture_enabled = True
    hp.pv_mixture_ratio = 0.25
    hp.pv_mixture_start_episode = 1
    hp.pv_mixture_budget_mode = "fixed_total"
    hp.pv_mixture_sampling_mode = "uniform"
    hp.pv_mixture_seed = 123
    hp.pv_mixture_preserve_rng = True
    hp.pv_mixture_stratified_validation = True
    hp.pv_target_grid_val_fraction = 0.25
    hp.max_firm_train_units = 0
    hp.pv_eta_resample_enabled = False
    return hp


def test_fixed_total_split_preserves_parent_budget_and_source_metadata():
    episode = _episode(_hp_mixture(), episode_id=1)
    sim_pool = episode._tensor_to_parent_group_pool(_firm_table(8), source_id=0)
    coverage_pool = episode._tensor_to_parent_group_pool(_firm_table(8, source_offset=100), source_id=1)

    train_pool, val_pool, summary = build_fixed_total_mixture_split(
        sim_pool,
        coverage_pool,
        coverage_ratio=0.25,
        val_fraction=0.25,
        generator=torch.Generator().manual_seed(7),
        stratified_validation=True,
    )

    assert len(train_pool) + len(val_pool) == len(sim_pool)
    assert summary["coverage_parent_groups_selected"] == 2.0
    assert summary["sim_parent_groups_selected"] == 6.0
    assert set(torch.unique(train_pool.source_id).tolist() + torch.unique(val_pool.source_id).tolist()) == {0, 1}
    assert torch.allclose(train_pool.children[0][:, :2] - train_pool.parent[:, :2], train_pool.children[0][:, :2] - train_pool.parent[:, :2])


def test_episode_mixture_batches_are_fixed_total_and_rng_is_restored():
    hp = _hp_mixture()
    episode = _episode(hp, episode_id=1)
    sim_table = _firm_table(8)
    coverage_pool = episode._tensor_to_parent_group_pool(_firm_table(8, source_offset=100), source_id=1)
    episode._build_pv_coverage_pool = lambda *args, **kwargs: coverage_pool

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    train_batches, val_batches, summary = episode._prepare_mixed_policy_value_batches(
        sim_table,
        batch_size=3,
        n_branches=2,
    )

    assert summary["enabled"] is True
    assert summary["mixed_parent_groups_selected"] == 8.0
    assert summary["coverage_parent_groups_selected"] == 2.0
    assert sum(batch["parent"].shape[0] for batch in train_batches + val_batches) == 8
    source_ids = torch.cat([batch["source_id"] for batch in train_batches + val_batches])
    assert int((source_ids == 0).sum().item()) == 6
    assert int((source_ids == 1).sum().item()) == 2
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_episode0_and_zero_ratio_keep_mixture_disabled():
    hp = _hp_mixture()
    hp.pv_mixture_ratio = 0.0
    episode = _episode(hp, episode_id=1)
    train_batches, val_batches, summary = episode._prepare_mixed_policy_value_batches(
        _firm_table(4),
        batch_size=2,
        n_branches=2,
    )
    assert summary["enabled"] is False
    assert summary["reason"] == "zero_ratio"
    assert val_batches == []
    assert sum(batch["parent"].shape[0] for batch in train_batches) == 4

    hp.pv_mixture_ratio = 0.25
    episode0 = _episode(hp, episode_id=0)
    assert episode0._pv_mixture_enabled_for_episode() == (False, "episode0_unchanged")


def test_bp_target_cache_preserves_source_metadata(monkeypatch):
    episode = _episode(_hp_mixture(), episode_id=1)
    batch = episode._parent_group_pool_to_batches(
        episode._tensor_to_parent_group_pool(_firm_table(2), source_id=1),
        batch_size=2,
        eta_resample=False,
        shuffle=False,
    )[0]

    class _FakeTeacherResult(dict):
        pass

    class _FakeTeacher:
        @classmethod
        def from_hyperparams(cls, *args, **kwargs):
            return cls()

        def compute(self, parent_state, **kwargs):
            n = parent_state.shape[0]
            return {
                "bp_star": torch.full((n, 1), 0.2, device=parent_state.device),
                "confidence": torch.ones(n, 1, device=parent_state.device),
            }

    monkeypatch.setattr("training.episode.BPGridTeacher", _FakeTeacher)
    cache = episode._build_bp_target_cache([batch], episode.models["policy_value"])

    assert cache[0]["source_id"] is not None
    assert cache[0]["source_index"] is not None
    assert torch.equal(cache[0]["source_id"], batch["source_id"].cpu())
    assert torch.equal(cache[0]["source_index"], batch["source_index"].cpu())
