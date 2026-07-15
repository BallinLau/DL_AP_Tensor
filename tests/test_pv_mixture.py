from pathlib import Path
import random
import sys

import numpy as np
import pytest
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


def _firm_table(
    n_parent: int,
    *,
    source_offset: int = 0,
    parent_branch: int = -1,
    value_offset: float = 0.0,
) -> TensorTable:
    rows = []
    for idx in range(n_parent):
        path = idx + source_offset
        firm = 10_000 + idx + source_offset
        parent = [
            float(path),
            0.0,
            float(parent_branch),
            0.0,
            0.10 + value_offset + idx * 0.01,
            -0.20 - value_offset + idx * 0.02,
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
    hp.pv_training_flow = "staged"
    hp.max_firm_train_units = 0
    hp.pv_eta_resample_enabled = False
    return hp


def _source_pairs(batches):
    pairs = set()
    for batch in batches:
        for sid, sidx in zip(batch["source_id"].cpu().tolist(), batch["source_index"].cpu().tolist()):
            pairs.add((int(sid), int(sidx)))
    return pairs


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
    assert torch.allclose(train_pool.children[0][:, 0], train_pool.parent[:, 0] + 0.01)
    assert torch.allclose(train_pool.children[0][:, 1], train_pool.parent[:, 1] + 0.02)
    assert torch.allclose(train_pool.children[1][:, 0], train_pool.parent[:, 0] + 0.02)
    assert torch.allclose(train_pool.children[1][:, 1], train_pool.parent[:, 1] + 0.04)


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
    for batch in train_batches + val_batches:
        assert batch["parent"].shape[0] == batch["child0"].shape[0]
        assert batch["parent"].shape[0] == batch["child1"].shape[0]
        assert batch["parent"].shape[0] == batch["source_id"].shape[0]
        assert batch["parent"].shape[0] == batch["source_index"].shape[0]
    assert _source_pairs(train_batches).isdisjoint(_source_pairs(val_batches))


def test_max_firm_train_units_applies_before_coverage_generation():
    hp = _hp_mixture()
    hp.max_firm_train_units = 5
    hp.pv_mixture_ratio = 0.4
    hp.pv_target_grid_val_fraction = 0.2
    episode = _episode(hp, episode_id=1)
    requested = []

    def _coverage(n_parent_groups, n_branches):
        requested.append(int(n_parent_groups))
        return episode._tensor_to_parent_group_pool(
            _firm_table(n_parent_groups, source_offset=100, value_offset=1.0),
            source_id=1,
        )

    episode._build_pv_coverage_pool = _coverage
    train_batches, val_batches, summary = episode._prepare_mixed_policy_value_batches(
        _firm_table(20),
        batch_size=10,
        n_branches=2,
    )

    assert requested == [2]
    assert summary["total_parent_budget"] == 5.0
    assert summary["sim_parent_groups_selected"] == 3.0
    assert summary["coverage_parent_groups_selected"] == 2.0
    assert sum(batch["parent"].shape[0] for batch in train_batches + val_batches) == 5


def test_same_seed_is_reproducible_and_episode_seed_changes_selection():
    hp = _hp_mixture()
    hp.max_firm_train_units = 6
    hp.pv_mixture_ratio = 0.5
    sim_table = _firm_table(12)
    coverage_table = _firm_table(6, source_offset=100, value_offset=1.0)

    def _run(ep_id):
        episode = _episode(hp, episode_id=ep_id)
        coverage_pool = episode._tensor_to_parent_group_pool(coverage_table, source_id=1)
        episode._build_pv_coverage_pool = lambda *args, **kwargs: coverage_pool
        train, val, _ = episode._prepare_mixed_policy_value_batches(sim_table, batch_size=10, n_branches=2)
        return sorted(_source_pairs(train + val))

    assert _run(1) == _run(1)
    assert _run(1) != _run(2)


def test_episode0_disabled_but_zero_ratio_uses_experiment_control_pipeline():
    hp = _hp_mixture()
    hp.pv_mixture_ratio = 0.0
    episode = _episode(hp, episode_id=1)
    coverage_calls = []
    episode._build_pv_coverage_pool = lambda n, n_branches: coverage_calls.append(n) or None
    train_batches, val_batches, summary = episode._prepare_mixed_policy_value_batches(
        _firm_table(4),
        batch_size=2,
        n_branches=2,
    )
    assert summary["enabled"] is True
    assert summary["reason"] == "enabled_experiment_control"
    assert coverage_calls == []
    assert summary["coverage_parent_groups_selected"] == 0.0
    assert summary["actual_coverage_ratio"] == 0.0
    assert summary["train_coverage_ratio"] == 0.0
    assert summary["validation_coverage_ratio"] == 0.0
    assert sum(batch["parent"].shape[0] for batch in train_batches + val_batches) == 4

    hp.pv_mixture_ratio = 0.25
    episode0 = _episode(hp, episode_id=0)
    assert episode0._pv_mixture_enabled_for_episode() == (False, "episode0_unchanged")


def test_control_and_treatment_rng_parity():
    sim_table = _firm_table(8)
    coverage_pool = _episode(_hp_mixture(), episode_id=1)._tensor_to_parent_group_pool(
        _firm_table(8, source_offset=100, value_offset=1.0),
        source_id=1,
    )

    def _run(ratio):
        hp = _hp_mixture()
        hp.pv_mixture_ratio = ratio
        episode = _episode(hp, episode_id=1)
        if ratio > 0:
            episode._build_pv_coverage_pool = lambda *args, **kwargs: coverage_pool
        random.seed(333)
        np.random.seed(333)
        torch.manual_seed(333)
        before = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        episode._prepare_mixed_policy_value_batches(sim_table, batch_size=4, n_branches=2)
        after = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        return before, after

    control_before, control_after = _run(0.0)
    treatment_before, treatment_after = _run(0.25)

    assert control_after[0] == control_before[0] == treatment_before[0] == treatment_after[0]
    assert np.array_equal(control_after[1][1], control_before[1][1])
    assert np.array_equal(treatment_after[1][1], treatment_before[1][1])
    assert np.array_equal(control_after[1][1], treatment_after[1][1])
    assert torch.equal(control_after[2], control_before[2])
    assert torch.equal(treatment_after[2], treatment_before[2])
    assert torch.equal(control_after[2], treatment_after[2])


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


def test_bp_cache_hash_includes_source_metadata(monkeypatch):
    episode = _episode(_hp_mixture(), episode_id=1)
    batch = episode._parent_group_pool_to_batches(
        episode._tensor_to_parent_group_pool(_firm_table(2), source_id=1),
        batch_size=2,
        eta_resample=False,
        shuffle=False,
    )[0]

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
    original_hash = episode._bp_cache_hash(cache)
    cache[0]["source_index"] = cache[0]["source_index"] + 1
    assert episode._bp_cache_hash(cache) != original_hash


def test_real_coverage_sample_child_m_is_finite():
    hp = _hp_mixture()
    episode = _episode(hp, episode_id=1)

    class _SDFStub:
        def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=False):
            n = x_prev.shape[0]
            m = torch.full((n,), 0.99, dtype=x_prev.dtype, device=x_prev.device)
            return (
                torch.zeros_like(m),
                torch.zeros_like(m),
                m,
                hatcf_prev + 0.01,
                lnkf_prev + 0.02,
            )

    episode.models["sdf_fc1"] = _SDFStub()
    pool = episode._build_pv_coverage_pool(n_parent_groups=4, n_branches=2)

    assert pool is not None
    assert len(pool) == 4
    child_m = torch.cat([child[:, 7] for child in pool.children])
    assert torch.isfinite(child_m).all()
    assert torch.allclose(child_m, torch.full_like(child_m, 0.99))


def test_mixture_rejects_eta_resampling_and_joint_flow():
    hp = _hp_mixture()
    hp.pv_eta_resample_enabled = True
    episode = _episode(hp, episode_id=1)
    with pytest.raises(ValueError, match="pv_eta_resample_enabled=False"):
        episode._prepare_mixed_policy_value_batches(_firm_table(4), batch_size=2, n_branches=2)

    hp = _hp_mixture()
    hp.pv_training_flow = "joint"
    episode = _episode(hp, episode_id=1)
    with pytest.raises(ValueError, match="pv_training_flow='staged'"):
        episode._prepare_mixed_policy_value_batches(_firm_table(4), batch_size=2, n_branches=2)


def test_coverage_generation_restores_model_modes():
    hp = _hp_mixture()
    episode = _episode(hp, episode_id=1)

    class _SDFModule(torch.nn.Module):
        def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=False):
            assert self.training is False
            n = x_prev.shape[0]
            m = torch.ones(n, dtype=x_prev.dtype, device=x_prev.device)
            return (
                torch.zeros_like(m),
                torch.zeros_like(m),
                m,
                hatcf_prev,
                lnkf_prev,
            )

    sdf = _SDFModule()
    sdf.train(True)
    episode.models["sdf_fc1"] = sdf
    episode.models["policy_value"].train(True)

    pool = episode._build_pv_coverage_pool(n_parent_groups=2, n_branches=2)

    assert pool is not None
    assert episode.models["policy_value"].training is True
    assert episode.models["sdf_fc1"].training is True
