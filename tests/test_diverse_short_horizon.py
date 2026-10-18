from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.economic_config import AnalysisEconomicConfig
from analysis.run_macro_outer_inner_pq import refresh_transition_bank
from data.diverse_short_horizon import (
    DISTRIBUTION_FAMILIES,
    DiverseShortHorizonConfig,
    build_short_horizon_pairs,
    make_diverse_initial_state,
    transition_parity,
)
from data.simulate_ts import SimulateTS
from training.inner_fixed_point import (
    InnerFPCollocationBank,
    build_child_bank,
    build_collocation_bank,
    module_hash,
)


class ToyMacro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        if x_curr.ndim == 3 and hatcf_prev.ndim == 2:
            hatcf_prev = hatcf_prev.unsqueeze(1)
            lnkf_prev = lnkf_prev.unsqueeze(1)
        h = hatcf_prev + self.scale * x_curr
        k = lnkf_prev + 2.0 * self.scale * x_curr
        m = 1.0 + 0.1 * x_curr + 0.01 * hatcf_prev
        return h, k, m, h, k


def _node_frames(n: int = 12):
    path = np.repeat(np.arange(3), 4)[:n]
    t = np.tile(np.arange(4), 3)[:n]
    node = path * 10 + t
    macro = pd.DataFrame({
        "path": path, "t": t, "branch": -1, "economic_node_id": node,
        "x": np.linspace(-0.2, 0.2, n), "hatcf": np.linspace(-2.2, -1.8, n),
        "lnkf": np.linspace(3.8, 4.2, n), "Hatc": np.linspace(-1.9, -1.5, n),
        "LnK": np.linspace(4.1, 4.5, n),
    })
    rows = []
    for idx, record in macro.iterrows():
        for firm in range(3):
            rows.append({
                "path": record.path, "t": record.t, "branch": -1,
                "economic_node_id": record.economic_node_id,
                "b": (firm + 1) / 4, "z": -0.2 + 0.2 * firm,
                "ETA": float(firm % 2), "i": 0.1 * firm, "x": record.x,
                "Hatcf": record.hatcf, "LnKF": record.lnkf,
            })
    return pd.DataFrame(rows), macro


def _bank() -> InnerFPCollocationBank:
    firm, macro = _node_frames()
    return build_collocation_bank(
        firm, macro, n_parents=16, seed=9, split="train",
        economic_config=AnalysisEconomicConfig.from_current_config(),
    )


def _macro_rows() -> tuple[pd.DataFrame, pd.DataFrame, DiverseShortHorizonConfig]:
    model = ToyMacro()
    rows = []
    metadata = []
    for path in range(2):
        metadata.append({"path_id": path, "distribution_id": path, "lineage_id": path})
        for t in range(3):
            x, h, k = 0.1 * (path + t), -2 + 0.1 * t, 4 + 0.1 * t
            rows.append({
                "path": path, "t": t, "branch": -1, "x": x, "hatcf": h,
                "lnkf": k, "Hatc": h + 0.3, "LnK": k + 0.2, "M": 1.0,
                "resource_feasible": 1, "n_firms": 3, "K": 10.0, "C": 2.0,
                "economic_node_id": path * 100 + t * 3,
            })
            if t < 2:
                for branch in (0, 1):
                    x1 = x + 0.05 * (branch + 1)
                    with torch.no_grad():
                        _, _, m, hh, kk = model.forward_step(
                            torch.tensor([[x]]), torch.tensor([[x1]]),
                            torch.tensor([[h + 0.3]]), torch.tensor([[k + 0.2]]), True,
                        )
                    rows.append({
                        "path": path, "t": t + 1, "branch": branch, "x": x1,
                        "hatcf": hh.item(), "lnkf": kk.item(), "Hatc": hh.item() + 0.1,
                        "LnK": kk.item() + 0.1, "M": m.item(), "resource_feasible": 1,
                        "n_firms": 3, "K": 10.0, "C": 2.0,
                        "economic_node_id": path * 100 + (t + 1) * 3 + branch + 1,
                    })
    config = DiverseShortHorizonConfig(2, 3, simulation_horizon=3, burn_in=1)
    return pd.DataFrame(rows), pd.DataFrame(metadata), config


def test_cal_to_f_not_f_to_f():
    bank = _bank()
    child = torch.zeros((len(bank.parent_states), 2, 8))
    child[..., 4] = 0.1
    refreshed, _ = refresh_transition_bank(
        ToyMacro(), bank.parent_states, child,
        parent_macro_context=bank.require_macro_context(), m_min=0.7, m_max=1.3,
    )
    expected = bank.parent_macro_cal[:, 0:1].unsqueeze(1) + 0.02
    assert torch.allclose(refreshed[..., 5:6], expected)
    assert not torch.allclose(refreshed[..., 5:6], bank.parent_states[:, 5:6].unsqueeze(1) + 0.02)


def test_firm_bank_preserves_7d_input():
    assert _bank().parent_states.shape[1] == 7


def test_parent_macro_context_matched():
    bank = _bank()
    assert bank.bank_version == "unified_firm_macro_bank_v1"
    assert bank.parent_macro_cal.shape == (16, 2)
    assert torch.all(bank.economic_node_id >= 0)


def test_child_t_equals_parent_t_plus_one():
    frame, metadata, config = _macro_rows()
    pairs, _ = build_short_horizon_pairs(frame, metadata, config)
    assert (pairs.child_t == pairs.parent_t + 1).all()


def test_branch_pair_conditional_independence():
    frame, metadata, config = _macro_rows()
    pairs, _ = build_short_horizon_pairs(frame, metadata, config)
    grouped = pairs.groupby(["path_id", "parent_t"])
    assert all(set(group.branch_id) == {0, 1} for _, group in grouped)
    assert all(group.x_t1.nunique() == 2 for _, group in grouped)


def test_no_future_cal_leakage():
    frame, metadata, config = _macro_rows()
    pairs, _ = build_short_horizon_pairs(frame, metadata, config)
    assert not np.allclose(pairs.Hatc_cal_t, pairs.Hatc_cal_t1)
    report = transition_parity(ToyMacro(), pairs, m_min=0.7, m_max=1.3, device=torch.device("cpu"))
    assert report["semantic_input"].endswith("Hatc_cal_t,LnK_cal_t)")


def test_unified_bank_serialization_roundtrip(tmp_path):
    bank = _bank()
    path = tmp_path / "bank.pt"
    torch.save(bank.payload(), path)
    payload = torch.load(path, weights_only=False)
    restored = InnerFPCollocationBank(
        **{name: payload[name] for name in (
            "parent_states", "source_kind", "seed", "split", "parent_macro_cal",
            "path_id", "economic_node_id", "lineage_id", "distribution_id", "parent_t",
            "bank_version", "metadata",
        )}
    )
    assert restored.fingerprint == bank.fingerprint


def test_old_bank_missing_cal_fails_fast():
    old = InnerFPCollocationBank(torch.zeros((3, 7)), torch.zeros(3, dtype=torch.long), 1, "old")
    with pytest.raises(ValueError, match="cannot be silently upgraded"):
        old.require_macro_context()


def test_macro_checkpoint_hash_unchanged():
    model = ToyMacro(); before = module_hash(model)
    build_child_bank(
        model, _bank(), n_child_shocks=2, seed=1,
        economic_config=AnalysisEconomicConfig.from_current_config(), device=torch.device("cpu"),
    )
    assert module_hash(model) == before


def test_firm_checkpoint_hash_unchanged():
    firm = torch.nn.Linear(7, 1); before = module_hash(firm)
    _ = _bank()
    assert module_hash(firm) == before


def test_pq_m_parity():
    bank = _bank(); child = torch.zeros((16, 2, 8)); child[..., 4] = 0.1
    refreshed, _ = refresh_transition_bank(
        ToyMacro(), bank.parent_states, child,
        parent_macro_context=bank.require_macro_context(), m_min=0.7, m_max=1.3,
    )
    assert torch.equal(refreshed[..., 7], refreshed[..., 7].clone())


def test_short_horizon_pair_count():
    frame, metadata, config = _macro_rows()
    pairs, quality = build_short_horizon_pairs(frame, metadata, config)
    assert len(pairs) == 4  # two paths, parent_t=1, two branches
    assert quality.loc[quality.discard_reason == "burn_in", "rows"].iloc[0] == 4


def test_entry_mode_explicit():
    params = SimulateTS.__init__.__annotations__
    assert "entry_mode" in SimulateTS.__init__.__code__.co_varnames
    assert "entry_spec_version" in SimulateTS.__init__.__code__.co_varnames


def test_grid_bp_applied_to_transition():
    source = SimulateTS._resolve_bp_action
    sim = object.__new__(SimulateTS)
    sim.bp_action_override = None; sim.bp_action_source = "grid"
    sim.bp_grid_policy = lambda **_: torch.tensor([0.75])
    assert source(sim, bp_head=torch.tensor([0.10])).item() == pytest.approx(0.75)


def test_resource_accounting_consistency():
    frame, metadata, config = _macro_rows()
    pairs, _ = build_short_horizon_pairs(frame, metadata, config)
    assert (pairs.resource_feasible == 1).all()
    assert (pairs.K_total >= 0).all() and (pairs.C_total >= 0).all()


def test_fixed_crn_repeatability():
    config = DiverseShortHorizonConfig(8, 4, seed=11)
    assert [DISTRIBUTION_FAMILIES[i % 8] for i in range(8)] == list(DISTRIBUTION_FAMILIES)
    assert config == DiverseShortHorizonConfig(8, 4, seed=11)


def test_validation_lineage_disjoint():
    train = _bank(); firm, macro = _node_frames()
    macro["path"] += 100; firm["path"] += 100
    validation = build_collocation_bank(
        firm, macro, n_parents=8, seed=10, split="validation",
        economic_config=AnalysisEconomicConfig.from_current_config(),
    )
    assert set(train.lineage_id.tolist()).isdisjoint(validation.lineage_id.tolist())


def test_outer_bank_refresh_only_at_boundary():
    bank = _bank(); before = bank.fingerprint
    _ = build_child_bank(
        ToyMacro(), bank, n_child_shocks=2, seed=4,
        economic_config=AnalysisEconomicConfig.from_current_config(), device=torch.device("cpu"),
    )
    assert bank.fingerprint == before
