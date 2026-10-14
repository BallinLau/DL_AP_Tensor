from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from analysis.economic_config import AnalysisEconomicConfig
from training.episode import Episode
from training.inner_fixed_point import (
    InnerFPCollocationBank,
    InnerFPChildBank,
    InnerFixedPointSolver,
    InnerFPSolverConfig,
    build_collocation_bank,
    module_hash,
    named_parameter_hash,
)


class _Parts(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_encoder = nn.Linear(1, 1)
        self.v0_head = nn.Linear(1, 1)
        self.vi_head = nn.Linear(1, 1)
        self.barz_model = nn.Linear(1, 1)
        self.bari_model = nn.Linear(1, 1)
        self.q_encoder = nn.Linear(1, 1)
        self.q_head = nn.Linear(1, 1)
        self.policy_encoder = nn.Linear(1, 1)
        self.bp0_head = nn.Linear(1, 1)
        self.bpi_head = nn.Linear(1, 1)
        self.q_parameterization = "direct"


def _parent_bank(seed: int = 1) -> InnerFPCollocationBank:
    generator = torch.Generator().manual_seed(seed)
    return InnerFPCollocationBank(
        parent_states=torch.rand((8, 7), generator=generator),
        source_kind=torch.tensor([0, 1] * 4),
        seed=seed,
        split="test",
    )


def _child_bank(seed: int = 2) -> InnerFPChildBank:
    generator = torch.Generator().manual_seed(seed)
    shape = (8, 3, 1)
    return InnerFPChildBank(
        children=torch.rand((8, 3, 8), generator=generator),
        eps_x=torch.rand(shape, generator=generator),
        eps_z=torch.rand(shape, generator=generator),
        u_eta=torch.rand(shape, generator=generator),
        u_i=torch.rand(shape, generator=generator),
        seed=seed,
    )


def test_frozen_macro_hash_changes_if_and_only_if_module_changes():
    module = nn.Linear(2, 1)
    before = module_hash(module)
    assert module_hash(module) == before
    with torch.no_grad():
        module.weight.add_(1.0)
    assert module_hash(module) != before


def test_fixed_parent_bank_fingerprint_detects_mutation():
    bank = _parent_bank()
    before = bank.fingerprint
    bank.parent_states[0, 0] += 1
    assert bank.fingerprint != before


def test_fixed_child_shock_bank_fingerprint_detects_mutation():
    bank = _child_bank()
    before = bank.fingerprint
    bank.children[0, 0, 0] += 1
    assert bank.fingerprint != before


def test_p_partition_hash_changes_without_q_partition_change():
    model = _Parts()
    q_before = named_parameter_hash(model, InnerFixedPointSolver.Q_PREFIXES)
    with torch.no_grad():
        model.v0_head.weight.add_(1)
    assert named_parameter_hash(model, InnerFixedPointSolver.Q_PREFIXES) == q_before


def test_q_partition_hash_changes_without_p_partition_change():
    model = _Parts()
    p_before = named_parameter_hash(model, InnerFixedPointSolver.P_PREFIXES)
    with torch.no_grad():
        model.q_head.weight.add_(1)
    assert named_parameter_hash(model, InnerFixedPointSolver.P_PREFIXES) == p_before


def test_bp_partition_remains_unchanged_during_p_and_q_mutations():
    model = _Parts()
    before = named_parameter_hash(model, InnerFixedPointSolver.BP_PREFIXES)
    with torch.no_grad():
        model.v0_head.bias.add_(1)
        model.q_head.bias.add_(1)
    assert named_parameter_hash(model, InnerFixedPointSolver.BP_PREFIXES) == before


def test_sdf_fc1_hash_is_independent_of_firm_model_updates():
    sdf = nn.Linear(2, 1)
    firm = _Parts()
    before = module_hash(sdf)
    with torch.no_grad():
        firm.v0_head.weight.add_(1)
        firm.q_head.weight.add_(1)
    assert module_hash(sdf) == before


def test_q_stage_reuses_one_previous_sweep_target_for_every_phase():
    episode = object.__new__(Episode)
    online = _Parts()
    fixed = deepcopy(online).eval().requires_grad_(False)
    episode.models = {"policy_value": online}
    episode.hyperparams = SimpleNamespace(
        q_target_refresh_mode="phase",
        q_epoch_validation_enabled=False,
        q_zero_boundary_epochs=1,
        q_default_pretrain_epochs=1,
        q_survival_aio_epochs=1,
        q_mixed_polish_epochs=1,
        q_require_zero_phase=False,
        q_require_default_phase=False,
        q_require_survival_phase=False,
        policy_lr=1e-3,
        q_stage_lr=None,
        save_intermediate_stage_checkpoints=False,
    )
    episode.device = torch.device("cpu")
    episode._q_no_improvement_streak = 0
    episode._intermediate_stage_states = {}
    episode._stage_component_hashes = {}
    episode._state_dict_hash = lambda model: module_hash(model)
    episode._policy_value_stage_params = lambda _stage: list(online.q_head.parameters())
    episode._params_content_hash = lambda params: str(sum(float(p.detach().sum()) for p in params))
    observed = []

    def phase(**kwargs):
        observed.append(kwargs["q_target_model"])
        return {"phase": kwargs["phase"], "status": "accepted", "optimizer_steps": 0,
                "coverage": {}}

    episode._run_q_regime_phase = phase
    summary = episode._run_q_regime_training(
        [{"parent": torch.zeros((2, 7))}], deepcopy(online),
        fixed_q_target_model=fixed,
    )
    assert summary["q_target_refresh_mode"] == "fixed_sweep_snapshot"
    assert len(observed) == 4 and all(item is fixed for item in observed)


def test_inner_solver_source_contains_no_simulation_or_entry_call():
    source = inspect.getsource(InnerFixedPointSolver.run)
    assert "simulate(" not in source
    assert "run_episode(" not in source
    assert "entry" not in source.lower()


def test_current_residual_evaluator_uses_current_model_on_both_sides():
    import training.inner_fixed_point as module
    source = inspect.getsource(module.evaluate_inner_operator)
    assert "q_target_model=model" in source
    assert "pred_p0, pred_pi = model._value_outputs(parent)" in source
    assert "episode._compute_q_bellman_signed_residuals" in source


def test_collocation_bank_is_deterministic_for_same_seed():
    econ = AnalysisEconomicConfig.from_current_config()
    rng = np.random.default_rng(4)
    firm = pd.DataFrame({"z": rng.normal(size=100)})
    macro = pd.DataFrame({
        "x": rng.normal(size=100),
        "hatcf": rng.normal(size=100),
        "lnkf": rng.normal(size=100),
    })
    first = build_collocation_bank(
        firm, macro, n_parents=64, seed=99, split="train", economic_config=econ
    )
    second = build_collocation_bank(
        firm, macro, n_parents=64, seed=99, split="train", economic_config=econ
    )
    assert first.fingerprint == second.fingerprint
    assert first.metadata["exact_b0_count"] > 0
    assert set(first.parent_states[:, 2].tolist()) == {0.0, 1.0}


def test_checkpoint_resume_restores_exact_sweep_state(tmp_path: Path):
    original = object.__new__(InnerFixedPointSolver)
    original.output_dir = tmp_path
    original.model = _Parts()
    original.sdf_hash = "sdf"
    original.bank_hashes = {"bank": "fixed"}
    original.source_provenance = {"source": "ep1"}
    original.config = InnerFPSolverConfig(max_sweeps=2)
    original.history = [{"sweep": 1}]
    original.previous_values = {"q": torch.tensor([1.0])}
    original.previous_previous_values = {"q": torch.tensor([0.5])}
    original.previous_boundary = {"default": torch.tensor([False])}
    original.previous_previous_boundary = {"default": torch.tensor([False])}
    original.previous_policies = {"b0_star": torch.tensor([0.2])}
    original.consecutive_passes = 1
    path = original._save_checkpoint(1, "resume.pt", {"stage": "post_q"})

    resumed = object.__new__(InnerFixedPointSolver)
    resumed.episode = SimpleNamespace(device=torch.device("cpu"))
    resumed.model = _Parts()
    resumed.sdf_hash = "sdf"
    resumed.bank_hashes = {"bank": "fixed"}
    assert resumed.load_resume(path) == 2
    for key, value in original.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key])
    assert resumed.history == original.history
    assert torch.equal(resumed.previous_values["q"], torch.tensor([1.0]))
