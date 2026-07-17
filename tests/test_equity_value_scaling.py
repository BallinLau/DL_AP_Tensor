from pathlib import Path

import pytest
import torch

from analysis.checkpoint_loader import load_analysis_checkpoint
from config import HyperParams, SIMMODEL
from losses import P0Loss
from models import PolicyValueModel


def _states() -> torch.Tensor:
    return torch.tensor(
        [
            [0.2, -1.0, 1.0, 0.1, -2.0, 0.0, 4.0],
            [0.7, 2.0, 0.0, 0.3, -1.5, 0.2, 4.2],
            [0.4, 3.0, 1.0, 0.2, -1.0, -0.1, 4.1],
        ],
        dtype=torch.float32,
    )


def test_none_mode_matches_legacy_value_outputs():
    torch.manual_seed(1)
    model = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, value_scale_mode="none")
    states = _states()

    raw_v0, raw_vi = model._raw_value_outputs(states)
    out = model(states)
    comp = model.forward_value_components(states)

    assert torch.allclose(out.P0, raw_v0)
    assert torch.allclose(out.PI, raw_vi)
    assert torch.allclose(comp["V0_physical"], raw_v0)
    assert torch.allclose(comp["V0_normalized"], raw_v0)
    assert torch.allclose(comp["value_scale"], torch.ones_like(raw_v0))


def test_exp_xz_physical_output_equals_scale_times_normalized():
    torch.manual_seed(2)
    model = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, value_scale_mode="exp_xz", value_scale_log_max=20.0)
    states = _states()
    comp = model.forward_value_components(states)
    out = model(states)

    expected_scale = 1.0 + torch.exp(states[:, SIMMODEL.X:SIMMODEL.X + 1] + states[:, SIMMODEL.Z:SIMMODEL.Z + 1])
    assert torch.allclose(comp["value_scale"], expected_scale)
    assert torch.allclose(comp["V0_physical"], comp["value_scale"] * comp["V0_normalized"])
    assert torch.allclose(comp["VI_physical"], comp["value_scale"] * comp["VI_normalized"])
    assert torch.allclose(out.P0, comp["V0_physical"])
    assert torch.allclose(out.PI, comp["VI_physical"])


def test_bellman_residual_scale_preserves_zero_and_normalizes():
    loss = P0Loss()
    scale = torch.tensor([[2.0], [4.0]])
    p0 = torch.tensor([[10.0], [20.0]])
    cf = torch.tensor([[1.0], [2.0]])
    m = torch.ones_like(p0)
    child = p0 - cf

    physical = loss.compute_bellman_residual(p0, cf, [m], [child], [torch.zeros_like(p0)])[0]
    normalized = loss.compute_bellman_residual(p0, cf, [m], [child], [torch.zeros_like(p0)], residual_scale=scale)[0]

    assert torch.allclose(physical, torch.zeros_like(physical))
    assert torch.allclose(normalized, physical / scale)

    child_shifted = child - torch.tensor([[1.0], [8.0]])
    physical = loss.compute_bellman_residual(p0, cf, [m], [child_shifted], [torch.zeros_like(p0)])[0]
    normalized = loss.compute_bellman_residual(
        p0, cf, [m], [child_shifted], [torch.zeros_like(p0)], residual_scale=scale
    )[0]
    assert torch.allclose(normalized, physical / scale)


def test_raw_none_checkpoint_refuses_exp_xz_analysis_load(tmp_path: Path):
    ckpt = tmp_path / "policy.pt"
    sdf = tmp_path / "sdf.pt"
    hp_json = tmp_path / "hp.json"

    model = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, value_scale_mode="none")
    torch.save(model.state_dict(), ckpt)
    torch.save({}, sdf)
    hp = HyperParams()
    hp.pv_value_scale_mode = "exp_xz"
    hp.pv_bellman_normalize_by_value_scale = True
    hp_json.write_text(
        '{"pv_value_scale_mode":"exp_xz","pv_bellman_normalize_by_value_scale":true}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="value_parameterization"):
        load_analysis_checkpoint(
            policy_checkpoint=ckpt,
            sdf_checkpoint=sdf,
            hyperparams_json=hp_json,
            device="cpu",
        )
