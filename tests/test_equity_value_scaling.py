from pathlib import Path

import pytest
import torch

from analysis.checkpoint_loader import load_analysis_checkpoint
from analysis.economic_config import AnalysisEconomicConfig
from config import HyperParams, SIMMODEL
from experiments.run_utils import build_models
from experiments.run_scaled_value_ablation import _bellman_loss_and_metrics, _split_value_state
from losses import P0Loss, PILoss
from models import PolicyValueModel
from training.trainer import Trainer


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


def test_pi_bellman_residual_scale_normalizes():
    loss = PILoss()
    scale = torch.tensor([[2.0], [5.0]])
    pi = torch.tensor([[8.0], [12.0]])
    cf = torch.tensor([[1.0], [2.0]])
    m = torch.ones_like(pi)
    child = torch.tensor([[3.0], [4.0]])

    physical = loss.compute_bellman_residual(pi, cf, [m], [child], [torch.zeros_like(pi)])[0]
    normalized = loss.compute_bellman_residual(pi, cf, [m], [child], [torch.zeros_like(pi)], residual_scale=scale)[0]

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


def test_trainer_checkpoint_mode_mismatch_rejected(tmp_path: Path):
    hp_none = HyperParams()
    model = PolicyValueModel()
    trainer = Trainer({"policy_value": model}, hyperparams=hp_none, save_dir=tmp_path / "ckpt", log_dir=tmp_path / "log", device=torch.device("cpu"))
    trainer.save_checkpoint("none")

    hp_scaled = HyperParams()
    hp_scaled.pv_value_scale_mode = "exp_xz"
    hp_scaled.pv_bellman_normalize_by_value_scale = True
    scaled = PolicyValueModel(value_scale_mode="exp_xz")
    trainer_scaled = Trainer({"policy_value": scaled}, hyperparams=hp_scaled, save_dir=tmp_path / "ckpt", log_dir=tmp_path / "log2", device=torch.device("cpu"))

    with pytest.raises(ValueError, match="value_parameterization"):
        trainer_scaled.load_checkpoint("none")


def test_trainer_evaluate_preserves_scaled_mode_and_physical_output(tmp_path: Path):
    hp = HyperParams()
    hp.pv_value_scale_mode = "exp_xz"
    hp.pv_bellman_normalize_by_value_scale = True
    model = PolicyValueModel(value_scale_mode="exp_xz")
    trainer = Trainer({"policy_value": model}, hyperparams=hp, save_dir=tmp_path / "ckpt", log_dir=tmp_path / "log", device=torch.device("cpu"))
    states = _states()
    before = model(states).P0.detach().clone()
    df = torch.zeros(3, 1).numpy()
    import pandas as pd
    eval_df = pd.DataFrame({"b": [0.1, 0.2, 0.3], "P0": [1.0, 2.0, 3.0], "PI": [1.0, 1.5, 2.0]})

    trainer.evaluate(df=eval_df)

    after = model(states).P0.detach()
    assert model.value_scale_mode == "exp_xz"
    assert torch.allclose(before, after)


def _combined_checkpoint(path: Path, *, mode: str = "none") -> None:
    models = build_models(torch.device("cpu"))
    pv = models["policy_value"]
    pv.configure_value_parameterization(mode=mode, log_max=20.0)
    hp = HyperParams()
    hp.pv_value_scale_mode = mode
    hp.pv_value_scale_log_max = 20.0
    hp.pv_bellman_normalize_by_value_scale = mode == "exp_xz"
    torch.save(
        {
            "models": {
                "policy_value": pv.state_dict(),
                "sdf_fc1": models["sdf_fc1"].state_dict(),
                "firm_target": pv.state_dict(),
            },
            "hyperparams": hp.__dict__,
            "config_snapshot": AnalysisEconomicConfig.from_current_config().to_dict(),
            "value_parameterization": {
                "mode": mode,
                "scale_formula": "1+exp(clamp(x+z,max=20))" if mode == "exp_xz" else "1",
                "bellman_normalization": mode == "exp_xz",
                "log_max": 20.0,
            },
        },
        path,
    )


def test_warmstart_scaled_full_checkpoint_reload(tmp_path: Path):
    from scripts.warmstart_scaled_equity_value import main as warmstart_main
    import sys

    baseline = tmp_path / "baseline.pt"
    output = tmp_path / "scaled.pt"
    _combined_checkpoint(baseline, mode="none")
    argv = [
        "warm",
        "--baseline-policy-checkpoint", str(baseline),
        "--output-checkpoint", str(output),
        "--n-states", "16",
        "--epochs", "1",
        "--batch-size", "8",
        "--device", "cpu",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        warmstart_main()
    finally:
        sys.argv = old

    payload = torch.load(output, map_location="cpu")
    assert {"policy_value", "sdf_fc1", "firm_target"}.issubset(payload["models"])
    assert payload["value_parameterization"]["mode"] == "exp_xz"
    load_analysis_checkpoint(output, allow_current_config=True, device="cpu")


def test_bellman_ablation_only_value_params_change_and_teacher_fixed():
    torch.manual_seed(0)
    student = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, value_scale_mode="exp_xz")
    teacher = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, value_scale_mode="exp_xz")
    teacher.load_state_dict(student.state_dict())
    teacher.eval().requires_grad_(False)
    _, non_value_before = _split_value_state(student)
    teacher_before = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    parent = _states()
    batch = {"parent": parent, "children": [parent.clone(), parent.clone()], "m_list": [torch.ones(3, 1), torch.ones(3, 1)]}
    opt = torch.optim.AdamW(
        [p for n, p in student.named_parameters() if n.startswith(("value_encoder", "v0_head", "vi_head"))],
        lr=1e-4,
    )
    loss, metrics = _bellman_loss_and_metrics(student, teacher, [batch], normalize=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    _, non_value_after = _split_value_state(student)

    assert set(metrics).issuperset({"p0_physical_mean_abs", "pi_physical_mean_abs", "p0_normalized_mean_abs", "pi_normalized_mean_abs"})
    assert all(torch.equal(non_value_before[k], non_value_after[k]) for k in non_value_before)
    assert all(torch.equal(teacher_before[k], v) for k, v in teacher.state_dict().items())
