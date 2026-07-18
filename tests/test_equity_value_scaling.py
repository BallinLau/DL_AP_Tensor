import json
import sys
from pathlib import Path

import pytest
import torch

from analysis.checkpoint_loader import load_analysis_checkpoint
from analysis.economic_config import AnalysisEconomicConfig
from config import Config, HyperParams, SIMMODEL
from experiments.run_utils import build_models
from experiments.run_scaled_value_ablation import (
    _accept_round,
    _bellman_economic_spec,
    _bellman_loss_and_metrics,
    _load_batches,
    _set_full_eval_mode,
    _set_value_training_mode,
    _split_value_state,
    _validate_cli_args,
)
from losses import P0Loss, PILoss
from models import PolicyValueModel, build_policy_value_from_checkpoint_spec
from training.trainer import Trainer


def _econ(**overrides) -> dict:
    spec = {
        "DELTA": 0.02,
        "TAU": 0.2,
        "KAPPA_B": 0.004,
        "KAPPA_E": 0.025,
        "AIO_WEIGHT": 0.5,
        "G": 1.14,
    }
    spec.update(overrides)
    return spec


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


def test_cal_phats_uses_instance_i_grid_not_global_config(monkeypatch):
    model = PolicyValueModel(
        share_hidden_dims=[8],
        share_output_dim=8,
        i_grid_size=3,
        i_threshold=0.3,
    )
    states = _states()
    seen_i = []

    def fake_value_outputs(firm_state):
        seen_i.append(firm_state[:, SIMMODEL.I].detach().clone())
        value = firm_state[:, SIMMODEL.I:SIMMODEL.I + 1]
        return value, value

    monkeypatch.setattr(model, "_value_outputs", fake_value_outputs)
    monkeypatch.setattr("models.policy_value.Config.PV_I_GRID_SIZE", 99, raising=False)
    monkeypatch.setattr("models.policy_value.Config.I_THRESHOLD", 9.0, raising=False)

    model.cal_phats(states)

    assert len(seen_i) == 3
    grid = torch.stack([x[0] for x in seen_i]).reshape(-1)
    assert torch.allclose(grid, torch.tensor([0.0, 0.15, 0.3]))


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
            "optimizers": {"policy_value": {"legacy": True}, "sdf_fc1": {"keep": True}},
            "hyperparams": hp.__dict__,
            "config_snapshot": AnalysisEconomicConfig.from_current_config().to_dict(),
            "policy_value_model_spec": pv.model_spec(),
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
    assert payload["warmstart"]["resume_optimizer_compatible"] is False
    assert "policy_value" not in payload.get("optimizers", {})
    assert "sdf_fc1" in payload.get("optimizers", {})
    for key in payload["models"]["policy_value"]:
        assert torch.equal(payload["models"]["policy_value"][key], payload["models"]["firm_target"][key])
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
    batch = {
        "parent": parent,
        "children": [parent.clone(), parent.clone()],
        "m_list": [torch.ones(3, 1), torch.ones(3, 1)],
        "branch_weights": torch.full((3, 2), 0.5),
    }
    opt = torch.optim.AdamW(
        [p for n, p in student.named_parameters() if n.startswith(("value_encoder", "v0_head", "vi_head"))],
        lr=1e-4,
    )
    loss, metrics = _bellman_loss_and_metrics(student, teacher, [batch], normalize=True, economic_spec=_econ())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    _, non_value_after = _split_value_state(student)

    assert set(metrics).issuperset({
        "p0_physical_conditional_mean_abs",
        "pi_physical_conditional_mean_abs",
        "p0_normalized_conditional_mean_abs",
        "pi_normalized_conditional_mean_abs",
    })
    assert "regions" in metrics
    assert all(torch.equal(non_value_before[k], non_value_after[k]) for k in non_value_before)
    assert all(torch.equal(teacher_before[k], v) for k, v in teacher.state_dict().items())


def test_value_training_mode_only_enables_value_modules():
    model = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, dropout=0.5)
    _set_value_training_mode(model)
    assert model.value_encoder.training is True
    assert model.v0_head.training is True
    assert model.vi_head.training is True
    assert model.q_encoder.training is False
    assert model.policy_encoder.training is False
    assert model.q_head.training is False
    assert model.bp0_head.training is False
    assert model.bpi_head.training is False
    assert model.barz_model.training is False
    assert model.bari_model.training is False
    _set_full_eval_mode(model)
    assert model.training is False


def test_dropout_validation_metrics_are_repeatable_and_leave_eval_mode():
    torch.manual_seed(123)
    student = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, dropout=0.5, value_scale_mode="exp_xz")
    teacher = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, dropout=0.5, value_scale_mode="exp_xz")
    teacher.load_state_dict(student.state_dict())
    parent = _states()
    batch = {
        "parent": parent,
        "children": [parent.clone(), parent.clone()],
        "m_list": [torch.ones(3, 1), torch.ones(3, 1)],
        "branch_weights": torch.full((3, 2), 0.5),
    }
    _set_full_eval_mode(student)
    _set_full_eval_mode(teacher)
    with torch.no_grad():
        _, first = _bellman_loss_and_metrics(student, teacher, [batch], normalize=True, economic_spec=_econ())
        _, second = _bellman_loss_and_metrics(student, teacher, [batch], normalize=True, economic_spec=_econ())
    for key in ("p0_physical_conditional_mean_abs", "pi_physical_conditional_mean_abs"):
        assert first[key] == pytest.approx(second[key])
    assert student.training is False


def test_scaled_ablation_batch_loader_requires_explicit_m_and_weights(tmp_path: Path):
    parent = _states()
    batch = {
        "parent": parent,
        "children": [parent.clone(), parent.clone()],
        "m_list": [torch.ones(3, 1), torch.ones(3, 1)],
        "branch_weights": torch.full((3, 2), 0.5),
    }
    path = tmp_path / "batches.pt"
    torch.save(
        {
            "train": [batch],
            "validation": [batch],
            "metadata": {"m_semantics": "raw", "shock_bank_hash": "abc"},
        },
        path,
    )
    train, val, meta = _load_batches(path, torch.device("cpu"))
    assert len(train) == 1
    assert len(val) == 1
    assert meta["m_semantics"] == "raw"

    missing_m = dict(batch)
    missing_m.pop("m_list")
    torch.save(
        {
            "train": [missing_m],
            "validation": [batch],
            "metadata": {"m_semantics": "raw", "shock_bank_hash": "abc"},
        },
        path,
    )
    with pytest.raises(ValueError, match="m_list|M_list|M"):
        _load_batches(path, torch.device("cpu"))

    missing_w = dict(batch)
    missing_w.pop("branch_weights")
    torch.save(
        {
            "train": [missing_w],
            "validation": [batch],
            "metadata": {"m_semantics": "raw", "shock_bank_hash": "abc"},
        },
        path,
    )
    with pytest.raises(ValueError, match="branch_weights"):
        _load_batches(path, torch.device("cpu"))
    train, _, _ = _load_batches(path, torch.device("cpu"), assume_equal_branch_weights=True)
    assert torch.allclose(train[0]["branch_weights"], torch.full((3, 2), 0.5))


def test_checkpoint_economic_parameters_override_runtime_config(monkeypatch):
    parent = _states()
    batch = {
        "parent": parent,
        "children": [parent.clone(), parent.clone()],
        "m_list": [torch.ones(3, 1), torch.ones(3, 1)],
        "branch_weights": torch.full((3, 2), 0.5),
    }
    model = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, value_scale_mode="none")
    model.eval()
    spec = _econ(DELTA=0.123, TAU=0.456, KAPPA_B=0.078, KAPPA_E=0.091, AIO_WEIGHT=0.3, G=1.07)
    _, before = _bellman_loss_and_metrics(model, model, [batch], normalize=False, economic_spec=spec)
    monkeypatch.setattr(Config, "DELTA", 9.0, raising=False)
    monkeypatch.setattr(Config, "TAU", 9.0, raising=False)
    monkeypatch.setattr(Config, "KAPPA_B", 9.0, raising=False)
    monkeypatch.setattr(Config, "KAPPA_E", 9.0, raising=False)
    monkeypatch.setattr(Config, "AIO_WEIGHT", 9.0, raising=False)
    monkeypatch.setattr(Config, "G", 9.0, raising=False)
    _, after = _bellman_loss_and_metrics(model, model, [batch], normalize=False, economic_spec=spec)
    for key in ("p0_physical_conditional_mean_abs", "pi_physical_conditional_mean_abs"):
        assert before[key] == pytest.approx(after[key])


def test_z_region_defaults_are_minus_two_and_two():
    states = torch.tensor(
        [
            [0.2, -3.0, 1.0, 0.1, -2.0, 0.0, 4.0],
            [0.2, 0.0, 1.0, 0.1, -2.0, 0.0, 4.0],
            [0.2, 3.0, 1.0, 0.1, -2.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    model = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8)
    batch = {
        "parent": states,
        "children": [states.clone(), states.clone()],
        "m_list": [torch.ones(3, 1), torch.ones(3, 1)],
        "branch_weights": torch.full((3, 2), 0.5),
    }
    _, metrics = _bellman_loss_and_metrics(model, model, [batch], normalize=False, economic_spec=_econ())
    assert metrics["regions"]["p0"]["low_z"]["n_parent"] == 1
    assert metrics["regions"]["p0"]["mid_z"]["n_parent"] == 1
    assert metrics["regions"]["p0"]["high_z"]["n_parent"] == 1


def test_candidate_improves_current_teacher_even_if_above_historical_best():
    historical_best = 1.0
    start_score = 10.0
    candidate_score = 8.0
    assert candidate_score > historical_best
    assert _accept_round(start_score, candidate_score, min_round_improvement=0.0) is True


def test_scaled_ablation_cli_validation_rejects_invalid_inputs():
    import argparse

    args = argparse.Namespace(min_round_improvement=0.0, low_z_cutoff=-2.0, high_z_cutoff=2.0, rounds=1, epochs_per_round=1, lr=1e-5)
    _validate_cli_args(args)
    for field, value, match in [
        ("min_round_improvement", -1.0, "nonnegative"),
        ("low_z_cutoff", 3.0, "low-z-cutoff"),
        ("rounds", 0, "rounds"),
        ("epochs_per_round", 0, "epochs-per-round"),
        ("lr", 0.0, "lr"),
    ]:
        bad = argparse.Namespace(**vars(args))
        setattr(bad, field, value)
        with pytest.raises(ValueError, match=match):
            _validate_cli_args(bad)


def test_scaled_ablation_rejects_non_double_sampling_and_weighted_aio(tmp_path: Path):
    parent = _states()
    base_batch = {
        "parent": parent,
        "children": [parent.clone(), parent.clone()],
        "m_list": [torch.ones(3, 1), torch.ones(3, 1)],
        "branch_weights": torch.full((3, 2), 0.5),
    }
    path = tmp_path / "batches.pt"
    bad_three = dict(base_batch)
    bad_three["children"] = [parent.clone(), parent.clone(), parent.clone()]
    bad_three["m_list"] = [torch.ones(3, 1), torch.ones(3, 1), torch.ones(3, 1)]
    bad_three["branch_weights"] = torch.full((3, 3), 1.0 / 3.0)
    torch.save({"train": [bad_three], "validation": [base_batch], "metadata": {"m_semantics": "raw", "shock_bank_hash": "x"}}, path)
    with pytest.raises(ValueError, match="len\\(children\\) must equal 2"):
        _load_batches(path, torch.device("cpu"))

    bad_weight = dict(base_batch)
    bad_weight["branch_weights"] = torch.tensor([[0.75, 0.25], [0.5, 0.5], [0.5, 0.5]])
    torch.save({"train": [bad_weight], "validation": [base_batch], "metadata": {"m_semantics": "raw", "shock_bank_hash": "x"}}, path)
    with pytest.raises(ValueError, match="weighted AiO"):
        _load_batches(path, torch.device("cpu"))


def test_policy_value_model_spec_round_trip_covers_all_head_dims():
    model = PolicyValueModel(
        share_hidden_dims=[7],
        share_output_dim=9,
        q_head_dims=[5],
        p0_head_dims=[6],
        pi_head_dims=[4],
        bp0_head_dims=[3],
        bpi_head_dims=[2],
        barz_hidden_dims=[8, 4],
        bari_hidden_dims=[5, 3],
        tau_i=0.11,
        tau_z=0.22,
        i_grid_size=5,
        i_threshold=0.4,
        delta=0.03,
        phi=0.44,
        g=1.05,
    )
    spec = model.model_spec()
    rebuilt = build_policy_value_from_checkpoint_spec({"policy_value_model_spec": spec})
    assert rebuilt.model_spec() == spec
    for key in (
        "q_head_dims",
        "p0_head_dims",
        "pi_head_dims",
        "bp0_head_dims",
        "bpi_head_dims",
        "barz_hidden_dims",
        "bari_hidden_dims",
    ):
        assert key in spec


def _hash_state(state):
    import hashlib

    h = hashlib.sha256()
    if state is None:
        return None
    for key in sorted(state):
        value = state[key]
        h.update(key.encode())
        if torch.is_tensor(value):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def _file_sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_legacy_annotation_preserves_state_hashes_and_warmstarts(tmp_path: Path):
    from scripts.annotate_legacy_equity_checkpoint import main as annotate_main
    from scripts.warmstart_scaled_equity_value import main as warmstart_main

    legacy = tmp_path / "legacy.pt"
    annotated = tmp_path / "annotated.pt"
    warmed = tmp_path / "warmed.pt"
    _combined_checkpoint(legacy, mode="none")
    legacy_payload = torch.load(legacy, map_location="cpu")
    legacy_payload.pop("value_parameterization")
    legacy_payload.pop("policy_value_model_spec")
    legacy_payload.pop("config_snapshot")
    torch.save(legacy_payload, legacy)

    model_spec_json = tmp_path / "model_spec.json"
    config_json = tmp_path / "config.json"
    model_spec_json.write_text(json.dumps({"policy_value_model_spec": build_models(torch.device("cpu"))["policy_value"].model_spec()}), encoding="utf-8")
    config_json.write_text(json.dumps({"config_snapshot": AnalysisEconomicConfig.from_current_config().to_dict()}), encoding="utf-8")

    old_argv = sys.argv
    try:
        sys.argv = [
            "annotate",
            "--legacy-checkpoint", str(legacy),
            "--model-spec-json", str(model_spec_json),
            "--economic-config-json", str(config_json),
            "--output-checkpoint", str(annotated),
        ]
        annotate_main()
    finally:
        sys.argv = old_argv

    ann = torch.load(annotated, map_location="cpu")
    for module_key in ("policy_value", "sdf_fc1", "firm_target"):
        assert _hash_state(ann["models"].get(module_key)) == _hash_state(legacy_payload["models"].get(module_key))
    assert _hash_state(ann.get("optimizers")) == _hash_state(legacy_payload.get("optimizers"))
    assert ann["value_parameterization"]["mode"] == "none"

    try:
        sys.argv = [
            "warm",
            "--baseline-policy-checkpoint", str(annotated),
            "--output-checkpoint", str(warmed),
            "--n-states", "16",
            "--epochs", "1",
            "--batch-size", "8",
            "--device", "cpu",
        ]
        warmstart_main()
    finally:
        sys.argv = old_argv
    assert warmed.exists()


def test_raw_components_legacy_annotation_packaging_smoke(tmp_path: Path):
    from scripts.annotate_legacy_equity_checkpoint import main as annotate_main

    models = build_models(torch.device("cpu"))
    policy_path = tmp_path / "policy.pt"
    sdf_path = tmp_path / "sdf.pt"
    hp_path = tmp_path / "hp.json"
    model_spec_json, config_json = _write_json_payloads(tmp_path)
    output = tmp_path / "raw_annotated.pt"
    torch.save(models["policy_value"].state_dict(), policy_path)
    torch.save(models["sdf_fc1"].state_dict(), sdf_path)
    hp_path.write_text(json.dumps(HyperParams().__dict__, default=str), encoding="utf-8")

    old_argv = sys.argv
    try:
        sys.argv = [
            "annotate",
            "--policy-checkpoint", str(policy_path),
            "--sdf-checkpoint", str(sdf_path),
            "--hyperparams-json", str(hp_path),
            "--model-spec-json", str(model_spec_json),
            "--economic-config-json", str(config_json),
            "--output-checkpoint", str(output),
        ]
        annotate_main()
    finally:
        sys.argv = old_argv

    payload = torch.load(output, map_location="cpu")
    assert payload["legacy_annotation"]["source_format"] == "raw_components"
    assert _hash_state(payload["models"]["policy_value"]) == _hash_state(torch.load(policy_path, map_location="cpu"))
    assert _hash_state(payload["models"]["sdf_fc1"]) == _hash_state(torch.load(sdf_path, map_location="cpu"))
    assert _hash_state(payload["models"]["firm_target"]) == _hash_state(torch.load(policy_path, map_location="cpu"))


def test_legacy_annotation_refuses_existing_scaled_checkpoint(tmp_path: Path):
    from scripts.annotate_legacy_equity_checkpoint import main as annotate_main

    scaled = tmp_path / "scaled.pt"
    _combined_checkpoint(scaled, mode="exp_xz")
    model_spec_json, config_json = _write_json_payloads(tmp_path)
    old_argv = sys.argv
    try:
        sys.argv = [
            "annotate",
            "--legacy-checkpoint", str(scaled),
            "--model-spec-json", str(model_spec_json),
            "--economic-config-json", str(config_json),
            "--output-checkpoint", str(tmp_path / "bad.pt"),
        ]
        with pytest.raises(ValueError, match="Refusing to annotate"):
            annotate_main()
    finally:
        sys.argv = old_argv


def _write_json_payloads(tmp_path: Path) -> tuple[Path, Path]:
    model_spec_json = tmp_path / "model_spec.json"
    config_json = tmp_path / "config.json"
    model_spec_json.write_text(
        json.dumps({"policy_value_model_spec": build_models(torch.device("cpu"))["policy_value"].model_spec()}),
        encoding="utf-8",
    )
    config_json.write_text(json.dumps({"config_snapshot": AnalysisEconomicConfig.from_current_config().to_dict()}), encoding="utf-8")
    return model_spec_json, config_json


def _annotate_legacy(tmp_path: Path, legacy: Path, annotated: Path) -> None:
    from scripts.annotate_legacy_equity_checkpoint import main as annotate_main

    model_spec_json, config_json = _write_json_payloads(tmp_path)
    old_argv = sys.argv
    try:
        sys.argv = [
            "annotate",
            "--legacy-checkpoint", str(legacy),
            "--model-spec-json", str(model_spec_json),
            "--economic-config-json", str(config_json),
            "--output-checkpoint", str(annotated),
        ]
        annotate_main()
    finally:
        sys.argv = old_argv


def _warmstart(tmp_path: Path, baseline: Path, warmed: Path) -> None:
    from scripts.warmstart_scaled_equity_value import main as warmstart_main

    old_argv = sys.argv
    try:
        sys.argv = [
            "warm",
            "--baseline-policy-checkpoint", str(baseline),
            "--output-checkpoint", str(warmed),
            "--n-states", "16",
            "--epochs", "1",
            "--batch-size", "8",
            "--device", "cpu",
        ]
        warmstart_main()
    finally:
        sys.argv = old_argv


def test_minimal_scaled_ablation_smoke_and_posttrain_strict_reload(tmp_path: Path):
    from experiments.run_scaled_value_ablation import main as ablation_main

    legacy = tmp_path / "legacy.pt"
    annotated = tmp_path / "annotated.pt"
    warmed = tmp_path / "warmed.pt"
    _combined_checkpoint(legacy, mode="none")
    legacy_payload = torch.load(legacy, map_location="cpu")
    legacy_payload.pop("value_parameterization")
    legacy_payload.pop("policy_value_model_spec")
    legacy_payload.pop("config_snapshot")
    torch.save(legacy_payload, legacy)
    _annotate_legacy(tmp_path, legacy, annotated)
    _warmstart(tmp_path, annotated, warmed)

    parent = torch.cat([_states(), _states(), _states(), _states(), _states(), _states()[:1]], dim=0)[:16]
    branch_weights = torch.full((16, 2), 0.5)
    batch = {
        "parent": parent,
        "children": [parent.clone(), parent.clone()],
        "m_list": [torch.ones(16, 1), torch.ones(16, 1)],
        "branch_weights": branch_weights,
    }
    batches = tmp_path / "batches.pt"
    torch.save(
        {
            "train": [batch, batch],
            "validation": [batch],
            "metadata": {
                "m_semantics": "fixed",
                "shock_bank_hash": "smoke-shock",
                "source_checkpoint_sha256": _file_sha(annotated),
                "sdf_state_hash": _hash_state(torch.load(annotated, map_location="cpu")["models"]["sdf_fc1"]),
            },
        },
        batches,
    )
    out_dir = tmp_path / "out"
    old_argv = sys.argv
    try:
        sys.argv = [
            "ablate",
            "--baseline-checkpoint", str(annotated),
            "--scaled-checkpoint", str(warmed),
            "--batch-data", str(batches),
            "--output-dir", str(out_dir),
            "--rounds", "2",
            "--epochs-per-round", "1",
            "--lr", "1e-8",
            "--force-reject-round", "2",
            "--device", "cpu",
        ]
        ablation_main()
    finally:
        sys.argv = old_argv

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rounds = [row for row in summary["history"] if row.get("stage") == "scaled_posttrain_round"]
    assert [row["accepted"] for row in rounds] == [True, False]
    assert all(rounds[1]["rollback_restore_checks"].values())
    assert summary["batch_provenance_verified"] is True
    assert summary["z_region_cutoffs"] == {"low_z_cutoff": -2.0, "high_z_cutoff": 2.0}
    assert summary["bellman_economic_spec"]["DELTA"] == pytest.approx(float(AnalysisEconomicConfig.from_current_config().DELTA))
    loaded = load_analysis_checkpoint(out_dir / "scaled_posttrain_combined.pt", device="cpu")
    assert loaded.metadata["value_parameterization"]["checkpoint"]["mode"] == "exp_xz"


def test_positive_lr_value_update_smoke_records_hash_invariants(tmp_path: Path):
    from experiments.run_scaled_value_ablation import main as ablation_main

    torch.manual_seed(7)
    models = build_models(torch.device("cpu"))
    pv = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8, dropout=0.25, value_scale_mode="none")
    legacy = tmp_path / "legacy_dropout.pt"
    torch.save(
        {
            "models": {
                "policy_value": pv.state_dict(),
                "sdf_fc1": models["sdf_fc1"].state_dict(),
                "firm_target": pv.state_dict(),
            },
            "hyperparams": HyperParams().__dict__,
        },
        legacy,
    )
    annotated = tmp_path / "annotated_dropout.pt"
    warmed = tmp_path / "warmed_dropout.pt"
    model_spec_json = tmp_path / "dropout_spec.json"
    config_json = tmp_path / "dropout_config.json"
    model_spec_json.write_text(json.dumps({"policy_value_model_spec": pv.model_spec()}), encoding="utf-8")
    config_json.write_text(json.dumps({"config_snapshot": AnalysisEconomicConfig.from_current_config().to_dict()}), encoding="utf-8")

    from scripts.annotate_legacy_equity_checkpoint import main as annotate_main
    old_argv = sys.argv
    try:
        sys.argv = [
            "annotate",
            "--legacy-checkpoint", str(legacy),
            "--model-spec-json", str(model_spec_json),
            "--economic-config-json", str(config_json),
            "--output-checkpoint", str(annotated),
        ]
        annotate_main()
    finally:
        sys.argv = old_argv
    _warmstart(tmp_path, annotated, warmed)

    parent = torch.cat([_states(), _states(), _states(), _states(), _states(), _states()[:1]], dim=0)[:16]
    batch = {
        "parent": parent,
        "children": [parent.clone(), parent.clone()],
        "m_list": [torch.ones(16, 1), torch.ones(16, 1)],
        "branch_weights": torch.full((16, 2), 0.5),
    }
    batches = tmp_path / "positive_lr_batches.pt"
    torch.save(
        {
            "train": [batch, batch],
            "validation": [batch],
            "metadata": {
                "m_semantics": "fixed",
                "shock_bank_hash": "positive-lr",
                "source_checkpoint_sha256": _file_sha(annotated),
                "sdf_state_hash": _hash_state(torch.load(annotated, map_location="cpu")["models"]["sdf_fc1"]),
            },
        },
        batches,
    )
    out_dir = tmp_path / "positive_lr"
    try:
        sys.argv = [
            "ablate",
            "--baseline-checkpoint", str(annotated),
            "--scaled-checkpoint", str(warmed),
            "--batch-data", str(batches),
            "--output-dir", str(out_dir),
            "--rounds", "1",
            "--epochs-per-round", "1",
            "--lr", "1e-5",
            "--device", "cpu",
        ]
        ablation_main()
    finally:
        sys.argv = old_argv

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    round1 = [row for row in summary["history"] if row.get("stage") == "scaled_posttrain_round"][0]
    assert round1["value_hash_candidate"] != round1["value_hash_before"]
    assert round1["non_value_hash_candidate"] == round1["non_value_hash_before"]
    assert round1["teacher_hash_after_training_before_acceptance"] == round1["teacher_hash_before"]
    assert round1["validation_repeat_1"]["p0_physical_conditional_mean_abs"] == pytest.approx(
        round1["validation_repeat_2"]["p0_physical_conditional_mean_abs"]
    )


def test_trainer_complete_config_snapshot_and_bellman_normalization_guard(tmp_path: Path):
    hp = HyperParams()
    model = PolicyValueModel()
    trainer = Trainer({"policy_value": model}, hyperparams=hp, save_dir=tmp_path / "ckpt", log_dir=tmp_path / "log", device=torch.device("cpu"))
    trainer.save_checkpoint("none")
    payload = torch.load(tmp_path / "ckpt" / "none.pt", map_location="cpu")
    assert "AIO_WEIGHT" in payload["config_snapshot"]
    assert "KAPPA_E" in payload["config_snapshot"]
    payload["value_parameterization"]["bellman_normalization"] = True
    torch.save(payload, tmp_path / "ckpt" / "bad_norm.pt")
    with pytest.raises(ValueError, match="bellman_normalization"):
        trainer.load_checkpoint("bad_norm")


def test_trainer_resume_rejects_config_mismatch_unless_evaluation_override(tmp_path: Path):
    hp = HyperParams()
    model = PolicyValueModel()
    trainer = Trainer({"policy_value": model}, hyperparams=hp, save_dir=tmp_path / "ckpt", log_dir=tmp_path / "log", device=torch.device("cpu"))
    trainer.save_checkpoint("none")
    payload = torch.load(tmp_path / "ckpt" / "none.pt", map_location="cpu")
    payload["config_snapshot"]["DELTA"] = float(payload["config_snapshot"]["DELTA"]) + 0.01
    torch.save(payload, tmp_path / "ckpt" / "bad_config.pt")
    with pytest.raises(ValueError, match="config_snapshot mismatch"):
        trainer.load_checkpoint("bad_config")
    trainer.load_checkpoint("bad_config", evaluation_only=True, allow_config_mismatch=True)


def test_raw_analysis_loader_requires_explicit_model_spec_json(tmp_path: Path):
    models = build_models(torch.device("cpu"))
    policy_path = tmp_path / "policy.pt"
    sdf_path = tmp_path / "sdf.pt"
    hp_path = tmp_path / "hp.json"
    config_path = tmp_path / "config.json"
    spec_path = tmp_path / "spec.json"
    torch.save(models["policy_value"].state_dict(), policy_path)
    torch.save(models["sdf_fc1"].state_dict(), sdf_path)
    hp_path.write_text(json.dumps(HyperParams().__dict__, default=str), encoding="utf-8")
    config_path.write_text(json.dumps(AnalysisEconomicConfig.from_current_config().to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="model_spec_json"):
        load_analysis_checkpoint(
            policy_checkpoint=policy_path,
            sdf_checkpoint=sdf_path,
            hyperparams_json=hp_path,
            config_json=config_path,
            device="cpu",
        )

    spec_path.write_text(json.dumps(models["policy_value"].model_spec()), encoding="utf-8")
    loaded = load_analysis_checkpoint(
        policy_checkpoint=policy_path,
        sdf_checkpoint=sdf_path,
        hyperparams_json=hp_path,
        config_json=config_path,
        model_spec_json=spec_path,
        device="cpu",
    )
    assert loaded.metadata["policy_value_model_spec"] == models["policy_value"].model_spec()
