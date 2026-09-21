from pathlib import Path
import math
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config, HyperParams  # noqa: E402
from losses.sdf_loss import (  # noqa: E402
    SDFLoss,
    compute_pooled_moment_constraints,
    moment_penalty,
    phr_augmented_lagrangian,
)
from models.sdf_fc1 import SDFFC1Combined  # noqa: E402
from training.episode import Episode, SDFTrainingPhase  # noqa: E402


BOUNDS = dict(mu_lo=-0.025, mu_hi=0.0, var_hi=0.25, eps=1e-8)


def _constraints(values: torch.Tensor):
    return compute_pooled_moment_constraints(values, **BOUNDS)


def test_pooled_constraints_use_flattened_branches_and_centered_variance():
    m = torch.tensor([[0.5, 1.5], [0.7, 1.3]], dtype=torch.float64)
    result = _constraints(m)
    pooled = m.reshape(-1)
    expected_mu = pooled.mean()
    expected_var = ((pooled - expected_mu) ** 2).mean()

    assert torch.allclose(result["mu"], expected_mu)
    assert torch.allclose(result["var"], expected_var)
    assert not torch.allclose(result["mu"], m[:, 0].mean())
    assert result["g_mean_low"] < 0
    assert result["g_mean_high"] == 0
    assert result["g_var_high"] < 0


@pytest.mark.parametrize(
    ("values", "positive_constraint"),
    [
        (torch.tensor([0.97, 0.99, 0.98, 0.98]), None),
        (torch.full((4,), 0.95), "g_mean_low"),
        (torch.full((4,), 1.01), "g_mean_high"),
        (torch.tensor([0.0] * 9 + [9.8]), "g_var_high"),
    ],
)
def test_constraint_signs(values, positive_constraint):
    result = _constraints(values)
    if positive_constraint is None:
        assert result["g_mean_low"] < 0
        assert result["g_mean_high"] < 0
        assert result["g_var_high"] < 0
        assert bool(result["feasible"])
    else:
        assert result[positive_constraint] > 0
        assert not bool(result["feasible"])


def test_phr_dual_update_increases_for_violation_and_projects_at_zero():
    episode = Episode.__new__(Episode)
    episode.hyperparams = HyperParams(sdf_al_rho=10.0, sdf_al_lambda_init=0.2)
    episode._reset_sdf_al_state()

    after = episode._update_sdf_al_duals({
        "g_mean_low": 0.1,
        "g_mean_high": -0.01,
        "g_var_high": -1.0,
    })

    assert after["lambda_mean_low"] > 0.2
    assert after["lambda_mean_high"] <= 0.2
    assert after["lambda_var_high"] == 0.0
    assert min(after[key] for key in after if key.startswith("lambda_")) >= 0.0


def test_inactive_phr_constraint_has_zero_gradient():
    m = torch.tensor([0.98, 0.98], dtype=torch.float64, requires_grad=True)
    constraints = _constraints(m)
    total, terms = phr_augmented_lagrangian(
        constraints,
        {
            "lambda_mean_low": 0.0,
            "lambda_mean_high": 0.0,
            "lambda_var_high": 0.0,
        },
        rho=10.0,
    )
    total.backward()

    assert total.item() == pytest.approx(0.0)
    assert all(term.item() == pytest.approx(0.0) for term in terms.values())
    assert torch.equal(m.grad, torch.zeros_like(m))


class _ToyCombined(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sdf_model = torch.nn.Linear(1, 1, bias=False)
        self.value_model = torch.nn.Linear(1, 1)
        self.fc1_model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.sdf_model.weight.fill_(1.0)
            self.value_model.weight.fill_(0.1)
            self.value_model.bias.fill_(1.0)

    def forward_step(self, x_prev, x_curr, hatcf_prev, lnkf_prev, return_physical=True):
        n_children = x_curr.shape[1]
        c_children = hatcf_prev.unsqueeze(1) + 0.01 * x_curr
        k_children = lnkf_prev.unsqueeze(1) + 0.02 * x_curr
        parent_surplus = torch.nn.functional.softplus(self.value_model(x_prev)) + 1.0
        child_surplus = torch.nn.functional.softplus(self.value_model(x_curr)) + 1.0
        w_parent = torch.exp(hatcf_prev) + parent_surplus
        w_children = torch.exp(c_children) + child_surplus
        m = torch.exp(self.sdf_model(x_curr))
        assert m.shape == (x_prev.shape[0], n_children, 1)
        return w_parent, w_children, m, c_children, k_children


def _batch():
    parent = torch.tensor(
        [
            [0.2, 0.1, 1.0, 0.0, -0.9, -0.20, 4.50, -0.10, 4.60],
            [0.4, 0.2, 0.0, 0.0, -0.8, -0.15, 4.55, -0.12, 4.62],
        ],
        dtype=torch.float32,
    )
    child0 = parent.clone()
    child1 = parent.clone()
    child0[:, 4] = torch.tensor([-0.9, -0.7])
    child1[:, 4] = torch.tensor([-1.1, -0.6])
    return {"parent": parent, "children": [child0, child1]}


def _loss_episode(mode: str) -> Episode:
    hp = HyperParams()
    hp.sdf_moment_constraint_mode = mode
    hp.sdf_fresh_pair_enabled = False
    hp.sdf_wealth_residual_mode = "normalized_ratio"
    hp.sdf_euler_weight = 1.0
    episode = Episode.__new__(Episode)
    episode.models = {"sdf_fc1": _ToyCombined()}
    episode.loss_fns = {"sdf": SDFLoss(wealth_loss_mode="signed_aio")}
    episode.hyperparams = hp
    episode.config = Config
    episode.device = torch.device("cpu")
    episode.episode_id = 1
    episode.add_FC1loss = False
    episode._fc1_teacher_forcing_stage = False
    episode.sdf_training_phase = SDFTrainingPhase.SDF_TRUE_ONLY
    episode._current_epoch_idx = 0
    episode.step_count = 0
    episode.sdf_fc1_step_count = 0
    episode._latest_sdf_terms = {}
    episode._latest_sdf_diag = {}
    episode._reset_sdf_al_state()
    return episode


def test_al_sdf_true_objective_keeps_signed_aio_and_fc1_frozen():
    episode = _loss_episode("augmented_lagrangian")
    model = episode.models["sdf_fc1"]
    episode._set_sdf_training_phase_freeze()
    fc1_before = {name: value.detach().clone() for name, value in model.fc1_model.state_dict().items()}

    loss = episode._compute_sdf_loss(_batch())
    loss.backward()

    terms = episode._latest_sdf_terms
    assert terms["sdf_wealth_loss_mode_signed_aio"] == 1.0
    assert terms["sdf_wealth_residual_mode_normalized_ratio"] == 1.0
    assert terms["sdf_moment_weight_effective"] == 0.0
    assert terms["sdf_anchor_weight_effective"] == 0.0
    assert terms["sdf_al_active"] == 1.0
    assert terms["sdf_al_term_total"] > 0.0
    assert loss.item() == pytest.approx(
        terms["sdf_main_weight_effective"] * terms["sdf_true_state_main_loss"]
        + terms["sdf_al_term_total"],
        rel=1e-5,
        abs=1e-6,
    )
    assert all(parameter.grad is None for parameter in model.fc1_model.parameters())
    assert any(parameter.grad is not None for parameter in model.sdf_model.parameters())
    assert any(parameter.grad is not None for parameter in model.value_model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.sdf_model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.value_model.parameters())
    assert all(torch.equal(value, fc1_before[name]) for name, value in model.fc1_model.state_dict().items())


def test_production_combined_model_gradient_routing_matches_current_architecture():
    torch.manual_seed(12345)
    episode = _loss_episode("augmented_lagrangian")
    episode.models["sdf_fc1"] = SDFFC1Combined(
        sdf_input_dim=4,
        fc1_input_dim=4,
        sdf_hidden_dims=[4],
        fc1_hidden_dims=[4],
        w_hidden_dims=[4],
    )
    model = episode.models["sdf_fc1"]
    episode._set_sdf_training_phase_freeze()

    loss = episode._compute_sdf_loss(_batch())
    loss.backward()

    value_grads = [parameter.grad for parameter in model.value_model.parameters()]
    assert any(grad is not None for grad in value_grads)
    assert all(grad is None or torch.isfinite(grad).all() for grad in value_grads)
    # Production M is structural and depends on ValueFunctionW; the historical
    # SDFModel.network is not called by SDFFC1Combined.forward_step().
    assert all(parameter.grad is None for parameter in model.sdf_model.network.parameters())
    assert all(parameter.grad is None for parameter in model.fc1_model.parameters())


def test_legacy_mode_preserves_fixed_penalty_and_anchor_objective():
    episode = _loss_episode("legacy_penalty")
    loss = episode._compute_sdf_loss(_batch())
    terms = episode._latest_sdf_terms

    assert terms["sdf_al_active"] == 0.0
    assert terms["sdf_al_term_total"] == 0.0
    assert terms["sdf_moment_weight_effective"] == episode.hyperparams.sdf_true_moment_weight
    assert terms["sdf_anchor_weight_effective"] == episode.hyperparams.sdf_true_anchor_weight
    expected = (
        terms["sdf_main_weight_effective"] * terms["sdf_true_state_main_loss"]
        + terms["sdf_moment_weight_effective"] * terms["sdf_moment_loss"]
        + terms["sdf_anchor_weight_effective"] * terms["sdf_mean_anchor_loss"]
    )
    assert loss.item() == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_dual_estimator_reforwards_fixed_training_split_at_current_theta():
    episode = _loss_episode("augmented_lagrangian")
    batch = _batch()
    estimate = episode._estimate_sdf_al_constraints([batch])

    with torch.no_grad():
        parent = batch["parent"]
        children = torch.stack(batch["children"], dim=1)
        _, _, m, _, _ = episode.models["sdf_fc1"].forward_step(
            x_prev=parent[:, 4:5],
            x_curr=children[:, :, 4:5],
            hatcf_prev=parent[:, 7:8],
            lnkf_prev=parent[:, 8:9],
            return_physical=True,
        )
        expected = _constraints(m)

    assert estimate["n_batches"] == 1
    assert estimate["n_observations"] == 4
    for key in ("mu", "var", "g_mean_low", "g_mean_high", "g_var_high"):
        assert estimate[key] == pytest.approx(expected[key].item())


def test_validation_metrics_report_centered_variance_and_logs():
    episode = _loss_episode("augmented_lagrangian")
    metrics = episode._evaluate_sdf_fc1_batches([_batch()], prefix="eval")
    root = "eval_primary_true_state_M"
    assert metrics[f"{root}_finite_ratio"] == 1.0
    assert metrics[f"{root}_var"] >= 0.0
    assert math.isfinite(metrics[f"{root}_log_mean"])
    assert math.isfinite(metrics[f"{root}_log_var"])


def _gate_episode() -> Episode:
    episode = Episode.__new__(Episode)
    episode.hyperparams = HyperParams(
        sdf_moment_constraint_mode="augmented_lagrangian",
        sdf_al_gate_tolerance=0.0,
        sdf_log_mean_error_max=0.005,
        sdf_signed_t_abs_max=2.0,
        sdf_gate_m_finite_ratio_min=1.0,
    )
    episode.loss_fns = {"sdf": SDFLoss()}
    return episode


def _gate_metrics(mean: float, var: float, signed_t: float):
    prefix = "validation"
    root = f"{prefix}_primary_true_state_M"
    return prefix, {
        f"{root}_mean": mean,
        f"{root}_var": var,
        f"{root}_log_mean": math.log(max(mean, 1e-8)),
        f"{root}_log_var": math.log(max(var, 1e-8)),
        f"{root}_finite_ratio": 1.0,
        f"{root}_p99": mean,
        f"{root}_max": mean,
        f"{prefix}_primary_true_state_normalized_signed_aio_t": signed_t,
    }


def test_al_strict_gate_uses_inequalities_not_point_anchor():
    episode = _gate_episode()
    prefix, metrics = _gate_metrics(mean=0.99, var=0.01, signed_t=0.0)
    passed, diag = episode._sdf_gate_passed(metrics, prefix, SDFTrainingPhase.SDF_TRUE_ONLY)
    assert passed
    assert diag["log_mean_error"] > episode.hyperparams.sdf_log_mean_error_max
    assert diag["moment_feasible"]

    prefix, metrics = _gate_metrics(mean=0.99, var=math.exp(0.25) + 0.01, signed_t=0.0)
    passed, diag = episode._sdf_gate_passed(metrics, prefix, SDFTrainingPhase.SDF_TRUE_ONLY)
    assert not passed
    assert diag["g_var_high"] > 0

    prefix, metrics = _gate_metrics(mean=0.99, var=0.01, signed_t=3.0)
    passed, _ = episode._sdf_gate_passed(metrics, prefix, SDFTrainingPhase.SDF_TRUE_ONLY)
    assert not passed
