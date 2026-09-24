from types import SimpleNamespace

import pytest
import torch

from config import Config
from config.hyperparams import HyperParams
from losses.q_loss import QLoss, classify_q_parent_regimes
from models.policy_value import (
    PolicyValueModel,
    build_policy_value_from_checkpoint_spec,
)
from training.episode import Episode
from training.trainer import Trainer


def _batch(device=torch.device("cpu")):
    parent = torch.tensor(
        [
            [0.2, 0.1, 1.0, 0.2, -2.0, -1.0, 4.0],
            [0.7, -0.2, 0.0, 0.4, -2.0, -1.0, 4.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    children = []
    for dz, m in ((0.1, 0.98), (-0.1, 1.02)):
        child = parent.clone()
        child[:, 1] += dz
        children.append(torch.cat([child, torch.full((2, 1), m, device=device)], dim=1))
    return {
        "parent": torch.cat([parent, torch.ones((2, 1), device=device)], dim=1),
        "children": children,
    }


def _episode():
    device = torch.device("cpu")
    hp = HyperParams()
    hp.policy_lr = 1e-3
    hp.q_zero_boundary_epochs = 1
    hp.q_default_pretrain_epochs = 0
    hp.q_survival_aio_epochs = 0
    hp.q_mixed_polish_epochs = 0
    online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=hp.policy_lr)
    return Episode(
        models={"policy_value": online},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=hp,
        device=device,
        firm_target=target,
    )


def _constant_q(model: PolicyValueModel, value: float) -> None:
    linear = [m for m in model.q_head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(value)


def test_direct_q_head_outputs_total_q_without_b_multiplier():
    model = PolicyValueModel(
        share_hidden_dims=[], share_output_dim=6, q_head_dims=[], q_parameterization="direct"
    )
    _constant_q(model, -0.25)
    states = torch.zeros((2, 7))
    states[:, 0] = torch.tensor([0.0, 0.8])
    q = model._q_output(states)
    torch.testing.assert_close(q, torch.full_like(q, -0.25))
    assert model.model_spec()["q_parameterization"] == "direct"


def test_missing_q_parameterization_rebuilds_legacy_model():
    direct = PolicyValueModel(share_hidden_dims=[], share_output_dim=6, q_head_dims=[])
    spec = direct.model_spec()
    spec.pop("q_parameterization")
    rebuilt = build_policy_value_from_checkpoint_spec(
        {"policy_value_model_spec": spec},
        value_scale_mode="none",
        value_scale_log_max=20.0,
    )
    assert rebuilt.q_parameterization == "b_times_unit"
    states = torch.zeros((1, 7))
    assert rebuilt._q_output(states).item() == 0.0


def test_trainer_rejects_legacy_checkpoint_for_direct_q():
    stub = SimpleNamespace(hyperparams=SimpleNamespace(q_parameterization="direct"))
    with pytest.raises(ValueError, match="Explicit migration is required"):
        Trainer._assert_checkpoint_q_parameterization(
            stub,
            {"policy_value_model_spec": {"base_state_dim": 6}},
        )


def test_q_regime_masks_are_mutually_exclusive_and_zero_has_priority():
    b = torch.tensor([[0.0], [0.2], [0.3], [-0.1]])
    phat = torch.tensor([[2.0], [-1.0], [1.0], [-1.0]])
    masks = classify_q_parent_regimes(b, phat)
    assert masks["zero"].reshape(-1).tolist() == [True, False, False, False]
    assert masks["default"].reshape(-1).tolist() == [False, True, False, False]
    assert masks["survival"].reshape(-1).tolist() == [False, False, True, False]
    assigned = sum(masks[name].to(torch.int8) for name in ("zero", "default", "survival"))
    assert int(assigned.max()) == 1


def test_zero_and_default_objectives_do_not_call_aio(monkeypatch):
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("AiO must not be called by Q0/QD")

    monkeypatch.setattr("losses.q_loss.compute_aio_residual", _forbidden)
    loss_fn = QLoss(recovery_normalization_mode="asset_only")
    q = torch.tensor([[-0.1], [0.3]], requires_grad=True)
    zero = loss_fn.compute_zero_debt_objective(q)
    default = loss_fn.compute_default_parent_objective(
        Q=q,
        b=torch.tensor([[0.2], [0.8]]),
        x=torch.zeros_like(q),
        z=torch.zeros_like(q),
    )
    assert torch.isfinite(zero["loss"])
    assert torch.isfinite(default["loss"])


def test_q_zero_phase_changes_only_q_and_keeps_frozen_phat():
    episode = _episode()
    model = episode.models["policy_value"]
    frozen_p = PolicyValueModel(**{
        "share_hidden_dims": [8], "share_output_dim": 8,
        "q_parameterization": "direct",
    })
    frozen_p.load_state_dict(model.state_dict())
    frozen_p.eval()
    states = _batch()["parent"][:, :7]
    with torch.no_grad():
        phat_before = frozen_p.forward_equity(states)["Phat"].clone()
    q_params = episode._policy_value_stage_params("q")
    non_q = [p for p in model.parameters() if id(p) not in {id(q) for q in q_params}]
    q_before = episode._snapshot_params(q_params)
    non_q_before = episode._snapshot_params(non_q)
    summary = episode._run_q_regime_phase(
        phase="zero",
        batches=[_batch()],
        frozen_p_model=frozen_p,
        q_target_model=frozen_p,
        epochs=1,
    )
    with torch.no_grad():
        phat_after = frozen_p.forward_equity(states)["Phat"]
    assert summary["optimizer_steps"] == 1
    assert episode._param_max_change_from_snapshot(q_params, q_before) > 0.0
    assert episode._param_max_change_from_snapshot(non_q, non_q_before) == 0.0
    torch.testing.assert_close(phat_after, phat_before)
    assert summary["frozen_p_hash_before"] == summary["frozen_p_hash_after"]


class _RecordingTarget(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.states = []

    def forward(self, state):
        self.states.append(state.detach().clone())
        return self.model(state)


def test_survival_continuation_uses_old_bond_leverage_not_bp_transition():
    episode = _episode()
    batch = _batch()
    target = _RecordingTarget(episode.firm_target)
    with torch.no_grad():
        parent_state = batch["parent"][:, :7]
        parent_out = episode.firm_target(parent_state)
        expected = parent_state[:, 0:1] / (
            1.0 + parent_out.bar_i * (float(episode.loss_fns["q"].g) - 1.0)
        )
    episode._compute_q_survival_bellman_loss(
        batch,
        create_graph=False,
        q_target_model=target,
    )
    # First call is parent target evaluation; remaining calls are continuation children.
    # Exact future-eta integration may expand each exogenous child into two eta
    # states; every resulting continuation state must keep the same old-bond b_sp.
    assert len(target.states) > 1
    for child_state in target.states[1:]:
        torch.testing.assert_close(child_state[:, 0:1], expected)


class _PhatFromZ(torch.nn.Module):
    def forward_equity(self, state):
        return {"Phat": state[:, 1:2]}


@pytest.mark.parametrize(
    ("b_value", "z_value", "expected_path"),
    [(0.0, 1.0, "zero"), (0.4, -1.0, "default")],
)
def test_q_dispatcher_zero_and_default_never_enter_survival_path(
    monkeypatch, b_value, z_value, expected_path
):
    episode = _episode()
    parent = torch.zeros((3, 8))
    parent[:, 0] = b_value
    parent[:, 1] = z_value
    batch = {"parent": parent}
    called = []

    monkeypatch.setattr(
        episode,
        "_compute_q_zero_loss",
        lambda state: called.append(("zero", state[:, 0].clone())) or torch.tensor(1.0),
    )
    monkeypatch.setattr(
        episode,
        "_compute_q_default_loss",
        lambda state: called.append(("default", state[:, 0].clone())) or torch.tensor(1.0),
    )

    def _forbidden_survival(*_args, **_kwargs):
        raise AssertionError("Q0/QD must not construct Q Bellman children or call SDF/AiO")

    monkeypatch.setattr(episode, "_compute_q_survival_bellman_loss", _forbidden_survival)
    loss = episode._compute_q_loss(
        batch,
        regime="mixed",
        frozen_p_model=_PhatFromZ(),
    )
    assert loss.item() == pytest.approx(1.0)
    assert [name for name, _ in called] == [expected_path]


def test_q_mixed_dispatch_is_sample_level_regime_exclusive(monkeypatch):
    episode = _episode()
    parent = torch.zeros((3, 8))
    parent[:, 0] = torch.tensor([0.0, 0.4, 0.6])
    parent[:, 1] = torch.tensor([1.0, -1.0, 1.0])
    children = [parent.clone(), parent.clone()]
    batch = {"parent": parent, "children": children}
    observed = {}

    def _record(name, state):
        observed[name] = state[:, :2].clone()
        return torch.tensor(1.0)

    monkeypatch.setattr(episode, "_compute_q_zero_loss", lambda state: _record("zero", state))
    monkeypatch.setattr(episode, "_compute_q_default_loss", lambda state: _record("default", state))

    def _survival(subset, **_kwargs):
        return _record("survival", subset["parent"])

    monkeypatch.setattr(episode, "_compute_q_survival_bellman_loss", _survival)
    loss = episode._compute_q_loss(
        batch,
        regime="mixed",
        frozen_p_model=_PhatFromZ(),
    )
    assert loss.item() == pytest.approx(1.0)
    assert set(observed) == {"zero", "default", "survival"}
    torch.testing.assert_close(observed["zero"], torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(observed["default"], torch.tensor([[0.4, -1.0]]))
    torch.testing.assert_close(observed["survival"], torch.tensor([[0.6, 1.0]]))
