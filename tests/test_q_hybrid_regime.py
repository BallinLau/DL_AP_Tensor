from copy import deepcopy
import types

import pytest
import torch

from config import Config
from config.hyperparams import HyperParams
from analysis.economic_config import AnalysisEconomicConfig
from evaluation.bellman_diagnostics import evaluate_bellman_residuals
from evaluation.bp_diagnostics import FrozenTransitionData
from evaluation.grids import ReferenceFirmState, build_frozen_grid
from experiments.run_utils import resolve_raw_q_parameterization
from losses import P0Loss, PILoss
from models.policy_value import PolicyValueModel
from training.bp_grid_teacher import BPGridTeacher, _target_q_claim
from training.episode import Episode


def _state(b_values=(0.0, 0.2, 0.8)) -> torch.Tensor:
    rows = []
    for b in b_values:
        rows.append([b, 0.1, 1.0, 0.2, 0.0, -2.0, 4.0])
    return torch.tensor(rows, dtype=torch.float32)


def _small_model(mode: str) -> PolicyValueModel:
    return PolicyValueModel(
        q_parameterization=mode,
        share_hidden_dims=[4],
        share_output_dim=4,
        q_head_dims=[4],
        p0_head_dims=[4],
        pi_head_dims=[4],
        bp0_head_dims=[4],
        bpi_head_dims=[4],
        barz_hidden_dims=[4],
        bari_hidden_dims=[4],
        i_grid_size=2,
    )


def test_hybrid_zero_default_survival_and_nonnegative_unit():
    model = _small_model("hybrid_regime")
    state = _state()
    phat = torch.tensor([[-1.0], [-1.0], [1.0]])

    effective = model._q_effective_output(state, phat=phat)
    recovery = model._q_recovery_output(state)
    unit = model._q_unit_output(state)
    claim = model._q_claim_output(state)

    assert effective[0].item() == 0.0
    assert claim[0].item() == 0.0
    torch.testing.assert_close(effective[1], recovery[1])
    torch.testing.assert_close(effective[2], state[2, 0:1] * unit[2])
    assert bool((unit >= 0.0).all())


def test_hybrid_default_value_is_independent_of_q_head_weights():
    model = _small_model("hybrid_regime")
    state = _state((0.01, 0.8))
    phat = torch.full((2, 1), -1.0)
    before = model._q_effective_output(state, phat=phat)
    with torch.no_grad():
        for parameter in model.q_head.parameters():
            parameter.fill_(37.0)
    after = model._q_effective_output(state, phat=phat)
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    torch.testing.assert_close(after, model._q_recovery_output(state))


class _CandidateGate(torch.nn.Module):
    def forward_equity(self, state):
        # Low candidate debt survives; high candidate debt defaults.
        phat = 0.5 - state[:, 0:1]
        return {
            "Phat": phat,
            "P": torch.relu(phat),
            "bar_z": (phat <= 0).to(state.dtype),
        }


def test_teacher_claim_q_ignores_candidate_specific_equity_gate():
    q_model = _small_model("hybrid_regime")
    gate = _CandidateGate()
    candidates = _state((0.2, 0.9))
    result = _target_q_claim(q_model, candidates, equity_model=gate)
    expected_claim = q_model._q_claim_output(candidates)
    torch.testing.assert_close(result, expected_claim)

    teacher = BPGridTeacher(
        gate,
        P0Loss(),
        PILoss(),
        q_target_model=q_model,
        coarse_size=2,
        refine=False,
    )
    parent = _state((0.1,))
    claim_grid, diagnostics = teacher._q_issue_grid(
        parent,
        torch.tensor([[0.2, 0.9]], dtype=parent.dtype),
    )
    assert set(diagnostics) >= {
        "q_issue_claim_grid",
        "q_issue_unit_grid",
        "q_issue_realized_default_mask_grid",
        "q_issue_recovery_grid",
        "q_issue_candidate_phat_grid",
        "candidate_phat_gate_used_for_q_issue",
    }
    torch.testing.assert_close(claim_grid, diagnostics["q_issue_claim_grid"])
    assert diagnostics["q_issue_realized_default_mask_grid"].tolist() == [[0.0, 1.0]]
    assert diagnostics["candidate_phat_gate_used_for_q_issue"].count_nonzero() == 0
    torch.testing.assert_close(claim_grid, q_model._q_claim_output(candidates).reshape(1, 2))


class _FlippedCandidateGate(torch.nn.Module):
    def forward_equity(self, state):
        phat = state[:, 0:1] - 0.5
        return {
            "Phat": phat,
            "P": torch.relu(phat),
            "bar_z": (phat <= 0).to(state.dtype),
        }


def test_changing_candidate_classifier_does_not_change_issue_claim_price():
    q_model = _small_model("hybrid_regime")
    parent = _state((0.1,))
    grid = torch.tensor([[0.2, 0.9]], dtype=parent.dtype)
    first = BPGridTeacher(
        _CandidateGate(), P0Loss(), PILoss(), q_target_model=q_model,
        coarse_size=2, refine=False,
    )
    second = BPGridTeacher(
        _FlippedCandidateGate(), P0Loss(), PILoss(), q_target_model=q_model,
        coarse_size=2, refine=False,
    )
    first_q, first_diag = first._q_issue_grid(parent, grid)
    second_q, second_diag = second._q_issue_grid(parent, grid)
    torch.testing.assert_close(first_q, second_q, rtol=0.0, atol=0.0)
    assert not torch.equal(
        first_diag["q_issue_realized_default_mask_grid"],
        second_diag["q_issue_realized_default_mask_grid"],
    )


def test_hybrid_checkpoint_round_trip_and_raw_semantic_selector(tmp_path):
    model = _small_model("hybrid_regime")
    path = tmp_path / "hybrid.pt"
    torch.save({"spec": model.model_spec(), "state": model.state_dict()}, path)
    payload = torch.load(path, map_location="cpu")
    restored = PolicyValueModel(**payload["spec"])
    restored.load_state_dict(payload["state"], strict=True)
    state = _state((0.0, 0.3, 0.9))
    phat = torch.tensor([[1.0], [1.0], [-1.0]])
    torch.testing.assert_close(
        model._q_effective_output(state, phat=phat),
        restored._q_effective_output(state, phat=phat),
    )
    assert restored.q_parameterization == "hybrid_regime"
    assert resolve_raw_q_parameterization("hybrid_regime") == "hybrid_regime"
    # A legacy raw state_dict remains legacy unless the caller explicitly says otherwise.
    assert resolve_raw_q_parameterization("b_times_unit") == "b_times_unit"


def test_existing_direct_and_legacy_q_semantics_are_unchanged():
    state = _state((0.0, 0.4))
    direct = _small_model("direct")
    legacy = _small_model("b_times_unit")
    torch.testing.assert_close(direct._q_output(state), direct._q_unit_output(state))
    torch.testing.assert_close(
        legacy._q_output(state),
        state[:, 0:1] * legacy._q_unit_output(state),
    )


def test_hybrid_bellman_uses_claim_q_not_effective_q(monkeypatch):
    episode = _episode_for_gate()
    online = episode.models["policy_value"]
    q_target = deepcopy(online)
    frozen_p = _small_model("direct")
    calls = {"online": 0, "target": 0}
    online_claim = online._q_claim_output
    target_claim = q_target._q_claim_output

    def online_spy(state):
        calls["online"] += 1
        return online_claim(state)

    def target_spy(state):
        calls["target"] += 1
        return target_claim(state)

    monkeypatch.setattr(online, "_q_claim_output", online_spy)
    monkeypatch.setattr(q_target, "_q_claim_output", target_spy)
    monkeypatch.setattr(
        q_target,
        "_q_effective_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("survival Bellman must not consume effective child Q")
        ),
    )
    parent = _state((0.2, 0.4))
    child0 = parent.clone()
    child1 = parent.clone()
    child0[:, 1] += 0.1
    child1[:, 1] -= 0.1
    batch = {
        "parent": torch.cat([parent, torch.ones(2, 1)], dim=1),
        "children": [
            torch.cat([child0, torch.ones(2, 1)], dim=1),
            torch.cat([child1, torch.ones(2, 1)], dim=1),
        ],
    }
    loss = episode._compute_q_survival_bellman_loss(
        batch,
        create_graph=False,
        frozen_p_model=frozen_p,
        q_target_model=q_target,
    )
    assert torch.isfinite(loss)
    assert calls["online"] == 1
    assert calls["target"] >= 2


def test_hybrid_bellman_evaluator_uses_claim_q_and_never_recovery_over_b(
    monkeypatch,
):
    model = _small_model("hybrid_regime")
    reference = ReferenceFirmState(
        eta=1.0,
        i_low=0.1,
        i_mid=0.2,
        i_high=0.3,
        x=0.0,
        hatcf=-2.0,
        lnkf=4.0,
        hatc_cal=-2.0,
        lnk_cal=4.0,
        n_parent_rows=1,
        source="fixture",
        macro_source="fixture",
    )
    grid = build_frozen_grid(
        reference,
        b_min=0.2,
        b_max=0.8,
        b_points=2,
        z_min=-0.2,
        z_max=0.2,
        z_points=2,
        device=torch.device("cpu"),
    )
    parent = grid.base_states
    children = torch.cat(
        [parent, torch.ones(parent.shape[0], 1, dtype=parent.dtype)], dim=1
    )
    transition = FrozenTransitionData(
        children=[children.clone(), children.clone()],
        m_raw_list=[torch.ones(parent.shape[0], 1)] * 2,
        m_used_list=[torch.ones(parent.shape[0], 1)] * 2,
        branch_weights=torch.full((parent.shape[0], 2), 0.5),
        metadata={},
    )

    original_forward = model.forward

    def forward_with_wrong_effective_q(states):
        output = original_forward(states)
        return output._replace(Q=torch.full_like(output.Q, 123.0))

    monkeypatch.setattr(model, "forward", forward_with_wrong_effective_q)
    surfaces, _ = evaluate_bellman_residuals(
        model,
        grid,
        transition,
        AnalysisEconomicConfig.from_current_config(),
    )

    expected_claim = model._q_claim_output(parent).detach().numpy().reshape(grid.shape)
    expected_unit = model._q_unit_output(parent).detach().numpy().reshape(grid.shape)
    assert (surfaces["Q_effective"] == 123.0).all()
    assert (abs(surfaces["Q_claim"] - expected_claim) < 1e-6).all()
    assert (abs(surfaces["q_unit_pred"] - expected_unit) < 1e-6).all()
    assert torch.isnan(torch.from_numpy(surfaces["q_unit_recovery_target"])).all()
    assert torch.isnan(torch.from_numpy(surfaces["q_unit_minus_recovery_target"])).all()
    assert (abs(surfaces["RQ_training_signed"] - surfaces["RQ_bellman_signed"]) < 1e-6).all()


def _q_batch(b_values=(0.2, 0.4, 0.7, 0.9)):
    parent = _state(b_values)
    child0 = parent.clone()
    child1 = parent.clone()
    child0[:, 1] += 0.1
    child1[:, 1] -= 0.1
    return {
        "parent": torch.cat([parent, torch.ones(parent.shape[0], 1)], dim=1),
        "children": [
            torch.cat([child0, torch.ones(parent.shape[0], 1)], dim=1),
            torch.cat([child1, torch.ones(parent.shape[0], 1)], dim=1),
        ],
    }


def test_hybrid_claim_batch_excludes_realized_default_positive_debt_parents():
    episode = _episode_for_gate()
    episode.hyperparams.q_claim_coverage_enabled = False
    batch = _q_batch()
    claim_batch, diagnostics = episode._build_q_survival_batch(
        batch,
        _CandidateGate(),
        return_diagnostics=True,
    )
    assert claim_batch is not None
    assert diagnostics["realized_parent_survival_count"] == 2
    assert diagnostics["realized_parent_default_count"] == 2
    assert diagnostics["claim_ondist_sample_count"] == 2
    assert diagnostics["claim_total_sample_count"] == 2
    torch.testing.assert_close(claim_batch["parent"], batch["parent"][:2])
    default_state = batch["parent"][2:, :7]
    default_phat = _CandidateGate().forward_equity(default_state)["Phat"]
    torch.testing.assert_close(
        episode.models["policy_value"]._q_effective_output(
            default_state, phat=default_phat
        ),
        episode.models["policy_value"]._q_recovery_output(default_state),
    )


def test_hybrid_claim_batch_reports_realized_zero_survival_default_partition():
    episode = _episode_for_gate()
    episode.hyperparams.q_claim_coverage_enabled = False
    claim_batch, diagnostics = episode._build_q_survival_batch(
        _q_batch((0.0, 0.2, 0.7)), _CandidateGate(), return_diagnostics=True
    )
    assert claim_batch is not None
    assert diagnostics["realized_parent_zero_b_count"] == 1
    assert diagnostics["realized_parent_survival_count"] == 1
    assert diagnostics["realized_parent_default_count"] == 1
    assert diagnostics["claim_total_sample_count"] == 1


def test_hybrid_claim_coverage_spans_bins_and_produces_q_gradient():
    episode = _episode_for_gate()
    episode.hyperparams.q_claim_coverage_enabled = True
    episode.hyperparams.q_survival_ondist_share = 0.5
    episode.hyperparams.q_claim_coverage_b_bins = 4
    batch = _q_batch((0.1, 0.2, 0.3, 0.4))
    claim_batch, diagnostics = episode._build_q_survival_batch(
        batch, _CandidateGate(),
        return_diagnostics=True,
    )
    assert claim_batch is not None
    assert diagnostics["claim_synthetic_sample_count"] == 4
    assert diagnostics["claim_total_sample_count"] == 8
    assert diagnostics["claim_coverage_bins_occupied"] == 4
    assert diagnostics["claim_coverage_fraction_bins_occupied"] == 1.0
    assert diagnostics["synthetic_candidate_phat_negative_share"] == pytest.approx(0.5)
    assert diagnostics["synthetic_candidate_phat_used_for_filter"] is False

    q_target = deepcopy(episode.models["policy_value"])
    loss = episode._compute_q_survival_bellman_loss(
        claim_batch,
        frozen_p_model=deepcopy(episode.firm_target),
        q_target_model=q_target,
    )
    loss.backward()
    grad_total = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in episode.models["policy_value"].q_head.parameters()
        if parameter.grad is not None
    )
    assert torch.isfinite(loss)
    assert grad_total > 0.0


def test_hybrid_synthetic_sources_only_use_realized_survivor_contexts():
    episode = _episode_for_gate()
    episode.hyperparams.q_survival_ondist_share = 0.5
    episode.hyperparams.q_claim_coverage_b_bins = 2
    batch = _q_batch((0.2, 0.4, 0.7, 0.9))
    batch["parent"][:, 1] = torch.tensor([1.0, 2.0, 7.0, 9.0])
    for child in batch["children"]:
        child[:, 1] = batch["parent"][:, 1]
    result, diagnostics = episode._build_q_survival_batch(
        batch, _CandidateGate(), return_diagnostics=True
    )
    assert diagnostics["realized_parent_survival_count"] == 2
    assert diagnostics["synthetic_source_context_count"] == 2
    assert set(result["parent"][-2:, 1].tolist()) == {1.0, 2.0}
    assert not set(result["parent"][-2:, 1].tolist()) & {7.0, 9.0}


class _TwoCallGate(torch.nn.Module):
    def __init__(self, synthetic_default: bool):
        super().__init__()
        self.synthetic_default = synthetic_default
        self.calls = 0

    def forward_equity(self, state):
        self.calls += 1
        phat = torch.ones_like(state[:, 0:1])
        if self.calls > 1 and self.synthetic_default:
            phat = -phat
        return {"Phat": phat, "P": torch.relu(phat), "bar_z": (phat <= 0).float()}


def test_hybrid_synthetic_candidate_phat_never_filters_claim_replay():
    episode = _episode_for_gate()
    episode.hyperparams.q_survival_ondist_share = 0.5
    episode.hyperparams.q_claim_coverage_b_bins = 4
    batch = _q_batch((0.1, 0.2, 0.3, 0.4))
    positive, positive_diag = episode._build_q_survival_batch(
        batch, _TwoCallGate(False), return_diagnostics=True
    )
    negative, negative_diag = episode._build_q_survival_batch(
        batch, _TwoCallGate(True), return_diagnostics=True
    )
    assert positive_diag["claim_synthetic_sample_count"] == 4
    assert negative_diag["claim_synthetic_sample_count"] == 4
    torch.testing.assert_close(positive["parent"], negative["parent"])
    assert positive_diag["synthetic_candidate_phat_positive_share"] == 1.0
    assert negative_diag["synthetic_candidate_phat_negative_share"] == 1.0
    assert negative_diag["synthetic_candidate_phat_used_for_filter"] is False
    synthetic_state = negative["parent"][-4:, :7]
    torch.testing.assert_close(
        episode.models["policy_value"]._q_claim_output(synthetic_state),
        synthetic_state[:, 0:1]
        * episode.models["policy_value"]._q_unit_output(synthetic_state),
    )


def test_hybrid_synthetic_parent_b_rebuilds_child_old_bond_leverage(monkeypatch):
    episode = _episode_for_gate()
    episode.hyperparams.q_survival_ondist_share = 0.5
    episode.hyperparams.q_claim_coverage_b_bins = 2
    episode.hyperparams.pv_exact_eta_integration_enabled = False
    batch = _q_batch((0.2, 0.4))
    for child in batch["children"]:
        child[:, 0] = 9.0  # Deliberately wrong; the Bellman path must overwrite it.
    claim_batch, diagnostics = episode._build_q_survival_batch(
        batch, _TwoCallGate(False), return_diagnostics=True
    )
    assert diagnostics["claim_synthetic_sample_count"] == 2

    q_target = deepcopy(episode.models["policy_value"])
    captured_child_b = []
    original_claim = q_target._q_claim_output

    def capture_claim(state):
        captured_child_b.append(state[:, 0:1].detach().clone())
        return original_claim(state)

    monkeypatch.setattr(q_target, "_q_claim_output", capture_claim)
    frozen_p = deepcopy(episode.firm_target)
    with torch.no_grad():
        bar_i = frozen_p(claim_batch["parent"][:, :7]).bar_i
    multiplier = bar_i * (float(episode.loss_fns["q"].g) - 1.0) + 1.0
    expected = claim_batch["parent"][:, 0:1] / multiplier.clamp_min(1e-6)
    episode._compute_q_survival_bellman_loss(
        claim_batch, frozen_p_model=frozen_p, q_target_model=q_target
    )
    assert captured_child_b
    for used_b in captured_child_b:
        torch.testing.assert_close(used_b, expected)
        assert not torch.equal(used_b, torch.full_like(used_b, 9.0))
    assert episode._latest_q_terms["q_child_b_sp_reconstruction_max_error"] == 0.0


class _DebtDependentBarITarget(torch.nn.Module):
    """Frozen-P double with different investment weights at original/synthetic b."""

    def forward_equity(self, state):
        one = torch.ones_like(state[:, 0:1])
        return {
            "Phat": one,
            "P": one,
            "bar_z": torch.zeros_like(one),
            "survival_prob": one,
        }

    def forward(self, state):
        one = torch.ones_like(state[:, 0:1])
        bar_i = torch.where(
            state[:, 0:1] < 0.5,
            torch.full_like(one, 0.8),
            torch.full_like(one, 0.2),
        )
        return types.SimpleNamespace(
            bar_i=bar_i,
            Phat=one,
            bar_z=torch.zeros_like(one),
        )


def test_hybrid_synthetic_b_recomputes_bar_i_before_child_b_sp(monkeypatch):
    episode = _episode_for_gate()
    episode.hyperparams.q_survival_ondist_share = 0.5
    episode.hyperparams.q_claim_coverage_b_bins = 2
    episode.hyperparams.bp_grid_min = 0.6
    episode.hyperparams.bp_grid_max = 1.0
    episode.hyperparams.pv_exact_eta_integration_enabled = False
    batch = _q_batch((0.2,))
    for child in batch["children"]:
        child[:, 0] = 9.0

    frozen_p = _DebtDependentBarITarget()
    claim_batch, diagnostics = episode._build_q_survival_batch(
        batch, frozen_p, return_diagnostics=True
    )
    assert diagnostics["claim_ondist_sample_count"] == 1
    assert diagnostics["claim_synthetic_sample_count"] == 1
    b_original = claim_batch["parent"][0:1, 0:1]
    b_syn = claim_batch["parent"][1:2, 0:1]
    torch.testing.assert_close(b_original, torch.tensor([[0.2]]))
    torch.testing.assert_close(b_syn, torch.tensor([[0.7]]))

    q_target = deepcopy(episode.models["policy_value"])
    captured_child_b = []
    original_claim = q_target._q_claim_output

    def capture_claim(state):
        captured_child_b.append(state[:, 0:1].detach().clone())
        return original_claim(state)

    monkeypatch.setattr(q_target, "_q_claim_output", capture_claim)
    episode._compute_q_survival_bellman_loss(
        claim_batch,
        frozen_p_model=frozen_p,
        q_target_model=q_target,
    )

    g = float(episode.loss_fns["q"].g)
    expected_b_sp = b_syn / (1.0 + 0.2 * (g - 1.0))
    wrong_original_bar_i_b_sp = b_syn / (1.0 + 0.8 * (g - 1.0))
    assert not torch.allclose(expected_b_sp, wrong_original_bar_i_b_sp)
    assert captured_child_b
    for used_b in captured_child_b:
        actual_synthetic_b_sp = used_b[1:2]
        torch.testing.assert_close(actual_synthetic_b_sp, expected_b_sp)
        assert not torch.allclose(actual_synthetic_b_sp, wrong_original_bar_i_b_sp)
        assert not torch.equal(actual_synthetic_b_sp, torch.full_like(actual_synthetic_b_sp, 9.0))


def test_episode0_postbootstrap_q_teacher_does_not_replace_equity_teacher(monkeypatch):
    episode = _episode_for_gate()
    episode.hyperparams.pv_training_flow = "staged"
    episode.hyperparams.firm_target_update = "stage_hard"
    episode.episode_id = 0
    online = episode.models["policy_value"]
    old_target_q = next(episode.firm_target.q_head.parameters()).detach().clone()
    captured = {}

    def fake_bootstrap(_batches):
        with torch.no_grad():
            next(online.q_head.parameters()).add_(3.0)
        return {"status": "accepted", "optimizer_steps": 1}

    def fake_p_stage(_train, _val, equity_teacher, _epochs, *, q_target_model=None):
        captured["equity"] = next(equity_teacher.q_head.parameters()).detach().clone()
        captured["q"] = next(q_target_model.q_head.parameters()).detach().clone()
        return {"status": "accepted"}

    monkeypatch.setattr(episode, "_run_q_bootstrap_stage", fake_bootstrap)
    monkeypatch.setattr(episode, "_run_policy_value_evaluation_stage", fake_p_stage)
    monkeypatch.setattr(
        episode,
        "_run_q_regime_training",
        lambda *args, **kwargs: {
            "status": "rejected_no_survival_bellman",
            "q_stage_rejection_reason": "test_stop",
        },
    )
    batch = {"parent": _state((0.2, 0.4))}
    episode._run_policy_value_staged([batch], [batch], n_epochs=1)
    torch.testing.assert_close(captured["equity"], old_target_q)
    assert not torch.equal(captured["q"], old_target_q)


def _episode_for_gate() -> Episode:
    hp = HyperParams()
    hp.q_parameterization = "hybrid_regime"
    hp.q_zero_boundary_epochs = 1
    hp.q_default_pretrain_epochs = 1
    hp.q_survival_aio_epochs = 1
    hp.q_mixed_polish_epochs = 0
    hp.q_min_default_samples = 1
    hp.q_min_survival_samples = 1
    model = _small_model("hybrid_regime")
    target = deepcopy(model)
    return Episode(
        models={"policy_value": model},
        optimizers={"policy_value": torch.optim.AdamW(model.parameters(), lr=1e-3)},
        config=Config,
        hyperparams=hp,
        device=torch.device("cpu"),
        firm_target=target,
    )


@pytest.mark.parametrize("survival_steps,expected", [(1, "accepted"), (0, "rejected_no_claim_bellman")])
def test_hybrid_required_gate_accepts_structural_zero_step_phases(
    survival_steps, expected
):
    episode = _episode_for_gate()

    def fake_phase(self, *, phase, **kwargs):
        if phase == "zero":
            return {
                "phase": phase,
                "status": "structural_verified",
                "optimizer_steps": 0,
                "metrics": {"q_structural_abs_max": 0.0},
                "coverage": {},
            }
        if phase == "default":
            return {
                "phase": phase,
                "status": "structural_verified",
                "optimizer_steps": 0,
                "metrics": {"q_structural_abs_max": 0.0},
                "coverage": {"default_candidates_selected": 2},
            }
        return {
            "phase": phase,
            "status": "accepted" if survival_steps else "skipped_no_samples",
            "optimizer_steps": survival_steps,
            "metrics": {},
            "coverage": {
                "survival_parent_count": 2 if survival_steps else 0,
                "claim_parent_count": 2 if survival_steps else 0,
                "claim_total_sample_count": 2 if survival_steps else 0,
                "realized_parent_survival_count": 2 if survival_steps else 0,
            },
        }

    episode._run_q_regime_phase = types.MethodType(fake_phase, episode)
    dummy = {"parent": _state((0.2, 0.4))}
    result = episode._run_q_regime_training([dummy], deepcopy(episode.firm_target))
    assert result["status"] == expected
    assert result["q_zero_optimizer_steps"] == 0
    assert result["q_default_optimizer_steps"] == 0


def test_hybrid_required_gate_allows_no_realized_default_observation():
    episode = _episode_for_gate()

    def fake_phase(self, *, phase, **kwargs):
        if phase == "zero":
            return {
                "phase": phase,
                "status": "structural_verified",
                "optimizer_steps": 0,
                "metrics": {"q_structural_abs_max": 0.0},
                "coverage": {},
            }
        if phase == "default":
            return {
                "phase": phase,
                "status": "no_realized_default_observed",
                "optimizer_steps": 0,
                "metrics": {},
                "coverage": {"realized_parent_default_count": 0},
            }
        return {
            "phase": phase,
            "status": "accepted",
            "optimizer_steps": 1,
            "metrics": {},
            "coverage": {
                "claim_total_sample_count": 2,
                "realized_parent_survival_count": 2,
            },
        }

    episode._run_q_regime_phase = types.MethodType(fake_phase, episode)
    result = episode._run_q_regime_training(
        [{"parent": _state((0.2, 0.4))}], deepcopy(episode.firm_target)
    )
    assert result["status"] == "accepted"
    assert result["q_default_coverage"]["realized_parent_default_count"] == 0


def test_hybrid_default_phase_reports_no_realized_default_without_failure():
    episode = _episode_for_gate()
    batch = _q_batch((0.1, 0.2, 0.3))
    result = episode._run_q_regime_phase(
        phase="default",
        batches=[batch],
        frozen_p_model=_CandidateGate(),
        q_target_model=deepcopy(episode.models["policy_value"]),
        epochs=1,
    )
    assert result["status"] == "no_realized_default_observed"
    assert result["optimizer_steps"] == 0
    assert result["coverage"]["realized_parent_default_count"] == 0


def test_hybrid_claim_gate_fails_when_no_realized_survivor_exists():
    episode = _episode_for_gate()
    episode.hyperparams.q_survival_ondist_share = 0.5
    batch = _q_batch((0.7, 0.8, 0.9))
    claim_batch, diagnostics = episode._build_q_survival_batch(
        batch, _CandidateGate(), return_diagnostics=True
    )
    assert claim_batch is None
    assert diagnostics["realized_parent_survival_count"] == 0
    assert diagnostics["realized_parent_default_count"] == 3
    assert diagnostics["claim_total_sample_count"] == 0
