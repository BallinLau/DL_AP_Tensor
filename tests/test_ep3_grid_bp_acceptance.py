"""Acceptance tests for the ep3-baseline Grid BP configuration.

Layout follows the Codex follow-up prompt (EP3 主线使用 Grid BP):

- Test A: BP Head 完全不训练 (BP_HEAD_OPTIMIZER_STEPS=0, P/Q steps > 0)
- Test B: Grid 确实被使用 (bp_actual == bp_grid; child fallback is a
  read-only diagnostic and never enters the economic transition)
- Test C/D: exercised by the GPU smoke on the real ep3 checkpoint + fixed CRN
  (they require the formal tensor simulator and a CUDA device).
- Test E: 原版回归 (head source + head training + inner_cycles=1 keeps the
  ep3 baseline semantics)
"""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from data.simulate_ts import SimulateTS  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402
from training.bp_simulation_policy import GridBPSimulationPolicy  # noqa: E402
from training.episode import Episode  # noqa: E402


def _small_hyperparams() -> HyperParams:
    hp = HyperParams()
    hp.pv_bp_training_mode = "target_grid"
    hp.pv_training_flow = "staged"
    hp.firm_target_update = "stage_hard"
    hp.bp_grid_coarse_size = 3
    hp.bp_grid_refine_enabled = False
    hp.bp_grid_candidate_chunk_size = 2
    hp.bp_grid_parent_chunk_size = 2
    hp.bp_grid_max_expanded_states = 16
    hp.bp_grid_mix_policy_weight = 1.0
    hp.bp_grid_margin_scale = 1e-4
    hp.bp_grid_policy_loss_space = "logit"
    hp.bp_grid_logit_target_eps = 1e-4
    hp.pv_use_clipped_m = False
    hp.pv_target_grid_val_fraction = 0.0
    hp.bp_distill_epochs = 1
    hp.bp_distill_patience = 1
    hp.pv_eval_epochs = 1
    hp.policy_lr = 1e-3
    hp.q_require_zero_phase = False
    hp.q_require_default_phase = False
    hp.q_require_survival_phase = False
    return hp


def _batch(device: torch.device):
    parent_state = torch.tensor(
        [
            [0.10, 0.20, 1.00, 0.20, 0.10, -2.00, 4.00],
            [0.30, 0.10, 0.00, 0.40, 0.00, -1.80, 4.20],
        ],
        device=device,
        dtype=torch.float32,
    )
    child0 = parent_state.clone()
    child1 = parent_state.clone()
    child0[:, 1:2] = child0[:, 1:2] + 0.05
    child1[:, 1:2] = child1[:, 1:2] - 0.03
    child0[:, 2:3] = torch.tensor([[1.0], [0.0]], device=device)
    child1[:, 2:3] = torch.tensor([[0.0], [1.0]], device=device)
    m0 = torch.full((2, 1), 0.98, device=device)
    m1 = torch.full((2, 1), 1.02, device=device)
    return {
        "parent": torch.cat([parent_state, torch.ones(2, 1, device=device)], dim=1),
        "children": [
            torch.cat([child0, m0], dim=1),
            torch.cat([child1, m1], dim=1),
        ],
    }


def _episode() -> Episode:
    device = torch.device("cpu")
    Config.DEVICE = device
    torch.manual_seed(123)
    online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=1e-3)
    episode = Episode(
        models={"policy_value": online},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=_small_hyperparams(),
        device=device,
        firm_target=target,
    )
    episode._prepare_sdf_shock_bank_for_epoch = lambda *args, **kwargs: None
    episode.evaluate_bellman_convergence = lambda *args, **kwargs: {
        "passed": True,
        "equations": {
            key: {
                "conditional_train_m": {
                    "mean": 0.01,
                    "p90": 0.02,
                    "passed": True,
                    "n_parent": 2,
                }
            }
            for key in ("p0", "pi", "q")
        },
    }
    return episode


# --- Test A: BP Head 完全不训练 --------------------------------------------

def test_inner3_head_disabled_keeps_zero_head_steps_and_trains_pq(monkeypatch):
    episode = _episode()
    episode.hyperparams.pv_bp_head_training_enabled = False
    episode.hyperparams.firm_inner_cycles = 3
    batch = _batch(episode.device)
    model = episode.models["policy_value"]
    bp_head_params = list(model.bp0_head.parameters()) + list(
        model.bpi_head.parameters()
    )
    bp_before = [value.detach().clone() for value in bp_head_params]

    monkeypatch.setattr(
        episode,
        "_run_q_bootstrap_stage",
        lambda *_args, **_kwargs: {"status": "accepted", "optimizer_steps": 0},
    )
    monkeypatch.setattr(
        episode,
        "_run_q_regime_training",
        lambda *_args, **_kwargs: {
            "status": "accepted",
            "q_stage_required_gate_passed": True,
        },
    )
    monkeypatch.setattr(episode, "_update_firm_target_now", lambda *_args: None)

    result = episode._run_policy_value_staged([batch], [batch], n_epochs=1)

    metadata = result["metadata"]
    assert metadata["policy_value_stage_status"] == "accepted"

    # The BP distilation stage must not construct an optimizer at all.
    bp_stage = metadata["bp_distillation_stage"]
    assert bp_stage["status"] == "diagnostic_only_training_disabled"
    assert bp_stage["bp_head_optimizer_steps"] == 0
    assert bp_stage["optimizer_steps"] == 0
    assert all(
        torch.equal(before, after)
        for before, after in zip(bp_before, bp_head_params)
    )
    # The P/Q stages genuinely stepped the optimizer.
    assert metadata["policy_value_evaluation_stage"]["optimizer_steps"] > 0

    # All three frozen-macro cycles executed and each reported P0/PI/Q residuals.
    assert metadata["firm_inner_cycles"] == 3
    cycles = metadata["inner_cycle_diagnostics"]
    assert len(cycles) == 3
    for index, cycle in enumerate(cycles, start=1):
        for key in ("P0", "PI", "Q"):
            residual = cycle[f"Inner{index}_{key}_residual"]
            assert residual["mean"] is not None
            assert residual["passed"] is True


# --- Test B: Grid 确实被使用 ------------------------------------------------

def test_grid_policy_parent_returns_grid_action_and_child_falls_back_to_head():
    # The parent node (branch == -1) must return the grid-resolved action.
    marker = torch.full((2, 1), 0.42)
    policy = object.__new__(GridBPSimulationPolicy)
    policy.child_head_fallback_calls = 0
    policy.evaluate_actions = lambda **kwargs: {"bp": marker}

    resolved = policy(
        branch=-1,
        firm_state=torch.zeros(2, 7),
        hatc_cal=torch.zeros(2, 1),
        lnk_cal=torch.zeros(2, 1),
    )
    assert resolved is marker

    # Child branches only carry the read-only head diagnostic.
    head = torch.full((2, 1), 0.61)
    child_value = policy(branch=0, bp_head=head)
    assert child_value is head
    assert policy.child_head_fallback_calls == 1


def test_simulate_resolve_bp_action_grid_source_returns_policy_output():
    grid_marker = torch.full((2, 1), 0.55)
    head_value = torch.full((2, 1), 0.33)

    class StubGridPolicy:
        def __call__(self, **context):
            assert int(context["branch"]) == -1
            return grid_marker

    sim = SimulateTS.__new__(SimulateTS)
    sim.bp_action_override = None
    sim.bp_action_source = "grid"
    sim.bp_grid_policy = StubGridPolicy()

    resolved = sim._resolve_bp_action(branch=-1, bp_head=head_value)
    assert resolved is grid_marker
    assert not torch.equal(resolved, head_value)

    sim.bp_action_source = "head"
    resolved_head = sim._resolve_bp_action(branch=-1, bp_head=head_value)
    assert resolved_head is head_value


# --- Test E: 原版回归 --------------------------------------------------------

def test_head_configuration_keeps_ep3_baseline_simulation_semantics():
    episode = _episode()
    episode.episode_id = 4
    episode.hyperparams.simulation_bp_action_source = "head"
    episode.hyperparams.pv_bp_head_training_enabled = True
    episode.hyperparams.firm_inner_cycles = 1
    episode.hyperparams.initial_hstar_enabled = False

    kwargs = episode._prepare_simulation_kwargs({})

    assert kwargs["bp_action_source"] == "head"
    assert "bp_grid_policy" not in kwargs
    assert episode._bp_head_training_enabled() is True
    assert int(getattr(episode.hyperparams, "firm_inner_cycles", 1)) == 1