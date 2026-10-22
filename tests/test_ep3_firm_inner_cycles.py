"""回归测试：Firm Inner Cycle 诊断键、阶段重置与冻结哈希（e35e2b8 修正）。

- 修复了 logger 使用静态键 "Inner%d_P0_residual" 导致的 KeyError；
- 每个 Firm 阶段开始必须清空上一阶段的 inner-cycle 诊断；
- firm_inner_cycles=1 时保留原基线产物键名（post_p / post_q_final）。
"""

from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402
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


def _episode(firm_inner_cycles: int = 1, head_training: bool = False) -> Episode:
    device = torch.device("cpu")
    Config.DEVICE = device
    torch.manual_seed(123)
    online = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=1e-3)
    hyperparams = _small_hyperparams()
    hyperparams.firm_inner_cycles = firm_inner_cycles
    hyperparams.pv_bp_head_training_enabled = head_training
    episode = Episode(
        models={"policy_value": online},
        optimizers={"policy_value": optimizer},
        config=Config,
        hyperparams=hyperparams,
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
    # 一个真实可哈希的冻结模块，用于验证 FC1/SDF 哈希在三轮内不变。
    episode.models["sdf_fc1"] = nn.Linear(1, 1, bias=False).to(device)
    return episode


def _patch_stages(episode: Episode, monkeypatch) -> None:
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
            "optimizer_steps": 7,
        },
    )
    monkeypatch.setattr(episode, "_update_firm_target_now", lambda *_args: None)


def test_inner3_diagnostics_keys_and_stage_reset(monkeypatch):
    episode = _episode(firm_inner_cycles=3)
    episode._firm_inner_cycle_diagnostics = ["stale-from-previous-stage"]
    batch = _batch(episode.device)
    _patch_stages(episode, monkeypatch)
    sdf_hash_before = episode._state_dict_hash(episode.models["sdf_fc1"])

    result = episode._run_policy_value_staged([batch], [batch], n_epochs=1)

    metadata = result["metadata"]
    assert metadata["policy_value_stage_status"] == "accepted"
    cycles = metadata["inner_cycle_diagnostics"]
    assert len(cycles) == 3
    # 阶段开始已清空旧诊断：结果中不得混入上一阶段的记录。
    assert "stale-from-previous-stage" not in cycles
    assert episode._firm_inner_cycle_diagnostics == cycles
    for index, cycle in enumerate(cycles, start=1):
        # 旧实现在这里用静态键 "Inner%d_P0_residual" 触发 KeyError；
        # 本测试运行到这一步即证明 logger 读取的是格式化后的真实键。
        for key in ("P0", "PI", "Q"):
            residual = cycle[f"Inner{index}_{key}_residual"]
            assert residual["mean"] is not None
            assert residual["passed"] is True
        assert int(cycle["q_optimizer_steps"]) == 7
        assert int(cycle["p_optimizer_steps"]) > 0
        # 三轮内 FC1/SDF 冻结哈希完全一致。
        assert cycle["sdf_fc1_hash"] == sdf_hash_before
        assert cycle["sdf_fc1_hash_after"] == sdf_hash_before
    # P/Q 网络确实被更新（函数漂移非零）。
    assert cycles[-1]["function_drift"]["l2_delta"] > 0.0


def test_inner_cycles_1_keeps_ep3_baseline_artifact_keys(monkeypatch):
    episode = _episode(firm_inner_cycles=1)
    batch = _batch(episode.device)
    _patch_stages(episode, monkeypatch)

    result = episode._run_policy_value_staged([batch], [batch], n_epochs=1)

    metadata = result["metadata"]
    assert metadata["firm_inner_cycles"] == 1
    assert len(metadata["inner_cycle_diagnostics"]) == 1
    # 单循环保持原基线产物键，不引入 cycle 后缀。
    assert "post_p" in metadata["stage_component_hashes"]
    assert "post_q_final" in metadata["stage_component_hashes"]


def test_head_training_disabled_reports_zero_head_steps(monkeypatch):
    episode = _episode(firm_inner_cycles=3, head_training=False)
    batch = _batch(episode.device)
    model = episode.models["policy_value"]
    bp_head_params = list(model.bp0_head.parameters()) + list(
        model.bpi_head.parameters()
    )
    bp_before = [value.detach().clone() for value in bp_head_params]
    _patch_stages(episode, monkeypatch)

    result = episode._run_policy_value_staged([batch], [batch], n_epochs=1)

    bp_stage = result["metadata"]["bp_distillation_stage"]
    assert bp_stage["status"] == "diagnostic_only_training_disabled"
    assert bp_stage["bp_head_optimizer_steps"] == 0
    assert all(
        torch.equal(before, after)
        for before, after in zip(bp_before, bp_head_params)
    )