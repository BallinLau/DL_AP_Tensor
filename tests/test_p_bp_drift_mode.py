"""PV_BP_DRIFT_MODE 执行层性能开关回归测试（纯观测口径降级）。

greedy BP drift 无任何消费方；该开关只裁剪每次 P Policy Iteration 的纯观测
冗余计算（_recompute_bp_star 贪心重解），绝不改变训练 / Acceptance
（frozen_improved 只看 Validation 残差）/ 回滚语义。默认 full 保持现状逐位一致。

Test 1：off 模式 0 次 _recompute_bp_star，greedy_bp_drift 各统计为 NaN +
        mode=off；廉价的 value_output_drift 照常计算。
Test 2：subset 模式重解行数 ≤ p_bp_drift_subset_rows，metadata 标注
        mode=subset 与 n_rows。
Test 3：full（默认）保持现状：逐 train batch × 2 分支全量重解，summary 形状
        不带任何新开关字段。
Test 4：非法取值 fail fast（ValueError，消息清晰），含 p_bp_drift_subset_rows。
"""

from pathlib import Path
import math
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from models.policy_value import PolicyValueModel  # noqa: E402
from training import frozen_policy_p_solver as solver  # noqa: E402
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
    # Hybrid P solver：微小预算保持 CPU 测试快速。
    hp.p_solver = "frozen_policy_dynamic_residual"
    hp.p_policy_iterations_per_inner = 3
    hp.p_root_adam_steps = 2
    hp.p_root_adam_lr = 1e-3
    hp.p_root_adam_clip = 10.0
    hp.p_root_eval_every = 1
    hp.p_root_lbfgs_max_iter = 2
    hp.p_root_lbfgs_chunk_iterations = 2
    hp.p_root_lbfgs_history_size = 5
    hp.q_solver = "ep3_regime_staged"
    hp.pv_bp_head_training_enabled = False
    return hp


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
    episode.models["sdf_fc1"] = nn.Linear(1, 1, bias=False).to(device)
    return episode


def _set_head_bias(model: PolicyValueModel, value: float) -> None:
    with torch.no_grad():
        for head in (model.v0_head, model.vi_head):
            linears = [m for m in head.network.modules() if isinstance(m, nn.Linear)]
            linears[-1].bias.fill_(float(value))


def _batch(device: torch.device, n_rows: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    parent_state = torch.zeros(n_rows, 7, device=device)
    parent_state[:, 0] = torch.rand(n_rows, generator=g) * 0.5
    parent_state[:, 1] = torch.randn(n_rows, generator=g) * 0.3
    parent_state[:, 2] = 0.0
    parent_state[: max(n_rows // 2, 1), 2] = 1.0
    parent_state[:, 3] = torch.rand(n_rows, generator=g) * 0.4
    parent_state[:, 4] = torch.randn(n_rows, generator=g) * 0.1
    parent_state[:, 5] = -2.0 + torch.randn(n_rows, generator=g) * 0.2
    parent_state[:, 6] = 4.0 + torch.randn(n_rows, generator=g) * 0.1
    child0 = parent_state.clone()
    child1 = parent_state.clone()
    child0[:, 1:2] += 0.05
    child1[:, 1:2] -= 0.03
    child0[:, 2:3] = 1.0 - parent_state[:, 2:3]
    child1[:, 2:3] = parent_state[:, 2:3]
    m0 = torch.full((n_rows, 1), 0.98, device=device)
    m1 = torch.full((n_rows, 1), 1.02, device=device)
    return {
        "parent": torch.cat([parent_state, torch.ones(n_rows, 1, device=device)], dim=1),
        "children": [
            torch.cat([child0, m0], dim=1),
            torch.cat([child1, m1], dim=1),
        ],
        "source_id": torch.zeros(n_rows, dtype=torch.long, device=device),
    }


def _run_iteration(episode: Episode, model, train, validation, **kwargs):
    params = dict(
        train_batches=train,
        validation_batches=validation,
        q_model=model,
        inner_index=1,
        iteration_index=1,
        adam_steps=1,
        adam_lr=1e-3,
        adam_clip=10.0,
        eval_every=1,
        lbfgs_max_iter=1,
        lbfgs_chunk_iterations=1,
        lbfgs_history_size=2,
    )
    params.update(kwargs)
    return solver.run_p_policy_iteration(episode, model, **params)


def _count_calls(monkeypatch, name: str):
    calls = {"n": 0, "rows": 0}
    original = getattr(solver, name)

    def _wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        calls["n"] += 1
        if torch.is_tensor(result):
            calls["rows"] += int(result.shape[0])
        return result

    monkeypatch.setattr(solver, name, _wrapped)
    return calls


# --- Test 1: off 模式 --------------------------------------------------------

def test_bp_drift_mode_off_skips_recompute_reports_nan(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    episode.hyperparams.pv_bp_drift_mode = "off"
    train = _batch(episode.device, n_rows=6, seed=1)
    calls = _count_calls(monkeypatch, "_recompute_bp_star")

    summary = _run_iteration(episode, model, [train], [train])

    assert calls["n"] == 0
    drift = summary["greedy_bp_drift"]
    assert drift["mode"] == "off"
    for key in ("p0_mean_abs", "pi_mean_abs", "p0_p90", "pi_p90"):
        assert math.isnan(float(drift[key])), key
    # value_output_drift（廉价）保持计算。
    assert math.isfinite(float(summary["function_drift"]["p0_output_mean_abs"]))
    assert math.isfinite(float(summary["function_drift"]["pi_output_mean_abs"]))
    # Acceptance / 回滚不受影响。
    assert summary["status"] in {"accepted_improved", "accepted_reverted"}


# --- Test 2: subset 模式 -----------------------------------------------------

def test_bp_drift_mode_subset_limits_recomputed_rows(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    episode.hyperparams.pv_bp_drift_mode = "subset"
    episode.hyperparams.p_bp_drift_subset_rows = 5
    train = [
        _batch(episode.device, n_rows=6, seed=1),
        _batch(episode.device, n_rows=7, seed=2),
    ]  # 13 行 > 5
    calls = _count_calls(monkeypatch, "_recompute_bp_star")

    summary = _run_iteration(episode, model, train, train)

    drift = summary["greedy_bp_drift"]
    assert drift["mode"] == "subset"
    assert int(drift["n_rows"]) <= 5
    # 每个重解分支恰好 n_rows 行（p0 + pi 两个分支）。
    assert calls["rows"] == 2 * int(drift["n_rows"])
    for key in ("p0_mean_abs", "pi_mean_abs", "p0_p90", "pi_p90"):
        assert math.isfinite(float(drift[key])), key
    assert summary["status"] in {"accepted_improved", "accepted_reverted"}


# --- Test 3: full 默认保持现状 -----------------------------------------------

def test_bp_drift_mode_full_default_keeps_legacy_behaviour(monkeypatch):
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    assert episode.hyperparams.pv_bp_drift_mode == "full"
    assert episode.hyperparams.p_bp_drift_subset_rows == 4096
    train = [
        _batch(episode.device, n_rows=6, seed=1),
        _batch(episode.device, n_rows=4, seed=2),
    ]
    calls = _count_calls(monkeypatch, "_recompute_bp_star")

    summary = _run_iteration(episode, model, train, train)

    # 全量：每 train batch × 2 分支各重解一次。
    assert calls["n"] == 2 * len(train)
    # 默认 summary 形状逐位保持原样：不携带任何新开关字段。
    drift = summary["greedy_bp_drift"]
    assert set(drift) == {"p0_mean_abs", "pi_mean_abs", "p0_p90", "pi_p90"}
    for key in drift:
        assert math.isfinite(float(drift[key])), key
    assert summary["status"] in {"accepted_improved", "accepted_reverted"}


# --- Test 4: 非法取值 fail fast ----------------------------------------------

@pytest.mark.parametrize(
    "field,value,message",
    [
        ("pv_bp_drift_mode", "sometimes", "pv_bp_drift_mode"),
    ],
)
def test_illegal_perf_switch_values_fail_fast(field, value, message):
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    setattr(episode.hyperparams, field, value)
    batch = _batch(episode.device, n_rows=4, seed=1)
    with pytest.raises(ValueError, match=message):
        _run_iteration(episode, model, [batch], [batch])


def test_illegal_bp_drift_subset_rows_fails_fast():
    episode = _episode()
    model = episode.models["policy_value"]
    _set_head_bias(model, 2.0)
    episode.hyperparams.pv_bp_drift_mode = "subset"
    episode.hyperparams.p_bp_drift_subset_rows = 0
    batch = _batch(episode.device, n_rows=4, seed=1)
    with pytest.raises(ValueError, match="p_bp_drift_subset_rows"):
        _run_iteration(episode, model, [batch], [batch])
