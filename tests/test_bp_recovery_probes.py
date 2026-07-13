from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.run_bp_recovery_probes import (
    PROBE_BASELINE,
    PROBE_LOGIT,
    PROBE_RECENTER,
    ProbeConfig,
    apply_bias_recenter,
    forward_policy_bundle,
    logit_space_loss,
    output_space_loss,
    run_probe_suite,
)
from models.policy_value import PolicyValueModel
from models.share_layer import BpHead


def test_bp_head_forward_logits_matches_forward_without_state_dict_change():
    torch.manual_seed(7)
    head = BpHead(input_dim=3, hidden_dims=[4], requires_i=False)
    keys_before = set(head.state_dict().keys())
    h = torch.randn(5, 3)
    torch.testing.assert_close(head(h), torch.sigmoid(head.forward_logits(h)), rtol=0, atol=0)
    assert set(head.state_dict().keys()) == keys_before

    head_i = BpHead(input_dim=3, hidden_dims=[4], requires_i=True)
    h_i = torch.randn(5, 3)
    i = torch.randn(5, 1)
    torch.testing.assert_close(head_i(h_i, i), torch.sigmoid(head_i.forward_logits(h_i, i)), rtol=0, atol=0)


def test_bias_recenter_translates_logits_without_changing_pairwise_distances():
    torch.manual_seed(11)
    model = PolicyValueModel(share_hidden_dims=[], share_output_dim=3)
    parent = torch.tensor(
        [
            [0.1, -0.2, 1.0, 0.0, 0.2, 0.1, 0.0],
            [0.2, -0.1, 1.0, 0.1, 0.3, 0.2, 0.1],
            [0.3, 0.0, 1.0, 0.2, 0.4, 0.3, 0.2],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        model.bp0_head.network.network[-2].weight.zero_()
        model.bp0_head.network.network[-2].bias.fill_(-30.0)
        model.bpi_head.network.network[-2].weight.zero_()
        model.bpi_head.network.network[-2].bias.fill_(-31.0)

    train_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    before = forward_policy_bundle(model, parent)
    stats = apply_bias_recenter(
        model,
        parent,
        train_idx,
        ProbeConfig(
            episode=0,
            steps=0,
            record_steps=[0],
            learning_rate=1e-3,
            weight_decay=0.0,
            holdout_fraction=0.0,
            seed=1,
            initial_bp_target=0.5,
            logit_target_eps=1e-4,
            logit_huber_delta=1.0,
            probes=[PROBE_RECENTER],
        ),
    )
    after = forward_policy_bundle(model, parent)
    for branch, shift_key in [("bp0", "bp0_bias_shift"), ("bpI", "bpI_bias_shift")]:
        diff = after[f"{branch}_logit"] - before[f"{branch}_logit"]
        torch.testing.assert_close(diff, torch.full_like(diff, stats[shift_key]), rtol=1e-5, atol=1e-5)
    assert abs(float(after["bp0_logit"].median().item())) < 1e-5
    assert abs(float(after["bpI_logit"].median().item())) < 1e-5


def test_logit_loss_has_large_gradient_when_output_loss_is_saturated():
    hp = SimpleNamespace(
        bp_grid_policy_huber_delta=0.05,
        bp_grid_policy_weight=1.0,
        bp_grid_mix_policy_weight=1.0,
    )
    cfg = ProbeConfig(
        episode=0,
        steps=0,
        record_steps=[0],
        learning_rate=1e-3,
        weight_decay=0.0,
        holdout_fraction=0.0,
        seed=1,
        initial_bp_target=0.5,
        logit_target_eps=1e-4,
        logit_huber_delta=1.0,
        probes=[PROBE_LOGIT],
    )
    target = torch.tensor([[0.4]])
    confidence = torch.ones_like(target)
    targets = {
        "bp0": target,
        "bpI": target,
        "bp_mix": target,
        "bp0_confidence": confidence,
        "bpI_confidence": confidence,
        "bp_mix_confidence": confidence,
        "bp_mix_survival_weight": confidence,
    }
    indices = torch.tensor([0], dtype=torch.long)

    output_logit = torch.tensor([[-30.0]], requires_grad=True)
    output_bundle = {
        "bp0": torch.sigmoid(output_logit),
        "bpI": torch.sigmoid(output_logit.detach()),
        "bp_mix": torch.sigmoid(output_logit.detach()),
        "bp0_logit": output_logit,
        "bpI_logit": output_logit.detach(),
    }
    output_loss, _ = output_space_loss(output_bundle, targets, indices, hp)
    output_loss.backward()
    output_grad = abs(float(output_logit.grad.item()))

    logit = torch.tensor([[-30.0]], requires_grad=True)
    logit_bundle = {
        "bp0": torch.sigmoid(logit),
        "bpI": torch.sigmoid(logit.detach()),
        "bp_mix": torch.sigmoid(logit.detach()),
        "bp0_logit": logit,
        "bpI_logit": logit.detach(),
    }
    loss, _ = logit_space_loss(logit_bundle, targets, indices, hp, target_eps=1e-4, huber_delta=1.0)
    loss.backward()
    logit_grad = abs(float(logit.grad.item()))
    assert logit_grad > output_grad * 1e8


def test_probe_suite_is_deterministic_and_does_not_mutate_online_model_or_targets():
    torch.manual_seed(13)
    model = PolicyValueModel(share_hidden_dims=[], share_output_dim=4)
    parent = torch.tensor(
        [
            [0.10, -0.2, 1.0, 0.0, 0.2, 0.1, 0.0],
            [0.20, -0.1, 1.0, 0.1, 0.3, 0.2, 0.1],
            [0.30, 0.0, 1.0, 0.2, 0.4, 0.3, 0.2],
            [0.40, 0.1, 1.0, 0.3, 0.5, 0.4, 0.3],
        ],
        dtype=torch.float32,
    )
    source_index = torch.arange(parent.shape[0], dtype=torch.long)
    target = torch.full((parent.shape[0], 1), 0.45)
    confidence = torch.ones_like(target)
    targets = {
        "bp0": target.clone(),
        "bpI": target.clone(),
        "bp_mix": target.clone(),
        "bp0_confidence": confidence.clone(),
        "bpI_confidence": confidence.clone(),
        "bp_mix_confidence": confidence.clone(),
        "bp_mix_survival_weight": confidence.clone(),
    }
    target_before = {key: value.clone() for key, value in targets.items()}
    online_before = {key: value.clone() for key, value in model.state_dict().items()}
    hp = SimpleNamespace(
        bp_grid_policy_huber_delta=0.05,
        bp_grid_policy_weight=1.0,
        bp_grid_mix_policy_weight=1.0,
    )
    cfg = ProbeConfig(
        episode=2,
        steps=2,
        record_steps=[0, 1, 2],
        learning_rate=1e-3,
        weight_decay=0.0,
        holdout_fraction=0.25,
        seed=99,
        initial_bp_target=0.5,
        logit_target_eps=1e-4,
        logit_huber_delta=1.0,
        probes=[PROBE_BASELINE, PROBE_RECENTER, PROBE_LOGIT],
    )

    history1, states1, summary1 = run_probe_suite(
        online_model=model,
        parent_state=parent,
        source_index=source_index,
        targets=targets,
        hp=hp,
        cfg=cfg,
    )
    history2, states2, summary2 = run_probe_suite(
        online_model=model,
        parent_state=parent,
        source_index=source_index,
        targets=targets,
        hp=hp,
        cfg=cfg,
    )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, online_before[key], rtol=0, atol=0)
    for key, value in targets.items():
        torch.testing.assert_close(value, target_before[key], rtol=0, atol=0)
    assert set(summary1["probe"]) == {PROBE_BASELINE, PROBE_RECENTER, PROBE_LOGIT}
    assert set(history1["branch"]) == {"p0", "pi", "mix", "combined"}
    assert "bp0_bias_shift" in states1.columns
    assert "bp0_target_logit" in states1.columns
    torch.testing.assert_close(
        torch.tensor(history1["output_mae"].to_numpy()),
        torch.tensor(history2["output_mae"].to_numpy()),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.tensor(summary1["final_train_mae"].to_numpy()),
        torch.tensor(summary2["final_train_mae"].to_numpy()),
        rtol=0,
        atol=0,
    )
