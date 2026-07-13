from __future__ import annotations

import pandas as pd
import torch

from experiments.export_bp_fixed_state_cross_episode import (
    attach_flip_flags,
    build_fixed_probe_panel,
)


def _mock_tensors() -> dict[str, torch.Tensor]:
    n = 18
    parent = torch.zeros(n, 8)
    parent[:, 0] = torch.linspace(0.05, 0.95, n)
    parent[:, 1] = torch.linspace(-1.0, 1.0, n)
    parent[:, 2] = 1.0
    parent[:, 3] = 0.1
    parent[:, 4] = 0.2
    parent[:, 5] = 0.3
    parent[:, 6] = 0.4
    parent[:, 7] = 0.9
    child0 = parent.clone()
    child1 = parent.clone()
    child0[:, 1] += 0.1
    child1[:, 1] -= 0.1
    return {
        "parent": parent,
        "child0": child0,
        "child1": child1,
        "source_index": torch.arange(100, 100 + n),
    }


def test_fixed_probe_panel_uses_reference_tensors_and_unique_keys():
    tensors = _mock_tensors()
    panel = build_fixed_probe_panel(tensors, states_per_group=3)
    states = panel["panel"]
    assert not states.empty
    assert states["source_index"].is_unique
    assert set(states["probe_group"]).issubset(
        {"low-z", "middle-z", "high-z", "low-b", "middle-b", "high-b"}
    )
    selected_source = states["source_index"].tolist()
    reference_source = set(tensors["source_index"].tolist())
    assert set(selected_source).issubset(reference_source)
    assert panel["parent"].shape[0] == len(states)
    assert panel["children"][0].shape[0] == len(states)
    assert panel["m_list"][0].shape[0] == len(states)


def test_flip_flags_and_teacher_semantics_are_explicit():
    rows = []
    for ep, pred, teacher in [(1, 0.1, 0.2), (2, 0.6, 0.25)]:
        rows.append(
            {
                "episode": ep,
                "source_index": 7,
                "branch": "mix",
                "probe_group": "middle-z",
                "teacher_semantics": "checkpoint_online_greedy_proxy",
                "bp_pred": pred,
                "bp_teacher": teacher,
            }
        )
    df = attach_flip_flags(pd.DataFrame(rows), threshold=0.2)
    last = df[df["episode"] == 2].iloc[0]
    assert bool(last["policy_flip"])
    assert not bool(last["teacher_flip"])
    assert bool(last["policy_flip_without_teacher_flip"])
    assert not bool(last["teacher_flip_not_followed_by_policy"])
    assert set(df["teacher_semantics"]) == {"checkpoint_online_greedy_proxy"}
    assert not df.duplicated(["episode", "source_index", "branch"]).any()
