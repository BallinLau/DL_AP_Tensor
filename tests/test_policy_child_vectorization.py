import copy
import unittest
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models import PolicyValueModel
from losses.p0_loss import P0Loss
from losses.pi_loss import PILoss
from config import Config, HyperParams
from evaluation.bp_diagnostics import (
    BP_EVAL_ALLOWED_EXPANDED_STATES,
    BP_EVAL_SAFE_DEFAULT_EXPANDED_STATES,
    resolve_bp_eval_max_expanded_states,
)
from training.bp_grid_teacher import (
    BPGridTeacher,
    _forward_equity_grid_children,
    resolve_grid_chunk_plan,
)
from training.episode import Episode
from utils.firm_transition import apply_refinancing_policy


def _make_children(batch_size: int = 5, n_children: int = 3):
    torch.manual_seed(123)
    children = []
    for j in range(n_children):
        child = torch.randn(batch_size, 8, dtype=torch.float64)
        child[:, 2:3] = torch.sigmoid(child[:, 2:3])
        child[:, 7:8] = 0.8 + 0.1 * j
        children.append(child)
    return children


def _reference_child_states(children, bp, b_parent, eta_current=None):
    if eta_current is None:
        eta_current = torch.ones_like(b_parent)
    reference_states = []
    for child in children:
        raw = child[:, :7] if child.shape[1] > 7 else child
        b_next = apply_refinancing_policy(
            b_current=b_parent,
            bp_candidate=bp,
            eta_current=eta_current,
        )
        state = torch.cat(
            [
                b_next,
                raw[:, 1:],
            ],
            dim=1,
        )
        reference_states.append(state)
    return torch.stack(reference_states, dim=1)


class _CountingTargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.equity_calls = 0
        self.q_calls = 0

    def _q_output(self, state):
        self.q_calls += 1
        return (
            0.70
            + 0.03 * state[:, 0:1]
            - 0.02 * state[:, 1:2]
            + 0.01 * state[:, 4:5]
        )

    def forward_equity(self, state):
        self.equity_calls += 1
        p = (
            1.10
            + 0.20 * state[:, 0:1]
            - 0.10 * state[:, 1:2]
            + 0.05 * state[:, 3:4]
            + 0.03 * state[:, 4:5]
        )
        bar_z = torch.sigmoid(0.40 * state[:, 1:2] - 0.30 * state[:, 0:1])
        return {"P": p, "bar_z": bar_z, "Q": self._q_output(state)}


def _reference_target_grid_chunk(teacher, parent_state, children, m_list, bp_grid, branch, mix_weight=None):
    batch_size, n_grid = bp_grid.shape
    q_current = teacher.target_model._q_output(parent_state)
    b_parent = parent_state[:, 0:1]
    eta_current = parent_state[:, 2:3].clamp(0.0, 1.0)
    issue_state = parent_state.unsqueeze(1).expand(batch_size, n_grid, parent_state.shape[-1]).reshape(batch_size * n_grid, -1).clone()
    issue_state[:, 0:1] = bp_grid.reshape(-1, 1)
    q_issue = teacher.target_model._q_output(issue_state).reshape(batch_size, n_grid)

    value_grid = torch.zeros(batch_size, n_grid, dtype=parent_state.dtype, device=parent_state.device)
    cashflow_grid = torch.zeros_like(value_grid)
    continuation_grid = torch.zeros_like(value_grid)
    p_sum = torch.zeros_like(value_grid)
    default_sum = torch.zeros_like(value_grid)
    x_parent = parent_state[:, 4:5]
    z_parent = parent_state[:, 1:2]
    i_parent = parent_state[:, 3:4]
    q_current_grid = q_current.expand(batch_size, n_grid)
    mix_w = None if mix_weight is None else mix_weight.clamp(0.0, 1.0).expand(batch_size, n_grid)

    for child, m in zip(children, m_list):
        child_raw = child[:, :7] if child.shape[1] > 7 else child
        child_state = child_raw.unsqueeze(1).expand(batch_size, n_grid, child_raw.shape[-1]).reshape(batch_size * n_grid, -1).clone()
        child_b_grid = apply_refinancing_policy(
            b_current=b_parent.unsqueeze(1),
            bp_candidate=bp_grid.unsqueeze(-1),
            eta_current=eta_current.unsqueeze(1),
        )
        child_state[:, 0:1] = child_b_grid.reshape(-1, 1)
        out = teacher.target_model.forward_equity(child_state)
        p_child = out["P"].reshape(batch_size, n_grid)
        bar_z_child = out["bar_z"].reshape(batch_size, n_grid).clamp(0.0, 1.0)

        cf0 = teacher.p0_loss_fn.compute_cashflow_p0(
            x_parent.expand(batch_size, n_grid).reshape(-1, 1),
            z_parent.expand(batch_size, n_grid).reshape(-1, 1),
            b_parent.expand(batch_size, n_grid).reshape(-1, 1),
            q_current_grid.reshape(-1, 1),
            q_issue.reshape(-1, 1),
            eta_current.expand(batch_size, n_grid).reshape(-1, 1),
        ).reshape(batch_size, n_grid)
        continuation0 = m.expand(batch_size, n_grid) * p_child
        value0 = cf0 + continuation0

        cfi = teacher.pi_loss_fn.compute_cashflow_pi(
            x_parent.expand(batch_size, n_grid).reshape(-1, 1),
            z_parent.expand(batch_size, n_grid).reshape(-1, 1),
            b_parent.expand(batch_size, n_grid).reshape(-1, 1),
            i_parent.expand(batch_size, n_grid).reshape(-1, 1),
            q_current_grid.reshape(-1, 1),
            q_issue.reshape(-1, 1),
            eta_current.expand(batch_size, n_grid).reshape(-1, 1),
        ).reshape(batch_size, n_grid)
        continuationi = Config.G * m.expand(batch_size, n_grid) * p_child
        valuei = cfi + continuationi

        if branch == "p0":
            branch_cashflow = cf0
            branch_continuation = continuation0
            branch_value = value0
        elif branch == "pi":
            branch_cashflow = cfi
            branch_continuation = continuationi
            branch_value = valuei
        else:
            branch_cashflow = (1.0 - mix_w) * cf0 + mix_w * cfi
            branch_continuation = (1.0 - mix_w) * continuation0 + mix_w * continuationi
            branch_value = (1.0 - mix_w) * value0 + mix_w * valuei

        value_grid = value_grid + branch_value
        cashflow_grid = cashflow_grid + branch_cashflow
        continuation_grid = continuation_grid + branch_continuation
        p_sum = p_sum + p_child
        default_sum = default_sum + bar_z_child

    n_children = len(children)
    return {
        "bp_grid": bp_grid,
        "value_grid": value_grid / n_children,
        "cashflow_grid_mean": cashflow_grid / n_children,
        "continuation_grid_mean": continuation_grid / n_children,
        "q_issue_grid": q_issue,
        "p_child_grid_mean": p_sum / n_children,
        "default_grid_mean": default_sum / n_children,
        "argmax_index": (value_grid / n_children).argmax(dim=1, keepdim=True),
    }


class PolicyChildVectorizationTest(unittest.TestCase):
    def test_grid_chunk_plan_enforces_hard_cap_at_realistic_child_counts(self):
        for n_children in (256, 128, 64):
            plan = resolve_grid_chunk_plan(
                n_parent=10201,
                n_grid=21,
                n_children=n_children,
                configured_parent_chunk=2048,
                configured_candidate_chunk=0,
                max_expanded_states=65536,
            )
            self.assertLessEqual(plan.expanded_states_per_forward, 65536)
            self.assertLessEqual(
                plan.parent_chunk_effective
                * plan.candidate_chunk_effective
                * n_children,
                65536,
            )

    def test_grid_chunk_plan_rejects_even_one_parent_child_bundle_above_cap(self):
        with self.assertRaisesRegex(ValueError, "smaller than one parent"):
            resolve_grid_chunk_plan(
                n_parent=10,
                n_grid=21,
                n_children=256,
                configured_parent_chunk=2048,
                configured_candidate_chunk=0,
                max_expanded_states=255,
            )

    def test_policy_value_model_default_has_no_batchnorm_or_dropout(self):
        model = PolicyValueModel()
        self.assertFalse(any(isinstance(m, nn.BatchNorm1d) for m in model.modules()))
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in model.modules()))

    def test_vectorized_child_state_matches_reference_loop(self):
        episode = Episode.__new__(Episode)
        children = _make_children()
        bp = torch.linspace(0.1, 0.5, steps=5, dtype=torch.float64).reshape(-1, 1)
        b_parent = torch.linspace(0.2, 0.6, steps=5, dtype=torch.float64).reshape(-1, 1)
        eta_current = torch.tensor([[0.0], [1.0], [0.0], [1.0], [1.0]], dtype=torch.float64)
        vectorized_states, eta_children = episode._build_vectorized_policy_child_states(
            children=children,
            bp=bp,
            b_parent=b_parent,
            eta_current=eta_current,
        )
        reference_states = _reference_child_states(children, bp, b_parent, eta_current)

        torch.testing.assert_close(vectorized_states, reference_states, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            eta_children,
            torch.stack([child[:, 2:3] for child in children], dim=1),
            rtol=0.0,
            atol=0.0,
        )

    def test_canonical_transition_uses_current_eta(self):
        b_parent = torch.tensor([[0.6]], dtype=torch.float64)
        bp = torch.tensor([[0.2]], dtype=torch.float64)

        # TEST A: eta_t = 0 keeps b_t, eta_t = 1 applies bp_t.
        torch.testing.assert_close(
            apply_refinancing_policy(
                b_current=torch.tensor([[0.4], [0.4]], dtype=torch.float64),
                bp_candidate=torch.tensor([[0.2], [0.9]], dtype=torch.float64),
                eta_current=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            ),
            torch.tensor([[0.4], [0.9]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            apply_refinancing_policy(
                b_current=b_parent,
                bp_candidate=bp,
                eta_current=torch.tensor([[0.0]], dtype=torch.float64),
            ),
            b_parent,
        )
        torch.testing.assert_close(
            apply_refinancing_policy(
                b_current=b_parent,
                bp_candidate=bp,
                eta_current=torch.tensor([[1.0]], dtype=torch.float64),
            ),
            bp,
        )

        # eta_t = 0 collapses the whole candidate row to b_parent; eta_t = 1
        # realizes each candidate. The child eta axis must not change either.
        candidate_grid = torch.tensor([[[0.2], [0.5], [0.8]]], dtype=torch.float64)
        child_b_eta0 = apply_refinancing_policy(
            b_current=torch.tensor([[[0.4]]], dtype=torch.float64),
            bp_candidate=candidate_grid,
            eta_current=torch.tensor([[[0.0]]], dtype=torch.float64),
        )
        child_b_eta1 = apply_refinancing_policy(
            b_current=torch.tensor([[[0.4]]], dtype=torch.float64),
            bp_candidate=candidate_grid,
            eta_current=torch.tensor([[[1.0]]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            child_b_eta0[0],
            torch.tensor([[0.4], [0.4], [0.4]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            child_b_eta1[0],
            torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64),
        )

        model = PolicyValueModel()
        eta_current = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        b_current = torch.tensor([[0.4], [0.4]], dtype=torch.float64)
        bp_candidate = torch.tensor([[0.2], [0.8]], dtype=torch.float64)
        torch.testing.assert_close(
            model.update_leverage(b_current, bp_candidate, eta_current),
            apply_refinancing_policy(b_current, bp_candidate, eta_current),
        )

    def test_child_eta_does_not_gate_leverage_when_parent_eta_is_zero(self):
        episode = Episode.__new__(Episode)
        child0 = torch.tensor([[0.0, 0.1, 0.0, 0.2, 0.3, -1.0, 4.0, 0.9]], dtype=torch.float64)
        child1 = torch.tensor([[0.0, 0.2, 1.0, 0.4, 0.5, -1.1, 4.1, 1.1]], dtype=torch.float64)
        states, eta_next = episode._build_vectorized_policy_child_states(
            children=[child0, child1],
            bp=torch.tensor([[0.4]], dtype=torch.float64),
            b_parent=torch.tensor([[0.6]], dtype=torch.float64),
            eta_current=torch.tensor([[0.0]], dtype=torch.float64),
        )

        # Both children share b_parent because the CURRENT parent eta_t = 0.
        torch.testing.assert_close(states[0, :, 0], torch.tensor([0.6, 0.6], dtype=torch.float64))
        # The child eta_{t+1} shock is still carried for the expectation.
        torch.testing.assert_close(eta_next[0, :, 0], torch.tensor([0.0, 1.0], dtype=torch.float64))

    def test_vectorized_forward_matches_reference_loop(self):
        torch.manual_seed(456)
        episode = Episode.__new__(Episode)
        children = _make_children(batch_size=6, n_children=2)
        bp = torch.randn(6, 1, dtype=torch.float64)
        b_parent = torch.randn(6, 1, dtype=torch.float64)
        model = nn.Sequential(
            nn.Linear(7, 16),
            nn.Tanh(),
            nn.Linear(16, 9),
        ).to(dtype=torch.float64)
        target_model = copy.deepcopy(model)

        reference_states = _reference_child_states(children, bp, b_parent)
        reference_output = torch.stack(
            [model(reference_states[:, j, :]) for j in range(reference_states.shape[1])],
            dim=1,
        )
        pack = episode._forward_vectorized_policy_children(
            children=children,
            bp=bp,
            b_parent=b_parent,
            eta_current=torch.ones_like(b_parent),
            model=model,
            target_model=target_model,
        )
        vectorized_output = pack["online_output_flat"].reshape(6, 2, 9)

        torch.testing.assert_close(
            vectorized_output,
            reference_output,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_vectorized_forward_preserves_bp_and_model_gradients(self):
        torch.manual_seed(789)
        episode = Episode.__new__(Episode)
        children = _make_children(batch_size=4, n_children=2)
        b_parent = torch.randn(4, 1, dtype=torch.float64)
        bp_reference = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
        bp_vectorized = bp_reference.detach().clone().requires_grad_(True)
        model_reference = nn.Sequential(
            nn.Linear(7, 16),
            nn.Tanh(),
            nn.Linear(16, 9),
        ).to(dtype=torch.float64)
        model_vectorized = copy.deepcopy(model_reference)
        target_model = copy.deepcopy(model_reference)

        reference_states = _reference_child_states(children, bp_reference, b_parent)
        reference_output = torch.stack(
            [
                model_reference(reference_states[:, j, :])
                for j in range(reference_states.shape[1])
            ],
            dim=1,
        )
        reference_loss = reference_output.square().mean()
        reference_loss.backward()
        reference_model_grads = [
            p.grad.detach().clone()
            for p in model_reference.parameters()
        ]

        pack = episode._forward_vectorized_policy_children(
            children=children,
            bp=bp_vectorized,
            b_parent=b_parent,
            eta_current=torch.ones_like(b_parent),
            model=model_vectorized,
            target_model=target_model,
        )
        vectorized_loss = pack["online_output_flat"].square().mean()
        vectorized_loss.backward()
        vectorized_model_grads = [
            p.grad.detach().clone()
            for p in model_vectorized.parameters()
        ]

        torch.testing.assert_close(
            bp_vectorized.grad,
            bp_reference.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        for grad_vec, grad_ref in zip(vectorized_model_grads, reference_model_grads):
            torch.testing.assert_close(
                grad_vec,
                grad_ref,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_target_grid_child_equity_forward_is_vectorized(self):
        children = _make_children(batch_size=4, n_children=3)
        bp_grid = torch.linspace(0.05, 0.95, steps=5, dtype=torch.float64).reshape(1, -1).expand(4, -1)
        b_parent = torch.linspace(0.10, 0.40, steps=4, dtype=torch.float64).reshape(-1, 1)
        eta_current = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float64)
        model = _CountingTargetModel().to(dtype=torch.float64)

        p_child, bar_z_child, child_b_grid = _forward_equity_grid_children(
            model,
            children,
            bp_grid,
            b_parent,
            eta_current,
        )
        self.assertEqual(model.equity_calls, 1)

        reference_states = []
        for child in children:
            raw = child[:, :7]
            states = raw.unsqueeze(1).expand(4, bp_grid.shape[1], 7).clone()
            states[:, :, 0:1] = apply_refinancing_policy(
                b_current=b_parent.unsqueeze(1),
                bp_candidate=bp_grid.unsqueeze(-1),
                eta_current=eta_current.unsqueeze(1),
            )
            reference_states.append(states)
        reference_states = torch.stack(reference_states, dim=2)
        reference_out = model.forward_equity(reference_states.reshape(-1, 7))

        torch.testing.assert_close(
            p_child,
            reference_out["P"].reshape(4, bp_grid.shape[1], len(children)),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            bar_z_child,
            reference_out["bar_z"].reshape(4, bp_grid.shape[1], len(children)),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(child_b_grid, reference_states[..., 0])

    def test_candidate_chunk_accounts_for_children(self):
        teacher = BPGridTeacher(
            target_model=None,
            p0_loss_fn=None,
            pi_loss_fn=None,
            refine=False,
            max_expanded_states=24,
        )

        chunk = teacher._resolve_candidate_chunk_size(
            batch_size=4,
            n_grid=10,
            n_children=3,
        )

        self.assertEqual(chunk, 2)
        self.assertLessEqual(4 * chunk * 3, 24)

    def test_target_grid_vectorized_chunk_matches_reference_child_loop(self):
        torch.manual_seed(321)
        batch_size = 3
        n_children = 4
        parent_state = torch.randn(batch_size, 7, dtype=torch.float64)
        parent_state[:, 0:1] = torch.sigmoid(parent_state[:, 0:1])
        parent_state[:, 2:3] = torch.sigmoid(parent_state[:, 2:3])
        children = _make_children(batch_size=batch_size, n_children=n_children)
        m_list = [
            torch.full((batch_size, 1), 0.90 + 0.03 * j, dtype=torch.float64)
            for j in range(n_children)
        ]
        bp_grid = torch.tensor(
            [
                [0.05, 0.20, 0.60],
                [0.10, 0.40, 0.90],
                [0.00, 0.50, 1.00],
            ],
            dtype=torch.float64,
        )
        mix_weight = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64)

        for branch in ("p0", "pi", "mix"):
            vector_model = _CountingTargetModel().to(dtype=torch.float64)
            reference_model = _CountingTargetModel().to(dtype=torch.float64)
            vector_teacher = BPGridTeacher(
                vector_model,
                P0Loss(),
                PILoss(),
                refine=False,
            )
            reference_teacher = BPGridTeacher(
                reference_model,
                P0Loss(),
                PILoss(),
                refine=False,
            )

            vector = vector_teacher._evaluate_grid_chunk(
                parent_state,
                children,
                m_list,
                bp_grid,
                branch=branch,
                mix_weight=mix_weight if branch == "mix" else None,
            )
            tensor_model = _CountingTargetModel().to(dtype=torch.float64)
            tensor_teacher = BPGridTeacher(
                tensor_model, P0Loss(), PILoss(), refine=False,
            )
            tensor_result = tensor_teacher._evaluate_grid_chunk(
                parent_state,
                torch.stack(children, dim=1),
                torch.stack(m_list, dim=1),
                bp_grid,
                branch=branch,
                mix_weight=mix_weight if branch == "mix" else None,
            )
            reference = _reference_target_grid_chunk(
                reference_teacher,
                parent_state,
                children,
                m_list,
                bp_grid,
                branch=branch,
                mix_weight=mix_weight if branch == "mix" else None,
            )

            for key in (
                "bp_grid",
                "value_grid",
                "cashflow_grid_mean",
                "continuation_grid_mean",
                "q_issue_grid",
                "p_child_grid_mean",
                "default_grid_mean",
            ):
                torch.testing.assert_close(vector[key], reference[key], rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(tensor_result[key], vector[key], rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(vector["argmax_index"], reference["argmax_index"], rtol=0.0, atol=0.0)
            torch.testing.assert_close(tensor_result["argmax_index"], vector["argmax_index"], rtol=0.0, atol=0.0)
            self.assertEqual(vector_model.equity_calls, 1)
            self.assertEqual(tensor_model.equity_calls, 1)
            self.assertEqual(reference_model.equity_calls, n_children)

    def test_dynamic_parent_candidate_chunks_match_one_shot_teacher(self):
        torch.manual_seed(912)
        batch_size = 5
        n_children = 4
        parent_state = torch.randn(batch_size, 7, dtype=torch.float64)
        parent_state[:, 0:1] = torch.sigmoid(parent_state[:, 0:1])
        parent_state[:, 2:3] = 1.0
        children = _make_children(batch_size=batch_size, n_children=n_children)
        m_list = [
            torch.full((batch_size, 1), 0.92 + 0.01 * index, dtype=torch.float64)
            for index in range(n_children)
        ]
        bp_pred = torch.linspace(0.1, 0.9, batch_size, dtype=torch.float64).reshape(-1, 1)
        common = dict(
            p0_loss_fn=P0Loss(),
            pi_loss_fn=PILoss(),
            coarse_size=7,
            fine_size=5,
            refine=True,
            quadratic_refine=False,
            parent_chunk_size=5,
        )
        one_shot = BPGridTeacher(
            target_model=_CountingTargetModel().to(dtype=torch.float64),
            max_expanded_states=10_000,
            **common,
        ).compute(parent_state, children, m_list, branch="p0", bp_pred=bp_pred)
        dynamically_chunked = BPGridTeacher(
            target_model=_CountingTargetModel().to(dtype=torch.float64),
            max_expanded_states=8,
            **common,
        ).compute(parent_state, children, m_list, branch="p0", bp_pred=bp_pred)

        for key in (
            "bp_star",
            "coarse_value_grid",
            "value_grid",
            "regret",
            "top2_margin",
            "q_issue_at_star",
            "p_child_at_star",
            "default_at_star",
        ):
            torch.testing.assert_close(
                dynamically_chunked[key], one_shot[key], rtol=1e-5, atol=1e-6
            )

    def test_multi_j_branch_reuse_matches_legacy_single_j_compute(self):
        torch.manual_seed(2024)
        batch_size = 4
        n_children_max = 16  # expanded children of Jmax=8
        children = _make_children(batch_size=batch_size, n_children=n_children_max)
        m_list = [
            torch.full((batch_size, 1), 0.88 + 0.005 * index, dtype=torch.float64)
            for index in range(n_children_max)
        ]
        base_parent = torch.randn(batch_size, 7, dtype=torch.float64)
        base_parent[:, 0:1] = torch.sigmoid(base_parent[:, 0:1])
        # compute_multi_j_branches shares one candidate grid, so the parent eta_t
        # must be uniform; use the refinancing-active case.
        base_parent[:, 2:3] = 1.0
        branch_states = []
        bp_preds = []
        for i_value in (0.15, 0.35):
            state = base_parent.clone()
            state[:, 3] = i_value
            branch_states.append(state)
            bp_preds.append(
                torch.linspace(0.1, 0.9, batch_size, dtype=torch.float64).reshape(-1, 1)
            )
        prefix_counts = [4, 8, 16]
        weights = torch.linspace(1.0, 2.0, n_children_max, dtype=torch.float64)
        common = dict(
            p0_loss_fn=P0Loss(),
            pi_loss_fn=PILoss(),
            coarse_size=7,
            fine_size=5,
            refine=True,
            quadratic_refine=False,
            parent_chunk_size=0,
            max_expanded_states=24,
        )
        multi_teacher = BPGridTeacher(
            target_model=_CountingTargetModel().to(dtype=torch.float64),
            **common,
        )
        bundles = multi_teacher.compute_multi_j_branches(
            branch_states,
            torch.stack(children, dim=1),
            torch.stack(m_list, dim=1),
            branches=["p0", "pi"],
            prefix_child_counts=prefix_counts,
            child_weights=weights,
            bp_preds=bp_preds,
        )
        stats = multi_teacher.forward_stats()
        self.assertTrue(stats["bp_multi_j_reuse_enabled"])
        self.assertTrue(stats["bp_branch_reuse_enabled"])
        self.assertLessEqual(
            stats["bp_max_actual_expanded_states"], stats["bp_max_expanded_states"]
        )

        keys = (
            "bp_star",
            "bp_star_grid",
            "value_star",
            "regret",
            "top2_margin",
            "confidence",
            "q_issue_at_star",
            "p_child_at_star",
            "default_at_star",
            "coarse_value_grid",
            "coarse_bp_grid",
        )
        legacy_equity_calls = 0
        for index, branch in enumerate(("p0", "pi")):
            for count in prefix_counts:
                legacy_teacher = BPGridTeacher(
                    target_model=_CountingTargetModel().to(dtype=torch.float64),
                    **common,
                )
                legacy = legacy_teacher.compute(
                    branch_states[index],
                    children[:count],
                    m_list[:count],
                    branch=branch,
                    bp_pred=bp_preds[index],
                    child_weights=weights[:count],
                )
                legacy_equity_calls += legacy_teacher.target_model.equity_calls
                optimized = bundles[index][count]
                for key in keys:
                    torch.testing.assert_close(
                        optimized[key], legacy[key], rtol=1e-5, atol=1e-6
                    )
        self.assertLess(
            multi_teacher.target_model.equity_calls, legacy_equity_calls
        )

    def test_bp_eval_budget_override_is_evaluator_only_and_numerically_neutral(self):
        self.assertEqual(BP_EVAL_ALLOWED_EXPANDED_STATES, (65536, 131072, 262144, 524288))
        self.assertEqual(BP_EVAL_SAFE_DEFAULT_EXPANDED_STATES, 65536)
        cpu = torch.device("cpu")
        budget, resolution = resolve_bp_eval_max_expanded_states(None, cpu)
        self.assertEqual((budget, resolution["mode"]), (65536, "cpu_safe_default"))
        budget, resolution = resolve_bp_eval_max_expanded_states(262144, cpu)
        self.assertEqual((budget, resolution["mode"]), (262144, "explicit"))
        with self.assertRaises(ValueError):
            resolve_bp_eval_max_expanded_states(1234, cpu)

        hyperparams = HyperParams()
        hyperparams.bp_grid_max_expanded_states = 65536
        small = BPGridTeacher.from_hyperparams(
            _CountingTargetModel().to(dtype=torch.float64),
            P0Loss(), PILoss(), hyperparams,
            max_expanded_states_override=8,
        )
        large = BPGridTeacher.from_hyperparams(
            _CountingTargetModel().to(dtype=torch.float64),
            P0Loss(), PILoss(), hyperparams,
            max_expanded_states_override=65536,
        )
        self.assertEqual(small.max_expanded_states, 8)
        self.assertEqual(large.max_expanded_states, 65536)
        # The override is evaluator-only: checkpoint hyperparams stay untouched.
        self.assertEqual(int(hyperparams.bp_grid_max_expanded_states), 65536)

        torch.manual_seed(11)
        batch_size = 4
        children = _make_children(batch_size=batch_size, n_children=8)
        m_list = [torch.full((batch_size, 1), 0.9, dtype=torch.float64) for _ in range(8)]
        parent = torch.randn(batch_size, 7, dtype=torch.float64)
        parent[:, 0:1] = torch.sigmoid(parent[:, 0:1])
        parent[:, 2:3] = 1.0
        bp_pred = torch.linspace(0.1, 0.9, batch_size, dtype=torch.float64).reshape(-1, 1)
        budgeted = small.compute(parent, children, m_list, branch="p0", bp_pred=bp_pred)
        unbudgeted = large.compute(parent, children, m_list, branch="p0", bp_pred=bp_pred)
        self.assertLess(
            small.forward_stats()["bp_max_actual_expanded_states"],
            large.forward_stats()["bp_max_actual_expanded_states"],
        )
        for key in ("bp_star", "bp_star_grid", "value_star", "regret", "coarse_value_grid"):
            torch.testing.assert_close(budgeted[key], unbudgeted[key], rtol=1e-6, atol=1e-8)

    def test_parent_eta_zero_runs_no_bp_grid_and_forces_child_leverage(self):
        batch_size = 2
        parent_state = torch.tensor(
            [
                [0.60, 0.10, 0.0, 0.20, 0.30, -1.0, 4.0],
                [0.40, -0.20, 0.0, 0.10, -0.10, -1.2, 4.2],
            ],
            dtype=torch.float64,
        )
        child0 = parent_state.clone()
        child1 = parent_state.clone()
        child0[:, 2:3] = 0.0
        child1[:, 2:3] = 1.0
        children = [
            torch.cat([child0, torch.ones(batch_size, 1, dtype=torch.float64)], dim=1),
            torch.cat([child1, torch.ones(batch_size, 1, dtype=torch.float64)], dim=1),
        ]
        m_list = [torch.ones(batch_size, 1, dtype=torch.float64) for _ in children]
        teacher = BPGridTeacher(
            _CountingTargetModel().to(dtype=torch.float64),
            P0Loss(),
            PILoss(),
            coarse_size=5,
            refine=False,
            confidence_min=0.0,
        )

        grid = teacher.compute(
            parent_state=parent_state,
            children=children,
            m_list=m_list,
            branch="p0",
            bp_pred=torch.full((batch_size, 1), 0.9, dtype=torch.float64),
        )

        # eta_t = 0 has no bp choice: no candidate comparison exists, so the
        # grid-shaped diagnostics are NaN/sentinel and refi_active/confidence are 0.
        torch.testing.assert_close(grid["refi_active"], torch.zeros_like(grid["refi_active"]))
        torch.testing.assert_close(grid["confidence"], torch.zeros_like(grid["confidence"]))
        self.assertTrue(bool(torch.isnan(grid["coarse_value_grid"]).all()))
        self.assertTrue(bool(torch.isnan(grid["continuation_grid_mean"]).all()))
        # child eta_{t+1} no longer gates leverage, so both conditional child
        # leverage means collapse to the forced b_parent on every candidate slot.
        torch.testing.assert_close(
            grid["child_b_eta0_mean"],
            parent_state[:, 0:1].expand_as(grid["child_b_eta0_mean"]),
        )
        torch.testing.assert_close(
            grid["child_b_eta1_mean"],
            parent_state[:, 0:1].expand_as(grid["child_b_eta1_mean"]),
        )
        torch.testing.assert_close(grid["bp_star"], parent_state[:, 0:1])
        # The forced Bellman value target is still a valid target.
        self.assertTrue(bool(torch.isfinite(grid["value_star"]).all()))
        torch.testing.assert_close(
            grid["eta_next_active_share"],
            torch.full((batch_size, 1), 0.5, dtype=torch.float64),
        )

    def test_exact_candidate_child_leverage_matrix_and_gradient(self):
        b_parent = torch.tensor([[0.4]], dtype=torch.float64)
        bp_grid = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float64)
        child0 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        child1 = child0.clone()
        child1[:, 2] = 1.0

        from training.bp_grid_teacher import _expand_grid_children

        # Parent eta_t = 1: every candidate is realized for BOTH children, even
        # though the two children carry different eta_{t+1} coordinates.
        eta_current = torch.tensor([[1.0]], dtype=torch.float64)
        child_states, child_b = _expand_grid_children(
            [child0, child1], bp_grid, b_parent, eta_current
        )
        torch.testing.assert_close(
            child_b[0],
            torch.tensor([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            child_states[0, :, :, 0],
            torch.tensor([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], dtype=torch.float64),
        )
        # Child eta_{t+1} is preserved untouched in the child state.
        torch.testing.assert_close(
            child_states[0, :, :, 2],
            torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
        )

        bp = torch.tensor([[0.3], [0.3]], dtype=torch.float64, requires_grad=True)
        b_parent_two = torch.tensor([[0.4], [0.4]], dtype=torch.float64)
        eta_two = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        b_child = apply_refinancing_policy(
            b_current=b_parent_two,
            bp_candidate=bp,
            eta_current=eta_two,
        )
        gradient = torch.autograd.grad(b_child.sum(), bp, retain_graph=True)[0]
        torch.testing.assert_close(gradient, eta_two)
        branch_grad = torch.autograd.grad(
            b_child,
            bp,
            grad_outputs=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            retain_graph=True,
        )[0]
        torch.testing.assert_close(branch_grad, torch.zeros_like(bp))

    def test_child_eta_zero_does_not_flatten_continuation_when_parent_refinances(self):
        parent = torch.tensor([[0.4, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        child0 = parent.clone()
        child1 = parent.clone()
        child0[:, 2] = 0.0
        child1[:, 2] = 0.0
        teacher = BPGridTeacher(
            _CountingTargetModel().to(dtype=torch.float64),
            P0Loss(),
            PILoss(),
            coarse_size=3,
            refine=False,
        )
        grid = teacher.compute(
            parent, [child0, child1], [torch.ones(1, 1), torch.ones(1, 1)], branch="p0"
        )
        # eta_t = 1: leverage follows the candidate grid for both children, so the
        # continuation varies along the candidate axis even though every child
        # draws eta_{t+1} = 0.
        assert not torch.allclose(
            grid["continuation_grid_mean"],
            grid["continuation_grid_mean"][:, :1].expand_as(grid["continuation_grid_mean"]),
        )
        assert grid["eta_next_active_share"].item() == 0.0

    def test_foc_autograd_chain_uses_current_eta_once(self):
        bp = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
        b_parent = torch.tensor([[0.4]], dtype=torch.float64)

        def _residuals(eta_current):
            p_children = [
                apply_refinancing_policy(b_parent, bp, eta_current).square()
                for _ in range(2)
            ]
            return P0Loss().compute_foc_residual_from_bp(
                CF0p=bp * 0.0,
                M_list=[torch.ones_like(bp), torch.ones_like(bp)],
                P_children=p_children,
                bar_z_children=[torch.zeros_like(bp), torch.zeros_like(bp)],
                bp=bp,
            )

        # eta_t = 1: db_child/dbp = eta_current = 1 enters the chain exactly once,
        # so the residual is 2*bp rather than 2*eta*bp or 2*eta^2*bp.
        active = _residuals(torch.tensor([[1.0]], dtype=torch.float64))
        torch.testing.assert_close(active[0], 2.0 * bp)
        torch.testing.assert_close(active[1], 2.0 * bp)

        # eta_t = 0: bp is not a control, so the debt-choice FOC residual vanishes.
        inactive = _residuals(torch.tensor([[0.0]], dtype=torch.float64))
        torch.testing.assert_close(inactive[0], torch.zeros_like(bp))
        torch.testing.assert_close(inactive[1], torch.zeros_like(bp))


if __name__ == "__main__":
    unittest.main()
