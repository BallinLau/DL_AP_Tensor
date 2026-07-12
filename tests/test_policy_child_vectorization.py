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
from config import Config
from training.bp_grid_teacher import BPGridTeacher, _forward_equity_grid_children
from training.episode import Episode


def _make_children(batch_size: int = 5, n_children: int = 3):
    torch.manual_seed(123)
    children = []
    for j in range(n_children):
        child = torch.randn(batch_size, 8, dtype=torch.float64)
        child[:, 2:3] = torch.sigmoid(child[:, 2:3])
        child[:, 7:8] = 0.8 + 0.1 * j
        children.append(child)
    return children


def _reference_child_states(children, bp, b_parent):
    reference_states = []
    for child in children:
        raw = child[:, :7] if child.shape[1] > 7 else child
        eta = child[:, 2:3]
        state = torch.cat(
            [
                eta * bp + (1.0 - eta) * b_parent,
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
    issue_state = parent_state.unsqueeze(1).expand(batch_size, n_grid, parent_state.shape[-1]).reshape(batch_size * n_grid, -1).clone()
    issue_state[:, 0:1] = bp_grid.reshape(-1, 1)
    q_issue = teacher.target_model._q_output(issue_state).reshape(batch_size, n_grid)

    value_grid = torch.zeros(batch_size, n_grid, dtype=parent_state.dtype, device=parent_state.device)
    p_sum = torch.zeros_like(value_grid)
    default_sum = torch.zeros_like(value_grid)
    b_parent = parent_state[:, 0:1]
    x_parent = parent_state[:, 4:5]
    z_parent = parent_state[:, 1:2]
    i_parent = parent_state[:, 3:4]
    q_current_grid = q_current.expand(batch_size, n_grid)
    mix_w = None if mix_weight is None else mix_weight.clamp(0.0, 1.0).expand(batch_size, n_grid)

    for child, m in zip(children, m_list):
        child_raw = child[:, :7] if child.shape[1] > 7 else child
        eta_grid = child[:, 2:3].clamp(0.0, 1.0).expand(batch_size, n_grid)
        child_state = child_raw.unsqueeze(1).expand(batch_size, n_grid, child_raw.shape[-1]).reshape(batch_size * n_grid, -1).clone()
        child_state[:, 0:1] = (
            eta_grid * bp_grid
            + (1.0 - eta_grid) * b_parent.expand(batch_size, n_grid)
        ).reshape(-1, 1)
        out = teacher.target_model.forward_equity(child_state)
        p_child = out["P"].reshape(batch_size, n_grid)
        bar_z_child = out["bar_z"].reshape(batch_size, n_grid).clamp(0.0, 1.0)

        cf0 = teacher.p0_loss_fn.compute_cashflow_p0(
            x_parent.expand(batch_size, n_grid).reshape(-1, 1),
            z_parent.expand(batch_size, n_grid).reshape(-1, 1),
            b_parent.expand(batch_size, n_grid).reshape(-1, 1),
            q_current_grid.reshape(-1, 1),
            q_issue.reshape(-1, 1),
            eta_grid.reshape(-1, 1),
        ).reshape(batch_size, n_grid)
        value0 = cf0 + m.expand(batch_size, n_grid) * p_child

        cfi = teacher.pi_loss_fn.compute_cashflow_pi(
            x_parent.expand(batch_size, n_grid).reshape(-1, 1),
            z_parent.expand(batch_size, n_grid).reshape(-1, 1),
            b_parent.expand(batch_size, n_grid).reshape(-1, 1),
            i_parent.expand(batch_size, n_grid).reshape(-1, 1),
            q_current_grid.reshape(-1, 1),
            q_issue.reshape(-1, 1),
            eta_grid.reshape(-1, 1),
        ).reshape(batch_size, n_grid)
        valuei = cfi + Config.G * m.expand(batch_size, n_grid) * p_child

        if branch == "p0":
            branch_value = value0
        elif branch == "pi":
            branch_value = valuei
        else:
            branch_value = (1.0 - mix_w) * value0 + mix_w * valuei

        value_grid = value_grid + branch_value
        p_sum = p_sum + p_child
        default_sum = default_sum + bar_z_child

    n_children = len(children)
    return {
        "bp_grid": bp_grid,
        "value_grid": value_grid / n_children,
        "q_issue_grid": q_issue,
        "p_child_grid_mean": p_sum / n_children,
        "default_grid_mean": default_sum / n_children,
        "argmax_index": (value_grid / n_children).argmax(dim=1, keepdim=True),
    }


class PolicyChildVectorizationTest(unittest.TestCase):
    def test_policy_value_model_default_has_no_batchnorm_or_dropout(self):
        model = PolicyValueModel()
        self.assertFalse(any(isinstance(m, nn.BatchNorm1d) for m in model.modules()))
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in model.modules()))

    def test_vectorized_child_state_matches_reference_loop(self):
        episode = Episode.__new__(Episode)
        children = _make_children()
        bp = torch.linspace(0.1, 0.5, steps=5, dtype=torch.float64).reshape(-1, 1)
        b_parent = torch.linspace(0.2, 0.6, steps=5, dtype=torch.float64).reshape(-1, 1)

        vectorized_states, eta_children = episode._build_vectorized_policy_child_states(
            children=children,
            bp=bp,
            b_parent=b_parent,
        )
        reference_states = _reference_child_states(children, bp, b_parent)

        torch.testing.assert_close(vectorized_states, reference_states, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            eta_children,
            torch.stack([child[:, 2:3] for child in children], dim=1),
            rtol=0.0,
            atol=0.0,
        )

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
        model = _CountingTargetModel().to(dtype=torch.float64)

        p_child, bar_z_child, eta_grid = _forward_equity_grid_children(
            model,
            children,
            bp_grid,
            b_parent,
        )
        self.assertEqual(model.equity_calls, 1)

        reference_states = []
        for child in children:
            raw = child[:, :7]
            states = raw.unsqueeze(1).expand(4, bp_grid.shape[1], 7).clone()
            eta = child[:, 2:3].expand(4, bp_grid.shape[1])
            states[:, :, 0:1] = (
                eta * bp_grid
                + (1.0 - eta) * b_parent.expand(4, bp_grid.shape[1])
            ).unsqueeze(-1)
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
        torch.testing.assert_close(
            eta_grid,
            torch.stack([child[:, 2:3].expand(4, bp_grid.shape[1]) for child in children], dim=2),
            rtol=0.0,
            atol=0.0,
        )

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
            reference = _reference_target_grid_chunk(
                reference_teacher,
                parent_state,
                children,
                m_list,
                bp_grid,
                branch=branch,
                mix_weight=mix_weight if branch == "mix" else None,
            )

            for key in ("bp_grid", "value_grid", "q_issue_grid", "p_child_grid_mean", "default_grid_mean"):
                torch.testing.assert_close(vector[key], reference[key], rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(vector["argmax_index"], reference["argmax_index"], rtol=0.0, atol=0.0)
            self.assertEqual(vector_model.equity_calls, 1)
            self.assertEqual(reference_model.equity_calls, n_children)


if __name__ == "__main__":
    unittest.main()
