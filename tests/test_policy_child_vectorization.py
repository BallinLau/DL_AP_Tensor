import copy
import unittest
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models import PolicyValueModel
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


if __name__ == "__main__":
    unittest.main()
