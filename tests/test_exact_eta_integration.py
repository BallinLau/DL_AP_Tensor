import copy
from types import SimpleNamespace

import pytest
import torch

from config import Config, HyperParams
from losses import P0Loss, PILoss, QLoss
from losses.utils import compute_aio_residual as real_compute_aio_residual
from models import PolicyValueModel
import training.episode as episode_module
from training.bp_grid_teacher import BPGridTeacher
from training.episode import Episode
from utils.firm_transition import (
    apply_refinancing_policy,
    exact_eta_pair_expectation,
    expand_children_exact_eta,
)


def _episode(*, zeta: float = 0.03, enabled: bool = True) -> Episode:
    episode = Episode.__new__(Episode)
    episode.config = SimpleNamespace(ZETA=zeta)
    episode.hyperparams = SimpleNamespace(
        pv_exact_eta_integration_enabled=enabled,
    )
    return episode


def _continuous_children(batch_size: int = 2, branch_count: int = 2):
    children = []
    for branch in range(branch_count):
        child = torch.zeros(batch_size, 8, dtype=torch.float64)
        child[:, 0] = torch.tensor([0.2, 0.7], dtype=torch.float64)[:batch_size]
        child[:, 1] = -0.2 + branch
        child[:, 2] = float(branch % 2)
        child[:, 3] = 0.1 + branch
        child[:, 4] = -2.0 + 0.1 * branch
        child[:, 5] = -2.2 + 0.1 * branch
        child[:, 6] = 4.0 + 0.1 * branch
        child[:, 7] = 0.9 + 0.1 * branch
        children.append(child)
    return children


def _training_episode_and_batch():
    model = PolicyValueModel(share_hidden_dims=[8], share_output_dim=8)
    target = copy.deepcopy(model)
    hp = HyperParams()
    hp.pv_bp_training_mode = "target_grid"
    hp.policy_value_bellman_only = True
    hp.pv_exact_eta_integration_enabled = True
    hp.pv_use_clipped_m = False
    episode = Episode(
        models={"policy_value": model},
        optimizers={"policy_value": torch.optim.AdamW(model.parameters(), lr=1e-3)},
        config=Config,
        hyperparams=hp,
        device=torch.device("cpu"),
        firm_target=target,
    )
    parent = torch.tensor(
        [
            [0.2, -0.1, 1.0, 0.2, -2.0, -2.2, 4.0, 1.0],
            [0.5, 0.3, 0.0, 0.4, -1.8, -2.0, 4.2, 1.0],
        ]
    )
    children = [child.float() for child in _continuous_children()]
    return episode, {"parent": parent, "children": children}


def test_exact_eta_expectation_matches_large_monte_carlo():
    zeta = 0.03
    eta0 = torch.tensor([1.5, -2.0, 4.0], dtype=torch.float64)
    eta1 = torch.tensor([7.0, 3.0, -1.0], dtype=torch.float64)
    exact = (1.0 - zeta) * eta0 + zeta * eta1

    generator = torch.Generator().manual_seed(12345)
    draws = torch.rand(500_000, generator=generator, dtype=torch.float64) < zeta
    simulated = torch.where(draws.unsqueeze(1), eta1, eta0).mean(dim=0)
    torch.testing.assert_close(simulated, exact, atol=1e-2, rtol=0.0)


def test_bp_teacher_exact_eta_matches_large_sampled_eta_benchmark():
    class LinearValueTarget:
        def _q_output(self, state):
            return torch.zeros_like(state[:, 0:1])

        def forward_equity(self, state):
            p = 1.0 + 2.0 * state[:, 0:1]
            return {"P": p, "Phat": p, "bar_z": torch.zeros_like(p)}

        def __call__(self, state):
            equity = self.forward_equity(state)
            return SimpleNamespace(Q=self._q_output(state), **equity)

    zeta = 0.03
    target = LinearValueTarget()
    teacher = BPGridTeacher(
        target, P0Loss(), PILoss(), coarse_size=3, refine=False,
        candidate_chunk_size=3, max_expanded_states=100_000,
    )
    parent = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    child = parent.clone()
    exact_children = [child.clone(), child.clone()]
    exact_children[0][:, 2] = 0.0
    exact_children[1][:, 2] = 1.0
    exact = teacher.compute(
        parent,
        exact_children,
        [torch.ones(1, 1), torch.ones(1, 1)],
        branch="p0",
        child_weights=torch.tensor([[1.0 - zeta, zeta]]),
    )

    generator = torch.Generator().manual_seed(24680)
    sampled_eta = (torch.rand(20_000, generator=generator) < zeta).float()
    sampled_children = []
    sampled_m = []
    for value in sampled_eta:
        sampled = child.clone()
        sampled[:, 2] = value
        sampled_children.append(sampled)
        sampled_m.append(torch.ones(1, 1))
    sampled = teacher.compute(parent, sampled_children, sampled_m, branch="p0")

    torch.testing.assert_close(
        sampled["coarse_continuation_grid_mean"],
        exact["coarse_continuation_grid_mean"],
        atol=1e-2,
        rtol=0.0,
    )


def test_training_expansion_duplicates_m_and_collapse_restores_independent_branch_count():
    episode = _episode()
    children = _continuous_children()
    raw_m = [child[:, 7:8] for child in children]
    used_m = [m.clamp(0.95, 1.05) for m in raw_m]

    expanded, raw_expanded, used_expanded, weights = (
        episode._expand_policy_expectation_children(children, raw_m, used_m)
    )
    assert len(expanded) == 4
    assert weights.shape == (2, 4)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(raw_expanded[0], raw_expanded[1])
    torch.testing.assert_close(raw_expanded[2], raw_expanded[3])
    torch.testing.assert_close(used_expanded[0], used_expanded[1])
    torch.testing.assert_close(used_expanded[2], used_expanded[3])

    eta_residuals = [torch.full((2, 1), float(index), dtype=torch.float64) for index in range(4)]
    independent_residuals = episode._collapse_policy_eta_pairs(eta_residuals)
    assert len(independent_residuals) == 2
    torch.testing.assert_close(independent_residuals[0], torch.full((2, 1), 0.03, dtype=torch.float64))
    torch.testing.assert_close(independent_residuals[1], torch.full((2, 1), 2.03, dtype=torch.float64))


@pytest.mark.parametrize("loss_fn,growth", [(P0Loss(), 1.0), (PILoss(), 1.02)])
def test_exact_eta_foc_autograd_has_single_zeta_multiplier(loss_fn, growth):
    zeta = 0.03
    episode = _episode(zeta=zeta)
    b_parent = torch.tensor([[0.4]], dtype=torch.float64)
    bp = torch.tensor([[0.8]], dtype=torch.float64, requires_grad=True)
    children = _continuous_children(batch_size=1, branch_count=2)
    raw_m = [torch.tensor([[0.9]], dtype=torch.float64), torch.tensor([[1.1]], dtype=torch.float64)]
    expanded, _raw_expanded, m_expanded, _weights = (
        episode._expand_policy_expectation_children(children, raw_m, raw_m)
    )

    slope = 2.5
    p_children = []
    bar_z_children = []
    cashflows = []
    for child in expanded:
        b_child = apply_refinancing_policy(b_parent, bp, child[:, 2:3])
        p_children.append(slope * b_child + 1.0)
        bar_z_children.append(torch.zeros_like(b_child))
        cashflows.append(bp * 0.0)
    if isinstance(loss_fn, P0Loss):
        expanded_foc = loss_fn.compute_foc_residual_from_bp(
            CF0p=cashflows,
            M_list=m_expanded,
            P_children=p_children,
            bar_z_children=bar_z_children,
            bp=bp,
        )
    else:
        loss_fn.g = growth
        expanded_foc = loss_fn.compute_foc_residual_from_bp(
            CFip=cashflows,
            M_list=m_expanded,
            P_children=p_children,
            bar_z_children=bar_z_children,
            bp=bp,
        )
    foc = episode._collapse_policy_eta_pairs(expanded_foc)

    assert len(foc) == 2
    for actual, m in zip(foc, raw_m):
        torch.testing.assert_close(actual, zeta * growth * m * slope)


def test_exact_eta_bellman_residual_is_collapsed_before_aio():
    zeta = 0.03
    episode = _episode(zeta=zeta)
    current = torch.tensor([[10.0]])
    cashflows = [torch.tensor([[1.0]]) for _ in range(4)]
    m_expanded = [torch.tensor([[2.0]]), torch.tensor([[2.0]]), torch.tensor([[3.0]]), torch.tensor([[3.0]])]
    p_expanded = [torch.tensor([[4.0]]), torch.tensor([[8.0]]), torch.tensor([[6.0]]), torch.tensor([[10.0]])]
    bars = [torch.zeros(1, 1) for _ in range(4)]

    expanded = P0Loss().compute_bellman_residual(
        current, cashflows, m_expanded, p_expanded, bars
    )
    residuals = episode._collapse_policy_eta_pairs(expanded)

    assert len(residuals) == 2
    expected0 = current - 1.0 - 2.0 * ((1.0 - zeta) * 4.0 + zeta * 8.0)
    expected1 = current - 1.0 - 3.0 * ((1.0 - zeta) * 6.0 + zeta * 10.0)
    torch.testing.assert_close(residuals[0], expected0)
    torch.testing.assert_close(residuals[1], expected1)


@pytest.mark.parametrize("equation", ["p0", "q"])
def test_training_aio_receives_only_independent_continuous_branches(monkeypatch, equation):
    episode, batch = _training_episode_and_batch()
    branch_counts = []

    def capture(residuals, aio_weight):
        branch_counts.append(len(residuals))
        return real_compute_aio_residual(residuals, aio_weight)

    monkeypatch.setattr(episode_module, "compute_aio_residual", capture)
    if equation == "p0":
        loss = episode._compute_p0_loss(batch)
    else:
        loss = episode._compute_q_loss(batch)
    assert torch.isfinite(loss)
    assert branch_counts == [2]


def test_exact_eta_q_residual_is_collapsed_before_aio():
    zeta = 0.03
    episode = _episode(zeta=zeta)
    loss_fn = QLoss(g=1.0, delta=0.1, phi=0.0)
    q = torch.tensor([[2.0]])
    b = torch.tensor([[0.5]])
    bar_i = torch.zeros_like(q)
    m = [torch.ones_like(q) for _ in range(4)]
    qsp = [torch.tensor([[1.0]]), torch.tensor([[3.0]]), torch.tensor([[2.0]]), torch.tensor([[6.0]])]
    bars = [torch.zeros_like(q) for _ in range(4)]
    xs = [torch.zeros_like(q) for _ in range(4)]
    zs = [torch.zeros_like(q) for _ in range(4)]

    expanded = loss_fn.compute_main_residual(q, b, bar_i, m, qsp, bars, xs, zs)
    residuals = episode._collapse_policy_eta_pairs(expanded)

    assert len(residuals) == 2
    expected0 = (1.0 - zeta) * expanded[0] + zeta * expanded[1]
    expected1 = (1.0 - zeta) * expanded[2] + zeta * expanded[3]
    torch.testing.assert_close(residuals[0], expected0)
    torch.testing.assert_close(residuals[1], expected1)


def test_legacy_sampled_eta_ablation_remains_available():
    episode = _episode(enabled=False)
    children = _continuous_children()
    m_list = [child[:, 7:8] for child in children]
    actual_children, raw_m, used_m, weights = episode._expand_policy_expectation_children(
        children, m_list, m_list
    )
    assert actual_children is children
    assert raw_m is m_list
    assert used_m is m_list
    assert weights is None


@pytest.mark.parametrize("zeta", [0.0, 1.0])
def test_exact_eta_pair_expectation_edge_cases(zeta):
    values = torch.tensor([[[1.0], [9.0], [2.0], [8.0]]])
    collapsed = exact_eta_pair_expectation(values, zeta=zeta)
    expected = torch.tensor([[[1.0], [2.0]]]) if zeta == 0.0 else torch.tensor([[[9.0], [8.0]]])
    torch.testing.assert_close(collapsed, expected)
