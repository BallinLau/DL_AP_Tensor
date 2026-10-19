from __future__ import annotations

import math

import pytest
import torch

from data.boundary_aware_distribution import (
    PostDecisionTransportAtoms,
    PreDecisionDistribution,
    gaussian_mixture_moments,
    gauss_legendre_probability_nodes,
    normal_cdf,
    normal_quantile,
    propagate_ar1_to_predecision,
    propagate_ar1_with_eta_mixture,
)


DTYPE = torch.float64


def _toy_transport_atoms(points: int = 192) -> PostDecisionTransportAtoms:
    """Analytic fixture from the experiment contract, represented exactly.

    z1 is integrated on the survival probability interval (0.5,1).  Each z
    quadrature atom is then split into the independent investment and
    non-investment regions, each of probability 0.5 under U(0,0.5).
    """
    lower = torch.tensor([0.5], dtype=DTYPE)
    upper = torch.tensor([1.0], dtype=DTYPE)
    u, probability_weight = gauss_legendre_probability_nodes(lower, upper, points)
    z1 = normal_quantile(u, torch.tensor([0.0], dtype=DTYPE), torch.tensor([0.1], dtype=DTYPE))
    z1 = z1.reshape(-1)
    probability_weight = probability_weight.reshape(-1)

    source_z = torch.repeat_interleave(z1, 2)[None, :]
    base_weight = torch.repeat_interleave(probability_weight * 100.0, 2)
    # i <= 0.25 under U(0,0.5) and its complement each have mass one half.
    weight = (0.5 * base_weight)[None, :]
    investment = torch.arange(2 * points) % 2 == 0
    capital = torch.where(
        investment, torch.full((2 * points,), 1.14, dtype=DTYPE), torch.ones(2 * points, dtype=DTYPE)
    )[None, :]
    b_next = torch.where(
        investment,
        torch.full((2 * points,), 0.4 / 1.14, dtype=DTYPE),
        torch.full((2 * points,), 0.4, dtype=DTYPE),
    )[None, :]
    shape = weight.shape
    zeros_long = torch.zeros(shape, dtype=torch.long)
    return PostDecisionTransportAtoms(
        transport_weight=weight,
        source_z_current=source_z,
        b_next=b_next,
        capital_next_per_unit=capital,
        atom_mask=torch.ones(shape, dtype=torch.bool),
        source_component_id=zeros_long,
        survival_region_id=torch.zeros(shape, dtype=torch.long),
        investment_region_id=investment.to(torch.long)[None, :],
        policy_branch_id=investment.to(torch.long)[None, :],
        economic_node_id=zeros_long,
        source_time=torch.ones(shape, dtype=torch.long),
        destination_time=torch.full(shape, 2, dtype=torch.long),
        cohort=zeros_long,
        birth_time=zeros_long,
        maintenance_eligibility=torch.ones(shape, dtype=torch.bool),
        origin_kind=zeros_long,
        origin_source_id=zeros_long,
    ).validate()


def test_two_period_selection_transport_matches_analytic_fixture():
    atoms = _toy_transport_atoms()
    investment = atoms.investment_region_id == 1
    noninvestment = ~investment
    assert atoms.total_mass().item() == pytest.approx(50.0, abs=1e-10)
    assert atoms.transport_weight[investment].sum().item() == pytest.approx(25.0, abs=1e-10)
    assert atoms.transport_weight[noninvestment].sum().item() == pytest.approx(25.0, abs=1e-10)
    assert atoms.total_capital().item() == pytest.approx(53.5, abs=1e-10)
    assert torch.unique(atoms.b_next[investment]).item() == pytest.approx(0.4 / 1.14)
    assert torch.unique(atoms.b_next[noninvestment]).item() == pytest.approx(0.4)

    t2 = propagate_ar1_to_predecision(
        atoms, rho_z=0.8, sigma_z=0.1, z_bar=0.0
    )
    moments = gaussian_mixture_moments(t2)
    expected_mean = 0.8 * 0.1 * math.sqrt(2.0 / math.pi)
    expected_variance = 0.8 ** 2 * 0.1 ** 2 * (1.0 - 2.0 / math.pi) + 0.1 ** 2
    assert moments["mass"].item() == pytest.approx(50.0, abs=1e-10)
    assert moments["mean"].item() == pytest.approx(expected_mean, abs=2e-6)
    assert moments["variance"].item() == pytest.approx(expected_variance, abs=2e-6)

    # Propagating only the selected mean drops Var(E[z2|z1]); this assertion is
    # the intentional guard against the old representative-point shortcut.
    mean_only_variance = 0.1 ** 2
    assert moments["variance"].item() - mean_only_variance > 0.002


def test_t2_branches_integrate_their_own_full_conditional_densities():
    atoms = _toy_transport_atoms()
    t2 = propagate_ar1_to_predecision(
        atoms, rho_z=0.8, sigma_z=0.1, z_bar=0.0
    )
    investment = atoms.investment_region_id == 1
    # Different default thresholds for the two selected capital/leverage
    # branches.  Each component uses its own conditional N(mu_r, sigma^2).
    threshold = torch.where(
        investment,
        torch.full_like(t2.z_conditional_mean, -0.01),
        torch.full_like(t2.z_conditional_mean, 0.03),
    )
    default_probability = normal_cdf(
        threshold, t2.z_conditional_mean, t2.z_conditional_std
    )
    default_mass_invest = (
        t2.component_mass * default_probability * investment.to(DTYPE)
    ).sum()
    default_mass_noinvest = (
        t2.component_mass * default_probability * (~investment).to(DTYPE)
    ).sum()
    assert default_mass_invest.item() > 0.0
    assert default_mass_noinvest.item() > 0.0
    assert abs(default_mass_invest.item() - default_mass_noinvest.item()) > 1.0

    # A mean-only whole-component decision would produce only 0 or 25 in each
    # branch; the boundary integral must remain fractional.
    assert 0.0 < default_mass_invest.item() < 25.0
    assert 0.0 < default_mass_noinvest.item() < 25.0


def test_z_dependent_investment_selection_survives_into_t2():
    points = 192
    u, probability_weight = gauss_legendre_probability_nodes(
        torch.tensor([0.5], dtype=DTYPE),
        torch.tensor([1.0], dtype=DTYPE),
        points,
    )
    z1 = normal_quantile(
        u, torch.tensor([0.0], dtype=DTYPE), torch.tensor([0.1], dtype=DTYPE)
    ).reshape(-1)
    probability_weight = probability_weight.reshape(-1)
    # The original cost density remains U(0,0.5), but the value boundary is
    # state dependent.  p_I(z) is therefore retained jointly with source z.
    i_star = (0.20 + z1).clamp(0.0, 0.5)
    p_invest = i_star / 0.5
    weight_i = 100.0 * probability_weight * p_invest
    weight_n = 100.0 * probability_weight * (1.0 - p_invest)
    mean_z_i = (weight_i * z1).sum() / weight_i.sum()
    mean_z_n = (weight_n * z1).sum() / weight_n.sum()
    assert mean_z_i.item() > mean_z_n.item()

    source_z = torch.cat((z1, z1))[None, :]
    weight = torch.cat((weight_i, weight_n))[None, :]
    invest = torch.cat(
        (torch.ones(points, dtype=torch.bool), torch.zeros(points, dtype=torch.bool))
    )[None, :]
    shape = weight.shape
    zeros = torch.zeros(shape, dtype=torch.long)
    atoms = PostDecisionTransportAtoms(
        transport_weight=weight,
        source_z_current=source_z,
        b_next=torch.where(invest, torch.full_like(weight, 0.4 / 1.14), torch.full_like(weight, 0.4)),
        capital_next_per_unit=torch.where(invest, torch.full_like(weight, 1.14), torch.ones_like(weight)),
        atom_mask=weight > 0.0,
        source_component_id=zeros,
        survival_region_id=zeros,
        investment_region_id=invest.to(torch.long),
        policy_branch_id=invest.to(torch.long),
        economic_node_id=zeros,
        source_time=torch.ones(shape, dtype=torch.long),
        destination_time=torch.full(shape, 2, dtype=torch.long),
        cohort=zeros,
        birth_time=zeros,
        maintenance_eligibility=torch.ones(shape, dtype=torch.bool),
        origin_kind=zeros,
        origin_source_id=zeros,
    ).validate()
    t2 = propagate_ar1_to_predecision(atoms, rho_z=0.8, sigma_z=0.1, z_bar=0.0)
    t2_mean_i = (
        t2.component_mass[invest] * t2.z_conditional_mean[invest]
    ).sum() / t2.component_mass[invest].sum()
    t2_mean_n = (
        t2.component_mass[~invest] * t2.z_conditional_mean[~invest]
    ).sum() / t2.component_mass[~invest].sum()
    assert t2_mean_i.item() > t2_mean_n.item()


def test_distribution_serialization_round_trip_is_exact(tmp_path):
    atoms = _toy_transport_atoms(points=16)
    distribution = propagate_ar1_to_predecision(
        atoms, rho_z=0.8, sigma_z=0.1, z_bar=0.0
    )
    path = tmp_path / "distribution.pt"
    torch.save(distribution.state_dict(), path)
    restored = PreDecisionDistribution.from_state_dict(torch.load(path, weights_only=True))
    assert restored.fingerprint() == distribution.fingerprint()


def test_next_eta_is_enumerated_after_b_transition_without_changing_mass():
    atoms = _toy_transport_atoms(points=16)
    distribution = propagate_ar1_with_eta_mixture(
        atoms, rho_z=0.8, sigma_z=0.1, z_bar=0.0, zeta=0.2
    )
    assert distribution.total_mass().item() == pytest.approx(atoms.total_mass().item())
    eta_one_mass = distribution.component_mass[distribution.eta_state == 1.0].sum()
    assert eta_one_mass.item() == pytest.approx(0.2 * atoms.total_mass().item())
    # Each atom's already-realized leverage is duplicated unchanged across the
    # new eta state, rather than eta_{t+1} refinancing it a second time.
    torch.testing.assert_close(
        distribution.b[:, 0::2], distribution.b[:, 1::2], atol=0, rtol=0
    )


def test_fp64_mass_does_not_promote_firm_network_state_dtype():
    atoms = _toy_transport_atoms(points=8)
    atoms = PostDecisionTransportAtoms(
        **{
            **atoms.state_dict(),
            "b_next": atoms.b_next.float(),
            "capital_next_per_unit": atoms.capital_next_per_unit.float(),
            "source_z_current": atoms.source_z_current.float(),
        }
    ).validate()
    distribution = propagate_ar1_with_eta_mixture(
        atoms, rho_z=0.8, sigma_z=0.1, z_bar=0.0, zeta=0.2
    )
    assert distribution.component_mass.dtype == torch.float64
    assert distribution.b.dtype == torch.float32
    assert distribution.eta_state.dtype == torch.float32

