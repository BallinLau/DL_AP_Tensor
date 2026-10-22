"""Elementwise parity: batched i-grid ``cal_phats`` vs the legacy per-i loop.

``PolicyValueModel.cal_phats`` used to evaluate ``_value_outputs`` once per
i-grid point (sequential per-i forwards).  It now expands the state rows and
runs chunked batched forwards instead.  These tests pin the elementwise
equivalence of the two paths (forward outputs and autograd gradients) against
a literal reproduction of the legacy loop semantics: for every i only the
``SIMMODEL.I`` column is replaced, all other state columns stay untouched.
"""

import torch

from config import SIMMODEL
from models.policy_value import PolicyValueModel


def _legacy_value_stacks(model, firm_state, i_values):
    """Reproduce the pre-batching ``cal_phats`` i-loop verbatim."""
    v0_rows = []
    vi_rows = []
    for i_val in i_values:
        modified = firm_state.clone()
        modified[:, SIMMODEL.I] = i_val
        v0_i, vi_i = model._value_outputs(modified)
        v0_rows.append(v0_i)
        vi_rows.append(vi_i)
    return torch.stack(v0_rows, dim=0), torch.stack(vi_rows, dim=0)


def _legacy_cal_phats(model, firm_state, i_values):
    v0_stack, vi_stack = _legacy_value_stacks(model, firm_state, i_values)
    phat = torch.max(v0_stack, vi_stack).mean(dim=0)
    p = model.derived.total_equity(phat, hard=True)
    survival_prob = model.derived.survival(phat)
    bar_z = 1.0 - survival_prob
    return phat, p, bar_z, survival_prob


def _default_i_grid(model, *, dtype, device):
    return torch.linspace(
        0.0,
        model.i_threshold,
        steps=max(int(model.i_grid_size), 2),
        device=device,
        dtype=dtype,
    )


def _make_model(*, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    return PolicyValueModel().to(dtype=dtype).eval()


def _make_state(batch, *, seed=1, dtype=torch.float64, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    # s_i = (b, z, eta, i, x, hatcf, lnkf) — 7 columns.
    state = torch.randn(batch, 7, generator=generator, dtype=torch.float64)
    state[:, SIMMODEL.ETA] = (
        torch.rand(batch, generator=generator) > 0.5
    ).to(torch.float64)
    state[:, SIMMODEL.I] = torch.rand(batch, generator=generator) * 0.5
    return state.to(device=device, dtype=dtype)


def _assert_outputs_close(got, want, *, rtol, atol):
    names = ("Phat", "P", "bar_z", "survival_prob")
    for name, g, w in zip(names, got, want):
        torch.testing.assert_close(g, w, rtol=rtol, atol=atol, msg=lambda m, n=name: f"{n}: {m}")


@torch.no_grad()
def test_default_grid_matches_legacy_loop():
    model = _make_model()
    state = _make_state(257, seed=2)
    grid = _default_i_grid(model, dtype=state.dtype, device=state.device)

    got = model.cal_phats(state)  # simulated_i=None -> built-in linspace grid
    want = _legacy_cal_phats(model, state, grid)

    _assert_outputs_close(got, want, rtol=1e-9, atol=1e-9)


@torch.no_grad()
def test_custom_grid_matches_legacy_loop():
    model = _make_model(seed=3)
    state = _make_state(512, seed=4)
    # Non-uniform grid with values outside [0, i_threshold], passed as a
    # column vector exactly like production ``forward_equity`` callers do.
    grid = torch.tensor(
        [[-0.25], [0.0], [0.07], [0.13], [0.31], [0.5], [0.77]],
        dtype=state.dtype,
        device=state.device,
    )

    got = model.cal_phats(state, grid)
    want = _legacy_cal_phats(model, state, grid)

    _assert_outputs_close(got, want, rtol=1e-9, atol=1e-9)


@torch.no_grad()
def test_multi_chunk_batching_matches_legacy_loop():
    # batch=30000 -> group = 262144 // 30000 = 8 < 11 i-points, so the
    # batched path must iterate at least two row-chunks to cover the grid.
    model = _make_model(seed=5)
    state = _make_state(30000, seed=6)
    grid = _default_i_grid(model, dtype=state.dtype, device=state.device)

    got = model.cal_phats(state, grid)
    want = _legacy_cal_phats(model, state, grid)

    _assert_outputs_close(got, want, rtol=1e-9, atol=1e-9)


def test_gradients_match_legacy_loop():
    model = _make_model(seed=7)
    state = _make_state(128, seed=8)
    grid = torch.tensor([0.05, 0.19, 0.33, 0.44], dtype=state.dtype, device=state.device)
    weights = torch.linspace(1.0, 2.0, 128, dtype=state.dtype, device=state.device).unsqueeze(-1)

    def run(fn):
        model.zero_grad(set_to_none=True)
        state_var = state.clone().requires_grad_(True)
        loss = (fn(model, state_var)[0] * weights).sum()
        loss.backward()
        param_grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        return state_var.grad.detach().clone(), param_grads, float(loss.detach())

    state_grad_b, param_grads_b, loss_b = run(lambda m, s: m.cal_phats(s, grid))
    state_grad_l, param_grads_l, loss_l = run(lambda m, s: _legacy_cal_phats(m, s, grid))

    assert abs(loss_b - loss_l) <= 1e-9 * max(1.0, abs(loss_l))
    torch.testing.assert_close(state_grad_b, state_grad_l, rtol=1e-7, atol=1e-9)
    assert set(param_grads_b) == set(param_grads_l)
    for name in param_grads_b:
        torch.testing.assert_close(
            param_grads_b[name],
            param_grads_l[name],
            rtol=1e-7,
            atol=1e-9,
            msg=lambda m, n=name: f"{n}: {m}",
        )


@torch.no_grad()
def test_float32_matches_legacy_loop():
    # Production runs float32; allow kernel-level accumulation slack but keep
    # the tolerance tight enough to catch any semantic (indexing) bug.
    model = _make_model(seed=9, dtype=torch.float32)
    state = _make_state(4096, seed=10, dtype=torch.float32)
    grid = _default_i_grid(model, dtype=state.dtype, device=state.device)

    got = model.cal_phats(state, grid)
    want = _legacy_cal_phats(model, state, grid)

    _assert_outputs_close(got, want, rtol=1e-5, atol=1e-6)
