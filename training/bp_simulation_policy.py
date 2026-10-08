"""Vectorized grid-policy resolver for formal firm simulation.

This module deliberately contains no Bellman equations.  It builds the fixed
child shock bank used by a simulation rollout and delegates every candidate
evaluation and argmax to :class:`training.bp_grid_teacher.BPGridTeacher`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import torch

from analysis.convergence_transition import (
    ConvergenceShockBank,
    MacroTransitionContext,
    build_child_exogenous_bundle,
)
from utils.firm_transition import expand_children_exact_eta_tensor

from .bp_grid_teacher import BPGridTeacher


def _module_state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def compose_grid_simulation_action(
    p0_star: torch.Tensor,
    mix_star: torch.Tensor,
    survival_probability: torch.Tensor,
) -> torch.Tensor:
    """Apply the same survival mixing used by ``PolicyValueModel.forward``."""
    survival = survival_probability.detach().clamp(0.0, 1.0)
    return survival * mix_star + (1.0 - survival) * p0_star


class GridBPSimulationPolicy:
    """Frozen, RNG-neutral, GPU-vectorized BP policy for one rollout.

    The policy model and SDF/FC1 model are read-only.  A single continuous
    shock bank is created at construction and reused for every parent batch in
    the rollout.  Future eta is integrated exactly by duplicating each
    continuous child with eta=0/1 and its Bernoulli probability mass.
    """

    def __init__(
        self,
        *,
        target_model: torch.nn.Module,
        sdf_fc1_model: torch.nn.Module,
        p0_loss: Any,
        pi_loss: Any,
        hyperparams: Any,
        economic_config: Any,
        n_child_shocks: int,
        shock_seed: int,
        require_cuda: bool = True,
    ) -> None:
        if sdf_fc1_model is None:
            raise ValueError("grid simulation BP requires sdf_fc1_model")
        if int(n_child_shocks) < 1:
            raise ValueError("simulation grid BP requires at least one child shock")
        try:
            device = next(target_model.parameters()).device
        except StopIteration as exc:  # pragma: no cover - production models have parameters
            raise ValueError("target_model must expose parameters") from exc
        if require_cuda and device.type != "cuda":
            raise RuntimeError(
                "simulation_bp_action_source='grid' requires a CUDA target model; "
                "CPU fallback is intentionally disabled for formal runs"
            )
        sdf_device = next(sdf_fc1_model.parameters()).device
        if sdf_device != device:
            raise ValueError(
                f"policy target and sdf_fc1 must share a device, got {device} and {sdf_device}"
            )

        self.target_model = target_model
        self.sdf_fc1_model = sdf_fc1_model
        self.hyperparams = hyperparams
        self.economic_config = economic_config
        self.device = device
        self.n_child_shocks = int(n_child_shocks)
        self.shock_seed = int(shock_seed)
        self.teacher = BPGridTeacher.from_hyperparams(
            target_model,
            p0_loss,
            pi_loss,
            hyperparams,
        )
        dtype = next(target_model.parameters()).dtype
        self._base_shock_bank = ConvergenceShockBank.create(
            1,
            self.n_child_shocks,
            seed=self.shock_seed,
            device=device,
            dtype=dtype,
        )
        self._target_hash = _module_state_hash(target_model)
        self._sdf_hash = _module_state_hash(sdf_fc1_model)
        self.calls = 0
        self.parent_rows = 0
        self.refinancing_active_rows = 0
        self.coarse_candidate_evaluations = 0
        self.fine_candidate_evaluations = 0

    def _shock_bank_for_rows(self, n_rows: int) -> ConvergenceShockBank:
        def expand(value: torch.Tensor) -> torch.Tensor:
            return value.expand(int(n_rows), -1, -1)

        return ConvergenceShockBank(
            eps_x=expand(self._base_shock_bank.eps_x),
            eps_z=expand(self._base_shock_bank.eps_z),
            u_eta=expand(self._base_shock_bank.u_eta),
            u_i=expand(self._base_shock_bank.u_i),
            seed=self._base_shock_bank.seed,
        )

    def _build_transition(
        self,
        parent_state: torch.Tensor,
        hatc_cal: torch.Tensor,
        lnk_cal: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        n_parent = int(parent_state.shape[0])
        bundle = build_child_exogenous_bundle(
            self.sdf_fc1_model,
            parent_state,
            MacroTransitionContext(
                hatc_cal=hatc_cal.reshape(-1, 1),
                lnk_cal=lnk_cal.reshape(-1, 1),
            ),
            self._shock_bank_for_rows(n_parent),
            economic_config=self.economic_config,
        )
        continuous_children = torch.stack(
            [
                parent_state[:, 0:1].expand(-1, self.n_child_shocks),
                bundle.z_next[..., 0],
                torch.zeros_like(bundle.eta_next[..., 0]),
                bundle.i_next[..., 0],
                bundle.x_next[..., 0],
                bundle.hatcf_next[..., 0],
                bundle.lnkf_next[..., 0],
            ],
            dim=-1,
        )
        expansion = expand_children_exact_eta_tensor(
            continuous_children,
            zeta=float(self.economic_config.ZETA),
            child_weights=bundle.branch_weights,
        )
        m_raw = bundle.m_raw
        if bool(getattr(self.hyperparams, "pv_use_clipped_m", True)):
            m_used = m_raw.clamp(
                float(getattr(self.hyperparams, "pv_m_clamp_min", 0.7)),
                float(getattr(self.hyperparams, "pv_m_clamp_max", 1.3)),
            )
        else:
            m_used = m_raw
        m_exact = (
            m_used.unsqueeze(2)
            .expand(-1, -1, 2, -1)
            .reshape(n_parent, 2 * self.n_child_shocks, 1)
        )
        children_tensor = expansion.children_tensor
        if children_tensor is None:  # pragma: no cover - tensor helper always returns it
            raise RuntimeError("exact eta expansion did not return children_tensor")
        return (
            list(children_tensor.unbind(dim=1)),
            list(m_exact.unbind(dim=1)),
            expansion.branch_weights,
        )

    @torch.inference_mode()
    def evaluate_actions(
        self,
        *,
        firm_state: torch.Tensor,
        hatc_cal: torch.Tensor,
        lnk_cal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if firm_state.device != self.device:
            raise ValueError(
                f"parent states must remain on {self.device}, got {firm_state.device}"
            )
        output = self.target_model(firm_state)
        children, m_list, child_weights = self._build_transition(
            firm_state,
            hatc_cal,
            lnk_cal,
        )
        p0 = self.teacher.compute(
            firm_state,
            children,
            m_list,
            branch="p0",
            bp_pred=output.bp0,
            child_weights=child_weights,
        )
        mixed = self.teacher.compute(
            firm_state,
            children,
            m_list,
            branch="mix",
            bp_pred=output.bp_cond,
            mix_weight=output.bar_i_cond,
            child_weights=child_weights,
        )
        action = compose_grid_simulation_action(
            p0["bp_star"],
            mixed["bp_star"],
            output.survival_prob,
        )
        active = int((firm_state[:, 2] > 0.5).sum().item())
        self.calls += 1
        self.parent_rows += int(firm_state.shape[0])
        self.refinancing_active_rows += active
        self.coarse_candidate_evaluations += 2 * active * int(self.teacher.coarse_size)
        if self.teacher.refine:
            self.fine_candidate_evaluations += 2 * active * int(self.teacher.fine_size)
        return {
            "bp": action.detach(),
            "bp0_star": p0["bp_star"].detach(),
            "mix_star": mixed["bp_star"].detach(),
            "bp_head": output.bp.detach(),
            "bp0_head": output.bp0.detach(),
            "bpi_head": output.bpI.detach(),
            "mix_head": output.bp_cond.detach(),
            "survival_probability": output.survival_prob.detach(),
            "top2_margin": mixed["top2_margin"].detach(),
            "regret": mixed["regret"].detach(),
            "refi_active": mixed["refi_active"].detach(),
        }

    @torch.inference_mode()
    def __call__(self, **context: Any) -> torch.Tensor:
        if int(context["branch"]) != -1:
            return context["bp_head"]
        return self.evaluate_actions(
            firm_state=context["firm_state"],
            hatc_cal=context["hatc_cal"],
            lnk_cal=context["lnk_cal"],
        )["bp"]

    def verify_immutable(self) -> None:
        if _module_state_hash(self.target_model) != self._target_hash:
            raise RuntimeError("grid simulation mutated the frozen policy/value target")
        if _module_state_hash(self.sdf_fc1_model) != self._sdf_hash:
            raise RuntimeError("grid simulation mutated the SDF/FC1 model")

    def instrumentation(self) -> Dict[str, Any]:
        return {
            "simulation_bp_source": "grid",
            "device": str(self.device),
            "n_child_shocks": self.n_child_shocks,
            "shock_seed": self.shock_seed,
            "eta_integration": "exact",
            "calls": int(self.calls),
            "num_parent_states": int(self.parent_rows),
            "refinancing_active_rows": int(self.refinancing_active_rows),
            "num_candidate_evaluations": int(
                self.coarse_candidate_evaluations + self.fine_candidate_evaluations
            ),
            "coarse_eval_count": int(self.coarse_candidate_evaluations),
            "fine_eval_count": int(self.fine_candidate_evaluations),
            **self.teacher.forward_stats(),
        }
