"""Isolated Bellman value-only ablation for exp_xz equity value scaling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HyperParams, SIMMODEL
from losses import P0Loss, PILoss
from losses.utils import compute_aio_residual
from models import PolicyValueModel, build_policy_value_from_checkpoint_spec
from utils.firm_transition import apply_refinancing_policy


VALUE_KEYS = ("value_encoder", "v0_head", "vi_head")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_hp_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, HyperParams):
        return dict(payload.__dict__)
    raise ValueError("checkpoint hyperparams must be a dict or HyperParams")


def _value_meta(mode: str, log_max: float) -> Dict[str, Any]:
    return {
        "mode": str(mode).lower(),
        "scale_formula": f"1+exp(clamp(x+z,max={float(log_max):g}))" if str(mode).lower() == "exp_xz" else "1",
        "bellman_normalization": str(mode).lower() == "exp_xz",
        "log_max": float(log_max),
    }


def _module_hashes(state: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for prefix in prefixes:
        h = hashlib.sha256()
        matched = False
        for key, value in sorted(state.items()):
            if key.startswith(prefix + "."):
                matched = True
                h.update(key.encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
        if not matched:
            raise ValueError(f"policy_value state_dict is missing module prefix {prefix!r}")
        out[prefix] = h.hexdigest()
    return out


def _state_dict_hash(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _require_value_metadata(payload: Dict[str, Any], *, expected_mode: str, expected_log_max: float, path: Path) -> None:
    actual = payload.get("value_parameterization")
    expected = _value_meta(expected_mode, expected_log_max)
    if not isinstance(actual, dict):
        raise ValueError(f"{path} is missing value_parameterization")
    keys = ("mode", "scale_formula") if expected["mode"] == "none" else ("mode", "scale_formula", "log_max")
    for key in keys:
        if actual.get(key) != expected[key]:
            raise ValueError(f"{path} value_parameterization.{key} mismatch: {actual.get(key)!r} != {expected[key]!r}")
    if bool(actual.get("bellman_normalization", expected["bellman_normalization"])) != bool(expected["bellman_normalization"]):
        raise ValueError(f"{path} value_parameterization.bellman_normalization mismatch")


def _load_combined_policy(path: Path, *, mode: str, log_max: float, device: torch.device) -> Tuple[PolicyValueModel, Dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "models" not in payload or "policy_value" not in payload["models"]:
        raise ValueError(f"{path} must be a combined checkpoint with models.policy_value")
    _require_value_metadata(payload, expected_mode=mode, expected_log_max=log_max, path=path)
    model = build_policy_value_from_checkpoint_spec(payload, value_scale_mode=mode, value_scale_log_max=log_max).to(device)
    model.load_state_dict(payload["models"]["policy_value"], strict=True)
    return model, payload


def _normalize_branch_weights(
    branch_weights: torch.Tensor,
    *,
    n_parent: int,
    n_child: int,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.as_tensor(branch_weights, dtype=torch.float32, device=device)
    if weights.ndim == 1:
        if weights.shape[0] != n_child:
            raise ValueError(f"branch_weights length must be {n_child}, got {weights.shape[0]}")
        weights = weights.reshape(1, n_child).expand(n_parent, n_child)
    elif weights.ndim == 2:
        if weights.shape != (n_parent, n_child):
            raise ValueError(f"branch_weights shape must be {(n_parent, n_child)}, got {tuple(weights.shape)}")
    else:
        raise ValueError("branch_weights must have shape [J] or [B,J]")
    if not torch.isfinite(weights).all():
        raise ValueError("branch_weights must be finite")
    if (weights < 0).any():
        raise ValueError("branch_weights must be nonnegative")
    rowsum = weights.sum(dim=1, keepdim=True)
    if (rowsum <= 0).any():
        raise ValueError("branch_weights rows must have positive mass")
    return weights / rowsum


def _load_batches(
    path: Path,
    device: torch.device,
    *,
    assume_equal_branch_weights: bool = False,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "train" in payload:
        train = payload["train"]
        val = payload.get("validation", payload.get("val", []))
        metadata = dict(payload.get("metadata") or {})
    else:
        raise ValueError("batch data must be a dict with nonempty train/validation and metadata")
    if not train:
        raise ValueError("batch data train split must be nonempty")
    if not val:
        raise ValueError("batch data validation split must be nonempty")
    m_semantics = metadata.get("m_semantics")
    if m_semantics not in {"raw", "train_clipped", "fixed"}:
        raise ValueError("batch metadata must include m_semantics in {'raw','train_clipped','fixed'}")
    if not metadata.get("shock_bank_hash"):
        raise ValueError("batch metadata must include shock_bank_hash")

    def _normalize(batch: Dict[str, Any]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        parent = torch.as_tensor(batch["parent"], dtype=torch.float32, device=device)
        if parent.ndim != 2 or parent.shape[1] < 7 or not torch.isfinite(parent).all():
            raise ValueError("batch parent must be finite with shape [B,>=7]")
        children = batch.get("children")
        if children is None:
            children = [batch["child0"], batch["child1"]]
        children = [torch.as_tensor(c, dtype=torch.float32, device=device) for c in children]
        if len(children) < 2:
            raise ValueError("each batch must have at least two child branches")
        for child in children:
            if child.ndim != 2 or child.shape[0] != parent.shape[0] or child.shape[1] < 7 or not torch.isfinite(child).all():
                raise ValueError("batch children must be finite with shape [B,>=7]")
        m_list = batch.get("m_list", batch.get("M_list", None))
        if m_list is None and "M" in batch:
            M = torch.as_tensor(batch["M"], dtype=torch.float32, device=device)
            m_list = [M[:, j:j + 1] for j in range(M.shape[1])]
        if m_list is None:
            raise ValueError("each batch must include explicit m_list/M_list or M; ones fallback is disabled")
        m_list = [torch.as_tensor(m, dtype=torch.float32, device=device).reshape(parent.shape[0], 1) for m in m_list]
        if len(m_list) != len(children):
            raise ValueError("len(m_list) must equal len(children)")
        for m in m_list:
            if m.shape != (parent.shape[0], 1) or not torch.isfinite(m).all():
                raise ValueError("each M branch must be finite with shape [B,1]")
        if "branch_weights" not in batch:
            if not assume_equal_branch_weights:
                raise ValueError("branch_weights are required unless --assume-equal-branch-weights is set")
            weights = torch.full((parent.shape[0], len(children)), 1.0 / len(children), dtype=torch.float32, device=device)
        else:
            weights = _normalize_branch_weights(
                batch["branch_weights"],
                n_parent=parent.shape[0],
                n_child=len(children),
                device=device,
            )
        return {
            "parent": parent[:, :7],
            "children": [c[:, :7] for c in children],
            "m_list": m_list,
            "branch_weights": weights,
        }

    return [_normalize(b) for b in train], [_normalize(b) for b in val], metadata


def _freeze_non_value(model: PolicyValueModel) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for name in VALUE_KEYS:
        for p in getattr(model, name).parameters():
            p.requires_grad = True


def _state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _split_value_state(model: PolicyValueModel) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    value = {}
    non_value = {}
    for key, value_tensor in model.state_dict().items():
        target = value if key.startswith(VALUE_KEYS) else non_value
        target[key] = value_tensor.detach().cpu().clone()
    return value, non_value


def _bellman_loss_and_metrics(
    model: PolicyValueModel,
    target: PolicyValueModel,
    batches: List[Dict[str, Any]],
    *,
    normalize: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    p0_loss = P0Loss()
    pi_loss = PILoss()
    losses = []
    p0_phys_cond = []
    pi_phys_cond = []
    p0_norm_cond = []
    pi_norm_cond = []
    z_values = []
    for batch in batches:
        parent = batch["parent"]
        children = batch["children"]
        m_list = batch["m_list"]
        branch_weights = batch["branch_weights"]
        out = model(parent)
        scale = model.equity_value_scale(parent).clamp_min(1e-12)
        b_parent = parent[:, SIMMODEL.B:SIMMODEL.B + 1]
        eta = parent[:, SIMMODEL.ETA:SIMMODEL.ETA + 1].clamp(0.0, 1.0)
        bp0 = out.bp0
        bpI = out.bpI
        q_parent = out.Q
        p0 = out.P0
        pi = out.PI

        p_children_p0 = []
        p_children_pi = []
        with torch.no_grad():
            for child in children:
                child_p0 = child.clone()
                child_p0[:, SIMMODEL.B:SIMMODEL.B + 1] = apply_refinancing_policy(b_parent, bp0, eta)
                child_pi = child.clone()
                child_pi[:, SIMMODEL.B:SIMMODEL.B + 1] = apply_refinancing_policy(b_parent, bpI, eta)
                p_children_p0.append(target(child_p0).P.detach())
                p_children_pi.append(target(child_pi).P.detach())
            p0_q_state = parent.clone()
            p0_q_state[:, SIMMODEL.B:SIMMODEL.B + 1] = apply_refinancing_policy(b_parent, bp0, eta)
            pi_q_state = parent.clone()
            pi_q_state[:, SIMMODEL.B:SIMMODEL.B + 1] = apply_refinancing_policy(b_parent, bpI, eta)
            q_p0 = target(p0_q_state).Q.detach()
            q_pi = target(pi_q_state).Q.detach()

        cf0 = p0_loss.compute_cashflow_p0(parent[:, SIMMODEL.X:SIMMODEL.X + 1], parent[:, SIMMODEL.Z:SIMMODEL.Z + 1], b_parent, q_parent, q_p0, eta)
        cfi = pi_loss.compute_cashflow_pi(
            parent[:, SIMMODEL.X:SIMMODEL.X + 1],
            parent[:, SIMMODEL.Z:SIMMODEL.Z + 1],
            b_parent,
            parent[:, SIMMODEL.I:SIMMODEL.I + 1],
            q_parent,
            q_pi,
            eta,
        )
        rz = [torch.zeros_like(p0) for _ in children]
        r0_phys = p0_loss.compute_bellman_residual(p0, cf0, m_list, p_children_p0, rz)
        ri_phys = pi_loss.compute_bellman_residual(pi, cfi, m_list, p_children_pi, rz)
        r0_train = [r / scale for r in r0_phys] if normalize else r0_phys
        ri_train = [r / scale for r in ri_phys] if normalize else ri_phys
        losses.append(compute_aio_residual(r0_train, p0_loss.aio_weight).mean())
        losses.append(compute_aio_residual(ri_train, pi_loss.aio_weight).mean())

        weights = branch_weights.to(device=parent.device, dtype=parent.dtype)
        r0_stack = torch.stack([r.reshape(parent.shape[0], 1) for r in r0_phys], dim=1).squeeze(-1)
        ri_stack = torch.stack([r.reshape(parent.shape[0], 1) for r in ri_phys], dim=1).squeeze(-1)
        r0_norm_stack = r0_stack / scale.reshape(parent.shape[0], 1)
        ri_norm_stack = ri_stack / scale.reshape(parent.shape[0], 1)
        p0_phys_cond.append((weights * r0_stack).sum(dim=1).detach())
        pi_phys_cond.append((weights * ri_stack).sum(dim=1).detach())
        p0_norm_cond.append((weights * r0_norm_stack).sum(dim=1).detach())
        pi_norm_cond.append((weights * ri_norm_stack).sum(dim=1).detach())
        z_values.append(parent[:, SIMMODEL.Z].detach())
    total = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=next(model.parameters()).device)

    def _stats(vals: List[torch.Tensor], prefix: str) -> Dict[str, float]:
        v = torch.cat(vals) if vals else torch.empty(0)
        if v.numel() == 0:
            return {
                f"{prefix}_conditional_signed_mean": float("nan"),
                f"{prefix}_conditional_mean_abs": float("nan"),
                f"{prefix}_conditional_p90_abs": float("nan"),
                f"{prefix}_conditional_max_abs": float("nan"),
                f"{prefix}_finite_ratio": 0.0,
                f"{prefix}_n_parent": 0,
            }
        finite = torch.isfinite(v)
        vf = v[finite]
        if vf.numel() == 0:
            return {
                f"{prefix}_conditional_signed_mean": float("nan"),
                f"{prefix}_conditional_mean_abs": float("nan"),
                f"{prefix}_conditional_p90_abs": float("nan"),
                f"{prefix}_conditional_max_abs": float("nan"),
                f"{prefix}_finite_ratio": 0.0,
                f"{prefix}_n_parent": int(v.numel()),
            }
        av = vf.abs()
        return {
            f"{prefix}_conditional_signed_mean": float(vf.mean().item()),
            f"{prefix}_conditional_mean_abs": float(av.mean().item()),
            f"{prefix}_conditional_p90_abs": float(torch.quantile(av, 0.9).item()),
            f"{prefix}_conditional_max_abs": float(av.max().item()),
            f"{prefix}_finite_ratio": float(finite.to(torch.float32).mean().item()),
            f"{prefix}_n_parent": int(v.numel()),
        }

    def _region_stats(z: torch.Tensor, values: torch.Tensor, normalized: torch.Tensor, prefix: str) -> Dict[str, Dict[str, float]]:
        regions = {
            "low_z": z < -1.0,
            "mid_z": (z >= -1.0) & (z <= 1.0),
            "high_z": z > 1.0,
            "full": torch.ones_like(z, dtype=torch.bool),
        }
        out: Dict[str, Dict[str, float]] = {}
        for name, mask in regions.items():
            vals = values[mask]
            norm = normalized[mask]
            base = _stats([vals], prefix)
            norm_stats = _stats([norm], f"{prefix}_normalized")
            out[name] = {
                "conditional_signed_mean": base[f"{prefix}_conditional_signed_mean"],
                "conditional_mean_abs": base[f"{prefix}_conditional_mean_abs"],
                "conditional_p90_abs": base[f"{prefix}_conditional_p90_abs"],
                "conditional_max_abs": base[f"{prefix}_conditional_max_abs"],
                "normalized_conditional_mean_abs": norm_stats[f"{prefix}_normalized_conditional_mean_abs"],
                "finite_ratio": base[f"{prefix}_finite_ratio"],
                "n_parent": base[f"{prefix}_n_parent"],
            }
        return out

    metrics = {}
    metrics.update(_stats(p0_phys_cond, "p0_physical"))
    metrics.update(_stats(pi_phys_cond, "pi_physical"))
    metrics.update(_stats(p0_norm_cond, "p0_normalized"))
    metrics.update(_stats(pi_norm_cond, "pi_normalized"))
    if z_values:
        z_all = torch.cat(z_values)
        p0_all = torch.cat(p0_phys_cond)
        pi_all = torch.cat(pi_phys_cond)
        p0_norm_all = torch.cat(p0_norm_cond)
        pi_norm_all = torch.cat(pi_norm_cond)
        metrics["regions"] = {
            "p0": _region_stats(z_all, p0_all, p0_norm_all, "p0"),
            "pi": _region_stats(z_all, pi_all, pi_norm_all, "pi"),
        }
    return total, metrics


def _validate_checkpoint_lineage(
    *,
    baseline_path: Path,
    baseline_payload: Dict[str, Any],
    scaled_payload: Dict[str, Any],
    log_max: float,
) -> Dict[str, Any]:
    _require_value_metadata(baseline_payload, expected_mode="none", expected_log_max=log_max, path=baseline_path)
    if baseline_payload["value_parameterization"].get("scale_formula") != "1":
        raise ValueError("baseline checkpoint must use scale_formula='1'")
    expected_scaled = _value_meta("exp_xz", log_max)
    for key in ("mode", "scale_formula", "log_max", "bellman_normalization"):
        if scaled_payload.get("value_parameterization", {}).get(key) != expected_scaled[key]:
            raise ValueError(f"scaled checkpoint value_parameterization.{key} mismatch")
    warmstart = scaled_payload.get("warmstart")
    if not isinstance(warmstart, dict):
        raise ValueError("scaled checkpoint is missing warmstart metadata")
    expected_source_hash = _sha256(baseline_path)
    if warmstart.get("source_sha256") != expected_source_hash:
        raise ValueError("scaled warmstart source hash does not match baseline checkpoint")
    if baseline_payload.get("config_snapshot") != scaled_payload.get("config_snapshot"):
        raise ValueError("baseline/scaled config_snapshot mismatch")
    if baseline_payload.get("policy_value_model_spec") != scaled_payload.get("policy_value_model_spec"):
        raise ValueError("baseline/scaled policy_value_model_spec mismatch")
    baseline_models = baseline_payload.get("models") or {}
    scaled_models = scaled_payload.get("models") or {}
    if "sdf_fc1" in baseline_models or "sdf_fc1" in scaled_models:
        if _state_dict_hash(baseline_models.get("sdf_fc1", {})) != _state_dict_hash(scaled_models.get("sdf_fc1", {})):
            raise ValueError("baseline/scaled sdf_fc1 hash mismatch")
    non_value_modules = ("q_encoder", "q_head", "policy_encoder", "bp0_head", "bpi_head", "barz_model", "bari_model")
    baseline_non_value = _module_hashes(baseline_models["policy_value"], non_value_modules)
    scaled_non_value = _module_hashes(scaled_models["policy_value"], non_value_modules)
    if baseline_non_value != scaled_non_value:
        raise ValueError("baseline/scaled non-value policy modules differ")
    return {
        "baseline_sha256": expected_source_hash,
        "sdf_fc1_hash": _state_dict_hash(baseline_models.get("sdf_fc1", {})) if "sdf_fc1" in baseline_models else None,
        "non_value_module_hashes": baseline_non_value,
    }


def _score(metrics: Dict[str, Any]) -> float:
    return float(metrics["p0_physical_conditional_mean_abs"] + metrics["pi_physical_conditional_mean_abs"])


def _physical_rmse_and_max(a: PolicyValueModel, b: PolicyValueModel, batches: List[Dict[str, Any]]) -> Dict[str, float]:
    v0_err = []
    vi_err = []
    with torch.no_grad():
        for batch in batches:
            parent = batch["parent"]
            ca = a.forward_value_components(parent)
            cb = b.forward_value_components(parent)
            v0_err.append((ca["V0_physical"] - cb["V0_physical"]).reshape(-1))
            vi_err.append((ca["VI_physical"] - cb["VI_physical"]).reshape(-1))
    v0 = torch.cat(v0_err) if v0_err else torch.empty(0)
    vi = torch.cat(vi_err) if vi_err else torch.empty(0)
    return {
        "v0_physical_rmse": float(torch.sqrt(torch.mean(v0.square())).item()) if v0.numel() else float("nan"),
        "vi_physical_rmse": float(torch.sqrt(torch.mean(vi.square())).item()) if vi.numel() else float("nan"),
        "v0_physical_max_abs": float(v0.abs().max().item()) if v0.numel() else float("nan"),
        "vi_physical_max_abs": float(vi.abs().max().item()) if vi.numel() else float("nan"),
    }


def _residual_metric_differences(a: Dict[str, Any], b: Dict[str, Any], prefix: str) -> Dict[str, float]:
    out = {}
    for eq in ("p0", "pi"):
        key = f"{eq}_physical_conditional_mean_abs"
        out[f"{prefix}_{eq}_conditional_mean_abs_diff"] = float(b[key] - a[key])
    return out


def _save_combined(
    path: Path,
    payload: Dict[str, Any],
    student: PolicyValueModel,
    teacher: PolicyValueModel,
    hp: HyperParams,
    source: Path,
    seed: int,
) -> None:
    out = copy.deepcopy(payload)
    out.setdefault("models", {})
    out["models"]["policy_value"] = student.state_dict()
    out["models"]["firm_target"] = teacher.state_dict()
    hyperparams_payload = _as_hp_dict(out.get("hyperparams"))
    hyperparams_payload["pv_value_scale_mode"] = "exp_xz"
    hyperparams_payload["pv_value_scale_log_max"] = float(hp.pv_value_scale_log_max)
    hyperparams_payload["pv_bellman_normalize_by_value_scale"] = True
    out["hyperparams"] = hyperparams_payload
    optimizers = dict(out.get("optimizers") or {})
    optimizers.pop("policy_value", None)
    if optimizers:
        out["optimizers"] = optimizers
    elif "optimizers" in out:
        del out["optimizers"]
    out["policy_value_model_spec"] = student.model_spec()
    out["value_parameterization"] = {
        "mode": "exp_xz",
        "scale_formula": f"1+exp(clamp(x+z,max={float(hp.pv_value_scale_log_max):g}))",
        "bellman_normalization": bool(hp.pv_bellman_normalize_by_value_scale),
        "log_max": float(hp.pv_value_scale_log_max),
    }
    out["scaled_value_ablation"] = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "seed": int(seed),
        "resume_optimizer_compatible": False,
        "resume_optimizer_incompatibility_reason": "value_parameterization_migration_or_value_only_training",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--scaled-checkpoint", required=True)
    p.add_argument("--batch-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--epochs-per-round", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cpu")
    p.add_argument("--value-scale-log-max", type=float, default=20.0)
    p.add_argument("--assume-equal-branch-weights", action="store_true")
    p.add_argument("--force-reject-round", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(args.baseline_checkpoint)
    scaled_path = Path(args.scaled_checkpoint)
    baseline, baseline_payload = _load_combined_policy(baseline_path, mode="none", log_max=args.value_scale_log_max, device=device)
    student, scaled_payload = _load_combined_policy(scaled_path, mode="exp_xz", log_max=args.value_scale_log_max, device=device)
    lineage = _validate_checkpoint_lineage(
        baseline_path=baseline_path,
        baseline_payload=baseline_payload,
        scaled_payload=scaled_payload,
        log_max=args.value_scale_log_max,
    )
    baseline.eval().requires_grad_(False)
    teacher = copy.deepcopy(student).to(device).eval().requires_grad_(False)
    _freeze_non_value(student)
    train_batches, val_batches, batch_metadata = _load_batches(
        Path(args.batch_data),
        device,
        assume_equal_branch_weights=bool(args.assume_equal_branch_weights),
    )
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)

    eval_batches = val_batches
    _, metrics_a = _bellman_loss_and_metrics(baseline, baseline, eval_batches, normalize=False)
    _, metrics_b = _bellman_loss_and_metrics(student, teacher, eval_batches, normalize=True)
    metrics_b.update(_physical_rmse_and_max(baseline, student, eval_batches))
    metrics_b.update(_residual_metric_differences(metrics_a, metrics_b, "b_vs_a"))
    value_before, non_value_before = _split_value_state(student)
    teacher_hash_before = _state_hash(teacher)
    best_score = _score(metrics_b)
    history = [{"stage": "baseline_unscaled", **metrics_a}, {"stage": "scaled_migrated_pretrain", **metrics_b}]

    for round_idx in range(int(args.rounds)):
        model_state = copy.deepcopy(student.state_dict())
        opt_state = copy.deepcopy(opt.state_dict())
        py_rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        teacher_round_hash = _state_hash(teacher)
        _, start_metrics = _bellman_loss_and_metrics(student, teacher, eval_batches, normalize=True)
        start_score = _score(start_metrics)
        for _ in range(int(args.epochs_per_round)):
            for batch in train_batches:
                loss, _ = _bellman_loss_and_metrics(student, teacher, [batch], normalize=True)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        _, candidate_metrics = _bellman_loss_and_metrics(student, teacher, eval_batches, normalize=True)
        score = _score(candidate_metrics)
        accepted = bool(score <= min(best_score, start_score))
        if int(args.force_reject_round) == round_idx + 1:
            accepted = False
        teacher_unchanged_within_round = _state_hash(teacher) == teacher_round_hash
        if accepted:
            best_score = score
            teacher.load_state_dict(student.state_dict(), strict=True)
            teacher.eval().requires_grad_(False)
            _, self_metrics = _bellman_loss_and_metrics(student, teacher, eval_batches, normalize=True)
            restored_metrics = None
        else:
            student.load_state_dict(model_state, strict=True)
            opt.load_state_dict(opt_state)
            random.setstate(py_rng_state)
            np.random.set_state(np_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            _, restored_metrics = _bellman_loss_and_metrics(student, teacher, eval_batches, normalize=True)
            self_metrics = None
        history.append({
            "stage": "scaled_posttrain_round",
            "round": round_idx + 1,
            "accepted": accepted,
            "teacher_hash_unchanged_within_round": teacher_unchanged_within_round,
            "start_score": start_score,
            "candidate_score": score,
            "start_metrics": start_metrics,
            "candidate_metrics": candidate_metrics,
            "restored_metrics": restored_metrics,
            "self_metrics_after_accept": self_metrics,
        })

    _, metrics_c = _bellman_loss_and_metrics(student, teacher, eval_batches, normalize=True)
    metrics_c.update(_residual_metric_differences(metrics_a, metrics_c, "c_vs_a"))
    for region in ("low_z", "mid_z", "high_z", "full"):
        c_p0 = metrics_c.get("regions", {}).get("p0", {}).get(region, {}).get("conditional_mean_abs", float("nan"))
        a_p0 = metrics_a.get("regions", {}).get("p0", {}).get(region, {}).get("conditional_mean_abs", float("nan"))
        c_pi = metrics_c.get("regions", {}).get("pi", {}).get(region, {}).get("conditional_mean_abs", float("nan"))
        a_pi = metrics_a.get("regions", {}).get("pi", {}).get(region, {}).get("conditional_mean_abs", float("nan"))
        metrics_c[f"c_vs_a_{region}_p0_relative_improvement"] = float((a_p0 - c_p0) / max(abs(a_p0), 1e-12))
        metrics_c[f"c_vs_a_{region}_pi_relative_improvement"] = float((a_pi - c_pi) / max(abs(a_pi), 1e-12))
    _, non_value_after = _split_value_state(student)
    non_value_unchanged = all(torch.equal(non_value_before[k], v) for k, v in non_value_after.items())
    value_after, _ = _split_value_state(student)
    value_changed = any(not torch.equal(value_before[k], value_after[k]) for k in value_before)
    metrics_c["only_value_params_change"] = bool(non_value_unchanged and value_changed)
    metrics_c["pi_high_b_penalty_effective_weight"] = 0.0
    denom = max(_score(metrics_a), 1e-12)
    metrics_c["physical_relative_improvement_vs_baseline"] = float((denom - _score(metrics_c)) / denom)
    history.append({"stage": "scaled_posttrain", **metrics_c})

    checkpoint_path = out_dir / "scaled_posttrain_combined.pt"
    hp = HyperParams()
    hp.pv_value_scale_mode = "exp_xz"
    hp.pv_value_scale_log_max = float(args.value_scale_log_max)
    hp.pv_bellman_normalize_by_value_scale = True
    _save_combined(checkpoint_path, scaled_payload, student, teacher, hp, scaled_path, args.seed)
    summary = {
        "baseline": metrics_a,
        "scaled_migrated_pretrain": metrics_b,
        "scaled_posttrain": metrics_c,
        "shock_hash": hashlib.sha256(Path(args.batch_data).read_bytes()).hexdigest(),
        "batch_metadata": batch_metadata,
        "lineage": lineage,
        "posttrain_checkpoint": str(checkpoint_path),
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
