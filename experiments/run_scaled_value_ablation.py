"""Isolated Bellman value-only ablation for exp_xz equity value scaling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HyperParams, SIMMODEL
from losses import P0Loss, PILoss
from losses.utils import compute_aio_residual
from models import PolicyValueModel
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


def _load_combined_policy(path: Path, *, mode: str, log_max: float, device: torch.device) -> Tuple[PolicyValueModel, Dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "models" not in payload or "policy_value" not in payload["models"]:
        raise ValueError(f"{path} must be a combined checkpoint with models.policy_value")
    model = PolicyValueModel(value_scale_mode=mode, value_scale_log_max=log_max).to(device)
    model.load_state_dict(payload["models"]["policy_value"], strict=True)
    return model, payload


def _load_batches(path: Path, device: torch.device) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "train" in payload:
        train = payload["train"]
        val = payload.get("validation", payload.get("val", []))
    elif isinstance(payload, list):
        train, val = payload, []
    else:
        raise ValueError("batch data must be a list or dict with train/validation")

    def _normalize(batch: Dict[str, Any]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        parent = torch.as_tensor(batch["parent"], dtype=torch.float32, device=device)
        children = batch.get("children")
        if children is None:
            children = [batch["child0"], batch["child1"]]
        children = [torch.as_tensor(c, dtype=torch.float32, device=device) for c in children]
        m_list = batch.get("m_list", batch.get("M_list", None))
        if m_list is None:
            if "M" in batch:
                M = torch.as_tensor(batch["M"], dtype=torch.float32, device=device)
                m_list = [M[:, j:j + 1] for j in range(M.shape[1])]
            else:
                m_list = [torch.ones(parent.shape[0], 1, dtype=torch.float32, device=device) for _ in children]
        m_list = [torch.as_tensor(m, dtype=torch.float32, device=device).reshape(parent.shape[0], 1) for m in m_list]
        return {"parent": parent[:, :7], "children": [c[:, :7] for c in children], "m_list": m_list}

    return [_normalize(b) for b in train], [_normalize(b) for b in val]


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
    p0_phys = []
    pi_phys = []
    p0_norm = []
    pi_norm = []
    for batch in batches:
        parent = batch["parent"]
        children = batch["children"]
        m_list = batch["m_list"]
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
        p0_phys.extend([r.detach().reshape(-1) for r in r0_phys])
        pi_phys.extend([r.detach().reshape(-1) for r in ri_phys])
        p0_norm.extend([(r / scale).detach().reshape(-1) for r in r0_phys])
        pi_norm.extend([(r / scale).detach().reshape(-1) for r in ri_phys])
    total = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=next(model.parameters()).device)

    def _stats(vals: List[torch.Tensor], prefix: str) -> Dict[str, float]:
        v = torch.cat(vals) if vals else torch.empty(0)
        if v.numel() == 0:
            return {f"{prefix}_signed_mean": float("nan"), f"{prefix}_mean_abs": float("nan"), f"{prefix}_p90_abs": float("nan"), f"{prefix}_max_abs": float("nan")}
        av = v.abs()
        return {
            f"{prefix}_signed_mean": float(v.mean().item()),
            f"{prefix}_mean_abs": float(av.mean().item()),
            f"{prefix}_p90_abs": float(torch.quantile(av, 0.9).item()),
            f"{prefix}_max_abs": float(av.max().item()),
        }

    metrics = {}
    metrics.update(_stats(p0_phys, "p0_physical"))
    metrics.update(_stats(pi_phys, "pi_physical"))
    metrics.update(_stats(p0_norm, "p0_normalized"))
    metrics.update(_stats(pi_norm, "pi_normalized"))
    return total, metrics


def _save_combined(path: Path, payload: Dict[str, Any], model: PolicyValueModel, hp: HyperParams, source: Path, seed: int) -> None:
    out = copy.deepcopy(payload)
    out.setdefault("models", {})
    out["models"]["policy_value"] = model.state_dict()
    out.setdefault("models", {}).setdefault("firm_target", model.state_dict())
    out["hyperparams"] = {**_as_hp_dict(out.get("hyperparams")), **hp.__dict__}
    out["value_parameterization"] = {
        "mode": "exp_xz",
        "scale_formula": f"1+exp(clamp(x+z,max={float(hp.pv_value_scale_log_max):g}))",
        "bellman_normalization": bool(hp.pv_bellman_normalize_by_value_scale),
        "log_max": float(hp.pv_value_scale_log_max),
    }
    out["scaled_value_ablation"] = {"source": str(source), "source_sha256": _sha256(source), "seed": int(seed)}
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
    baseline.eval().requires_grad_(False)
    teacher = copy.deepcopy(student).to(device).eval().requires_grad_(False)
    _freeze_non_value(student)
    train_batches, val_batches = _load_batches(Path(args.batch_data), device)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)

    _, metrics_a = _bellman_loss_and_metrics(baseline, baseline, val_batches or train_batches, normalize=False)
    _, metrics_b = _bellman_loss_and_metrics(student, teacher, val_batches or train_batches, normalize=True)
    value_before, non_value_before = _split_value_state(student)
    teacher_hash_before = _state_hash(teacher)
    best_score = metrics_b["p0_physical_mean_abs"] + metrics_b["pi_physical_mean_abs"]
    history = [{"stage": "baseline_unscaled", **metrics_a}, {"stage": "scaled_migrated_pretrain", **metrics_b}]

    for round_idx in range(int(args.rounds)):
        model_state = copy.deepcopy(student.state_dict())
        opt_state = copy.deepcopy(opt.state_dict())
        rng_state = torch.random.get_rng_state()
        teacher_round_hash = _state_hash(teacher)
        for _ in range(int(args.epochs_per_round)):
            for batch in train_batches:
                loss, _ = _bellman_loss_and_metrics(student, teacher, [batch], normalize=True)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        _, val_metrics = _bellman_loss_and_metrics(student, teacher, val_batches or train_batches, normalize=True)
        score = val_metrics["p0_physical_mean_abs"] + val_metrics["pi_physical_mean_abs"]
        accepted = bool(score <= best_score)
        teacher_unchanged_within_round = _state_hash(teacher) == teacher_round_hash
        if accepted:
            best_score = score
            teacher.load_state_dict(student.state_dict(), strict=True)
            teacher.eval().requires_grad_(False)
        else:
            student.load_state_dict(model_state, strict=True)
            opt.load_state_dict(opt_state)
            torch.random.set_rng_state(rng_state)
        history.append({
            "stage": "scaled_posttrain_round",
            "round": round_idx + 1,
            "accepted": accepted,
            "teacher_hash_unchanged_within_round": teacher_unchanged_within_round,
            **val_metrics,
        })

    _, metrics_c = _bellman_loss_and_metrics(student, teacher, val_batches or train_batches, normalize=True)
    _, non_value_after = _split_value_state(student)
    non_value_unchanged = all(torch.equal(non_value_before[k], v) for k, v in non_value_after.items())
    value_after, _ = _split_value_state(student)
    value_changed = any(not torch.equal(value_before[k], value_after[k]) for k in value_before)
    metrics_c["only_value_params_change"] = bool(non_value_unchanged and value_changed)
    metrics_c["pi_high_b_penalty_effective_weight"] = 0.0
    denom = max(metrics_a["p0_physical_mean_abs"] + metrics_a["pi_physical_mean_abs"], 1e-12)
    metrics_c["physical_relative_improvement_vs_baseline"] = float((denom - (metrics_c["p0_physical_mean_abs"] + metrics_c["pi_physical_mean_abs"])) / denom)
    history.append({"stage": "scaled_posttrain", **metrics_c})

    checkpoint_path = out_dir / "scaled_posttrain_combined.pt"
    hp = HyperParams()
    hp.pv_value_scale_mode = "exp_xz"
    hp.pv_value_scale_log_max = float(args.value_scale_log_max)
    hp.pv_bellman_normalize_by_value_scale = True
    _save_combined(checkpoint_path, scaled_payload, student, hp, scaled_path, args.seed)
    summary = {
        "baseline": metrics_a,
        "scaled_migrated_pretrain": metrics_b,
        "scaled_posttrain": metrics_c,
        "shock_hash": hashlib.sha256(Path(args.batch_data).read_bytes()).hexdigest(),
        "posttrain_checkpoint": str(checkpoint_path),
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
