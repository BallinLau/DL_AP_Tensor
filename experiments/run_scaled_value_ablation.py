"""Isolated value-only ablation for equity value scale parameterization.

This runner intentionally does not execute outer episodes. It compares a
control policy-value checkpoint and an exp_xz-scaled warm-start checkpoint on
the same fixed parent states, while freezing non-value modules.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models import PolicyValueModel


def _load_states(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "parent" in payload:
        payload = payload["parent"]
    states = torch.as_tensor(payload, dtype=torch.float32)
    if states.shape[-1] < 7:
        raise ValueError("parent-state tensor must have at least seven columns")
    return states[:, :7].float()


def _load_policy(path: str, *, mode: str, log_max: float, device: torch.device) -> PolicyValueModel:
    payload = torch.load(path, map_location=device)
    state = payload["models"]["policy_value"] if isinstance(payload, dict) and "models" in payload else payload
    model = PolicyValueModel(value_scale_mode=mode, value_scale_log_max=log_max).to(device)
    model.load_state_dict(state, strict=True)
    return model


def _freeze_non_value(model: PolicyValueModel) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for module in (model.value_encoder, model.v0_head, model.vi_head):
        for p in module.parameters():
            p.requires_grad = True


def _eval_physical_mae(student: PolicyValueModel, teacher: PolicyValueModel, states: torch.Tensor) -> dict[str, float]:
    student.eval()
    teacher.eval()
    with torch.no_grad():
        s = student.forward_value_components(states)
        t = teacher.forward_value_components(states)
        return {
            "v0_physical_mae": float((s["V0_physical"] - t["V0_physical"]).abs().mean().item()),
            "vi_physical_mae": float((s["VI_physical"] - t["VI_physical"]).abs().mean().item()),
            "value_scale_mean": float(s["value_scale"].mean().item()),
            "value_scale_max": float(s["value_scale"].max().item()),
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-policy-checkpoint", required=True)
    p.add_argument("--scaled-policy-checkpoint", required=True)
    p.add_argument("--parent-states", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cpu")
    p.add_argument("--value-scale-log-max", type=float, default=20.0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher = _load_policy(args.baseline_policy_checkpoint, mode="none", log_max=args.value_scale_log_max, device=device)
    student = _load_policy(args.scaled_policy_checkpoint, mode="exp_xz", log_max=args.value_scale_log_max, device=device)
    teacher.eval().requires_grad_(False)
    _freeze_non_value(student)
    states = _load_states(args.parent_states).to(device)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)

    history = []
    best = None
    best_state = None
    for epoch in range(int(args.epochs)):
        student.train()
        perm = torch.randperm(states.shape[0], device=device)
        epoch_loss = []
        for start in range(0, states.shape[0], int(args.batch_size)):
            batch = states[perm[start:start + int(args.batch_size)]]
            with torch.no_grad():
                t = teacher.forward_value_components(batch)
                scale = student.equity_value_scale(batch)
                target_v0 = t["V0_physical"] / scale.clamp_min(1e-12)
                target_vi = t["VI_physical"] / scale.clamp_min(1e-12)
            s = student.forward_value_components(batch)
            loss = F.smooth_l1_loss(s["V0_normalized"], target_v0) + F.smooth_l1_loss(s["VI_normalized"], target_vi)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss.append(float(loss.detach().item()))
        metrics = _eval_physical_mae(student, teacher, states)
        score = metrics["v0_physical_mae"] + metrics["vi_physical_mae"]
        if best is None or score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(epoch_loss)), **metrics})

    if best_state is not None:
        student.load_state_dict(best_state, strict=True)
    final_metrics = _eval_physical_mae(student, teacher, states)
    torch.save(student.state_dict(), output_dir / "scaled_value_ablation_policy_value.pt")
    (output_dir / "summary.json").write_text(
        json.dumps({"history": history, "final": final_metrics}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
