"""Warm-start exp_xz-scaled equity value heads from a physical-value checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HyperParams
from models import PolicyValueModel
from analysis.economic_config import AnalysisEconomicConfig


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_hp_dict(payload: object) -> dict:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, HyperParams):
        return dict(payload.__dict__)
    raise ValueError("combined baseline checkpoint hyperparams must be a dict or HyperParams")


def _load_parent_states(path: str | None, n_states: int, seed: int) -> torch.Tensor:
    if path:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "parent" in payload:
            states = torch.as_tensor(payload["parent"], dtype=torch.float32)
        else:
            states = torch.as_tensor(payload, dtype=torch.float32)
        if states.shape[-1] < 7:
            raise ValueError("sample states must have at least seven firm-state columns")
        return states[:, :7].float()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    b = torch.rand(n_states, 1, generator=gen)
    z = -4.0 + 8.0 * torch.rand(n_states, 1, generator=gen)
    eta = torch.randint(0, 2, (n_states, 1), generator=gen).float()
    i = 0.5 * torch.rand(n_states, 1, generator=gen)
    x = -2.0 + 0.5 * torch.randn(n_states, 1, generator=gen)
    hatcf = torch.randn(n_states, 1, generator=gen)
    lnkf = 4.0 + torch.randn(n_states, 1, generator=gen)
    return torch.cat([b, z, eta, i, x, hatcf, lnkf], dim=1)


def _copy_non_value_modules(student: PolicyValueModel, teacher: PolicyValueModel) -> None:
    for name in ("q_encoder", "q_head", "policy_encoder", "bp0_head", "bpi_head"):
        getattr(student, name).load_state_dict(getattr(teacher, name).state_dict())


def _set_value_trainable(student: PolicyValueModel) -> None:
    for param in student.parameters():
        param.requires_grad = False
    for module in (student.value_encoder, student.v0_head, student.vi_head):
        for param in module.parameters():
            param.requires_grad = True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-policy-checkpoint", required=True)
    p.add_argument("--output-checkpoint", required=True)
    p.add_argument("--sample-states")
    p.add_argument("--n-states", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cpu")
    p.add_argument("--value-scale-log-max", type=float, default=20.0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    baseline_path = Path(args.baseline_policy_checkpoint)
    payload = torch.load(baseline_path, map_location=device)
    if not isinstance(payload, dict) or "models" not in payload or "policy_value" not in payload["models"]:
        raise ValueError("warm-start requires a combined baseline checkpoint with models.policy_value")
    source_value_meta = payload.get("value_parameterization") or {"mode": "none", "scale_formula": "1", "log_max": 20.0}
    if str(source_value_meta.get("mode", "none")).lower() != "none":
        raise ValueError("warm-start baseline must be mode='none'; scaled inputs must not be migrated again")
    if "hyperparams" not in payload:
        raise ValueError("combined baseline checkpoint is missing hyperparams")
    if "config_snapshot" not in payload:
        raise ValueError("combined baseline checkpoint is missing config_snapshot")

    teacher = PolicyValueModel(value_scale_mode="none").to(device)
    state = payload["models"]["policy_value"]
    teacher.load_state_dict(state, strict=True)
    teacher.eval().requires_grad_(False)

    student = PolicyValueModel(value_scale_mode="exp_xz", value_scale_log_max=args.value_scale_log_max).to(device)
    _copy_non_value_modules(student, teacher)
    _set_value_trainable(student)
    student.train()

    states = _load_parent_states(args.sample_states, args.n_states, args.seed).to(device)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(int(args.epochs)):
        perm = torch.randperm(states.shape[0], device=device)
        losses = []
        for start in range(0, states.shape[0], int(args.batch_size)):
            batch = states[perm[start:start + int(args.batch_size)]]
            with torch.no_grad():
                teacher_comp = teacher.forward_value_components(batch)
                scale = student.equity_value_scale(batch)
                target_v0 = teacher_comp["V0_physical"] / scale.clamp_min(1e-12)
                target_vi = teacher_comp["VI_physical"] / scale.clamp_min(1e-12)
            student_comp = student.forward_value_components(batch)
            loss = F.smooth_l1_loss(student_comp["V0_normalized"], target_v0) + F.smooth_l1_loss(
                student_comp["VI_normalized"], target_vi
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().item()))
        if epoch == 0 or (epoch + 1) == int(args.epochs) or (epoch + 1) % 25 == 0:
            print(f"epoch={epoch + 1} loss={np.mean(losses):.6g}")

    student.eval()
    with torch.no_grad():
        teacher_comp = teacher.forward_value_components(states)
        student_comp = student.forward_value_components(states)
        v0_mae = (student_comp["V0_physical"] - teacher_comp["V0_physical"]).abs().mean()
        vi_mae = (student_comp["VI_physical"] - teacher_comp["VI_physical"]).abs().mean()
        clamp_ratio = student.equity_value_scale_diagnostics(states)["value_scale_clamp_ratio"]
    print(f"physical_parity_v0_mae={float(v0_mae):.6g}")
    print(f"physical_parity_vi_mae={float(vi_mae):.6g}")
    print(f"value_scale_clamp_ratio={float(clamp_ratio):.6g}")

    hp = HyperParams()
    hp.pv_value_scale_mode = "exp_xz"
    hp.pv_value_scale_log_max = float(args.value_scale_log_max)
    hp.pv_bellman_normalize_by_value_scale = True
    out = Path(args.output_checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    models_payload = dict(payload["models"])
    models_payload["policy_value"] = student.state_dict()
    if "firm_target" not in models_payload:
        models_payload["firm_target"] = student.state_dict()
    torch.save(
        {
            **{k: v for k, v in payload.items() if k not in {"models", "hyperparams", "value_parameterization"}},
            "models": models_payload,
            "hyperparams": {**_as_hp_dict(payload.get("hyperparams")), **hp.__dict__},
            "config_snapshot": payload.get("config_snapshot", AnalysisEconomicConfig.from_current_config().to_dict()),
            "value_parameterization": {
                "mode": "exp_xz",
                "scale_formula": f"1+exp(clamp(x+z,max={float(args.value_scale_log_max):g}))",
                "bellman_normalization": True,
                "log_max": float(args.value_scale_log_max),
            },
            "warmstart": {
                "source": str(baseline_path),
                "source_sha256": _sha256(baseline_path),
                "seed": int(args.seed),
                "copied_modules": ["q_encoder", "q_head", "policy_encoder", "bp0_head", "bpi_head"],
                "trained_modules": ["value_encoder", "v0_head", "vi_head"],
            },
        },
        out,
    )
    print(f"saved={out}")


if __name__ == "__main__":
    main()
