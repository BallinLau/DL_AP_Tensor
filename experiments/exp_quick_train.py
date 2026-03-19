"""
Quick experimental runner for a tiny Episode-0 training on sample data.

This is meant as a smoke test: it builds fresh models, runs one short
`run_episode` on sample-generated data, and prints loss summaries.
"""

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config, HyperParams  # noqa: E402
from models import SDFFC1Combined, PolicyValueModel, FC2Model  # noqa: E402
from training.episode import Episode  # noqa: E402


def build_models(device: torch.device):
    """Instantiate core models."""
    models = {
        'sdf_fc1': SDFFC1Combined(
            sdf_input_dim=Config.FC1_INPUT_DIM,  # both SDF/FC1 take 4-d inputs here
            fc1_input_dim=Config.FC1_INPUT_DIM,
            sdf_hidden_dims=Config.SDF_HIDDEN_DIMS,
            fc1_hidden_dims=Config.FC1_HIDDEN_DIMS,
            w_hidden_dims=Config.SDF_HIDDEN_DIMS
        ).to(device),
        'policy_value': PolicyValueModel().to(device),
        'fc2': FC2Model(
            input_dim=Config.FC2_INPUT_DIM,
            hidden_dims=Config.FC2_HIDDEN_DIMS,
            quantile_num=Config.QUANTILE_NUM
        ).to(device)
    }
    return models


def build_optimizers(models, hyperparams: HyperParams | None = None):
    """Simple AdamW optimizers for each model."""
    hp = hyperparams or HyperParams()
    optimizers = {}
    for name, model in models.items():
        if name == 'sdf_fc1':
            optimizers[name] = torch.optim.AdamW(
                [
                    {
                        'params': model.sdf_model.parameters(),
                        'lr': hp.sdf_lr,
                        'base_lr': hp.sdf_lr,
                        'weight_decay': hp.sdf_weight_decay,
                        'group_name': 'sdf_core',
                    },
                    {
                        'params': model.value_model.parameters(),
                        'lr': hp.sdf_lr,
                        'base_lr': hp.sdf_lr,
                        'weight_decay': hp.sdf_weight_decay,
                        'group_name': 'value',
                    },
                    {
                        'params': model.fc1_model.parameters(),
                        'lr': hp.fc1_lr,
                        'base_lr': hp.fc1_lr,
                        'weight_decay': hp.fc1_weight_decay,
                        'group_name': 'fc1',
                    },
                ]
            )
        elif name == 'policy_value':
            optimizers[name] = torch.optim.AdamW(
                model.parameters(),
                lr=hp.policy_lr,
                weight_decay=hp.policy_weight_decay,
            )
        elif name == 'fc2':
            optimizers[name] = torch.optim.AdamW(
                model.parameters(),
                lr=hp.fc2_lr,
                weight_decay=hp.fc2_weight_decay,
            )
        else:
            optimizers[name] = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=1e-4)
    return optimizers


def build_hyperparams():
    """
    Episode expects some attributes not present in HyperParams by default;
    set them explicitly for this quick run.
    """
    hp = HyperParams(
        n_samples=2000,
        n_paths=200,
        simulate_horizon=5,
        batch_size=128,
        epochs=1,
    )
    # Required extras for schedulers/weights
    hp.lr = 1e-3
    hp.min_lr = 1e-6
    hp.warmup_steps = 0
    hp.max_steps = 10_000
    hp.w_sdf = 1.0
    hp.w_p0 = 1.0
    hp.w_pi = 1.0
    hp.w_q = 1.0
    hp.w_fc2 = 1.0
    return hp


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Config.DEVICE = device

    hyperparams = build_hyperparams()
    models = build_models(device)
    optimizers = build_optimizers(models, hyperparams)

    episode = Episode(
        models=models,
        optimizers=optimizers,
        config=Config,
        hyperparams=hyperparams,
        device=device,
        episode_id=0
    )

    summary = episode.run_episode(
        n_epochs=1,
        batch_size=128,
        log_interval=10,
        n_samples=hyperparams.n_samples,
        n_paths=hyperparams.n_paths,
        group_size=2,
        n_branches=2,
        train_modules=['policy_value'],  # change to include 'sdf_fc1', 'fc2' when ready
        simulate_kwargs=None
    )

    print("Episode summary:")
    print(summary['module_summaries'])


if __name__ == "__main__":
    main()
