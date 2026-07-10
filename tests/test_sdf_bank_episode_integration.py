from types import SimpleNamespace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import Config  # noqa: E402
from config.hyperparams import HyperParams  # noqa: E402
from training.episode import Episode  # noqa: E402


def _minimal_episode() -> Episode:
    ep = Episode.__new__(Episode)
    ep.hyperparams = SimpleNamespace(
        sdf_fresh_pair_enabled=True,
        sdf_child_bank_size=4,
        sdf_signed_aio_n_children=2,
        sdf_child_bank_seed=123,
        sdf_child_bank_refresh_epochs=1,
        sdf_child_bank_wealth_only=True,
    )
    ep.device = torch.device("cpu")
    ep.episode_id = 0
    ep._current_epoch_idx = 0
    ep._sdf_shock_bank = None
    ep._sdf_shock_bank_n_parents = 0
    ep._sdf_shock_bank_epoch = None
    ep._sdf_shock_bank_episode_id = None
    ep._sdf_shock_bank_key = None
    ep._sdf_pair_generator = None
    ep.config = Config
    ep.add_FC1loss = False
    return ep


def test_batch_parent_index_is_compact_after_sparse_selection():
    ep = _minimal_episode()
    ep.hyperparams.max_firm_train_units = 3
    ep.hyperparams.pv_eta_resample_enabled = False
    parent = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    children = [parent.clone(), parent.clone()]

    batches = ep._build_batches_from_parent_children(parent, children, batch_size=2, eta_resample=False)
    compact = torch.cat([b["parent_index"] for b in batches])
    source = torch.cat([b["parent_source_index"] for b in batches])

    assert torch.equal(compact, torch.arange(compact.numel()))
    assert source.max().item() >= compact.max().item()


def test_lazy_sdf_bank_expands_for_larger_later_parent_index():
    ep = _minimal_episode()
    parent = torch.zeros(1, 7)

    ep._ensure_sdf_bank_capacity(torch.tensor([0]), dtype=parent.dtype, epoch=0)
    assert ep._sdf_shock_bank.eps.shape[0] == 1
    ep._ensure_sdf_bank_capacity(torch.tensor([3]), dtype=parent.dtype, epoch=0)
    assert ep._sdf_shock_bank.eps.shape[0] == 4


def test_reset_sdf_shock_bank_clears_episode_bank_state():
    ep = _minimal_episode()
    ep._ensure_sdf_bank_capacity(torch.tensor([2]), dtype=torch.float32, epoch=0)
    assert ep._sdf_shock_bank is not None

    ep.episode_id = 1
    ep.reset_sdf_shock_bank()

    assert ep._sdf_shock_bank is None
    assert ep._sdf_shock_bank_epoch is None
    assert ep._sdf_shock_bank_episode_id == 1


def test_formal_sdf_defaults_use_signed_aio_fresh_pairs():
    hp = HyperParams()

    assert hp.sdf_wealth_loss_mode == "signed_aio"
    assert hp.sdf_fresh_pair_enabled is True


def test_signed_aio_requires_fresh_pair_enabled():
    hp = HyperParams()
    hp.sdf_wealth_loss_mode = "signed_aio"
    hp.sdf_fresh_pair_enabled = False

    try:
        Episode(models={}, optimizers={}, config=Config, hyperparams=hp, device=torch.device("cpu"))
    except ValueError as exc:
        assert "signed_aio requires" in str(exc)
    else:
        raise AssertionError("signed_aio without fresh pair should raise")


if __name__ == "__main__":
    test_batch_parent_index_is_compact_after_sparse_selection()
    test_lazy_sdf_bank_expands_for_larger_later_parent_index()
    test_reset_sdf_shock_bank_clears_episode_bank_state()
    test_formal_sdf_defaults_use_signed_aio_fresh_pairs()
    test_signed_aio_requires_fresh_pair_enabled()
