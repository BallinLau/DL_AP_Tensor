# DL-APSSH Codebase Analysis

## 1. Project Structure

- Technology stack:
  - Python single-repo project (detected manifest: `requirements.txt`).
  - Core runtime libs: `torch`, `numpy`, `pandas`, `tqdm`, `matplotlib`, `scipy`.
- Root layout (by responsibility):
  - `config/`: global economic constants and training hyperparameters.
  - `models/`: neural architectures (`sdf_fc1`, `policy_value`, `fc2`, shared heads/layers).
  - `losses/`: objective functions and FC2 aggregation pipeline.
  - `data/`: sample generator and time-series simulator.
  - `training/`: episode-level and multi-episode training orchestration.
  - `utils/`: logging, checkpoints, metrics, plotting.
  - `experiments/`: runnable experiment scripts and visualization artifact generation.
  - `tests/`: simulation-focused test and notebooks.
  - `skills/`: local skill metadata/docs (not runtime core).
  - `checkpoints/`, `cachedir/`: generated artifacts/cache (data/model outputs).
- Repository shape:
  - No multi-package/workspace manager detected.
  - Flat Python package-style modules under a single root.

## 2. Core Modules

- `config/constants.py` + `config/hyperparams.py`
  - Why core: highest fan-in; parameter source for most modules.
  - Evidence: internal import count shows `config` referenced most (`23` import occurrences across Python files).
- `training/episode.py`
  - Why core: central pipeline orchestrator for data generation, model forward, all loss computations, and stage execution.
  - Evidence: imported/instantiated by `training/trainer.py` and multiple experiment runners (`experiments/run_multi_episode.py`, `run_multi_episode_job.py`, `run_episode0_full.py`, `exp_quick_train.py`).
- `models/policy_value.py`, `models/sdf_fc1.py`, `models/fc2.py`
  - Why core: primary domain models used in both training and simulation.
  - Evidence: constructed in `main.py` and `experiments/run_utils.py`; consumed by episode training and `data/simulate_ts.py`.
- `data/sample.py` + `data/simulate_ts.py`
  - Why core: canonical data contracts feeding training stages.
  - Evidence: `training/episode.py` imports `Sample` and `SimulateTS` directly.
- `losses/sdf_loss.py`, `losses/p0_loss.py`, `losses/pi_loss.py`, `losses/q_loss.py`, `losses/fc2_loss.py`
  - Why core: objective layer defining optimization targets.
  - Evidence: `training/episode.py` imports all major loss classes.

## 3. Entry Files

- Runtime CLI entry:
  - `main.py`
  - Why entry: defines `argparse` arguments and `main()` dispatch (`train/eval/simulate`), guarded by `if __name__ == '__main__'`.
  - Invocation: `python main.py --mode train ...`
- Experiment/script entries:
  - `experiments/run_multi_episode.py`
  - `experiments/run_multi_episode_job.py`
  - `experiments/run_episode0_full.py`
  - `experiments/exp_quick_train.py`
  - `experiments/fill_fullN_entrants.py`
  - Why entry: each contains top-level `main()` flow and `__main__` guard; used for staged training, batch jobs, quick smoke runs, and post-processing.
  - Invocation: `python experiments/<script>.py [args]`
- Test-like executable entry:
  - `tests/test_simulate_ts.py`
  - Why entry: contains `main()` and `if __name__ == "__main__"` for direct execution.

## 4. Dependency Relationships

- External dependencies:
  - Runtime (declared): `torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `scipy` (`requirements.txt`).
  - Dev/test/tooling (observed, undeclared in manifest): `pytest` usage implied by `tests/` naming and previous pytest invocation.
- Internal dependency chains (dominant):
  - Training CLI chain:
    - `main.py` -> `training.Trainer` -> `training.Episode` -> (`data.Sample` / `data.SimulateTS`) + (`losses.*`) + (`models.*`).
  - Multi-episode experiment chain:
    - `experiments/run_multi_episode*.py` -> `experiments/run_utils.py` (build/save/plot helpers) -> `training.Episode` + `data.SimulateTS`.
  - Simulation chain:
    - `data/simulate_ts.py` -> `models.PolicyValueModel` + `models.SDFFC1Combined` + `data/data_utils.py`.
- Coupling and risk patterns (evidence-based):
  - High global coupling to `Config`: broad cross-module dependence on shared mutable globals.
  - Path-coupling via `sys.path.append(...)` appears in many files (`data/`, `models/`, `losses/`, `training/`, `experiments/`, `tests/`).
  - Interface drift risk in runtime entry path:
    - `main.py` references config/model args not found in current definitions (`SDF_INPUT_DIM`, `FIRM_STATE_DIM`, `FC2_N_QUANTILES`, and model ctor arg names).
    - `training/trainer.py` optimizer init references hyperparams fields not defined (`weight_decay`, `beta1`, `beta2`).
  - Assumption: given existing experiment scripts work around some constructor mismatches, day-to-day training likely relies more on `experiments/*.py` than `main.py`.
