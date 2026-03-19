# config

Central configuration for economic constants and training hyperparameters.

## constants.py
Defines economy- and model-level constants used across the codebase. Typical fields:
- Macros and shocks: `RHO_X`, `SIGMA_X`, `XBAR`, `RHO_Z`, `SIGMA_Z`, `ZBAR`
- Preferences and technology: `GAMMA`, `KAPPA`, `SIGMA`, `BETA`, `DELTA`, `PHI`, `G`
- Financing/frictions: `TAU`, `KAPPA_B`, `KAPPA_E`, `ZETA`
- Model sizes: feature dimensions and hidden layer defaults

These are treated as module-level constants under `Config`.

## hyperparams.py
Defines training hyperparameters via the `HyperParams` dataclass:
- Learning rates per module: `sdf_lr`, `fc1_lr`, `policy_lr`, `fc2_lr`
- Weight decay per module and gradient clipping thresholds
- Learning rate scheduling: plateau controls and cooldown options
- Training phases: stage-wise epochs for policy/value modules
- Data defaults: `n_samples`, `n_paths`, `simulate_horizon`

## Usage
Import `Config` for constants and `HyperParams` for training settings:

```python
from config import Config, HyperParams
hp = HyperParams()
```
