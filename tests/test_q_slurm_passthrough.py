"""Formal scaled-value Slurm chain must pass the direct-Q baseline knobs to Python.

The scaled-value chain is
``..._eta25_bp500.slurm -> ..._scaled_value_pilot.slurm -> run_normalized_sdf_aio_bp_logit.slurm``
and the wrappers only ``export`` environment variables. The base script therefore has to
forward ``Q_SHAPE_WEIGHT_*`` / ``Q_BOOTSTRAP_*`` / ``Q_REQUIRE_*`` as CLI flags.
"""

import re
from pathlib import Path

SLURM_DIR = Path(__file__).resolve().parents[1] / "slurm"
BASE_SLURM = SLURM_DIR / "run_normalized_sdf_aio_bp_logit.slurm"
PILOT_SLURM = SLURM_DIR / "run_normalized_sdf_aio_bp_logit_scaled_value_pilot.slurm"
RUN_JOB = Path(__file__).resolve().parents[1] / "experiments" / "run_multi_episode_job.py"

ENV_VARS = (
    "Q_SHAPE_WEIGHT_Z",
    "Q_SHAPE_WEIGHT_B_LOW",
    "Q_SHAPE_WEIGHT_B_HIGH",
    "Q_BOOTSTRAP_EPOCHS",
    "Q_BOOTSTRAP_MODE",
    "Q_BOOTSTRAP_UNIT_VALUE",
    "Q_BOOTSTRAP_NONNEGATIVE_WEIGHT",
    "Q_REQUIRE_ZERO_PHASE",
    "Q_REQUIRE_DEFAULT_PHASE",
    "Q_REQUIRE_SURVIVAL_PHASE",
)

CLI_FLAGS = (
    ("--q-shape-weight-z", "Q_SHAPE_WEIGHT_Z"),
    ("--q-shape-weight-b-low", "Q_SHAPE_WEIGHT_B_LOW"),
    ("--q-shape-weight-b-high", "Q_SHAPE_WEIGHT_B_HIGH"),
    ("--q-bootstrap-epochs", "Q_BOOTSTRAP_EPOCHS"),
    ("--q-bootstrap-mode", "Q_BOOTSTRAP_MODE"),
    ("--q-bootstrap-unit-value", "Q_BOOTSTRAP_UNIT_VALUE"),
    ("--q-bootstrap-nonnegative-weight", "Q_BOOTSTRAP_NONNEGATIVE_WEIGHT"),
    ("--q-require-zero-phase", "Q_REQUIRE_ZERO_PHASE"),
    ("--q-require-default-phase", "Q_REQUIRE_DEFAULT_PHASE"),
    ("--q-require-survival-phase", "Q_REQUIRE_SURVIVAL_PHASE"),
)

BASELINE_DEFAULTS = {
    "Q_SHAPE_WEIGHT_Z": "0",
    "Q_SHAPE_WEIGHT_B_LOW": "0",
    "Q_SHAPE_WEIGHT_B_HIGH": "0",
    "Q_BOOTSTRAP_EPOCHS": "5",
    "Q_BOOTSTRAP_MODE": "constant_unit",
    "Q_BOOTSTRAP_UNIT_VALUE": "1.0",
}


def _base_text() -> str:
    return BASE_SLURM.read_text(encoding="utf-8")


def test_T7_base_slurm_declares_every_q_env_var_with_a_default():
    text = _base_text()
    for name in ENV_VARS:
        assert f'{name}="${{{name}:-' in text, f"missing default assignment for {name}"


def test_T7b_base_slurm_echoes_effective_values():
    text = _base_text()
    for name in ENV_VARS:
        assert f'echo "  {name}=' in text, f"missing echo for {name}"


def test_T7c_base_slurm_forwards_every_var_to_the_python_cli():
    text = _base_text()
    for flag, var in CLI_FLAGS:
        assert re.search(rf'{re.escape(flag)}\s+"\${var}"', text), (flag, var)


def test_T7d_run_job_actually_accepts_the_forwarded_flags():
    src = RUN_JOB.read_text(encoding="utf-8")
    for flag, _ in CLI_FLAGS:
        assert f'"{flag}"' in src, f"run_multi_episode_job.py does not accept {flag}"


def test_T7e_defaults_are_the_direct_q_baseline():
    text = _base_text()
    for name, value in BASELINE_DEFAULTS.items():
        assert f'{name}="${{{name}:-{value}}}"' in text, (name, value)


def test_T7f_pilot_wrapper_still_overrides_through_the_environment():
    pilot = PILOT_SLURM.read_text(encoding="utf-8")
    assert "run_normalized_sdf_aio_bp_logit.slurm" in pilot
    # wrapper 只 export 环境变量，不透传 CLI；override 依赖 base 的 ${VAR:-default}。
    for _, var in CLI_FLAGS:
        assert f"--q-{var}" not in pilot
    assert "export PV_VALUE_SCALE_MODE" in pilot
