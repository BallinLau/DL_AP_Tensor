from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT
    / "slurm"
    / "run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_"
    "investment_compare_fresh100.slurm"
)


def test_fresh100_wrapper_pins_the_ablation_contract():
    script = WRAPPER.read_text()

    required = (
        "#SBATCH -J dl-entry-fresh100",
        "dl_entry_fresh100_output.%j",
        "dl_entry_fresh100_error.%j",
        "normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_"
        "investment_compare_fresh100_${STAMP}",
        'export N_EPISODES="${N_EPISODES:-4}"',
        "export SIMULATE_GROUP_SIZE=200",
        "export SIMULATE_HORIZON=100",
        "export POST0_MODE=modeb",
        "export STATE_BANK_ENABLED=0",
        "export STATE_BANK_SDF_PUBLICATION_GATE_ENABLED=0",
        "export STATE_BANK_PQ_DECOUPLING_ENABLED=0",
        "export STATE_BANK_ECONOMIC_FEASIBILITY_FILTER_ENABLED=0",
        "export PV_MIXTURE_ENABLED=1",
        "export PV_MIXTURE_RATIO=0.20",
        "export SDF_CONTINUE_CONSTRAINT_TOL=0.02",
        "export TRANSITION_RNG_MODE=stable_firm_identity",
        "export COMMON_TRANSITION_SEED=424242",
        "export ENTRY_RNG_MODE=stable_economic_identity",
        "export PRESERVE_GLOBAL_RNG_AROUND_ENTRY=1",
        "export ENTRY_MODE=investment_compare",
        "export ENTRY_CAPITAL_RATIO=0.10",
        "export ENTRY_SIZE_RATIO=1.0",
        "export CONSUMPTION_AGGREGATION_MODE=raw",
        "export CONSUMPTION_FLOOR=1e-4",
        "export NODE_ACCOUNTING_MODE=economic_node_ledger",
        "export ENTRY_INDEPENDENT_RESOURCE_VALIDATION_ENABLED=0",
        "normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_20261008_184305",
        'export INITIAL_CHECKPOINT_EPISODE="${INITIAL_CHECKPOINT_EPISODE:-9}"',
        'export INITIAL_CHECKPOINT_MODE="${INITIAL_CHECKPOINT_MODE:-model_init}"',
        'export START_EPISODE="${START_EPISODE:-0}"',
        "run_normalized_sdf_aio_bp_logit_scaled_value_eta25_bp500_"
        "investment_compare.slurm",
    )
    for token in required:
        assert token in script

    unused_state_bank_controls = (
        "STATE_BANK_SHORT_HORIZON",
        "STATE_BANK_BOOTSTRAP_BURN_IN",
        "STATE_BANK_REFRESH_BURN_IN",
        "STATE_BANK_PARENT_TRANSITION_BUDGET",
        "STATE_BANK_MAX_ENTRIES",
    )
    for token in unused_state_bank_controls:
        assert token not in script


def test_existing_fresh_initializer_and_modeb_path_are_explicit():
    episode_source = (ROOT / "training" / "episode.py").read_text()
    simulator_source = (ROOT / "data" / "simulate_ts_parallel.py").read_text()

    assert "if self._state_bank_rollout_plan is None:" in episode_source
    assert "initial_state=initial_state" in episode_source
    assert "if initial_state is None:" in simulator_source
    assert "state = _initialize_batched_state(sim, max_firms)" in simulator_source
    assert "n0 = sim.group_size" in simulator_source
    assert "alive[:, :n0] = True" in simulator_source
