# Configuration Files

This directory contains all configuration files for the PINN project, organized by purpose.

## Directory Structure

```
configs/
├── data_generation/     # Data generation configurations
│   ├── data_generation.yaml        # Production config
│   ├── quick.yaml                 # Quick scale (100 trajectories, ~1 hour)
│   ├── moderate.yaml              # Moderate scale (250 trajectories, ~2.5 hours)
│   └── comprehensive.yaml         # Comprehensive scale (1000 trajectories, ~10 hours)
├── publication/         # Publication-ready configurations
│   ├── smib_delta20_omega40.yaml  # SMIB fixed Delta=20, Omega=40
│   └── kundur_2area.yaml          # Kundur 2-area, 4-machine
├── training/           # Training-only configurations
│   ├── trajectory_config.yaml
│   ├── parameter_config.yaml
│   └── cct_config.yaml
└── experiments/       # Full experiment pipeline configurations
    ├── baseline_trajectory.yaml
    ├── colab_quick.yaml
    ├── colab_moderate.yaml
    ├── colab_comprehensive.yaml
    ├── template.yaml
    └── ...
```

## Usage

### Data Generation Configs

**Three-Tier Test Configs** (following same pattern as experiment configs):

- **`quick.yaml`**: Quick scale (100 trajectories, ~1 hour) - Quick testing and validation
- **`moderate.yaml`**: Moderate scale (250 trajectories, ~2.5 hours) - Moderate testing and validation
- **`comprehensive.yaml`**: Comprehensive scale (1000 trajectories, ~10 hours) - Comprehensive testing and validation

**Production Config:**
- **`data_generation.yaml`**: Full production config (840+ samples, 2-4 hours)

**Usage:**

```bash
# Quick test (recommended for first test)
python scripts/quick_test_data.py --level quick

# Or directly:
python -m data_generation.parameter_sweep \
    --config configs/data_generation/quick.yaml \
    --output data/generated/quick_test

# Moderate test
python scripts/quick_test_data.py --level moderate

# Comprehensive test
python scripts/quick_test_data.py --level comprehensive

# Production data
python -m data_generation.parameter_sweep \
    --config configs/data_generation/data_generation.yaml \
    --output data/generated/full_dataset
```

### Publication Configs

- **`smib_delta20_omega40.yaml`**: SMIB with fixed loss scaling (Delta=20, Omega=40) for journal experiments.
- **`experiments/smib/pinn_ml_fair_loss_tune.yaml`**: SMIB; same `model` / `loss` / PINN `training` as `smib_delta20_omega40.yaml`; `ml_baseline` uses `pe_direct` (9-D, same as PINN) and publication-aligned ML loss knobs. Verify PINN YAML parity with `python scripts/verify_pinn_sections_match.py configs/publication/smib_delta20_omega40.yaml configs/experiments/smib/pinn_ml_fair_loss_tune.yaml`.
- **`kundur_2area.yaml`**: Kundur 2-area, 4-machine system. Use with multimachine data generation and `training/train_multimachine_pe_input.py` (full experiment pipeline multimachine support is optional).

```bash
# SMIB full experiment
python scripts/run_complete_experiment.py --config configs/publication/smib_delta20_omega40.yaml

# Kundur multimachine (same objective as SMIB, for 4-machine verification):
#   1. Generate data (one command):
#        python scripts/generate_multimachine_data.py --config configs/publication/kundur_2area.yaml --output data/multimachine/kundur
#   2. Preprocess and train:
#        python scripts/preprocess_data.py --data-path data/multimachine/kundur/parameter_sweep_data_*.csv ...
#        python training/train_multimachine_pe_input.py --data-dir <processed_dir> --num-machines 4
```

### Training Configs

For training individual models:

```bash
# Trajectory prediction
python training/train_trajectory.py --config configs/training/trajectory_config.yaml

# Parameter estimation
python training/train_parameters.py --config configs/training/parameter_config.yaml

# CCT estimation
python training/train_cct.py --config configs/training/cct_config.yaml
```

### Experiment Configs

For full experiment pipelines (data generation + training + evaluation):

```bash
python scripts/run_experiment.py --config configs/experiments/baseline_trajectory.yaml
```

## Notes

- **Data generation configs** (`data_generation/`): Settings for ANDES simulation and data extraction
- **Training configs** (`training/`): Model architecture, training hyperparameters, loss weights
- **Experiment configs** (`experiments/`): Complete pipeline configurations that combine data generation, training, and evaluation settings

