# Scripts Directory - Workflow Organization

> **Script documentation and workflow organization**  
> **📖 For experiment commands organized by type**, see: [Experiment Commands Guide](../docs/EXPERIMENT_COMMANDS.md)  
> **📖 For quick command lookup**, see: [Command Reference](../docs/COMMAND_REFERENCE.md)  
> **📖 For master experiments index**, see: [Experiments Index](../docs/EXPERIMENTS_INDEX.md)

This directory contains organized workflow scripts for the complete PINN project pipeline.

---

## 📋 Workflow Overview

### Individual Steps (Step-by-Step)
```
1. Generate Data     → 2. Verify Data  → 3. Analyze Data
                                              ↓
4. Preprocess Data    → 5. Train Model  → 6. Evaluate Model
   (Split train/val/test)
```

### Complete Workflow (All-in-One)
```
run_experiment.py:
1. Generate Data → 2. Analyze Data → 3. Preprocess Data → 4. Train Model → 5. Evaluate Model
   (All steps in one command, can skip any step)
```

**Key Points:**
- Each step can use explicit file paths for reproducibility
- Data can come from local generation, Colab, or online GPU
- Preprocessing step creates explicit train/val/test splits
- **Complete workflow** (`run_experiment.py`) now includes all steps:
  - ✅ Data generation
  - ✅ Data analysis (optional, generates statistics & figures)
  - ✅ Data preprocessing (optional, creates train/val/test splits)
  - ✅ Model training
  - ✅ Model evaluation
- Can load existing data with `--data-path` or `--data-dir`
- Can load existing models with `--model-path`
- See [Data Selection](#-data-selection) section for details

---

## 🔄 Step-by-Step Workflow Scripts

### 1. Data Generation

**Script**: `generate_data.py`

Generate training datasets using YAML configuration files.

**Usage:**
```bash
# Using pre-configured levels
python scripts/generate_data.py --level quick          # ~1 hour, 100 trajectories
python scripts/generate_data.py --level moderate        # ~2.5 hours, 250 trajectories
python scripts/generate_data.py --level comprehensive   # ~10 hours, 1000 trajectories

# Using custom config file
python scripts/generate_data.py --config configs/data_generation/my_config.yaml

# Custom output directory
python scripts/run_experiment.py --config configs/experiments/baseline_trajectory.yaml
```

**Config Files:**
- `configs/data_generation/quick.yaml`
- `configs/data_generation/moderate.yaml`
- `configs/data_generation/comprehensive.yaml`

**Output**: `data/common/trajectory_data_{n_samples}_{key_params}_{fingerprint}_{timestamp}.csv`

**Note**: The timestamp (YYYYMMDD_HHMMSS) ensures unique filenames. Use this full filename in subsequent steps for reproducibility.

---

**Script**: `run_load_level_simulation.py` ⭐ **NEW**

Run ANDES time domain simulations at different load levels for SMIB systems. Supports both batch data generation and single analysis runs.

**Usage:**
```bash
# Single simulation (analysis mode)
python scripts/run_load_level_simulation.py \\
    --case smib/SMIB.json \\
    --load 0.7 \\
    --mode analysis \\
    --plot

# Load level sweep (data generation mode) - saves to data/common/
python scripts/run_load_level_simulation.py \\
    --case smib/SMIB.json \\
    --load-range 0.4 0.9 10 \\
    --mode batch \\
    --task trajectory

# With generator parameters
python scripts/run_load_level_simulation.py \\
    --case smib/SMIB.json \\
    --load-range 0.4 0.9 10 \\
    --H 5.0 \\
    --D 1.0 \\
    --fault-start 1.0 \\
    --fault-clearing-times 1.15 1.18 1.20 1.22 1.25 \\
    --task trajectory
```

**Features:**
- ✅ Automatic load detection and addition
- ✅ Power flow convergence validation
- ✅ Voltage and generator capability checks
- ✅ Data quality and physics validation
- ✅ Stability detection
- ✅ Saves to `data/common/` with timestamped filenames
- ✅ Standardized output format matching existing pipeline
- ✅ Analysis plots with timestamped filenames

**Output Locations:**

**Batch Mode:**
- Data saved to: `data/common/trajectory_data_{n_samples}_{key_params}_{fingerprint}_{timestamp}.csv`
- Data is reusable across experiments via fingerprinting

**Analysis Mode:**
- Plots saved to: `outputs/analysis/` (default) or custom `--output` directory
- Filename format: `load_level_analysis_YYYYMMDD_HHMMSS.png`
- Example: `load_level_analysis_20251222_143022.png`
- Directory is created automatically if it doesn't exist

---

### 2. Data Verification

**Script**: `verify_data.py`

Verify generated data quality, check for missing trajectories, and validate CCT values.

**Usage:**
```bash
# Verify latest data file (defaults to quick_test)
python scripts/verify_data.py

# Verify specific file (recommended for reproducibility)
python scripts/verify_data.py data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv

# Verify file from different source (Colab, online GPU, etc.)
python scripts/verify_data.py data/generated/colab_data/parameter_sweep_data_YYYYMMDD_HHMMSS.csv
```

**Data Selection:**
- If no file specified, uses latest file in `data/generated/quick_test/`
- Can use explicit path to any file (local, Colab, etc.)
- See [Data Selection](#-data-selection) section below for details

**What it checks:**
- Total rows and columns
- Unique scenarios and parameter combinations
- Scenarios per combination (expected vs actual)
- Missing trajectories
- CCT values (unique, duplicate, missing)
- Stability distribution

---

### 3. Data Analysis

**Script**: `analyze_data.py`

Comprehensive dataset analysis for publication with statistical analysis and figures.

**Usage:**
```bash
# Analyze latest data file (defaults to quick_test)
python scripts/analyze_data.py

# Analyze specific file (recommended for reproducibility)
python scripts/analyze_data.py data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv

# Analyze by level (uses latest file from that level)
python scripts/analyze_data.py --level moderate
python scripts/analyze_data.py --level comprehensive

# Analyze file from different source (Colab, online GPU, etc.)
python scripts/analyze_data.py data/generated/colab_data/parameter_sweep_data_YYYYMMDD_HHMMSS.csv

# Analyze from custom directory
python scripts/analyze_data.py --data-dir data/generated/moderate_test

# Generate PDF figures too
python scripts/analyze_data.py --format png pdf

# Analyze experiment data and save figures to experiment directory
python scripts/analyze_data.py "outputs/experiments/exp_YYYYMMDD_HHMMSS/data/parameter_sweep_data_YYYYMMDD_HHMMSS.csv" \
    --output-dir "outputs/experiments/exp_YYYYMMDD_HHMMSS/analysis"
```

**Data Selection:**
- If no file specified, uses latest file in `data/generated/quick_test/`
- Use `--level` to select from quick/moderate/comprehensive
- Use `--data-dir` to specify custom directory
- Can use explicit path to any file (local, Colab, etc.)
- See [Data Selection](#-data-selection) section below for details

**Output:**
- Dataset statistics
- Trajectory-level analysis
- CCT correlation analysis
- Publication-ready figures (PNG/PDF)
- Summary report

**Figures Generated:**
- **Parameter Space Coverage** (`parameter_space_coverage_*.png`): 3D scatter plots showing H vs D, H vs Pm, D vs Pm
- **Parameter Distributions** (`parameter_distributions_*.png`): Histograms showing the distribution of H, D, and Pm parameters
- **Stable Trajectories** (`stable_trajectories_*.png`): Sample stable trajectory plots with rotor angle and speed
- **Unstable Trajectories** (`unstable_trajectories_*.png`): Sample unstable trajectory plots
- **CCT Distribution** (`cct_distribution_*.png`): Histogram of Critical Clearing Time values
- **CCT vs Parameters** (`cct_vs_parameters_*.png`): Correlation plots showing CCT relationship with H, D, Pm

**Output Location:**
- Default: `results/analysis/figures/` (if using default `--output-dir`)
- Custom: `{output_dir}/figures/` (if using `--output-dir` option)
- Experiment: `outputs/experiments/exp_XXX/analysis/figures/` (if analyzing experiment data)

**Note**: All figures are saved with timestamps in the filename (e.g., `parameter_space_coverage_20251207_143022.png`) for traceability.

---

### 4. Data Preprocessing & Splitting

**Script**: `preprocess_data.py`

Preprocess and split data into train/validation/test sets. This is a critical step that should be explicit.

**Usage:**
```bash
# Basic preprocessing and splitting (explicit file recommended)
python scripts/preprocess_data.py --data-path data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv

# Using file from Colab/online GPU
python scripts/preprocess_data.py --data-path data/generated/colab_data/parameter_sweep_data_YYYYMMDD_HHMMSS.csv

# Custom split ratios
python scripts/preprocess_data.py \
    --data-path data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

# With feature engineering
python scripts/preprocess_data.py \
    --data-path data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv \
    --engineer-features
```

**What it does:**
- Loads data from `generate_data.py` output
- Optionally engineers features
- Splits into train/val/test sets (default: 70/15/15)
- Optionally normalizes data (or let training handle it)
- Saves splits and metadata

**Output:**
- `data/preprocessed/{level}_test/train_data_YYYYMMDD_HHMMSS.csv` - Training set (70% by default)
- `data/preprocessed/{level}_test/val_data_YYYYMMDD_HHMMSS.csv` - Validation set (15% by default)
- `data/preprocessed/{level}_test/test_data_YYYYMMDD_HHMMSS.csv` - Test set (15% by default)
- `data/preprocessed/{level}_test/preprocessing_metadata_YYYYMMDD_HHMMSS.json` - Split metadata

**Important Notes:**
- ✅ **Recommended**: Use this step for explicit, reproducible train/val/test splits
- ⚠️ **Alternative**: If you skip this step, `train_model.py` will split data automatically (less reproducible)
- 📝 **Best Practice**: Use the train set for training, val set for validation during training, test set for final evaluation

**Note**: Always use explicit file paths (with timestamps) for reproducibility. The `--data-path` argument is required.

---

### 5. Model Training

**Script**: `train_model.py`

Train PINN models on generated data with flexible configuration.

**Key Features:**
- ✅ **Early stopping**: Automatically stops training if validation loss plateaus
- ✅ **Angle penalty**: Prevents unrealistic rotor angle predictions
- ✅ **Time tracking**: Shows elapsed time and ETA during training
- ✅ **Adaptive batch size**: Automatically calculates optimal batch size
- ✅ **Adaptive loss weights**: Dynamically adjusts physics loss weight
- ✅ **Learning rate scheduling**: Automatically reduces LR when validation loss plateaus

**Usage:**
```bash
# Basic training with preprocessed data (reactance-based, default)
python scripts/train_model.py \
    --data-path data/preprocessed/quick_test/train_data_YYYYMMDD_HHMMSS.csv \
    --epochs 100

# Pe-based approach (7 input dimensions)
python scripts/train_model.py \
    --data-path data/preprocessed/quick_test/train_data_YYYYMMDD_HHMMSS.csv \
    --input-method pe_direct \
    --epochs 100

# With data directory (uses latest file)
python scripts/train_model.py \
    --data-dir data/preprocessed/quick_test \
    --epochs 100

# Custom training parameters (with manual batch size)
python scripts/train_model.py \
    --data-path data/preprocessed/quick_test/train_data_YYYYMMDD_HHMMSS.csv \
    --input-method pe_direct \
    --epochs 200 \
    --batch-size 12 \
    --learning-rate 0.0005

# Automatic batch size (default - recommended)
python scripts/train_model.py \
    --data-path data/preprocessed/quick_test/train_data_YYYYMMDD_HHMMSS.csv \
    --input-method pe_direct \
    --epochs 100
    # Batch size will be auto-calculated from dataset size

# Quick test (1 epoch)
python scripts/train_model.py \
    --data-dir data/preprocessed/quick_test \
    --input-method pe_direct \
    --epochs 1
```

**Training Configuration:**

#### Input Method Selection

Choose between two input approaches:

1. **Pe-based** (default, `--input-method pe_direct`):
   - **Input dimensions**: 7
   - **Input features**: `[t, δ₀, ω₀, H, D, Pm, Pe(t)]`
   - **Use when**: You have Pe(t) measurements and want direct power input (recommended)
   - **Model**: `TrajectoryPredictionPINN_PeInput`
   - **Default**: Yes (both PINN and ML baseline use this by default)

2. **Reactance-based** (optional, `--input-method reactance`):
   - **Input dimensions**: 11
   - **Input features**: `[t, δ₀, ω₀, H, D, Pm, Xprefault, Xfault, Xpostfault, tf, tc]`
   - **Use when**: You have reactance values and want physics-based input
   - **Model**: `TrajectoryPredictionPINN`

#### Batch Size Configuration

Batch size refers to **number of scenarios per batch** (not data points). 

**Automatic Batch Size (Default):**
- If `--batch-size` is not specified, batch size is **automatically calculated** from your dataset size
- Target: 6-8 batches per epoch for optimal learning
- The script will print the calculated batch size and batches per epoch when training starts
- Example output: `✓ Adaptive batch size: 12 (calculated from 69 scenarios) → 5.8 batches per epoch`

**Manual Batch Size:**
- You can override with `--batch-size <number>` to use a specific batch size
- Choose based on your dataset size using the table below:

| Dataset Size          | Recommended Batch Size | Batches/Epoch | Notes                                 |
| --------------------- | ---------------------- | ------------- | ------------------------------------- |
| **< 50 scenarios**    | 4-8                    | 5-12          | Small datasets need more batches      |
| **50-100 scenarios**  | 8-16                   | 5-12          | Balanced for medium datasets          |
| **100-200 scenarios** | 16-24                  | 5-12          | Good for larger datasets              |
| **> 200 scenarios**   | 24-32                  | 8-15          | Large datasets can use larger batches |

**Guidelines:**
- **Target**: 4-12 batches per epoch for optimal learning
- **Too few batches** (< 3): Slower convergence, fewer gradient updates
- **Too many batches** (> 15): May slow training, but generally acceptable
- **Formula**: `batch_size ≈ num_scenarios / desired_batches_per_epoch`

**Example**: With 69 training scenarios:
- Batch size 32 → ~2 batches/epoch ❌ (too few)
- Batch size 12 → ~6 batches/epoch ✅ (good)
- Batch size 8 → ~9 batches/epoch ✅ (excellent)

#### Epoch Configuration

- **Quick test**: `--epochs 1` (verify pipeline works)
- **Quick training**: `--epochs 50-100` (for small datasets or prototyping)
- **Standard training**: `--epochs 100-200` (for most cases)
- **Full training**: `--epochs 200-500` (for publication-quality results)

#### Learning Rate

- **Default**: `0.001` (works well for most cases)
- **Lower**: `0.0005` (if training is unstable or loss oscillates)
- **Higher**: `0.002` (may speed up but risk instability)
- **Note**: Learning rate scheduler automatically reduces LR when validation loss plateaus

#### Early Stopping

Training includes automatic early stopping to prevent overfitting and save time:

- **Configuration**: Set `early_stopping_patience` in your config file (default: `200`)
  ```yaml
  training:
    early_stopping_patience: 200  # Stop if no improvement for 200 epochs
  ```
- **Behavior**: Training stops automatically if validation loss doesn't improve for the specified number of epochs
- **Default**: `200` epochs (disabled if set to `null`)
- **Output**: Shows best validation loss and the epoch where it occurred
- **Example**: If `early_stopping_patience: 50`, training stops after 50 epochs without improvement

#### Angle Penalty

The training includes an angle penalty to prevent unrealistic rotor angle predictions:

- **Configuration**: Set in your config file:
  ```yaml
  training:
    max_training_angle_degrees: 720.0  # Maximum angle in degrees
    lambda_angle: 0.1  # Weight for angle penalty
  ```
- **Purpose**: Penalizes predictions where rotor angle exceeds physical limits
- **Default**: `720.0` degrees maximum angle, `0.1` penalty weight
- **How it works**: Adds a penalty term to the loss function when predicted angles exceed the limit

#### Time Tracking

Training now includes detailed time tracking with ETA (Estimated Time to Arrival):

- **Display**: Shows time information every 10 epochs:
  ```
  ⏱️  Time: Epoch=2m 15s | Elapsed=5m 30s | Remaining≈45m
  ```
- **Information provided**:
  - **Epoch time**: Time taken for the current epoch
  - **Elapsed time**: Total time since training started
  - **Remaining time**: Estimated time to complete all epochs (based on average epoch time)
- **Final summary**: After training completes, shows:
  - Total training time
  - Average time per epoch
  - Best validation loss and epoch number

**Note**: 
- Use preprocessed train data (from `preprocess_data.py`) for explicit splits
- If using raw data, training will split automatically (less reproducible)
- Batch size should be adjusted based on your dataset size (see table above)

**Output:**
- Trained model: `outputs/experiments/exp_*/model/best_model_*.pth`
- Training history: `outputs/experiments/exp_*/model/training_history_*.json`
- Loss curves: `outputs/experiments/exp_*/results/figures/`

---

### 6. Model Evaluation

**Script**: `evaluate_model.py`

Evaluate trained models on test data and generate performance metrics.

**Usage:**
```bash
# Evaluate on test set (recommended)
python scripts/evaluate_model.py \
    --model-path outputs/experiments/exp_*/model/best_model_*.pth \
    --data-path data/preprocessed/quick_test/test_data_YYYYMMDD_HHMMSS.csv

# Evaluate on full dataset
python scripts/evaluate_model.py \
    --model-path outputs/experiments/exp_*/model/best_model_*.pth \
    --data-path data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv

# Evaluate more scenarios
python scripts/evaluate_model.py \
    --model-path outputs/experiments/exp_*/model/best_model_*.pth \
    --data-path data/preprocessed/quick_test/test_data_YYYYMMDD_HHMMSS.csv \
    --n-scenarios 20
```

**Note**: Use test set from preprocessing step for proper evaluation on unseen data.

**Output:**
- Performance metrics (RMSE, MAE for delta and omega)
- Comparison plots (predicted vs true)
- Metrics CSV: `results/evaluation/evaluation_metrics_*.csv`

---

### 7. ML Baseline Evaluation

**Script**: `evaluate_ml_baseline.py`

Evaluate trained ML baseline models (StandardNN, LSTM) on test data and generate scenario visualizations.

**Note**: ML baseline models now save configuration files (`config.yaml`) similar to PINN, including:
- Model architecture configuration
- Training hyperparameters (epochs, learning rate, weight decay)
- Loss weights (`lambda_data=1.0`, `lambda_ic=10.0`, `lambda_physics=None`)
- Input method and data paths
- Experiment metadata

**Usage:**
```bash
# Basic evaluation
python scripts/evaluate_ml_baseline.py \
    --model-path outputs/ml_baselines/exp_20251215_111443/standard_nn/model.pth \
    --test-data data/common/full_trajectory_data_30_*.csv \
    --output-dir outputs/ml_baselines/exp_20251215_111443/standard_nn/evaluation

# With pre-split test data (recommended for reproducibility)
python scripts/evaluate_ml_baseline.py \
    --model-path outputs/ml_baselines/exp_20251215_111443/standard_nn/model.pth \
    --test-data data/common/full_trajectory_data_30_*.csv \
    --test-split-path outputs/experiments/exp_20251208_234830/data/val_data_*.csv \
    --output-dir outputs/ml_baselines/exp_20251215_111443/standard_nn/evaluation
```

**Output:**
- `metrics.json` - Trajectory metrics (RMSE, R², MAPE, etc.)
- `test_scenario_ids.json` - Test scenario IDs for reproducibility
- `figures/test_scenarios_predictions_*.png` - Scenario visualizations

**See**: `docs/guides/ML_BASELINE_EVALUATION_GUIDE.md` for detailed documentation.

---

### 8. Model Comparison

**Script**: `compare_models.py`

Compare ML baseline vs PINN vs ANDES ground truth with statistical analysis and overlaid trajectory plots.

**Usage:**
```bash
# Pre-split test CSV (recommended). Omit --full-trajectory-data unless you need a separate combined trajectory file.
python scripts/compare_models.py \
    --ml-baseline-model outputs/ml_baselines/exp_20251215_111443/standard_nn/model.pth \
    --pinn-model outputs/experiments/exp_20251208_234830/model.pth \
    --test-split-path outputs/experiments/exp_20251208_234830/data/test_data_*.csv \
    --output-dir outputs/comparison/exp_20251215_120000

# With PINN config and optional combined trajectory (when test CSV is not enough for full trajectories)
python scripts/compare_models.py \
    --ml-baseline-model outputs/ml_baselines/exp_20251215_111443/standard_nn/model.pth \
    --pinn-model outputs/experiments/exp_20251208_234830/model.pth \
    --pinn-config outputs/experiments/exp_20251208_234830/config.json \
    --test-split-path outputs/experiments/exp_20251208_234830/data/test_data_*.csv \
    --full-trajectory-data data/common/full_trajectory_data_30_*.csv \
    --output-dir outputs/comparison/exp_20251215_120000
```

**Output:**
- `comparison_results.json` - Statistical comparison (mean, std, CI, p-values)
- `test_scenario_ids.json` - Test scenario IDs for reproducibility
- `figures/model_comparison_overlaid_*.png` - Overlaid trajectory plots

**Key Features:**
- ✅ Pre-split test CSV via **`--test-split-path`** (recommended); optional **`--full-trajectory-data`** / **`--test-data`** for combined trajectory or on-the-fly split
- ✅ Ensures same test scenarios for fair comparison
- ✅ Statistical significance tests (paired/independent t-tests)
- ✅ 95% confidence intervals
- ✅ Relative improvement percentages
- ✅ Overlaid plots (ANDES, ML Baseline, PINN on same axes)
- ✅ Delta-only figures default to **separate** stable/unstable PNGs (when `is_stable` is in test data); use **`--combine-delta-only-figure`** for one combined figure

**See**: `docs/guides/ML_BASELINE_COMPARISON_GUIDE.md` for detailed documentation.

---

## 🚀 Complete Experiment Workflow

### Option 1: PINN Only Workflow

**Script**: `run_experiment.py`

Run complete PINN pipeline: data generation → data analysis → preprocessing → training → evaluation in one command.

### Option 2: Complete Comparison Workflow (PINN + ML Baselines)

**Script**: `run_complete_experiment.py`

Run complete pipeline with both PINN and ML baseline models: data generation → data analysis → PINN training → ML baseline training → evaluation → comparison in one command.

**Usage:**
```bash
# Full workflow (generate data, train both models, evaluate, compare)
# ⭐ RECOMMENDED: Use hyperparameter_tuning.yaml for publication-quality experiments
python scripts/run_complete_experiment.py \
    --config configs/experiments/hyperparameter_tuning.yaml \
    --pinn-epochs 400 \
    --ml-baseline-epochs 200

# Quick test workflow (faster, smaller dataset)
python scripts/run_complete_experiment.py \
    --config configs/experiments/quick.yaml \
    --pinn-epochs 50 \
    --ml-baseline-epochs 50

# Reuse existing data
python scripts/run_complete_experiment.py --config config.yaml \
    --skip-data-generation --data-dir path/to/data

# Reuse existing models (skip training, only evaluate and compare)
python scripts/run_complete_experiment.py --config config.yaml \
    --skip-data-generation --data-dir path/to/data \
    --skip-pinn-training --pinn-model-path outputs/experiments/exp_XXX/model/best_model_*.pth \
    --skip-ml-baseline-training --ml-baseline-model-path outputs/ml_baselines/exp_XXX/standard_nn/model.pth

# Train multiple ML baseline models
python scripts/run_complete_experiment.py --config config.yaml \
    --ml-baseline-models standard_nn,lstm

# Skip specific steps
python scripts/run_complete_experiment.py --config config.yaml \
    --skip-data-analysis \
    --skip-pinn-evaluation \
    --skip-ml-baseline-evaluation \
    --skip-comparison
```

**Arguments:**
- `--config` (required): YAML config file path
- `--skip-data-generation`: Reuse existing data (requires `--data-dir` or `--data-path`)
- `--data-dir`: Path to existing data directory
- `--data-path`: Direct path to data file
- `--skip-data-analysis`: Skip ANDES data analysis
- `--epochs`: Override epochs for both models
- `--pinn-epochs`: Override epochs for PINN only
- `--ml-baseline-epochs`: Override epochs for ML baseline only
- `--skip-pinn-training`: Skip PINN training (use `--pinn-model-path`)
- `--pinn-model-path`: Path to existing PINN model (supports wildcards)
- `--skip-pinn-evaluation`: Skip PINN evaluation
- `--skip-ml-baseline-training`: Skip ML baseline training (use `--ml-baseline-model-path`)
- `--ml-baseline-model-path`: Path to existing ML baseline model
- `--ml-baseline-models`: ML baseline models to train (default: `standard_nn`, comma-separated)
- `--skip-ml-baseline-evaluation`: Skip ML baseline evaluation
- `--skip-comparison`: Skip model comparison
- `--output-dir`: Base output directory (default: `outputs/complete_experiments`)

**Output Structure:**
```
outputs/complete_experiments/exp_YYYYMMDD_HHMMSS/
├── config.yaml
├── experiment_summary.json
├── data/
├── analysis/
├── pinn/
│   ├── model/
│   └── evaluation/
├── ml_baseline/
│   ├── standard_nn/
│   └── lstm/ (if trained)
└── comparison/
    └── figures/
```

---

**Script**: `run_experiment.py` (PINN only)

Run complete PINN pipeline: data generation → data analysis → preprocessing → training → evaluation in one command.

### 📋 Quick Reference

| Scenario                                  | Command                                                                                                                            | `--data-dir` needed? | `--model-path` needed? |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------- |
| **Full experiment** (generate everything) | `python scripts/run_experiment.py --config configs/experiments/comprehensive.yaml`                                                 | ❌ No                 | ❌ No                   |
| **Use existing data** (skip generation)   | `python scripts/run_experiment.py --config ... --skip-data-generation --data-dir "path/to/data"`                                   | ✅ **Yes**            | ❌ No                   |
| **Figures only** (skip training/eval)     | `python scripts/run_experiment.py --config ... --skip-data-generation --skip-training --skip-evaluation --data-dir "path/to/data"` | ✅ **Yes**            | ❌ No                   |
| **Evaluation only** (skip training)       | `python scripts/run_experiment.py --config ... --skip-training --model-path "path/to/model.pth"`                                   | ❌ No*                | ✅ **Yes**              |

\* Unless you also use `--skip-data-generation`, then `--data-dir` is required.

**Usage:**

**⚠️ Important: When to use `--data-dir`:**
- ✅ **Full experiment (no skips)**: `--data-dir` is **NOT needed** - data will be generated automatically
- ✅ **Skip data generation**: `--data-dir` **IS REQUIRED** - must specify where existing data is located
- ✅ **Skip training + evaluation (figures only)**: `--data-dir` **IS REQUIRED**, but `--model-path` is **NOT needed**

```bash
# Full experiment with config file (all steps) - NO --data-dir needed!
python scripts/run_experiment.py --config configs/experiments/comprehensive.yaml

# Skip data analysis (faster, but less insights)
python scripts/run_experiment.py --config configs/experiments/baseline_trajectory.yaml \
    --skip-data-analysis

# Skip preprocessing (training will do internal split)
python scripts/run_experiment.py --config configs/experiments/baseline_trajectory.yaml \
    --skip-preprocessing

# Use existing data (from directory)
python scripts/run_experiment.py \
    --config configs/experiments/baseline_trajectory.yaml \
    --skip-data-generation \
    --data-dir data/generated/quick_test

# Use existing data (direct file path)
python scripts/run_experiment.py \
    --config configs/experiments/baseline_trajectory.yaml \
    --skip-data-generation \
    --data-path data/generated/quick_test/parameter_sweep_data_20251205_170908.csv

# Use existing model (skip training)
python scripts/run_experiment.py \
    --config configs/experiments/baseline_trajectory.yaml \
    --skip-training \
    --model-path outputs/experiments/exp_20251205_170908/model/best_model_20251205_170908.pth

# Use existing data and model (evaluation only)
python scripts/run_experiment.py \
    --config configs/experiments/baseline_trajectory.yaml \
    --skip-data-generation \
    --skip-data-analysis \
    --skip-preprocessing \
    --data-path data/preprocessed/quick_test/test_data_20251205_170908.csv \
    --skip-training \
    --model-path outputs/experiments/exp_20251205_170908/model/best_model_20251205_170908.pth

# Override epochs from command line (useful for quick tests)
python scripts/run_experiment.py \
    --config configs/experiments/quick.yaml \
    --skip-data-generation \
    --data-path data/generated/quick_test/parameter_sweep_data_20251205_170908.csv \
    --epochs 100

# Use quick config with existing data
python scripts/run_experiment.py \
    --config configs/experiments/quick.yaml \
    --skip-data-generation \
    --data-dir data/generated/quick_test \
    --epochs 100

# Generate figures only from existing data (no training/evaluation)
python scripts/run_experiment.py \
    --config configs/experiments/comprehensive.yaml \
    --skip-data-generation \
    --skip-training \
    --skip-evaluation \
    --data-dir "outputs/experiments/exp_YYYYMMDD_HHMMSS/data"
    # Note: Model path not required when both training and evaluation are skipped
```

**What it does:**
1. Loads configuration
2. Generates/loads training data
3. **Analyzes data** (optional, generates statistics and figures)
4. **Preprocesses and splits data** (optional, creates train/val/test sets)
5. Trains model
6. Evaluates on test set
7. Saves all results with experiment ID

**Command-Line Options:**

| Option                   | Description                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| `--config`               | Path to experiment configuration YAML file (required)                                       |
| `--output-dir`           | Base directory for experiment outputs (default: `outputs/experiments`)                      |
| `--epochs`               | Number of training epochs (overrides config file setting)                                   |
| `--skip-data-generation` | Skip data generation (**requires `--data-dir` or `--data-path`**)                           |
| `--skip-data-analysis`   | Skip data analysis step                                                                     |
| `--skip-preprocessing`   | Skip data preprocessing/splitting                                                           |
| `--skip-training`        | Skip training (requires `--model-path` **unless `--skip-evaluation` is also used**)         |
| `--skip-evaluation`      | Skip evaluation                                                                             |
| `--data-dir`             | Directory with existing data (**required when `--skip-data-generation` is used**)           |
| `--data-path`            | Direct path to data file (overrides `--data-dir` if both provided)                          |
| `--model-path`           | Path to existing model file (required if `--skip-training` **and NOT `--skip-evaluation`**) |

**Output Structure:**
```
outputs/experiments/exp_YYYYMMDD_HHMMSS/
├── config.yaml                    # Exact config used
├── data/                          # Generated data
│   ├── parameter_sweep_data_*.csv
│   └── preprocessed/              # Preprocessed splits (if preprocessing enabled)
│       ├── train_data_*.csv
│       ├── val_data_*.csv
│       ├── test_data_*.csv
│       └── preprocessing_metadata_*.json
├── analysis/                      # Data analysis results (if analysis enabled)
│   ├── figures/                   # Analysis figures (6 total)
│   │   ├── parameter_space_coverage_*.png
│   │   ├── parameter_distributions_*.png
│   │   ├── stable_trajectories_*.png
│   │   ├── unstable_trajectories_*.png
│   │   ├── cct_distribution_*.png
│   │   └── cct_vs_parameters_*.png
│   ├── trajectory_statistics_*.csv
│   └── analysis_summary_report_*.txt
├── model/                         # Trained model
│   ├── best_model_*.pth
│   ├── training_history_*.json
│   └── scalers (in checkpoint)
└── results/                       # Evaluation results
    ├── figures/                   # Evaluation figures
    └── metrics_*.json
```

**Generating Figures from Existing Data:**

If you have already generated data and only want to create analysis figures, you have two options:

**Option 1: Using `analyze_data.py` (Recommended - Simpler)**
```bash
# Analyze data and generate figures
python scripts/analyze_data.py "outputs/experiments/exp_YYYYMMDD_HHMMSS/data/parameter_sweep_data_YYYYMMDD_HHMMSS.csv" \
    --output-dir "outputs/experiments/exp_YYYYMMDD_HHMMSS/analysis"
```
This will create figures in: `outputs/experiments/exp_YYYYMMDD_HHMMSS/analysis/figures/`

**Option 2: Using `run_experiment.py` (Full workflow, figures only)**
```bash
# Generate figures using experiment workflow (creates new experiment directory)
python scripts/run_experiment.py \
    --config configs/experiments/comprehensive.yaml \
    --skip-data-generation \
    --skip-training \
    --skip-evaluation \
    --data-dir "outputs/experiments/exp_YYYYMMDD_HHMMSS/data"
```
This will create a new experiment directory with figures in: `outputs/experiments/exp_NEW_ID/analysis/figures/`

**Important Notes:**
- When using `--skip-training` and `--skip-evaluation` together, the `--model-path` argument is **NOT required**
- When using `--skip-data-generation`, you **MUST** provide `--data-dir` or `--data-path`
- For a full experiment (no skips), you **DO NOT** need `--data-dir` - data will be generated automatically

---

## 🔧 Utility Scripts

### Compare Experiments

**Script**: `compare_experiments.py`

Compare multiple experiment results and generate comparison reports.

**Usage:**
```bash
python scripts/compare_experiments.py \
    --experiments outputs/experiments/exp_* \
    --metric val_loss \
    --output-dir outputs/comparisons
```

### Statistical Validation ⭐ **NEW - Publication Quality**

**Script**: `run_statistical_validation.py`

Run multiple experiments with different random seeds to assess reproducibility and compute confidence intervals for metrics. Essential for publication-quality research.

**Usage:**
```bash
# Recommended: Use exp_20251208_234830 (balanced performance configuration)
python scripts/run_statistical_validation.py \
    --config configs/experiments/hyperparameter_tuning.yaml \
    --data-path outputs/experiments/exp_20251208_234830/data/preprocessed/train_data_*.csv \
    --seeds 0 1 2 3 4 \
    --output-dir outputs/statistical_validation

# Alternative: Auto-find latest preprocessed data
python scripts/run_statistical_validation.py \
    --config configs/experiments/hyperparameter_tuning.yaml \
    --skip-data-generation \
    --seeds 0 1 2 3 4 \
    --output-dir outputs/statistical_validation

# Single line (PowerShell)
python scripts/run_statistical_validation.py --config configs/experiments/hyperparameter_tuning.yaml --data-path outputs/experiments/exp_20251208_234830/data/preprocessed/train_data_*.csv --seeds 0 1 2 3 4 --output-dir outputs/statistical_validation
```

**Recommended Experiment: `exp_20251208_234830`**
- **Configuration**: n_samples=30, epochs=500, architecture=[256,256,128,128], λ_physics=0.5 (Fixed)
- **Performance**: R² Delta = 0.866, R² Omega = 0.587
- **Why Recommended**: This is the **Balanced Performance** configuration (Option 2) from `COMPREHENSIVE_COMPARISON_TABLE.md`, providing excellent performance on both delta and omega predictions. Ideal for demonstrating reproducibility.

**What it does:**
1. Runs 5 separate experiments with different random seeds (0, 1, 2, 3, 4)
2. Trains models with identical configuration but different initialization
3. Computes statistical summary: mean, std, median, min, max, 95% confidence intervals
4. Generates visualizations: box plots, confidence intervals, distributions
5. Saves results to `outputs/statistical_validation/`

**Output Files:**
- `raw_results.csv` - Individual experiment results
- `statistical_summary.json` - Aggregated statistics with confidence intervals
- `figures/box_plots.png` - Metric distributions
- `figures/confidence_intervals.png` - CI plots
- `figures/distributions.png` - Histograms

**Expected Output:**
```
STATISTICAL VALIDATION SUMMARY
Number of successful runs: 5/5

Metrics Summary:
  delta_r2:
    Mean: 0.866000
    Std:  0.003000
    95% CI: [0.863000, 0.869000]
  omega_r2:
    Mean: 0.587000
    Std:  0.005000
    95% CI: [0.582000, 0.592000]
```

**Command-Line Options:**

| Option                   | Description                                                                     |
| ------------------------ | ------------------------------------------------------------------------------- |
| `--config`               | Experiment configuration file (required)                                        |
| `--data-path`            | Path to training data CSV file (supports wildcards: `train_data_*.csv`)         |
| `--seeds`                | Random seeds to use (default: `0 1 2 3 4`)                                      |
| `--output-dir`           | Output directory for results (default: `outputs/statistical_validation`)        |
| `--skip-data-generation` | Skip data generation (use existing data, auto-finds latest if no `--data-path`) |

**Best Practices:**
- ✅ Use the recommended experiment (`exp_20251208_234830`) for balanced performance
- ✅ Run with 5 seeds minimum for publication quality
- ✅ Use wildcard patterns (`train_data_*.csv`) to automatically find latest file
- ✅ Check `statistical_summary.json` for confidence intervals
- ✅ Include results in paper: "R² Delta = 0.866 ± 0.003 (95% CI [0.863, 0.869])"

**See Also:**
- `COMPREHENSIVE_COMPARISON_TABLE.md` - Full comparison of all experiments
- `docs/publication/STEP_BY_STEP_PUBLICATION_PLAN.md` - Complete publication guide
- `docs/publication/QUICK_REFERENCE_COMMANDS.md` - Quick command reference

---

### Hyperparameter Sweep

**Script**: `run_hyperparameter_sweep.py`

Run multiple experiments with different combinations of hyperparameters (n_samples and epochs) to find optimal settings.

**Usage:**
```bash
# Basic sweep: test different n_samples and epochs
python scripts/run_hyperparameter_sweep.py \
    --n-samples-range 20 30 40 \
    --epochs-range 100 200 300 \
    --config configs/experiments/hyperparameter_tuning.yaml
```

**Reusing Existing Data:**

The script supports three methods for reusing previously generated data:

**Option 1: Auto-find data directories (Recommended)**
```bash
# Automatically finds data for each n_samples value from experiment directories
python scripts/run_hyperparameter_sweep.py \
    --n-samples-range 20 30 40 \
    --epochs-range 100 200 300 \
    --config configs/experiments/hyperparameter_tuning.yaml \
    --skip-data-generation \
    --data-dir-base "outputs/experiments"
```
- Searches all experiment directories in `outputs/experiments/`
- Matches each `n_samples` value to experiments with that value in their config
- Uses the most recent matching experiment's data directory
- Skips experiments if no matching data is found

**Option 2: Explicit mapping**
```bash
# Map each n_samples to a specific data directory
python scripts/run_hyperparameter_sweep.py \
    --n-samples-range 20 30 40 \
    --epochs-range 100 200 300 \
    --config configs/experiments/hyperparameter_tuning.yaml \
    --skip-data-generation \
    --data-dir "20:outputs/experiments/exp_XXX_20/data,30:outputs/experiments/exp_XXX_30/data,40:outputs/experiments/exp_XXX_40/data"
```
- Format: `"n_samples1:path1,n_samples2:path2,..."`
- Useful when you know exactly which experiment directories to use

**Option 3: Single data directory (all experiments use same data)**
```bash
# Use the same data for all experiments
python scripts/run_hyperparameter_sweep.py \
    --n-samples-range 20 30 40 \
    --epochs-range 100 200 300 \
    --config configs/experiments/hyperparameter_tuning.yaml \
    --skip-data-generation \
    --data-dir "outputs/experiments/exp_20251208_023045/data"
```
- ⚠️ **Note**: This only compares different `epochs` values, not `n_samples` (all use same dataset)

**Command-Line Options:**

| Option                   | Description                                                                     |
| ------------------------ | ------------------------------------------------------------------------------- |
| `--config`               | Base configuration file (default: `hyperparameter_tuning.yaml`)                 |
| `--n-samples-range`      | List of n_samples values to test (e.g., `20 30 40`)                             |
| `--epochs-range`         | List of epochs values to test (e.g., `100 200 300`)                             |
| `--skip-data-generation` | Skip data generation (use existing data)                                        |
| `--data-dir`             | Data directory mapping or single directory (see options above)                  |
| `--data-dir-base`        | Base directory to auto-search for experiment data (e.g., `outputs/experiments`) |

**Output:**

Each experiment creates its own directory: `outputs/experiments/exp_YYYYMMDD_HHMMSS/`

After completion, compare all results:
```bash
python scripts/compare_experiments.py \
    --experiments outputs/experiments/exp_20251208_21* outputs/experiments/exp_20251208_22* \
    --output-dir outputs/comparisons
```

**Example Workflow:**

1. **First run** (generate data for each n_samples):
   ```bash
   python scripts/run_hyperparameter_sweep.py \
       --n-samples-range 20 30 40 \
       --epochs-range 100 200 300 \
       --config configs/experiments/hyperparameter_tuning.yaml
   ```

2. **Rerun failed experiments** (reuse existing data):
   ```bash
   python scripts/run_hyperparameter_sweep.py \
       --n-samples-range 20 30 40 \
       --epochs-range 100 200 300 \
       --config configs/experiments/hyperparameter_tuning.yaml \
       --skip-data-generation \
       --data-dir-base "outputs/experiments"
   ```

3. **Compare results**:
   ```bash
   python scripts/compare_experiments.py \
       --experiments outputs/experiments/exp_20251208_* \
       --metric best_val_loss
   ```

### Environment Check

**Script**: `utils/check_environment.py`

Check environment (Colab/local, GPU availability) and suggest workflow.

**Usage:**
```bash
python scripts/utils/check_environment.py
```

### Colab Sync

**Script**: `utils/sync_for_colab.py`

Sync local changes to GitHub for Colab use.

**Usage:**
```bash
python scripts/utils/sync_for_colab.py [commit_message]
```

### Compare Approaches

**Script**: `comparison/compare_approaches.py`

Compare Pe Input vs Reactance approaches on same test data.

**Usage:**
```bash
python scripts/comparison/compare_approaches.py \
    --data-dir data \
    --model-pe-input model_pe.pth \
    --model-reactance model_reactance.pth
```

### Analysis and Comparison Utilities

**Scripts**: Analysis and comparison utilities for experiment results

- `compare_epoch_experiments.py` - Compare experiments with different epoch counts
- `create_comparison.py` - Create comprehensive comparison tables
- `generate_table.py` - Generate comparison tables from experiment data
- `update_comparison_table.py` - Update comparison tables with new experiments
- `update_table_from_ablation_studies.py` - Update tables from ablation studies
- `extract_all_dec8_experiments.py` - Extract experiment data from December 8 experiments
- `generate_figures.py` - Generate figures from experiment results
- `debug_figures.py` - Debug figure generation

**Usage:**
```bash
# Compare epoch experiments
python scripts/compare_epoch_experiments.py

# Generate comparison table
python scripts/generate_table.py

# Update comparison table
python scripts/update_comparison_table.py

# Generate figures
python scripts/generate_figures.py
```

---

## 📁 Directory Structure

```
scripts/
├── generate_data.py          # Step 1: Data generation
├── verify_data.py            # Step 2: Data verification
├── analyze_data.py           # Step 3: Data analysis
├── preprocess_data.py        # Step 4: Data preprocessing & splitting
├── train_model.py            # Step 5: Model training
├── evaluate_model.py         # Step 6: Model evaluation
├── run_experiment.py         # Complete workflow (all steps)
├── compare_experiments.py    # Compare multiple experiments
├── core/                      # Core modules (used by main scripts)
│   ├── data_generation.py    # Data generation logic
│   ├── data_utils.py         # Data selection utilities
│   ├── training.py           # Training logic
│   ├── evaluation.py         # Evaluation logic
│   ├── utils.py              # Shared utilities
│   ├── visualization.py       # Visualization wrapper
│   └── experiment_tracker.py # Experiment tracking
├── utils/                    # Standalone utility scripts
│   ├── check_environment.py  # Environment check utility
│   └── sync_for_colab.py     # Colab sync utility
├── comparison/               # Comparison utilities
│   └── compare_approaches.py  # Compare Pe Input vs Reactance approaches
├── compare_epoch_experiments.py  # Compare epoch experiments
├── create_comparison.py          # Create comparison tables
├── generate_table.py             # Generate comparison tables
├── update_comparison_table.py     # Update comparison tables
├── update_table_from_ablation_studies.py  # Update from ablation studies
├── extract_all_dec8_experiments.py       # Extract experiment data
├── generate_figures.py           # Generate figures
└── debug_figures.py               # Debug figure generation
```

---

## 🎯 Quick Start Examples

### Complete Workflow (Recommended for First Time)

```bash
# 1. Generate data
python scripts/generate_data.py --level quick
# Output: data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv

# 2. Verify data (using latest file or explicit path)
python scripts/verify_data.py
# Or: python scripts/verify_data.py data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv

# 3. Analyze data (using latest file or explicit path)
python scripts/analyze_data.py
# Or: python scripts/analyze_data.py data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv

# 4. Preprocess & Split (RECOMMENDED for reproducibility)
python scripts/preprocess_data.py --data-path data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv
# Output: data/preprocessed/quick_test/{train,val,test}_data_YYYYMMDD_HHMMSS.csv

# 5. Train (using pre-split data)
python scripts/train_model.py --data-path data/preprocessed/quick_test/train_data_YYYYMMDD_HHMMSS.csv --epochs 100

# 6. Evaluate (using test set)
python scripts/evaluate_model.py \
    --model-path outputs/experiments/exp_*/model/best_model_*.pth \
    --data-path data/preprocessed/quick_test/test_data_YYYYMMDD_HHMMSS.csv
```

**Note**: 
- Steps 3-6 can also be done automatically by `run_experiment.py`, which now includes:
  - **Data analysis** (Step 3) - Generates statistics and figures automatically
  - **Data preprocessing** (Step 4) - Creates train/val/test splits automatically
  - **Training** (Step 5) - Trains model automatically
  - **Evaluation** (Step 6) - Evaluates model automatically
- You can still run steps individually for better control and debugging
- **Use explicit file paths** (with timestamps) for reproducibility instead of relying on "latest" files
- The complete workflow (`run_experiment.py`) now matches all individual steps, ensuring consistency
```

### Single Command (Complete Experiment)

```bash
# Run everything in one command (full workflow)
python scripts/run_experiment.py --config configs/experiments/baseline_trajectory.yaml

# Use existing data (from directory)
python scripts/run_experiment.py \
    --config configs/experiments/baseline_trajectory.yaml \
    --skip-data-generation \
    --data-dir data/generated/quick_test

# Use existing data (direct file path)
python scripts/run_experiment.py \
    --config configs/experiments/baseline_trajectory.yaml \
    --skip-data-generation \
    --data-path data/generated/quick_test/parameter_sweep_data_20251205_170908.csv

# Use existing model (skip training)
python scripts/run_experiment.py \
    --config configs/experiments/baseline_trajectory.yaml \
    --skip-training \
    --model-path outputs/experiments/exp_20251205_170908/model/best_model_20251205_170908.pth

# Use existing data and model (evaluation only)
python scripts/run_experiment.py \
    --config configs/experiments/baseline_trajectory.yaml \
    --skip-data-generation \
    --skip-data-analysis \
    --skip-preprocessing \
    --data-path data/preprocessed/quick_test/test_data_20251205_170908.csv \
    --skip-training \
    --model-path outputs/experiments/exp_20251205_170908/model/best_model_20251205_170908.pth

# Override epochs from command line
python scripts/run_experiment.py \
    --config configs/experiments/quick.yaml \
    --skip-data-generation \
    --data-path data/generated/quick_test/parameter_sweep_data_20251205_170908.csv \
    --epochs 100

---

## 📝 Configuration Files

### ⭐ **Which Config Should I Use?**

**For `run_complete_experiment.py` (Complete Comparison Workflow):**

🎯 **Best Choice for Publication:** `configs/experiments/hyperparameter_tuning.yaml`
- ✅ Balanced dataset: 30 samples × 5 trajectories = 150 total
- ✅ Stratified splitting (`stratify_by: "is_stable"`) for balanced train/test
- ✅ Optimized hyperparameters (lambda_physics=0.5 from ablation study)
- ✅ Large network: [256, 256, 128, 128]
- ✅ Pe_direct input (7 dimensions, simpler and faster)
- ✅ 400 epochs (sufficient for training)

**Recommended Command:**
```bash
python scripts/run_complete_experiment.py \
    --config configs/experiments/hyperparameter_tuning.yaml \
    --pinn-epochs 400 \
    --ml-baseline-epochs 200
```

🚀 **Quick Testing:** `configs/experiments/quick.yaml`
- ✅ Fast: 20 samples × 5 = 100 trajectories (~1 hour data generation)
- ✅ Good for testing the complete workflow

**Recommended Command:**
```bash
python scripts/run_complete_experiment.py \
    --config configs/experiments/quick.yaml \
    --pinn-epochs 50 \
    --ml-baseline-epochs 50
```

**See:** `docs/analysis/CONFIG_SELECTION_GUIDE.md` for detailed comparison of all configs and selection criteria.

---

### Config File Types

#### Data Generation Configs
- `configs/data_generation/quick.yaml` - Quick scale (100 trajectories)
- `configs/data_generation/moderate.yaml` - Moderate scale (250 trajectories)
- `configs/data_generation/comprehensive.yaml` - Comprehensive scale (1000 trajectories)

**Note:** These are for standalone data generation only. For complete experiments, use experiment configs below.

#### Training Configs
- `configs/training/trajectory_config.yaml` - Trajectory prediction training
- `configs/training/parameter_config.yaml` - Parameter estimation training

**Note:** These are for individual training scripts only. For complete experiments, use experiment configs below.

#### Experiment Configs ⭐ **USE THESE FOR COMPLETE EXPERIMENTS**

These are **full pipeline configurations** that include data generation, model architecture, training, and evaluation settings.

**Recommended Configs:**
- `configs/experiments/hyperparameter_tuning.yaml` ⭐ **BEST** - Publication-quality experiments
- `configs/experiments/quick.yaml` - Quick testing and validation
- `configs/experiments/comprehensive.yaml` - Large-scale validation
- `configs/experiments/baseline_trajectory.yaml` - Baseline with reactance-based input

**Other Configs:**
- `configs/experiments/minimal_test.yaml` - Minimal test config (1 sample, 1 epoch, for pipeline verification)
- `configs/experiments/template.yaml` - Template for new experiments
- `configs/experiments/colab_quick.yaml` - Quick Colab experiment (small dataset)
- `configs/experiments/colab_moderate.yaml` - Moderate Colab experiment (medium dataset)
- `configs/experiments/colab_comprehensive.yaml` - Comprehensive Colab experiment (large dataset)
- `configs/experiments/large_model.yaml` - Large model configuration

**All experiment configs include:**
- Early stopping configuration (`early_stopping_patience`)
- Angle penalty settings (`max_training_angle_degrees`, `lambda_angle`)
- Data preprocessing settings (`train_ratio`, `val_ratio`, `test_ratio`)

**Key Differences:**
- **Input Method**: `pe_direct` (7 dims) vs `reactance` (11 dims)
- **Stratified Splitting**: `hyperparameter_tuning.yaml` has `stratify_by: "is_stable"` (important for fair comparison)
- **Network Size**: Small ([64,64,64]) vs Large ([256,256,128,128])
- **Dataset Size**: 50-150 trajectories (varies by config)

---

## 💡 Best Practices

### Workflow Best Practices

1. **Use Step-by-Step for Learning**: Run each step separately to understand the process
2. **Use Complete Workflow for Production**: Use `run_experiment.py` for reproducible experiments
   - Now includes data analysis and preprocessing automatically
   - Matches all individual steps for consistency
3. **Verify Before Training**: Always verify data quality before training
4. **Analyze After Generation**: Analyze data to understand distributions before training
   - Now done automatically in `run_experiment.py` (can skip with `--skip-data-analysis`)
5. **Use Appropriate Configs**: Choose quick/moderate/comprehensive based on your needs
6. **Use Existing Data/Models**: Load existing data with `--data-path` or `--data-dir`, and existing models with `--model-path`
7. **Proper Test Sets**: Enable preprocessing (`--skip-preprocessing=false`) to get proper train/val/test separation

### Data Selection Best Practices

6. **Use Explicit File Paths**: For reproducibility, specify exact file paths (with timestamps) instead of relying on "latest"
   ```bash
   # ✅ Good: Explicit path
   python scripts/analyze_data.py data/generated/quick_test/parameter_sweep_data_20251205_170908.csv
   
   # ⚠️ Less reproducible: Uses latest
   python scripts/analyze_data.py
   ```

7. **Document Data Sources**: Keep track of which data files you used (local, Colab, online GPU, etc.)

8. **Use Preprocessed Data for Training**: Use train/val/test splits from `preprocess_data.py` for explicit control
   ```bash
   # ✅ Good: Use preprocessed train data
   python scripts/train_model.py --data-path data/preprocessed/quick_test/train_data_YYYYMMDD_HHMMSS.csv
   ```

9. **Use Test Set for Evaluation**: Always evaluate on the test set, not training data
   ```bash
   # ✅ Good: Use test set
   python scripts/evaluate_model.py --data-path data/preprocessed/quick_test/test_data_YYYYMMDD_HHMMSS.csv
   ```

### Training Best Practices

10. **Choose Appropriate Input Method**:
    - **Pe-based** (`--input-method pe_direct`, **default**): Recommended for most use cases
      - Input: 7 dimensions `[t, δ₀, ω₀, H, D, Pm, Pe(t)]`
      - Simpler and faster (fewer input dimensions)
      - Uses Pe(t) directly from ANDES simulation
      - **Default for both PINN and ML baseline models**
    - **Reactance-based** (`--input-method reactance`, optional): Use when you have reactance values (Xprefault, Xfault, Xpostfault)
      - Input: 11 dimensions `[t, δ₀, ω₀, H, D, Pm, Xprefault, Xfault, Xpostfault, tf, tc]`
      - More interpretable (explicit network parameters)

11. **Adjust Batch Size Based on Dataset Size**:
    - **Calculate**: `batches_per_epoch = num_scenarios / batch_size`
    - **Target**: 4-12 batches per epoch for optimal learning
    - **Guidelines**:
      - < 50 scenarios → batch_size 4-8
      - 50-100 scenarios → batch_size 8-16
      - 100-200 scenarios → batch_size 16-24
      - > 200 scenarios → batch_size 24-32
    - **Example**: With 69 scenarios, use `--batch-size 8` or `--batch-size 12` (not 32)

12. **Start with Quick Test**:
    - Always test with `--epochs 1` first to verify the pipeline works
    - Then run full training with appropriate epochs
    - You can override epochs from command line: `--epochs 100` (overrides config file setting)

13. **Epoch Configuration**:
    - **Quick test**: `--epochs 1` (verify pipeline)
    - **Quick training**: `--epochs 50-100` (prototyping)
    - **Standard**: `--epochs 100-200` (most cases)
    - **Full training**: `--epochs 200-500` (publication-quality)
    - **Note**: Use `--epochs` command-line argument in `run_experiment.py` to override config file setting

14. **Learning Rate**:
    - **Default**: `0.001` (works well for most cases)
    - **Lower** (`0.0005`): If training is unstable or loss oscillates
    - **Higher** (`0.002`): May speed up but risk instability
    - Learning rate scheduler automatically reduces LR when validation loss plateaus

15. **Early Stopping**:
    - Configure `early_stopping_patience` in your config file (default: `200`)
    - Training stops automatically if validation loss doesn't improve
    - Saves time and prevents overfitting
    - Set to `null` to disable early stopping
    - Example: `early_stopping_patience: 50` stops after 50 epochs without improvement

16. **Angle Penalty**:
    - Configure `max_training_angle_degrees` (default: `720.0`) and `lambda_angle` (default: `0.1`)
    - Prevents unrealistic rotor angle predictions
    - Automatically penalizes predictions exceeding physical limits
    - Adjust `lambda_angle` to control penalty strength (higher = stronger penalty)

17. **Monitor Training Progress**:
    - Check training history: `outputs/experiments/exp_*/model/training_history_*.json`
    - Review loss curves: `outputs/experiments/exp_*/results/figures/`
    - Watch for overfitting (val loss increasing while train loss decreases)
    - Use time tracking to estimate remaining training time
    - Look for early stopping messages if validation loss plateaus

---

## 🔄 Workflow Diagram

### Individual Steps (Step-by-Step)

```
┌─────────────────┐
│ generate_data.py│  →  Generates CSV data file
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  verify_data.py │  →  Validates data quality
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ analyze_data.py │  →  Generates analysis & figures
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│preprocess_data.py│ →  Splits into train/val/test
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ train_model.py  │  →  Trains PINN model
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│evaluate_model.py│  →  Evaluates model performance
└─────────────────┘
```

### Complete Workflow (All-in-One)

```
┌──────────────────────┐
│  run_experiment.py   │
└──────────┬───────────┘
           │
           ├─→ STEP 1: Data Generation
           │
           ├─→ STEP 1B: Data Analysis (optional)
           │   └─→ Generates statistics & figures
           │
           ├─→ STEP 1C: Data Preprocessing (optional)
           │   └─→ Splits into train/val/test
           │
           ├─→ STEP 2: Model Training
           │
           ├─→ STEP 3: Model Evaluation
           │
           └─→ STEP 4: Save Experiment Metadata
```

**Key Features:**
- ✅ All individual steps are included
- ✅ Can skip any step with `--skip-*` flags
- ✅ Can load existing data with `--data-dir` or `--data-path`
- ✅ Can load existing model with `--model-path`
- ✅ Proper test set separation when preprocessing is enabled

---

## 📁 Subdirectories

### `core/` - Core Modules

Core modules used by main scripts:
- `data_generation.py` - Data generation logic
- `data_utils.py` - Data selection utilities (finding files, etc.)
- `training.py` - Training logic
- `evaluation.py` - Evaluation logic
- `utils.py` - Shared utilities
- `visualization.py` - Visualization wrapper
- `experiment_tracker.py` - Experiment tracking

**Note**: These are imported by main scripts, not run directly. They provide the core functionality for the workflow.

### `utils/` - Utility Scripts

Standalone utility scripts:
- `check_environment.py` - Environment check utility
- `sync_for_colab.py` - Colab sync utility

See `scripts/utils/README.md` for details.

### `comparison/` - Comparison Utilities

Comparison utilities for model approaches:
- `compare_approaches.py` - Compare Pe Input vs Reactance approaches

See `scripts/comparison/README.md` for details.

---

## 📂 Data Selection

### How Scripts Select Data

After `generate_data.py` creates timestamped files like:
```
data/generated/quick_test/parameter_sweep_data_20251205_170908.csv
data/generated/moderate_test/parameter_sweep_data_20251205_180000.csv
data/generated/comprehensive_test/parameter_sweep_data_20251205_190000.csv
```

Subsequent scripts can select data using multiple methods:

#### Method 1: Explicit File Path (Recommended for Reproducibility)

**Most Flexible**: Specify the exact file you want to use.

```bash
# Verify specific file
python scripts/verify_data.py data/generated/quick_test/parameter_sweep_data_20251205_170908.csv

# Analyze specific file
python scripts/analyze_data.py data/generated/moderate_test/parameter_sweep_data_20251205_180000.csv

# Preprocess specific file
python scripts/preprocess_data.py --data-path data/generated/comprehensive_test/parameter_sweep_data_20251205_190000.csv
```

**Benefits:**
- ✅ Explicit and clear
- ✅ Works with any file (local, Colab, etc.)
- ✅ Reproducible (always uses same file)

#### Method 2: Level-Based Selection (Convenient)

**Quick Iteration**: Use latest file from a specific level.

```bash
# Analyze latest moderate test data
python scripts/analyze_data.py --level moderate

# Analyze latest comprehensive test data
python scripts/analyze_data.py --level comprehensive
```

**Supported Levels:**
- `quick` (default)
- `moderate`
- `comprehensive`

#### Method 3: Directory-Based Selection (Flexible)

**Custom Locations**: Specify the directory to search.

```bash
# Use latest file in a specific directory
python scripts/analyze_data.py --data-dir data/generated/moderate_test

# Use latest file in custom directory
python scripts/analyze_data.py --data-dir data/generated/colab_data
```

#### Method 4: Default (Latest in quick_test)

**Quick Testing**: Uses latest file in `data/generated/quick_test/`.

```bash
# Uses latest file automatically
python scripts/verify_data.py
python scripts/analyze_data.py
```

**Note**: Not recommended for production workflows as it may select unexpected files.

---

### Using Data from Colab/Online GPU

You can use data generated on Colab or other online GPU services:

#### Step 1: Download Data from Colab

```python
# In Colab notebook
from google.colab import files
files.download('parameter_sweep_data_20251205_170908.csv')
```

#### Step 2: Place in Project Directory

```bash
# Create directory for Colab data
mkdir -p data/generated/colab_data

# Copy downloaded file
cp ~/Downloads/parameter_sweep_data_20251205_170908.csv \
   data/generated/colab_data/parameter_sweep_data_20251205_170908.csv
```

#### Step 3: Use Explicit Path

```bash
# Verify Colab data
python scripts/verify_data.py data/generated/colab_data/parameter_sweep_data_20251205_170908.csv

# Analyze Colab data
python scripts/analyze_data.py data/generated/colab_data/parameter_sweep_data_20251205_170908.csv

# Preprocess Colab data
python scripts/preprocess_data.py --data-path data/generated/colab_data/parameter_sweep_data_20251205_170908.csv
```

---

### Using Previously Generated Data

#### Find Available Files

```bash
# List files in quick_test
ls -lt data/generated/quick_test/parameter_sweep_data_*.csv

# List files in moderate_test
ls -lt data/generated/moderate_test/parameter_sweep_data_*.csv

# List files in comprehensive_test
ls -lt data/generated/comprehensive_test/parameter_sweep_data_*.csv
```

#### Use Specific File

```bash
# Use file from yesterday
python scripts/analyze_data.py data/generated/quick_test/parameter_sweep_data_20251204_120000.csv

# Use file from last week
python scripts/train_model.py --data-path data/generated/moderate_test/parameter_sweep_data_20251128_150000.csv
```

---

### Data Selection Priority

When a script needs to find data, it uses this priority:

1. **Explicit file path** (if provided) - Highest priority
2. **Directory specified** (if provided via `--data-dir`)
3. **Level specified** (if provided via `--level`)
4. **Default** (latest in `quick_test`) - Lowest priority

---

### Best Practices for Data Selection

| Method               | Use Case                    | Reproducibility | Works with Colab?     |
| -------------------- | --------------------------- | --------------- | --------------------- |
| **Explicit path**    | Production, reproducibility | ✅ High          | ✅ Yes                 |
| **Level flag**       | Quick iteration             | ⚠️ Medium        | ⚠️ If placed correctly |
| **Directory flag**   | Custom locations            | ✅ High          | ✅ Yes                 |
| **Default (latest)** | Quick testing               | ❌ Low           | ⚠️ If placed correctly |

**Recommendation**: 
- ✅ **Use explicit file paths** for production workflows and when using data from Colab/online GPU
- ✅ **Document which files you used** in experiment logs
- ⚠️ **Avoid relying on "latest"** for important experiments

### Configuration Selection Best Practices

18. **Choose the Right Config File**:
    - ✅ **Publication experiments**: Use `hyperparameter_tuning.yaml` (stratified splitting, optimized hyperparameters)
    - ✅ **Quick testing**: Use `quick.yaml` (fast, smaller dataset)
    - ✅ **Baseline comparison**: Use `baseline_trajectory.yaml` (reactance-based input)
    - ⚠️ **Always use stratified splitting** (`stratify_by: "is_stable"`) for fair comparison
    - ⚠️ **Ensure input method consistency** between PINN and ML baseline models

19. **Config File Types**:
    - ✅ **Experiment configs** (`configs/experiments/*.yaml`): Use for `run_complete_experiment.py` and `run_experiment.py`
    - ⚠️ **Data generation configs** (`configs/data_generation/*.yaml`): Only for standalone data generation
    - ⚠️ **Training configs** (`configs/training/*.yaml`): Only for individual training scripts

**See**: `docs/analysis/CONFIG_SELECTION_GUIDE.md` for detailed comparison and selection criteria.

---

**See `docs/DATA_SELECTION_GUIDE.md` for complete details and examples.**

---

## 📚 Additional Resources

### Workflow Guides
- **Data Generation Guide**: `docs/guides/QUICK_TEST_DATA_GENERATION.md`
- **Training Guide**: `docs/guides/QUICK_TRAINING_GUIDE.md`
- **Analysis Guide**: `docs/guides/DATA_ANALYSIS_FOR_PUBLICATIONS.md`
- **Complete Workflow**: `docs/guides/COMPLETE_WORKFLOW_PROCEDURE.md`
- **Workflow Organization**: `docs/WORKFLOW_ORGANIZATION.md`

### Configuration Guides
- **Config Selection Guide**: `docs/analysis/CONFIG_SELECTION_GUIDE.md` ⭐ **NEW** - Which config to use for your experiments

### Data Management
- **Data Selection Guide**: `docs/DATA_SELECTION_GUIDE.md` - Complete guide on selecting and using data files
- **Data Normalization**: `docs/guides/DATA_NORMALIZATION_EXPLANATION.md`

### Project Organization
- **Script Organization**: `docs/SCRIPTS_ORGANIZATION_COMPLETE.md`
- **AI Agent Guidelines**: `docs/AI_AGENT_GUIDELINES.md`

---

## 🔍 Quick Reference

### Finding Your Data Files

```bash
# List all generated data files
find data/generated -name "parameter_sweep_data_*.csv" -type f | sort

# List files by level
ls -lt data/generated/quick_test/parameter_sweep_data_*.csv
ls -lt data/generated/moderate_test/parameter_sweep_data_*.csv
ls -lt data/generated/comprehensive_test/parameter_sweep_data_*.csv

# List preprocessed files
find data/preprocessed -name "*_data_*.csv" -type f | sort
```

### Common Workflow Patterns

```bash
# Pattern 1: Quick iteration (uses latest files)
python scripts/generate_data.py --level quick
python scripts/verify_data.py
python scripts/analyze_data.py

# Pattern 2: Reproducible workflow (explicit paths)
python scripts/generate_data.py --level quick
# Note the output file: parameter_sweep_data_20251205_170908.csv
python scripts/verify_data.py data/generated/quick_test/parameter_sweep_data_20251205_170908.csv
python scripts/analyze_data.py data/generated/quick_test/parameter_sweep_data_20251205_170908.csv
python scripts/preprocess_data.py --data-path data/generated/quick_test/parameter_sweep_data_20251205_170908.csv
python scripts/train_model.py --data-path data/preprocessed/quick_test/train_data_YYYYMMDD_HHMMSS.csv

# Pattern 3: Using Colab data
# 1. Download from Colab
# 2. Place in: data/generated/colab_data/
# 3. Use explicit path:
python scripts/verify_data.py data/generated/colab_data/parameter_sweep_data_YYYYMMDD_HHMMSS.csv
```

---

**Last Updated**: December 2024  
**Status**: Active Documentation
