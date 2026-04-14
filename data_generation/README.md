# Data Generation Module

This directory contains modules for generating training data from ANDES power system simulations. It handles parameter sweeps, data extraction, preprocessing, and validation.

## 📑 Table of Contents

- [Overview](#overview)
- [Module Files](#module-files)
  - [andes_extractor.py](#andes_extractorpy)
  - [parameter_sweep.py](#parameter_sweeppy)
  - [preprocessing.py](#preprocessingpy)
  - [sampling_strategies.py](#sampling_strategiespy)
  - [validation.py](#validationpy)
  - [andes_utils/ Subdirectory](#andes_utils-subdirectory)
- [Module Exports](#module-exports)
- [How Data Generation Works](#how-data-generation-works)
- [Comparison: `parameter_sweep.py` vs `smib_albert_cct.py`](#comparison-parameter_sweeppy-vs-smib_albert_cctpy)
- [Data Generation Best Practices](#data-generation-best-practices)
  - [Choosing the Right Strategy](#choosing-the-right-strategy)
  - [Why Task-Specific Strategies Matter](#why-task-specific-strategies-matter)
  - [Sampling Strategy Comparison](#sampling-strategy-comparison)
- [Usage Examples](#usage-examples)
  - [Complete Data Generation Workflow](#complete-data-generation-workflow)
  - [CCT Data Generation with Boundary Focus](#cct-data-generation-with-boundary-focus)
- [Configuration](#configuration)
- [Notes](#notes)

---

## Overview

The `data_generation/` module provides:
- **ANDES Integration** - Extract trajectories and system parameters from ANDES simulations
- **Parameter Sampling** - Generate diverse parameter combinations (H, D, fault scenarios)
- **Data Preprocessing** - Normalize, split, and prepare data for PINN training
- **Validation** - Quality checks and coverage analysis for generated datasets

## Module Files

### `andes_extractor.py`
**Purpose**: Extract trajectories and system parameters from ANDES TDS simulation results and format them into structured DataFrames for PINN training.

This module serves as a **bridge between ANDES simulation outputs and the PINN training pipeline**, converting raw ANDES results into structured, training-ready data. It handles multiple ANDES API versions and provides robust fallback mechanisms for reliable data extraction.

#### Key Functions

##### `extract_trajectories(ss, gen_idx=None)`
Extracts time-series trajectories from completed ANDES TDS simulations.

**Extracted Data**:
- **Rotor angle (δ)**: In radians and degrees (`delta`, `delta_deg`)
- **Rotor speed (ω)**: In per-unit (`omega`)
- **Speed deviation**: Deviation from 1.0 pu (`omega_deviation`)
- **Electrical power (Pe)**: In per-unit
- **Mechanical power (Pm)**: In per-unit (typically constant for GENCLS)

**Robust Extraction Strategy** (with multiple fallbacks):
1. **Primary Method**: Uses `ss.dae.ts` (ANDES recommended approach)
   - Time: `ss.dae.ts.t`
   - States: `ss.dae.ts.x[:, variable.a]` (for δ, ω)
   - Algebraic: `ss.dae.ts.y[:, variable.a]` (for Pe)
2. **Fallback 1**: Uses `ss.TDS.plt` (plotter interface) if `ss.dae.ts` unavailable
3. **Fallback 2**: Direct `.v` attribute access (last resort)

**Parameters**:
- `ss`: ANDES system object after TDS simulation
- `gen_idx`: Generator index (string name or int position). Default: 0 (first GENCLS generator)

**Returns**: Dictionary with keys: `time`, `delta`, `omega`, `Pe`, `Pm`, `delta_deg`, `omega_deviation`

##### `extract_system_reactances(ss, fault_idx=None)`
Extracts system reactances for different fault states.

**Extracted Data**:
- **Xprefault**: Pre-fault equivalent reactance (pu)
- **Xfault**: Fault reactance (pu)
- **Xpostfault**: Post-fault equivalent reactance (pu)
- **tf**: Fault start time (seconds)
- **tc**: Fault clearing time (seconds)

**Calculation Method**:
- For parallel lines: Calculates equivalent reactance using `1/Xeq = Σ(1/Xi)`
- Falls back to default values if line data unavailable

##### `label_system_states(time, tf, tc)`
Labels each time point with system state for training.

**State Labels**:
- `0` = Pre-fault
- `1` = During fault
- `2` = Post-fault

**Returns**: NumPy array of state labels (same length as `time`)

##### `extract_complete_dataset(ss, gen_idx=None, fault_idx=None)`
**Main function** - Combines all extraction functions into a single DataFrame.

**Complete Dataset Includes**:
- **Trajectories**: `time`, `delta`, `delta_deg`, `omega`, `omega_deviation`, `Pe`, `Pm`
- **System Reactances**: `Xprefault`, `Xfault`, `Xpostfault`
- **Fault Timing**: `tf`, `tc`
- **State Labels**: `state` (0=pre-fault, 1=during-fault, 2=post-fault)
- **Generator Parameters**: `M`, `D`, `H` (extracted from GENCLS)
- **Initial Conditions**: `delta0`, `omega0`

**Note**: Stability labels (`is_stable`, `is_stable_from_cct`) are **not** added by `extract_complete_dataset()` itself, but are automatically added by `parameter_sweep.py` during data generation for all operating points.

**Returns**: `pd.DataFrame` ready for PINN training

#### Integration with Data Generation Pipeline

This module is automatically called by `parameter_sweep.py` during data generation:

```python
# Inside parameter_sweep.py after TDS simulation:
from .andes_extractor import extract_complete_dataset

# After running ss.TDS.run()
df = extract_complete_dataset(ss, gen_idx=gen_idx, fault_idx=fault_idx)

# Then combined with parameter metadata:
df["param_M"] = M
df["param_D"] = D
df["param_H"] = M / 2.0
df["param_tc"] = tc
# ... and CCT information if using CCT-based sampling
# ... and stability labels (is_stable, is_stable_from_cct) automatically added
```

#### Usage Examples

**Basic Usage**:
```python
from data_generation.andes_extractor import (
    extract_trajectories,
    extract_system_reactances,
    extract_complete_dataset
)

# After running ANDES TDS simulation
ss.TDS.run()

# Extract complete dataset (recommended)
df = extract_complete_dataset(ss, gen_idx=0, fault_idx=None)

# Or extract individual components
trajectories = extract_trajectories(ss, gen_idx=0)
reactances = extract_system_reactances(ss, fault_idx=None)
```

**Accessing Extracted Data**:
```python
# From complete dataset
time = df["time"]
delta = df["delta"]  # radians
delta_deg = df["delta_deg"]  # degrees
omega = df["omega"]  # pu
Pe = df["Pe"]  # pu
Pm = df["Pm"]  # pu
state = df["state"]  # 0, 1, or 2

# System parameters
M = df["M"].iloc[0]  # Inertia constant
D = df["D"].iloc[0]  # Damping coefficient
H = df["H"].iloc[0]  # H = M/2

# Reactances
Xprefault = df["Xprefault"].iloc[0]
Xfault = df["Xfault"].iloc[0]
Xpostfault = df["Xpostfault"].iloc[0]
```

#### Why This Module is Important

1. **Standardization**: Provides consistent interface regardless of ANDES version
2. **Robustness**: Multiple fallback methods ensure data extraction succeeds even if ANDES API changes
3. **Completeness**: Extracts all necessary data (trajectories, parameters, states) in one call
4. **PINN-Ready Format**: Outputs DataFrames directly usable for neural network training
5. **Error Handling**: Validates generator existence and handles edge cases gracefully

#### Notes

- The module automatically handles multiple generators by using `gen_idx` parameter
- All arrays are ensured to have the same length (padded if necessary)
- Default values are used if certain data is unavailable (with warnings)
- The extraction follows ANDES documentation recommendations for best compatibility

---

### `parameter_sweep.py`
**Purpose**: Generate diverse training data by varying system parameters and running ANDES simulations.

**Key Functions**:
- `generate_trajectory_data()` - Generate data for trajectory prediction task
- `generate_parameter_estimation_data()` - Generate data for parameter estimation task (with decorrelated H-D sampling)
- `generate_cct_data()` - Generate data for CCT estimation (boundary-focused sampling)
- `generate_multi_task_data()` - Generate combined dataset for multi-task learning
- `generate_parameter_sweep()` - Generic parameter sweep generator

**Sampling Strategies**:
- **Full factorial**: Exhaustive combination of parameter ranges
- **Latin Hypercube Sampling (LHS)**: Space-filling design
- **Sobol sequences**: Quasi-random low-discrepancy sequences
- **Boundary-focused**: Concentrated sampling near stability boundaries
- **Decorrelated**: Low correlation between H and D for parameter estimation

**Usage**:
```python
from data_generation import generate_trajectory_data, generate_cct_data

# Generate trajectory prediction data
trajectory_data = generate_trajectory_data(
    n_samples=1000,
    H_range=(2.0, 10.0),
    D_range=(0.1, 2.0),
    sampling_method="lhs"
)

# Generate CCT data with boundary-focused sampling
cct_data = generate_cct_data(
    n_samples=500,
    boundary_focus=True,
    stability_threshold=180.0
)
```

#### Generated Data Structure

The `generate_parameter_sweep()` and related functions return a `pd.DataFrame` with the following structure:

**Data Organization**:
- **Each row** = One time point in a trajectory
- **Multiple rows with same `scenario_id`** = One operating point (one complete simulation)
- **Metadata columns** = Scalar values (same for all time points in a scenario)

**Stored Information** (automatically added for all operating points):

1. **Trajectory Data** (time-series, varies per time point):
   - `time`: Time stamps (seconds)
   - `delta`: Rotor angle (radians)
   - `delta_deg`: Rotor angle (degrees)
   - `omega`: Rotor speed (per-unit)
   - `omega_deviation`: Speed deviation from 1.0 pu
   - `Pe`: Electrical power (per-unit)
   - `Pm`: Mechanical power (per-unit)
   - `state`: System state label (0=pre-fault, 1=during-fault, 2=post-fault)

2. **System Parameters** (scalar per scenario):
   - `param_M`: Inertia constant M (seconds)
   - `param_D`: Damping coefficient D (per-unit)
   - `param_H`: Inertia constant H = M/2 (seconds)
   - `param_Pm`: Mechanical power (per-unit)
   - `param_tc`: Fault clearing time (seconds, absolute time)
   - `param_fault_bus`: Bus where fault occurs
   - `scenario_id`: Unique identifier for each operating point

3. **CCT Information** (scalar per scenario, when CCT-based sampling is used):
   - `param_cct_absolute`: Absolute CCT time (fault_start_time + cct_duration)
   - `param_cct_duration`: CCT duration (seconds)
   - `param_cct_uncertainty`: CCT uncertainty from bisection (seconds)
   - `param_small_delta`: Small delta (ε) used for sampling (seconds)
   - `param_offset_from_cct`: Offset of clearing time from CCT (seconds)

4. **Stability Labels** (scalar per scenario, **automatically computed for all operating points**):
   - `is_stable` (bool): **Trajectory-based stability classification** (primary method)
     - `True` if `max(|delta|) < π` AND `max(|omega - 1.0|) < 1.5`
     - Based on actual trajectory behavior
     - **Always computed** (works with or without CCT-based sampling)
   - `is_stable_from_cct` (bool, optional): **CCT-based stability classification** (validation method)
     - `True` if `clearing_time < CCT`
     - Based on physics (clearing time relative to CCT)
     - Only available when CCT-based sampling is used and CCT is found
     - Set to `NaN` otherwise

5. **System Reactances** (scalar per scenario):
   - `Xprefault`: Pre-fault equivalent reactance (per-unit)
   - `Xfault`: Fault reactance (per-unit)
   - `Xpostfault`: Post-fault equivalent reactance (per-unit)
   - `tf`: Fault start time (seconds)
   - `tc`: Fault clearing time (seconds, from ANDES)

6. **Initial Conditions** (scalar per scenario):
   - `delta0`: Initial rotor angle (radians)
   - `omega0`: Initial rotor speed (per-unit)

**Example Data Structure**:
```
scenario_id | time  | delta | omega | param_M | param_D | param_Pm | param_tc | is_stable | is_stable_from_cct
-----------|-------|-------|-------|---------|---------|----------|----------|-----------|-------------------
     0     | 0.00  | 0.5   | 1.0   |   5.0   |   2.0   |   0.8    |   1.2    |   True    |       True
     0     | 0.01  | 0.52  | 1.01  |   5.0   |   2.0   |   0.8    |   1.2    |   True    |       True
     0     | 0.02  | 0.54  | 1.02  |   5.0   |   2.0   |   0.8    |   1.2    |   True    |       True
   ...     | ...   | ...   | ...   |   ...   |   ...   |   ...    |   ...    |   ...     |       ...
     1     | 0.00  | 0.5   | 1.0   |   5.0   |   2.0   |   0.8    |   1.25   |  False    |      False
     1     | 0.01  | 0.55  | 1.05  |   5.0   |   2.0   |   0.8    |   1.25   |  False    |      False
   ...     | ...   | ...   | ...   |   ...   |   ...   |   ...    |   ...    |   ...     |       ...
```

**Key Points**:
- ✅ **Stability labels are stored for ALL operating points** (every successful simulation)
- ✅ **Same storage pattern as other metadata** (scalar value per scenario, repeated for all time points)
- ✅ **Dual validation**: Both trajectory-based and CCT-based stability (when available)
- ✅ **Automatic computation**: No post-processing needed

**Accessing Stability Information**:
```python
# Get unique operating points with their stability
unique_scenarios = df.groupby('scenario_id').first()[
    ['param_M', 'param_D', 'param_Pm', 'param_tc', 'is_stable', 'is_stable_from_cct']
]

# Count stable vs unstable cases
stability_counts = df.groupby('scenario_id')['is_stable'].first().value_counts()
print(f"Stable cases: {stability_counts.get(True, 0)}")
print(f"Unstable cases: {stability_counts.get(False, 0)}")

# Check consistency between trajectory-based and CCT-based stability
if 'is_stable_from_cct' in df.columns:
    consistent = df.groupby('scenario_id').apply(
        lambda x: x['is_stable'].iloc[0] == x['is_stable_from_cct'].iloc[0]
    )
    print(f"Consistency: {consistent.sum()}/{len(consistent)} scenarios")
```

---

### `preprocessing.py`
**Purpose**: Data preprocessing utilities for PINN training.

**Key Functions**:
- `preprocess_data()` - Complete preprocessing pipeline (normalization, feature engineering)
- `normalize_data()` - Normalize features and targets to [0, 1] or standardize
- `split_dataset()` - Split data into train/validation/test sets

**Preprocessing Steps**:
1. Feature normalization (standardization or min-max scaling)
2. Target normalization (for stable training)
3. Feature engineering (derived features, time encoding)
4. Data splitting with stratification (if applicable)

**Usage**:
```python
from data_generation import preprocess_data, split_dataset

# Preprocess data
processed_data, scalers = preprocess_data(
    raw_data,
    normalize_features=True,
    normalize_targets=True
)

# Split dataset
train_data, val_data, test_data = split_dataset(
    processed_data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
```

---

### `sampling_strategies.py`
**Purpose**: Advanced parameter sampling strategies for diverse training data.

**Key Functions**:
- `latin_hypercube_sample()` - LHS sampling for uniform parameter space coverage
- `sobol_sequence_sample()` - Sobol quasi-random sequences
- `boundary_focused_sample()` - Concentrated sampling near stability boundaries
- `decorrelated_sample()` - Low correlation sampling (for parameter estimation)
- `correlation_analysis()` - Analyze parameter correlations in samples
- `validate_sample_quality()` - Check sampling quality metrics

**Usage**:
```python
from data_generation import (
    latin_hypercube_sample,
    boundary_focused_sample,
    validate_sample_quality
)

# LHS sampling
samples = latin_hypercube_sample(
    n_samples=1000,
    bounds={"H": (2.0, 10.0), "D": (0.1, 2.0)}
)

# Boundary-focused sampling
boundary_samples = boundary_focused_sample(
    n_samples=500,
    stability_data=stability_results,
    focus_radius=0.1
)

# Validate quality
quality_metrics = validate_sample_quality(samples)
```

---

### `validation.py`
**Purpose**: Data quality validation and coverage analysis.

**Key Functions**:
- `validate_data_for_task()` - Task-specific data validation
- `validate_cct_data_quality()` - Validate CCT estimation dataset quality
- `detect_stability_boundary()` - Identify stability boundaries in parameter space
- `estimate_cct_from_data()` - Estimate CCT from trajectory data
- `analyze_parameter_coverage()` - Analyze parameter space coverage
- `generate_validation_report()` - Generate comprehensive validation report
- `generate_data_quality_report()` - Generate detailed quality metrics report
- `track_quality_incremental()` - Track quality metrics incrementally during generation

**Validation Checks**:
- Parameter range coverage
- Stability/instability balance
- Data completeness
- Outlier detection
- Boundary representation
- Time step consistency
- Correlation analysis

**Usage**:
```python
from data_generation import (
    validate_data_for_task,
    analyze_parameter_coverage,
    generate_validation_report
)

# Validate trajectory prediction data
validation_results = validate_data_for_task(
    data,
    task="trajectory_prediction",
    required_columns=["delta", "omega", "H", "D"]
)

# Analyze coverage
coverage = analyze_parameter_coverage(
    data,
    parameters=["H", "D"],
    bins=20
)

# Generate report
report = generate_validation_report(data, task="trajectory_prediction")
```

---

### `andes_utils/` Subdirectory

The `andes_utils/` subdirectory contains ANDES-specific utility modules:

#### `system_manager.py`
**Purpose**: Manage ANDES system setup, configuration, and simulation execution.

#### `data_extractor.py`
**Purpose**: Extract data from ANDES output files (`.lst`, `.npz`, `.txt`).

#### `data_exporter.py`
**Purpose**: Export extracted data to standardized formats (CSV, JSON).

#### `data_validator.py`
**Purpose**: Validate ANDES simulation outputs and extracted data.

**Key Functions**:
- `validate_data_quality()` - Low-level data validation (NaN/Inf, time consistency, trajectory length)
- `check_stability()` - Check system stability with various criteria

**Note**: For high-level validation (task-specific, coverage analysis), use functions from `validation.py` instead.

#### `cct_finder.py`
**Purpose**: Find Critical Clearing Time from ANDES simulation results.

---

## Module Exports

The `data_generation/__init__.py` module exports:

```python
from data_generation import (
    # Data extraction
    extract_trajectories,
    extract_system_reactances,

    # Parameter sweep generation
    generate_parameter_sweep,
    generate_trajectory_data,
    generate_parameter_estimation_data,
    generate_cct_data,
    generate_multi_task_data,

    # Preprocessing
    preprocess_data,
    normalize_data,
    split_dataset,

    # Sampling strategies
    latin_hypercube_sample,
    sobol_sequence_sample,
    boundary_focused_sample,
    correlation_analysis,
    validate_sample_quality,

    # Validation
    validate_data_quality,  # Low-level validation (NaN/Inf, time consistency)
    detect_stability_boundary,
    estimate_cct_from_data,
    validate_cct_data_quality,
    analyze_parameter_coverage,
    validate_data_for_task,
    generate_validation_report,
    generate_data_quality_report,  # Comprehensive quality metrics
    track_quality_incremental,  # Incremental quality tracking
)
```

## How Data Generation Works

The data generation process works as follows:

1. **Loads SMIB Case from ANDES**: The code loads the `smib/SMIB.json` case file directly from ANDES using `andes.load()`. This case file contains the base SMIB system structure (buses, lines, generator model, etc.) with default parameters (e.g., default M = 5.7512, D = 1.0).

2. **Uses Case as Template**: The SMIB.json file serves as a **base template** for the system structure. It defines:
   - System topology (3 buses, transmission lines, generator)
   - Default component models (GENCLS generator, Fault model)
   - Initial operating conditions

3. **Manually Overrides Parameters**: After loading the case, the code **manually overrides** the generator parameters (H, D, and P_m) and fault parameters with the values you specify:
   - **H (Inertia constant)**: Converted to M (M = 2×H for 60 Hz systems) and set via `ss.GENCLS.M.v[0] = M`
   - **D (Damping coefficient)**: Set directly via `ss.GENCLS.D.v[0] = D`
   - **P_m (Mechanical power)**: Set via `ss.GENCLS.tm0.v[0] = Pm` and `ss.GENCLS.P0.v[0] = Pm` (typically 0.4-0.9 pu)
   - **Fault clearing time**: Set via `ss.Fault.tc.v[0] = tc`
   - **Fault location**: Set via `ss.Fault.bus.v[0] = fault_bus`

4. **Generates Parameter Combinations**: The `H_range`, `D_range`, and `Pm_range` parameters you provide are used to generate different combinations:
   - **For trajectory prediction**: 3D Sobol sampling (H, D, P_m) or full factorial sampling (all combinations)
   - **For parameter estimation**: Decorrelated sampling (low H-D correlation to help the model learn distinct effects)
   - **P_m variations**: Allows the model to learn how different power levels affect system stability

5. **Runs Simulations**: For each parameter combination:
   - Loads the SMIB case (fresh copy each time)
   - Overrides M, D, P_m, and fault parameters
   - Runs power flow to establish initial conditions
   - Runs time-domain simulation (TDS) to generate trajectories
   - Extracts trajectory data (δ, ω, time, reactances, etc.)

**Key Point**: The SMIB.json file provides the **system structure**, while the **H, D, P_m, and fault parameters are provided manually** via the function arguments. This allows you to generate diverse training data by sweeping these parameters across different ranges. P_m variations (typically 0.4-0.9 pu) enable the model to learn how different operating conditions affect transient stability.

## Comparison: `parameter_sweep.py` vs `smib_albert_cct.py`

This module (`parameter_sweep.py`) and the standalone script `examples/scripts/smib/smib_albert_cct.py` serve different purposes and generate cases differently:

### `parameter_sweep.py` - Automatic Parameter Sweeping

**Purpose**: Generate diverse training data (trajectories) for PINN training.

**How it generates cases**:
- **Automatic parameter combination generation**:
  - Takes parameter ranges as input (e.g., `H_range=(2.0, 10.0, 5)`, `D_range=(0.5, 3.0, 5)`)
  - Automatically generates all combinations using:
    - Full factorial sampling (all combinations)
    - Latin Hypercube Sampling (LHS)
    - Sobol sequences
    - Boundary-focused sampling
    - Decorrelated sampling
- **Sweeps multiple parameters simultaneously**:
  - H (or M) values
  - D values
  - P_m (mechanical power) values (optional, typically 0.4-0.9 pu)
  - Fault clearing times
  - Fault locations (optional)
- **Output**: CSV files with complete trajectory data for training

**Example Usage**:
```python
from data_generation import generate_trajectory_data

# Automatically generates all combinations
data = generate_trajectory_data(
    case_file="smib/SMIB.json",
    H_range=(2.0, 10.0, 5),  # 5 values
    D_range=(0.5, 3.0, 5),    # 5 values
    Pm_range=(0.4, 0.9, 5),    # 5 values (optional, enables 3D sampling)
    fault_clearing_times=[1.15, 1.18, 1.20, 1.22, 1.25],
    sampling_strategy="full_factorial"  # 5×5×5 = 125 combinations (or 3D Sobol)
)
```

### `smib_albert_cct.py` - Manual Parameter Input for CCT Finding

**Purpose**: Find Critical Clearing Time (CCT) boundaries for specific operating points.

**How it generates cases**:
- **Manual parameter input required**:
  - **Single case mode** (`main()` function): Hardcoded single operating point
    ```python
    Pm = 0.8  # User must manually change these values
    M = 6.0
    D = 1.0
    ```
  - **Batch mode** (`batch_cct_finding_in_memory()` function): Requires user to provide a list of (Pm, M, D) tuples
    ```python
    # User must manually create this list
    param_combinations = [
        (0.6, 5.0, 1.0),
        (0.7, 6.0, 1.5),
        (0.8, 7.0, 2.0),
    ]

    results = batch_cct_finding_in_memory(
        case_path,
        param_combinations,  # Pass the list you created
        ...
    )
    ```
- **Does NOT automatically generate parameter combinations**
- **Uses binary search** to find CCT for each provided (Pm, M, D) combination
- **Output**: CCT values, stability results, and plots (not training data)

### Key Differences Summary

| Aspect | `parameter_sweep.py` | `smib_albert_cct.py` |
|--------|---------------------|----------------------|
| **Case Generation** | ✅ Automatic (sweeps parameters) | ❌ Manual (user provides combinations) |
| **Purpose** | Generate training trajectories | Find CCT boundaries |
| **Parameter Input** | Ranges (min, max, num_points) | List of specific tuples |
| **Sampling Strategies** | Full factorial, LHS, Sobol, etc. | None (user-provided list) |
| **Output** | Training data CSV files | CCT values + stability results |
| **Use Case** | PINN training data generation | CCT analysis and validation |

### When to Use Each

- **Use `parameter_sweep.py`** when:
  - You need diverse training data for PINN training
  - You want automatic parameter combination generation
  - You need trajectory data across parameter space
  - You're doing trajectory prediction or parameter estimation tasks

- **Use `smib_albert_cct.py`** when:
  - You need to find CCT for specific operating points
  - You want to validate CCT boundaries
  - You need detailed CCT analysis with plots
  - You're doing CCT boundary studies (not training data generation)

### Generating Multiple Cases with `smib_albert_cct.py`

If you want to use `smib_albert_cct.py` for multiple cases, you need to manually generate the parameter combinations:

```python
import itertools
from examples.scripts.smib.smib_albert_cct import batch_cct_finding_in_memory

# Manually generate parameter combinations
Pm_values = [0.6, 0.7, 0.8]
M_values = [5.0, 6.0, 7.0]
D_values = [1.0, 1.5, 2.0]

# Generate all combinations
param_combinations = list(itertools.product(Pm_values, M_values, D_values))
# Result: 3×3×3 = 27 combinations

# Pass to batch function
results = batch_cct_finding_in_memory(
    case_path="smib/SMIB.json",
    param_combinations=param_combinations,
    ...
)
```

**Note**: For training data generation, `parameter_sweep.py` is recommended as it handles parameter combination generation automatically and is optimized for PINN training workflows.

## Data Generation Best Practices

### Choosing the Right Strategy

**Two-Model Architecture:**

1. **Trajectory Prediction** (Forward Problem):
   - **Strategy**: Full factorial or Latin Hypercube Sampling (LHS)
   - **Focus**: Uniform coverage of parameter space
   - **H-D Correlation**: Not critical (model learns forward mapping)
   - **Example**: Predicting δ(t), ω(t) from known H, D, fault_clearing_time
   - **Complexity**: Basic PINN with physics loss, straightforward data generation
   - **Model**: `TrajectoryPredictionPINN` - standalone forward model

2. **CCT Estimation** (Via Binary Search):
   - **No separate model needed** - Uses binary search with the trained trajectory model
   - **Strategy**: Uses the trajectory prediction model iteratively
   - **Focus**: Binary search to find maximum stable fault clearing time
   - **H-D Correlation**: Not applicable (uses trajectory model)
   - **Example**: Finding CCT by testing different clearing times with trajectory model
   - **Complexity**: Algorithm-based (no separate training), requires trained trajectory model
   - **Depends on**: Trajectory prediction model and stability checking
   - **Implementation**: `utils.cct_binary_search.estimate_cct_binary_search()`

3. **Parameter Estimation** (Inverse Problem):
   - **Strategy**: Decorrelated sampling (LHS or Sobol sequence)
   - **Focus**: Low correlation between H and D (target: |r| < 0.3)
   - **H-D Correlation**: Critical (model must learn distinct effects)
   - **Example**: Estimating H, D from observed δ(t), ω(t) trajectories
   - **Complexity**: Most complex (sequence models, LSTM optional), requires trajectory data
   - **Depends on**: Trajectory data and understanding of system dynamics
   - **Model**: `ParameterEstimationPINN` - inverse model using trajectory sequences

### Why Task-Specific Strategies Matter

**Parameter Estimation Challenge**: When H and D are highly correlated in training data, the model struggles to learn their distinct effects. For example, if high H always co-occurs with high D, the model may learn a combined effect rather than separate contributions.

**Solution**: Use decorrelated sampling to ensure:
- Low correlation between H and D (|r| < 0.3)
- Good coverage of parameter space
- Sufficient samples for both parameters independently

### Sampling Strategy Comparison

| Strategy | Use Case | H-D Correlation | Coverage | Computational Cost |
|----------|----------|-----------------|----------|-------------------|
| Full Factorial | Trajectory prediction | High (if correlated) | Excellent | High (all combinations) |
| Latin Hypercube | Parameter estimation | Low (decorrelated) | Good | Medium |
| Sobol Sequence | Parameter estimation, Trajectory+CCT | Low (decorrelated) | Excellent | Medium |
| Boundary-Focused | CCT estimation (NOT recommended) | Medium | Good (boundaries) | Medium |

### Sampling Strategy Selection for Trajectory + CCT Prediction

**Important:** For trajectory + CCT prediction tasks, use **CCT-based sampling** with **Sobol sequences**, NOT boundary-focused sampling.

#### Two Types of Boundaries

1. **Temporal Boundary (Clearing Time)** - ✅ Handled by CCT-based sampling:
   - The bisection method finds CCT for each (H, D, Pm) combination
   - Samples clearing times around CCT (e.g., CCT ± 0.002s)
   - This is the **CRITICAL boundary** for CCT prediction
   - Already implemented via `use_cct_based_sampling=True`

2. **Parameter Space Boundary (H, D, Pm values)** - Would be handled by boundary-focused sampling:
   - Focuses on parameter combinations near range limits (e.g., H near 2.0 or 10.0)
   - **Less critical** because:
     - Sobol/LHS already covers boundaries well
     - Bisection finds CCT regardless of where (H, D, Pm) is in parameter space
     - CCT-based sampling already captures boundary behavior (unstable cases)

#### Why NOT Use Boundary-Focused Sampling for Trajectory + CCT?

- ✅ CCT-based sampling already handles the temporal boundary (the critical one)
- ✅ Sobol sequences naturally cover parameter space, including boundaries
- ✅ Bisection finds CCT for any (H, D, Pm) combination, not just near boundaries
- ⚠️ Boundary-focused sampling may reduce interior coverage

#### Optimal Strategy for Trajectory + CCT Prediction

- **Sobol sequences**: Best parameter space coverage (better than LHS)
- **CCT-based sampling**: Handles temporal boundary via bisection (already implemented)
- **Correlation analysis**: Diagnostic to ensure H and D are distinguishable (target: < 0.5)
- **Coverage analysis**: Validates uniform sampling
- **Stability balance**: Critical for CCT (target: 20-80% stable ratio)

## Usage Examples

### Complete Data Generation Workflow
```python
from data_generation import (
    generate_trajectory_data,
    preprocess_data,
    split_dataset,
    validate_data_for_task
)

# 1. Generate raw data
raw_data = generate_trajectory_data(
    n_samples=1000,
    H_range=(2.0, 10.0),
    D_range=(0.1, 2.0),
    sampling_method="lhs"
)

# 2. Preprocess
processed_data, scalers = preprocess_data(raw_data)

# 3. Split dataset
train_data, val_data, test_data = split_dataset(
    processed_data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# 4. Validate
validation_results = validate_data_for_task(
    train_data,
    task="trajectory_prediction"
)
```

### CCT Data Generation with Boundary Focus
```python
from data_generation import (
    generate_cct_data,
    validate_cct_data_quality,
    validate_data_quality,
    generate_data_quality_report,
)

# Generate CCT data with boundary-focused sampling
cct_data = generate_cct_data(
    n_samples=500,
    boundary_focus=True,
    stability_threshold=180.0,
    H_range=(2.0, 10.0),
    D_range=(0.1, 2.0)
)

# Low-level validation (NaN/Inf, time consistency)
is_valid, issues = validate_data_quality(sample_data)
if not is_valid:
    print(f"Data quality issues: {issues}")

# High-level validation (task-specific)
quality_metrics = validate_cct_data_quality(cct_data)
print(f"Stable samples: {quality_metrics['stable_ratio']:.2%}")
print(f"Unstable samples: {1 - quality_metrics['stable_ratio']:.2%}")

# Comprehensive quality report
quality_report = generate_data_quality_report(all_data)
print(f"Completeness: {quality_report['completeness']:.2%}")
print(f"Correlation H-D: {quality_report['correlation_H_D']:.4f}")
```

## Configuration

Data generation parameters can be configured via `configs/data_generation/data_generation.yaml`:
- Parameter ranges (H, D, fault scenarios)
- Sampling strategies
- Number of samples
- ANDES system configuration
- Output paths

## Notes

- Requires ANDES power system simulation tool
- Generated data is saved to `data/processed/` directory
- Supports both SMIB (Single Machine Infinite Bus) and multimachine systems
- Sampling strategies can be customized for specific tasks
- Validation ensures data quality before training
- Preprocessing scalers should be saved for inference-time normalization

## Related Documentation

- **[Decorrelated Sampling Guide](DECORRELATED_SAMPLING_GUIDE.md)** - Comprehensive explanation of why decorrelated sampling is critical for parameter estimation but not needed for trajectory prediction or CCT estimation

---

**[⬆ Back to Top](#data-generation-module)**
