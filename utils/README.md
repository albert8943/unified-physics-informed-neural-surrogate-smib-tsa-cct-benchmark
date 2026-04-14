# Utilities Module

This directory contains reusable utility modules for PINN training, evaluation, and analysis. These modules are imported and used throughout the application codebase.

## 📑 Table of Contents

- [Overview](#overview)
- [Module Files](#module-files)
  - [metrics.py](#metricspy)
  - [data_utils.py](#data_utilspy)
  - [visualization.py](#visualizationpy)
  - [stability_checker.py](#stability_checkerpy)
  - [cct_binary_search.py](#cct_binary_searchpy)
- [Module Exports](#module-exports)
- [Usage Examples](#usage-examples)
  - [Complete Training Workflow](#complete-training-workflow)
  - [CCT Estimation Workflow](#cct-estimation-workflow)
- [Notes](#notes)

---

## Overview

The `utils/` module provides essential functionality for:
- **Metrics computation** - Evaluation metrics for trajectory prediction, CCT estimation, and parameter estimation
- **Data handling** - Dataset loading, saving, and batch preparation
- **Visualization** - Plotting functions for trajectories, loss curves, and comparisons
- **Stability checking** - Functions to check system stability from trajectories
- **CCT estimation** - Binary search-based CCT estimation using trained trajectory models

## Module Files

### `metrics.py`
**Purpose**: Comprehensive evaluation metrics for PINN models.

**Key Functions**:
- `compute_trajectory_metrics()` - Computes RMSE, MAE, MAPE, R², and first swing metrics for trajectory predictions (δ, ω)
- `compute_cct_metrics()` - Evaluates CCT estimation accuracy (absolute error, relative error, classification metrics)
- `compute_parameter_metrics()` - Assesses parameter estimation accuracy for H and D
- `compute_metrics()` - Generic metrics computation wrapper

**Usage**:
```python
from utils import compute_trajectory_metrics, compute_cct_metrics

metrics = compute_trajectory_metrics(delta_pred, omega_pred, delta_true, omega_true)
cct_metrics = compute_cct_metrics(cct_pred, cct_true)
```

---

### `data_utils.py`
**Purpose**: Data loading, saving, and batch preparation utilities.

**Key Components**:
- `TrajectoryDataset` - PyTorch Dataset class for trajectory prediction data
- `load_dataset()` - Load datasets from CSV files
- `save_dataset()` - Save datasets to CSV files
- `prepare_batch()` - Prepare batches for training (device placement)
- `create_dataloader()` - Create PyTorch DataLoader instances

**Usage**:
```python
from utils import load_dataset, TrajectoryDataset, create_dataloader

data = load_dataset("data/processed/training_data.csv")
dataset = TrajectoryDataset(data, feature_columns, target_columns)
dataloader = create_dataloader(dataset, batch_size=32, shuffle=True)
```

---

### `visualization.py`
**Purpose**: Plotting and visualization functions for PINN results.

**Key Functions**:
- `plot_trajectories()` - Plot predicted vs true trajectories (δ, ω) with fault timing markers
- `plot_loss_curves()` - Visualize training/validation loss curves and components
- `plot_comparison()` - Compare multiple methods or models side-by-side

**Usage**:
```python
from utils import plot_trajectories, plot_loss_curves

fig, axes = plot_trajectories(
    time, delta_pred, omega_pred,
    delta_true, omega_true,
    tf=fault_time, tc=clear_time,
    save_path="outputs/figures/trajectory.png"
)
```

---

### `stability_checker.py`
**Purpose**: Stability analysis functions for power system trajectories.

**Key Functions**:
- `check_stability()` - Determine if a single trajectory is stable or unstable
- `check_stability_batch()` - Batch processing for multiple trajectories

**Usage**:
```python
from utils import check_stability, check_stability_batch

is_stable = check_stability(delta_trajectory, omega_trajectory, threshold=180.0)
stability_results = check_stability_batch(trajectories)
```

---

### `cct_binary_search.py`
**Purpose**: Critical Clearing Time (CCT) estimation using binary search with trained trajectory models.

**Key Functions**:
- `estimate_cct_binary_search()` - Estimate CCT for a single scenario using binary search
- `estimate_cct_batch()` - Batch processing for multiple scenarios

**Algorithm**: Uses binary search to find the maximum fault clearing time that results in stable trajectories, leveraging the trained trajectory prediction PINN model.

**Usage**:
```python
from utils import estimate_cct_binary_search

cct_estimate = estimate_cct_binary_search(
    model=trajectory_model,
    initial_conditions=ic,
    system_params=params,
    fault_params=fault_params,
    tolerance=0.01,
    max_iterations=20
)
```

---

## Module Exports

The `utils/__init__.py` module exports the following functions:

```python
from utils import (
    # Metrics
    compute_metrics,
    compute_trajectory_metrics,
    compute_cct_metrics,
    compute_parameter_metrics,

    # Visualization
    plot_trajectories,
    plot_loss_curves,
    plot_comparison,

    # Data utilities
    load_dataset,
    save_dataset,
    prepare_batch,

    # Stability checking
    check_stability,
    check_stability_batch,

    # CCT estimation
    estimate_cct_binary_search,
    estimate_cct_batch,
)
```

## Usage Examples

### Complete Training Workflow
```python
from utils import (
    load_dataset, TrajectoryDataset, create_dataloader,
    compute_trajectory_metrics, plot_trajectories
)

# Load data
data = load_dataset("data/processed/training_data.csv")
dataset = TrajectoryDataset(data, feature_columns, target_columns)
dataloader = create_dataloader(dataset, batch_size=32)

# After training, evaluate
metrics = compute_trajectory_metrics(delta_pred, omega_pred, delta_true, omega_true)
print(f"RMSE: {metrics['delta_rmse']:.4f}")

# Visualize results
plot_trajectories(time, delta_pred, omega_pred, delta_true, omega_true)
```

### CCT Estimation Workflow
```python
from utils import estimate_cct_binary_search, compute_cct_metrics

# Estimate CCT using trained trajectory model
cct_estimate = estimate_cct_binary_search(
    model=trajectory_model,
    initial_conditions=ic,
    system_params=params,
    fault_params=fault_params
)

# Evaluate accuracy
cct_metrics = compute_cct_metrics(cct_estimate, cct_true)
print(f"CCT Error: {cct_metrics['absolute_error']:.4f} s")
```

## Notes

- All modules are designed to work with PyTorch tensors and NumPy arrays
- Functions are optimized for batch processing where applicable
- Visualization functions support both interactive display and file saving
- Metrics functions return dictionaries with standardized key names
- The CCT binary search implementation is the recommended approach (replaces the deprecated CCTEstimationPINN class)

---

**[⬆ Back to Top](#utilities-module)**
