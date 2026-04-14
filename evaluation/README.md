# Evaluation Module

This directory contains comprehensive evaluation frameworks for PINN models, including baseline comparisons, ablation studies, statistical analysis, and scalability assessments.

## 📑 Table of Contents

- [Overview](#overview)
- [Module Files](#module-files)
  - [comprehensive_evaluation.py](#comprehensive_evaluationpy)
  - [baseline_comparison.py](#baseline_comparisonpy)
  - [ablation_studies.py](#ablation_studiespy)
  - [statistical_analysis.py](#statistical_analysispy)
  - [scalability_analysis.py](#scalability_analysispy)
- [Module Exports](#module-exports)
- [Usage Examples](#usage-examples)
  - [Complete Evaluation Workflow](#complete-evaluation-workflow)
  - [Baseline Comparison](#baseline-comparison)
  - [Ablation Study](#ablation-study)
  - [Statistical Analysis](#statistical-analysis)
- [Output Formats](#output-formats)
  - [Evaluation Reports](#evaluation-reports)
  - [Report Structure](#report-structure)
- [Integration with Other Modules](#integration-with-other-modules)
- [Best Practices](#best-practices)
- [Notes](#notes)

---

## Overview

The `evaluation/` module provides:
- **Baseline Comparisons** - Compare PINN against traditional methods (EAC, TDS, standard ML)
- **Ablation Studies** - Analyze impact of different PINN components
- **Statistical Analysis** - Comprehensive statistical evaluation and reporting
- **Scalability Analysis** - Assess performance on larger systems
- **Comprehensive Evaluation** - End-to-end evaluation framework
- **Statistical Validation** ⭐ **NEW** - Multiple runs with different seeds for reproducibility
- **ML Baselines** ⭐ **NEW** - Standard NN and LSTM implementations without physics
- **GENROU Validation** ⭐ **NEW** - Validate PINN trained on GENCLS against GENROU
- **Energy Validation** ⭐ **NEW** - Energy-based physics consistency validation
- **CCT Comparison** ⭐ **NEW** - CCT estimation comparison framework
- **Failure Analysis** ⭐ **NEW** - Identify and analyze failure cases
- **Speed Benchmarking** ⭐ **NEW** - Computational speed comparison

## Module Files

### **New Publication Frameworks** ⭐ **NEW**

#### `baselines/ml_baselines.py`
**Purpose**: Standard ML baseline implementations (no physics constraints).

**Key Classes**:
- `StandardNN` - Feedforward neural network without physics
- `LSTMModel` - LSTM model for sequence prediction
- `MLBaselineTrainer` - Trainer for ML baseline models

**Usage**:
```bash
# Train ML baselines
python scripts/train_ml_baselines.py \
    --data-path data/train_data.csv \
    --models standard_nn lstm \
    --epochs 400
```

#### `genrou_validation.py`
**Purpose**: Validate PINN trained on GENCLS against GENROU (detailed generator model).

**Key Functions**:
- `validate_pinn_on_genrou()` - Main validation function
- `extract_pe_from_genrou()` - Extract Pe(t) from GENROU simulation
- `prepare_pinn_inputs()` - Prepare inputs for PINN prediction

**Usage**:
```bash
python scripts/validation/genrou_validation.py \
    --pinn-model outputs/experiments/exp_XXX/model/best_model.pth \
    --genrou-case smib/SMIB_genrou.xlsx \
    --test-scenarios data/test_data.csv
```

#### `energy_validation.py`
**Purpose**: Validate physics consistency using transient energy function.

**Key Functions**:
- `compute_transient_energy()` - Compute transient energy function
- `validate_energy_conservation()` - Validate energy conservation

#### `cct_comparison.py`
**Purpose**: Compare CCT estimation accuracy across methods.

**Key Functions**:
- `compare_cct_estimation()` - Compare PINN vs ANDES vs EAC
- `estimate_cct_pinn()` - Estimate CCT using PINN

#### `failure_analysis.py`
**Purpose**: Identify where/why PINN fails.

**Key Functions**:
- `identify_failure_cases()` - Find high error predictions
- `analyze_failure_patterns()` - Analyze failure patterns
- `analyze_near_instability_cases()` - Analyze near-instability cases

#### `benchmark_speed.py`
**Purpose**: Compare computational speed across methods.

**Key Functions**:
- `benchmark_inference_time()` - Benchmark model inference time
- `compare_speed()` - Compare speed across methods
- `compute_speedup()` - Compute speedup factors

**For complete usage, see**: [`docs/publication/STEP_BY_STEP_PUBLICATION_PLAN.md`](../docs/publication/STEP_BY_STEP_PUBLICATION_PLAN.md)

---

### `comprehensive_evaluation.py`
**Purpose**: Complete evaluation framework that combines all evaluation components.

**Key Components**:
- `ComprehensiveEvaluator` - Main evaluation class
  - PINN model evaluation with metrics
  - Baseline method comparisons
  - Statistical analysis
  - Report generation

**Features**:
- Evaluates PINN models across multiple tasks
- Compares against multiple baseline methods
- Generates publication-ready reports
- Supports batch evaluation

**Usage**:
```python
from evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(
    model_path="outputs/models/trajectory_model.pth",
    task="trajectory_prediction"
)

results = evaluator.evaluate(
    test_data=test_dataset,
    baselines=["EAC", "TDS", "ML"],
    generate_report=True
)
```

---

### `baseline_comparison.py`
**Purpose**: Compare PINN models against traditional power system analysis methods.

**Key Classes**:
- `BaselineComparator` - Main comparison framework
- `EACBaseline` - Equal Area Criterion baseline
- `TDSBaseline` - Time Domain Simulation baseline (ANDES)
- `MLBaseline` - Standard machine learning baseline (no physics)

**Baseline Methods**:
1. **EAC (Equal Area Criterion)**: Classical analytical method for CCT estimation
2. **TDS (Time Domain Simulation)**: Full numerical simulation (ground truth)
3. **Standard ML**: Neural network without physics constraints

**Metrics**: Compares accuracy, speed, and computational cost

**Usage**:
```python
from evaluation import BaselineComparator, EACBaseline, TDSBaseline

comparator = BaselineComparator()

# Compare CCT estimation
results = comparator.compare_cct_estimation(
    pinn_model=trajectory_model,
    eac_baseline=EACBaseline(),
    tds_baseline=TDSBaseline(),
    test_scenarios=test_data
)

# Compare trajectory prediction
trajectory_results = comparator.compare_trajectory_prediction(
    pinn_model=trajectory_model,
    tds_baseline=TDSBaseline(),
    test_scenarios=test_data
)
```

---

### `ablation_studies.py`
**Purpose**: Analyze the impact of different PINN components through ablation studies.

**Key Classes**:
- `AblationStudy` - Base class for ablation studies
- `PhysicsLossAblation` - Study impact of physics loss weight
- `ArchitectureAblation` - Study impact of network architecture
- `CollocationAblation` - Study impact of collocation point strategy

**Ablation Dimensions**:
1. **Physics Loss Weight** (`lambda_physics`): 0.0 (no physics) to 1.0 (strong physics)
2. **Network Architecture**: Depth, width, activation functions
3. **Collocation Points**: Number, distribution, sampling strategy
4. **Loss Components**: Data loss, physics loss, IC loss, boundary loss

**Usage**:
```python
from evaluation import (
    PhysicsLossAblation,
    ArchitectureAblation,
    CollocationAblation
)

# Physics loss ablation
physics_study = PhysicsLossAblation()
results = physics_study.run(
    base_model=trajectory_model,
    lambda_physics_range=[0.0, 0.01, 0.1, 0.5, 1.0],
    test_data=test_dataset
)

# Architecture ablation
arch_study = ArchitectureAblation()
results = arch_study.run(
    architectures=[
        {"hidden_dims": [32, 32]},
        {"hidden_dims": [64, 64, 64]},
        {"hidden_dims": [128, 128, 128, 128]}
    ],
    test_data=test_dataset
)
```

---

### `statistical_analysis.py`
**Purpose**: Comprehensive statistical analysis and reporting.

**Key Functions**:
- `compute_statistics()` - Compute descriptive statistics
- `compare_methods()` - Statistical comparison between methods
- `generate_statistical_report()` - Generate detailed statistical report
- `hypothesis_testing()` - Perform statistical hypothesis tests

**Statistical Measures**:
- Mean, median, standard deviation
- Confidence intervals
- Percentiles (25th, 75th, 95th)
- Correlation analysis
- Distribution analysis

**Usage**:
```python
from evaluation import (
    compute_statistics,
    compare_methods,
    generate_statistical_report
)

# Compute statistics
stats = compute_statistics(
    predictions=pinn_predictions,
    ground_truth=true_values,
    metrics=["rmse", "mae", "r2"]
)

# Compare methods
comparison = compare_methods(
    method_results={
        "PINN": pinn_results,
        "EAC": eac_results,
        "TDS": tds_results
    },
    metric="cct_error"
)

# Generate report
report = generate_statistical_report(
    all_results=comparison,
    output_path="outputs/results/statistical_report.md"
)
```

---

### `scalability_analysis.py`
**Purpose**: Assess PINN performance and scalability on larger power systems.

**Key Analyses**:
- **System Size Scaling**: Performance vs number of machines
- **Computation Time**: Training and inference time analysis
- **Memory Usage**: Memory requirements for different system sizes
- **Accuracy Scaling**: How accuracy changes with system complexity

**Usage**:
```python
from evaluation import ScalabilityAnalyzer

analyzer = ScalabilityAnalyzer()

# Analyze scalability
results = analyzer.analyze(
    models={
        "SMIB": smib_model,
        "2-machine": two_machine_model,
        "3-machine": three_machine_model
    },
    test_systems=test_systems,
    metrics=["accuracy", "inference_time", "memory"]
)

# Generate scalability report
analyzer.generate_report(
    results=results,
    output_path="outputs/results/scalability_analysis.md"
)
```

---

## Module Exports

The `evaluation/__init__.py` module exports:

```python
from evaluation import (
    # Baseline comparison
    BaselineComparator,
    EACBaseline,
    TDSBaseline,
    MLBaseline,

    # Ablation studies
    AblationStudy,
    PhysicsLossAblation,
    ArchitectureAblation,
    CollocationAblation,
)
```

---

## Usage Examples

### Complete Evaluation Workflow
```python
from evaluation import ComprehensiveEvaluator
from utils import load_dataset

# Load test data
test_data = load_dataset("data/processed/test_data.csv")

# Initialize evaluator
evaluator = ComprehensiveEvaluator(
    model_path="outputs/models/trajectory_model.pth",
    task="trajectory_prediction"
)

# Run comprehensive evaluation
results = evaluator.evaluate(
    test_data=test_data,
    baselines=["EAC", "TDS"],
    generate_report=True,
    output_dir="outputs/results"
)

# Results include:
# - PINN metrics
# - Baseline comparisons
# - Statistical analysis
# - Visualization plots
```

### Baseline Comparison
```python
from evaluation import BaselineComparator, EACBaseline, TDSBaseline

comparator = BaselineComparator()

# Compare CCT estimation methods
cct_comparison = comparator.compare_cct_estimation(
    pinn_model=trajectory_model,
    eac_baseline=EACBaseline(),
    tds_baseline=TDSBaseline(),
    test_scenarios=test_data
)

print(f"PINN CCT Error: {cct_comparison['pinn']['mean_error']:.4f} s")
print(f"EAC CCT Error: {cct_comparison['eac']['mean_error']:.4f} s")
print(f"TDS (ground truth)")

# Compare speed
print(f"PINN Inference Time: {cct_comparison['pinn']['inference_time']:.4f} s")
print(f"EAC Computation Time: {cct_comparison['eac']['computation_time']:.4f} s")
```

### Ablation Study
```python
from evaluation import PhysicsLossAblation

# Study impact of physics loss weight
study = PhysicsLossAblation()
results = study.run(
    base_model=trajectory_model,
    lambda_physics_range=[0.0, 0.01, 0.1, 0.5, 1.0],
    test_data=test_dataset
)

# Analyze results
for lambda_phys, metrics in results.items():
    print(f"λ_physics={lambda_phys}: RMSE={metrics['rmse']:.4f}")
```

### Statistical Analysis
```python
from evaluation import compute_statistics, compare_methods

# Compute statistics for PINN
pinn_stats = compute_statistics(
    predictions=pinn_predictions,
    ground_truth=true_values
)

# Compare multiple methods
comparison = compare_methods(
    method_results={
        "PINN": pinn_results,
        "EAC": eac_results,
        "Standard ML": ml_results
    },
    metric="cct_error"
)

# Generate report
from evaluation import generate_statistical_report
report = generate_statistical_report(
    all_results=comparison,
    output_path="outputs/results/statistical_report.md"
)
```

---

## Output Formats

### Evaluation Reports
- **Markdown reports**: Human-readable evaluation summaries
- **JSON results**: Machine-readable results for further analysis
- **CSV metrics**: Tabular format for spreadsheet analysis
- **Visualization plots**: Comparison plots, error distributions, etc.

### Report Structure
1. **Executive Summary**: Key findings and metrics
2. **Method Comparison**: Detailed comparison tables
3. **Statistical Analysis**: Statistical tests and confidence intervals
4. **Visualizations**: Plots and figures
5. **Appendix**: Detailed results and configurations

---

## Integration with Other Modules

The evaluation module integrates with:
- **`utils.metrics`**: Uses metric computation functions
- **`utils.visualization`**: Uses plotting functions
- **`pinn`**: Evaluates trained PINN models
- **`data_generation`**: Uses generated test datasets

---

## Best Practices

1. **Use representative test sets**: Ensure test data covers parameter space
2. **Compare multiple baselines**: EAC, TDS, and standard ML
3. **Run statistical tests**: Use hypothesis testing for method comparison
4. **Report confidence intervals**: Include uncertainty estimates
5. **Visualize results**: Generate plots for better understanding
6. **Document configurations**: Save evaluation configurations for reproducibility

---

## Notes

- All evaluation scripts support both CPU and GPU
- Results are saved to `outputs/results/` directory
- Baseline comparisons require ANDES for TDS baseline
- Ablation studies can be computationally expensive (multiple model trainings)
- Statistical analysis includes proper hypothesis testing and confidence intervals
- Scalability analysis helps assess practical applicability to larger systems

---

**[⬆ Back to Top](#evaluation-module)**
