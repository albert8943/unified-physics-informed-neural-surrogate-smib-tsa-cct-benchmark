# Comparison Directory

This directory contains comparison utilities for model approaches.

---

## Scripts

### `compare_approaches.py`

**Purpose**: Compare Pe Input vs Reactance Approaches.

**What it does:**
- Loads two trained models (Pe Input and Reactance-based)
- Evaluates both on the same test data
- Compares performance metrics (RMSE, MAE, R²)
- Generates comparison report

**Usage:**
```bash
python scripts/comparison/compare_approaches.py \
    --data-dir data \
    --model-pe-input outputs/models/pe_input_model.pth \
    --model-reactance outputs/models/reactance_model.pth \
    --output-dir outputs/comparison
```

**Output:**
- Comparison metrics (RMSE, MAE, R² for both approaches)
- Comparison report: `outputs/comparison/comparison_report.txt`

---

## When to Use

Use this script when you want to:
- Compare different model architectures
- Evaluate Pe Input vs Reactance approaches
- Generate comparison reports for publication

---

**Last Updated**: December 2024

