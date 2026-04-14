# Importance of Decorrelated Sampling for Each Task

This guide explains why decorrelated sampling is critical for parameter estimation but not needed for trajectory prediction or CCT estimation.

## What is Correlation Coefficient (r)?

**`r`** is the **Pearson correlation coefficient** between H and D parameters. It measures how linearly related H and D are in your training data.

### Correlation Coefficient Range
- **r = +1.0**: Perfect positive correlation (when H increases, D always increases proportionally)
- **r = 0.0**: No correlation (H and D vary independently)
- **r = -1.0**: Perfect negative correlation (when H increases, D always decreases proportionally)

### How It's Calculated
The code uses `numpy.corrcoef()` which computes:
```
r = Σ[(H_i - H_mean) × (D_i - D_mean)] / √[Σ(H_i - H_mean)² × Σ(D_i - D_mean)²]
```

### What |r| < 0.3 Means
- **|r|** = absolute value of correlation (ignores positive/negative)
- **|r| < 0.3** means the correlation is between -0.3 and +0.3
- This indicates **weak correlation** - H and D vary somewhat independently

### Examples

**High Correlation (r ≈ 0.95) - ❌ BAD for Parameter Estimation:**
```
Sample  H    D
1       2.0  0.5
2       4.0  1.0
3       6.0  1.5
4       8.0  2.0
5       10.0 2.5

Pattern: H and D increase together (r ≈ 0.95)
Problem: Model learns "high H always means high D"
```

**Low Correlation (r ≈ 0.1) - ✅ GOOD for Parameter Estimation:**
```
Sample  H    D
1       2.0  2.5
2       4.0  0.5
3       6.0  2.0
4       8.0  1.0
5       10.0 1.5

Pattern: H and D vary independently (r ≈ 0.1)
Solution: Model learns distinct H and D effects
```

## Quick Summary

| Task | Decorrelated Sampling Needed? | Why? |
|------|------------------------------|------|
| **Trajectory Prediction** | ❌ **NO** | Forward problem - model learns mapping from (H, D) → trajectories |
| **CCT Estimation** | ❌ **NO** | Uses binary search with trajectory model - no separate training needed |
| **Parameter Estimation** | ✅ **YES - CRITICAL** | Inverse problem - model must distinguish H and D effects separately |

---

## 1. Trajectory Prediction (Forward Problem)

### What the Model Learns
- **Input**: (H, D, fault_clearing_time, initial conditions)
- **Output**: Trajectories δ(t), ω(t)
- **Task**: Predict system response given known parameters

### Why Decorrelated Sampling is NOT Needed

**The model learns a forward mapping:**
```
(H, D, tc) → [δ(t), ω(t)]
```

**Example with correlated data:**
- If high H always co-occurs with high D in training data:
  - Model sees: (H=8.0, D=2.0) → trajectory_A
  - Model sees: (H=8.0, D=2.0) → trajectory_B
  - Model learns: "When H=8.0 and D=2.0, predict trajectory_X"
  - **This works fine!** The model doesn't need to separate H and D effects.

**What matters:**
- ✅ Good coverage of parameter space (H, D combinations)
- ✅ Diverse fault scenarios
- ✅ Uniform sampling (full factorial or LHS)

**What doesn't matter:**
- ❌ H-D correlation (can be high or low)
- ❌ Whether H and D are independent

### Recommended Strategy
```python
from data_generation import generate_trajectory_data

# Full factorial or LHS - correlation doesn't matter
trajectory_data = generate_trajectory_data(
    case_file="smib/SMIB.json",
    H_range=(2.0, 10.0, 5),
    D_range=(0.5, 3.0, 5),
    sampling_strategy="full_factorial"  # or "latin_hypercube"
)
```

---

## 2. CCT Estimation (Via Binary Search)

### What the Model Uses
- **No separate model** - Uses the trained trajectory prediction model
- **Algorithm**: Binary search to find maximum stable clearing time
- **Task**: Find CCT by testing different clearing times

### Why Decorrelated Sampling is NOT Needed

**CCT estimation process:**
1. Load trained trajectory prediction model
2. For a given (H, D) combination, test different clearing times
3. Use binary search to find maximum stable clearing time
4. Return CCT estimate

**The trajectory model already handles (H, D) correctly:**
- Since trajectory prediction doesn't need decorrelated data, CCT estimation inherits this property
- CCT estimation is **algorithm-based**, not model-based
- No separate training data needed

### Recommended Strategy
```python
# Use the same data as trajectory prediction
# OR use boundary-focused sampling for better CCT boundary coverage
from data_generation import generate_cct_data

cct_data = generate_cct_data(
    case_file="smib/SMIB.json",
    boundary_focus=True,  # Focus on stability boundaries
    H_range=(2.0, 10.0, 5),
    D_range=(0.5, 3.0, 5)
)
```

---

## 3. Parameter Estimation (Inverse Problem) ⚠️ **CRITICAL**

### What the Model Learns
- **Input**: Observed trajectories [δ(t), ω(t)]
- **Output**: Estimated parameters (H, D)
- **Task**: Infer system parameters from observed behavior

### Why Decorrelated Sampling IS CRITICAL

**The model learns an inverse mapping:**
```
[δ(t), ω(t)] → (H, D)
```

**Problem with correlated data:**

**Scenario A: High H-D Correlation (❌ BAD)**
```
Training Data:
- (H=2.0, D=0.5) → trajectory_1
- (H=4.0, D=1.0) → trajectory_2
- (H=6.0, D=1.5) → trajectory_3
- (H=8.0, D=2.0) → trajectory_4
- (H=10.0, D=2.5) → trajectory_5

Correlation: r(H, D) ≈ 0.95 (very high!)
```

**What the model learns:**
- Model sees: "High H always comes with high D"
- Model learns: "If trajectory looks like trajectory_4, then H≈8.0 AND D≈2.0"
- **Problem**: Model learns a **combined effect** (H+D together)
- **Cannot distinguish**: Is the trajectory due to H=8.0 or D=2.0?

**Test case:**
- Given trajectory from (H=8.0, D=0.5) - **unseen combination**
- Model predicts: (H≈8.0, D≈2.0) ❌ **WRONG!**
- Model assumes: "High H means high D" (learned from training)

**Scenario B: Low H-D Correlation (✅ GOOD)**
```
Training Data:
- (H=2.0, D=2.5) → trajectory_1
- (H=4.0, D=0.5) → trajectory_2
- (H=6.0, D=2.0) → trajectory_3
- (H=8.0, D=1.0) → trajectory_4
- (H=10.0, D=1.5) → trajectory_5

Correlation: r(H, D) ≈ 0.1 (low!)
```

**What the model learns:**
- Model sees: "H and D vary independently"
- Model learns: "Trajectory features that indicate H vs. features that indicate D"
- **Success**: Model learns **distinct effects** of H and D separately
- **Can distinguish**: "This trajectory feature is due to H, that feature is due to D"

**Test case:**
- Given trajectory from (H=8.0, D=0.5) - **unseen combination**
- Model predicts: (H≈8.0, D≈0.5) ✅ **CORRECT!**
- Model recognizes: "High H signature" and "Low D signature" independently

### Mathematical Explanation

**High Correlation Problem:**
- If H and D are highly correlated, the model learns: `f(δ, ω) ≈ g(H + αD)` where α is a constant
- The model cannot separate: `f(δ, ω) = g₁(H) + g₂(D)`
- Result: Model predicts H and D as a combined quantity

**Low Correlation Solution:**
- With decorrelated data, the model learns: `f(δ, ω) = g₁(H) + g₂(D)`
- The model can identify:
  - Trajectory features that depend on H (inertia effects)
  - Trajectory features that depend on D (damping effects)
- Result: Model predicts H and D independently

### Recommended Strategy
```python
from data_generation import generate_parameter_estimation_data

# MUST use decorrelated sampling
param_data = generate_parameter_estimation_data(
    case_file="smib/SMIB.json",
    H_range=(2.0, 10.0, 5),
    D_range=(0.5, 3.0, 5),
    target_correlation=0.0,  # Target: zero correlation
    correlation_tolerance=0.1  # Acceptable: |r| < 0.1
)

# Validate correlation
from data_generation import correlation_analysis
corr_info = correlation_analysis(param_data[['H', 'D']].values)
print(f"H-D correlation: {corr_info['max_correlation']:.4f}")
# Should be: |correlation| < 0.3 (ideally < 0.1)
```

---

## Visual Comparison

### Trajectory Prediction (Forward Problem)
```
Training Data (correlated is OK):
H    D    →  Trajectory
2.0  0.5  →  [δ₁(t), ω₁(t)]
4.0  1.0  →  [δ₂(t), ω₂(t)]
6.0  1.5  →  [δ₃(t), ω₃(t)]
8.0  2.0  →  [δ₄(t), ω₄(t)]

Model learns: (H, D) → Trajectory
✅ Works fine even with high correlation
```

### Parameter Estimation (Inverse Problem)
```
Training Data (MUST be decorrelated):
Trajectory          →  H    D
[δ₁(t), ω₁(t)]  →  2.0  2.5  ← Low H, High D
[δ₂(t), ω₂(t)]  →  4.0  0.5  ← Medium H, Low D
[δ₃(t), ω₃(t)]  →  6.0  2.0  ← High H, Medium D
[δ₄(t), ω₄(t)]  →  8.0  1.0  ← High H, Low D

Model learns: Trajectory → (H, D)
❌ High correlation: Model learns combined effect
✅ Low correlation: Model learns distinct effects
```

---

## Key Takeaways

1. **Trajectory Prediction**:
   - Decorrelated sampling: **NOT needed**
   - Focus: Good parameter space coverage
   - Correlation: Doesn't matter

2. **CCT Estimation**:
   - Decorrelated sampling: **NOT needed**
   - Uses trajectory model (no separate training)
   - Focus: Boundary coverage (optional)

3. **Parameter Estimation**:
   - Decorrelated sampling: **CRITICAL** ⚠️
   - Target correlation: |r| < 0.3 (ideally |r| < 0.1)
   - Without it: Model cannot distinguish H and D effects
   - With it: Model learns independent H and D contributions

---

## Validation Checklist

**For Parameter Estimation Data:**
```python
from data_generation import validate_data_for_task
from data_generation import correlation_analysis
import numpy as np

# Method 1: Using validation function
validation = validate_data_for_task(
    data=param_data,
    task="parameter_estimation",
    max_correlation=0.3  # Maximum allowed |r| (absolute correlation)
)

if validation['validation_passed']:
    print("✅ Data is suitable for parameter estimation")
    print(f"   H-D correlation (r): {validation['max_correlation']:.4f}")
    print(f"   |r| = {abs(validation['max_correlation']):.4f} < 0.3 ✓")
else:
    print("❌ Data has high H-D correlation!")
    print(f"   |r| = {abs(validation['max_correlation']):.4f} >= 0.3")
    print("   Regenerate with decorrelated sampling")

# Method 2: Direct correlation calculation
H_values = param_data['H'].values
D_values = param_data['D'].values
r = np.corrcoef(H_values, D_values)[0, 1]  # Correlation coefficient

print(f"\nDirect calculation:")
print(f"   Correlation coefficient (r): {r:.4f}")
print(f"   Absolute correlation |r|: {abs(r):.4f}")
if abs(r) < 0.3:
    print(f"   ✅ |r| = {abs(r):.4f} < 0.3 (acceptable)")
    if abs(r) < 0.1:
        print(f"   ✅ |r| = {abs(r):.4f} < 0.1 (ideal!)")
else:
    print(f"   ❌ |r| = {abs(r):.4f} >= 0.3 (too high!)")
```

**Understanding the Output:**
- **r = 0.05**: Very low correlation (ideal) - H and D are nearly independent
- **r = 0.15**: Low correlation (good) - H and D vary somewhat independently
- **r = 0.25**: Moderate correlation (acceptable) - Some dependence, but still usable
- **r = 0.35**: High correlation (bad) - H and D are too dependent
- **r = 0.85**: Very high correlation (very bad) - H and D are almost perfectly correlated

---

## References

- See [`data_generation/README.md`](README.md#data-generation-best-practices) for detailed strategies
- See [`data_generation/sampling_strategies.py`](sampling_strategies.py) for decorrelated sampling implementation
- See [`data_generation/parameter_sweep.py`](parameter_sweep.py) for task-specific data generation functions
