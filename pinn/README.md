# PINN Module

This directory contains the core Physics-Informed Neural Network (PINN) implementations for power system transient stability analysis.

## 📑 Table of Contents

- [Overview](#overview)
- [Module Files](#module-files)
  - [core.py](#corepy)
  - [trajectory_prediction.py](#trajectory_predictionpy)
  - [parameter_estimation.py](#parameter_estimationpy)
  - [cct_estimation.py](#cct_estimationpy)
  - [multimachine.py](#multimachinepy)
  - [detailed_generator.py](#detailed_generatorpy)
- [Module Exports](#module-exports)
- [Usage Examples](#usage-examples)
  - [Trajectory Prediction](#trajectory-prediction)
  - [Parameter Estimation](#parameter-estimation)
- [Key Features](#key-features)
  - [Physics-Informed Loss Function](#physics-informed-loss-function)
  - [State-Aware Modeling](#state-aware-modeling)
- [Neural Network Architecture](#neural-network-architecture)
  - [Architecture Components](#architecture-components)
  - [Architecture Selection Rationale](#architecture-selection-rationale)
  - [Architecture Details](#architecture-details)
- [Training Considerations](#training-considerations)
- [Theoretical Background](#theoretical-background)
  - [Swing Equation](#swing-equation)
  - [Electrical Power with State-Aware Reactance](#electrical-power-with-state-aware-reactance)
- [Notes](#notes)

---

## Overview

The `pinn/` module implements PINN models for three main tasks:
- **Trajectory Prediction** - Forward problem: predict rotor angle (δ) and speed (ω) trajectories
- **Parameter Estimation** - Inverse problem: estimate machine inertia (H) and damping (D) from trajectories
- **CCT Estimation** - Critical Clearing Time estimation (deprecated in favor of binary search approach)

All models incorporate physics constraints based on the swing equation to ensure physically consistent predictions.

## Module Files

### `core.py`
**Purpose**: Base classes and physics-informed loss functions.

**Key Components**:
- `PINN` - Base class for all PINN models with common functionality
  - Neural network architecture management
  - Forward pass with physics constraints
  - Training loop utilities
- `PhysicsInformedLoss` - Loss function combining:
  - **Data loss**: MSE between predicted and observed trajectories
  - **Physics loss**: Residual of swing equation at collocation points
  - **Initial condition loss**: MSE for initial conditions
  - **Boundary condition loss**: Constraints at fault clearing times

**Physics Constraints**: Implements the swing equation:
```
M·d²δ/dt² + D·dδ/dt = Pm - Pe(δ)
```

**Usage**:
```python
from pinn import PINN, PhysicsInformedLoss

loss_fn = PhysicsInformedLoss(
    lambda_data=1.0,
    lambda_physics=0.1,
    lambda_ic=10.0,
    lambda_boundary=1.0
)
```

---

### `trajectory_prediction.py`
**Purpose**: PINN model for predicting rotor angle and speed trajectories under transient faults.

**Class**: `TrajectoryPredictionPINN`

**Input**:
- Time `t`
- Initial conditions: `δ₀`, `ω₀`
- System parameters: `H` (inertia), `D` (damping), `P_m` (mechanical power)
- Network reactances: `Xprefault`, `Xfault`, `Xpostfault`
- Fault parameters: `tf` (fault start), `tc` (fault clear)

**Input vector**: `[t, δ₀, ω₀, H, D, P_m, Xprefault, Xfault, Xpostfault, tf, tc]` (11 dimensions)

**Output**: Predicted trajectories `δ(t)`, `ω(t)`

**Architecture**:
- Fully connected neural network with residual connections
- Configurable hidden layers and activation functions
- Physics-informed loss with swing equation constraints

**Usage**:
```python
from pinn import TrajectoryPredictionPINN

model = TrajectoryPredictionPINN(
    input_dim=11,  # [t, δ₀, ω₀, H, D, P_m, Xprefault, Xfault, Xpostfault, tf, tc]
    hidden_dims=[64, 64, 64, 64],
    activation="tanh",
    use_residual=True
)

# Training
predictions = model(t, features)
loss = loss_fn(predictions, targets, t, collocation_points)
```

---

### `parameter_estimation.py`
**Purpose**: PINN model for estimating machine parameters (H, D) from observed trajectories.

**Class**: `ParameterEstimationPINN`

**Input**:
- Observed trajectories: `δ(t)`, `ω(t)` (time series)
- System configuration (network reactances, fault parameters)

**Output**: Estimated parameters `H` (or `M = 2H`) and `D`

**Architecture**:
- Sequence-to-value network (LSTM or fully connected)
- Processes time series data to extract parameter estimates
- Physics constraints ensure estimated parameters satisfy swing equation

**Usage**:
```python
from pinn import ParameterEstimationPINN

model = ParameterEstimationPINN(
    input_dim=2,  # delta and omega
    sequence_length=100,
    hidden_dims=[128, 128, 64],
    use_lstm=True
)

# Training
H_est, D_est = model(delta_trajectory, omega_trajectory)
```

---

### `cct_estimation.py`
**Purpose**: **DEPRECATED** - CCT estimation PINN model.

**Class**: `CCTEstimationPINN`

**Status**: This class is deprecated. The recommended approach is to use binary search with the trajectory prediction model (see `utils.cct_binary_search.estimate_cct_binary_search()`).

**Note**: Kept for reference but not recommended for new implementations.

---

### `multimachine.py`
**Purpose**: Extension of PINN framework for multimachine power systems.

**Status**: Extension module for scaling to multiple machines (beyond single-machine infinite bus systems).

---

### `detailed_generator.py`
**Purpose**: Advanced PINN architecture with detailed physics constraints.

**Status**: Enhanced generator with more sophisticated physics-informed components.

---

## Module Exports

The `pinn/__init__.py` module exports:

```python
from pinn import (
    PINN,                    # Base class
    PhysicsInformedLoss,      # Physics-informed loss function
    TrajectoryPredictionPINN, # Trajectory prediction model
    ParameterEstimationPINN,  # Parameter estimation model
    CCTEstimationPINN,        # Deprecated - use binary search instead
)
```

## Usage Examples

### Trajectory Prediction
```python
from pinn import TrajectoryPredictionPINN, PhysicsInformedLoss
import torch

# Initialize model
model = TrajectoryPredictionPINN(
    input_dim=11,  # [t, δ₀, ω₀, H, D, P_m, Xprefault, Xfault, Xpostfault, tf, tc]
    hidden_dims=[64, 64, 64, 64],
    activation="tanh"
)

# Initialize loss function
loss_fn = PhysicsInformedLoss(
    lambda_data=1.0,
    lambda_physics=0.1,
    lambda_ic=10.0
)

# Forward pass
t = torch.linspace(0, 2.0, 100)
features = torch.randn(100, 11)  # [t, δ₀, ω₀, H, D, P_m, Xpre, Xf, Xpost, tf, tc]
delta_pred, omega_pred = model(t, features)
```

### Parameter Estimation
```python
from pinn import ParameterEstimationPINN

# Initialize model
model = ParameterEstimationPINN(
    input_dim=2,  # [delta, omega] time series
    sequence_length=100,
    use_lstm=True
)

# Forward pass
delta_traj = torch.randn(100, 1)  # Time series
omega_traj = torch.randn(100, 1)  # Time series
H_est, D_est = model(delta_traj, omega_traj)
```

## Key Features

### Physics-Informed Loss Function

The framework implements a comprehensive physics-informed loss function that enforces:

- **Swing Equation**: M·d²δ/dt² + D·dδ/dt = Pm - Pe(t)
  - *Explanation*: Describes how a generator's rotor angle (δ) changes over time. The left side represents inertia (M) and damping (D) forces, while the right side is the difference between mechanical power input (Pm) and electrical power output (Pe). This is the fundamental equation of generator dynamics.

- **Electrical Power**: Pe(t) = (V₁·V₂/X(t))·sin(δ(t)) with state-aware reactance switching
  - *Explanation*: Calculates the electrical power flowing between two buses with voltages V₁ and V₂. The power depends on the angle difference δ(t) and the transmission line reactance X(t), which changes during faults (pre-fault, during-fault, post-fault states).

- **Speed-Angle Relation**: dδ/dt = ω - 1.0
  - *Explanation*: Links the rotor angle (δ) to the rotor speed (ω). The derivative of angle equals the speed deviation from the synchronous speed (normalized to 1.0 per unit). This connects the generator's rotational motion to its electrical angle.

- **Initial Conditions**: δ(t=0) = δ₀ and ω(t=0) = ω₀
  - *Explanation*: Specifies the starting values at time zero: the initial rotor angle (δ₀) and initial rotor speed (ω₀). These values are needed to solve the differential equations and represent the system's state before any disturbance occurs.

- **Boundary Conditions**: Continuity at fault application/clearing times
  - *Explanation*: Ensures that the rotor angle and speed remain continuous (no sudden jumps) when a fault is applied or cleared. The system transitions smoothly between pre-fault, during-fault, and post-fault states, maintaining physical consistency.

### State-Aware Modeling

The framework handles three distinct system states during three-phase short-circuit studies:

- **Pre-fault**: X(t) = Xprefault (all lines in service)
  - *Explanation*: The normal operating condition before any fault occurs. All transmission lines are connected and operating normally, so the reactance X(t) equals the pre-fault value. This represents the steady-state condition of the power system.

- **During fault**: X(t) = Xfault (very small, ~0.0001 pu)
  - *Explanation*: When a short-circuit fault occurs (e.g., a three-phase fault), the effective reactance between buses becomes very small (approximately 0.0001 per unit). This creates a low-impedance path that dramatically increases current flow and reduces the electrical power transfer capability.

- **Post-fault**: X(t) = Xpostfault (may differ if line is tripped)
  - *Explanation*: After the fault is cleared by protective devices, the system enters the post-fault state. If a transmission line was disconnected (tripped) to clear the fault, the reactance X(t) will be different from the pre-fault value because the system topology has changed. This represents the new steady-state configuration after the disturbance.

## Neural Network Architecture

The framework uses deep feedforward neural networks (also known as multilayer perceptrons) as the core architecture for learning the physics-informed mappings. These networks are specifically designed to handle the smooth, continuous nature of power system dynamics.

### Architecture Components

**Multi-layer Feedforward Networks**:
- The networks consist of multiple fully-connected (dense) layers that transform inputs through a series of nonlinear transformations
- **Configurable depth/width**: The number of layers (depth) and neurons per layer (width) can be adjusted based on problem complexity
  - Deeper networks can capture more complex patterns but require more data and training time
  - Wider networks provide more capacity but may be prone to overfitting
  - Default configuration: 4 layers with 64 neurons each (balanced for SMIB systems)

**Understanding Overfitting and Underfitting**:
- **Overfitting**: Occurs when a neural network learns the training data too well, including noise and specific patterns that don't generalize to new data. The model performs excellently on training data but poorly on validation/test data. This often happens with networks that are too complex (too many parameters) relative to the amount of training data available. In PINN applications, overfitting can cause the network to memorize training trajectories rather than learning the underlying physics.
- **Underfitting**: Occurs when a neural network is too simple to capture the underlying patterns in the data. The model performs poorly on both training and validation data because it lacks sufficient capacity to learn the complex relationships. This typically happens with networks that are too small or shallow for the problem complexity.
- **Balanced Fit**: The goal is to find a network architecture that is complex enough to learn the physics-informed relationships but not so complex that it memorizes training data. Regularization techniques (e.g., weight decay, dropout) and appropriate network sizing help achieve this balance.

**Residual Connections** (optional):
- Skip connections that allow information to flow directly from earlier layers to later layers
- Help with training deeper networks by mitigating the vanishing gradient problem
- Particularly useful for learning complex temporal dependencies in trajectory prediction
- Enable the network to learn both detailed features and global patterns

**Tanh Activation Function**:
- Hyperbolic tangent activation (tanh) produces smooth, bounded outputs in the range [-1, 1]
- **Why tanh for PINNs?**:
  - Smoothness is critical for physics-informed learning, as derivatives (used in physics loss) must be well-defined
  - Bounded outputs prevent numerical instabilities during training
  - Smooth activation enables accurate computation of higher-order derivatives needed for the swing equation (d²δ/dt²)
- Alternative activations like ReLU are less suitable here due to their non-smooth nature at zero

**Xavier Weight Initialization**:
- Also known as Glorot initialization, this method initializes network weights based on the number of input and output neurons
- Prevents weights from being too large (which causes saturation) or too small (which causes slow learning)
- Ensures that activations and gradients flow properly through the network at the start of training
- Particularly important for deep networks and helps achieve faster convergence

**Understanding Weight Initialization**:
- **General Idea**: Weight initialization sets the starting values of network parameters (weights and biases) before training begins. This is crucial because random initialization can significantly impact training success. Poor initialization can lead to vanishing gradients (weights too small), exploding gradients (weights too large), or saturation (activations stuck at extreme values).

- **Why Xavier/Glorot Initialization?**:
  - Xavier initialization (named after Xavier Glorot) is designed specifically for networks using tanh or sigmoid activation functions
  - It sets weights by sampling from a uniform or normal distribution with variance = 1/(fan_in + fan_out), where fan_in is the number of input neurons and fan_out is the number of output neurons
  - This variance choice ensures that the variance of activations and gradients remains roughly constant as they propagate through the network, preventing the vanishing/exploding gradient problems
  - For tanh activations (used in this PINN framework), Xavier initialization is particularly well-suited because it maintains the variance of activations in the linear region of tanh

- **Alternative Initialization Methods**:
  - **He Initialization** (Kaiming): Designed for ReLU and its variants. Uses variance = 2/fan_in, accounting for ReLU's zero half-plane that reduces variance by half
  - **LeCun Initialization**: Similar to Xavier but uses variance = 1/fan_in. Good for tanh/sigmoid but less commonly used than Xavier
  - **Random Initialization**: Simple uniform or normal distribution with fixed bounds (e.g., [-0.1, 0.1]). Works for small networks but can fail for deep networks
  - **Orthogonal Initialization**: Initializes weights as orthogonal matrices, preserving gradient norms. Useful for recurrent networks
  - **Zero Initialization**: Setting all weights to zero. This is problematic because all neurons would learn the same thing (symmetry breaking issue)

- **Why It Matters for PINNs**:
  - Proper initialization ensures that the network starts in a regime where gradients flow well, enabling effective learning of physics constraints
  - Since PINNs compute derivatives (for physics loss), maintaining good gradient flow from the start is essential
  - Poor initialization can cause the network to get stuck in poor local minima or fail to learn the physics relationships

**PyTorch Implementation**:
- Built on PyTorch for flexibility and efficient automatic differentiation
- Automatic differentiation is essential for computing physics loss terms (derivatives of network outputs)
- Supports both CPU and GPU training for scalability
- Enables easy integration with existing deep learning workflows

### Architecture Selection Rationale

The framework uses a two-model architecture with unified data generation:

1. **Trajectory Prediction Model** (Forward Problem):
   - Network maps inputs (time, initial conditions, parameters) to outputs (δ, ω)
   - The smooth tanh activation ensures that computed derivatives accurately represent the physics (swing equation)
   - This is the foundational forward problem that enables CCT estimation via binary search
   - Input: (t, δ₀, ω₀, H, D, P_m, Xprefault, Xfault, Xpostfault, tf, tc) - 11 dimensions
   - Output: (δ(t), ω(t)) trajectories

2. **Parameter Estimation Model** (Inverse Problem):
   - Network processes trajectory sequences (δ(t), ω(t)) to infer system parameters (H, D)
   - Most complex architecture: sequence models with optional LSTM layers
   - Requires trajectory data as input (inverse problem)
   - Captures temporal dependencies in trajectory data to extract parameter information
   - Input: Observed (δ(t), ω(t)) trajectories
   - Output: Estimated (H, D) parameters

3. **CCT Estimation** (Via Binary Search):
   - **No separate model needed** - CCT is estimated using binary search with the trajectory model
   - Uses `utils.cct_binary_search.estimate_cct_binary_search()` function
   - Performs binary search to find maximum stable fault clearing time
   - More accurate and doesn't require separate CCT training data
   - See `examples/cct_pinn_example.py` for usage

**Note**: The separate CCT estimation model (`CCTEstimationPINN`) is deprecated but kept for reference.

## Architecture Details

### Network Architecture
- **Activation Functions**: tanh (default), ReLU, or custom
- **Residual Connections**: Optional skip connections for deeper networks
- **Dropout**: Optional regularization
- **Layer Normalization**: For stable training

### Physics Constraints
All models enforce:
1. **Swing Equation**: `M·d²δ/dt² + D·dδ/dt = Pm - Pe(δ)`
2. **Initial Conditions**: `δ(0) = δ₀`, `ω(0) = ω₀`
3. **Boundary Conditions**: Continuity at fault clearing times

### Collocation Points
Physics loss is evaluated at collocation points (typically randomly sampled in time domain) to enforce physics constraints throughout the solution domain.

## Training Considerations

1. **Loss Weighting**: Balance between data fit and physics constraints
   - `lambda_data`: Weight for data loss (typically 1.0)
   - `lambda_physics`: Weight for physics loss (typically 0.1-1.0)
   - `lambda_ic`: Weight for initial conditions (typically 10.0)

2. **Collocation Points**: Number and distribution affect physics enforcement
   - More points = stronger physics constraints but higher computational cost
   - Adaptive sampling near critical regions can improve accuracy

3. **Network Depth**: Deeper networks can capture more complex dynamics but require more data

## Theoretical Background

### Swing Equation

The framework enforces the classical swing equation:

```
M·d²δ/dt² + D·dδ/dt = Pm - Pe(t)
```

where:
- **M**: Inertia constant (seconds)
- **D**: Damping coefficient (pu)
- **δ**: Rotor angle (radians)
- **ω**: Rotor speed (pu)
- **Pm**: Mechanical power (pu)
- **Pe(t)**: Electrical power (pu)

### Electrical Power with State-Aware Reactance

```
Pe(t) = (V₁·V₂/X(t))·sin(δ(t))
```

where X(t) switches based on fault state:
- **Pre-fault**: X(t) = Xprefault
- **During fault**: X(t) = Xfault
- **Post-fault**: X(t) = Xpostfault

## Notes

- All models use PyTorch and support GPU acceleration
- Models are designed to work with the training scripts in `training/`
- Evaluation metrics are available in `utils.metrics`
- Visualization tools are available in `utils.visualization`
- For CCT estimation, use `utils.cct_binary_search` instead of the deprecated CCTEstimationPINN

---

**[⬆ Back to Top](#pinn-module)**
