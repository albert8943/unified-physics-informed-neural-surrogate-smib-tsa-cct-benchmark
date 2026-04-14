"""
Normalization Utilities for PINN Training.

Provides centralized normalization/denormalization functions and
a PhysicsNormalizer class that handles denormalization for physics loss
while preserving computation graphs.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def normalize_array(arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Normalize array using fitted scaler.

    Parameters:
    -----------
    arr : np.ndarray
        Array to normalize
    scaler : StandardScaler
        Fitted sklearn StandardScaler

    Returns:
    --------
    np.ndarray : Normalized array
    """
    if len(arr.shape) == 1:
        return scaler.transform(arr.reshape(-1, 1)).flatten()
    else:
        return scaler.transform(arr)


def denormalize_array(arr_norm: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Denormalize array back to physical units.

    Parameters:
    -----------
    arr_norm : np.ndarray
        Normalized array
    scaler : StandardScaler
        Fitted sklearn StandardScaler

    Returns:
    --------
    np.ndarray : Denormalized array in physical units
    """
    if len(arr_norm.shape) == 1:
        return scaler.inverse_transform(arr_norm.reshape(-1, 1)).flatten()
    else:
        return scaler.inverse_transform(arr_norm)


def normalize_value(val: float, scaler: StandardScaler) -> float:
    """
    Normalize single value.

    Parameters:
    -----------
    val : float
        Value to normalize
    scaler : StandardScaler
        Fitted sklearn StandardScaler

    Returns:
    --------
    float : Normalized value
    """
    return scaler.transform([[val]])[0, 0]


def denormalize_value(val_norm: float, scaler: StandardScaler) -> float:
    """
    Denormalize single value back to physical units.

    Parameters:
    -----------
    val_norm : float
        Normalized value
    scaler : StandardScaler
        Fitted sklearn StandardScaler

    Returns:
    --------
    float : Denormalized value in physical units
    """
    return scaler.inverse_transform([[val_norm]])[0, 0]


def normalize_tensor(
    tensor: torch.Tensor, scaler: StandardScaler, device: str = "cpu"
) -> torch.Tensor:
    """
    Normalize tensor using fitted scaler.

    Parameters:
    -----------
    tensor : torch.Tensor
        Tensor to normalize
    scaler : StandardScaler
        Fitted sklearn StandardScaler
    device : str
        Device for output tensor

    Returns:
    --------
    torch.Tensor : Normalized tensor
    """
    # Convert to numpy, normalize, convert back
    arr = tensor.detach().cpu().numpy()
    arr_norm = normalize_array(arr, scaler)
    return torch.tensor(arr_norm, dtype=torch.float32, device=device)


def denormalize_tensor(
    tensor_norm: torch.Tensor, scaler: StandardScaler, device: str = "cpu"
) -> torch.Tensor:
    """
    Denormalize tensor back to physical units while preserving gradients.

    CRITICAL: This function preserves the computation graph for backpropagation.
    Uses differentiable PyTorch operations instead of numpy conversion.

    Parameters:
    -----------
    tensor_norm : torch.Tensor
        Normalized tensor (must have requires_grad=True for gradients)
    scaler : StandardScaler
        Fitted sklearn StandardScaler
    device : str
        Device for output tensor

    Returns:
    --------
    torch.Tensor : Denormalized tensor in physical units (preserves gradients)
    """
    # Extract scaler statistics as tensors
    mean = torch.tensor(scaler.mean_[0], dtype=torch.float32, device=device)
    std = torch.tensor(scaler.scale_[0], dtype=torch.float32, device=device)

    # Denormalize: x_phys = x_norm * std + mean
    # This preserves gradients since it's a differentiable operation
    return tensor_norm * std + mean


class PhysicsNormalizer:
    """
    Handles normalization/denormalization for physics loss computation.

    Critical: Denormalizes values for physics equation while preserving
    computation graph for backpropagation.

    The physics equation needs physical units (radians, seconds, pu), but
    the model outputs normalized values. This class handles the conversion
    while keeping gradients intact.
    """

    def __init__(self, scalers: Dict[str, StandardScaler], device: str = "cpu"):
        """
        Initialize PhysicsNormalizer.

        Parameters:
        -----------
        scalers : dict
            Dictionary of fitted sklearn StandardScalers
            Must include: 'delta', 'omega', 'time'
        device : str
            Device for tensor operations
        """
        self.scalers = scalers
        self.device = device

        # Extract scaler statistics as tensors (for gradient-preserving operations)
        self.delta_mean = torch.tensor(
            scalers["delta"].mean_[0], dtype=torch.float32, device=device
        )
        self.delta_std = torch.tensor(
            scalers["delta"].scale_[0], dtype=torch.float32, device=device
        )
        self.omega_mean = torch.tensor(
            scalers["omega"].mean_[0], dtype=torch.float32, device=device
        )
        self.omega_std = torch.tensor(
            scalers["omega"].scale_[0], dtype=torch.float32, device=device
        )
        self.time_mean = torch.tensor(scalers["time"].mean_[0], dtype=torch.float32, device=device)
        self.time_std = torch.tensor(scalers["time"].scale_[0], dtype=torch.float32, device=device)

    def denormalize_for_physics(
        self, delta_norm: torch.Tensor, omega_norm: torch.Tensor, t_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Denormalize values for physics equation while preserving gradients.

        This is the key method that allows physics loss computation with
        physical units while maintaining backpropagation through the
        denormalization operations.

        Parameters:
        -----------
        delta_norm : torch.Tensor
            Normalized rotor angle
        omega_norm : torch.Tensor
            Normalized rotor speed
        t_norm : torch.Tensor
            Normalized time

        Returns:
        --------
        tuple : (delta_phys, omega_phys, t_phys, time_scale, time_scale_sq)
            delta_phys : Denormalized rotor angle (rad)
            omega_phys : Denormalized rotor speed (pu)
            t_phys : Denormalized time (s)
            time_scale : Derivative scaling factor d(delta_phys)/d(t_phys)
            time_scale_sq : 2nd derivative scaling factor d²(delta_phys)/d(t_phys)²
        """
        # Denormalize (keeps gradients because it's differentiable operations)
        # Formula: x = x_norm * std + mean
        delta_phys = delta_norm * self.delta_std + self.delta_mean
        omega_phys = omega_norm * self.omega_std + self.omega_mean
        t_phys = t_norm * self.time_std + self.time_mean

        # Derivative scaling factors
        # When we compute d(delta_norm)/d(t_norm), we need to scale to physical units
        # d(delta_phys)/d(t_phys) = d(delta_norm)/d(t_norm) * (std_delta / std_time)
        time_scale = self.delta_std / self.time_std

        # Second derivative scaling
        # d²(delta_phys)/d(t_phys)² = d²(delta_norm)/d(t_norm)² * (std_delta / std_time²)
        time_scale_sq = self.delta_std / (self.time_std**2)

        return delta_phys, omega_phys, t_phys, time_scale, time_scale_sq

    def denormalize_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize rotor angle only."""
        return delta_norm * self.delta_std + self.delta_mean

    def denormalize_omega(self, omega_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize rotor speed only."""
        return omega_norm * self.omega_std + self.omega_mean

    def denormalize_time(self, t_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize time only."""
        return t_norm * self.time_std + self.time_mean

    def get_time_scale(self) -> torch.Tensor:
        """Get derivative scaling factor."""
        return self.delta_std / self.time_std

    def get_time_scale_sq(self) -> torch.Tensor:
        """Get second derivative scaling factor."""
        return self.delta_std / (self.time_std**2)


def set_model_standardization_to_identity(
    model: nn.Module, input_dim: int, output_dim: int, device: str = "cpu"
):
    """
    Set model's internal standardization to identity (mean=0, std=1).

    This is critical when using sklearn scalers for normalization - we don't want
    the model to apply its own normalization on top of sklearn's normalization.

    Parameters:
    -----------
    model : nn.Module
        PINN model with input_standardization and output_scaling attributes
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    device : str
        Device for tensors
    """
    if hasattr(model, "input_standardization") and model.input_standardization is not None:
        model.input_standardization.mean.data = torch.zeros(
            input_dim, dtype=torch.float32, device=device
        )
        model.input_standardization.standard_deviation.data = torch.ones(
            input_dim, dtype=torch.float32, device=device
        )
        print("✓ Set input standardization to identity (mean=0, std=1)")

    if hasattr(model, "output_scaling") and model.output_scaling is not None:
        model.output_scaling.mean.data = torch.zeros(output_dim, dtype=torch.float32, device=device)
        model.output_scaling.standard_deviation.data = torch.ones(
            output_dim, dtype=torch.float32, device=device
        )
        print("✓ Set output scaling to identity (mean=0, std=1)")


def verify_normalization_consistency(
    model: nn.Module,
    scalers: Dict[str, StandardScaler],
    test_input: torch.Tensor,
    verbose: bool = True,
) -> bool:
    """
    Verify that normalization is consistent between training and inference.

    Parameters:
    -----------
    model : nn.Module
        PINN model
    scalers : dict
        Dictionary of fitted scalers
    test_input : torch.Tensor
        Test input tensor (normalized)
    verbose : bool
        Print diagnostic information

    Returns:
    --------
    bool : True if consistent, False otherwise
    """
    if verbose:
        print("\nNormalization Consistency Check:")

    # Check model standardization
    if hasattr(model, "input_standardization"):
        mean = model.input_standardization.mean.data
        std = model.input_standardization.standard_deviation.data

        is_identity = torch.allclose(mean, torch.zeros_like(mean)) and torch.allclose(
            std, torch.ones_like(std)
        )

        if verbose:
            if is_identity:
                print("  ✓ Model input standardization is identity")
            else:
                print("  ✗ Model input standardization is NOT identity")
                print(f"    Mean: {mean.cpu().numpy()}")
                print(f"    Std:  {std.cpu().numpy()}")

        return is_identity

    return True


if __name__ == "__main__":
    print("Normalization Utilities")
    print("Import and use PhysicsNormalizer, normalize_array(), denormalize_array(), etc.")
