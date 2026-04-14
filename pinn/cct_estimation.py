"""
CCT Estimation PINN.

Determines Critical Clearing Time (CCT) and transient stability margins.

.. deprecated:: 2.0
    This module is deprecated. Use binary search with trajectory model instead.
    See `utils.cct_binary_search.estimate_cct_binary_search()` for the new approach.
"""

import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .core import PINN

# Issue deprecation warning at module level
warnings.warn(
    "CCTEstimationPINN is deprecated. Use binary search with trajectory model instead. "
    "See utils.cct_binary_search.estimate_cct_binary_search() for the new approach.",
    DeprecationWarning,
    stacklevel=2,
)


class CCTEstimationPINN(nn.Module):
    """
    PINN for Critical Clearing Time (CCT) estimation.

    .. deprecated:: 2.0
        This class is deprecated. Use binary search with trajectory model instead.
        See `utils.cct_binary_search.estimate_cct_binary_search()` for the new approach.

    Input: Initial conditions, fault location, system parameters
    Output: Critical clearing time (CCT)

    Note:
        This class is kept for reference and backward compatibility.
        For new code, use the binary search approach with TrajectoryPredictionPINN.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list = [64, 64, 64],
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        """
        Initialize CCT estimation PINN.

        .. deprecated:: 2.0
            Use binary search with trajectory model instead.

        Parameters:
        -----------
        input_dim : int
            Input dimension (default: 8 for [δ₀, ω₀, H, D, Xprefault, "
            "Xfault, Xpostfault, fault_bus])
        hidden_dims : list
            Hidden layer dimensions
        activation : str
            Activation function
        dropout : float
            Dropout rate
        """
        warnings.warn(
            "CCTEstimationPINN is deprecated. Use binary search with trajectory model instead. "
            "See utils.cct_binary_search.estimate_cct_binary_search() for the new approach.",
            DeprecationWarning,
            stacklevel=2,
        )
        super(CCTEstimationPINN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unknown activation: {activation}")

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # Output layer (single value: CCT)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.ReLU())  # Ensure positive CCT

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def predict_cct(
        self,
        delta0: float,
        omega0: float,
        H: float,
        D: float,
        Xprefault: float,
        Xfault: float,
        Xpostfault: float,
        fault_bus: int = 3,
        device: str = "cpu",
    ) -> float:
        """
        Predict CCT for given parameters.

        .. deprecated:: 2.0
            Use binary search with trajectory model instead.
            See `utils.cct_binary_search.estimate_cct_binary_search()`.

        Parameters:
        -----------
        delta0 : float
            Initial rotor angle (rad)
        omega0 : float
            Initial rotor speed (pu)
        H : float
            Inertia constant (seconds)
        D : float
            Damping coefficient (pu)
        Xprefault : float
            Pre-fault reactance (pu)
        Xfault : float
            Fault reactance (pu)
        Xpostfault : float
            Post-fault reactance (pu)
        fault_bus : int
            Fault bus index
        device : str
            Device ('cpu' or 'cuda')

        Returns:
        --------
        float : Predicted CCT (seconds)
        """
        warnings.warn(
            "CCTEstimationPINN.predict_cct() is deprecated. "
            "Use utils.cct_binary_search.estimate_cct_binary_search() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.eval()

        # Prepare input
        x = torch.tensor(
            [[delta0, omega0, H, D, Xprefault, Xfault, Xpostfault, float(fault_bus)]],
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            cct_pred = self.forward(x)

        return cct_pred.cpu().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch_size, input_dim]

        Returns:
        --------
        torch.Tensor : CCT predictions [batch_size, 1]
        """
        return self.network(x)
