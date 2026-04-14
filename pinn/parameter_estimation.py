"""
Parameter Estimation PINN.

Estimates machine inertia (H/M) and damping (D) from time-domain responses.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .core import PINN


class ParameterEstimationPINN(nn.Module):
    """
    PINN for parameter estimation (H, D).

    Input: Observed δ(t), ω(t) trajectories
    Output: Estimated H (or M) and D
    """

    def __init__(
        self,
        input_dim: int = 2,  # delta and omega at each time point
        sequence_length: int = 100,  # Number of time points
        hidden_dims: list = [128, 128, 64],
        activation: str = "tanh",
        dropout: float = 0.0,
        use_lstm: bool = True,
    ):
        """
        Initialize parameter estimation PINN.

        Parameters:
        -----------
        input_dim : int
            Input dimension per time step (default: 2 for [δ, ω])
        sequence_length : int
            Number of time points in trajectory
        hidden_dims : list
            Hidden layer dimensions
        activation : str
            Activation function
        dropout : float
            Dropout rate
        use_lstm : bool
            Whether to use LSTM for sequence processing
        """
        super(ParameterEstimationPINN, self).__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dims = hidden_dims
        self.use_lstm = use_lstm

        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unknown activation: {activation}")

        if use_lstm:
            # LSTM for sequence processing
            lstm_hidden = hidden_dims[0] if len(hidden_dims) > 0 else 64
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=lstm_hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0,
            )
            prev_dim = lstm_hidden
        else:
            # Flatten sequence
            prev_dim = input_dim * sequence_length
            self.lstm = None

        # Build network
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0 and use_lstm:
                continue  # Already handled by LSTM
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # Output layer (2 values: H and D)
        layers.append(nn.Linear(prev_dim, 2))
        layers.append(nn.ReLU())  # Ensure positive values

        self.network = nn.Sequential(*layers) if len(layers) > 0 else None

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch_size, sequence_length, input_dim] or "
            "[batch_size, sequence_length * input_dim]

        Returns:
        --------
        torch.Tensor : Parameter predictions [batch_size, 2] (H, D)
        """
        if self.use_lstm:
            # x shape: [batch_size, sequence_length, input_dim]
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use last hidden state
            x = lstm_out[:, -1, :]  # [batch_size, lstm_hidden]
        else:
            # Flatten if needed
            if len(x.shape) > 2:
                x = x.view(x.shape[0], -1)

        if self.network is not None:
            output = self.network(x)
        else:
            # Direct output from LSTM
            output = nn.Linear(x.shape[1], 2).to(x.device)(x)

        return output

    def predict_parameters(
        self, delta_trajectory: np.ndarray, omega_trajectory: np.ndarray, device: str = "cpu"
    ) -> Tuple[float, float]:
        """
        Predict H and D from observed trajectories.

        Parameters:
        -----------
        delta_trajectory : np.ndarray
            Observed rotor angle trajectory
        omega_trajectory : np.ndarray
            Observed rotor speed trajectory
        device : str
            Device ('cpu' or 'cuda')

        Returns:
        --------
        tuple : (H_pred, D_pred)
        """
        self.eval()

        # Ensure trajectories have same length
        min_len = min(len(delta_trajectory), len(omega_trajectory))
        delta_trajectory = delta_trajectory[:min_len]
        omega_trajectory = omega_trajectory[:min_len]

        # Pad or truncate to sequence_length
        if len(delta_trajectory) < self.sequence_length:
            # Pad with last value
            delta_padded = np.pad(
                delta_trajectory,
                (0, self.sequence_length - len(delta_trajectory)),
                mode="constant",
                constant_values=delta_trajectory[-1],
            )
            omega_padded = np.pad(
                omega_trajectory,
                (0, self.sequence_length - len(omega_trajectory)),
                mode="constant",
                constant_values=omega_trajectory[-1],
            )
        else:
            # Truncate
            delta_padded = delta_trajectory[: self.sequence_length]
            omega_padded = omega_trajectory[: self.sequence_length]

        # Prepare input
        x = np.stack([delta_padded, omega_padded], axis=1)  # [sequence_length, 2]
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(
            0
        )  # [1, sequence_length, 2]

        with torch.no_grad():
            params_pred = self.forward(x)

        H_pred = params_pred[0, 0].cpu().item()
        D_pred = params_pred[0, 1].cpu().item()

        return H_pred, D_pred
