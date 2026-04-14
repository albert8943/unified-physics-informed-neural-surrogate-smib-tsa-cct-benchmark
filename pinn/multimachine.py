"""
Multimachine Extension for PINN.

Extends PINN framework from SMIB systems to multimachine networks.
Handles relative angles and COI (Center of Inertia) reference frame.

CRITICAL NOTES:
- COI computation requires M = 2*H (not H) for 60 Hz systems
- Physics loss should use COI-relative angles (δ'_i = δ_i - δ_COI)
- Current PhysicsInformedLoss may need extension for multimachine COI-referenced swing equation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .core import PINN


class MultimachinePINN(nn.Module):
    """
    PINN for multimachine transient stability analysis.

    Extends swing equations to N-machine system.
    Handles relative angles and COI (Center of Inertia) reference frame.
    """

    def __init__(
        self,
        num_machines: int,
        input_dim_per_machine: int = 10,
        hidden_dims: List[int] = [128, 128, 128, 64],
        activation: str = "tanh",
        dropout: float = 0.0,
        use_coi: bool = True,
        use_pe_as_input: bool = True,
    ):
        """
        Initialize multimachine PINN.

        Parameters:
        -----------
        num_machines : int
            Number of generators in the system
        input_dim_per_machine : int
            Input dimension per machine
        hidden_dims : list
            Hidden layer dimensions
        activation : str
            Activation function
        dropout : float
            Dropout rate
        use_coi : bool
            Whether to use Center of Inertia reference frame
        """
        super(MultimachinePINN, self).__init__()

        self.num_machines = num_machines
        self.use_pe_as_input = use_pe_as_input

        # Adjust input_dim_per_machine based on input method
        if use_pe_as_input:
            # Pe(t) as input: 7 dims [t, δ₀, ω₀, H, D, Pm, Pe(t)] or 9 dims [t, δ₀, ω₀, H, D, Pm_i, Pe(t), tf, tc]
            if input_dim_per_machine not in (7, 9):
                input_dim_per_machine = 9
        else:
            # Reactance-based: [t, δ₀, ω₀, H, D, Pm, Xprefault, Xfault, Xpostfault, tf, tc] = 11 dims
            if input_dim_per_machine != 11:
                input_dim_per_machine = 11

        self.input_dim_per_machine = input_dim_per_machine
        self.use_coi = use_coi

        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unknown activation: {activation}")

        # Shared feature extraction for each machine
        self.machine_encoders = nn.ModuleList(
            [
                self._create_encoder(input_dim_per_machine, hidden_dims, dropout)
                for _ in range(num_machines)
            ]
        )

        # COI computation layer (if using COI)
        if use_coi:
            self.coi_layer = nn.Linear(hidden_dims[-1] * num_machines, hidden_dims[-1])

        # Output layers (delta and omega for each machine)
        output_dim = hidden_dims[-1] if use_coi else hidden_dims[-1] * num_machines
        self.delta_output = nn.Linear(output_dim, num_machines)
        self.omega_output = nn.Linear(output_dim, num_machines)

        # Initialize weights
        self._initialize_weights()

    def _create_encoder(self, input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
        """Create encoder network for a single machine."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(self.activation)
            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_coi(
        self, deltas: torch.Tensor, omegas: torch.Tensor, M_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Center of Inertia (COI) reference frame.

        Parameters:
        -----------
        deltas : torch.Tensor
            Rotor angles [batch_size, num_machines]
        omegas : torch.Tensor
            Rotor speeds [batch_size, num_machines]
        M_values : torch.Tensor
            Inertia constants [batch_size, num_machines]
            **CRITICAL**: Must be M = 2*H (not H) for 60 Hz systems
            Example: If H = 3.0 s, then M = 6.0 s

        Returns:
        --------
        tuple : (delta_coi, omega_coi)
            delta_coi : COI angle [batch_size, 1]
            omega_coi : COI speed [batch_size, 1]

        Formula:
        --------
        δ_COI = (Σᵢ M_i · δ_i) / (Σᵢ M_i)
        ω_COI = (Σᵢ M_i · ω_i) / (Σᵢ M_i)

        Where M_i = 2·H_i for 60 Hz systems.
        """
        # Total inertia.
        M_total = torch.sum(M_values, dim=1, keepdim=True)  # [batch_size, 1]

        # COI angle: weighted average by inertia
        # Formula: δ_COI = (Σᵢ M_i · δ_i) / (Σᵢ M_i)
        delta_coi = torch.sum(M_values * deltas, dim=1, keepdim=True) / (M_total + 1e-8)

        # COI speed: weighted average by inertia
        # Formula: ω_COI = (Σᵢ M_i · ω_i) / (Σᵢ M_i)
        omega_coi = torch.sum(M_values * omegas, dim=1, keepdim=True) / (M_total + 1e-8)

        return delta_coi, omega_coi

    def compute_relative_angles(
        self, deltas: torch.Tensor, delta_coi: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relative angles with respect to COI.

        Parameters:
        -----------
        deltas : torch.Tensor
            Rotor angles [batch_size, num_machines]
        delta_coi : torch.Tensor
            COI angle [batch_size, 1]

        Returns:
        --------
        torch.Tensor : Relative angles [batch_size, num_machines]
        """
        return deltas - delta_coi

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch_size, num_machines, input_dim_per_machine]

        Returns:
        --------
        tuple : (delta_pred, omega_pred) [batch_size, num_machines]
        """
        batch_size = x.shape[0]

        # Encode each machine
        machine_features = []
        for i, encoder in enumerate(self.machine_encoders):
            machine_input = x[:, i, :]  # [batch_size, input_dim_per_machine]
            features = encoder(machine_input)  # [batch_size, hidden_dim]
            machine_features.append(features)

        # Concatenate or use COI
        if self.use_coi:
            # Concatenate all machine features
            combined_features = torch.cat(
                machine_features, dim=1
            )  # [batch_size, num_machines * hidden_dim]
            # Apply COI layer
            coi_features = self.coi_layer(combined_features)  # [batch_size, hidden_dim]
            output_features = coi_features
        else:
            # Concatenate all features
            output_features = torch.cat(machine_features, dim=1)

        # Predict delta and omega for each machine
        delta_pred = self.delta_output(output_features)  # [batch_size, num_machines]
        omega_pred = self.omega_output(output_features)  # [batch_size, num_machines]

        return delta_pred, omega_pred

    def predict_trajectories(
        self,
        t: torch.Tensor,
        initial_conditions: Dict[str, torch.Tensor],
        system_parameters: Dict[str, torch.Tensor],
        Pe_trajectories: Optional[Dict[int, torch.Tensor]] = None,  # NEW: Pe_i(t) for each machine
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectories for all machines.

        Parameters:
        -----------
        t : torch.Tensor
            Time points [batch_size]
        initial_conditions : dict
            Dictionary with 'delta0' and 'omega0' [batch_size, num_machines]
        system_parameters : dict
            Dictionary with system parameters (H, D, X, etc.)
            For Pe input: {'H': [batch, num_machines], 'D': [batch, num_machines], 'Pm': [batch, num_machines]}
            For reactance input: also includes 'Xprefault', 'Xfault', 'Xpostfault', 'tf', 'tc'
        Pe_trajectories : dict, optional
            Dictionary mapping machine index to Pe_i(t) [batch_size, time_steps]
            Required when use_pe_as_input=True
        device : str
            Device ('cpu' or 'cuda')

        Returns:
        --------
        tuple : (delta_pred, omega_pred) [batch_size, num_machines]
        """
        self.eval()

        batch_size = t.shape[0] if len(t.shape) > 0 else 1

        # Construct input tensor
        # Shape: [batch_size, num_machines, input_dim_per_machine]
        x = torch.zeros(batch_size, self.num_machines, self.input_dim_per_machine, device=device)

        # Extract parameters
        delta0 = initial_conditions.get(
            "delta0", torch.zeros(batch_size, self.num_machines, device=device)
        )
        omega0 = initial_conditions.get(
            "omega0", torch.ones(batch_size, self.num_machines, device=device)
        )
        H = system_parameters.get(
            "H", torch.ones(batch_size, self.num_machines, device=device) * 3.0
        )
        D = system_parameters.get(
            "D", torch.ones(batch_size, self.num_machines, device=device) * 1.0
        )
        Pm = system_parameters.get(
            "Pm", torch.ones(batch_size, self.num_machines, device=device) * 0.7
        )

        # Fill input tensor for each machine
        tf = system_parameters.get("tf", torch.ones(batch_size, device=device) * 1.0)
        tc = system_parameters.get("tc", torch.ones(batch_size, device=device) * 1.2)
        for i in range(self.num_machines):
            if self.use_pe_as_input:
                # Pe(t) as input: 7 dims [t, δ₀, ω₀, H, D, Pm, Pe(t)] or 9 dims [+ tf, tc]
                if Pe_trajectories is not None and i in Pe_trajectories:
                    Pe_i = Pe_trajectories[i]  # [batch_size, time_steps] or [batch_size]
                    if len(Pe_i.shape) == 1:
                        Pe_i = Pe_i.unsqueeze(1)  # [batch_size, 1]
                    # Use Pe at current time point (simplified - in practice would interpolate)
                    Pe_at_t = (
                        Pe_i[:, 0] if Pe_i.shape[1] > 0 else torch.zeros(batch_size, device=device)
                    )
                else:
                    Pe_at_t = torch.zeros(batch_size, device=device)

                base_inputs = [
                    t.flatten()[:batch_size] if len(t.shape) > 0 else t.expand(batch_size),
                    delta0[:, i] if delta0.shape[1] > i else delta0[:, 0],
                    omega0[:, i] if omega0.shape[1] > i else omega0[:, 0],
                    H[:, i] if H.shape[1] > i else H[:, 0],
                    D[:, i] if D.shape[1] > i else D[:, 0],
                    Pm[:, i] if Pm.shape[1] > i else Pm[:, 0],
                    Pe_at_t,
                ]
                if self.input_dim_per_machine >= 9:
                    base_inputs.extend([tf, tc])
                x[:, i, :] = torch.stack(base_inputs, dim=1)
            else:
                # Reactance-based input: [t, δ₀, ω₀, H, D, Pm, Xprefault, Xfault, Xpostfault, tf, tc]
                Xprefault = system_parameters.get(
                    "Xprefault", torch.ones(batch_size, device=device) * 0.5
                )
                Xfault = system_parameters.get(
                    "Xfault", torch.ones(batch_size, device=device) * 0.0001
                )
                Xpostfault = system_parameters.get(
                    "Xpostfault", torch.ones(batch_size, device=device) * 0.5
                )
                tf = system_parameters.get("tf", torch.ones(batch_size, device=device) * 1.0)
                tc = system_parameters.get("tc", torch.ones(batch_size, device=device) * 1.2)

                x[:, i, :] = torch.stack(
                    [
                        t.flatten()[:batch_size] if len(t.shape) > 0 else t.expand(batch_size),
                        delta0[:, i] if delta0.shape[1] > i else delta0[:, 0],
                        omega0[:, i] if omega0.shape[1] > i else omega0[:, 0],
                        H[:, i] if H.shape[1] > i else H[:, 0],
                        D[:, i] if D.shape[1] > i else D[:, 0],
                        Pm[:, i] if Pm.shape[1] > i else Pm[:, 0],
                        Xprefault,
                        Xfault,
                        Xpostfault,
                        tf,
                        tc,
                    ],
                    dim=1,
                )

        with torch.no_grad():
            delta_pred, omega_pred = self.forward(x)

        return delta_pred, omega_pred
