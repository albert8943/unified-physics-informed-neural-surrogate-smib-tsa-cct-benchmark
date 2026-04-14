"""
Trajectory Prediction PINN.

Predicts rotor angle (δ) and speed (ω) trajectories under transient faults.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .core import PINN, PhysicsInformedLoss


class TrajectoryPredictionPINN(PINN):
    """
    PINN for trajectory prediction (δ, ω).

    Input: Initial conditions (δ₀, ω₀), fault parameters, system parameters (H, D, P_m)
    Output: Predicted δ(t), ω(t) trajectories
    """

    def __init__(
        self,
        input_dim: int = 11,  # [t, δ₀, ω₀, H, D, Pm, Xprefault, Xfault, Xpostfault, tf, tc]
        hidden_dims: list = [64, 64, 64, 64],
        activation: str = "tanh",
        use_residual: bool = False,
        dropout: float = 0.0,
        use_standardization: bool = True,
    ):
        """
        Initialize trajectory prediction PINN.

        Parameters:
        -----------
        input_dim : int
            Input dimension (default: 11 for [t, δ₀, ω₀, H, D, P_m, "
            "Xprefault, Xfault, Xpostfault, tf, tc])
        hidden_dims : list
            Hidden layer dimensions
        activation : str
            Activation function
        use_residual : bool
            Whether to use residual connections
        dropout : float
            Dropout rate
        use_standardization : bool
            Whether to use input/output standardization (default: True, recommended)
        """
        super(TrajectoryPredictionPINN, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=2,  # [delta, omega]
            activation=activation,
            use_residual=use_residual,
            dropout=dropout,
            use_standardization=use_standardization,
        )

        self.loss_fn = PhysicsInformedLoss(
            lambda_data=1.0, lambda_physics=0.1, lambda_ic=10.0, lambda_boundary=1.0
        )

    def compute_loss(
        self,
        t_data: torch.Tensor,
        delta_pred: torch.Tensor,
        omega_pred: torch.Tensor,
        delta_obs: Optional[torch.Tensor] = None,
        omega_obs: Optional[torch.Tensor] = None,
        t_colloc: Optional[torch.Tensor] = None,
        delta_colloc: Optional[torch.Tensor] = None,
        omega_colloc: Optional[torch.Tensor] = None,
        t_ic: Optional[torch.Tensor] = None,
        delta0: Optional[torch.Tensor] = None,
        omega0: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        Pm: Optional[torch.Tensor] = None,
        Xprefault: Optional[torch.Tensor] = None,
        Xfault: Optional[torch.Tensor] = None,
        Xpostfault: Optional[torch.Tensor] = None,
        tf: Optional[torch.Tensor] = None,
        tc: Optional[torch.Tensor] = None,
        time_scale: Optional[torch.Tensor] = None,
        time_scale_sq: Optional[torch.Tensor] = None,
        delta_colloc_phys: Optional[torch.Tensor] = None,
        omega_colloc_phys: Optional[torch.Tensor] = None,
        t_colloc_phys: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for trajectory prediction.

        Parameters:
        -----------
        t_data : torch.Tensor
            Time points for data loss
        delta_pred : torch.Tensor
            Predicted rotor angles
        omega_pred : torch.Tensor
            Predicted rotor speeds
        delta_obs : torch.Tensor, optional
            Observed rotor angles
        omega_obs : torch.Tensor, optional
            Observed rotor speeds
        t_colloc : torch.Tensor, optional
            Collocation points (normalized for derivative computation)
        delta_colloc : torch.Tensor, optional
            Predicted delta at collocation points (normalized for derivative computation)
        omega_colloc : torch.Tensor, optional
            Predicted omega at collocation points
        t_ic : torch.Tensor, optional
            Initial time
        delta0 : torch.Tensor, optional
            Initial rotor angle
        omega0 : torch.Tensor, optional
            Initial rotor speed
        M : torch.Tensor, optional
            Inertia constant (denormalized for physics equation)
        D : torch.Tensor, optional
            Damping coefficient (denormalized for physics equation)
        Pm : torch.Tensor, optional
            Mechanical power (denormalized for physics equation)
        Xprefault : torch.Tensor, optional
            Pre-fault reactance (denormalized for physics equation)
        Xfault : torch.Tensor, optional
            Fault reactance (denormalized for physics equation)
        Xpostfault : torch.Tensor, optional
            Post-fault reactance (denormalized for physics equation)
        tf : torch.Tensor, optional
            Fault start time (denormalized for physics equation)
        tc : torch.Tensor, optional
            Fault clear time (denormalized for physics equation)
        time_scale : torch.Tensor, optional
            Scaling factor for first derivative (std_delta / std_time)
        time_scale_sq : torch.Tensor, optional
            Scaling factor for second derivative
        delta_colloc_phys : torch.Tensor, optional
            Denormalized delta at collocation points (for physics equation)
        omega_colloc_phys : torch.Tensor, optional
            Denormalized omega at collocation points (for physics equation)
        t_colloc_phys : torch.Tensor, optional
            Denormalized time at collocation points (for physics equation)

        Returns:
        --------
        dict : Loss components
        """
        return self.loss_fn(
            t_data=t_data,
            delta_pred=delta_pred,
            omega_pred=omega_pred,
            delta_obs=delta_obs,
            omega_obs=omega_obs,
            t_colloc=t_colloc,
            delta_colloc=delta_colloc,
            omega_colloc=omega_colloc,
            t_ic=t_ic,
            delta0=delta0,
            omega0=omega0,
            M=M,
            D=D,
            Pm=Pm,
            Xprefault=Xprefault,
            Xfault=Xfault,
            Xpostfault=Xpostfault,
            tf=tf,
            tc=tc,
            time_scale=time_scale,
            time_scale_sq=time_scale_sq,
            delta_colloc_phys=delta_colloc_phys,
            omega_colloc_phys=omega_colloc_phys,
            t_colloc_phys=t_colloc_phys,
        )

    def predict(
        self,
        t: np.ndarray,
        delta0: float,
        omega0: float,
        H: float,
        D: float,
        Pm: Optional[float] = None,
        Xprefault: float = 0.5,
        Xfault: float = 0.0001,
        Xpostfault: float = 0.5,
        tf: float = 1.0,
        tc: float = 1.2,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory for given parameters.

        Parameters:
        -----------
        t : np.ndarray
            Time points
        delta0 : float
            Initial rotor angle (rad)
        omega0 : float
            Initial rotor speed (pu)
        H : float
            Inertia constant (seconds)
        D : float
            Damping coefficient (pu)
        Pm : float
            Mechanical power (pu). **REQUIRED** - Must match training data range (typically 0.4-0.9)
        Xprefault : float
            Pre-fault reactance (pu)
        Xfault : float
            Fault reactance (pu)
        Xpostfault : float
            Post-fault reactance (pu)
        tf : float
            Fault start time (s)
        tc : float
            Fault clear time (s)
        device : str
            Device ('cpu' or 'cuda')

        Returns:
        --------
        tuple : (delta_pred, omega_pred) as numpy arrays
        """
        self.eval()

        # Pm is required - model was trained with varying Pm
        if Pm is None:
            raise ValueError(
                "Pm (mechanical power) is REQUIRED. "
                "The model was trained with varying Pm (0.4-0.9 pu). "
                "You must provide the correct Pm value for accurate predictions."
            )

        # Convert to tensors
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        delta0_tensor = torch.tensor(delta0, dtype=torch.float32, device=device)
        omega0_tensor = torch.tensor(omega0, dtype=torch.float32, device=device)
        H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
        D_tensor = torch.tensor(D, dtype=torch.float32, device=device)
        Pm_tensor = torch.tensor(Pm, dtype=torch.float32, device=device)
        Xprefault_tensor = torch.tensor(Xprefault, dtype=torch.float32, device=device)
        Xfault_tensor = torch.tensor(Xfault, dtype=torch.float32, device=device)
        Xpostfault_tensor = torch.tensor(Xpostfault, dtype=torch.float32, device=device)
        tf_tensor = torch.tensor(tf, dtype=torch.float32, device=device)
        tc_tensor = torch.tensor(tc, dtype=torch.float32, device=device)

        with torch.no_grad():
            delta_pred, omega_pred = self.predict_trajectory(
                t=t_tensor,
                delta0=delta0_tensor,
                omega0=omega0_tensor,
                H=H_tensor,
                D=D_tensor,
                Pm=Pm_tensor,
                Xprefault=Xprefault_tensor,
                Xfault=Xfault_tensor,
                Xpostfault=Xpostfault_tensor,
                tf=tf_tensor,
                tc=tc_tensor,
            )

        # Convert to numpy
        delta_pred_np = delta_pred.cpu().numpy()
        omega_pred_np = omega_pred.cpu().numpy()

        return delta_pred_np, omega_pred_np


class TrajectoryPredictionPINN_PeInput(PINN):
    """
    PINN with Pe(t) as input instead of reactances.

    Input (set ``input_dim``): either 7 dims
    [t, δ₀, ω₀, H, D, alpha, Pe(t)] or 9 dims
    [t, δ₀, ω₀, H, D, alpha, Pe(t), tf, tc] (tf, tc for extra steady-state context).
    Output: [δ(t), ω(t)]  # 2 dims

    This variant uses Pe(t) directly from ANDES as input, eliminating
    the need to compute electrical power from reactances.

    Note: Uses alpha (uniform load multiplier) instead of Pm (mechanical power) as input,
    which is more appropriate for load variation mode and works for both SMIB and multimachine.
    Pm is still used separately for physics loss computation.
    """

    def __init__(
        self,
        input_dim: int = 9,  # [t, δ₀, ω₀, H, D, alpha, Pe(t), tf, tc]
        hidden_dims: list = [64, 64, 64, 64],
        activation: str = "tanh",
        use_residual: bool = False,
        dropout: float = 0.0,
        use_standardization: bool = True,
    ):
        """
        Initialize trajectory prediction PINN with Pe(t) as input.

        Parameters:
        -----------
        input_dim : int
            Input dimension (default: 9 for [t, δ₀, ω₀, H, D, alpha, Pe(t), tf, tc])
        hidden_dims : list
            Hidden layer dimensions
        activation : str
            Activation function
        use_residual : bool
            Whether to use residual connections
        dropout : float
            Dropout rate
        use_standardization : bool
            Whether to use input/output standardization (default: True, recommended)
        """
        super(TrajectoryPredictionPINN_PeInput, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=2,  # [delta, omega]
            activation=activation,
            use_residual=use_residual,
            dropout=dropout,
            use_standardization=use_standardization,
        )

        self.loss_fn = PhysicsInformedLoss(
            lambda_data=1.0, lambda_physics=0.1, lambda_ic=10.0, lambda_boundary=1.0
        )

    def predict_trajectory(
        self,
        t: torch.Tensor,
        delta0: torch.Tensor,
        omega0: torch.Tensor,
        H: torch.Tensor,
        D: torch.Tensor,
        alpha: torch.Tensor,  # Changed from Pload to alpha (unified approach)
        Pe: torch.Tensor,  # Pe(t) as input
        tf: Optional[torch.Tensor] = None,  # Fault start time (for 9-dim input)
        tc: Optional[torch.Tensor] = None,  # Fault clear time (for 9-dim input)
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectory with Pe(t) as input.

        Parameters:
        -----------
        t : torch.Tensor
            Time points
        delta0 : torch.Tensor
            Initial rotor angle
        omega0 : torch.Tensor
            Initial rotor speed
        H : torch.Tensor
            Inertia constant
        D : torch.Tensor
            Damping coefficient
        alpha : torch.Tensor
            Uniform load multiplier (e.g., 0.7 = 70% of base load).
            Works for both SMIB and multimachine systems.
            - SMIB: alpha × P_base = actual load
            - Multimachine: alpha × P_base_i = actual load_i (for all loads)
        Pe : torch.Tensor
            Electrical power at time points t (shape: [batch, time_steps] or [batch])
        **kwargs : dict
            Additional parameters (ignored for this variant)

        Returns:
        --------
        tuple : (delta_pred, omega_pred)
        """
        # Prepare input.
        batch_size = t.shape[0] if len(t.shape) > 0 else 1

        # Expand scalars to match time dimension
        if isinstance(delta0, (int, float)):
            delta0 = torch.tensor(delta0, device=t.device)
        if isinstance(omega0, (int, float)):
            omega0 = torch.tensor(omega0, device=t.device)
        if isinstance(H, (int, float)):
            H = torch.tensor(H, device=t.device)
        if isinstance(D, (int, float)):
            D = torch.tensor(D, device=t.device)
        if alpha is None:
            raise ValueError("alpha (uniform load multiplier) is REQUIRED.")
        elif isinstance(alpha, (int, float)):
            alpha = torch.tensor(alpha, device=t.device)
        if Pe is None:
            raise ValueError("Pe (electrical power) is REQUIRED for Pe-input model.")
        elif isinstance(Pe, (int, float)):
            Pe = torch.tensor(Pe, device=t.device)
        # tf, tc: optional for backward compatibility (7-dim); required for 9-dim
        if tf is None:
            tf = torch.tensor(1.0, device=t.device)
        elif isinstance(tf, (int, float)):
            tf = torch.tensor(tf, device=t.device)
        if tc is None:
            tc = torch.tensor(1.2, device=t.device)
        elif isinstance(tc, (int, float)):
            tc = torch.tensor(tc, device=t.device)

        # Expand to match time dimension
        if len(t.shape) > 0:
            delta0 = delta0.expand(batch_size) if delta0.numel() == 1 else delta0
            omega0 = omega0.expand(batch_size) if omega0.numel() == 1 else omega0
            H = H.expand(batch_size) if H.numel() == 1 else H
            D = D.expand(batch_size) if D.numel() == 1 else D
            alpha = alpha.expand(batch_size) if alpha.numel() == 1 else alpha
            tf = tf.expand(batch_size) if tf.numel() == 1 else tf
            tc = tc.expand(batch_size) if tc.numel() == 1 else tc
            # Pe should already match time dimension, but ensure it's flattened
            Pe = (
                Pe.flatten()
                if Pe.numel() > batch_size
                else Pe.expand(batch_size)
                if Pe.numel() == 1
                else Pe
            )

        # Construct input: 7-D [t, δ₀, ω₀, H, D, alpha, Pe(t)] or 9-D with tf, tc appended
        if self.input_dim == 7:
            x = torch.stack(
                [
                    t.flatten(),
                    delta0.flatten(),
                    omega0.flatten(),
                    H.flatten(),
                    D.flatten(),
                    alpha.flatten(),
                    Pe.flatten(),
                ],
                dim=1,
            )
        elif self.input_dim == 9:
            x = torch.stack(
                [
                    t.flatten(),
                    delta0.flatten(),
                    omega0.flatten(),
                    H.flatten(),
                    D.flatten(),
                    alpha.flatten(),
                    Pe.flatten(),
                    tf.flatten(),
                    tc.flatten(),
                ],
                dim=1,
            )
        else:
            raise ValueError(
                f"TrajectoryPredictionPINN_PeInput supports input_dim 7 or 9, got {self.input_dim}"
            )

        # Forward pass
        output = self.forward(x)

        # Split output into delta and omega
        delta_pred = output[:, 0]
        omega_pred = output[:, 1]

        return delta_pred, omega_pred

    def compute_loss(
        self,
        t_data: torch.Tensor,
        delta_pred: torch.Tensor,
        omega_pred: torch.Tensor,
        delta_obs: Optional[torch.Tensor] = None,
        omega_obs: Optional[torch.Tensor] = None,
        t_colloc: Optional[torch.Tensor] = None,
        delta_colloc: Optional[torch.Tensor] = None,
        omega_colloc: Optional[torch.Tensor] = None,
        t_ic: Optional[torch.Tensor] = None,
        delta0: Optional[torch.Tensor] = None,
        omega0: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        Pm: Optional[torch.Tensor] = None,
        Pe_from_andes: Optional[torch.Tensor] = None,  # NEW: Pe(t) from ANDES
        time_scale: Optional[torch.Tensor] = None,
        time_scale_sq: Optional[torch.Tensor] = None,
        delta_colloc_phys: Optional[torch.Tensor] = None,
        omega_colloc_phys: Optional[torch.Tensor] = None,
        t_colloc_phys: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for trajectory prediction with Pe(t) as input.

        Parameters:
        -----------
        t_data : torch.Tensor
            Time points for data loss
        delta_pred : torch.Tensor
            Predicted rotor angles
        omega_pred : torch.Tensor
            Predicted rotor speeds
        delta_obs : torch.Tensor, optional
            Observed rotor angles
        omega_obs : torch.Tensor, optional
            Observed rotor speeds
        t_colloc : torch.Tensor, optional
            Collocation points
        delta_colloc : torch.Tensor, optional
            Predicted delta at collocation points
        omega_colloc : torch.Tensor, optional
            Predicted omega at collocation points
        t_ic : torch.Tensor, optional
            Initial time
        delta0 : torch.Tensor, optional
            Initial rotor angle
        omega0 : torch.Tensor, optional
            Initial rotor speed
        M : torch.Tensor, optional
            Inertia constant
        D : torch.Tensor, optional
            Damping coefficient
        Pm : torch.Tensor, optional
            Mechanical power
        Pe_from_andes : torch.Tensor, optional
            Electrical power from ANDES (used directly in physics loss)
        time_scale : torch.Tensor, optional
            Scaling factor for first derivative
        time_scale_sq : torch.Tensor, optional
            Scaling factor for second derivative
        delta_colloc_phys : torch.Tensor, optional
            Denormalized delta at collocation points
        omega_colloc_phys : torch.Tensor, optional
            Denormalized omega at collocation points
        t_colloc_phys : torch.Tensor, optional
            Denormalized time at collocation points

        Returns:
        --------
        dict : Loss components
        """
        return self.loss_fn(
            t_data=t_data,
            delta_pred=delta_pred,
            omega_pred=omega_pred,
            delta_obs=delta_obs,
            omega_obs=omega_obs,
            t_colloc=t_colloc,
            delta_colloc=delta_colloc,
            omega_colloc=omega_colloc,
            t_ic=t_ic,
            delta0=delta0,
            omega0=omega0,
            M=M,
            D=D,
            Pm=Pm,
            Pe_from_andes=Pe_from_andes,  # Pass Pe directly
            use_pe_direct=True,  # Use Pe direct mode
            time_scale=time_scale,
            time_scale_sq=time_scale_sq,
            delta_colloc_phys=delta_colloc_phys,
            omega_colloc_phys=omega_colloc_phys,
            t_colloc_phys=t_colloc_phys,
        )

    def predict(
        self,
        t: np.ndarray,
        delta0: float,
        omega0: float,
        H: float,
        D: float,
        Pm: float,
        Pe: np.ndarray,  # NEW: Pe(t) as input
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict trajectory for given parameters with Pe(t) as input.

        Parameters:
        -----------
        t : np.ndarray
            Time points
        delta0 : float
            Initial rotor angle (rad)
        omega0 : float
            Initial rotor speed (pu)
        H : float
            Inertia constant (seconds)
        D : float
            Damping coefficient (pu)
        Pm : float
            Mechanical power (pu)
        Pe : np.ndarray
            Electrical power at time points t (pu)
        device : str
            Device ('cpu' or 'cuda')

        Returns:
        --------
        tuple : (delta_pred, omega_pred) as numpy arrays
        """
        self.eval()

        # Convert to tensors
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
        delta0_tensor = torch.tensor(delta0, dtype=torch.float32, device=device)
        omega0_tensor = torch.tensor(omega0, dtype=torch.float32, device=device)
        H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
        D_tensor = torch.tensor(D, dtype=torch.float32, device=device)
        Pm_tensor = torch.tensor(Pm, dtype=torch.float32, device=device)
        Pe_tensor = torch.tensor(Pe, dtype=torch.float32, device=device)

        with torch.no_grad():
            # 6th input slot is `alpha` in predict_trajectory; SMIB pe_direct_7 uses Pm there.
            delta_pred, omega_pred = self.predict_trajectory(
                t=t_tensor,
                delta0=delta0_tensor,
                omega0=omega0_tensor,
                H=H_tensor,
                D=D_tensor,
                alpha=Pm_tensor,
                Pe=Pe_tensor,
            )

        # Convert to numpy
        delta_pred_np = delta_pred.cpu().numpy()
        omega_pred_np = omega_pred.cpu().numpy()

        return delta_pred_np, omega_pred_np
