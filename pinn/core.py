"""
PINN Core Framework.

Base classes for Physics-Informed Neural Networks with swing equation constraints.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Standardise(nn.Module):
    """
    Standardize input by mean and standard deviation normalization.

    Similar to PINNSim's Standardise layer for input normalization.
    """

    def __init__(self, n_neurons: int):
        """
        Initialize standardization layer.

        Parameters:
        -----------
        n_neurons : int
            Number of input features
        """
        super(Standardise, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)
        self.eps = 1e-8

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (input - mean) / (std + eps)
        """
        return (input - self.mean) / (self.standard_deviation + self.eps)

    def set_standardisation(self, mean: torch.Tensor, standard_deviation: torch.Tensor):
        """
        Set standardization parameters from dataset statistics.

        Parameters:
        -----------
        mean : torch.Tensor
            Mean values (1-D tensor)
        standard_deviation : torch.Tensor
            Standard deviation values (1-D tensor)
        """
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise ValueError("Input statistics must be 1-D tensors.")

        if torch.sum(standard_deviation != 0) != standard_deviation.shape[0]:
            raise ValueError("Standard deviation contains elements equal to 0.")

        with torch.no_grad():
            self.mean = nn.Parameter(data=mean.clone(), requires_grad=False)
            self.standard_deviation = nn.Parameter(
                data=standard_deviation.clone(), requires_grad=False
            )


class Scale(nn.Module):
    """
    Scale output by mean and standard deviation (denormalization).

    Similar to PINNSim's Scale layer for output denormalization.
    """

    def __init__(self, n_neurons: int):
        """
        Initialize scaling layer.

        Parameters:
        -----------
        n_neurons : int
            Number of output features
        """
        super(Scale, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)
        self.eps = 1e-8

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: mean + input * std
        """
        return self.mean + input * self.standard_deviation

    def set_scaling(self, mean: torch.Tensor, standard_deviation: torch.Tensor):
        """
        Set scaling parameters from dataset statistics.

        Parameters:
        -----------
        mean : torch.Tensor
            Mean values (1-D tensor)
        standard_deviation : torch.Tensor
            Standard deviation values (1-D tensor)
        """
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise ValueError("Input statistics must be 1-D tensors.")

        if torch.sum(standard_deviation != 0) != standard_deviation.shape[0]:
            raise ValueError("Standard deviation contains elements equal to 0.")

        with torch.no_grad():
            self.mean = nn.Parameter(data=mean.clone(), requires_grad=False)
            self.standard_deviation = nn.Parameter(
                data=standard_deviation.clone(), requires_grad=False
            )


class NormalizedStateLoss(nn.Module):
    """
    Normalized state loss function.

    Similar to PINNSim's LossNormedState, scales errors by component importance.
    This is critical for SMIB where delta (~0.1-1.0 rad) and omega (~0.001-0.01 pu)
    have very different scales.
    """

    def __init__(self, scale_to_norm: Optional[torch.Tensor] = None):
        """
        Initialize normalized state loss.

        Parameters:
        -----------
        scale_to_norm : torch.Tensor, optional
            Scaling factors [1, output_dim] to normalize state errors.
            If None, uses identity (no normalization).
        """
        super(NormalizedStateLoss, self).__init__()
        if scale_to_norm is None:
            # Default: normalize by typical scales
            # delta: ~0.1-1.0 rad, omega: ~0.001-0.01 pu
            self.scale_to_norm = torch.tensor([[1.0, 100.0]])  # Weight omega errors more
        else:
            self.scale_to_norm = scale_to_norm

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized state loss.

        Parameters:
        -----------
        inputs : torch.Tensor
            Predicted states [batch_size, output_dim]
        targets : torch.Tensor
            Target states [batch_size, output_dim]

        Returns:
        --------
        torch.Tensor : Normalized loss (scalar)
        """
        loss_full = (inputs - targets) * self.scale_to_norm
        loss_point_wise = torch.square(torch.linalg.norm(loss_full, ord=2, dim=1, keepdim=True))
        loss = torch.mean(loss_point_wise)
        return loss


class LossWeightScheduler:
    """
    Scheduler for physics regularizer weight.

    Similar to PINNSim's LossWeightScheduler, gradually increases physics weight
    during training to allow model to fit data first, then enforce physics.
    """

    def __init__(
        self,
        max_value: float = 0.1,
        epochs_to_tenfold: int = 20,
        initial_value: float = 1e-6,
    ):
        """
        Initialize loss weight scheduler.

        Parameters:
        -----------
        max_value : float
            Maximum physics regularizer value
        epochs_to_tenfold : int
            Number of epochs to increase weight by 10x
        initial_value : float
            Initial physics regularizer value
        """
        self.max_value = torch.tensor(max_value)
        assert epochs_to_tenfold > 0
        self.epoch_factor = torch.tensor(10.0) ** (1.0 / epochs_to_tenfold)
        self.current_value = torch.tensor(initial_value) if max_value > 0.0 else torch.tensor(0.0)

    def step(self) -> float:
        """
        Update scheduler and return current value.

        Returns:
        --------
        float : Current physics regularizer value
        """
        self.current_value = torch.minimum(self.current_value * self.epoch_factor, self.max_value)
        return self.current_value.item()

    def get_value(self) -> float:
        """Get current physics regularizer value."""
        return self.current_value.item()


class AdaptiveLossWeightScheduler:
    """
    Adaptive physics weight scheduler based on loss magnitudes.

    Ensures physics loss contributes meaningfully without dominating.
    Computes weight to achieve target ratio: physics_loss / total_loss.

    Similar to LossWeightScheduler but uses actual loss magnitudes to
    compute appropriate weight, preventing physics loss from dominating.
    """

    def __init__(
        self,
        initial_ratio: float = 0.0,
        final_ratio: float = 0.5,
        warmup_epochs: int = 30,
        gradual_increase_epochs: int = 70,
        normalize_losses: bool = True,
    ):
        """
        Initialize adaptive loss weight scheduler.

        Parameters:
        -----------
        initial_ratio : float
            Initial physics contribution ratio (0.0 = no physics)
        final_ratio : float
            Final physics contribution ratio (0.5 = physics is 50% of total)
        warmup_epochs : int
            Number of epochs before physics loss starts increasing
        gradual_increase_epochs : int
            Number of epochs over which to gradually increase lambda (after warmup)
        normalize_losses : bool
            If True, normalize loss magnitudes before computing lambda (Option 4)
        """
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.warmup_epochs = warmup_epochs
        self.gradual_increase_epochs = gradual_increase_epochs
        self.normalize_losses = normalize_losses
        self.current_epoch = 0

    def compute_weight(
        self, data_loss: torch.Tensor, physics_loss: torch.Tensor, epoch: int
    ) -> float:
        """
        Compute adaptive physics weight based on loss magnitudes.

        Parameters:
        -----------
        data_loss : torch.Tensor
            Current data loss value
        physics_loss : torch.Tensor
            Current physics loss value
        epoch : int
            Current epoch number

        Returns:
        --------
        float : Physics weight (lambda_physics)
        """
        self.current_epoch = epoch

        # Stage 1: Warmup - no physics loss
        if epoch < self.warmup_epochs:
            return 0.0

        # Stage 2: Gradually increase target ratio (step-by-step)
        # Progress from initial_ratio to final_ratio over gradual_increase_epochs
        epochs_since_warmup = epoch - self.warmup_epochs
        if epochs_since_warmup < self.gradual_increase_epochs:
            # Gradually increase target ratio
            progress = epochs_since_warmup / max(1, self.gradual_increase_epochs)
            progress = min(1.0, progress)
            target_ratio = self.initial_ratio + progress * (self.final_ratio - self.initial_ratio)
        else:
            # After gradual increase period, use final_ratio
            target_ratio = self.final_ratio

        # If target ratio is 0, return 0
        if target_ratio <= 0:
            return 0.0

        # Extract loss magnitudes
        data_mag = data_loss.detach().item()
        phys_mag = physics_loss.detach().item()

        # Avoid division by zero
        if phys_mag < 1e-8 or data_mag < 1e-8:
            return 0.0

        # Option 4: Normalize loss magnitudes to handle scale mismatch
        if self.normalize_losses:
            # Normalize physics loss to match data loss scale
            # This allows us to compute lambda without needing extremely large values
            # The idea: scale physics loss so it's comparable to data loss
            if phys_mag > 1e-8:
                # Scale physics loss to data loss scale
                # After scaling: scaled_phys = phys_mag * (data_mag / phys_mag) = data_mag
                # So we can think of it as: lambda * data_mag contributes to total loss
                # To achieve target_ratio: (lambda * data_mag) / (data_mag + lambda * data_mag) = target_ratio
                # Simplifies to: lambda / (1 + lambda) = target_ratio
                # Solving: lambda = target_ratio / (1 - target_ratio)
                if target_ratio >= 1.0:
                    lambda_phys = 10.0  # Cap for very high ratios
                else:
                    lambda_phys = target_ratio / (1.0 - target_ratio)
            else:
                lambda_phys = 0.0
        else:
            # Original method: compute lambda based on raw loss magnitudes
            # Compute lambda_physics to achieve target ratio:
            # Want: (lambda_phys * phys_mag) / (data_mag + lambda_phys * phys_mag) = target_ratio
            # Solving for lambda_phys:
            # lambda_phys * phys_mag = target_ratio * (data_mag + lambda_phys * phys_mag)
            # lambda_phys * phys_mag = target_ratio * data_mag + target_ratio * lambda_phys * phys_mag
            # lambda_phys * phys_mag * (1 - target_ratio) = target_ratio * data_mag
            # lambda_phys = (target_ratio * data_mag) / (phys_mag * (1 - target_ratio))

            denominator = phys_mag * (1 - target_ratio)
            if denominator < 1e-8:
                # Target ratio approaching 1.0, use a reasonable cap
                lambda_phys = 1.0
            else:
                lambda_phys = (target_ratio * data_mag) / denominator

        # Clamp to reasonable range
        # With normalization, lambda can be > 1.0 (e.g., target_ratio=0.5 gives lambda=1.0)
        # Without normalization, keep original clamp to [0, 1]
        max_lambda = 10.0 if self.normalize_losses else 1.0
        lambda_phys = max(0.0, min(lambda_phys, max_lambda))

        return lambda_phys

    def get_current_ratio(self) -> float:
        """Get current target ratio"""
        if self.current_epoch < self.warmup_epochs:
            return 0.0
        epochs_since_warmup = self.current_epoch - self.warmup_epochs
        if epochs_since_warmup < self.gradual_increase_epochs:
            progress = epochs_since_warmup / max(1, self.gradual_increase_epochs)
            progress = min(1.0, progress)
            return self.initial_ratio + progress * (self.final_ratio - self.initial_ratio)
        else:
            return self.final_ratio


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for swing equation constraints.

    Implements:
    - Data loss: MSE between predicted and observed trajectories
    - Physics loss: Residual of swing equation at collocation points
    - Initial condition loss: MSE for initial conditions
    - Optional normalized state loss (like PINNSim)
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_ic: float = 10.0,
        lambda_steady_state: float = 0.0,
        lambda_boundary: float = 1.0,
        fn: float = 60.0,
        use_normalized_loss: bool = False,
        scale_to_norm: Optional[torch.Tensor] = None,
        physics_normalizer: Optional[object] = None,
        use_pe_direct: bool = False,
        num_machines: int = 1,  # NEW: Number of machines (1 for SMIB, >1 for multimachine)
        use_coi: bool = False,  # NEW: Use COI reference frame for multimachine
    ):
        """
        Initialize physics-informed loss.

        Parameters:
        -----------
        lambda_data : float
            Weight for data loss
        lambda_physics : float
            Weight for physics loss (can be scheduled)
        lambda_ic : float
            Weight for initial condition loss
        lambda_boundary : float
            Weight for boundary condition loss (at fault times)
        fn : float
            System frequency (Hz)
        use_normalized_loss : bool
            Whether to use normalized state loss (scales errors by component importance)
        scale_to_norm : torch.Tensor, optional
            Scaling factors for normalized loss [1, output_dim]
        """
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_ic = lambda_ic
        self.lambda_steady_state = lambda_steady_state
        self.lambda_boundary = lambda_boundary
        self.fn = fn
        self.two_pi_fn = 2.0 * math.pi * fn
        self.use_normalized_loss = use_normalized_loss
        self.physics_normalizer = physics_normalizer
        self.use_pe_direct = use_pe_direct  # Use Pe(t) directly from ANDES
        self.num_machines = num_machines  # NEW: Number of machines
        self.use_coi = use_coi and (num_machines > 1)  # NEW: COI only for multimachine

        # Physics regularizer (can be updated during training)
        self.physics_regulariser = torch.tensor(lambda_physics)

        if use_normalized_loss:
            self.normalized_loss_fn = NormalizedStateLoss(scale_to_norm=scale_to_norm)
        else:
            self.normalized_loss_fn = None

    def _compute_multimachine_coi_swing_residual(
        self,
        delta: torch.Tensor,  # [batch, num_machines]
        omega: torch.Tensor,  # [batch, num_machines]
        ddelta_dt: torch.Tensor,  # [batch, num_machines]
        d2delta_dt2: torch.Tensor,  # [batch, num_machines]
        M: torch.Tensor,  # [batch, num_machines]
        D: torch.Tensor,  # [batch, num_machines]
        Pm: torch.Tensor,  # [batch, num_machines]
        Pe: torch.Tensor,  # [batch, num_machines] or [batch, num_machines, time_steps]
        t: torch.Tensor,  # [batch] or [batch, time_steps]
    ) -> torch.Tensor:
        """
        Compute COI-referenced swing equation residual for multimachine systems.

        Formula: M_i·d²δ'_i/dt² + D_i·dδ'_i/dt = P_mi - P_ei - M_i·d²δ_COI/dt²
        where δ'_i = δ_i - δ_COI

        Parameters:
        -----------
        delta : torch.Tensor
            Rotor angles [batch, num_machines]
        omega : torch.Tensor
            Rotor speeds [batch, num_machines]
        ddelta_dt : torch.Tensor
            First derivative of delta [batch, num_machines]
        d2delta_dt2 : torch.Tensor
            Second derivative of delta [batch, num_machines]
        M : torch.Tensor
            Inertia constants [batch, num_machines] (M = 2*H for 60 Hz)
        D : torch.Tensor
            Damping coefficients [batch, num_machines]
        Pm : torch.Tensor
            Mechanical power [batch, num_machines]
        Pe : torch.Tensor
            Electrical power [batch, num_machines] or [batch, num_machines, time_steps]
        t : torch.Tensor
            Time [batch] or [batch, time_steps]

        Returns:
        --------
        torch.Tensor
            Swing equation residual [batch, num_machines]
        """
        batch_size = delta.shape[0]
        num_machines = delta.shape[1]

        # 1. Compute COI angle and speed
        M_total = torch.sum(M, dim=1, keepdim=True)  # [batch, 1]
        delta_coi = torch.sum(M * delta, dim=1, keepdim=True) / (M_total + 1e-8)  # [batch, 1]
        omega_coi = torch.sum(M * omega, dim=1, keepdim=True) / (M_total + 1e-8)  # [batch, 1]

        # 2. Compute COI derivatives
        # COI first derivative: dδ_COI/dt
        ddelta_coi_dt = torch.sum(M * ddelta_dt, dim=1, keepdim=True) / (
            M_total + 1e-8
        )  # [batch, 1]

        # COI second derivative: d²δ_COI/dt²
        d2delta_coi_dt2 = torch.sum(M * d2delta_dt2, dim=1, keepdim=True) / (
            M_total + 1e-8
        )  # [batch, 1]

        # 3. Compute relative angles: δ'_i = δ_i - δ_COI
        delta_relative = delta - delta_coi  # [batch, num_machines]

        # 4. Compute relative angle derivatives
        ddelta_relative_dt = ddelta_dt - ddelta_coi_dt  # [batch, num_machines]
        d2delta_relative_dt2 = d2delta_dt2 - d2delta_coi_dt2  # [batch, num_machines]

        # 5. Handle Pe shape (may be [batch, num_machines] or [batch, num_machines, time_steps])
        if len(Pe.shape) == 3:
            # If Pe has time dimension, take mean or use appropriate time step
            # For now, use mean across time dimension
            Pe = Pe.mean(dim=2)  # [batch, num_machines]
        elif len(Pe.shape) == 1:
            # If Pe is [batch*num_machines], reshape
            if Pe.shape[0] == batch_size * num_machines:
                Pe = Pe.view(batch_size, num_machines)
            else:
                # Broadcast to all machines if single value
                Pe = Pe.unsqueeze(1).expand(-1, num_machines)

        # 6. Compute COI-referenced swing equation residual
        # M_i·d²δ'_i/dt² + D_i·dδ'_i/dt = P_mi - P_ei - M_i·d²δ_COI/dt²
        # Rearranged: M_i·d²δ'_i/dt² + D_i·dδ'_i/dt - (P_mi - P_ei) + M_i·d²δ_COI/dt² = 0
        swing_residual = (
            M * d2delta_relative_dt2
            + D * ddelta_relative_dt
            - (Pm - Pe)
            + M * d2delta_coi_dt2  # COI acceleration term
        )  # [batch, num_machines]

        return swing_residual

    def compute_derivatives(
        self, t: torch.Tensor, delta: torch.Tensor, omega: torch.Tensor, use_jvp: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute first and second derivatives using automatic differentiation.

        Parameters:
        -----------
        t : torch.Tensor
            Time tensor (must require grad)
        delta : torch.Tensor
            Rotor angle predictions
        omega : torch.Tensor
            Rotor speed predictions (not used but kept for API consistency)
        use_jvp : bool
            Whether to use jvp (Jacobian-vector product) for more efficient computation

        Returns:
        --------
        tuple : (ddelta_dt, d2delta_dt2)
        """
        # Ensure t requires grad for derivative computation
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        # Check if delta is connected to t in the computation graph
        # If delta doesn't require grad or has no grad_fn, it's not connected to t
        if not delta.requires_grad and delta.grad_fn is None:
            raise RuntimeError(
                "delta tensor is not connected to t in the computation graph. "
                "This usually happens when:\n"
                "1. delta was computed with torch.no_grad() context\n"
                "2. delta was detached from the computation graph\n"
                "3. delta was computed from a model in eval() mode with "
                "operations that detach gradients\n"
                "Solution: Ensure delta_colloc is computed from the model "
                "using t_colloc "
                "with requires_grad=True, and the model is in train() mode."
            )

        if use_jvp:
            # More efficient computation using jvp (like PINNSim)
            # First derivative
            (ddelta_dt,) = torch.autograd.functional.jvp(
                func=lambda t_in: delta,
                inputs=t,
                v=torch.ones_like(t),
                create_graph=True,
            )

            # Second derivative
            (d2delta_dt2,) = torch.autograd.functional.jvp(
                func=lambda t_in: ddelta_dt,
                inputs=t,
                v=torch.ones_like(t),
                create_graph=True,
            )
        else:
            # Standard approach using autograd.grad
            # First derivative: dδ/dt.
            ddelta_dt = torch.autograd.grad(
                outputs=delta,
                inputs=t,
                grad_outputs=torch.ones_like(delta),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Second derivative: d²δ/dt²
            d2delta_dt2 = torch.autograd.grad(
                outputs=ddelta_dt,
                inputs=t,
                grad_outputs=torch.ones_like(ddelta_dt),
                create_graph=True,
                retain_graph=True,
            )[0]

        return ddelta_dt, d2delta_dt2

    def compute_electrical_power(
        self,
        delta: torch.Tensor,
        t: torch.Tensor,
        Xprefault: torch.Tensor,
        Xfault: torch.Tensor,
        Xpostfault: torch.Tensor,
        tf: torch.Tensor,
        tc: torch.Tensor,
        V1: float = 1.05,
        V2: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute electrical power Pe(t) = (V1*V2/X(t))*sin(δ(t)).
        where X(t) switches based on fault state.

        Parameters:
        -----------
        delta : torch.Tensor
            Rotor angle
        t : torch.Tensor
            Time
        Xprefault : torch.Tensor
            Pre-fault reactance
        Xfault : torch.Tensor
            Fault reactance
        Xpostfault : torch.Tensor
            Post-fault reactance
        tf : torch.Tensor
            Fault start time
        tc : torch.Tensor
            Fault clear time
        V1 : float
            Generator voltage (pu)
        V2 : float
            Infinite bus voltage (pu)

        Returns:
        --------
        torch.Tensor : Electrical power Pe(t)
        """
        # Determine system state for each time point.
        # Pre-fault: t < tf
        # During fault: tf <= t <= tc
        # Post-fault: t > tc

        # Create masks for each state
        pre_fault_mask = (t < tf).float()
        during_fault_mask = ((t >= tf) & (t <= tc)).float()
        post_fault_mask = (t > tc).float()

        # Select appropriate reactance
        X = pre_fault_mask * Xprefault + during_fault_mask * Xfault + post_fault_mask * Xpostfault

        # Compute Pe = (V1*V2/X)*sin(δ)
        Pe = (V1 * V2 / (X + 1e-8)) * torch.sin(delta)

        return Pe

    def forward(
        self,
        t_data: torch.Tensor,
        delta_pred: torch.Tensor,
        omega_pred: torch.Tensor,
        delta_obs: Optional[torch.Tensor] = None,
        omega_obs: Optional[torch.Tensor] = None,
        t_colloc: Optional[torch.Tensor] = None,
        delta_colloc: Optional[torch.Tensor] = None,
        omega_colloc: Optional[torch.Tensor] = None,
        delta_colloc_phys: Optional[torch.Tensor] = None,  # Denormalized delta for physics equation
        omega_colloc_phys: Optional[torch.Tensor] = None,  # Denormalized omega for physics equation
        t_colloc_phys: Optional[torch.Tensor] = None,  # Denormalized time for physics equation
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
        tf_norm: Optional[
            torch.Tensor
        ] = None,  # Normalized fault start time (for steady-state loss)
        time_scale: Optional[torch.Tensor] = None,
        time_scale_sq: Optional[torch.Tensor] = None,
        Pe_from_andes: Optional[torch.Tensor] = None,  # NEW: Pe(t) from ANDES
        use_pe_direct: Optional[bool] = None,  # NEW: Override instance setting
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Parameters:
        -----------
        t_data : torch.Tensor
            Time points for data loss
        delta_pred : torch.Tensor
            Predicted rotor angles
        omega_pred : torch.Tensor
            Predicted rotor speeds
        delta_obs : torch.Tensor, optional
            Observed rotor angles (for data loss)
        omega_obs : torch.Tensor, optional
            Observed rotor speeds (for data loss)
        t_colloc : torch.Tensor, optional
            Collocation points for physics loss
        delta_colloc : torch.Tensor, optional
            Predicted delta at collocation points (normalized)
        omega_colloc : torch.Tensor, optional
            Predicted omega at collocation points (normalized)
        delta_colloc_phys : torch.Tensor, optional
            Denormalized delta at collocation points (for physics equation)
        omega_colloc_phys : torch.Tensor, optional
            Denormalized omega at collocation points (for physics equation)
        t_colloc_phys : torch.Tensor, optional
            Denormalized time at collocation points (for physics equation)
        t_ic : torch.Tensor, optional
            Initial time point
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
        Xprefault : torch.Tensor, optional
            Pre-fault reactance
        Xfault : torch.Tensor, optional
            Fault reactance
        Xpostfault : torch.Tensor, optional
            Post-fault reactance
        tf : torch.Tensor, optional
            Fault start time
        tc : torch.Tensor, optional
            Fault clear time

        Returns:
        --------
        dict : Dictionary of loss components
        """
        losses = {}

        # Data loss
        if delta_obs is not None and omega_obs is not None:
            if self.use_normalized_loss and self.normalized_loss_fn is not None:
                # Use normalized state loss (like PINNSim)
                pred_states = torch.stack([delta_pred, omega_pred], dim=1)
                obs_states = torch.stack([delta_obs, omega_obs], dim=1)
                loss_data = self.normalized_loss_fn(pred_states, obs_states)
            else:
                # Standard MSE loss
                loss_data = torch.mean((delta_pred - delta_obs) ** 2) + torch.mean(
                    (omega_pred - omega_obs) ** 2
                )
            losses["data"] = loss_data
        else:
            losses["data"] = torch.tensor(0.0, device=delta_pred.device)

        # Physics loss at collocation points
        if (
            t_colloc is not None
            and delta_colloc is not None
            and omega_colloc is not None
            and M is not None
            and D is not None
        ):
            # === IMPROVED PHYSICS LOSS COMPUTATION ===
            # Use PhysicsNormalizer if available for proper denormalization

            # 1. Compute derivatives with normalized values (preserves computation graph)
            use_jvp = False  # Can be made configurable
            ddelta_dt_norm, d2delta_dt2_norm = self.compute_derivatives(
                t_colloc, delta_colloc, omega_colloc, use_jvp=use_jvp
            )

            # 2. Denormalize for physics equation
            if self.physics_normalizer is not None:
                # Use PhysicsNormalizer for gradient-preserving denormalization
                (
                    delta_for_physics,
                    omega_for_physics,
                    t_for_physics,
                    time_scale_factor,
                    time_scale_sq_factor,
                ) = self.physics_normalizer.denormalize_for_physics(
                    delta_colloc, omega_colloc, t_colloc
                )
                # Scale derivatives to physical units
                ddelta_dt = ddelta_dt_norm * time_scale_factor
                d2delta_dt2 = d2delta_dt2_norm * time_scale_sq_factor
            elif delta_colloc_phys is not None and omega_colloc_phys is not None:
                # Fallback: Use pre-computed denormalized values
                delta_for_physics = delta_colloc_phys
                omega_for_physics = omega_colloc_phys
                t_for_physics = t_colloc_phys if t_colloc_phys is not None else t_colloc
                # Scale derivatives if factors provided
                if time_scale is not None and time_scale_sq is not None:
                    ddelta_dt = ddelta_dt_norm * time_scale
                    d2delta_dt2 = d2delta_dt2_norm * time_scale_sq
                else:
                    ddelta_dt = ddelta_dt_norm
                    d2delta_dt2 = d2delta_dt2_norm
            else:
                # No denormalization available - assume already in physical units
                delta_for_physics = delta_colloc
                omega_for_physics = omega_colloc
                t_for_physics = t_colloc
                ddelta_dt = ddelta_dt_norm
                d2delta_dt2 = d2delta_dt2_norm

            # Compute electrical power (needs physical units for time comparisons)
            # Use Pe directly from ANDES if use_pe_direct is True
            use_pe_direct_mode = use_pe_direct if use_pe_direct is not None else self.use_pe_direct

            if use_pe_direct_mode and Pe_from_andes is not None:
                # Use Pe(t) directly from ANDES (no computation from reactances)
                Pe = Pe_from_andes
                # Ensure Pe matches collocation point dimensions
                if Pe.shape[0] != delta_colloc.shape[0]:
                    # Interpolate or expand Pe to match collocation points
                    # For now, assume Pe is already at collocation points
                    if len(Pe.shape) == 1 and Pe.shape[0] == delta_colloc.shape[0]:
                        Pe = Pe
                    else:
                        # Fallback: use first value or interpolate
                        Pe = (
                            Pe.flatten()[: delta_colloc.shape[0]]
                            if Pe.numel() >= delta_colloc.shape[0]
                            else torch.zeros_like(delta_colloc)
                        )
            elif (
                Xprefault is not None
                and Xfault is not None
                and Xpostfault is not None
                and tf is not None
                and tc is not None
                and Pm is not None
            ):
                # Compute Pe from reactances (existing behavior)
                Pe = self.compute_electrical_power(
                    delta_for_physics, t_for_physics, Xprefault, Xfault, Xpostfault, tf, tc
                )
            else:
                # Default Pe if not provided
                Pe = torch.zeros_like(delta_colloc)

            # Swing equation computation
            # For SMIB: M·d²δ/dt² + D·dδ/dt = Pm - Pe
            # For multimachine with COI: M_i·d²δ'_i/dt² + D_i·dδ'_i/dt = P_mi - P_ei - M_i·d²δ_COI/dt²
            # where δ'_i = δ_i - δ_COI

            if self.use_coi and self.num_machines > 1:
                # MULTIMACHINE COI-REFERENCED SWING EQUATION
                # Check if inputs are multimachine (shape: [batch, num_machines])
                if (
                    len(delta_for_physics.shape) >= 2
                    and delta_for_physics.shape[1] == self.num_machines
                ):
                    swing_residual = self._compute_multimachine_coi_swing_residual(
                        delta_for_physics,
                        omega_for_physics,
                        ddelta_dt,
                        d2delta_dt2,
                        M,
                        D,
                        Pm,
                        Pe,
                        t_for_physics,
                    )
                else:
                    # Fallback to SMIB if shape doesn't match
                    swing_residual = M * d2delta_dt2 + D * ddelta_dt - (Pm - Pe)
            else:
                # SMIB SWING EQUATION (original implementation)
                swing_residual = M * d2delta_dt2 + D * ddelta_dt - (Pm - Pe)

            # dδ/dt = 2π·fn·(ω - 1) - FIXED: Use denormalized omega_for_physics
            # CRITICAL FIX: omega_colloc is normalized, but physics equation needs physical units
            # For multimachine with COI: use relative speeds ω'_i = ω_i - ω_COI
            if (
                self.use_coi
                and self.num_machines > 1
                and len(omega_for_physics.shape) >= 2
                and omega_for_physics.shape[1] == self.num_machines
            ):
                # Compute COI speed for angle-speed relation
                if M is not None and len(M.shape) >= 2 and M.shape[1] == self.num_machines:
                    M_total = torch.sum(M, dim=1, keepdim=True)  # [batch, 1]
                    omega_coi = torch.sum(M * omega_for_physics, dim=1, keepdim=True) / (
                        M_total + 1e-8
                    )  # [batch, 1]
                    omega_relative = omega_for_physics - omega_coi  # [batch, num_machines]
                    # Angle-speed relation for relative angles: dδ'_i/dt = 2π·fn·(ω'_i)
                    angle_speed_relation = ddelta_dt - self.two_pi_fn * omega_relative
                else:
                    # Fallback to standard relation if M not available
                    angle_speed_relation = ddelta_dt - self.two_pi_fn * (omega_for_physics - 1.0)
            else:
                # SMIB angle-speed relation (original)
                angle_speed_relation = ddelta_dt - self.two_pi_fn * (omega_for_physics - 1.0)

            # === IMPROVED ADAPTIVE RESIDUAL NORMALIZATION ===
            # Normalize residuals based on data loss magnitude to prevent domination

            with torch.no_grad():
                # Get data loss magnitude for adaptive scaling
                data_magnitude = losses.get("data", torch.tensor(1.0)).detach()

                # Compute residual magnitudes
                swing_residual_mag = torch.abs(swing_residual).mean()
                angle_speed_mag = torch.abs(angle_speed_relation).mean()

                # Adaptive scaling: normalize so physics loss is comparable to data loss
                # Target: normalized residuals have similar scale to data loss
                swing_scale = torch.maximum(
                    swing_residual_mag / 10.0,  # Allow normalized residual ~10
                    data_magnitude * 0.1,  # Scale relative to data loss
                )
                swing_scale = swing_scale.clamp(min=0.01)

                angle_scale = torch.maximum(angle_speed_mag / 10.0, data_magnitude * 0.1)
                angle_scale = angle_scale.clamp(min=0.01)

            # Normalize residuals
            swing_norm = swing_residual / (swing_scale + 1e-8)
            angle_norm = angle_speed_relation / (angle_scale + 1e-8)

            # Clamp to prevent extremes (after clamping, max squared value is 100)
            swing_norm = torch.clamp(swing_norm, min=-10.0, max=10.0)
            angle_norm = torch.clamp(angle_norm, min=-10.0, max=10.0)

            # Compute physics loss
            loss_physics = torch.mean(swing_norm**2) + torch.mean(angle_norm**2)

            # Store physics residuals (unnormalized) for monitoring
            losses["physics_residual_swing"] = torch.abs(swing_residual).mean().detach()
            losses["physics_residual_angle"] = torch.abs(angle_speed_relation).mean().detach()

            losses["physics"] = loss_physics
        else:
            losses["physics"] = torch.tensor(0.0, device=delta_pred.device)

        # Initial condition loss
        if t_ic is not None and delta0 is not None and omega0 is not None:
            # Get predictions at initial time
            # Assuming t_ic is a single point or we need to interpolate
            if len(t_ic.shape) == 0:
                t_ic = t_ic.unsqueeze(0)

            # Find closest time point in predictions
            # For simplicity, assume first point is initial condition
            delta_pred_ic = delta_pred[0] if len(delta_pred) > 0 else delta_pred
            omega_pred_ic = omega_pred[0] if len(omega_pred) > 0 else omega_pred

            loss_ic = (delta_pred_ic - delta0) ** 2 + (omega_pred_ic - omega0) ** 2
            losses["ic"] = loss_ic
        else:
            losses["ic"] = torch.tensor(0.0, device=delta_pred.device)

        # Steady-state auxiliary loss: enforce (delta, omega) = (delta0, omega0) for all t < tf
        if (
            self.lambda_steady_state > 0
            and t_data is not None
            and tf_norm is not None
            and delta0 is not None
            and omega0 is not None
        ):
            t_obs_s = t_data.squeeze()
            tf_s = tf_norm.squeeze()
            if t_obs_s.dim() == 0:
                t_obs_s = t_obs_s.unsqueeze(0)
            pre_fault = (t_obs_s < tf_s).float()
            n_pre = pre_fault.sum().clamp(min=1.0)
            d0 = delta0.squeeze()
            o0 = omega0.squeeze()
            ss_delta = ((delta_pred - d0) ** 2 * pre_fault).sum() / n_pre
            ss_omega = ((omega_pred - o0) ** 2 * pre_fault).sum() / n_pre
            losses["steady_state"] = ss_delta + ss_omega
        else:
            losses["steady_state"] = torch.tensor(0.0, device=delta_pred.device)

        # Boundary condition loss (continuity at fault times)
        if tf is not None and tc is not None and t_colloc is not None and delta_colloc is not None:
            # This would require more sophisticated handling
            # For now, skip boundary loss
            losses["boundary"] = torch.tensor(0.0, device=delta_pred.device)
        else:
            losses["boundary"] = torch.tensor(0.0, device=delta_pred.device)

        # Total loss (use physics_regulariser which can be updated during training)
        total_loss = (
            self.lambda_data * losses["data"]
            + self.physics_regulariser * losses["physics"]
            + self.lambda_ic * losses["ic"]
            + self.lambda_steady_state * losses["steady_state"]
            + self.lambda_boundary * losses["boundary"]
        )

        losses["total"] = total_loss

        return losses


class _SameDimResidualBlock(nn.Module):
    """Two-layer same-width block with residual: ``out = x + W2(act(D(W1(x))))``."""

    def __init__(self, dim: int, activation: nn.Module, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = activation
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.lin2(out)
        out = self.act(out)
        out = self.drop(out)
        return x + out


class _DimChangeBlock(nn.Module):
    """Width change with skip: ``out = f(x) + proj(x)`` (``f`` = linear, optional dropout, act)."""

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module, dropout: float):
        super().__init__()
        seq: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if dropout > 0:
            seq.append(nn.Dropout(dropout))
        seq.append(activation)
        self.f = nn.Sequential(*seq)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x) + self.proj(x)


class _ResidualFeatureNet(nn.Module):
    """
    Stem to ``hidden_dims[0]``, then residual / transition blocks matching ``hidden_dims``,
    then linear head to ``output_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: nn.Module,
        dropout: float,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty for residual backbone")
        stem: List[nn.Module] = [nn.Linear(input_dim, hidden_dims[0])]
        if dropout > 0:
            stem.append(nn.Dropout(dropout))
        stem.append(activation)
        self.stem = nn.Sequential(*stem)
        blocks: List[nn.Module] = []
        for i in range(1, len(hidden_dims)):
            if hidden_dims[i] == hidden_dims[i - 1]:
                blocks.append(_SameDimResidualBlock(hidden_dims[i], activation, dropout))
            else:
                blocks.append(
                    _DimChangeBlock(hidden_dims[i - 1], hidden_dims[i], activation, dropout)
                )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


class PINN(nn.Module):
    """
    Base Physics-Informed Neural Network for power system transient stability.

    Architecture: Multi-layer feedforward network with input/output standardization
    and state normalization, similar to PINNSim's approach.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [64, 64, 64, 64],
        output_dim: int = 2,
        activation: str = "tanh",
        use_residual: bool = False,
        dropout: float = 0.0,
        use_standardization: bool = True,
    ):
        """
        Initialize PINN.

        Parameters:
        -----------
        input_dim : int
            Input dimension (time + system parameters)
        hidden_dims : list
            Hidden layer dimensions
        output_dim : int
            Output dimension (typically 2 for δ and ω)
        activation : str
            Activation function: 'tanh', 'relu', 'sigmoid'
        use_residual : bool
            If True, use stem + residual / skip blocks (see ``_ResidualFeatureNet``).
            If False, use the flat ``nn.Sequential`` MLP (default). Set True only when
            matching older checkpoints trained with residual blocks.
        dropout : float
            Dropout rate (0 = no dropout)
        use_standardization : bool
            Whether to use input/output standardization layers
        """
        super(PINN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.use_standardization = use_standardization

        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Input standardization layer
        if use_standardization:
            self.input_standardization = Standardise(input_dim)
        else:
            self.input_standardization = None

        # Build backbone: real residual blocks when use_residual else legacy Sequential
        if use_residual:
            self.network = _ResidualFeatureNet(
                input_dim=input_dim,
                hidden_dims=list(hidden_dims),
                output_dim=output_dim,
                activation=self.activation,
                dropout=dropout,
            )
        else:
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.append(self.activation)
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        # Output scaling layer
        if use_standardization:
            self.output_scaling = Scale(output_dim)
        else:
            self.output_scaling = None

        # State normalization parameters (for delta and omega separately)
        # These are set from dataset statistics
        self.state_mean = nn.Parameter(torch.zeros((1, output_dim)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((1, output_dim)), requires_grad=False)

        # Normalization scales for delta and omega (similar to PINNSim's norm_to_scale)
        # These are used to scale state variables to appropriate ranges
        self.norm_to_scale_delta = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.norm_to_scale_omega = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == nn.Tanh():
                    nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("tanh"))
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional input/output standardization.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch_size, input_dim]

        Returns:
        --------
        torch.Tensor : Output tensor [batch_size, output_dim]
        """
        # Input standardization
        if self.input_standardization is not None:
            x = self.input_standardization(x)

        # Network forward pass
        output = self.network(x)

        # Output scaling
        if self.output_scaling is not None:
            output = self.output_scaling(output)

        return output

    def transform_state_to_norm(self, state: torch.Tensor) -> torch.Tensor:
        """
        Transform state to normalized form.

        Parameters:
        -----------
        state : torch.Tensor
            State tensor [batch_size, output_dim] in physical units

        Returns:
        --------
        torch.Tensor : Normalized state
        """
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def transform_state_to_scale(self, state: torch.Tensor) -> torch.Tensor:
        """
        Transform normalized state back to physical units.

        Parameters:
        -----------
        state : torch.Tensor
            Normalized state tensor [batch_size, output_dim]

        Returns:
        --------
        torch.Tensor : State in physical units
        """
        return state * self.state_std + self.state_mean

    def transform_state_to_scale_only(self, state: torch.Tensor) -> torch.Tensor:
        """
        Scale normalized state by std only (no mean shift).

        Parameters:
        -----------
        state : torch.Tensor
            Normalized state tensor [batch_size, output_dim]

        Returns:
        --------
        torch.Tensor : Scaled state (no mean added)
        """
        return state * self.state_std

    def transform_state_to_norm_only(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state by std only (no mean shift).

        Parameters:
        -----------
        state : torch.Tensor
            State tensor [batch_size, output_dim] in physical units

        Returns:
        --------
        torch.Tensor : Normalized state (no mean subtracted)
        """
        return state / (self.state_std + 1e-8)

    def update_input_standardization(self, input_data: torch.Tensor):
        """
        Update input standardization statistics from dataset.

        Parameters:
        -----------
        input_data : torch.Tensor
            Input data tensor [n_samples, input_dim]
        """
        if self.input_standardization is None:
            return

        input_std, input_mean = torch.std_mean(input_data, dim=0)
        self.input_standardization.set_standardisation(input_mean, input_std)

    def update_output_scaling(self, output_data: torch.Tensor):
        """
        Update output scaling statistics from dataset.

        Parameters:
        -----------
        output_data : torch.Tensor
            Output data tensor [n_samples, output_dim]
        """
        if self.output_scaling is None:
            return

        output_std, output_mean = torch.std_mean(output_data, dim=0)
        self.output_scaling.set_scaling(output_mean, output_std)

    def update_state_scaling(
        self,
        state_data: torch.Tensor,
        norm_to_scale_delta: Optional[float] = None,
        norm_to_scale_omega: Optional[float] = None,
    ):
        """
        Update state normalization parameters from dataset.

        Parameters:
        -----------
        state_data : torch.Tensor
            State data tensor [n_samples, output_dim] where output_dim=2 for [delta, omega]
        norm_to_scale_delta : float, optional
            Normalization scale for delta (radians). If None, computed from data.
        norm_to_scale_omega : float, optional
            Normalization scale for omega (pu). If None, computed from data.
        """
        # Compute state mean and std
        state_std, state_mean = torch.std_mean(state_data, dim=0, keepdim=True)

        with torch.no_grad():
            self.state_mean = nn.Parameter(state_mean.clone(), requires_grad=False)
            self.state_std = nn.Parameter(state_std.clone(), requires_grad=False)

        # Set normalization scales for delta and omega separately
        if norm_to_scale_delta is not None:
            self.norm_to_scale_delta = nn.Parameter(
                torch.tensor(norm_to_scale_delta), requires_grad=False
            )
        else:
            # Use std of delta as default scale
            self.norm_to_scale_delta = nn.Parameter(state_std[0, 0].clone(), requires_grad=False)

        if norm_to_scale_omega is not None:
            self.norm_to_scale_omega = nn.Parameter(
                torch.tensor(norm_to_scale_omega), requires_grad=False
            )
        else:
            # Use std of omega as default scale
            self.norm_to_scale_omega = nn.Parameter(state_std[0, 1].clone(), requires_grad=False)

    def adjust_to_dataset(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor,
        norm_to_scale_delta: Optional[float] = None,
        norm_to_scale_omega: Optional[float] = None,
    ):
        """
        Adjust all normalization parameters from dataset (convenience method).

        Similar to PINNSim's adjust_to_dataset method.

        Parameters:
        -----------
        input_data : torch.Tensor
            Input data tensor [n_samples, input_dim]
        output_data : torch.Tensor
            Output data tensor [n_samples, output_dim]
        norm_to_scale_delta : float, optional
            Normalization scale for delta
        norm_to_scale_omega : float, optional
            Normalization scale for omega
        """
        self.update_input_standardization(input_data)
        self.update_output_scaling(output_data)
        self.update_state_scaling(
            output_data,
            norm_to_scale_delta=norm_to_scale_delta,
            norm_to_scale_omega=norm_to_scale_omega,
        )

    def predict_trajectory(
        self,
        t: torch.Tensor,
        delta0: torch.Tensor,
        omega0: torch.Tensor,
        H: torch.Tensor,
        D: torch.Tensor,
        Pm: torch.Tensor,
        Xprefault: torch.Tensor = None,
        Xfault: torch.Tensor = None,
        Xpostfault: torch.Tensor = None,
        tf: torch.Tensor = None,
        tc: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectory given initial conditions and system parameters.

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
        Pm : torch.Tensor
            Mechanical power (pu). **REQUIRED** - Must match training data range (typically 0.4-0.9)
        Xprefault : torch.Tensor
            Pre-fault reactance
        Xfault : torch.Tensor
            Fault reactance
        Xpostfault : torch.Tensor
            Post-fault reactance
        tf : torch.Tensor
            Fault start time
        tc : torch.Tensor
            Fault clear time
        **kwargs : dict
            Additional parameters

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
        if Pm is None:
            raise ValueError(
                "Pm (mechanical power) is REQUIRED. "
                "The model was trained with varying Pm (0.4-0.9 pu). "
                "You must provide the correct Pm value for accurate predictions."
            )
        elif isinstance(Pm, (int, float)):
            Pm = torch.tensor(Pm, device=t.device)
        if Xprefault is None:
            Xprefault = torch.tensor(0.5, device=t.device)
        elif isinstance(Xprefault, (int, float)):
            Xprefault = torch.tensor(Xprefault, device=t.device)
        if Xfault is None:
            Xfault = torch.tensor(0.0001, device=t.device)
        elif isinstance(Xfault, (int, float)):
            Xfault = torch.tensor(Xfault, device=t.device)
        if Xpostfault is None:
            Xpostfault = torch.tensor(0.5, device=t.device)
        elif isinstance(Xpostfault, (int, float)):
            Xpostfault = torch.tensor(Xpostfault, device=t.device)
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
            Pm = Pm.expand(batch_size) if Pm.numel() == 1 else Pm
            Xprefault = Xprefault.expand(batch_size) if Xprefault.numel() == 1 else Xprefault
            Xfault = Xfault.expand(batch_size) if Xfault.numel() == 1 else Xfault
            Xpostfault = Xpostfault.expand(batch_size) if Xpostfault.numel() == 1 else Xpostfault
            tf = tf.expand(batch_size) if tf.numel() == 1 else tf
            tc = tc.expand(batch_size) if tc.numel() == 1 else tc

        # Construct input: [t, delta0, omega0, H, D, Pm, Xprefault, Xfault, Xpostfault, tf, tc]
        # Note: Model expects H (not M) in the input features
        x = torch.stack(
            [
                t.flatten(),
                delta0.flatten(),
                omega0.flatten(),
                H.flatten(),  # Use H directly (model expects H, not M)
                D.flatten(),
                Pm.flatten(),  # Added: mechanical power
                Xprefault.flatten(),
                Xfault.flatten(),
                Xpostfault.flatten(),
                tf.flatten(),
                tc.flatten(),
            ],
            dim=1,
        )

        # Forward pass
        output = self.forward(x)

        # Split output into delta and omega
        delta_pred = output[:, 0]
        omega_pred = output[:, 1]

        return delta_pred, omega_pred

    def forward_dt(
        self,
        t: torch.Tensor,
        delta0: torch.Tensor,
        omega0: torch.Tensor,
        H: torch.Tensor,
        D: torch.Tensor,
        Pm: torch.Tensor,
        Xprefault: Optional[torch.Tensor] = None,
        Xfault: Optional[torch.Tensor] = None,
        Xpostfault: Optional[torch.Tensor] = None,
        tf: Optional[torch.Tensor] = None,
        tc: Optional[torch.Tensor] = None,
        use_jvp: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with time derivative computation.

        Similar to PINNSim's forward_dt method, computes prediction and its time derivative.

        Parameters:
        -----------
        t : torch.Tensor
            Time points (must require grad)
        delta0 : torch.Tensor
            Initial rotor angle
        omega0 : torch.Tensor
            Initial rotor speed
        H : torch.Tensor
            Inertia constant
        D : torch.Tensor
            Damping coefficient
        Xprefault : torch.Tensor
            Pre-fault reactance
        Xfault : torch.Tensor
            Fault reactance
        Xpostfault : torch.Tensor
            Post-fault reactance
        tf : torch.Tensor
            Fault start time
        tc : torch.Tensor
            Fault clear time
        use_jvp : bool
            Whether to use jvp for derivative computation
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        tuple : (delta_pred, omega_pred, ddelta_dt, domega_dt)
        """
        # Ensure t requires grad
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        # Get predictions
        delta_pred, omega_pred = self.predict_trajectory(
            t=t,
            delta0=delta0,
            omega0=omega0,
            H=H,
            D=D,
            Pm=Pm,
            Xprefault=Xprefault,
            Xfault=Xfault,
            Xpostfault=Xpostfault,
            tf=tf,
            tc=tc,
            **kwargs,
        )

        # Compute derivatives
        if use_jvp:
            # Use jvp for efficiency
            (delta_dt,) = torch.autograd.functional.jvp(
                func=lambda t_in: self.predict_trajectory(
                    t=t_in,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    Pm=Pm,
                    Xprefault=Xprefault,
                    Xfault=Xfault,
                    Xpostfault=Xpostfault,
                    tf=tf,
                    tc=tc,
                    **kwargs,
                )[0],
                inputs=t,
                v=torch.ones_like(t),
                create_graph=True,
            )
            (omega_dt,) = torch.autograd.functional.jvp(
                func=lambda t_in: self.predict_trajectory(
                    t=t_in,
                    delta0=delta0,
                    omega0=omega0,
                    H=H,
                    D=D,
                    Pm=Pm,
                    Xprefault=Xprefault,
                    Xfault=Xfault,
                    Xpostfault=Xpostfault,
                    tf=tf,
                    tc=tc,
                    **kwargs,
                )[1],
                inputs=t,
                v=torch.ones_like(t),
                create_graph=True,
            )
        else:
            # Use standard autograd
            delta_dt = torch.autograd.grad(
                outputs=delta_pred,
                inputs=t,
                grad_outputs=torch.ones_like(delta_pred),
                create_graph=True,
                retain_graph=True,
            )[0]
            omega_dt = torch.autograd.grad(
                outputs=omega_pred,
                inputs=t,
                grad_outputs=torch.ones_like(omega_pred),
                create_graph=True,
                retain_graph=True,
            )[0]

        return delta_pred, omega_pred, delta_dt, omega_dt
