"""
Detailed Generator Model Support for PINN.

This module extends the PINN framework to support detailed generator models
(GENROU) with exciter and governor dynamics.

Phase 2: Complete SMIB Extension
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class DetailedGeneratorPhysicsLoss(nn.Module):
    """
    Physics-informed loss for detailed generator models.

    Extends the swing equation to include:
    - Field circuit dynamics
    - Exciter/AVR dynamics
    - Governor/turbine dynamics
    - Multi-time-scale interactions
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_ic: float = 10.0,
        lambda_boundary: float = 1.0,
        lambda_field: float = 0.1,
        lambda_exciter: float = 0.1,
        lambda_governor: float = 0.1,
        fn: float = 60.0,
    ):
        """
        Initialize detailed generator physics loss.

        Parameters:
        -----------
        lambda_data : float
            Weight for data loss
        lambda_physics : float
            Weight for swing equation physics loss
        lambda_ic : float
            Weight for initial condition loss
        lambda_boundary : float
            Weight for boundary condition loss
        lambda_field : float
            Weight for field circuit dynamics loss
        lambda_exciter : float
            Weight for exciter dynamics loss
        lambda_governor : float
            Weight for governor dynamics loss
        fn : float
            System frequency (Hz)
        """
        super(DetailedGeneratorPhysicsLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_ic = lambda_ic
        self.lambda_boundary = lambda_boundary
        self.lambda_field = lambda_field
        self.lambda_exciter = lambda_exciter
        self.lambda_governor = lambda_governor
        self.fn = fn
        self.two_pi_fn = 2.0 * np.pi * fn

    def compute_swing_equation_residual(
        self,
        t: torch.Tensor,
        delta: torch.Tensor,
        omega: torch.Tensor,
        M: torch.Tensor,
        D: torch.Tensor,
        Pm: torch.Tensor,
        Pe: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute swing equation residual.

        M·d²δ/dt² + D·dδ/dt = Pm - Pe
        """
        # Compute derivatives.
        ddelta_dt = torch.autograd.grad(
            outputs=delta,
            inputs=t,
            grad_outputs=torch.ones_like(delta),
            create_graph=True,
            retain_graph=True,
        )[0]

        d2delta_dt2 = torch.autograd.grad(
            outputs=ddelta_dt,
            inputs=t,
            grad_outputs=torch.ones_like(ddelta_dt),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Swing equation residual
        residual = M * d2delta_dt2 + D * ddelta_dt - (Pm - Pe)

        return residual

    def compute_field_dynamics_residual(
        self,
        t: torch.Tensor,
        eq: torch.Tensor,
        ed: torch.Tensor,
        id: torch.Tensor,
        iq: torch.Tensor,
        xd: float,
        xq: float,
        xd_prime: float,
        xq_prime: float,
        Td0_prime: float,
        Tq0_prime: float,
    ) -> torch.Tensor:
        """
        Compute field circuit dynamics residual.

        For GENROU model:
        dE'q/dt = (Efd - E'q - (xd - x'd)·id) / T'd0
        dE'd/dt = (-E'd - (xq - x'q)·iq) / T'q0
        """
        # Compute derivatives.
        deq_dt = torch.autograd.grad(
            outputs=eq,
            inputs=t,
            grad_outputs=torch.ones_like(eq),
            create_graph=True,
            retain_graph=True,
        )[0]

        ded_dt = torch.autograd.grad(
            outputs=ed,
            inputs=t,
            grad_outputs=torch.ones_like(ed),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Field dynamics residuals
        # Note: Efd would come from exciter model
        # For now, placeholder
        residual_q = deq_dt - (0.0 - eq - (xd - xd_prime) * id) / Td0_prime
        residual_d = ded_dt - (-ed - (xq - xq_prime) * iq) / Tq0_prime

        return residual_q, residual_d

    def compute_exciter_dynamics_residual(
        self,
        t: torch.Tensor,
        Efd: torch.Tensor,
        Vt: torch.Tensor,
        Vref: float,
        Ka: float,
        Ta: float,
        Kf: float,
        Tf: float,
    ) -> torch.Tensor:
        """
        Compute exciter/AVR dynamics residual.

        For EXST1 or IEEET1 exciter:
        dEfd/dt = (Ka·(Vref - Vt) - Efd) / Ta
        """
        # Compute derivative.
        dEfd_dt = torch.autograd.grad(
            outputs=Efd,
            inputs=t,
            grad_outputs=torch.ones_like(Efd),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Exciter dynamics residual
        residual = dEfd_dt - (Ka * (Vref - Vt) - Efd) / Ta

        return residual

    def compute_governor_dynamics_residual(
        self,
        t: torch.Tensor,
        Pm: torch.Tensor,
        omega: torch.Tensor,
        Pref: float,
        R: float,
        T1: float,
        T2: float,
        T3: float,
    ) -> torch.Tensor:
        """
        Compute governor/turbine dynamics residual.

        For TGOV1 governor:
        dPm/dt = (Pref - (omega - 1)/R - Pm) / T1
        """
        # Compute derivative.
        dPm_dt = torch.autograd.grad(
            outputs=Pm,
            inputs=t,
            grad_outputs=torch.ones_like(Pm),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Governor dynamics residual
        residual = dPm_dt - (Pref - (omega - 1.0) / R - Pm) / T1

        return residual

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
        # Detailed model variables
        eq_colloc: Optional[torch.Tensor] = None,
        ed_colloc: Optional[torch.Tensor] = None,
        Efd_colloc: Optional[torch.Tensor] = None,
        Pm_colloc: Optional[torch.Tensor] = None,
        # System parameters
        M: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        Pm: Optional[torch.Tensor] = None,
        Pe: Optional[torch.Tensor] = None,
        # Generator parameters
        xd: float = 1.8,
        xq: float = 1.7,
        xd_prime: float = 0.3,
        xq_prime: float = 0.55,
        Td0_prime: float = 8.0,
        Tq0_prime: float = 0.4,
        # Exciter parameters
        Ka: float = 200.0,
        Ta: float = 0.02,
        Vref: float = 1.0,
        # Governor parameters
        R: float = 0.05,
        T1: float = 0.3,
        Pref: float = 0.8,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for detailed generator model.

        Returns:
        --------
        losses : dict
            Dictionary of loss components
        """
        losses = {}

        # Data loss (same as simplified model)
        if delta_obs is not None and omega_obs is not None:
            loss_data = torch.mean((delta_pred - delta_obs) ** 2) + torch.mean(
                (omega_pred - omega_obs) ** 2
            )
            losses["data"] = loss_data
        else:
            losses["data"] = torch.tensor(0.0, device=delta_pred.device)

        # Swing equation physics loss
        if (
            t_colloc is not None
            and delta_colloc is not None
            and omega_colloc is not None
            and M is not None
            and D is not None
            and Pm is not None
            and Pe is not None
        ):
            swing_residual = self.compute_swing_equation_residual(
                t_colloc, delta_colloc, omega_colloc, M, D, Pm, Pe
            )
            loss_physics = torch.mean(swing_residual**2)
            losses["physics"] = loss_physics
        else:
            losses["physics"] = torch.tensor(0.0, device=delta_pred.device)

        # Field dynamics loss
        if eq_colloc is not None and ed_colloc is not None and t_colloc is not None:
            # Placeholder - would need id, iq from network solution
            # For now, skip
            losses["field"] = torch.tensor(0.0, device=delta_pred.device)
        else:
            losses["field"] = torch.tensor(0.0, device=delta_pred.device)

        # Exciter dynamics loss
        if Efd_colloc is not None and t_colloc is not None:
            # Placeholder - would need Vt from network solution
            # For now, skip
            losses["exciter"] = torch.tensor(0.0, device=delta_pred.device)
        else:
            losses["exciter"] = torch.tensor(0.0, device=delta_pred.device)

        # Governor dynamics loss
        if Pm_colloc is not None and omega_colloc is not None and t_colloc is not None:
            governor_residual = self.compute_governor_dynamics_residual(
                t_colloc, Pm_colloc, omega_colloc, Pref, R, T1, 0.0, 0.0
            )
            loss_governor = torch.mean(governor_residual**2)
            losses["governor"] = loss_governor
        else:
            losses["governor"] = torch.tensor(0.0, device=delta_pred.device)

        # Initial condition loss
        if len(delta_pred) > 0:
            # Placeholder
            losses["ic"] = torch.tensor(0.0, device=delta_pred.device)
        else:
            losses["ic"] = torch.tensor(0.0, device=delta_pred.device)

        # Boundary condition loss
        losses["boundary"] = torch.tensor(0.0, device=delta_pred.device)

        # Total loss
        total_loss = (
            self.lambda_data * losses["data"]
            + self.lambda_physics * losses["physics"]
            + self.lambda_ic * losses["ic"]
            + self.lambda_boundary * losses["boundary"]
            + self.lambda_field * losses["field"]
            + self.lambda_exciter * losses["exciter"]
            + self.lambda_governor * losses["governor"]
        )

        losses["total"] = total_loss

        return losses


class DetailedTrajectoryPredictionPINN(nn.Module):
    """
    PINN for trajectory prediction with detailed generator models.

    Extends TrajectoryPredictionPINN to support:
    - GENROU generator model
    - Exciter/AVR dynamics
    - Governor/turbine dynamics
    """

    def __init__(
        self,
        input_dim: int = 15,  # Extended input dimension
        hidden_dims: list = [64, 64, 64, 64],
        activation: str = "tanh",
        use_residual: bool = False,
        dropout: float = 0.0,
        output_dim: int = 6,  # delta, omega, eq, ed, Efd, Pm
    ):
        """
        Initialize detailed trajectory prediction PINN.

        Parameters:
        -----------
        input_dim : int
            Input dimension (extended for detailed model parameters)
        hidden_dims : list
            Hidden layer dimensions
        activation : str
            Activation function
        use_residual : bool
            Whether to use residual connections
        dropout : float
            Dropout rate
        output_dim : int
            Output dimension (delta, omega, eq, ed, Efd, Pm)
        """
        super(DetailedTrajectoryPredictionPINN, self).__init__()

        # Build network (similar to base PINN)
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

        # Physics loss
        self.loss_fn = DetailedGeneratorPhysicsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def predict_trajectory(
        self,
        t: torch.Tensor,
        delta0: torch.Tensor,
        omega0: torch.Tensor,
        eq0: torch.Tensor,
        ed0: torch.Tensor,
        Efd0: torch.Tensor,
        Pm0: torch.Tensor,
        # System parameters
        M: torch.Tensor,
        D: torch.Tensor,
        # Generator parameters
        xd: torch.Tensor,
        xq: torch.Tensor,
        xd_prime: torch.Tensor,
        xq_prime: torch.Tensor,
        Td0_prime: torch.Tensor,
        Tq0_prime: torch.Tensor,
        # Network parameters
        Xprefault: torch.Tensor,
        Xfault: torch.Tensor,
        Xpostfault: torch.Tensor,
        tf: torch.Tensor,
        tc: torch.Tensor,
        # Exciter parameters
        Ka: torch.Tensor,
        Ta: torch.Tensor,
        Vref: torch.Tensor,
        # Governor parameters
        R: torch.Tensor,
        T1: torch.Tensor,
        Pref: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Predict trajectory with detailed generator model.

        Returns:
        --------
        tuple : (delta, omega, eq, ed, Efd, Pm)
        """
        # Prepare input.
        batch_size = t.shape[0] if len(t.shape) > 0 else 1

        # Concatenate all inputs
        inputs = torch.cat(
            [
                t.unsqueeze(-1) if len(t.shape) == 1 else t,
                delta0.unsqueeze(-1) if len(delta0.shape) == 0 else delta0,
                omega0.unsqueeze(-1) if len(omega0.shape) == 0 else omega0,
                eq0.unsqueeze(-1) if len(eq0.shape) == 0 else eq0,
                ed0.unsqueeze(-1) if len(ed0.shape) == 0 else ed0,
                Efd0.unsqueeze(-1) if len(Efd0.shape) == 0 else Efd0,
                Pm0.unsqueeze(-1) if len(Pm0.shape) == 0 else Pm0,
                M.unsqueeze(-1) if len(M.shape) == 0 else M,
                D.unsqueeze(-1) if len(D.shape) == 0 else D,
                xd.unsqueeze(-1) if len(xd.shape) == 0 else xd,
                xq.unsqueeze(-1) if len(xq.shape) == 0 else xq,
                xd_prime.unsqueeze(-1) if len(xd_prime.shape) == 0 else xd_prime,
                xq_prime.unsqueeze(-1) if len(xq_prime.shape) == 0 else xq_prime,
                Td0_prime.unsqueeze(-1) if len(Td0_prime.shape) == 0 else Td0_prime,
                Tq0_prime.unsqueeze(-1) if len(Tq0_prime.shape) == 0 else Tq0_prime,
                Xprefault.unsqueeze(-1) if len(Xprefault.shape) == 0 else Xprefault,
                Xfault.unsqueeze(-1) if len(Xfault.shape) == 0 else Xfault,
                Xpostfault.unsqueeze(-1) if len(Xpostfault.shape) == 0 else Xpostfault,
                tf.unsqueeze(-1) if len(tf.shape) == 0 else tf,
                tc.unsqueeze(-1) if len(tc.shape) == 0 else tc,
                Ka.unsqueeze(-1) if len(Ka.shape) == 0 else Ka,
                Ta.unsqueeze(-1) if len(Ta.shape) == 0 else Ta,
                Vref.unsqueeze(-1) if len(Vref.shape) == 0 else Vref,
                R.unsqueeze(-1) if len(R.shape) == 0 else R,
                T1.unsqueeze(-1) if len(T1.shape) == 0 else T1,
                Pref.unsqueeze(-1) if len(Pref.shape) == 0 else Pref,
            ],
            dim=-1,
        )

        # Predict
        output = self.forward(inputs)

        # Split output
        delta = output[:, 0]
        omega = output[:, 1]
        eq = output[:, 2]
        ed = output[:, 3]
        Efd = output[:, 4]
        Pm = output[:, 5]

        return delta, omega, eq, ed, Efd, Pm
