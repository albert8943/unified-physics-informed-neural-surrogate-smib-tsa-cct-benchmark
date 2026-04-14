"""
Visualization Utilities for PINN Training and Evaluation.

This module provides plotting functions for trajectories, loss curves, and comparisons.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_trajectories(
    time: np.ndarray,
    delta_pred: np.ndarray,
    omega_pred: np.ndarray,
    delta_true: Optional[np.ndarray] = None,
    omega_true: Optional[np.ndarray] = None,
    tf: Optional[float] = None,
    tc: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot trajectory predictions vs true values.

    Parameters:
    -----------
    time : np.ndarray
        Time array
    delta_pred : np.ndarray
        Predicted rotor angles
    omega_pred : np.ndarray
        Predicted rotor speeds
    delta_true : np.ndarray, optional
        True rotor angles
    omega_true : np.ndarray, optional
        True rotor speeds
    tf : float, optional
        Fault start time
    tc : float, optional
        Fault clear time
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to show figure

    Returns:
    --------
    tuple : (figure, axes)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot delta
    axes[0].plot(time, np.degrees(delta_pred), "b-", label="Predicted", linewidth=2)
    if delta_true is not None:
        axes[0].plot(time, np.degrees(delta_true), "r--", label="True", linewidth=2)

    if tf is not None:
        axes[0].axvline(x=tf, color="g", linestyle=":", label="Fault Start", linewidth=2)
    if tc is not None:
        axes[0].axvline(x=tc, color="orange", linestyle=":", label="Fault Clear", linewidth=2)

    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Rotor Angle (degrees)")
    axes[0].set_title("Rotor Angle Trajectory")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot omega
    axes[1].plot(time, omega_pred, "b-", label="Predicted", linewidth=2)
    if omega_true is not None:
        axes[1].plot(time, omega_true, "r--", label="True", linewidth=2)

    if tf is not None:
        axes[1].axvline(x=tf, color="g", linestyle=":", label="Fault Start", linewidth=2)
    if tc is not None:
        axes[1].axvline(x=tc, color="orange", linestyle=":", label="Fault Clear", linewidth=2)

    axes[1].axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Nominal Speed")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Rotor Speed (pu)")
    axes[1].set_title("Rotor Speed Trajectory")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    loss_components: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot training loss curves.

    Parameters:
    -----------
    train_losses : list
        Training losses per epoch
    val_losses : list, optional
        Validation losses per epoch
    loss_components : dict, optional
        Dictionary of loss components (e.g., {'data': [...], 'physics': [...]})
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to show figure

    Returns:
    --------
    tuple : (figure, axes)
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    axes.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    if val_losses is not None:
        axes.plot(epochs, val_losses, "r--", label="Validation Loss", linewidth=2)

    if loss_components is not None:
        for name, losses in loss_components.items():
            axes.plot(epochs, losses, "--", label="{name} Loss", alpha=0.7)

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_title("Training Loss Curves")
    axes.set_yscale("log")
    axes.legend()
    axes.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


def plot_comparison(
    predictions: Dict,
    targets: Dict,
    task: str = "trajectory",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot comparison between predictions and targets.

    Parameters:
    -----------
    predictions : dict
        Dictionary of predictions
    targets : dict
        Dictionary of targets
    task : str
        Task type: 'trajectory', 'cct', or 'parameter'
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to show figure

    Returns:
    --------
    tuple : (figure, axes)
    """
    if task == "trajectory":
        return plot_trajectories(
            predictions.get("time"),
            predictions.get("delta"),
            predictions.get("omega"),
            targets.get("delta"),
            targets.get("omega"),
            save_path=save_path,
            show=show,
        )
    elif task == "cct":
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        cct_pred = predictions.get("cct", [])
        cct_true = targets.get("cct", [])

        ax.scatter(cct_true, cct_pred, alpha=0.6, s=50)

        # Perfect prediction line
        min_val = min(min(cct_true), min(cct_pred))
        max_val = max(max(cct_true), max(cct_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

        ax.set_xlabel("True CCT (s)")
        ax.set_ylabel("Predicted CCT (s)")
        ax.set_title("CCT Prediction Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax

    elif task == "parameter":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        H_pred = predictions.get("H", [])
        H_true = targets.get("H", [])
        D_pred = predictions.get("D", [])
        D_true = targets.get("D", [])

        # H comparison
        axes[0].scatter(H_true, H_pred, alpha=0.6, s=50)
        min_h = min(min(H_true), min(H_pred))
        max_h = max(max(H_true), max(H_pred))
        axes[0].plot([min_h, max_h], [min_h, max_h], "r--", label="Perfect Prediction")
        axes[0].set_xlabel("True H (s)")
        axes[0].set_ylabel("Predicted H (s)")
        axes[0].set_title("H Parameter Estimation")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # D comparison
        axes[1].scatter(D_true, D_pred, alpha=0.6, s=50)
        min_d = min(min(D_true), min(D_pred))
        max_d = max(max(D_true), max(D_pred))
        axes[1].plot([min_d, max_d], [min_d, max_d], "r--", label="Perfect Prediction")
        axes[1].set_xlabel("True D (pu)")
        axes[1].set_ylabel("Predicted D (pu)")
        axes[1].set_title("D Parameter Estimation")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, axes

    else:
        raise ValueError("Unknown task: {task}")
