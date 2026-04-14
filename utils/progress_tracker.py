"""
Progress Tracking Utilities for Data Generation and Training.

This module provides comprehensive progress tracking with detailed statistics
and ETAs for both data generation and model training processes.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def format_percentage(value: float, total: float) -> str:
    """Format percentage with proper handling of zero division."""
    if total == 0:
        return "0.0%"
    return f"{100.0 * value / total:.1f}%"


def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Create a text-based progress bar (ASCII-safe for Windows cp949 and other encodings)."""
    if total == 0:
        return "#" * width
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return bar


class DataGenerationTracker:
    """
    Tracks progress during data generation with power system and ML metrics.

    Attributes:
        total_samples: Total number of samples to generate
        completed_samples: Number of samples completed
        stable_count: Number of stable samples
        unstable_count: Number of unstable samples
        cct_values: List of CCT values found
        cct_uncertainties: List of CCT uncertainties
        max_angles: List of maximum rotor angles (degrees)
        max_freq_deviations: List of maximum frequency deviations (Hz)
        h_values: List of H (inertia) values
        d_values: List of D (damping) values
        pm_values: List of Pm (mechanical power) values
        failed_cct_searches: Number of failed CCT searches
        failed_simulations: Number of failed simulations
        start_time: Start time of data generation
        last_update_time: Time of last update
    """

    def __init__(self, total_samples: int, param_ranges: Optional[Dict] = None):
        """
        Initialize data generation tracker.

        Parameters:
        -----------
        total_samples : int
            Total number of samples to generate
        param_ranges : dict, optional
            Parameter ranges dictionary with keys like 'H', 'D', 'Pm'
        """
        self.total_samples = total_samples
        self.completed_samples = 0
        self.stable_count = 0
        self.unstable_count = 0

        # Power system metrics
        self.cct_values: List[float] = []
        self.cct_uncertainties: List[float] = []
        self.max_angles: List[float] = []  # degrees
        self.max_freq_deviations: List[float] = []  # Hz
        self.first_swing_times: List[float] = []

        # Parameter tracking
        self.h_values: List[float] = []
        self.d_values: List[float] = []
        self.pm_values: List[float] = []

        # Quality metrics
        self.failed_cct_searches = 0
        self.failed_simulations = 0
        self.convergence_warnings: List[str] = []

        # Timing
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None

        # Parameter ranges for coverage analysis
        self.param_ranges = param_ranges or {}

    def start(self):
        """Start tracking."""
        self.start_time = time.time()
        self.last_update_time = self.start_time

    def update(
        self,
        is_stable: bool,
        cct: Optional[float] = None,
        cct_uncertainty: Optional[float] = None,
        max_angle: Optional[float] = None,
        max_freq_dev: Optional[float] = None,
        first_swing_time: Optional[float] = None,
        H: Optional[float] = None,
        D: Optional[float] = None,
        Pm: Optional[float] = None,
        cct_found: bool = True,
        simulation_success: bool = True,
    ):
        """
        Update tracker with new sample information.

        Parameters:
        -----------
        is_stable : bool
            Whether the sample is stable
        cct : float, optional
            Critical clearing time (seconds)
        cct_uncertainty : float, optional
            CCT uncertainty (seconds)
        max_angle : float, optional
            Maximum rotor angle (degrees)
        max_freq_dev : float, optional
            Maximum frequency deviation (Hz)
        first_swing_time : float, optional
            Time to first swing peak (seconds)
        H : float, optional
            Inertia constant (seconds)
        D : float, optional
            Damping coefficient (pu)
        Pm : float, optional
            Mechanical power (pu)
        cct_found : bool
            Whether CCT was successfully found
        simulation_success : bool
            Whether simulation was successful
        """
        self.completed_samples += 1

        if is_stable:
            self.stable_count += 1
        else:
            self.unstable_count += 1

        if cct is not None:
            self.cct_values.append(cct)
        if cct_uncertainty is not None:
            self.cct_uncertainties.append(cct_uncertainty)
        if max_angle is not None:
            self.max_angles.append(max_angle)
        if max_freq_dev is not None:
            self.max_freq_deviations.append(max_freq_dev)
        if first_swing_time is not None:
            self.first_swing_times.append(first_swing_time)

        if H is not None:
            self.h_values.append(H)
        if D is not None:
            self.d_values.append(D)
        if Pm is not None:
            self.pm_values.append(Pm)

        if not cct_found:
            self.failed_cct_searches += 1
        if not simulation_success:
            self.failed_simulations += 1

        self.last_update_time = time.time()

    def get_progress_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_samples == 0:
            return 100.0
        return 100.0 * self.completed_samples / self.total_samples

    def get_eta(self) -> Optional[float]:
        """Calculate estimated time to completion."""
        if self.start_time is None or self.completed_samples == 0:
            return None

        elapsed = time.time() - self.start_time
        if self.completed_samples == 0:
            return None

        avg_time_per_sample = elapsed / self.completed_samples
        remaining_samples = self.total_samples - self.completed_samples
        eta_seconds = avg_time_per_sample * remaining_samples

        return eta_seconds

    def get_cct_statistics(self) -> Dict:
        """Get CCT distribution statistics."""
        if not self.cct_values:
            return {}

        cct_array = np.array(self.cct_values)
        stats = {
            "mean": float(np.mean(cct_array)),
            "std": float(np.std(cct_array)),
            "min": float(np.min(cct_array)),
            "max": float(np.max(cct_array)),
            "count": len(cct_array),
        }

        if self.cct_uncertainties:
            unc_array = np.array(self.cct_uncertainties)
            stats["mean_uncertainty"] = float(np.mean(unc_array))
            stats["max_uncertainty"] = float(np.max(unc_array))

        return stats

    def get_stability_statistics(self) -> Dict:
        """Get stability metrics statistics."""
        stats = {
            "stable_count": self.stable_count,
            "unstable_count": self.unstable_count,
            "total": self.completed_samples,
            "stable_ratio": (
                self.stable_count / self.completed_samples if self.completed_samples > 0 else 0.0
            ),
        }

        if self.max_angles:
            angle_array = np.array(self.max_angles)
            stats["max_angle_mean"] = float(np.mean(angle_array))
            stats["max_angle_max"] = float(np.max(angle_array))
            stats["max_angle_min"] = float(np.min(angle_array))

        if self.max_freq_deviations:
            freq_array = np.array(self.max_freq_deviations)
            stats["max_freq_dev_mean"] = float(np.mean(freq_array))
            stats["max_freq_dev_max"] = float(np.max(freq_array))
            stats["max_freq_dev_min"] = float(np.min(freq_array))

        return stats

    def get_parameter_coverage(self) -> Dict:
        """Get parameter space coverage statistics."""
        coverage = {}

        if self.h_values:
            h_array = np.array(self.h_values)
            coverage["H"] = {
                "min": float(np.min(h_array)),
                "max": float(np.max(h_array)),
                "mean": float(np.mean(h_array)),
                "std": float(np.std(h_array)),
            }
            if "H" in self.param_ranges:
                req_min, req_max = self.param_ranges["H"][:2]
                coverage["H"]["requested"] = (req_min, req_max)
                coverage["H"]["coverage_ok"] = (
                    np.min(h_array) <= req_min + 0.1 and np.max(h_array) >= req_max - 0.1
                )

        if self.d_values:
            d_array = np.array(self.d_values)
            coverage["D"] = {
                "min": float(np.min(d_array)),
                "max": float(np.max(d_array)),
                "mean": float(np.mean(d_array)),
                "std": float(np.std(d_array)),
            }
            if "D" in self.param_ranges:
                req_min, req_max = self.param_ranges["D"][:2]
                coverage["D"]["requested"] = (req_min, req_max)
                coverage["D"]["coverage_ok"] = (
                    np.min(d_array) <= req_min + 0.1 and np.max(d_array) >= req_max - 0.1
                )

        if self.h_values and self.d_values:
            # Calculate H-D correlation
            h_array = np.array(self.h_values)
            d_array = np.array(self.d_values)
            if len(h_array) == len(d_array) and len(h_array) > 1:
                correlation = np.corrcoef(h_array, d_array)[0, 1]
                coverage["H_D_correlation"] = float(correlation)

        return coverage

    def get_quality_warnings(self) -> List[str]:
        """Get quality warnings based on current statistics."""
        warnings = []

        # Class imbalance warning
        if self.completed_samples > 0:
            stable_ratio = self.stable_count / self.completed_samples
            if stable_ratio < 0.3:
                warnings.append(
                    f"⚠️  Class imbalance: Only {stable_ratio*100:.1f}% "
                    f"stable samples (target: 30-70%)"
                )
            elif stable_ratio > 0.7:
                warnings.append(
                    f"⚠️  Class imbalance: {stable_ratio*100:.1f}% stable samples (target: 30-70%)"
                )

        # Failed CCT searches warning
        if self.completed_samples > 0:
            fail_rate = self.failed_cct_searches / self.completed_samples
            if fail_rate > 0.1:
                warnings.append(
                    f"⚠️  High CCT failure rate: {fail_rate*100:.1f}% "
                    f"({self.failed_cct_searches}/{self.completed_samples})"
                )

        # Failed simulations warning
        if self.completed_samples > 0:
            fail_rate = self.failed_simulations / self.completed_samples
            if fail_rate > 0.05:
                warnings.append(
                    f"⚠️  High simulation failure rate: {fail_rate*100:.1f}% "
                    f"({self.failed_simulations}/{self.completed_samples})"
                )

        # Max angle warning
        if self.max_angles:
            max_angle = np.max(self.max_angles)
            if max_angle > 180:
                warnings.append(
                    f"⚠️  Some samples exceed 180° rotor angle "
                    f"(max: {max_angle:.1f}°) - loss of synchronism detected"
                )

        # Max frequency deviation warning
        if self.max_freq_deviations:
            max_freq = np.max(self.max_freq_deviations)
            if max_freq > 0.5:
                warnings.append(
                    f"⚠️  High frequency deviation detected: {max_freq:.2f} Hz "
                    f"(typical limit: 0.5 Hz)"
                )

        return warnings

    def display_progress(self, current_params: Optional[Dict] = None) -> str:
        """
        Generate formatted progress display string.

        Parameters:
        -----------
        current_params : dict, optional
            Current parameters being processed (H, D, Pm)

        Returns:
        --------
        str : Formatted progress string
        """
        lines = []

        # Progress bar and percentage
        progress_pct = self.get_progress_percentage()
        progress_bar = create_progress_bar(self.completed_samples, self.total_samples)
        lines.append(
            f"Progress: {progress_bar} {progress_pct:.1f}%"
            f"({self.completed_samples}/{self.total_samples})"
        )

        # Current parameters
        if current_params:
            param_str = ", ".join(
                [f"{k}={v:.3f}" for k, v in current_params.items() if v is not None]
            )
            if param_str:
                lines.append(f"Current: {param_str}")

        # CCT statistics
        if self.cct_values:
            cct_stats = self.get_cct_statistics()
            current_cct = self.cct_values[-1] if self.cct_values else None
            if current_cct is not None:
                cct_unc = (
                    self.cct_uncertainties[-1]
                    if self.cct_uncertainties
                    and len(self.cct_uncertainties) == len(self.cct_values)
                    else None
                )
                if cct_unc is not None:
                    lines.append(
                        f"CCT: {current_cct:.3f}s ± {cct_unc:.3f}s | Mean: {cct_stats['mean']:.3f}s"
                        f"[{cct_stats['min']:.3f}s, {cct_stats['max']:.3f}s]"
                    )
                else:
                    lines.append(
                        f"CCT: {current_cct:.3f}s | Mean: {cct_stats['mean']:.3f}s"
                        f"[{cct_stats['min']:.3f}s, {cct_stats['max']:.3f}s]"
                    )

        # Stability statistics
        stable_ratio_pct = (
            format_percentage(self.stable_count, self.completed_samples)
            if self.completed_samples > 0
            else "0%"
        )
        lines.append(
            f"Stability: Stable: {self.stable_count} | Unstable: {self.unstable_count} | Ratio:"
            f"{stable_ratio_pct}"
        )

        # Power system metrics
        if self.max_angles:
            current_angle = self.max_angles[-1]
            mean_angle = np.mean(self.max_angles)
            lines.append(f"Max Angle: {current_angle:.1f}° | Mean: {mean_angle:.1f}°")

        if self.max_freq_deviations:
            current_freq = self.max_freq_deviations[-1]
            mean_freq = np.mean(self.max_freq_deviations)
            lines.append(f"Max Freq Dev: {current_freq:.2f} Hz | Mean: {mean_freq:.2f} Hz")

        # Timing
        if self.start_time:
            elapsed = time.time() - self.start_time
            eta = self.get_eta()
            eta_str = format_time(eta) if eta else "calculating..."
            lines.append(f"Elapsed: {format_time(elapsed)} | ETA: {eta_str}")

        # Quality warnings
        warnings = self.get_quality_warnings()
        if warnings:
            lines.append("")
            lines.extend(warnings)
        else:
            lines.append("")
            lines.append("Quality Warnings: None ✓")

        return "\n".join(lines)

    def display_summary(self) -> str:
        """Generate comprehensive final summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("Data Generation Summary")
        lines.append("=" * 70)

        # Timing
        if self.start_time:
            total_time = time.time() - self.start_time
            lines.append(f"Total Time: {format_time(total_time)}")

        lines.append(f"Total Samples: {self.completed_samples}/{self.total_samples}")

        # CCT statistics
        if self.cct_values:
            cct_stats = self.get_cct_statistics()
            lines.append("")
            lines.append("CCT Statistics:")
            lines.append(f"  Mean: {cct_stats['mean']:.3f}s ± {cct_stats['std']:.3f}s")
            lines.append(f"  Range: [{cct_stats['min']:.3f}s, {cct_stats['max']:.3f}s]")
            if "mean_uncertainty" in cct_stats:
                lines.append(f"  Mean Uncertainty: {cct_stats['mean_uncertainty']:.3f}s")
            lines.append(
                f"Failed Searches: {self.failed_cct_searches}/{self.completed_samples}"
                f"({format_percentage(self.failed_cct_searches, self.completed_samples)})"
            )

        # Stability statistics
        stats = self.get_stability_statistics()
        lines.append("")
        lines.append("Stability Statistics:")
        stable_pct = format_percentage(stats["stable_count"], stats["total"])
        unstable_pct = format_percentage(stats["unstable_count"], stats["total"])
        lines.append(f"Stable: {stats['stable_count']} ({stable_pct})")
        lines.append(f"Unstable: {stats['unstable_count']} ({unstable_pct})")
        if "max_angle_mean" in stats:
            lines.append(
                f"  Max Angle: Mean={stats['max_angle_mean']:.1f}°, Max={stats['max_angle_max']:.1f}°"
            )
        if "max_freq_dev_mean" in stats:
            lines.append(
                f"Max Freq Dev: Mean={stats['max_freq_dev_mean']:.2f} Hz,"
                f"Max={stats['max_freq_dev_max']:.2f} Hz"
            )

        # Parameter coverage
        coverage = self.get_parameter_coverage()
        if coverage:
            lines.append("")
            lines.append("Parameter Coverage:")
            for param, info in coverage.items():
                if param != "H_D_correlation":
                    if "requested" in info:
                        req_min, req_max = info["requested"]
                        status = "✓" if info.get("coverage_ok", False) else "⚠️"
                        lines.append(
                            f"{param}: [{info['min']:.2f}, {info['max']:.2f}] (requested:"
                            f"[{req_min:.2f}, {req_max:.2f}]) {status}"
                        )
                    else:
                        lines.append(f"  {param}: [{info['min']:.2f}, {info['max']:.2f}]")
            if "H_D_correlation" in coverage:
                corr = coverage["H_D_correlation"]
                status = "✓" if abs(corr) < 0.3 else "⚠️"
                lines.append(f"  H-D Correlation: {corr:.3f} {status}")

        # Quality warnings
        warnings = self.get_quality_warnings()
        if warnings:
            lines.append("")
            lines.append("Quality Warnings:")
            for warning in warnings:
                lines.append(f"  {warning}")

        lines.append("=" * 70)

        return "\n".join(lines)


class TrainingTracker:
    """
    Tracks progress during model training with PINN-specific metrics.

    Attributes:
        total_epochs: Total number of training epochs
        current_epoch: Current epoch number
        train_losses: List of training losses
        val_losses: List of validation losses
        data_losses: List of data loss components
        physics_losses: List of physics loss components
        ic_losses: List of initial condition loss components
        best_val_loss: Best validation loss so far
        best_epoch: Epoch with best validation loss
        learning_rates: List of learning rates
        early_stopping_counter: Early stopping patience counter
        start_time: Start time of training
    """

    def __init__(self, total_epochs: int, initial_lr: float = 0.001):
        """
        Initialize training tracker.

        Parameters:
        -----------
        total_epochs : int
            Total number of training epochs
        initial_lr : float
            Initial learning rate
        """
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Loss tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.data_losses: List[float] = []
        self.physics_losses: List[float] = []
        self.ic_losses: List[float] = []

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Learning rate tracking
        self.learning_rates: List[float] = [initial_lr]
        self.current_lr = initial_lr

        # Early stopping
        self.early_stopping_counter = 0
        self.early_stopping_patience = 20

        # Timing
        self.start_time: Optional[float] = None

    def start(self):
        """Start tracking."""
        self.start_time = time.time()

    def update_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        data_loss: Optional[float] = None,
        physics_loss: Optional[float] = None,
        ic_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        """
        Update tracker with epoch results.

        Parameters:
        -----------
        epoch : int
            Current epoch number (0-indexed)
        train_loss : float
            Training loss
        val_loss : float
            Validation loss
        data_loss : float, optional
            Data loss component
        physics_loss : float, optional
            Physics loss component
        ic_loss : float, optional
            Initial condition loss component
        learning_rate : float, optional
            Current learning rate
        """
        self.current_epoch = epoch + 1
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if data_loss is not None:
            self.data_losses.append(data_loss)
        if physics_loss is not None:
            self.physics_losses.append(physics_loss)
        if ic_loss is not None:
            self.ic_losses.append(ic_loss)

        if learning_rate is not None:
            self.current_lr = learning_rate
            self.learning_rates.append(learning_rate)

        # Update best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = self.current_epoch
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

    def get_progress_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_epochs == 0:
            return 100.0
        return 100.0 * self.current_epoch / self.total_epochs

    def get_eta(self) -> Optional[float]:
        """Calculate estimated time to completion."""
        if self.start_time is None or self.current_epoch == 0:
            return None

        elapsed = time.time() - self.start_time
        if self.current_epoch == 0:
            return None

        avg_time_per_epoch = elapsed / self.current_epoch
        remaining_epochs = self.total_epochs - self.current_epoch
        eta_seconds = avg_time_per_epoch * remaining_epochs

        return eta_seconds

    def get_loss_ratios(self) -> Dict[str, float]:
        """Get loss component ratios."""
        if not self.train_losses:
            return {}

        ratios = {}

        if self.data_losses and self.physics_losses and self.ic_losses:
            # Use most recent losses
            data = self.data_losses[-1]
            physics = self.physics_losses[-1]
            ic = self.ic_losses[-1]
            total = data + physics + ic

            if total > 0:
                ratios["data"] = data / total
                ratios["physics"] = physics / total
                ratios["ic"] = ic / total

        return ratios

    def get_train_val_gap(self) -> Optional[float]:
        """Get training-validation loss gap."""
        if not self.train_losses or not self.val_losses:
            return None
        return self.val_losses[-1] - self.train_losses[-1]

    def get_quality_warnings(self) -> List[str]:
        """Get quality warnings based on current training state."""
        warnings = []

        # Loss component imbalance
        ratios = self.get_loss_ratios()
        if ratios:
            if ratios.get("physics", 0) > 0.7:
                warnings.append(
                    "⚠️  Physics loss dominates (>70%) - model may not be learning from data"
                )
            elif ratios.get("data", 0) > 0.8:
                warnings.append(
                    "⚠️  Data loss dominates (>80%) - model may be ignoring physics constraints"
                )

        # Overfitting detection
        gap = self.get_train_val_gap()
        if gap is not None and gap > 0.001:
            warnings.append(f"⚠️  Large train-val gap: {gap:.6f} - possible overfitting")

        # Early stopping warning
        if self.early_stopping_counter >= self.early_stopping_patience * 0.8:
            warnings.append(
                f"⚠️ Early stopping approaching:"
                f"{self.early_stopping_counter}/{self.early_stopping_patience}"
            )

        return warnings

    def display_progress(self) -> str:
        """Generate formatted progress display string."""
        lines = []

        # Progress bar and percentage
        progress_pct = self.get_progress_percentage()
        progress_bar = create_progress_bar(self.current_epoch, self.total_epochs)
        lines.append(
            f"Progress: {progress_bar} {progress_pct:.1f}%"
            f"({self.current_epoch}/{self.total_epochs} epochs)"
        )

        # Losses
        if self.train_losses and self.val_losses:
            train_loss = self.train_losses[-1]
            val_loss = self.val_losses[-1]
            best_str = (
                f" (Best: {self.best_val_loss:.6f} @ epoch {self.best_epoch})"
                if self.best_epoch > 0
                else ""
            )
            lines.append(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}{best_str}")

        # Loss components
        if self.data_losses and self.physics_losses and self.ic_losses:
            data = self.data_losses[-1]
            physics = self.physics_losses[-1]
            ic = self.ic_losses[-1]
            total = data + physics + ic
            if total > 0:
                data_pct = 100 * data / total
                physics_pct = 100 * physics / total
                ic_pct = 100 * ic / total
                lines.append(
                    f"└─ Data: {data:.6f} ({data_pct:.0f}%) | Physics: {physics:.6f}"
                    f"({physics_pct:.0f}%) | IC: {ic:.6f} ({ic_pct:.0f}%)"
                )

            # Loss ratios
            ratios = self.get_loss_ratios()
            if ratios:
                physics_ratio = (
                    ratios.get("physics", 0) / ratios.get("data", 1)
                    if ratios.get("data", 0) > 0
                    else 0
                )
                ic_ratio = (
                    ratios.get("ic", 0) / ratios.get("data", 1) if ratios.get("data", 0) > 0 else 0
                )
                lines.append(
                    f"  └─ Ratios: Data:Physics:IC = 1.0:" f"{physics_ratio:.2f}:{ic_ratio:.2f}"
                )

        # Train-val gap
        gap = self.get_train_val_gap()
        if gap is not None:
            status = "✓" if abs(gap) < 0.001 else "⚠️"
            lines.append(f"Train-Val Gap: {gap:.6f} {status}")

        # Learning rate and early stopping
        lines.append(
            f"LR: {self.current_lr:.2e} | Early Stop:"
            f"{self.early_stopping_counter}/{self.early_stopping_patience}"
        )

        # Timing
        if self.start_time:
            elapsed = time.time() - self.start_time
            eta = self.get_eta()
            eta_str = format_time(eta) if eta else "calculating..."
            lines.append(f"Elapsed: {format_time(elapsed)} | ETA: {eta_str}")

        # Quality warnings
        warnings = self.get_quality_warnings()
        if warnings:
            lines.append("")
            lines.extend(warnings)
        else:
            lines.append("")
            lines.append("Quality Warnings: None ✓")

        return "\n".join(lines)

    def display_summary(self) -> str:
        """Generate comprehensive final summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("Training Summary")
        lines.append("=" * 70)

        # Timing
        if self.start_time:
            total_time = time.time() - self.start_time
            lines.append(f"Total Time: {format_time(total_time)}")

        lines.append(f"Total Epochs: {self.current_epoch}/{self.total_epochs}")

        # Final losses
        if self.train_losses and self.val_losses:
            lines.append("")
            lines.append("Final Losses:")
            lines.append(f"  Training: {self.train_losses[-1]:.6f}")
            lines.append(f"  Validation: {self.val_losses[-1]:.6f}")
            lines.append(f"  Best Validation: {self.best_val_loss:.6f} @ epoch {self.best_epoch}")

        # Loss components
        if self.data_losses and self.physics_losses and self.ic_losses:
            lines.append("")
            lines.append("Final Loss Components:")
            lines.append(f"  Data: {self.data_losses[-1]:.6f}")
            lines.append(f"  Physics: {self.physics_losses[-1]:.6f}")
            lines.append(f"  IC: {self.ic_losses[-1]:.6f}")

            ratios = self.get_loss_ratios()
            if ratios:
                lines.append("")
                lines.append("Loss Component Ratios:")
                lines.append(f"  Data: {ratios.get('data', 0)*100:.1f}%")
                lines.append(f"  Physics: {ratios.get('physics', 0)*100:.1f}%")
                lines.append(f"  IC: {ratios.get('ic', 0)*100:.1f}%")

        # Train-val gap
        gap = self.get_train_val_gap()
        if gap is not None:
            lines.append("")
            gap_status = "(acceptable)" if abs(gap) < 0.001 else "(possible overfitting)"
            lines.append(f"Train-Val Gap: {gap:.6f} {gap_status}")

        # Quality warnings
        warnings = self.get_quality_warnings()
        if warnings:
            lines.append("")
            lines.append("Quality Warnings:")
            for warning in warnings:
                lines.append(f"  {warning}")

        lines.append("=" * 70)

        return "\n".join(lines)
