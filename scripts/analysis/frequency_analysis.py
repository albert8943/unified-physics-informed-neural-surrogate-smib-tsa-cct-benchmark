"""
Frequency Analysis for Delta and Omega Predictions.

Analyzes the frequency content of Delta and Omega signals to understand
their different characteristics and explain the trade-off.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy import signal
from scipy.fft import fft, fftfreq


def load_trajectory_data(experiment_dir: Path, scenario_id: Optional[str] = None) -> Optional[Dict]:
    """
    Load trajectory data from an experiment.

    Parameters:
    -----------
    experiment_dir : Path
        Experiment directory path
    scenario_id : str, optional
        Specific scenario ID to load (if None, loads first available)

    Returns:
    --------
    data : dict or None
        Dictionary with 'time', 'delta_true', 'omega_true', 'delta_pred', 'omega_pred'
    """
    # Try to find trajectory data in results directory
    results_dir = experiment_dir / "results"
    if not results_dir.exists():
        return None

    # Look for trajectory CSV files
    traj_files = list(results_dir.glob("*trajectory*.csv"))
    if not traj_files:
        return None

    import pandas as pd

    df = pd.read_csv(traj_files[0])

    # Extract data
    if scenario_id:
        df = df[df.get("scenario_id", "") == scenario_id]
        if len(df) == 0:
            return None

    data = {
        "time": df["time"].values if "time" in df else np.arange(len(df)),
        "delta_true": df["delta_true"].values if "delta_true" in df else None,
        "omega_true": df["omega_true"].values if "omega_true" in df else None,
        "delta_pred": df["delta_pred"].values if "delta_pred" in df else None,
        "omega_pred": df["omega_pred"].values if "omega_pred" in df else None,
    }

    return data


def compute_frequency_spectrum(
    signal_data: np.ndarray, time: np.ndarray, sampling_rate: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency spectrum using FFT.

    Parameters:
    -----------
    signal_data : np.ndarray
        Signal data
    time : np.ndarray
        Time array
    sampling_rate : float, optional
        Sampling rate (Hz). If None, computed from time array.

    Returns:
    --------
    frequencies : np.ndarray
        Frequency array (Hz)
    magnitude : np.ndarray
        Magnitude spectrum
    """
    if sampling_rate is None:
        dt = time[1] - time[0] if len(time) > 1 else 0.01
        sampling_rate = 1.0 / dt

    # Compute FFT
    n = len(signal_data)
    fft_vals = fft(signal_data)
    frequencies = fftfreq(n, 1.0 / sampling_rate)

    # Only return positive frequencies
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    magnitude = np.abs(fft_vals[positive_freq_idx])

    return frequencies, magnitude


def analyze_frequency_content(
    delta_true: np.ndarray,
    omega_true: np.ndarray,
    delta_pred: np.ndarray,
    omega_pred: np.ndarray,
    time: np.ndarray,
) -> Dict:
    """
    Analyze frequency content of Delta and Omega signals.

    Parameters:
    -----------
    delta_true : np.ndarray
        True delta values
    omega_true : np.ndarray
        True omega values
    delta_pred : np.ndarray
        Predicted delta values
    omega_pred : np.ndarray
        Predicted omega values
    time : np.ndarray
        Time array

    Returns:
    --------
    analysis : dict
        Dictionary with frequency analysis results
    """
    # Compute frequency spectra
    freq_delta_true, mag_delta_true = compute_frequency_spectrum(delta_true, time)
    freq_omega_true, mag_omega_true = compute_frequency_spectrum(omega_true, time)
    freq_delta_pred, mag_delta_pred = compute_frequency_spectrum(delta_pred, time)
    freq_omega_pred, mag_omega_pred = compute_frequency_spectrum(omega_pred, time)

    # Find dominant frequencies
    def find_dominant_frequencies(frequencies, magnitude, n_peaks=3):
        """Find top N dominant frequencies."""
        # Find peaks
        peaks, properties = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
        if len(peaks) == 0:
            return [], []

        # Sort by magnitude
        peak_magnitudes = magnitude[peaks]
        sorted_idx = np.argsort(peak_magnitudes)[::-1]
        top_peaks = peaks[sorted_idx[:n_peaks]]

        return frequencies[top_peaks], magnitude[top_peaks]

    delta_dom_freqs, delta_dom_mags = find_dominant_frequencies(freq_delta_true, mag_delta_true)
    omega_dom_freqs, omega_dom_mags = find_dominant_frequencies(freq_omega_true, mag_omega_true)

    # Compute frequency bands
    def compute_band_energy(frequencies, magnitude, freq_min, freq_max):
        """Compute energy in a frequency band."""
        mask = (frequencies >= freq_min) & (frequencies <= freq_max)
        return np.sum(magnitude[mask] ** 2)

    # Define frequency bands
    low_freq_band = (0.1, 2.0)  # Low frequency (0.1-2 Hz) - typical for delta
    high_freq_band = (2.0, 10.0)  # High frequency (2-10 Hz) - typical for omega

    delta_low_energy = compute_band_energy(freq_delta_true, mag_delta_true, *low_freq_band)
    delta_high_energy = compute_band_energy(freq_delta_true, mag_delta_true, *high_freq_band)
    omega_low_energy = compute_band_energy(freq_omega_true, mag_omega_true, *low_freq_band)
    omega_high_energy = compute_band_energy(freq_omega_true, mag_omega_true, *high_freq_band)

    analysis = {
        "delta": {
            "frequencies": freq_delta_true.tolist(),
            "magnitude_true": mag_delta_true.tolist(),
            "magnitude_pred": mag_delta_pred.tolist(),
            "dominant_frequencies": delta_dom_freqs.tolist(),
            "dominant_magnitudes": delta_dom_mags.tolist(),
            "low_freq_energy": float(delta_low_energy),
            "high_freq_energy": float(delta_high_energy),
            "low_freq_ratio": float(
                delta_low_energy / (delta_low_energy + delta_high_energy + 1e-10)
            ),
        },
        "omega": {
            "frequencies": freq_omega_true.tolist(),
            "magnitude_true": mag_omega_true.tolist(),
            "magnitude_pred": mag_omega_pred.tolist(),
            "dominant_frequencies": omega_dom_freqs.tolist(),
            "dominant_magnitudes": omega_dom_mags.tolist(),
            "low_freq_energy": float(omega_low_energy),
            "high_freq_energy": float(omega_high_energy),
            "low_freq_ratio": float(
                omega_low_energy / (omega_low_energy + omega_high_energy + 1e-10)
            ),
        },
    }

    return analysis


def generate_frequency_analysis_plot(
    analysis: Dict, output_path: Path, title: str = "Frequency Content Analysis: Delta vs Omega"
) -> None:
    """
    Generate frequency analysis plots.

    Parameters:
    -----------
    analysis : dict
        Frequency analysis results
    output_path : Path
        Path to save the plot
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Delta frequency spectrum
    ax = axes[0, 0]
    delta_freqs = np.array(analysis["delta"]["frequencies"])
    delta_mag_true = np.array(analysis["delta"]["magnitude_true"])
    delta_mag_pred = np.array(analysis["delta"]["magnitude_pred"])

    ax.plot(delta_freqs, delta_mag_true, "b-", label="True", linewidth=2)
    ax.plot(delta_freqs, delta_mag_pred, "r--", label="Predicted", linewidth=2, alpha=0.7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Delta Frequency Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 10])
    ax.axvspan(0.1, 2.0, alpha=0.2, color="green", label="Low Freq Band")
    ax.axvspan(2.0, 10.0, alpha=0.2, color="orange", label="High Freq Band")

    # Plot 2: Omega frequency spectrum
    ax = axes[0, 1]
    omega_freqs = np.array(analysis["omega"]["frequencies"])
    omega_mag_true = np.array(analysis["omega"]["magnitude_true"])
    omega_mag_pred = np.array(analysis["omega"]["magnitude_pred"])

    ax.plot(omega_freqs, omega_mag_true, "b-", label="True", linewidth=2)
    ax.plot(omega_freqs, omega_mag_pred, "r--", label="Predicted", linewidth=2, alpha=0.7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Omega Frequency Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 10])
    ax.axvspan(0.1, 2.0, alpha=0.2, color="green", label="Low Freq Band")
    ax.axvspan(2.0, 10.0, alpha=0.2, color="orange", label="High Freq Band")

    # Plot 3: Energy comparison
    ax = axes[1, 0]
    categories = ["Delta\nLow Freq", "Delta\nHigh Freq", "Omega\nLow Freq", "Omega\nHigh Freq"]
    energies = [
        analysis["delta"]["low_freq_energy"],
        analysis["delta"]["high_freq_energy"],
        analysis["omega"]["low_freq_energy"],
        analysis["omega"]["high_freq_energy"],
    ]
    colors = ["lightblue", "lightcoral", "lightgreen", "lightyellow"]
    ax.bar(categories, energies, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Energy")
    ax.set_title("Frequency Band Energy Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Low frequency ratio comparison
    ax = axes[1, 1]
    categories = ["Delta", "Omega"]
    ratios = [
        analysis["delta"]["low_freq_ratio"],
        analysis["omega"]["low_freq_ratio"],
    ]
    colors = ["blue", "red"]
    ax.bar(categories, ratios, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Low Frequency Energy Ratio")
    ax.set_title("Low vs High Frequency Energy Distribution")
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="50% threshold")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved frequency analysis plot to: {output_path}")
