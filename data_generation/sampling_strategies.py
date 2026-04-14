"""
Sampling Strategies for PINN Data Generation.

This module provides various sampling strategies for generating diverse
training data for different PINN tasks:
- Latin Hypercube Sampling (LHS) for better coverage
- Sobol sequences for quasi-random sampling
- Boundary-focused sampling for CCT estimation
- Correlation analysis utilities
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import qmc


def latin_hypercube_sample(
    n_samples: int, bounds: List[Tuple[float, float]], seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate samples using Latin Hypercube Sampling (LHS)

    LHS ensures better coverage of the parameter space compared to
    random sampling by dividing each dimension into n_samples intervals
    and ensuring one sample per interval.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    bounds : list of tuples
        List of (min, max) bounds for each dimension
        Example: [(2.0, 10.0), (0.5, 3.0)] for H and D
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Array of shape (n_samples, n_dims) with sampled values
    """
    n_dims = len(bounds)

    # Use scipy's LHS implementation
    if seed is not None:
        np.random.seed(seed)

    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)

    # Scale samples to bounds
    scaled_samples = np.zeros_like(samples)
    for i, (min_val, max_val) in enumerate(bounds):
        scaled_samples[:, i] = samples[:, i] * (max_val - min_val) + min_val

    return scaled_samples


def sobol_sequence_sample(
    n_samples: int, bounds: List[Tuple[float, float]], seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate samples using Sobol sequence (quasi-random sampling)

    Sobol sequences provide low-discrepancy sampling, ensuring better
    coverage of the parameter space with fewer samples.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    bounds : list of tuples
        List of (min, max) bounds for each dimension
    seed : int, optional
        Random seed for reproducibility (Sobol sequences are deterministic)

    Returns:
    --------
    np.ndarray
        Array of shape (n_samples, n_dims) with sampled values
    """
    n_dims = len(bounds)

    # Use scipy's Sobol sequence
    sampler = qmc.Sobol(d=n_dims, seed=seed, scramble=True)
    samples = sampler.random(n=n_samples)

    # Scale samples to bounds
    scaled_samples = np.zeros_like(samples)
    for i, (min_val, max_val) in enumerate(bounds):
        scaled_samples[:, i] = samples[:, i] * (max_val - min_val) + min_val

    return scaled_samples


def boundary_focused_sample(
    n_samples: int,
    bounds: List[Tuple[float, float]],
    boundary_fraction: float = 0.4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate samples with focus on parameter space boundaries.

    This strategy is useful for CCT estimation where stability boundaries
    are critical. It generates a mix of boundary and interior samples.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    bounds : list of tuples
        List of (min, max) bounds for each dimension
    boundary_fraction : float
        Fraction of samples to place near boundaries (0.0 to 1.0)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Array of shape (n_samples, n_dims) with sampled values
    """
    if seed is not None:
        np.random.seed(seed)

    n_dims = len(bounds)
    n_boundary = int(n_samples * boundary_fraction)
    n_interior = n_samples - n_boundary

    samples = np.zeros((n_samples, n_dims))

    # Generate boundary samples (near min or max of each dimension)
    boundary_threshold = 0.1  # 10% of range from boundaries
    for i in range(n_boundary):
        sample = np.zeros(n_dims)
        for j, (min_val, max_val) in enumerate(bounds):
            range_val = max_val - min_val
            # Randomly choose near min or max boundary
            if np.random.random() < 0.5:
                # Near min boundary
                sample[j] = min_val + np.random.uniform(0, boundary_threshold * range_val)
            else:
                # Near max boundary
                sample[j] = max_val - np.random.uniform(0, boundary_threshold * range_val)
        samples[i] = sample

    # Generate interior samples using LHS for better coverage
    if n_interior > 0:
        interior_samples = latin_hypercube_sample(n_interior, bounds, seed=seed)
        samples[n_boundary:] = interior_samples

    # Shuffle to mix boundary and interior samples
    np.random.shuffle(samples)

    return samples


def full_factorial_sample(values_list: List[np.ndarray]) -> np.ndarray:
    """
    Generate full factorial design (all combinations)

    This is the current approach - generates all combinations of
    parameter values. Useful for trajectory prediction.

    Parameters:
    -----------
    values_list : list of arrays
        List of arrays, each containing values for one dimension
        Example: [H_values, D_values, tc_values]

    Returns:
    --------
    np.ndarray
        Array of shape (n_combinations, n_dims) with all combinations
    """
    from itertools import product

    combinations = list(product(*values_list))
    return np.array(combinations)


def correlation_analysis(
    samples: np.ndarray, parameter_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Analyze correlation between parameters in sampled data.

    For parameter estimation tasks, we want low correlation between
    H and D to ensure the model can learn distinct effects.

    Parameters:
    -----------
    samples : np.ndarray
        Array of shape (n_samples, n_dims) with parameter values
    parameter_names : list of str, optional
        Names of parameters for reporting

    Returns:
    --------
    dict
        Dictionary with correlation matrix and summary statistics
    """
    if parameter_names is None:
        parameter_names = ["Param_{i}" for i in range(samples.shape[1])]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(samples.T)

    # Extract pairwise correlations
    n_dims = samples.shape[1]
    correlations = {}

    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            key = "{parameter_names[i]}_{parameter_names[j]}"
            correlations[key] = corr_matrix[i, j]

    # Compute maximum absolute correlation (excluding diagonal)
    max_corr = 0.0
    max_corr_pair = None
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            abs_corr = abs(corr_matrix[i, j])
            if abs_corr > max_corr:
                max_corr = abs_corr
                max_corr_pair = (parameter_names[i], parameter_names[j])

    return {
        "correlation_matrix": corr_matrix,
        "pairwise_correlations": correlations,
        "max_correlation": max_corr,
        "max_correlation_pair": max_corr_pair,
        "parameter_names": parameter_names,
    }


def decorrelated_sample(
    n_samples: int,
    bounds: List[Tuple[float, float]],
    target_correlation: float = 0.0,
    tolerance: float = 0.1,
    max_iterations: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate samples with controlled correlation between parameters.

    For parameter estimation, we want H and D to be decorrelated.
    This function uses iterative adjustment to achieve target correlation.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    bounds : list of tuples
        List of (min, max) bounds for each dimension
    target_correlation : float
        Target correlation coefficient (typically 0.0 for decorrelation)
    tolerance : float
        Acceptable deviation from target correlation
    max_iterations : int
        Maximum number of iterations to achieve target
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (samples, correlation_info) where samples is the array and
        correlation_info contains correlation analysis
    """
    if seed is not None:
        np.random.seed(seed)

    n_dims = len(bounds)

    if n_dims != 2:
        # For more than 2 dimensions, use LHS (which tends to be decorrelated)
        samples = latin_hypercube_sample(n_samples, bounds, seed=seed)
        corr_info = correlation_analysis(samples)
        return samples, corr_info

    # For 2D case (e.g., H and D), use iterative approach
    # Start with LHS
    samples = latin_hypercube_sample(n_samples, bounds, seed=seed)

    # Compute current correlation
    corr_info = correlation_analysis(samples)
    current_corr = corr_info["max_correlation"]

    # If already within tolerance, return
    if abs(current_corr - target_correlation) <= tolerance:
        return samples, corr_info

    # Iteratively adjust to achieve target correlation
    for iteration in range(max_iterations):
        # Compute correlation
        corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]

        if abs(corr - target_correlation) <= tolerance:
            break

        # Adjust samples to reduce correlation
        # Use rank-based transformation
        ranks_0 = stats.rankdata(samples[:, 0])
        ranks_1 = stats.rankdata(samples[:, 1])

        # Rescale ranks to original bounds
        min_0, max_0 = bounds[0]
        min_1, max_1 = bounds[1]

        samples[:, 0] = (ranks_0 - 1) / (n_samples - 1) * (max_0 - min_0) + min_0
        samples[:, 1] = (ranks_1 - 1) / (n_samples - 1) * (max_1 - min_1) + min_1

        # Add small random perturbation to break perfect correlation
        if abs(corr) > 0.9:
            noise_scale = 0.01 * np.array([max_0 - min_0, max_1 - min_1])
            samples += np.random.normal(0, noise_scale, samples.shape)
            # Clip to bounds
            samples[:, 0] = np.clip(samples[:, 0], min_0, max_0)
            samples[:, 1] = np.clip(samples[:, 1], min_1, max_1)

    # Final correlation analysis
    corr_info = correlation_analysis(samples)

    return samples, corr_info


def validate_sample_quality(
    samples: np.ndarray,
    bounds: List[Tuple[float, float]],
    min_correlation: float = -0.3,
    max_correlation: float = 0.3,
) -> Dict[str, any]:
    """
    Validate quality of generated samples.

    Checks:
    - Coverage of parameter space
    - Correlation between parameters
    - Uniformity of distribution

    Parameters:
    -----------
    samples : np.ndarray
        Array of shape (n_samples, n_dims) with parameter values
    bounds : list of tuples
        List of (min, max) bounds for each dimension
    min_correlation : float
        Minimum acceptable correlation (for parameter estimation)
    max_correlation : float
        Maximum acceptable correlation (for parameter estimation)

    Returns:
    --------
    dict
        Dictionary with validation results and quality metrics
    """
    n_samples, n_dims = samples.shape

    # Check bounds
    within_bounds = True
    for i, (min_val, max_val) in enumerate(bounds):
        if np.any(samples[:, i] < min_val) or np.any(samples[:, i] > max_val):
            within_bounds = False
            break

    # Correlation analysis
    corr_info = correlation_analysis(samples)
    max_corr = corr_info["max_correlation"]
    correlation_ok = min_correlation <= max_corr <= max_correlation

    # Coverage analysis (check if samples cover the space)
    coverage_scores = []
    for i, (min_val, max_val) in enumerate(bounds):
        range_val = max_val - min_val
        # Check coverage in each quartile
        q1 = min_val + 0.25 * range_val
        q2 = min_val + 0.50 * range_val
        q3 = min_val + 0.75 * range_val

        in_q1 = np.sum((samples[:, i] >= min_val) & (samples[:, i] < q1))
        in_q2 = np.sum((samples[:, i] >= q1) & (samples[:, i] < q2))
        in_q3 = np.sum((samples[:, i] >= q2) & (samples[:, i] < q3))
        in_q4 = np.sum((samples[:, i] >= q3) & (samples[:, i] <= max_val))

        # Ideal: 25% in each quartile
        quartile_counts = [in_q1, in_q2, in_q3, in_q4]
        expected_per_quartile = n_samples / 4
        coverage_score = 1.0 - np.std(quartile_counts) / expected_per_quartile
        coverage_scores.append(coverage_score)

    avg_coverage = np.mean(coverage_scores)

    return {
        "within_bounds": within_bounds,
        "correlation_ok": correlation_ok,
        "max_correlation": max_corr,
        "coverage_score": avg_coverage,
        "correlation_info": corr_info,
        "n_samples": n_samples,
        "n_dims": n_dims,
        "validation_passed": within_bounds and correlation_ok and avg_coverage > 0.5,
    }


def filter_extreme_combinations(
    samples: np.ndarray,
    bounds: List[Tuple[float, float]],
    H_idx: int = 0,
    D_idx: int = 1,
    Load_idx: Optional[int] = None,
    H_threshold: float = 0.3,
    D_threshold: float = 0.3,
    Load_threshold: float = 0.8,
    verbose: bool = False,
    # Priority 1: Absolute thresholds based on physical reality
    H_min_absolute: float = 2.0,  # seconds - minimum realistic H
    D_min_absolute: float = 0.5,  # pu - minimum realistic D
    D_min_critical: float = 0.3,  # pu - critical minimum D
    Load_max_absolute: float = 0.9,  # pu - maximum safe loading
) -> Tuple[np.ndarray, int]:
    """
    Filter out extreme parameter combinations from sampled data.

    Removes combinations that are likely to cause convergence issues:
    - High load (>= threshold) + Low H (<= threshold) + Low D (<= threshold)

    This is a post-sampling filter that can be applied to any sampling strategy.

    Parameters:
    -----------
    samples : np.ndarray
        Array of shape (n_samples, n_dims) with parameter values
    bounds : list of tuples
        List of (min, max) bounds for each dimension
    H_idx : int
        Index of H (inertia) dimension in samples array (default: 0)
    D_idx : int
        Index of D (damping) dimension in samples array (default: 1)
    Load_idx : int, optional
        Index of Load dimension in samples array (None if not present)
    H_threshold : float
        Normalized threshold for low H (0.0 to 1.0, default: 0.3 = 30% of range)
    D_threshold : float
        Normalized threshold for low D (0.0 to 1.0, default: 0.3 = 30% of range)
    Load_threshold : float
        Normalized threshold for high load (0.0 to 1.0, default: 0.8 = 80% of range)
    verbose : bool
        Whether to print filtering statistics

    Returns:
    --------
    tuple
        (filtered_samples, n_filtered) where:
        - filtered_samples: Array with extreme combinations removed
        - n_filtered: Number of samples that were filtered out
    """
    if samples.shape[0] == 0:
        return samples, 0

    n_samples, n_dims = samples.shape

    # Normalize samples to [0, 1] range
    normalized = np.zeros_like(samples)
    for i, (min_val, max_val) in enumerate(bounds):
        if max_val > min_val:
            normalized[:, i] = (samples[:, i] - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 0.0

    # Create mask for valid samples (not extreme)
    valid_mask = np.ones(n_samples, dtype=bool)

    # PRIORITY 1: Check absolute thresholds FIRST (physical reality)
    # These are independent of input ranges

    # Check absolute H threshold
    if H_idx < n_dims:
        H_values = samples[:, H_idx]
        low_H_absolute = H_values < H_min_absolute
        valid_mask = valid_mask & ~low_H_absolute
        n_low_H = np.sum(low_H_absolute)
        if verbose and n_low_H > 0:
            print(
                f"[FILTER] Removed {n_low_H} samples with H < {H_min_absolute}s (absolute"
                f"threshold)"
            )

    # Check absolute D threshold (critical minimum)
    if D_idx < n_dims:
        D_values = samples[:, D_idx]
        low_D_critical = D_values < D_min_critical
        valid_mask = valid_mask & ~low_D_critical
        n_low_D_critical = np.sum(low_D_critical)
        if verbose and n_low_D_critical > 0:
            print(
                f"[FILTER] Removed {n_low_D_critical} samples with D < {D_min_critical}pu (critical"
                f"minimum)"
            )

    # Check absolute load threshold
    if Load_idx is not None and Load_idx < n_dims:
        Load_values = samples[:, Load_idx]
        high_load_absolute = Load_values > Load_max_absolute
        valid_mask = valid_mask & ~high_load_absolute
        n_high_load = np.sum(high_load_absolute)
        if verbose and n_high_load > 0:
            print(
                f"[FILTER] Removed {n_high_load} samples with Load > {Load_max_absolute}pu"
                f"(absolute threshold)"
            )

    # Check for extreme combinations using normalized thresholds (relative to range)
    # This catches combinations that are extreme relative to the specified range
    if Load_idx is not None and Load_idx < n_dims:
        # Extreme: High load + Low H + Low D (normalized)
        high_load_norm = normalized[:, Load_idx] >= Load_threshold
        low_H_norm = normalized[:, H_idx] <= H_threshold
        low_D_norm = normalized[:, D_idx] <= D_threshold

        extreme_mask_norm = high_load_norm & low_H_norm & low_D_norm
        valid_mask = valid_mask & ~extreme_mask_norm

        n_extreme_norm = np.sum(extreme_mask_norm)
        if verbose and n_extreme_norm > 0:
            print(
                f"[FILTER] Removed {n_extreme_norm} extreme combinations (normalized): "
                f"High load (≥{Load_threshold:.0%}) + Low H (≤{H_threshold:.0%}) + "
                f"Low D (≤{D_threshold:.0%})"
            )
    else:
        # If no Load dimension, only check H and D (less critical)
        if verbose:
            print(
                "[FILTER] No Load dimension found, skipping normalized extreme combination filter"
            )

    filtered_samples = samples[valid_mask]
    n_filtered = n_samples - len(filtered_samples)

    return filtered_samples, n_filtered
