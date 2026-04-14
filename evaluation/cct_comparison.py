"""
CCT Estimation Comparison.

Compares CCT estimation accuracy across methods:
- PINN (using binary search)
- ANDES TDS (using binary search)
- EAC (analytical)

Extended with damping analysis and quantitative comparison tools.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from scipy import stats
from evaluation.baseline_comparison import EACBaseline, TDSBaseline
from utils.metrics import compute_cct_metrics


def estimate_cct_pinn(
    pinn_model,
    scenario: Dict,
    cct_min: float = 0.1,
    cct_max: float = 0.5,
    tolerance: float = 0.001,
    max_iter: int = 20,
) -> Optional[float]:
    """
    Estimate CCT using PINN with binary search.

    Parameters:
    -----------
    pinn_model : nn.Module
        Trained PINN model
    scenario : dict
        Scenario parameters
    cct_min : float
        Minimum CCT guess
    cct_max : float
        Maximum CCT guess
    tolerance : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns:
    --------
    cct : float or None
        Estimated CCT
    """
    # Binary search
    low, high = cct_min, cct_max

    for _ in range(max_iter):
        if high - low < tolerance:
            return (low + high) / 2

        tc = (low + high) / 2

        # Predict trajectory with this clearing time
        # (Implementation depends on PINN interface)
        # For now, placeholder
        is_stable = True  # Would use PINN to predict stability

        if is_stable:
            low = tc
        else:
            high = tc

    return (low + high) / 2


def compare_cct_estimation(
    test_scenarios: List[Dict],
    pinn_model=None,
    true_cct: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compare CCT estimation across methods.

    Parameters:
    -----------
    test_scenarios : list
        List of test scenarios
    pinn_model : nn.Module, optional
        Trained PINN model
    true_cct : np.ndarray, optional
        True CCT values

    Returns:
    --------
    results : dict
        Comparison results
    """
    results = {}

    # EAC baseline
    eac_baseline = EACBaseline()
    cct_eac = []
    eac_failures = []
    for i, scenario in enumerate(test_scenarios):
        # Get M (inertia coefficient): prefer M if available, otherwise compute from H
        M = scenario.get("M", None)
        if M is None:
            H = scenario.get("H", 6.0)
            M = 2.0 * H
        try:
            cct = eac_baseline.estimate_cct(
                Pm=scenario.get("Pm", 0.8),
                M=float(M),
                D=scenario.get("D", 1.0),
                Xprefault=scenario.get("Xprefault", 0.5),
                Xfault=scenario.get("Xfault", 0.0001),
                Xpostfault=scenario.get("Xpostfault", 0.5),
                V1=scenario.get("V1", 1.05),
                V2=scenario.get("V2", 1.0),
            )
            if cct is None:
                eac_failures.append(i)
                if i < 3:  # Print first 3 failures for debugging
                    print(
                        f"  [DEBUG] EAC returned None for scenario {i}: Pm={scenario.get('Pm', 'N/A')}, M={M}, Xprefault={scenario.get('Xprefault', 'N/A')}, V1={scenario.get('V1', 'N/A')}, V2={scenario.get('V2', 'N/A')}"
                    )
            cct_eac.append(cct if cct is not None else np.nan)
        except Exception as e:
            eac_failures.append(i)
            if i < 3:  # Print first 3 exceptions for debugging
                print(f"  [DEBUG] EAC exception for scenario {i}: {e}")
                print(f"    Scenario keys: {list(scenario.keys())}")
                print(
                    f"    Pm={scenario.get('Pm', 'N/A')}, M={M}, Xprefault={scenario.get('Xprefault', 'N/A')}, V1={scenario.get('V1', 'N/A')}, V2={scenario.get('V2', 'N/A')}"
                )
            cct_eac.append(np.nan)

    if eac_failures:
        print(f"⚠️  EAC calculation failed for {len(eac_failures)}/{len(test_scenarios)} scenarios")

    results["EAC"] = {
        "cct_predictions": np.array(cct_eac),
        "method": "analytical",
    }

    # TDS baseline
    tds_baseline = TDSBaseline()
    if tds_baseline.andes_available:
        cct_tds = []
        for scenario in test_scenarios:
            # Get M (inertia coefficient): prefer M if available, otherwise compute from H
            M = scenario.get("M", None)
            if M is None:
                H = scenario.get("H", 6.0)
                M = 2.0 * H
            cct = tds_baseline.find_cct(
                Pm=scenario.get("Pm", 0.8),
                M=float(M),
                D=scenario.get("D", 1.0),
            )
            cct_tds.append(cct if cct is not None else np.nan)

        results["TDS"] = {
            "cct_predictions": np.array(cct_tds),
            "method": "simulation",
        }

    # PINN baseline
    if pinn_model is not None:
        cct_pinn = []
        for scenario in test_scenarios:
            cct = estimate_cct_pinn(pinn_model, scenario)
            cct_pinn.append(cct if cct is not None else np.nan)

        results["PINN"] = {
            "cct_predictions": np.array(cct_pinn),
            "method": "neural_network",
        }

    # Compute metrics if true CCT available
    if true_cct is not None:
        for method_name, method_results in results.items():
            predictions = method_results["cct_predictions"]
            # Remove NaN values
            valid_mask = ~np.isnan(predictions)
            if np.sum(valid_mask) > 0:
                metrics = compute_cct_metrics(
                    cct_pred=predictions[valid_mask],
                    cct_true=true_cct[valid_mask],
                )
                method_results["metrics"] = metrics

    return results


def check_damping_distribution(
    test_scenarios: List[Dict], thresholds: List[float] = [0.3, 0.5, 1.5]
) -> Dict[str, int]:
    """
    Check distribution of damping coefficients in test scenarios.

    Parameters:
    -----------
    test_scenarios : list
        List of test scenarios with 'D' key
    thresholds : list
        Damping thresholds to check (default: [0.3, 0.5, 1.5] pu)

    Returns:
    --------
    distribution : dict
        Dictionary with counts for each threshold range
    """
    damping_values = [scenario.get("D", 0.0) for scenario in test_scenarios]
    damping_values = np.array(damping_values)

    # Handle empty array case
    if len(damping_values) == 0:
        return {
            "total": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
        }

    distribution = {
        "total": len(damping_values),
        "min": float(np.min(damping_values)),
        "max": float(np.max(damping_values)),
        "mean": float(np.mean(damping_values)),
        "median": float(np.median(damping_values)),
    }

    # Count by thresholds (only if we have data)
    if len(damping_values) > 0:
        for i, threshold in enumerate(thresholds):
            if i == 0:
                count = np.sum(damping_values < threshold)
                distribution[f"d_lt_{threshold}"] = int(count)
            else:
                prev_threshold = thresholds[i - 1]
                count = np.sum((damping_values >= prev_threshold) & (damping_values < threshold))
                distribution[f"d_{prev_threshold}_to_{threshold}"] = int(count)

    # Count above last threshold
    if len(thresholds) > 0:
        last_threshold = thresholds[-1]
        count = np.sum(damping_values >= last_threshold)
        distribution[f"d_ge_{last_threshold}"] = int(count)

    return distribution


def categorize_by_damping(
    test_scenarios: List[Dict], damping_values: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Categorize scenarios by damping level.

    Parameters:
    -----------
    test_scenarios : list
        List of test scenarios
    damping_values : np.ndarray
        Array of damping coefficients

    Returns:
    --------
    categories : dict
        Dictionary with boolean masks for each category:
        - 'low': D < 0.5 pu
        - 'medium': 0.5 <= D < 1.5 pu
        - 'high': D >= 1.5 pu
    """
    categories = {
        "low": damping_values < 0.5,
        "medium": (damping_values >= 0.5) & (damping_values < 1.5),
        "high": damping_values >= 1.5,
    }
    return categories


def analyze_eac_damping_correlation(
    eac_errors: np.ndarray, damping_values: np.ndarray
) -> Dict[str, float]:
    """
    Analyze correlation between EAC error and damping coefficient.

    Parameters:
    -----------
    eac_errors : np.ndarray
        Array of EAC CCT errors (absolute errors in ms)
    damping_values : np.ndarray
        Array of damping coefficients

    Returns:
    --------
    analysis : dict
        Dictionary with correlation coefficient, p-value, and regression parameters
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(eac_errors) | np.isnan(damping_values))
    eac_errors_clean = eac_errors[valid_mask]
    damping_clean = damping_values[valid_mask]

    if len(eac_errors_clean) < 3:
        return {"error": "Insufficient data for correlation analysis"}

    # Pearson correlation
    correlation, p_value = stats.pearsonr(damping_clean, eac_errors_clean)

    # Linear regression
    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
        damping_clean, eac_errors_clean
    )

    analysis = {
        "correlation": float(correlation),
        "p_value": float(p_value),
        "r_squared": float(r_value**2),
        "slope": float(slope),
        "intercept": float(intercept),
        "std_err": float(std_err),
        "n_samples": int(len(eac_errors_clean)),
        "significant": p_value < 0.05,
    }

    return analysis


def compute_eac_success_rate(
    eac_predictions: np.ndarray,
) -> Dict[str, float]:
    """
    Compute EAC success rate (percentage of valid predictions).

    Parameters:
    -----------
    eac_predictions : np.ndarray
        Array of EAC CCT predictions (may contain NaN)

    Returns:
    --------
    stats : dict
        Dictionary with success rate and failure count
    """
    total = len(eac_predictions)
    valid = np.sum(~np.isnan(eac_predictions))
    failed = total - valid

    return {
        "total": int(total),
        "successful": int(valid),
        "failed": int(failed),
        "success_rate": float(valid / total) if total > 0 else 0.0,
    }


def compare_cct_estimation_with_damping_analysis(
    test_scenarios: List[Dict],
    pinn_model=None,
    true_cct: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compare CCT estimation with comprehensive damping analysis.

    Extended version of compare_cct_estimation() that includes:
    - Damping distribution analysis
    - EAC error correlation with damping
    - Categorization by damping level
    - Success rate analysis

    Parameters:
    -----------
    test_scenarios : list
        List of test scenarios with 'D' key for damping
    pinn_model : nn.Module, optional
        Trained PINN model
    true_cct : np.ndarray, optional
        True CCT values from ANDES

    Returns:
    --------
    results : dict
        Extended comparison results with damping analysis
    """
    # Get basic comparison results
    results = compare_cct_estimation(test_scenarios, pinn_model, true_cct)

    # Extract damping values
    damping_values = np.array([scenario.get("D", 0.0) for scenario in test_scenarios])

    # Check damping distribution
    results["damping_distribution"] = check_damping_distribution(test_scenarios)

    # Categorize by damping
    results["damping_categories"] = categorize_by_damping(test_scenarios, damping_values)

    # Analyze EAC if available
    if "EAC" in results and true_cct is not None:
        eac_predictions = results["EAC"]["cct_predictions"]
        valid_mask = ~np.isnan(eac_predictions)

        if np.sum(valid_mask) > 0:
            # Compute errors (convert to ms)
            eac_errors = np.abs(eac_predictions[valid_mask] - true_cct[valid_mask]) * 1000
            damping_valid = damping_values[valid_mask]

            # Correlation analysis
            results["eac_damping_correlation"] = analyze_eac_damping_correlation(
                eac_errors, damping_valid
            )

            # Success rate
            results["eac_success_rate"] = compute_eac_success_rate(eac_predictions)

            # Metrics by damping category
            categories = results["damping_categories"]
            results["eac_metrics_by_category"] = {}

            for category_name, category_mask in categories.items():
                category_mask_valid = category_mask & valid_mask
                if np.sum(category_mask_valid) > 0:
                    eac_pred_cat = eac_predictions[category_mask_valid]
                    true_cct_cat = true_cct[category_mask_valid]
                    category_errors = np.abs(eac_pred_cat - true_cct_cat) * 1000
                    results["eac_metrics_by_category"][category_name] = {
                        "count": int(np.sum(category_mask_valid)),
                        "mean_error_ms": float(np.mean(category_errors)),
                        "std_error_ms": float(np.std(category_errors)),
                        "max_error_ms": float(np.max(category_errors)),
                    }

    # PINN analysis by category (if available)
    if "PINN" in results and true_cct is not None:
        pinn_predictions = results["PINN"]["cct_predictions"]
        valid_mask = ~np.isnan(pinn_predictions)

        if np.sum(valid_mask) > 0:
            categories = results["damping_categories"]
            results["pinn_metrics_by_category"] = {}

            for category_name, category_mask in categories.items():
                category_mask_valid = category_mask & valid_mask
                if np.sum(category_mask_valid) > 0:
                    pinn_pred_cat = pinn_predictions[category_mask_valid]
                    true_cct_cat = true_cct[category_mask_valid]
                    category_errors = np.abs(pinn_pred_cat - true_cct_cat) * 1000
                    results["pinn_metrics_by_category"][category_name] = {
                        "count": int(np.sum(category_mask_valid)),
                        "mean_error_ms": float(np.mean(category_errors)),
                        "std_error_ms": float(np.std(category_errors)),
                        "max_error_ms": float(np.max(category_errors)),
                    }

    return results
