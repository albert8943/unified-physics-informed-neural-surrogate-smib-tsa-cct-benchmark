"""
Statistical Analysis Tools for PINN Evaluation.

This module provides statistical analysis tools including:
- Multiple runs with different seeds
- Statistical significance testing
- Confidence intervals
- Error distribution analysis
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_statistics(values: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Compute statistical summary for a set of values.

    Parameters:
    -----------
    values : np.ndarray
        Array of values
    confidence_level : float
        Confidence level (default 0.95 for 95% CI)

    Returns:
    --------
    stats_dict : dict
        Dictionary with mean, std, CI, etc.
    """
    stats_dict = {}

    stats_dict["mean"] = np.mean(values)
    stats_dict["std"] = np.std(values, ddof=1)  # Sample std
    stats_dict["median"] = np.median(values)
    stats_dict["min"] = np.min(values)
    stats_dict["max"] = np.max(values)
    stats_dict["q25"] = np.percentile(values, 25)
    stats_dict["q75"] = np.percentile(values, 75)

    # Confidence interval
    n = len(values)
    if n > 1:
        se = stats_dict["std"] / np.sqrt(n)
        t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        stats_dict["ci_lower"] = stats_dict["mean"] - t_critical * se
        stats_dict["ci_upper"] = stats_dict["mean"] + t_critical * se
        stats_dict["ci_width"] = stats_dict["ci_upper"] - stats_dict["ci_lower"]
    else:
        stats_dict["ci_lower"] = stats_dict["mean"]
        stats_dict["ci_upper"] = stats_dict["mean"]
        stats_dict["ci_width"] = 0.0

    return stats_dict


def compare_methods(
    method1_results: List[Dict],
    method2_results: List[Dict],
    metric_name: str,
    test_type: str = "t-test",
) -> Dict[str, float]:
    """
    Compare two methods using statistical tests.

    Parameters:
    -----------
    method1_results : list
        List of result dictionaries from method 1
    method2_results : list
        List of result dictionaries from method 2
    metric_name : str
        Name of metric to compare
    test_type : str
        Type of test: 't-test', 'mann-whitney', 'wilcoxon'

    Returns:
    --------
    comparison : dict
        Dictionary with test results
    """
    # Extract metric values.
    values1 = [r[metric_name] for r in method1_results if metric_name in r]
    values2 = [r[metric_name] for r in method2_results if metric_name in r]

    if len(values1) == 0 or len(values2) == 0:
        return {"error": "Insufficient data"}

    values1 = np.array(values1)
    values2 = np.array(values2)

    comparison = {}

    # Basic statistics
    comparison["method1_mean"] = np.mean(values1)
    comparison["method1_std"] = np.std(values1, ddof=1)
    comparison["method2_mean"] = np.mean(values2)
    comparison["method2_std"] = np.std(values2, ddof=1)
    comparison["difference"] = comparison["method1_mean"] - comparison["method2_mean"]
    comparison["relative_improvement"] = (
        (comparison["method2_mean"] - comparison["method1_mean"])
        / (comparison["method2_mean"] + 1e-8)
        * 100
    )

    # Statistical tests
    if test_type == "t-test":
        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(values1, values2)
        comparison["test_statistic"] = t_stat
        comparison["p_value"] = p_value
        comparison["significant"] = p_value < 0.05
    elif test_type == "mann-whitney":
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(values1, values2, alternative="two-sided")
        comparison["test_statistic"] = u_stat
        comparison["p_value"] = p_value
        comparison["significant"] = p_value < 0.05
    elif test_type == "wilcoxon":
        # Wilcoxon signed-rank test (paired)
        if len(values1) == len(values2):
            w_stat, p_value = stats.wilcoxon(values1, values2)
            comparison["test_statistic"] = w_stat
            comparison["p_value"] = p_value
            comparison["significant"] = p_value < 0.05
        else:
            comparison["error"] = "Wilcoxon test requires paired samples"

    return comparison


def multiple_runs_analysis(
    all_results: List[List[Dict]], metric_names: List[str]
) -> Dict[str, Dict]:
    """
    Analyze results from multiple runs (different seeds)

    Parameters:
    -----------
    all_results : list
        List of result lists (one per run)
    metric_names : list
        List of metric names to analyze

    Returns:
    --------
    analysis : dict
        Dictionary with statistics for each metric
    """
    analysis = {}

    for metric_name in metric_names:
        # Extract metric values across all runs
        metric_values = []
        for run_results in all_results:
            for result in run_results:
                if metric_name in result:
                    metric_values.append(result[metric_name])

        if len(metric_values) > 0:
            analysis[metric_name] = compute_statistics(np.array(metric_values))

    return analysis


def error_distribution_analysis(errors: np.ndarray) -> Dict[str, float]:
    """
    Analyze error distribution.

    Parameters:
    -----------
    errors : np.ndarray
        Array of errors

    Returns:
    --------
    analysis : dict
        Dictionary with distribution statistics
    """
    analysis = {}

    analysis["mean"] = np.mean(errors)
    analysis["std"] = np.std(errors, ddof=1)
    analysis["skewness"] = stats.skew(errors)
    analysis["kurtosis"] = stats.kurtosis(errors)
    analysis["median"] = np.median(errors)
    analysis["mad"] = np.median(np.abs(errors - analysis["median"]))  # Median absolute deviation

    # Test for normality
    if len(errors) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(errors)
        analysis["shapiro_statistic"] = shapiro_stat
        analysis["shapiro_p_value"] = shapiro_p
        analysis["is_normal"] = shapiro_p > 0.05

    # Percentiles
    analysis["p95"] = np.percentile(np.abs(errors), 95)
    analysis["p99"] = np.percentile(np.abs(errors), 99)

    return analysis


def generate_statistical_report(
    results: Dict[str, List[Dict]], output_file: Optional[str] = None
) -> str:
    """
    Generate comprehensive statistical report.

    Parameters:
    -----------
    results : dict
        Dictionary of results (method_name -> list of result dicts)
    output_file : str, optional
        File path to save report

    Returns:
    --------
    report : str
        Formatted report string
    """
    report = "=" * 80 + "\n"
    report += "STATISTICAL ANALYSIS REPORT\n"
    report += "=" * 80 + "\n\n"

    # For each method, compute statistics
    for method_name, method_results in results.items():
        report += "{method_name}\n"
        report += "-" * 80 + "\n"

        # Extract all metric names
        if len(method_results) > 0:
            metric_names = set()
            for result in method_results:
                metric_names.update(result.keys())

            # Compute statistics for each metric
            for metric_name in sorted(metric_names):
                values = [r[metric_name] for r in method_results if metric_name in r]
                if len(values) > 0:
                    stats_dict = compute_statistics(np.array(values))
                    report += "\n{metric_name}:\n"
                    report += "  Mean: {stats_dict['mean']:.6f}\n"
                    report += "  Std: {stats_dict['std']:.6f}\n"
                    report += (
                        "  95% CI: [{stats_dict['ci_lower']:.6f}, {stats_dict['ci_upper']:.6f}]\n"
                    )
                    report += "  Median: {stats_dict['median']:.6f}\n"
                    report += "  Range: [{stats_dict['min']:.6f}, {stats_dict['max']:.6f}]\n"

        report += "\n"

    # Method comparisons
    if len(results) > 1:
        report += "\n" + "=" * 80 + "\n"
        report += "METHOD COMPARISONS\n"
        report += "=" * 80 + "\n\n"

        method_names = list(results.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method1 = method_names[i]
                method2 = method_names[j]

                report += "{method1} vs {method2}\n"
                report += "-" * 80 + "\n"

                # Find common metrics
                metrics1 = set()
                for r in results[method1]:
                    metrics1.update(r.keys())
                metrics2 = set()
                for r in results[method2]:
                    metrics2.update(r.keys())
                common_metrics = metrics1 & metrics2

                for metric_name in sorted(common_metrics):
                    comparison = compare_methods(
                        results[method1], results[method2], metric_name, test_type="t-test"
                    )
                    if "error" not in comparison:
                        report += f"\n{metric_name}:\n"
                        report += (
                            f"  {method1}: {comparison['method1_mean']:.6f} "
                            f"± {comparison['method1_std']:.6f}\n"
                        )
                        report += (
                            f"  {method2}: {comparison['method2_mean']:.6f} "
                            f"± {comparison['method2_std']:.6f}\n"
                        )
                        report += f"  Difference: {comparison['difference']:.6f}\n"
                        report += f"  p-value: {comparison['p_value']:.6f}\n"
                        report += f"  Significant: {comparison['significant']}\n"

                report += "\n"

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)

    return report
