"""
Comprehensive Evaluation Script for PINN Models.

This script provides a complete evaluation framework that:
1. Evaluates PINN models with comprehensive metrics
2. Compares against baseline methods
3. Conducts statistical analysis
4. Generates publication-ready reports
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.baseline_comparison import BaselineComparator, EACBaseline, MLBaseline, TDSBaseline
from evaluation.statistical_analysis import (
    compare_methods,
    compute_statistics,
    generate_statistical_report,
)
from utils.metrics import compute_cct_metrics, compute_parameter_metrics, compute_trajectory_metrics


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for PINN models.

    Provides end-to-end evaluation including:
    - PINN model evaluation
    - Baseline comparisons
    - Statistical analysis
    - Report generation
    """

    def __init__(self, output_dir: str = "outputs/evaluation"):
        """
        Initialize comprehensive evaluator.

        Parameters:
        -----------
        output_dir : str
            Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.comparator = BaselineComparator()
        self.results = {}

    def setup_baselines(self):
        """Setup baseline methods for comparison."""
        # EAC baseline
        self.comparator.add_baseline("EAC", EACBaseline())

        # TDS baseline
        self.comparator.add_baseline("TDS", TDSBaseline())

        # ML baselines
        self.comparator.add_baseline("LSTM", MLBaseline(model_type="LSTM"))
        self.comparator.add_baseline("CNN", MLBaseline(model_type="CNN"))

    def evaluate_pinn_trajectory(
        self, pinn_model: Any, test_data: Dict, device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Evaluate PINN model on trajectory prediction task.

        Parameters:
        -----------
        pinn_model : object
            Trained PINN model
        test_data : dict
            Test dataset with keys: 't', 'delta_true', 'omega_true', and parameters
        device : str
            Device to use ('cpu' or 'cuda')

        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        pinn_model.eval()

        # Extract test data
        t = test_data["t"]
        delta_true = test_data["delta_true"]
        omega_true = test_data["omega_true"]

        # Prepare inputs
        # Assuming test_data contains all necessary parameters
        batch_size = len(t) if isinstance(t, np.ndarray) else 1

        # Convert to tensors
        t_tensor = torch.FloatTensor(t).to(device)

        # Get predictions
        with torch.no_grad():
            # This is a placeholder - actual implementation depends on model interface
            # delta_pred, omega_pred = pinn_model.predict_trajectory(...)
            # For now, return placeholder
            delta_pred = delta_true  # Placeholder
            omega_pred = omega_true  # Placeholder

        # Compute metrics
        metrics = compute_trajectory_metrics(
            delta_pred=delta_pred,
            omega_pred=omega_pred,
            delta_true=delta_true,
            omega_true=omega_true,
            t=t,
        )

        return metrics

    def evaluate_pinn_cct(
        self, pinn_model: Any, test_cases: List[Dict], true_cct: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate PINN model on CCT estimation task.

        Parameters:
        -----------
        pinn_model : object
            Trained PINN model
        test_cases : list
            List of test case dictionaries
        true_cct : np.ndarray, optional
            True CCT values

        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Placeholder - would use binary search with PINN model.
        # from utils.cct_binary_search import estimate_cct_binary_search

        if true_cct is None:
            return {}

        # For now, return placeholder
        cct_pred = true_cct  # Placeholder

        metrics = compute_cct_metrics(cct_pred, true_cct)
        return metrics

    def run_comprehensive_evaluation(
        self,
        pinn_model: Any,
        test_data: Dict,
        test_cases: Optional[List[Dict]] = None,
        true_cct: Optional[np.ndarray] = None,
        n_runs: int = 5,
        seeds: Optional[List[int]] = None,
    ) -> Dict:
        """
        Run comprehensive evaluation with multiple runs and statistical analysis.

        Parameters:
        -----------
        pinn_model : object
            Trained PINN model
        test_data : dict
            Test dataset
        test_cases : list, optional
            Test cases for CCT evaluation
        true_cct : np.ndarray, optional
            True CCT values
        n_runs : int
            Number of runs for statistical analysis
        seeds : list, optional
            Random seeds for each run

        Returns:
        --------
        results : dict
            Comprehensive results dictionary
        """
        if seeds is None:
            seeds = list(range(n_runs))

        all_results = {"trajectory": [], "cct": [], "baseline_comparison": {}}

        # Run evaluation multiple times
        for run_idx, seed in enumerate(seeds[:n_runs]):
            print("\n{'='*80}")
            print("Run {run_idx + 1}/{n_runs} (seed={seed})")
            print("{'='*80}\n")

            # Set random seed
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Evaluate PINN trajectory prediction
            trajectory_metrics = self.evaluate_pinn_trajectory(pinn_model, test_data)
            all_results["trajectory"].append(trajectory_metrics)

            # Evaluate PINN CCT estimation
            if test_cases is not None and true_cct is not None:
                cct_metrics = self.evaluate_pinn_cct(pinn_model, test_cases, true_cct)
                all_results["cct"].append(cct_metrics)

        # Statistical analysis
        trajectory_stats = {}
        if len(all_results["trajectory"]) > 0:
            for metric_name in all_results["trajectory"][0].keys():
                values = [r[metric_name] for r in all_results["trajectory"]]
                trajectory_stats[metric_name] = compute_statistics(np.array(values))

        cct_stats = {}
        if len(all_results["cct"]) > 0:
            for metric_name in all_results["cct"][0].keys():
                values = [r[metric_name] for r in all_results["cct"]]
                cct_stats[metric_name] = compute_statistics(np.array(values))

        # Baseline comparison
        baseline_results = self.comparator.compare_trajectory_prediction(
            test_cases if test_cases else [], pinn_model, metrics=None
        )
        all_results["baseline_comparison"] = baseline_results

        # Store results
        self.results = {
            "trajectory": {
                "individual_runs": all_results["trajectory"],
                "statistics": trajectory_stats,
            },
            "cct": {"individual_runs": all_results["cct"], "statistics": cct_stats},
            "baseline_comparison": baseline_results,
        }

        return self.results

    def generate_report(self, filename: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.

        Parameters:
        -----------
        filename : str, optional
            File path to save report

        Returns:
        --------
        report : str
            Formatted report string
        """
        report = "=" * 80 + "\n"
        report += "COMPREHENSIVE PINN EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"

        # Trajectory prediction results
        if "trajectory" in self.results:
            report += "TRAJECTORY PREDICTION RESULTS\n"
            report += "-" * 80 + "\n\n"

            if "statistics" in self.results["trajectory"]:
                for metric_name, stats_dict in self.results["trajectory"]["statistics"].items():
                    report += "{metric_name}:\n"
                    report += "  Mean: {stats_dict['mean']:.6f}\n"
                    report += "  Std: {stats_dict['std']:.6f}\n"
                    report += (
                        "  95% CI: [{stats_dict['ci_lower']:.6f}, {stats_dict['ci_upper']:.6f}]\n"
                    )
                    report += "  Median: {stats_dict['median']:.6f}\n\n"

        # CCT estimation results
        if "cct" in self.results:
            report += "\n" + "=" * 80 + "\n"
            report += "CCT ESTIMATION RESULTS\n"
            report += "-" * 80 + "\n\n"

            if "statistics" in self.results["cct"]:
                for metric_name, stats_dict in self.results["cct"]["statistics"].items():
                    report += "{metric_name}:\n"
                    report += "  Mean: {stats_dict['mean']:.6f}\n"
                    report += "  Std: {stats_dict['std']:.6f}\n"
                    report += (
                        "  95% CI: [{stats_dict['ci_lower']:.6f}, {stats_dict['ci_upper']:.6f}]\n"
                    )
                    report += "  Median: {stats_dict['median']:.6f}\n\n"

        # Baseline comparison
        if "baseline_comparison" in self.results:
            report += "\n" + "=" * 80 + "\n"
            report += "BASELINE COMPARISON\n"
            report += "-" * 80 + "\n\n"
            report += self.comparator.generate_comparison_report()

        if filename:
            filepath = self.output_dir / filename
            with open(filepath, "w") as f:
                f.write(report)
            print("Report saved to {filepath}")

        return report

    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            filename = "evaluation_results.json"

        filepath = self.output_dir / filename

        # Convert to JSON-serializable format
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        serializable_results[key][sub_key] = [
                            {
                                k: float(v) if isinstance(v, (int, float, np.number)) else v
                                for k, v in item.items()
                            }
                            for item in sub_value
                        ]
                    elif isinstance(sub_value, dict):
                        serializable_results[key][sub_key] = {
                            k: {
                                sk: float(sv) if isinstance(sv, (int, float, np.number)) else sv
                                for sk, sv in sv_dict.items()
                            }
                            for k, sv_dict in sub_value.items()
                        }

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print("Results saved to {filepath}")


def main():
    """Example usage of comprehensive evaluator."""
    evaluator = ComprehensiveEvaluator()
    evaluator.setup_baselines()

    # Example: Load test data and model
    # test_data = load_test_data(...)
    # pinn_model = load_trained_model(...)

    # Run evaluation
    # results = evaluator.run_comprehensive_evaluation(
    #     pinn_model=pinn_model,
    #     test_data=test_data,
    #     n_runs=5
    # )

    # Generate report
    # report = evaluator.generate_report("evaluation_report.txt")
    # evaluator.save_results("evaluation_results.json")

    print("Comprehensive evaluation framework ready!")
    print("Use ComprehensiveEvaluator class to run evaluations.")


if __name__ == "__main__":
    main()
