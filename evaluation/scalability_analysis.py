"""
Scalability Analysis Framework.

This module provides tools for analyzing PINN scalability:
- Computational complexity analysis
- Training time vs. system size
- Inference time vs. system size
- Memory requirements
- Comparison with traditional methods
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ScalabilityAnalyzer:
    """
    Analyzer for PINN scalability across different system sizes.
    """

    def __init__(self, output_dir: str = "outputs/scalability"):
        """
        Initialize scalability analyzer.

        Parameters:
        -----------
        output_dir : str
            Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def measure_training_time(
        self, model: Any, train_data: Dict, epochs: int = 100, device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Measure training time for a model.

        Parameters:
        -----------
        model : object
            Model to train
        train_data : dict
            Training dataset
        epochs : int
            Number of training epochs
        device : str
            Device to use

        Returns:
        --------
        timing : dict
            Dictionary with timing information
        """
        model = model.to(device)

        # Measure training time
        start_time = time.time()

        # Placeholder for actual training
        # In practice, this would run the actual training loop
        # for epoch in range(epochs):
        #     train_one_epoch(model, train_data, device)

        elapsed_time = time.time() - start_time

        timing = {
            "total_time": elapsed_time,
            "time_per_epoch": elapsed_time / epochs if epochs > 0 else 0.0,
            "epochs": epochs,
        }

        return timing

    def measure_inference_time(
        self, model: Any, test_data: Dict, n_runs: int = 100, device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Measure inference time for a model.

        Parameters:
        -----------
        model : object
            Model to evaluate
        test_data : dict
            Test dataset
        n_runs : int
            Number of inference runs
        device : str
            Device to use

        Returns:
        --------
        timing : dict
            Dictionary with timing information
        """
        model = model.to(device)
        model.eval()

        # Warm-up run
        with torch.no_grad():
            # Placeholder inference
            pass

        # Measure inference time
        inference_times = []

        with torch.no_grad():
            for _ in range(n_runs):
                start_time = time.time()
                # Placeholder inference
                # output = model(input_data)
                elapsed_time = time.time() - start_time
                inference_times.append(elapsed_time)

        inference_times = np.array(inference_times)

        timing = {
            "mean_time": np.mean(inference_times),
            "std_time": np.std(inference_times),
            "min_time": np.min(inference_times),
            "max_time": np.max(inference_times),
            "median_time": np.median(inference_times),
            "p95_time": np.percentile(inference_times, 95),
            "p99_time": np.percentile(inference_times, 99),
            "n_runs": n_runs,
        }

        return timing

    def measure_memory_usage(self, model: Any, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """
        Measure memory usage of a model.

        Parameters:
        -----------
        model : object
            Model to analyze
        input_shape : tuple
            Input tensor shape

        Returns:
        --------
        memory : dict
            Dictionary with memory information
        """
        # Count parameters.
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate memory (rough)
        # 4 bytes per float32 parameter
        param_memory_mb = n_params * 4 / (1024**2)

        # Estimate activation memory (rough)
        # This would require actual forward pass
        activation_memory_mb = 0.0  # Placeholder

        total_memory_mb = param_memory_mb + activation_memory_mb

        memory = {
            "n_parameters": n_params,
            "n_trainable": n_trainable,
            "parameter_memory_mb": param_memory_mb,
            "activation_memory_mb": activation_memory_mb,
            "total_memory_mb": total_memory_mb,
        }

        return memory

    def analyze_scalability(
        self,
        system_sizes: List[int],
        models: Dict[str, Any],
        train_data: Dict[int, Dict],
        test_data: Dict[int, Dict],
    ) -> Dict[str, Dict]:
        """
        Analyze scalability across different system sizes.

        Parameters:
        -----------
        system_sizes : list
            List of system sizes (e.g., [1, 14, 39, 118] for number of generators)
        models : dict
            Dictionary of models for each system size
        train_data : dict
            Dictionary of training data for each system size
        test_data : dict
            Dictionary of test data for each system size

        Returns:
        --------
        analysis : dict
            Dictionary with scalability analysis results
        """
        analysis = {"training_time": {}, "inference_time": {}, "memory_usage": {}, "complexity": {}}

        for size in system_sizes:
            print("\nAnalyzing system size: {size}")

            if size not in models or size not in train_data or size not in test_data:
                print("  Warning: Missing data for size {size}, skipping...")
                continue

            model = models[size]
            train = train_data[size]
            test = test_data[size]

            # Training time
            print("  Measuring training time...")
            training_timing = self.measure_training_time(model, train, epochs=10)
            analysis["training_time"][size] = training_timing

            # Inference time
            print("  Measuring inference time...")
            inference_timing = self.measure_inference_time(model, test, n_runs=10)
            analysis["inference_time"][size] = inference_timing

            # Memory usage
            print("  Measuring memory usage...")
            # Estimate input shape from test data
            input_shape = (1, 10)  # Placeholder
            memory_info = self.measure_memory_usage(model, input_shape)
            analysis["memory_usage"][size] = memory_info

            # Computational complexity
            n_params = memory_info["n_parameters"]
            analysis["complexity"][size] = {
                "n_parameters": n_params,
                "n_generators": size,
                "params_per_generator": n_params / size if size > 0 else 0,
            }

        self.results = analysis
        return analysis

    def compare_with_traditional_methods(
        self, system_sizes: List[int], pinn_times: Dict[int, float], tds_times: Dict[int, float]
    ) -> Dict[str, Any]:
        """
        Compare PINN with traditional TDS methods.

        Parameters:
        -----------
        system_sizes : list
            List of system sizes
        pinn_times : dict
            PINN inference times (size -> time)
        tds_times : dict
            TDS simulation times (size -> time)

        Returns:
        --------
        comparison : dict
            Comparison results
        """
        comparison = {"speedup": {}, "relative_speedup": {}}

        for size in system_sizes:
            if size in pinn_times and size in tds_times:
                speedup = tds_times[size] / pinn_times[size] if pinn_times[size] > 0 else 0.0
                comparison["speedup"][size] = speedup
                comparison["relative_speedup"][size] = (speedup - 1.0) * 100

        return comparison

    def generate_report(self, filename: Optional[str] = None) -> str:
        """
        Generate scalability analysis report.

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
        report += "SCALABILITY ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"

        # Training time analysis
        if "training_time" in self.results:
            report += "TRAINING TIME ANALYSIS\n"
            report += "-" * 80 + "\n"
            for size, timing in sorted(self.results["training_time"].items()):
                report += "\nSystem Size: {size}\n"
                report += "  Total time: {timing['total_time']:.2f}s\n"
                report += "  Time per epoch: {timing['time_per_epoch']:.4f}s\n"
            report += "\n"

        # Inference time analysis
        if "inference_time" in self.results:
            report += "\n" + "=" * 80 + "\n"
            report += "INFERENCE TIME ANALYSIS\n"
            report += "-" * 80 + "\n"
            for size, timing in sorted(self.results["inference_time"].items()):
                report += "\nSystem Size: {size}\n"
                report += "  Mean time: {timing['mean_time']*1000:.2f}ms\n"
                report += "  Std time: {timing['std_time']*1000:.2f}ms\n"
                report += "  Median time: {timing['median_time']*1000:.2f}ms\n"
                report += "  P95 time: {timing['p95_time']*1000:.2f}ms\n"
            report += "\n"

        # Memory usage analysis
        if "memory_usage" in self.results:
            report += "\n" + "=" * 80 + "\n"
            report += "MEMORY USAGE ANALYSIS\n"
            report += "-" * 80 + "\n"
            for size, memory in sorted(self.results["memory_usage"].items()):
                report += "\nSystem Size: {size}\n"
                report += "  Parameters: {memory['n_parameters']:,}\n"
                report += "  Parameter memory: {memory['parameter_memory_mb']:.2f} MB\n"
                report += "  Total memory: {memory['total_memory_mb']:.2f} MB\n"
            report += "\n"

        # Complexity analysis
        if "complexity" in self.results:
            report += "\n" + "=" * 80 + "\n"
            report += "COMPUTATIONAL COMPLEXITY\n"
            report += "-" * 80 + "\n"
            for size, complexity in sorted(self.results["complexity"].items()):
                report += "\nSystem Size: {size}\n"
                report += "  Parameters: {complexity['n_parameters']:,}\n"
                report += "  Parameters per generator: {complexity['params_per_generator']:.2f}\n"
            report += "\n"

        if filename:
            filepath = self.output_dir / filename
            with open(filepath, "w") as f:
                f.write(report)
            print("Report saved to {filepath}")

        return report

    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            filename = "scalability_results.json"

        filepath = self.output_dir / filename

        # Convert to JSON-serializable format
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    str(k): {
                        sk: float(sv) if isinstance(sv, (int, float, np.number)) else sv
                        for sk, sv in sv_dict.items()
                    }
                    for k, sv_dict in value.items()
                }

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print("Results saved to {filepath}")


def main():
    """Example usage of scalability analyzer."""
    analyzer = ScalabilityAnalyzer()

    print("Scalability analysis framework ready!")
    print("Use ScalabilityAnalyzer class to analyze model scalability.")


if __name__ == "__main__":
    main()
