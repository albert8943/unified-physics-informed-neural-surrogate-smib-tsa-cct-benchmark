"""
Computational Speed Benchmarking.

Compares inference time and computational speed across methods.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch


def benchmark_inference_time(
    model,
    test_inputs: torch.Tensor,
    n_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu",
) -> Dict:
    """
    Benchmark inference time for a model.

    Parameters:
    -----------
    model : nn.Module
        Model to benchmark
    test_inputs : torch.Tensor
        Test input tensor
    n_runs : int
        Number of benchmark runs
    warmup_runs : int
        Number of warmup runs
    device : str
        Device to use

    Returns:
    --------
    results : dict
        Benchmark results
    """
    model.eval()
    model = model.to(device)
    test_inputs = test_inputs.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(test_inputs)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(test_inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds

    times = np.array(times)

    results = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "n_runs": n_runs,
    }

    return results


def compare_speed(
    methods: Dict[str, any],
    test_inputs: Dict[str, torch.Tensor],
    n_runs: int = 100,
) -> Dict:
    """
    Compare speed across multiple methods.

    Parameters:
    -----------
    methods : dict
        Dictionary mapping method name to model/function
    test_inputs : dict
        Dictionary mapping method name to test inputs
    n_runs : int
        Number of benchmark runs

    Returns:
    --------
    comparison : dict
        Speed comparison results
    """
    comparison = {}

    for method_name, model in methods.items():
        inputs = test_inputs.get(method_name)
        if inputs is None:
            continue

        try:
            results = benchmark_inference_time(model, inputs, n_runs=n_runs)
            comparison[method_name] = results
        except Exception as e:
            print(f"Warning: Failed to benchmark {method_name}: {e}")
            comparison[method_name] = {"error": str(e)}

    return comparison


def compute_speedup(
    baseline_time_ms: float,
    comparison_time_ms: float,
) -> float:
    """
    Compute speedup factor.

    Parameters:
    -----------
    baseline_time_ms : float
        Baseline method time (milliseconds)
    comparison_time_ms : float
        Comparison method time (milliseconds)

    Returns:
    --------
    speedup : float
        Speedup factor (baseline / comparison)
    """
    if comparison_time_ms == 0:
        return float("inf")
    return baseline_time_ms / comparison_time_ms
