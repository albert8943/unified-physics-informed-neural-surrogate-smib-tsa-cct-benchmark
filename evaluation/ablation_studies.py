"""
Ablation Study Framework for PINN Models.

This module provides tools for conducting systematic ablation studies:
- Physics loss weighting impact
- Architecture variations
- Collocation point strategies
- State-aware modeling effectiveness
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class AblationStudy:
    """
    Base class for ablation studies.

    Provides common functionality for running systematic experiments
    and collecting results.
    """

    def __init__(self, name: str, output_dir: Optional[str] = None):
        """
        Initialize ablation study.

        Parameters:
        -----------
        name : str
            Study name
        output_dir : str, optional
            Output directory for results
        """
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/ablation_studies")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.configs = []

    def run_study(
        self, configs: List[Dict], train_fn: callable, eval_fn: callable, test_data: Any
    ) -> Dict:
        """
        Run ablation study with multiple configurations.

        Parameters:
        -----------
        configs : list
            List of configuration dictionaries
        train_fn : callable
            Training function that takes config and returns model
        eval_fn : callable
            Evaluation function that takes model and test_data, returns metrics
        test_data : any
            Test dataset

        Returns:
        --------
        results : dict
            Dictionary of results for each configuration
        """
        all_results = {}

        for i, config in enumerate(configs):
            print("\n{'='*80}")
            print("Running configuration {i+1}/{len(configs)}")
            print("Config: {config}")
            print("{'='*80}\n")

            start_time = time.time()

            # Train model with this configuration
            model = train_fn(config)

            # Evaluate model
            metrics = eval_fn(model, test_data)

            elapsed_time = time.time() - start_time

            # Store results
            result = {"config": config, "metrics": metrics, "training_time": elapsed_time}

            all_results["config_{i+1}"] = result
            self.results.append(result)
            self.configs.append(config)

            print("\nResults for config {i+1}:")
            for key, value in metrics.items():
                print("  {key}: {value:.6f}")
            print("  Training time: {elapsed_time:.2f}s\n")

        return all_results

    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            filename = "{self.name}_results.json"

        filepath = self.output_dir / filename

        # Convert results to JSON-serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {
                "config": result["config"],
                "metrics": {k: float(v) for k, v in result["metrics"].items()},
                "training_time": float(result["training_time"]),
            }
            serializable_results.append(serializable_result)

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print("Results saved to {filepath}")

    def generate_report(self, filename: Optional[str] = None) -> str:
        """
        Generate text report of ablation study results.

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
        report += "ABLATION STUDY: {self.name}\n"
        report += "=" * 80 + "\n\n"

        for i, result in enumerate(self.results):
            report += "Configuration {i+1}:\n"
            report += "-" * 80 + "\n"
            report += "Config: {json.dumps(result['config'], indent=2)}\n\n"
            report += "Metrics:\n"
            for key, value in result["metrics"].items():
                report += "  {key}: {value:.6f}\n"
            report += "Training time: {result['training_time']:.2f}s\n\n"

        if filename:
            filepath = self.output_dir / filename
            with open(filepath, "w") as f:
                f.write(report)

        return report


class PhysicsLossAblation(AblationStudy):
    """
    Ablation study for physics loss weighting impact.

    Tests different values of lambda_physics to understand its impact
    on model performance.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize physics loss ablation study."""
        super().__init__("PhysicsLossAblation", output_dir)

    def generate_configs(self, lambda_physics_values: List[float], base_config: Dict) -> List[Dict]:
        """
        Generate configurations with different physics loss weights.

        Parameters:
        -----------
        lambda_physics_values : list
            List of lambda_physics values to test
        base_config : dict
            Base configuration dictionary

        Returns:
        --------
        configs : list
            List of configuration dictionaries
        """
        configs = []

        for lambda_physics in lambda_physics_values:
            config = base_config.copy()
            config["lambda_physics"] = lambda_physics
            config["name"] = "lambda_physics_{lambda_physics}"
            configs.append(config)

        return configs


class ArchitectureAblation(AblationStudy):
    """
    Ablation study for architecture variations.

    Tests different network architectures (depth, width, activation, etc.)
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize architecture ablation study."""
        super().__init__("ArchitectureAblation", output_dir)

    def generate_configs(
        self,
        hidden_dims_variations: List[List[int]],
        activation_variations: List[str],
        use_residual_variations: List[bool],
        base_config: Dict,
    ) -> List[Dict]:
        """
        Generate configurations with different architectures.

        Parameters:
        -----------
        hidden_dims_variations : list
            List of hidden dimension configurations
        activation_variations : list
            List of activation functions to test
        use_residual_variations : list
            List of residual connection options
        base_config : dict
            Base configuration dictionary

        Returns:
        --------
        configs : list
            List of configuration dictionaries
        """
        configs = []

        for hidden_dims in hidden_dims_variations:
            for activation in activation_variations:
                for use_residual in use_residual_variations:
                    config = base_config.copy()
                    config["hidden_dims"] = hidden_dims
                    config["activation"] = activation
                    config["use_residual"] = use_residual
                    config["name"] = (
                        "dims_{len(hidden_dims)}_"
                        "width_{hidden_dims[0] if hidden_dims else 0}_"
                        "act_{activation}_"
                        "res_{use_residual}"
                    )
                    configs.append(config)

        return configs


class CollocationAblation(AblationStudy):
    """
    Ablation study for collocation point strategies.

    Tests different collocation point selection methods and densities.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize collocation ablation study."""
        super().__init__("CollocationAblation", output_dir)

    def generate_configs(
        self, n_colloc_points: List[int], colloc_strategies: List[str], base_config: Dict
    ) -> List[Dict]:
        """
        Generate configurations with different collocation strategies.

        Parameters:
        -----------
        n_colloc_points : list
            List of number of collocation points to test
        colloc_strategies : list
            List of collocation strategies ('uniform', 'random', 'adaptive')
        base_config : dict
            Base configuration dictionary

        Returns:
        --------
        configs : list
            List of configuration dictionaries
        """
        configs = []

        for n_colloc in n_colloc_points:
            for strategy in colloc_strategies:
                config = base_config.copy()
                config["n_colloc_points"] = n_colloc
                config["colloc_strategy"] = strategy
                config["name"] = "n_colloc_{n_colloc}_strategy_{strategy}"
                configs.append(config)

        return configs


class StateAwareAblation(AblationStudy):
    """
    Ablation study for state-aware modeling effectiveness.

    Compares models with and without state-aware reactance switching.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize state-aware ablation study."""
        super().__init__("StateAwareAblation", output_dir)

    def generate_configs(self, state_aware_options: List[bool], base_config: Dict) -> List[Dict]:
        """
        Generate configurations with/without state-aware modeling.

        Parameters:
        -----------
        state_aware_options : list
            List of boolean values (True/False for state-aware)
        base_config : dict
            Base configuration dictionary

        Returns:
        --------
        configs : list
            List of configuration dictionaries
        """
        configs = []

        for state_aware in state_aware_options:
            config = base_config.copy()
            config["state_aware"] = state_aware
            config["name"] = "state_aware_{state_aware}"
            configs.append(config)

        return configs
