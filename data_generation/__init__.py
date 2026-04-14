"""
Data Generation Module for PINN Training.

This module handles data extraction from ANDES simulations,
parameter sweeps, and data preprocessing for PINN training.

Submodules:
- andes_utils: ANDES-specific utility modules for data generation
  (CCT finding, data extraction, validation, export)
"""

from .andes_extractor import extract_system_reactances, extract_trajectories
from .andes_utils.data_validator import validate_data_quality
from .parameter_sweep import (
    generate_cct_data,
    generate_multi_task_data,
    generate_parameter_estimation_data,
    generate_parameter_sweep,
    generate_trajectory_data,
)
from .preprocessing import normalize_data, preprocess_data, split_dataset
from .sampling_strategies import (
    boundary_focused_sample,
    correlation_analysis,
    latin_hypercube_sample,
    sobol_sequence_sample,
    validate_sample_quality,
)
from .validation import (
    analyze_parameter_coverage,
    detect_stability_boundary,
    estimate_cct_from_data,
    generate_data_quality_report,
    generate_validation_report,
    track_quality_incremental,
    validate_cct_data_quality,
    validate_data_for_task,
)

__all__ = [
    "extract_trajectories",
    "extract_system_reactances",
    "generate_parameter_sweep",
    "generate_trajectory_data",
    "generate_parameter_estimation_data",
    "generate_cct_data",
    "generate_multi_task_data",
    "preprocess_data",
    "normalize_data",
    "split_dataset",
    "latin_hypercube_sample",
    "sobol_sequence_sample",
    "boundary_focused_sample",
    "correlation_analysis",
    "validate_sample_quality",
    "detect_stability_boundary",
    "estimate_cct_from_data",
    "validate_cct_data_quality",
    "analyze_parameter_coverage",
    "validate_data_for_task",
    "validate_data_quality",
    "generate_validation_report",
    "generate_data_quality_report",
    "track_quality_incremental",
]
