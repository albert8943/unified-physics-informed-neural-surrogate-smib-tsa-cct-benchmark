"""
Evaluation Framework for PINN Models.

This package provides comprehensive evaluation tools including:
- Baseline comparison methods (EAC, TDS, standard ML)
- Ablation study framework
- Statistical analysis tools
- Scalability analysis
"""

from .ablation_studies import (
    AblationStudy,
    ArchitectureAblation,
    CollocationAblation,
    PhysicsLossAblation,
)
from .baseline_comparison import BaselineComparator, EACBaseline, MLBaseline, TDSBaseline

__all__ = [
    "BaselineComparator",
    "EACBaseline",
    "TDSBaseline",
    "MLBaseline",
    "AblationStudy",
    "PhysicsLossAblation",
    "ArchitectureAblation",
    "CollocationAblation",
]
