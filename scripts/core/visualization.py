"""
Visualization Core Module.

This module integrates visualization generation into the experiment workflow.
Used by main scripts and run_experiment.py.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.publication_figures import generate_experiment_figures

from .utils import generate_timestamped_filename, save_json
from datetime import datetime


def generate_experiment_figures_wrapper(
    config: Dict,
    data_path: Optional[Path] = None,
    training_history: Optional[Dict] = None,
    evaluation_results: Optional[Dict] = None,
    output_dir: Path = Path("outputs/figures"),
) -> Dict[str, Path]:
    """
    Generate all publication-quality figures for an experiment.

    This is a wrapper around the visualization module that integrates
    with the workflow system.

    Parameters:
    -----------
    config : dict
        Experiment configuration
    data_path : Path, optional
        Path to training data CSV
    training_history : dict, optional
        Training history with losses and metrics
    evaluation_results : dict, optional
        Evaluation results with predictions and metrics
    output_dir : Path
        Directory to save figures

    Returns:
    --------
    figure_paths : dict
        Dictionary mapping figure names to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    figure_paths = generate_experiment_figures(
        config=config,
        data_path=data_path,
        training_history=training_history,
        evaluation_results=evaluation_results,
        output_dir=output_dir,
        figure_formats=["png"],  # Default: PNG only (PDF can be added if needed)
        dpi=300,
    )

    # Verify figures were actually saved
    saved_figures = list(output_dir.glob("*.png"))
    print(f"\n✓ Generated {len(figure_paths)} figure paths")
    print(f"  Actually saved: {len(saved_figures)} PNG files in {output_dir}")
    if len(saved_figures) > 0:
        print(f"  Saved files:")
        for fig in sorted(saved_figures):
            size = fig.stat().st_size if fig.exists() else 0
            print(f"    • {fig.name} ({size} bytes)")
    else:
        print(f"  ⚠️  WARNING: No PNG files found in {output_dir}")
        print(f"     This may indicate a save failure. Check for errors above.")

    # Save figure metadata with timestamp
    metadata = {
        "n_figures": len(figure_paths),
        "n_actual_files": len(saved_figures),
        "figure_paths": {name: str(path) for name, path in figure_paths.items()},
        "output_dir": str(output_dir),
    }
    metadata_filename = generate_timestamped_filename("figure_metadata", "json")
    metadata_path = output_dir / metadata_filename
    save_json(metadata, metadata_path)

    print(f"  Metadata: {metadata_path}")

    return figure_paths
