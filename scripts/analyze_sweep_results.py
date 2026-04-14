#!/usr/bin/env python
"""
Analyze comprehensive hyperparameter sweep results.

Extracts metrics from all sweep experiments and finds optimal configuration.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_sweep(sweep_n_samples=None, sweep_epochs=None, sweep_dir=None):
    """
    Analyze hyperparameter sweep results.

    Parameters:
    -----------
    sweep_n_samples : list, optional
        List of n_samples values in the sweep (e.g., [50, 60, 70, 80, 90, 100])
    sweep_epochs : list, optional
        List of epochs values in the sweep (e.g., [500, 600, 700, 800, 900, 1000])
    sweep_dir : str or Path, optional
        Directory containing sweep experiments (default: "outputs/experiments")
    """
    results = []

    print("=" * 70)
    print("ANALYZING SWEEP RESULTS")
    print("=" * 70)
    print("Scanning experiments...")

    # Determine directory to scan
    if sweep_dir is None:
        sweep_dir = Path("outputs/experiments")
    else:
        sweep_dir = Path(sweep_dir)

    if not sweep_dir.exists():
        print(f"[ERROR] Directory not found: {sweep_dir}")
        return

    for exp_dir in sorted(sweep_dir.glob("exp_*")):
        summary = exp_dir / "experiment_summary.json"
        if not summary.exists():
            continue

        try:
            data = json.load(open(summary))

            # Try to get config from summary, or load from config.yaml
            config = data.get("config", {})
            if not config:
                # Try loading from config.yaml in experiment directory
                config_path = exp_dir / "config.yaml"
                if config_path.exists():
                    import yaml

                    with open(config_path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                else:
                    # If no config found, skip this experiment
                    continue

            # Try multiple paths for metrics
            metrics = None
            if (
                "pinn" in data
                and "evaluation" in data["pinn"]
                and "metrics" in data["pinn"]["evaluation"]
            ):
                metrics = data["pinn"]["evaluation"]["metrics"]
            elif "evaluation" in data and "metrics" in data["evaluation"]:
                metrics = data["evaluation"]["metrics"]
            elif "results" in data and "metrics" in data["results"]:
                if "metrics" in data["results"]["metrics"]:
                    metrics = data["results"]["metrics"]["metrics"]
                else:
                    metrics = data["results"]["metrics"]
            elif "metrics" in data:
                if "metrics" in data["metrics"]:
                    metrics = data["metrics"]["metrics"]
                else:
                    metrics = data["metrics"]

            if not metrics:
                continue

            n_samples = config.get("data", {}).get("generation", {}).get("n_samples")
            epochs = config.get("training", {}).get("epochs")
            hidden_dims = config.get("model", {}).get("hidden_dims", [])

            if n_samples is None or epochs is None:
                continue

            # Filter to sweep experiments if specified
            if sweep_n_samples and n_samples not in sweep_n_samples:
                continue
            if sweep_epochs and epochs not in sweep_epochs:
                continue

            results.append(
                {
                    "experiment_id": exp_dir.name,
                    "n_samples": n_samples,
                    "epochs": epochs,
                    "hidden_dims": str(hidden_dims),
                    "rmse_delta": metrics.get("rmse_delta"),
                    "r2_delta": metrics.get("r2_delta"),
                    "rmse_omega": metrics.get("rmse_omega"),
                    "r2_omega": metrics.get("r2_omega"),
                    "mae_delta": metrics.get("mae_delta"),
                    "mae_omega": metrics.get("mae_omega"),
                }
            )
        except Exception as e:
            continue

    if not results:
        print("\n[WARNING] No sweep results found.")
        print("  Make sure experiments have completed and experiment_summary.json exists.")
        return

    df = pd.DataFrame(results)

    print(f"\n[OK] Found {len(df)} experiments")

    # Filter valid results
    df_valid = df[
        (df["r2_delta"].notna())
        & (df["r2_omega"].notna())
        & (df["rmse_delta"] > 0.0)
        & (df["r2_delta"] > 0.0)
    ].copy()

    if len(df_valid) == 0:
        print("\n[ERROR] No valid results found")
        return

    print(f"[OK] {len(df_valid)} experiments with valid metrics")

    # Best overall (highest R² Omega while maintaining good R² Delta)
    print("\n" + "=" * 70)
    print("BEST OVERALL CONFIGURATION")
    print("=" * 70)

    # Filter to good delta predictions (R² > 0.85)
    df_good = df_valid[df_valid["r2_delta"] > 0.85].copy()

    if len(df_good) > 0:
        best_overall = df_good.loc[df_good["r2_omega"].idxmax()]
        print(f"n_samples: {best_overall['n_samples']}")
        print(f"epochs: {best_overall['epochs']}")
        print(f"Network: {best_overall['hidden_dims']}")
        print(f"R² Delta: {best_overall['r2_delta']:.4f}")
        print(f"R² Omega: {best_overall['r2_omega']:.4f}")
        print(f"RMSE Delta: {best_overall['rmse_delta']:.4f} rad")
        print(f"RMSE Omega: {best_overall['rmse_omega']:.4f} pu")
        print(f"Experiment: {best_overall['experiment_id']}")
    else:
        # Fallback: best overall regardless of delta
        best_overall = df_valid.loc[df_valid["r2_omega"].idxmax()]
        print(f"n_samples: {best_overall['n_samples']}")
        print(f"epochs: {best_overall['epochs']}")
        print(f"Network: {best_overall['hidden_dims']}")
        print(f"R² Delta: {best_overall['r2_delta']:.4f}")
        print(f"R² Omega: {best_overall['r2_omega']:.4f}")
        print(f"RMSE Delta: {best_overall['rmse_delta']:.4f} rad")
        print(f"RMSE Omega: {best_overall['rmse_omega']:.4f} pu")
        print(f"Experiment: {best_overall['experiment_id']}")

    # Best by n_samples
    if "n_samples" in df_valid.columns:
        print("\n" + "=" * 70)
        print("BEST BY n_samples")
        print("=" * 70)
        for n_samples in sorted(df_valid["n_samples"].unique()):
            subset = df_valid[df_valid["n_samples"] == n_samples]
            if len(subset) > 0:
                best_n = subset.loc[subset["r2_omega"].idxmax()]
                print(
                    f"n_samples={n_samples:3d}: R² Omega={best_n['r2_omega']:.4f}, "
                    f"R² Delta={best_n['r2_delta']:.4f}, epochs={best_n['epochs']}, "
                    f"RMSE Delta={best_n['rmse_delta']:.4f}"
                )

    # Best by epochs
    if "epochs" in df_valid.columns:
        print("\n" + "=" * 70)
        print("BEST BY epochs")
        print("=" * 70)
        for epochs in sorted(df_valid["epochs"].unique()):
            subset = df_valid[df_valid["epochs"] == epochs]
            if len(subset) > 0:
                best_e = subset.loc[subset["r2_omega"].idxmax()]
                print(
                    f"epochs={epochs:4d}: R² Omega={best_e['r2_omega']:.4f}, "
                    f"R² Delta={best_e['r2_delta']:.4f}, n_samples={best_e['n_samples']}, "
                    f"RMSE Delta={best_e['rmse_delta']:.4f}"
                )

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"R² Omega range: {df_valid['r2_omega'].min():.4f} - {df_valid['r2_omega'].max():.4f}")
    print(f"R² Delta range: {df_valid['r2_delta'].min():.4f} - {df_valid['r2_delta'].max():.4f}")
    print(
        f"RMSE Delta range: {df_valid['rmse_delta'].min():.4f} - {df_valid['rmse_delta'].max():.4f}"
        f"rad"
    )
    print(
        f"RMSE Omega range: {df_valid['rmse_omega'].min():.4f} - {df_valid['rmse_omega'].max():.4f}"
        f"pu"
    )

    # Save to CSV
    output_dir = Path("outputs/sweep_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    from scripts.core.utils import generate_timestamped_filename

    csv_path = output_dir / generate_timestamped_filename("sweep_analysis", "csv")
    df_valid.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to: {csv_path}")

    # Create pivot table for easy comparison
    if len(df_valid) > 0 and "n_samples" in df_valid.columns and "epochs" in df_valid.columns:
        pivot_r2_omega = df_valid.pivot_table(
            values="r2_omega", index="n_samples", columns="epochs", aggfunc="max"
        )
        pivot_path = output_dir / generate_timestamped_filename("sweep_pivot_r2_omega", "csv")
        pivot_r2_omega.to_csv(pivot_path)
        print(f"[OK] R² Omega pivot table saved to: {pivot_path}")

        pivot_r2_delta = df_valid.pivot_table(
            values="r2_delta", index="n_samples", columns="epochs", aggfunc="max"
        )
        pivot_path2 = output_dir / generate_timestamped_filename("sweep_pivot_r2_delta", "csv")
        pivot_r2_delta.to_csv(pivot_path2)
        print(f"[OK] R² Delta pivot table saved to: {pivot_path2}")

    # Generate summary report
    summary_path = generate_sweep_summary_report(
        df_valid=df_valid,
        best_overall=best_overall,
        output_dir=output_dir,
        sweep_dir=sweep_dir,
        csv_path=csv_path,
        pivot_r2_omega_path=pivot_path
        if len(df_valid) > 0 and "n_samples" in df_valid.columns and "epochs" in df_valid.columns
        else None,
        pivot_r2_delta_path=pivot_path2
        if len(df_valid) > 0 and "n_samples" in df_valid.columns and "epochs" in df_valid.columns
        else None,
    )
    print(f"[OK] Summary report saved to: {summary_path}")

    return df_valid, best_overall


def generate_sweep_summary_report(
    df_valid: pd.DataFrame,
    best_overall: pd.Series,
    output_dir: Path,
    sweep_dir: Path,
    csv_path: Path,
    pivot_r2_omega_path: Path = None,
    pivot_r2_delta_path: Path = None,
) -> Path:
    """
    Generate a formatted markdown summary report for the sweep analysis.

    Parameters:
    -----------
    df_valid : pd.DataFrame
        DataFrame with valid experiment results
    best_overall : pd.Series
        Best overall configuration
    output_dir : Path
        Output directory for the report
    sweep_dir : Path
        Directory containing sweep experiments
    csv_path : Path
        Path to the CSV analysis file
    pivot_r2_omega_path : Path, optional
        Path to R² Omega pivot table
    pivot_r2_delta_path : Path, optional
        Path to R² Delta pivot table

    Returns:
    --------
    Path
        Path to generated summary report
    """
    from scripts.core.utils import generate_timestamped_filename

    report_path = output_dir / generate_timestamped_filename("sweep_summary", "md")

    # Get best by n_samples
    best_by_n_samples = []
    if "n_samples" in df_valid.columns:
        for n_samples in sorted(df_valid["n_samples"].unique()):
            subset = df_valid[df_valid["n_samples"] == n_samples]
            if len(subset) > 0:
                best_n = subset.loc[subset["r2_omega"].idxmax()]
                best_by_n_samples.append(
                    {
                        "n_samples": n_samples,
                        "r2_omega": best_n["r2_omega"],
                        "r2_delta": best_n["r2_delta"],
                        "epochs": best_n["epochs"],
                        "rmse_delta": best_n["rmse_delta"],
                    }
                )

    # Get best by epochs
    best_by_epochs = []
    if "epochs" in df_valid.columns:
        for epochs in sorted(df_valid["epochs"].unique()):
            subset = df_valid[df_valid["epochs"] == epochs]
            if len(subset) > 0:
                best_e = subset.loc[subset["r2_omega"].idxmax()]
                best_by_epochs.append(
                    {
                        "epochs": epochs,
                        "r2_omega": best_e["r2_omega"],
                        "r2_delta": best_e["r2_delta"],
                        "n_samples": best_e["n_samples"],
                        "rmse_delta": best_e["rmse_delta"],
                    }
                )

    # Generate markdown content
    md_content = f"""# Hyperparameter Sweep Analysis Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Sweep Directory**: `{sweep_dir}`  
**Total Experiments Analyzed**: {len(df_valid)}

---

## 🎯 Best Overall Configuration

- **n_samples**: {best_overall['n_samples']}
- **epochs**: {best_overall['epochs']}
- **Network Architecture**: {best_overall['hidden_dims']}
- **R² Omega**: {best_overall['r2_omega']:.4f}
- **R² Delta**: {best_overall['r2_delta']:.4f}
- **RMSE Delta**: {best_overall['rmse_delta']:.4f} rad
- **RMSE Omega**: {best_overall['rmse_omega']:.4f} pu
- **MAE Delta**: {best_overall['mae_delta']:.4f} rad
- **MAE Omega**: {best_overall['mae_omega']:.4f} pu
- **Experiment ID**: `{best_overall['experiment_id']}`

---

## 📊 Findings

### Best by n_samples

"""

    for item in best_by_n_samples:
        md_content += f"- **n_samples={item['n_samples']}**: R² Omega={item['r2_omega']:.4f}, R² Delta={item['r2_delta']:.4f}, epochs={item['epochs']}, RMSE Delta={item['rmse_delta']:.4f} rad\n"

    md_content += "\n### Best by epochs\n\n"

    for item in best_by_epochs:
        md_content += f"- **epochs={item['epochs']}**: R² Omega={item['r2_omega']:.4f}, R² Delta={item['r2_delta']:.4f}, n_samples={item['n_samples']}, RMSE Delta={item['rmse_delta']:.4f} rad\n"

    # Check if epochs 300, 400, 500 all achieve same result
    convergence_note = ""
    if len(best_by_epochs) >= 3:
        epochs_300_400_500 = [item for item in best_by_epochs if item["epochs"] in [300, 400, 500]]
        if len(epochs_300_400_500) == 3:
            r2_values = [item["r2_omega"] for item in epochs_300_400_500]
            if len(set([round(v, 4) for v in r2_values])) == 1:
                convergence_note = "\n> **Note**: Epochs 300, 400, and 500 all achieve the same best result, suggesting convergence around 300 epochs.\n"

    md_content += f"""
---

## 📈 Performance Range

- **R² Omega**: {df_valid['r2_omega'].min():.4f} - {df_valid['r2_omega'].max():.4f}
- **R² Delta**: {df_valid['r2_delta'].min():.4f} - {df_valid['r2_delta'].max():.4f}
- **RMSE Delta**: {df_valid['rmse_delta'].min():.4f} - {df_valid['rmse_delta'].max():.4f} rad
- **RMSE Omega**: {df_valid['rmse_omega'].min():.4f} - {df_valid['rmse_omega'].max():.4f} pu

{convergence_note}
---

## 📁 Generated Files

- **Analysis CSV**: `{csv_path.name}`
"""

    if pivot_r2_omega_path:
        md_content += f"- **R² Omega Pivot Table**: `{pivot_r2_omega_path.name}`\n"
    if pivot_r2_delta_path:
        md_content += f"- **R² Delta Pivot Table**: `{pivot_r2_delta_path.name}`\n"

    md_content += f"""
- **Summary Report**: `{report_path.name}`

---

## 🔍 Analysis Details

### Hyperparameter Ranges Tested

- **n_samples**: {', '.join(map(str, sorted(df_valid['n_samples'].unique())))}
- **epochs**: {', '.join(map(str, sorted(df_valid['epochs'].unique())))}

### Key Insights

1. **Optimal Configuration**: n_samples={best_overall['n_samples']}, epochs={best_overall['epochs']}
2. **Best Performance**: R² Omega = {best_overall['r2_omega']:.4f} (experiment `{best_overall['experiment_id']}`)
3. **Performance Range**: R² Omega spans {df_valid['r2_omega'].max() - df_valid['r2_omega'].min():.4f} across all configurations

---

**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Generated Automatically**: This file was generated by the sweep analysis script.
"""

    # Write to file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return report_path


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results")
    parser.add_argument(
        "--n-samples",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific n_samples values (e.g., 50 60 70)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific epochs values (e.g., 500 600 700)",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Directory containing sweep experiments (default: outputs/experiments)",
    )

    args = parser.parse_args()

    analyze_sweep(
        sweep_n_samples=args.n_samples, sweep_epochs=args.epochs, sweep_dir=args.sweep_dir
    )


if __name__ == "__main__":
    main()
