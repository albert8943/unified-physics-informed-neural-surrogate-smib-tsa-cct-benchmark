#!/usr/bin/env python
"""
Data Preprocessing Script

Preprocess and split data into train/validation/test sets.
This is a critical step in the ML workflow that should be explicit.

Usage:
    # Basic preprocessing and splitting
    python scripts/preprocess_data.py --data-path data/generated/quick_test/parameter_sweep_data_*.csv

    # Multimachine (Kundur) data: same workflow; columns are mapped automatically (delta_0 -> delta, etc.)
    python scripts/preprocess_data.py --data-path data/multimachine/kundur/exp_*/parameter_sweep_data_*.csv

    # Custom split ratios
    python scripts/preprocess_data.py --data-path data/.../file.csv --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

    # With feature engineering
    python scripts/preprocess_data.py --data-path data/.../file.csv --engineer-features

    # With rotor angle filtering (same as SMIB: limit training angles, 180 deg stability)
    python scripts/preprocess_data.py --data-path data/.../file.csv --filter-angles --max-angle-deg 360
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from data_generation.preprocessing import (
    engineer_features,
    map_multimachine_to_smib_columns,
    normalize_data,
    preprocess_data,
    split_dataset,
)
from scripts.core.utils import generate_timestamped_filename, save_json, generate_experiment_id

try:
    from utils.angle_filter import filter_trajectory_by_angle

    ANGLE_FILTER_AVAILABLE = True
except ImportError:
    ANGLE_FILTER_AVAILABLE = False
    filter_trajectory_by_angle = None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and split data into train/validation/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing
  python scripts/preprocess_data.py --data-path data/generated/quick_test/parameter_sweep_data_*.csv
  
  # Custom split ratios
  python scripts/preprocess_data.py --data-path data/.../file.csv --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
  
  # With feature engineering
  python scripts/preprocess_data.py --data-path data/.../file.csv --engineer-features
  
  # Save to custom directory
  python scripts/preprocess_data.py --data-path data/.../file.csv --output-dir data/processed/my_experiment
        """,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input data CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed data (default: data/processed/{input_dir_name})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="scenario_id",
        help="Column to stratify by (default: scenario_id). Set to None for random split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize data (default: False, normalization handled in training)",
    )
    parser.add_argument(
        "--engineer-features",
        action="store_true",
        help="Engineer additional features (default: False)",
    )
    parser.add_argument(
        "--normalization-method",
        type=str,
        choices=["standard", "minmax"],
        default="standard",
        help="Normalization method: standard (z-score) or minmax (0-1) (default: standard)",
    )
    parser.add_argument(
        "--filter-angles",
        action="store_true",
        help="Filter trajectories by rotor angle (same as SMIB): keep |delta| < max_angle_deg, update stability by 180 deg",
    )
    parser.add_argument(
        "--max-angle-deg",
        type=float,
        default=360.0,
        help="Max rotor angle (deg) to keep when --filter-angles (default: 360)",
    )
    parser.add_argument(
        "--stability-threshold-deg",
        type=float,
        default=180.0,
        help="Stability threshold (deg) when --filter-angles (default: 180)",
    )

    args = parser.parse_args()

    # Load data - handle glob patterns and placeholders
    data_path_str = args.data_path

    # Check if user provided placeholder (common mistake)
    if "YYYYMMDD_HHMMSS" in data_path_str or "YYYYMMDD" in data_path_str:
        print(f"❌ Error: Placeholder detected in path: {data_path_str}")
        print(f"\n💡 The path contains 'YYYYMMDD_HHMMSS' which is a placeholder.")
        print(f"   Replace it with the actual timestamp from your generated file.")
        print(f"\n   Example:")
        print(f"   ❌ Wrong: data/generated/quick_test/parameter_sweep_data_YYYYMMDD_HHMMSS.csv")
        print(
            f"   ✅ Right: data/common/trajectory_data_1000_H2-10_D0.5-3_abc12345_20251205_170908.csv"
        )
        print(f"   ✅ Or use: data/common/trajectory_data_*.csv (wildcard)")
        print(f"\n   Or use a glob pattern to find the latest file:")
        print(f"   ✅ Right: data/generated/quick_test/parameter_sweep_data_*.csv")
        print(f"\n   To find your actual file, run:")
        print(f"   Get-ChildItem data/generated/quick_test/parameter_sweep_data_*.csv")
        sys.exit(1)

    # Try to resolve the path (handles glob patterns)
    data_path = Path(data_path_str)

    # Handle glob patterns
    if "*" in str(data_path) or "?" in str(data_path):
        # Expand glob pattern
        matches = list(data_path.parent.glob(data_path.name))
        if matches:
            # Use latest matching file
            data_path = max(matches, key=lambda p: p.stat().st_mtime)
            print(f"📂 Using latest matching file: {data_path.name}")
        else:
            print(f"❌ No files found matching pattern: {data_path_str}")
            print(f"\n💡 Available files in {data_path.parent}:")
            all_csv = list(data_path.parent.glob("*.csv"))
            if all_csv:
                for f in sorted(all_csv, key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                    print(f"   - {f.name}")
            else:
                print(f"   (no CSV files found)")
            sys.exit(1)

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print(f"\n💡 Available files in {data_path.parent}:")
        all_csv = list(data_path.parent.glob("*.csv"))
        if all_csv:
            for f in sorted(all_csv, key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                print(f"   - {f.name}")
        else:
            print(f"   (no CSV files found)")
        sys.exit(1)

    print("=" * 70)
    print("DATA PREPROCESSING & SPLITTING")
    print("=" * 70)
    print(f"Input: {data_path}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Stratify by: {args.stratify_by}")
    print(f"Random state: {args.random_state}")
    print()

    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(data):,} rows, {len(data.columns)} columns")
    print(f"  Unique scenarios: {data['scenario_id'].nunique()}")
    print()

    # Determine base output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        # Default: data/preprocessed/{input_dir_name}
        input_dir = data_path.parent.name
        base_output_dir = PROJECT_ROOT / "data" / "processed" / input_dir

    # Create timestamped experiment folder (format: exp_YYYYMMDD_HHMMSS)
    experiment_id = generate_experiment_id()  # Returns "exp_YYYYMMDD_HHMMSS"
    output_dir = base_output_dir / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base output directory: {base_output_dir}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Output directory: {output_dir}")
    print()

    # Preprocess data
    print("Preprocessing data...")
    df = data.copy()

    # Map multimachine columns to SMIB-style (delta, omega, etc.) so downstream steps see consistent columns
    df = map_multimachine_to_smib_columns(df)

    # Rotor angle filtering (same as SMIB workflow: truncate at max_angle_deg, update stability by 180 deg)
    filter_stats = None
    if getattr(args, "filter_angles", False):
        if not ANGLE_FILTER_AVAILABLE:
            print("  [WARNING] angle_filter module not available. Skipping angle filtering.")
        elif "delta" not in df.columns:
            print("  [WARNING] 'delta' column not found. Skipping angle filtering.")
        else:
            print(
                f"  Filtering trajectories: max_angle_deg={args.max_angle_deg}, stability_threshold_deg={args.stability_threshold_deg}"
            )
            df, filter_stats = filter_trajectory_by_angle(
                data=df,
                max_angle_deg=args.max_angle_deg,
                stability_threshold_deg=args.stability_threshold_deg,
            )
            if filter_stats:
                print(
                    f"  Filtered: {filter_stats.get('original_points', 0):,} -> {filter_stats.get('filtered_points', 0):,} points,"
                    f" {filter_stats.get('original_scenarios', 0)} -> {filter_stats.get('filtered_scenarios', 0)} scenarios"
                )

    # Feature engineering
    if args.engineer_features:
        print("  Engineering features...")
        df = engineer_features(df)
        print(f"  ✓ Features engineered: {len(df.columns)} columns")

    # Normalization (optional, usually handled in training)
    scaler_dict = None
    if args.normalize:
        print("  Normalizing data...")
        df, scaler_dict = normalize_data(df, method=args.normalization_method)
        print("  ✓ Data normalized")
    else:
        print("  ⚠️  Normalization skipped (will be handled during training)")

    # Split dataset
    print()
    print("Splitting dataset...")
    stratify_by = args.stratify_by if args.stratify_by != "None" else None

    train_data, val_data, test_data = split_dataset(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_by=stratify_by,
        random_state=args.random_state,
    )

    print(f"  ✓ Training set: {len(train_data):,} rows ({len(train_data)/len(df)*100:.1f}%)")
    print(f"  ✓ Validation set: {len(val_data):,} rows ({len(val_data)/len(df)*100:.1f}%)")
    print(f"  ✓ Test set: {len(test_data):,} rows ({len(test_data)/len(df)*100:.1f}%)")
    print()

    # Save splits
    print("Saving preprocessed data...")
    timestamp = generate_timestamped_filename("", "").replace("_", "").replace(".", "")

    train_path = output_dir / f"train_data_{timestamp}.csv"
    val_path = output_dir / f"val_data_{timestamp}.csv"
    test_path = output_dir / f"test_data_{timestamp}.csv"

    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"  ✓ Training data: {train_path}")
    print(f"  ✓ Validation data: {val_path}")
    print(f"  ✓ Test data: {test_path}")

    # Save metadata
    metadata = {
        "experiment_id": experiment_id,
        "input_file": str(data_path),
        "preprocessing": {
            "feature_engineering": args.engineer_features,
            "normalization": args.normalize,
            "normalization_method": args.normalization_method if args.normalize else None,
            "filter_angles": getattr(args, "filter_angles", False),
            "max_angle_deg": getattr(args, "max_angle_deg", 360.0),
            "stability_threshold_deg": getattr(args, "stability_threshold_deg", 180.0),
            "filter_stats": filter_stats,
        },
        "splitting": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "stratify_by": stratify_by,
            "random_state": args.random_state,
        },
        "statistics": {
            "total_rows": len(data),
            "train_rows": len(train_data),
            "val_rows": len(val_data),
            "test_rows": len(test_data),
            "train_scenarios": (
                train_data["scenario_id"].nunique() if "scenario_id" in train_data.columns else None
            ),
            "val_scenarios": (
                val_data["scenario_id"].nunique() if "scenario_id" in val_data.columns else None
            ),
            "test_scenarios": (
                test_data["scenario_id"].nunique() if "scenario_id" in test_data.columns else None
            ),
        },
        "output_files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }

    metadata_path = output_dir / f"preprocessing_metadata_{timestamp}.json"
    save_json(metadata, metadata_path)
    print(f"  ✓ Metadata: {metadata_path}")

    # Save scalers if normalized
    if scaler_dict:
        import pickle

        scalers_path = output_dir / f"scalers_{timestamp}.pkl"
        with open(scalers_path, "wb") as f:
            pickle.dump(scaler_dict, f)
        print(f"  ✓ Scalers: {scalers_path}")

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print()
    print(f"Experiment ID: {experiment_id}")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Verify splits: Check the saved CSV files in {output_dir}")
    print(f"  2. Train model: python scripts/train_model.py --data-path {train_path}")
    print(f"  3. Evaluate model: Use {test_path} for final evaluation")
    print()
    print(f"   Or use in complete experiment:")
    print(
        f"python scripts/run_complete_experiment.py --skip-data-generation --data-dir"
        f'"{output_dir.parent}"'
    )


if __name__ == "__main__":
    main()
