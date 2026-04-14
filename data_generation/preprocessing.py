"""
Data Preprocessing for PINN Training.

This module handles data normalization, feature engineering,
and train/validation/test splits.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Import angle filtering if available
try:
    from utils.angle_filter import filter_trajectory_by_angle

    ANGLE_FILTER_AVAILABLE = True
except ImportError:
    ANGLE_FILTER_AVAILABLE = False
    filter_trajectory_by_angle = None


def map_multimachine_to_smib_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Map multimachine (Kundur) parameter_sweep columns to SMIB-style names so the same
    preprocessing and training pipeline can be used.

    If the data already has a "delta" column, returns unchanged. Otherwise, if it has
    delta_0, omega_0 (and optionally delta_rel_*), adds: delta, omega, delta0, omega0,
    time (from time_s if needed), tf, tc, Pm so that preprocess_data and train_model
    work without change. Uses generator 0 as the representative machine (equivalent
    single-machine view for PINN training).
    """
    df = data.copy()
    if "delta" in df.columns:
        return df
    if "delta_0" not in df.columns or "omega_0" not in df.columns:
        return df
    # Time column
    if "time" not in df.columns and "time_s" in df.columns:
        df["time"] = df["time_s"]
    if "time" not in df.columns:
        return df
    # Representative machine: generator 0 (or use COI-relative for gen 0 if present)
    if "delta_rel_0" in df.columns and "omega_0" in df.columns:
        df["delta"] = df["delta_rel_0"].values
    else:
        df["delta"] = df["delta_0"].values
    df["omega"] = df["omega_0"].values
    # Per-scenario initial conditions (first time point per scenario)
    if "scenario_id" in df.columns:
        first = df.groupby("scenario_id")[["delta", "omega"]].first()
        df["delta0"] = df["scenario_id"].map(first["delta"])
        df["omega0"] = df["scenario_id"].map(first["omega"])
    else:
        df["delta0"] = df["delta"].iloc[0]
        df["omega0"] = df["omega"].iloc[0]
    # Fault timing (training expects tf, tc)
    if "tf" not in df.columns and "param_tf" in df.columns:
        df["tf"] = df["param_tf"]
    elif "tf" not in df.columns:
        df["tf"] = 1.0
    if "tc" not in df.columns and "param_tc" in df.columns:
        df["tc"] = df["param_tc"]
    # Pm for physics (training uses Pm; multimachine has param_load or param_Pm)
    if "Pm" not in df.columns:
        if "param_load" in df.columns:
            df["Pm"] = df["param_load"]
        elif "param_Pm" in df.columns:
            df["Pm"] = df["param_Pm"]
    # Pe for Pe-input PINN (single-machine view uses generator 0)
    if "Pe" not in df.columns and "Pe_0" in df.columns:
        df["Pe"] = df["Pe_0"]
    return df


def normalize_data(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "standard",
    scaler_dict: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize data columns.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    columns : list, optional
        Columns to normalize. If None, normalizes common PINN features.
    method : str
        Normalization method: 'standard' (z-score) or 'minmax' (0-1)
    scaler_dict : dict, optional
        Pre-fitted scalers. If None, fits new scalers.

    Returns:
    --------
    tuple : (normalized_data, scaler_dict)
    """
    if columns is None:
        # Default columns to normalize
        columns = [
            "time",
            "delta",
            "omega",
            "Pe",
            "Pm",
            "Xprefault",
            "Xfault",
            "Xpostfault",
            "M",
            "D",
            "H",
            "delta0",
            "omega0",
            "tf",
            "tc",
        ]

    # Filter to existing columns
    columns = [col for col in columns if col in data.columns]

    if scaler_dict is None:
        scaler_dict = {}

    normalized_data = data.copy()

    for col in columns:
        if col not in data.columns:
            continue

        if col not in scaler_dict:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError("Unknown normalization method: {method}")

            # Fit scaler
            values = data[col].values.reshape(-1, 1)
            scaler.fit(values)
            scaler_dict[col] = scaler
        else:
            scaler = scaler_dict[col]

        # Transform
        values = data[col].values.reshape(-1, 1)
        normalized_data[col] = scaler.transform(values).flatten()

    return normalized_data, scaler_dict


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features for PINN training.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data

    Returns:
    --------
    pd.DataFrame : Data with additional features
    """
    df = data.copy()

    # Time-based features
    if "time" in df.columns:
        df["time_squared"] = df["time"] ** 2
        df["time_normalized"] = df["time"] / (df["time"].max() + 1e-8)

    # Angular velocity (dδ/dt)
    if "delta" in df.columns and "time" in df.columns:
        df["ddelta_dt"] = np.gradient(df["delta"], df["time"])
        df["ddelta_dt_squared"] = df["ddelta_dt"] ** 2

    # Angular acceleration (d²δ/dt²)
    if "ddelta_dt" in df.columns and "time" in df.columns:
        df["d2delta_dt2"] = np.gradient(df["ddelta_dt"], df["time"])

    # Speed deviation features
    if "omega_deviation" in df.columns:
        df["omega_deviation_squared"] = df["omega_deviation"] ** 2
        df["abs_omega_deviation"] = np.abs(df["omega_deviation"])

    # Power imbalance
    if "Pm" in df.columns and "Pe" in df.columns:
        df["power_imbalance"] = df["Pm"] - df["Pe"]
        df["abs_power_imbalance"] = np.abs(df["power_imbalance"])

    # State-based features
    if "state" in df.columns:
        df["is_prefault"] = (df["state"] == 0).astype(int)
        df["is_during_fault"] = (df["state"] == 1).astype(int)
        df["is_postfault"] = (df["state"] == 2).astype(int)

    # Reactance ratio features
    if "Xprefault" in df.columns and "Xfault" in df.columns:
        df["Xratio_prefault_fault"] = df["Xprefault"] / (df["Xfault"] + 1e-8)

    if "Xprefault" in df.columns and "Xpostfault" in df.columns:
        df["Xratio_prefault_postfault"] = df["Xprefault"] / (df["Xpostfault"] + 1e-8)

    # Parameter interactions
    if "M" in df.columns and "D" in df.columns:
        df["M_D_ratio"] = df["M"] / (df["D"] + 1e-8)
        df["M_times_D"] = df["M"] * df["D"]

    # Fault timing features
    if "time" in df.columns and "tf" in df.columns and "tc" in df.columns:
        df["time_to_fault"] = df["tf"] - df["time"]
        df["time_from_fault"] = df["time"] - df["tf"]
        df["time_to_clear"] = df["tc"] - df["time"]
        df["fault_duration"] = df["tc"] - df["tf"]

    return df


def split_dataset(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: Optional[str] = None,
    stratify_by_stability: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation/test sets.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    test_ratio : float
        Test set ratio
    stratify_by : str, optional
        Column name to stratify by (e.g., 'scenario_id')
    stratify_by_stability : bool
        If True and "is_stable" in data, split so train/val/test each get both stable and unstable scenarios (recommended for multimachine/SMIB).
    random_state : int
        Random seed

    Returns:
    --------
    tuple : (train_data, val_data, test_data)
    """
    # Verify ratios sum to 1.
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # If stratifying by scenario, group by scenario_id
    use_scenario_split = stratify_by is not None and stratify_by in data.columns

    if use_scenario_split:
        # Get unique scenarios (ensure we work with unique values only)
        # This handles cases where data was combined and may have duplicate scenario_ids
        unique_scenarios = data[stratify_by].unique()
        n_scenarios = len(unique_scenarios)

        # Always print debug info for diagnosis
        print(f"[DEBUG] Starting scenario-based split:")
        print(f"  Total unique scenarios: {n_scenarios}")
        print(f"  Total rows in data: {len(data):,}")
        print(f"  Stratify by column: '{stratify_by}'")
        print(f"  Column dtype: {data[stratify_by].dtype}")
        if n_scenarios <= 20:  # Print all if small enough
            print(f"  Unique scenario IDs: {sorted(unique_scenarios)}")
        else:
            print(f"  Scenario ID range: {min(unique_scenarios)} - {max(unique_scenarios)}")

        # Check if we have enough scenarios to split
        if n_scenarios < 3:
            # Not enough scenarios to split into train/val/test
            # Use simple random split instead
            print(
                f"⚠️  Warning: Only {n_scenarios} unique scenario(s) found. "
                f"Using simple random split instead of scenario-based stratification."
            )
            use_scenario_split = False
        else:
            # Optionally stratify by is_stable so train/val/test each get both stable and unstable
            stratify_vec = None
            if stratify_by_stability and "is_stable" in data.columns:
                scenario_stability = data.groupby("scenario_id")["is_stable"].first()
                stratify_vec = scenario_stability.reindex(unique_scenarios).values
                if np.any(pd.isna(stratify_vec)) or len(np.unique(stratify_vec)) < 2:
                    stratify_vec = None
                else:
                    print(
                        f"  Stratifying by is_stable so train/val/test include both stable and unstable."
                    )
            try:
                train_scenarios, temp_scenarios = train_test_split(
                    unique_scenarios,
                    test_size=(1 - train_ratio),
                    random_state=random_state,
                    stratify=stratify_vec,
                )
            except ValueError:
                stratify_vec = None
                train_scenarios, temp_scenarios = train_test_split(
                    unique_scenarios, test_size=(1 - train_ratio), random_state=random_state
                )

            # Debug output
            print(
                f"[DEBUG] Split scenarios: train={len(train_scenarios)}, temp={len(temp_scenarios)}"
            )

            # Check if we have enough scenarios for val/test split
            if len(temp_scenarios) < 2:
                # Not enough scenarios for val/test split
                # Put all temp scenarios in validation, test will be empty
                print(
                    f"⚠️  Warning: Only {len(temp_scenarios)} scenario(s) available "
                    f"for val/test split. Assigning all to validation."
                )
                val_scenarios = temp_scenarios
                test_scenarios = np.array([], dtype=unique_scenarios.dtype)
            else:
                val_size = val_ratio / (val_ratio + test_ratio)
                stratify_temp = None
                if stratify_vec is not None and "is_stable" in data.columns:
                    scenario_stability = data.groupby("scenario_id")["is_stable"].first()
                    stratify_temp = scenario_stability.reindex(temp_scenarios).values
                    if np.any(pd.isna(stratify_temp)) or len(np.unique(stratify_temp)) < 2:
                        stratify_temp = None
                try:
                    val_scenarios, test_scenarios = train_test_split(
                        temp_scenarios,
                        test_size=(1 - val_size),
                        random_state=random_state,
                        stratify=stratify_temp,
                    )
                except ValueError:
                    val_scenarios, test_scenarios = train_test_split(
                        temp_scenarios, test_size=(1 - val_size), random_state=random_state
                    )

                # Debug output
                print(
                    f"[DEBUG] Split temp scenarios: val={len(val_scenarios)},"
                    f"test={len(test_scenarios)}"
                )

            # Split data based on scenarios
            # Convert to sets for faster lookup and ensure proper type matching
            train_scenarios_set = set(train_scenarios)
            val_scenarios_set = set(val_scenarios)
            test_scenarios_set = set(test_scenarios) if len(test_scenarios) > 0 else set()

            # Verify no overlap in scenario assignments
            if train_scenarios_set & val_scenarios_set:
                raise ValueError(
                    f"BUG: train_scenarios and val_scenarios overlap: "
                    f"{train_scenarios_set & val_scenarios_set}"
                )
            if train_scenarios_set & test_scenarios_set:
                raise ValueError(
                    f"BUG: train_scenarios and test_scenarios overlap: "
                    f"{train_scenarios_set & test_scenarios_set}"
                )
            if val_scenarios_set & test_scenarios_set:
                raise ValueError(
                    f"BUG: val_scenarios and test_scenarios overlap: "
                    f"{val_scenarios_set & test_scenarios_set}"
                )

            # Split data based on scenarios
            # Debug: Check data types before filtering
            print(f"[DEBUG] Before filtering:")
            train_type = type(list(train_scenarios_set)[0]) if train_scenarios_set else "empty"
            val_type = type(list(val_scenarios_set)[0]) if val_scenarios_set else "empty"
            print(f"train_scenarios_set type: {train_type}")
            print(f"val_scenarios_set type: {val_type}")
            test_type = type(list(test_scenarios_set)[0]) if test_scenarios_set else "empty"
            print(f"test_scenarios_set type: {test_type}")
            print(f"  data['{stratify_by}'] dtype: {data[stratify_by].dtype}")

            # Ensure type compatibility
            # Convert scenario sets to match data dtype
            data_dtype = data[stratify_by].dtype
            if pd.api.types.is_integer_dtype(data_dtype):
                train_scenarios_set = {int(s) for s in train_scenarios_set}
                val_scenarios_set = {int(s) for s in val_scenarios_set}
                test_scenarios_set = {int(s) for s in test_scenarios_set}
            elif pd.api.types.is_float_dtype(data_dtype):
                train_scenarios_set = {float(s) for s in train_scenarios_set}
                val_scenarios_set = {float(s) for s in val_scenarios_set}
                test_scenarios_set = {float(s) for s in test_scenarios_set}

            train_data = data[data[stratify_by].isin(train_scenarios_set)].copy()
            val_data = data[data[stratify_by].isin(val_scenarios_set)].copy()
            if len(test_scenarios_set) > 0:
                test_data = data[data[stratify_by].isin(test_scenarios_set)].copy()
            else:
                test_data = pd.DataFrame(columns=data.columns)

            # Debug: Check what we actually got
            print(f"[DEBUG] After filtering:")
            print(
                f"Train rows: {len(train_data):,}, unique scenarios:"
                f"{train_data[stratify_by].nunique()}"
            )
            print(
                f"  Val rows: {len(val_data):,}, unique scenarios: {val_data[stratify_by].nunique()}"
            )
            print(
                f"Test rows: {len(test_data):,}, unique scenarios:"
                f"{test_data[stratify_by].nunique() if len(test_data) > 0 else 0}"
            )

            # Final verification: check that scenario counts match
            train_scenarios_actual = set(train_data[stratify_by].unique())
            val_scenarios_actual = set(val_data[stratify_by].unique())
            test_scenarios_actual = (
                set(test_data[stratify_by].unique()) if len(test_data) > 0 else set()
            )

            # Debug output
            print(f"[DEBUG] Final split results:")
            print(f"  Train: {len(train_scenarios_actual)} scenarios, {len(train_data)} rows")
            print(f"  Val: {len(val_scenarios_actual)} scenarios, {len(val_data)} rows")
            print(f"  Test: {len(test_scenarios_actual)} scenarios, {len(test_data)} rows")

            if train_scenarios_actual != train_scenarios_set:
                raise ValueError(
                    f"BUG: train_data scenarios don't match assigned scenarios. "
                    f"Expected {len(train_scenarios_set)}, got {len(train_scenarios_actual)}. "
                    f"Missing: {train_scenarios_set - train_scenarios_actual}, "
                    f"Extra: {train_scenarios_actual - train_scenarios_set}"
                )
            if val_scenarios_actual != val_scenarios_set:
                raise ValueError(
                    f"BUG: val_data scenarios don't match assigned scenarios. "
                    f"Expected {len(val_scenarios_set)}, got {len(val_scenarios_actual)}. "
                    f"Missing: {val_scenarios_set - val_scenarios_actual}, "
                    f"Extra: {val_scenarios_actual - val_scenarios_set}"
                )
            if len(test_scenarios_set) > 0 and test_scenarios_actual != test_scenarios_set:
                raise ValueError(
                    f"BUG: test_data scenarios don't match assigned scenarios. "
                    f"Expected {len(test_scenarios_set)}, got {len(test_scenarios_actual)}. "
                    f"Missing: {test_scenarios_set - test_scenarios_actual}, "
                    f"Extra: {test_scenarios_actual - test_scenarios_set}"
                )

            return train_data, val_data, test_data

    # Simple random split (used when not stratifying or when too few scenarios)
    train_data, temp_data = train_test_split(
        data, test_size=(1 - train_ratio), random_state=random_state
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data, test_size=(1 - val_size), random_state=random_state
    )

    return train_data, val_data, test_data


def preprocess_data(
    data: pd.DataFrame,
    normalize: bool = True,
    apply_feature_engineering: bool = True,
    split: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    normalization_method: str = "standard",
    stratify_by: Optional[str] = None,
    stratify_by_stability: bool = True,
    output_dir: Optional[str] = None,
    random_state: int = 42,
    filter_angles: bool = False,
    max_angle_deg: float = 360.0,
    stability_threshold_deg: float = 180.0,
) -> Dict:
    """
    Complete preprocessing pipeline.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    normalize : bool
        Whether to normalize data
    apply_feature_engineering : bool
        Whether to engineer features
    split : bool
        Whether to split dataset
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    test_ratio : float
        Test set ratio
    normalization_method : str
        Normalization method
    stratify_by : str, optional
        Column to stratify by (e.g. scenario_id)
    stratify_by_stability : bool
        If True, split so train/val/test each include both stable and unstable scenarios (default True when is_stable present).
    output_dir : str, optional
        Directory to save preprocessed data
    random_state : int
        Random seed
    filter_angles : bool
        Whether to filter trajectories by rotor angle (default: False)
    max_angle_deg : float
        Maximum angle in degrees to keep (default: 360.0)
    stability_threshold_deg : float
        Stability threshold in degrees (default: 180.0)

    Returns:
    --------
    dict : Dictionary containing preprocessed data and metadata
    """
    df = data.copy()

    # Map multimachine columns (delta_0, omega_0, param_*) to SMIB-style (delta, omega, tf, tc, Pm)
    # so the same pipeline works for both SMIB and Kundur multimachine data
    df = map_multimachine_to_smib_columns(df)

    # Angle filtering (applied first, before feature engineering)
    # COMMENTED OUT: keep full 5 s trajectories for trajectory plots and training.
    # Uncomment to truncate trajectories when |delta| >= max_angle_deg (e.g. 360°).
    filter_stats = None
    # if filter_angles:
    #     if not ANGLE_FILTER_AVAILABLE:
    #         print("⚠️  Warning: angle_filter module not available. Skipping angle filtering.")
    #     elif "delta" not in df.columns:
    #         print("⚠️  Warning: 'delta' column not found. Skipping angle filtering.")
    #     else:
    #         print(f"\n📊 Filtering trajectories: limiting angles to {max_angle_deg}°")
    #         print(f"   Stability threshold: {stability_threshold_deg}°")
    #         df, filter_stats = filter_trajectory_by_angle(
    #             data=df,
    #             max_angle_deg=max_angle_deg,
    #             stability_threshold_deg=stability_threshold_deg,
    #         )
    #         if filter_stats:
    #             print(
    #                 f"✓ Filtered: {filter_stats.get('original_points', 0)} →"
    #                 f"{filter_stats.get('filtered_points', 0)} points"
    #             )
    #             print(
    #                 f"✓ Scenarios: {filter_stats.get('original_scenarios', 0)} →"
    #                 f"{filter_stats.get('filtered_scenarios', 0)}"
    #             )
    #             print(
    #                 f"✓ Max angle: {filter_stats.get('max_angle_before', 0):.1f}° →"
    #                 f"{filter_stats.get('max_angle_after', 0):.1f}°"
    #             )

    # Feature engineering
    if apply_feature_engineering:
        df = engineer_features(df)

    # Normalization
    scaler_dict = None
    if normalize:
        df, scaler_dict = normalize_data(df, method=normalization_method)

    # Split dataset
    train_data = val_data = test_data = None
    if split:
        train_data, val_data, test_data = split_dataset(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_by=stratify_by,
            stratify_by_stability=stratify_by_stability,
            random_state=random_state,
        )

    # Save if output directory specified
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if split:
            train_data.to_csv(output_path / "train_data.csv", index=False)
            val_data.to_csv(output_path / "val_data.csv", index=False)
            test_data.to_csv(output_path / "test_data.csv", index=False)
        else:
            df.to_csv(output_path / "preprocessed_data.csv", index=False)

        if scaler_dict is not None:
            with open(output_path / "scalers.pkl", "wb") as f:
                pickle.dump(scaler_dict, f)

    result = {
        "data": df if not split else None,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "scaler_dict": scaler_dict,
        "feature_columns": list(df.columns),
        "filter_stats": filter_stats,
    }

    return result
