"""
Data Utilities for PINN Training.

This module provides utilities for loading, saving, and preparing data batches.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory prediction.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str] = ["delta", "omega"],
    ):
        """
        Initialize dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        feature_columns : list
            Column names for features
        target_columns : list
            Column names for targets
        """
        self.data = data
        self.feature_columns = feature_columns
        self.target_columns = target_columns

        # Extract features and targets
        self.features = data[feature_columns].values.astype(np.float32)
        self.targets = data[target_columns].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        return features, targets


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Parameters:
    -----------
    file_path : str
        Path to CSV file

    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    return pd.read_csv(file_path)


def save_dataset(data: pd.DataFrame, file_path: str):
    """
    Save dataset to CSV file.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to save
    file_path : str
        Path to save file
    """
    data.to_csv(file_path, index=False)


def prepare_batch(
    batch: Tuple[torch.Tensor, torch.Tensor], device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare batch for training.

    Parameters:
    -----------
    batch : tuple
        Batch of (features, targets)
    device : str
        Device ('cpu' or 'cuda')

    Returns:
    --------
    tuple : (features, targets) on specified device
    """
    features, targets = batch
    return features.to(device), targets.to(device)


def create_dataloader(
    dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader from dataset.

    Parameters:
    -----------
    dataset : Dataset
        PyTorch dataset
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle data
    num_workers : int
        Number of worker processes

    Returns:
    --------
    DataLoader : PyTorch DataLoader
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def generate_collocation_points(
    t_min: float,
    t_max: float,
    n_points: int,
    strategy: str = "uniform",
    fault_clearing_time: Optional[float] = None,
    emphasis_region: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate collocation points for physics loss computation.

    Similar to PINNSim's approach: generates points in time domain without
    requiring simulation. For SMIB, typically use 4-5x more collocation points
    than data points.

    Parameters:
    -----------
    t_min : float
        Minimum time (typically 0.0)
    t_max : float
        Maximum time (simulation duration)
    n_points : int
        Number of collocation points to generate
    strategy : str
        Sampling strategy: 'uniform', 'random', 'log', 'adaptive'
        - 'uniform': Uniformly spaced points
        - 'random': Random sampling
        - 'log': Logarithmic spacing (more points near t_min)
        - 'adaptive': Higher density near fault clearing time
    fault_clearing_time : float, optional
        Fault clearing time (for adaptive strategy)
    emphasis_region : tuple, optional
        (t_start, t_end) region to emphasize (for adaptive strategy)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray : Collocation time points [n_points]
    """
    if seed is not None:
        np.random.seed(seed)

    if strategy == "uniform":
        # Uniformly spaced points
        t_colloc = np.linspace(t_min, t_max, n_points)

    elif strategy == "random":
        # Random sampling
        t_colloc = np.random.uniform(t_min, t_max, n_points)
        t_colloc = np.sort(t_colloc)

    elif strategy == "log":
        # Logarithmic spacing (more points near t_min)
        t_colloc_log = np.log(t_min + 1e-8) + np.random.rand(n_points) * (
            np.log(t_max + 1e-8) - np.log(t_min + 1e-8)
        )
        t_colloc = np.exp(t_colloc_log) - 1e-8
        t_colloc = np.clip(t_colloc, t_min, t_max)
        t_colloc = np.sort(t_colloc)

    elif strategy == "adaptive":
        # Higher density near fault clearing time or emphasis region
        if fault_clearing_time is not None:
            # Emphasize region around fault clearing time
            emphasis_start = max(t_min, fault_clearing_time - 0.2)
            emphasis_end = min(t_max, fault_clearing_time + 0.2)
        elif emphasis_region is not None:
            emphasis_start, emphasis_end = emphasis_region
        else:
            # Default: emphasize middle region
            emphasis_start = (t_min + t_max) / 2 - 0.2
            emphasis_end = (t_min + t_max) / 2 + 0.2

        # Allocate 60% of points to emphasis region, 40% to rest
        n_emphasis = int(0.6 * n_points)
        n_rest = n_points - n_emphasis

        # Points in emphasis region
        t_emphasis = np.random.uniform(emphasis_start, emphasis_end, n_emphasis)

        # Points outside emphasis region
        t_rest = []
        if emphasis_start > t_min:
            n_before = n_rest // 2
            t_rest.append(np.random.uniform(t_min, emphasis_start, n_before))
        if emphasis_end < t_max:
            n_after = n_rest - len(t_rest) if t_rest else n_rest
            t_rest.append(np.random.uniform(emphasis_end, t_max, n_after))

        if t_rest:
            t_colloc = np.concatenate([t_emphasis] + t_rest)
        else:
            t_colloc = t_emphasis
        t_colloc = np.sort(t_colloc)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return t_colloc


def generate_collocation_batch(
    t_data: np.ndarray,
    n_colloc: Optional[int] = None,
    strategy: str = "uniform",
    fault_clearing_time: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate collocation points for a batch of trajectories.

    Convenience function that generates collocation points based on data time range.

    Parameters:
    -----------
    t_data : np.ndarray
        Time points from data [n_time_points]
    n_colloc : int, optional
        Number of collocation points. If None, uses 4-5x data points.
    strategy : str
        Sampling strategy (see generate_collocation_points)
    fault_clearing_time : float, optional
        Fault clearing time for adaptive strategy
    seed : int, optional
        Random seed

    Returns:
    --------
    np.ndarray : Collocation time points [n_colloc]
    """
    t_min = float(t_data.min())
    t_max = float(t_data.max())

    if n_colloc is None:
        # Default: 4-5x data points (PINNSim uses ~4:1 ratio)
        n_colloc = min(200, max(50, len(t_data) * 4))

    return generate_collocation_points(
        t_min=t_min,
        t_max=t_max,
        n_points=n_colloc,
        strategy=strategy,
        fault_clearing_time=fault_clearing_time,
        seed=seed,
    )
