"""Utilities for generating and partitioning simulated data."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple


def generate_simulated_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simulated classification data.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features per sample
        n_classes: Number of classes
        random_seed: Random seed for reproducibility
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
    """
    np.random.seed(random_seed)
    
    # Generate class centers
    centers = np.random.randn(n_classes, n_features) * 3
    
    # Generate samples
    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []
    
    for class_idx in range(n_classes):
        # Generate samples around each center with some noise
        class_samples = centers[class_idx] + np.random.randn(samples_per_class, n_features) * 0.5
        X_list.append(class_samples)
        y_list.append(np.full(samples_per_class, class_idx))
    
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def partition_data(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 3,
    iid: bool = True,
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data among clients.
    
    Args:
        X: Feature matrix
        y: Labels
        n_clients: Number of clients
        iid: If True, data is IID distributed; if False, non-IID
        random_seed: Random seed
    
    Returns:
        List of (X, y) tuples for each client
    """
    np.random.seed(random_seed)
    n_samples = len(X)
    
    if iid:
        # IID: Randomly shuffle and split equally
        indices = np.random.permutation(n_samples)
        splits = np.array_split(indices, n_clients)
    else:
        # Non-IID: Sort by label and distribute
        sorted_indices = np.argsort(y)
        # Give each client data primarily from certain classes
        splits = np.array_split(sorted_indices, n_clients)
    
    partitions = []
    for split in splits:
        partitions.append((X[split], y[split]))
    
    return partitions


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders from numpy arrays.
    
    Args:
        X: Feature matrix
        y: Labels
        batch_size: Batch size for DataLoader
        train_ratio: Ratio of data to use for training
    
    Returns:
        train_loader, test_loader
    """
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Split into train and test
    n_train = int(len(X) * train_ratio)
    
    # Handle edge cases where train_ratio is 0 or 1
    if train_ratio == 0.0:
        # All data goes to test
        test_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    elif train_ratio == 1.0:
        # All data goes to train
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = None
        return train_loader, test_loader
    
    X_train, X_test = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_test = y_tensor[:n_train], y_tensor[n_train:]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test data generation
    X, y = generate_simulated_data(n_samples=1000, n_features=20, n_classes=2)
    print(f"Generated data: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Test partitioning
    partitions = partition_data(X, y, n_clients=3, iid=True)
    for i, (X_i, y_i) in enumerate(partitions):
        print(f"Client {i}: {len(X_i)} samples, class dist: {np.bincount(y_i)}")
