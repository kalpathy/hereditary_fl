"""MedMNIST dataset utilities for federated learning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from typing import List, Tuple, Dict, Optional
import medmnist
from medmnist import INFO


# Available MedMNIST datasets with their properties
MEDMNIST_DATASETS = {
    "pathmnist": {
        "name": "PathMNIST",
        "description": "Colon Pathology (9 classes)",
        "n_channels": 3,
        "n_classes": 9,
        "task": "multi-class"
    },
    "dermamnist": {
        "name": "DermaMNIST",
        "description": "Dermatoscopy (7 classes)",
        "n_channels": 3,
        "n_classes": 7,
        "task": "multi-class"
    },
    "bloodmnist": {
        "name": "BloodMNIST",
        "description": "Blood Cell Microscopy (8 classes)",
        "n_channels": 3,
        "n_classes": 8,
        "task": "multi-class"
    },
    "breastmnist": {
        "name": "BreastMNIST",
        "description": "Breast Ultrasound (2 classes)",
        "n_channels": 1,
        "n_classes": 2,
        "task": "binary-class"
    },
    "organamnist": {
        "name": "OrganAMNIST",
        "description": "Abdominal CT - Axial (11 classes)",
        "n_channels": 1,
        "n_classes": 11,
        "task": "multi-class"
    },
    "organcmnist": {
        "name": "OrganCMNIST",
        "description": "Abdominal CT - Coronal (11 classes)",
        "n_channels": 1,
        "n_classes": 11,
        "task": "multi-class"
    },
    "organsmnist": {
        "name": "OrganSMNIST",
        "description": "Abdominal CT - Sagittal (11 classes)",
        "n_channels": 1,
        "n_classes": 11,
        "task": "multi-class"
    },
    "pneumoniamnist": {
        "name": "PneumoniaMNIST",
        "description": "Pediatric Chest X-Ray (2 classes)",
        "n_channels": 1,
        "n_classes": 2,
        "task": "binary-class"
    },
    "retinamnist": {
        "name": "RetinaMNIST",
        "description": "Retina OCT (5 ordinal classes)",
        "n_channels": 3,
        "n_classes": 5,
        "task": "ordinal-regression"
    },
    "tissuemnist": {
        "name": "TissueMNIST",
        "description": "Kidney Cortex Cells (8 classes)",
        "n_channels": 1,
        "n_classes": 8,
        "task": "multi-class"
    },
}


def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about a MedMNIST dataset."""
    if dataset_name not in MEDMNIST_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(MEDMNIST_DATASETS.keys())}")
    return MEDMNIST_DATASETS[dataset_name]


def load_medmnist(
    dataset_name: str,
    data_dir: str = None,
    download: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a MedMNIST dataset.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    import os
    
    # Use default data directory if not specified
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    
    # Load train, val, test splits
    train_dataset = DataClass(split="train", download=download, root=data_dir)
    val_dataset = DataClass(split="val", download=download, root=data_dir)
    test_dataset = DataClass(split="test", download=download, root=data_dir)
    
    # Extract images and labels
    X_train = train_dataset.imgs
    y_train = train_dataset.labels.flatten()
    
    X_val = val_dataset.imgs
    y_val = val_dataset.labels.flatten()
    
    X_test = test_dataset.imgs
    y_test = test_dataset.labels.flatten()
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_images(images: np.ndarray, n_channels: int) -> torch.Tensor:
    """
    Preprocess images for PyTorch CNN.
    
    Args:
        images: numpy array of shape (N, H, W) or (N, H, W, C)
        n_channels: expected number of channels
    
    Returns:
        Tensor of shape (N, C, H, W) normalized to [0, 1]
    """
    # Ensure 4D array
    if images.ndim == 3:
        images = images[..., np.newaxis]
    
    # Convert to float and normalize
    images = images.astype(np.float32) / 255.0
    
    # Transpose from (N, H, W, C) to (N, C, H, W)
    images = np.transpose(images, (0, 3, 1, 2))
    
    return torch.from_numpy(images)


def partition_medmnist_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data IID across clients.
    
    Returns:
        List of (X_client, y_client) tuples
    """
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    # Split indices evenly
    splits = np.array_split(indices, n_clients)
    
    client_data = []
    for split in splits:
        client_data.append((X[split], y[split]))
    
    return client_data


def partition_medmnist_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    classes_per_client: int = 2,
    alpha: float = 0.5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data non-IID across clients using Dirichlet distribution.
    
    Args:
        X: Image data
        y: Labels
        n_clients: Number of clients
        classes_per_client: Minimum classes per client (ignored with Dirichlet)
        alpha: Dirichlet concentration parameter (smaller = more non-IID)
    
    Returns:
        List of (X_client, y_client) tuples
    """
    n_classes = len(np.unique(y))
    n_samples = len(y)
    
    # Group indices by class
    class_indices = {c: np.where(y == c)[0] for c in range(n_classes)}
    
    # Sample from Dirichlet distribution for each class
    client_indices = [[] for _ in range(n_clients)]
    
    for c in range(n_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * n_clients)
        
        # Ensure minimum samples per client for this class
        proportions = np.maximum(proportions, 0.01)
        proportions = proportions / proportions.sum()
        
        # Split indices according to proportions
        splits = (proportions * len(indices)).astype(int)
        splits[-1] = len(indices) - splits[:-1].sum()  # Handle rounding
        
        start = 0
        for client_id, split_size in enumerate(splits):
            end = start + split_size
            client_indices[client_id].extend(indices[start:end])
            start = end
    
    # Create client datasets
    client_data = []
    for indices in client_indices:
        indices = np.array(indices)
        if len(indices) > 0:
            np.random.shuffle(indices)
            client_data.append((X[indices], y[indices]))
        else:
            # Handle empty clients by giving them a small random sample
            sample_indices = np.random.choice(n_samples, size=10, replace=False)
            client_data.append((X[sample_indices], y[sample_indices]))
    
    return client_data


def create_medmnist_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    n_channels: int,
    batch_size: int = 32,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders from numpy arrays.
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    n_samples = len(X)
    n_train = max(1, int(n_samples * train_ratio))
    
    # Preprocess images
    X_tensor = preprocess_images(X, n_channels)
    y_tensor = torch.from_numpy(y).long()
    
    # Split into train/test
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:] if n_train < n_samples else indices[:max(1, n_samples // 5)]
    
    # Create datasets
    train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
    test_dataset = TensorDataset(X_tensor[test_indices], y_tensor[test_indices])
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_client_distribution(client_data: List[Tuple[np.ndarray, np.ndarray]], n_classes: int) -> Dict:
    """
    Get label distribution for each client.
    
    Returns:
        Dictionary with client_id -> class_counts mapping
    """
    distribution = {}
    for client_id, (X, y) in enumerate(client_data):
        counts = np.bincount(y, minlength=n_classes)
        distribution[f"Client {client_id}"] = counts.tolist()
    return distribution


if __name__ == "__main__":
    # Test the utilities
    print("Testing MedMNIST utilities...")
    
    # List available datasets
    print("\nAvailable MedMNIST datasets:")
    for name, info in MEDMNIST_DATASETS.items():
        print(f"  {name}: {info['description']}")
    
    # Load a small dataset
    print("\nLoading PathMNIST...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_medmnist("pathmnist")
    print(f"  Train: {X_train.shape}, labels: {np.unique(y_train)}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Test IID partitioning
    print("\nTesting IID partitioning (3 clients)...")
    client_data = partition_medmnist_iid(X_train, y_train, n_clients=3)
    for i, (X, y) in enumerate(client_data):
        print(f"  Client {i}: {len(X)} samples, classes: {np.unique(y)}")
    
    # Test non-IID partitioning
    print("\nTesting non-IID partitioning (3 clients, alpha=0.5)...")
    client_data = partition_medmnist_non_iid(X_train, y_train, n_clients=3, alpha=0.5)
    for i, (X, y) in enumerate(client_data):
        print(f"  Client {i}: {len(X)} samples, class distribution: {np.bincount(y, minlength=9)}")
    
    print("\nMedMNIST utilities test complete!")
