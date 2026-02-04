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
    "colormnist": {
        "name": "ColorMNIST",
        "description": "Colored MNIST digits (10 classes)",
        "n_channels": 3,
        "n_classes": 10,
        "task": "multi-class",
        "synthetic": True
    },
}


def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about a MedMNIST dataset."""
    if dataset_name not in MEDMNIST_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(MEDMNIST_DATASETS.keys())}")
    return MEDMNIST_DATASETS[dataset_name]


def generate_colormnist(
    n_train: int = 50000,
    n_test: int = 10000,
    correlation: float = 0.9,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ColorMNIST dataset - MNIST digits with colored backgrounds.
    
    Each digit class gets a correlated color, useful for studying:
    - Spurious correlations in federated learning
    - Domain shift between clients
    - Non-IID effects from color distribution
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        correlation: How strongly color correlates with digit (0.0-1.0)
        seed: Random seed
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    from torchvision import datasets, transforms
    import os
    
    np.random.seed(seed)
    
    # Define 10 distinct colors (RGB) for 10 digit classes
    COLORS = np.array([
        [255, 0, 0],      # 0: Red
        [0, 255, 0],      # 1: Green
        [0, 0, 255],      # 2: Blue
        [255, 255, 0],    # 3: Yellow
        [255, 0, 255],    # 4: Magenta
        [0, 255, 255],    # 5: Cyan
        [255, 128, 0],    # 6: Orange
        [128, 0, 255],    # 7: Purple
        [0, 255, 128],    # 8: Spring Green
        [255, 128, 128],  # 9: Light Red/Pink
    ], dtype=np.uint8)
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Load original MNIST
    mnist_train = datasets.MNIST(root=data_dir, train=True, download=True)
    mnist_test = datasets.MNIST(root=data_dir, train=False, download=True)
    
    def colorize_mnist(images, labels, n_samples, corr):
        """Colorize MNIST images based on labels with some noise."""
        # Sample indices
        indices = np.random.choice(len(images), size=min(n_samples, len(images)), replace=False)
        
        colored_images = []
        sampled_labels = []
        
        for idx in indices:
            img = np.array(images[idx])
            label = int(labels[idx])
            
            # Decide if color correlates with label or is random
            if np.random.random() < corr:
                color = COLORS[label]
            else:
                # Random color from other classes
                other_colors = [i for i in range(10) if i != label]
                color = COLORS[np.random.choice(other_colors)]
            
            # Create RGB image: colored background, white digit
            colored = np.zeros((28, 28, 3), dtype=np.uint8)
            
            # Background color where digit is not
            mask = img < 128
            for c in range(3):
                colored[:, :, c] = np.where(mask, color[c], img)
            
            colored_images.append(colored)
            sampled_labels.append(label)
        
        return np.array(colored_images), np.array(sampled_labels)
    
    # Generate colored datasets
    X_train, y_train = colorize_mnist(
        mnist_train.data.numpy(), 
        mnist_train.targets.numpy(), 
        n_train, 
        correlation
    )
    
    X_test, y_test = colorize_mnist(
        mnist_test.data.numpy(), 
        mnist_test.targets.numpy(), 
        n_test, 
        correlation
    )
    
    # Create validation set from training
    n_val = min(5000, len(X_train) // 10)
    X_val, y_val = X_train[:n_val], y_train[:n_val]
    X_train, y_train = X_train[n_val:], y_train[n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_medmnist(
    dataset_name: str,
    data_dir: str = None,
    download: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a MedMNIST dataset (or ColorMNIST).
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    import os
    
    # Handle ColorMNIST separately
    if dataset_name == "colormnist":
        return generate_colormnist()
    
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
