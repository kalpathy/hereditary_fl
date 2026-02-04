"""
Run federated learning simulation using Flower.

This script demonstrates a complete FL pipeline with:
- Simulated data generation
- Data partitioning across multiple clients
- Federated training using FedAvg
- Model evaluation
"""

import flwr as fl
import numpy as np
import torch
from typing import Dict

from data_utils import generate_simulated_data, partition_data, create_data_loaders
from client import create_client_fn
from server import create_strategy, get_evaluate_fn


def run_fl_simulation(
    n_clients: int = 3,
    n_rounds: int = 5,
    n_samples: int = 1500,
    n_features: int = 20,
    n_classes: int = 2,
    iid: bool = True,
    random_seed: int = 42
) -> Dict:
    """
    Run a federated learning simulation.
    
    Args:
        n_clients: Number of FL clients
        n_rounds: Number of FL rounds
        n_samples: Total number of data samples
        n_features: Number of features per sample
        n_classes: Number of classes
        iid: Whether to use IID data distribution
        random_seed: Random seed for reproducibility
    
    Returns:
        History object with training metrics
    """
    print("=" * 60)
    print("Federated Learning Simulation with Flower")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Number of clients: {n_clients}")
    print(f"  - Number of rounds: {n_rounds}")
    print(f"  - Total samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - Classes: {n_classes}")
    print(f"  - IID distribution: {iid}")
    print("=" * 60)
    
    # Generate simulated data
    print("\n[1] Generating simulated data...")
    X, y = generate_simulated_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_seed=random_seed
    )
    print(f"    Generated {len(X)} samples with {n_features} features")
    
    # Partition data for clients
    print("\n[2] Partitioning data across clients...")
    partitions = partition_data(X, y, n_clients=n_clients, iid=iid, random_seed=random_seed)
    for i, (X_i, y_i) in enumerate(partitions):
        class_dist = np.bincount(y_i, minlength=n_classes)
        print(f"    Client {i}: {len(X_i)} samples, class distribution: {class_dist}")
    
    # Create a centralized test set for server-side evaluation
    print("\n[3] Creating centralized test set...")
    X_test, y_test = generate_simulated_data(
        n_samples=500,
        n_features=n_features,
        n_classes=n_classes,
        random_seed=random_seed + 100  # Different seed for test data
    )
    _, test_loader = create_data_loaders(X_test, y_test, train_ratio=0.0)
    print(f"    Created test set with {len(X_test)} samples")
    
    # Create client function
    print("\n[4] Setting up FL clients...")
    client_fn = create_client_fn(partitions, n_features=n_features, n_classes=n_classes)
    
    # Create strategy
    print("\n[5] Creating FedAvg strategy...")
    strategy = create_strategy(
        n_features=n_features,
        n_classes=n_classes,
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients
    )
    
    # Create client resources
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if torch.cuda.is_available():
        client_resources["num_gpus"] = 0.5
    
    # Run simulation
    print("\n[6] Starting FL simulation...")
    print("-" * 60)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    
    print("-" * 60)
    print("\n[7] Simulation complete!")
    
    # Print final results
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    if history.metrics_distributed:
        print("\nDistributed (Client) Metrics:")
        for round_num, metrics in enumerate(history.metrics_distributed.get("accuracy", []), 1):
            print(f"  Round {round_num}: Accuracy = {metrics[1]:.4f}")
    
    if history.losses_distributed:
        print("\nDistributed Losses:")
        for round_num, loss in enumerate(history.losses_distributed, 1):
            print(f"  Round {round_num}: Loss = {loss[1]:.4f}")
    
    return history


if __name__ == "__main__":
    # Run the simulation
    history = run_fl_simulation(
        n_clients=3,
        n_rounds=5,
        n_samples=1500,
        n_features=20,
        n_classes=2,
        iid=True
    )
