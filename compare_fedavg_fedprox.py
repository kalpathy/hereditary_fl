"""
Compare FedAvg vs FedProx on non-IID data.
"""

import flwr as fl
import numpy as np
import torch

from data_utils import generate_simulated_data, partition_data, create_data_loaders
from client import create_client_fn
from fedprox_client import create_fedprox_client_fn
from server import create_strategy


def run_comparison(
    n_clients: int = 3,
    n_rounds: int = 10,
    n_samples: int = 1500,
    mu: float = 0.5  # FedProx proximal coefficient
):
    """Compare FedAvg vs FedProx on non-IID data."""
    
    print("=" * 60)
    print("FedAvg vs FedProx Comparison on Non-IID Data")
    print("=" * 60)
    
    # Generate non-IID data
    X, y = generate_simulated_data(n_samples=n_samples)
    partitions = partition_data(X, y, n_clients=n_clients, iid=False)
    
    print("\nData distribution (non-IID):")
    for i, (X_i, y_i) in enumerate(partitions):
        print(f"  Client {i}: {np.bincount(y_i, minlength=2)}")
    
    strategy = create_strategy(min_fit_clients=n_clients, min_evaluate_clients=n_clients, min_available_clients=n_clients)
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    
    # Run FedAvg
    print("\n" + "-" * 60)
    print("Running FedAvg...")
    print("-" * 60)
    
    fedavg_history = fl.simulation.start_simulation(
        client_fn=create_client_fn(partitions),
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=create_strategy(min_fit_clients=n_clients, min_evaluate_clients=n_clients, min_available_clients=n_clients),
        client_resources=client_resources,
    )
    
    # Run FedProx
    print("\n" + "-" * 60)
    print(f"Running FedProx (mu={mu})...")
    print("-" * 60)
    
    fedprox_history = fl.simulation.start_simulation(
        client_fn=create_fedprox_client_fn(partitions, mu=mu),
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=create_strategy(min_fit_clients=n_clients, min_evaluate_clients=n_clients, min_available_clients=n_clients),
        client_resources=client_resources,
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print("\nLoss per round:")
    print(f"{'Round':<8} {'FedAvg':<15} {'FedProx':<15}")
    print("-" * 38)
    
    fedavg_losses = fedavg_history.losses_distributed
    fedprox_losses = fedprox_history.losses_distributed
    
    for i in range(min(len(fedavg_losses), len(fedprox_losses))):
        round_num = i + 1
        avg_loss = fedavg_losses[i][1]
        prox_loss = fedprox_losses[i][1]
        print(f"{round_num:<8} {avg_loss:<15.6f} {prox_loss:<15.6f}")
    
    # Final comparison
    print("\n" + "-" * 38)
    print(f"{'Final':<8} {fedavg_losses[-1][1]:<15.6f} {fedprox_losses[-1][1]:<15.6f}")
    
    improvement = ((fedavg_losses[-1][1] - fedprox_losses[-1][1]) / fedavg_losses[-1][1]) * 100
    if improvement > 0:
        print(f"\nFedProx achieved {improvement:.1f}% lower final loss!")
    else:
        print(f"\nFedAvg achieved {-improvement:.1f}% lower final loss")


if __name__ == "__main__":
    run_comparison(n_clients=3, n_rounds=10, mu=0.5)
