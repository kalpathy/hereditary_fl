"""
Using Flower's native FedProx strategy.

NOTE: Flower's FedProx strategy sends `proximal_mu` to clients via config,
but you STILL need to implement the proximal term in the client's training loop.
The strategy itself is identical to FedAvg - the difference is in the client.
"""

import flwr as fl
from flwr.server.strategy import FedProx  # Native FedProx!
import numpy as np
import torch

from data_utils import generate_simulated_data, partition_data
from model import SimpleNet, get_parameters
from fedprox_client import create_fedprox_client_fn  # Our client with proximal term


def run_native_fedprox(
    n_clients: int = 3,
    n_rounds: int = 5,
    mu: float = 0.5,
    iid: bool = False
):
    """Run FL with Flower's native FedProx strategy."""
    
    print("=" * 60)
    print("Using Flower's Native FedProx Strategy")
    print("=" * 60)
    
    # Generate non-IID data
    X, y = generate_simulated_data(n_samples=1500)
    partitions = partition_data(X, y, n_clients=n_clients, iid=iid)
    
    print(f"\nData distribution ({'IID' if iid else 'non-IID'}):")
    for i, (X_i, y_i) in enumerate(partitions):
        print(f"  Client {i}: {np.bincount(y_i, minlength=2)}")
    
    # Create initial model parameters
    initial_model = SimpleNet()
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(initial_model))
    
    # Use Flower's native FedProx strategy
    strategy = FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        initial_parameters=initial_params,
        proximal_mu=mu,  # This gets passed to clients via config
    )
    
    print(f"\nStrategy: FedProx (native)")
    print(f"Proximal mu: {mu}")
    print("-" * 60)
    
    # Run simulation - client must still implement proximal term!
    history = fl.simulation.start_simulation(
        client_fn=create_fedprox_client_fn(partitions, mu=mu),
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for i, (_, loss) in enumerate(history.losses_distributed, 1):
        print(f"  Round {i}: Loss = {loss:.6f}")
    
    return history


if __name__ == "__main__":
    run_native_fedprox(n_clients=3, n_rounds=5, mu=0.5, iid=False)
