"""Flower server implementation with custom strategy."""

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple
import numpy as np

from model import SimpleNet, get_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from clients using weighted average.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from each client
    
    Returns:
        Aggregated metrics dictionary
    """
    # Aggregate accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}


def create_strategy(
    n_features: int = 20,
    n_classes: int = 2,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2
) -> FedAvg:
    """
    Create a FedAvg strategy with initial parameters.
    
    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        fraction_fit: Fraction of clients to use for training
        fraction_evaluate: Fraction of clients to use for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
    
    Returns:
        FedAvg strategy
    """
    # Create initial model to get initial parameters
    initial_model = SimpleNet(n_features=n_features, n_classes=n_classes)
    initial_parameters = get_parameters(initial_model)
    
    # Convert to Flower parameters
    initial_params = fl.common.ndarrays_to_parameters(initial_parameters)
    
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=lambda x: {},  # No fit metrics aggregation
    )
    
    return strategy


def get_evaluate_fn(test_loader, n_features: int = 20, n_classes: int = 2):
    """
    Create a server-side evaluation function.
    
    This is called after each round to evaluate the global model
    on a centralized test set.
    """
    import torch
    from model import set_parameters, evaluate
    
    def evaluate_fn(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict
    ) -> Optional[Tuple[float, Dict]]:
        """Evaluate global model on centralized test set."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleNet(n_features=n_features, n_classes=n_classes).to(device)
        set_parameters(model, parameters)
        
        loss, accuracy = evaluate(model, test_loader, device)
        
        print(f"\n[Server] Round {server_round} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
        
        return loss, {"accuracy": accuracy}
    
    return evaluate_fn
