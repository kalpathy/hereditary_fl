"""Flower client implementation for federated learning."""

import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Tuple

from model import SimpleNet, get_parameters, set_parameters, train_one_epoch, evaluate
from data_utils import create_data_loaders


class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning."""
    
    def __init__(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int = 20,
        n_classes: int = 2,
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01
    ):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        self.model = SimpleNet(n_features=n_features, n_classes=n_classes).to(self.device)
        
        # Create data loaders
        self.train_loader, self.test_loader = create_data_loaders(X, y, batch_size=batch_size)
        
        print(f"Client {client_id} initialized with {len(X)} samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters."""
        return get_parameters(self.model)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on local data."""
        # Set global parameters
        set_parameters(self.model, parameters)
        
        # Get training config
        local_epochs = config.get("local_epochs", self.local_epochs)
        
        # Create optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        
        # Train for local epochs
        for epoch in range(local_epochs):
            loss = train_one_epoch(self.model, self.train_loader, optimizer, self.device)
            print(f"Client {self.client_id}, Epoch {epoch+1}/{local_epochs}, Loss: {loss:.4f}")
        
        # Return updated parameters and metrics
        return get_parameters(self.model), len(self.train_loader.dataset), {"loss": loss}
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model on local test data."""
        set_parameters(self.model, parameters)
        
        loss, accuracy = evaluate(self.model, self.test_loader, self.device)
        
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def create_client_fn(
    partitions: List[Tuple[np.ndarray, np.ndarray]],
    n_features: int = 20,
    n_classes: int = 2
):
    """Create a client function for Flower simulation."""
    
    def client_fn(cid: str) -> FlowerClient:
        """Create a single client instance."""
        client_id = int(cid)
        X, y = partitions[client_id]
        return FlowerClient(
            client_id=client_id,
            X=X,
            y=y,
            n_features=n_features,
            n_classes=n_classes
        )
    
    return client_fn
