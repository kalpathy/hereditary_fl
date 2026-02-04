"""FedProx client implementation - adds proximal term to handle non-IID data."""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

from model import SimpleNet, get_parameters, set_parameters, evaluate
from data_utils import create_data_loaders


def train_fedprox(
    model: nn.Module,
    global_params: List[torch.Tensor],
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mu: float = 0.1  # Proximal term coefficient
) -> float:
    """
    Train with FedProx - adds proximal term to handle heterogeneity.
    
    The proximal term penalizes local updates that deviate too far
    from the global model: loss += (mu/2) * ||w - w_global||^2
    """
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # Convert global params to tensors on device
    global_tensors = [torch.tensor(p, device=device) for p in global_params]
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        
        # Standard cross-entropy loss
        loss = criterion(outputs, y_batch)
        
        # Add proximal term: (mu/2) * ||w - w_global||^2
        proximal_term = 0.0
        for local_param, global_param in zip(model.parameters(), global_tensors):
            proximal_term += ((local_param - global_param) ** 2).sum()
        
        loss += (mu / 2) * proximal_term
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


class FedProxClient(fl.client.NumPyClient):
    """Flower client using FedProx algorithm."""
    
    def __init__(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int = 20,
        n_classes: int = 2,
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        mu: float = 0.1  # FedProx proximal coefficient
    ):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.mu = mu  # Proximal term weight
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNet(n_features=n_features, n_classes=n_classes).to(self.device)
        self.train_loader, self.test_loader = create_data_loaders(X, y, batch_size=batch_size)
        
        print(f"FedProx Client {client_id} initialized (mu={mu}) with {len(X)} samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.model)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set global parameters
        set_parameters(self.model, parameters)
        
        # Store global params for proximal term
        global_params = [p.copy() for p in parameters]
        
        local_epochs = config.get("local_epochs", self.local_epochs)
        # Get proximal_mu from native FedProx strategy config, fallback to self.mu
        mu = config.get("proximal_mu", config.get("mu", self.mu))
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        
        # Train with FedProx
        for epoch in range(local_epochs):
            loss = train_fedprox(
                self.model, 
                global_params, 
                self.train_loader, 
                optimizer, 
                self.device,
                mu=mu
            )
            print(f"FedProx Client {self.client_id}, Epoch {epoch+1}/{local_epochs}, Loss: {loss:.4f}")
        
        return get_parameters(self.model), len(self.train_loader.dataset), {"loss": loss}
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        set_parameters(self.model, parameters)
        loss, accuracy = evaluate(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def create_fedprox_client_fn(
    partitions: List[Tuple[np.ndarray, np.ndarray]],
    n_features: int = 20,
    n_classes: int = 2,
    mu: float = 0.1
):
    """Create a FedProx client function for simulation."""
    
    def client_fn(cid: str) -> FedProxClient:
        client_id = int(cid)
        X, y = partitions[client_id]
        return FedProxClient(
            client_id=client_id,
            X=X,
            y=y,
            n_features=n_features,
            n_classes=n_classes,
            mu=mu
        )
    
    return client_fn
