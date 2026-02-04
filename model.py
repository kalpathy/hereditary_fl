"""Simple neural network model for federated learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SimpleNet(nn.Module):
    """A simple fully connected neural network for classification."""
    
    def __init__(self, n_features: int = 20, n_hidden: int = 64, n_classes: int = 2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def get_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Extract model parameters as a list of numpy arrays."""
    return [param.cpu().detach().numpy() for param in model.parameters()]


def set_parameters(model: nn.Module, parameters: List) -> None:
    """Set model parameters from a list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    test_loader,
    device: torch.device
) -> tuple:
    """Evaluate the model and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy
