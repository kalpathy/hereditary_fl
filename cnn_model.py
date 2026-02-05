"""
CNN models for MedMNIST and ColorMNIST image classification in federated learning.

This module provides:
- MedMNISTCNN: A 3-layer CNN for 28x28 medical/colored images
- SimpleCNN: A smaller 2-layer CNN for faster training
- Utility functions for FL: get_parameters, set_parameters, train_one_epoch, evaluate

The models are designed to work with Flower FL framework:
    - get_parameters(): Extract model weights as NumPy arrays for sending to server
    - set_parameters(): Load NumPy arrays into model (from server aggregation)
    - train_one_epoch(): Client-side local training
    - evaluate(): Model evaluation on test data

Usage:
    model = get_cnn_model("simple", n_channels=3, n_classes=10)
    params = get_parameters(model)  # For sending to FL server
    set_parameters(model, new_params)  # After receiving from server

Author: Kalpathy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MedMNISTCNN(nn.Module):
    """
    A CNN for 28x28 medical images (MedMNIST).
    
    Architecture:
    - 3 convolutional layers with batch norm and max pooling
    - 2 fully connected layers with dropout
    - Suitable for MedMNIST datasets (PathMNIST, DermaMNIST, etc.)
    
    Input: (batch, n_channels, 28, 28)
    Output: (batch, n_classes) - logits (use CrossEntropyLoss)
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 9):
        super(MedMNISTCNN, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # First conv block: 28x28 -> 14x14
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second conv block: 14x14 -> 7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third conv block: 7x7 -> 3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pooling layers: 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SimpleCNN(nn.Module):
    """
    A simpler/smaller CNN for faster training.
    Good for quick experiments.
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 9):
        super(SimpleCNN, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Conv layers
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 2 pooling: 28 -> 14 -> 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResBlock(nn.Module):
    """Residual block for ResNet-style CNN."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MedMNISTResNet(nn.Module):
    """
    A small ResNet-style network for MedMNIST.
    More powerful than the basic CNN but still lightweight.
    """
    
    def __init__(self, n_channels: int = 3, n_classes: int = 9):
        super(MedMNISTResNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Initial conv
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)  # 28 -> 14
        self.layer3 = self._make_layer(64, 128, 2, stride=2)  # 14 -> 7
        
        # Global average pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, n_classes)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        current_channels = in_channels
        for s in strides:
            layers.append(ResBlock(current_channels, out_channels, s))
            current_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_cnn_model(
    model_name: str = "simple",
    n_channels: int = 3,
    n_classes: int = 9
) -> nn.Module:
    """
    Factory function to get a CNN model.
    
    Args:
        model_name: One of "simple", "cnn", "resnet"
        n_channels: Number of input channels (1 for grayscale, 3 for RGB)
        n_classes: Number of output classes
    
    Returns:
        PyTorch model
    """
    models = {
        "simple": SimpleCNN,
        "cnn": MedMNISTCNN,
        "resnet": MedMNISTResNet,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](n_channels=n_channels, n_classes=n_classes)


def get_parameters(model: nn.Module) -> List[torch.Tensor]:
    """Extract model parameters as a list of numpy arrays (includes all state)."""
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


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
            
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def evaluate_with_predictions(
    model: nn.Module,
    test_loader,
    device: torch.device
) -> tuple:
    """Evaluate model and return loss, accuracy, all predictions and true labels."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy, all_preds, all_labels


if __name__ == "__main__":
    # Test the models
    print("Testing CNN models...")
    
    # Test with random data
    batch_size = 8
    
    for model_name in ["simple", "cnn", "resnet"]:
        print(f"\nTesting {model_name} model:")
        
        # RGB input
        model = get_cnn_model(model_name, n_channels=3, n_classes=9)
        x = torch.randn(batch_size, 3, 28, 28)
        y = model(x)
        print(f"  RGB input: {x.shape} -> output: {y.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Grayscale input
        model = get_cnn_model(model_name, n_channels=1, n_classes=2)
        x = torch.randn(batch_size, 1, 28, 28)
        y = model(x)
        print(f"  Grayscale input: {x.shape} -> output: {y.shape}")
    
    print("\nCNN models test complete!")
