# Federated Learning Demo with Flower

A simple demonstration of federated learning using [Flower](https://flower.dev/) with simulated data.

## Overview

This project demonstrates federated learning concepts using:
- **Flower (flwr)**: A federated learning framework
- **PyTorch**: For the neural network model
- **Simulated Data**: Synthetic classification data for quick experimentation

## Project Structure

```
hereditary_fl/
├── data_utils.py        # Data generation and partitioning utilities
├── model.py             # Neural network model definition
├── client.py            # Flower client implementation
├── server.py            # Flower server and strategy setup
├── run_simulation.py    # Main script to run FL simulation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Simulation

Run the federated learning simulation:

```bash
python run_simulation.py
```

This will:
1. Generate synthetic classification data
2. Partition data across 3 simulated clients
3. Run 5 rounds of federated training using FedAvg
4. Display training metrics and final accuracy

## Configuration

You can modify the simulation parameters in `run_simulation.py`:

```python
history = run_fl_simulation(
    n_clients=3,       # Number of federated clients
    n_rounds=5,        # Number of FL rounds
    n_samples=1500,    # Total samples to generate
    n_features=20,     # Features per sample
    n_classes=2,       # Number of classes
    iid=True           # IID (True) or non-IID (False) data distribution
)
```

## Key Concepts

### Federated Averaging (FedAvg)
The default aggregation strategy where:
1. Server sends global model to clients
2. Each client trains locally on their data
3. Clients send model updates back to server
4. Server averages the updates weighted by dataset size

### IID vs Non-IID Data
- **IID**: Data is randomly distributed across clients (each client has similar class distribution)
- **Non-IID**: Data is distributed by class (each client has different class distributions)

Non-IID is more realistic but harder to train. Try `iid=False` to see the difference!

## Next Steps

- Add more sophisticated models
- Implement differential privacy
- Try different aggregation strategies (FedProx, FedOpt)
- Use real datasets (MNIST, CIFAR-10)
- Add client selection strategies
