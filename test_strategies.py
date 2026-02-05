#!/usr/bin/env python3
"""Test FL strategy implementations."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

# Test 1: Verify all strategy imports work
print("=" * 60)
print("Test 1: Verify strategy imports")
print("=" * 60)
try:
    from flwr.server.strategy import FedAvg, FedProx, FedAdagrad, FedAdam, FedYogi, FedAvgM
    print("✅ All Flower strategies imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Verify our Progress wrappers are defined
print("\n" + "=" * 60)
print("Test 2: Verify Progress wrapper classes")
print("=" * 60)
try:
    from app import (
        ProgressFedAvg, ProgressFedProx, 
        ProgressFedAdagrad, ProgressFedAdam, ProgressFedYogi, ProgressFedAvgM
    )
    print("✅ All Progress wrappers imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 3: Verify strategy instantiation with parameters
print("\n" + "=" * 60)
print("Test 3: Verify strategy instantiation")
print("=" * 60)

# Create dummy initial parameters
dummy_params = [np.random.randn(10, 5).astype(np.float32)]
import flwr as fl
initial_params = fl.common.ndarrays_to_parameters(dummy_params)

strategies_to_test = [
    ("FedAvg", lambda: ProgressFedAvg(
        total_rounds=5,
        initial_parameters=initial_params
    )),
    ("FedProx", lambda: ProgressFedProx(
        total_rounds=5,
        initial_parameters=initial_params,
        proximal_mu=0.5
    )),
    ("FedAdagrad", lambda: ProgressFedAdagrad(
        total_rounds=5,
        initial_parameters=initial_params,
        eta=0.01
    )),
    ("FedAdam", lambda: ProgressFedAdam(
        total_rounds=5,
        initial_parameters=initial_params,
        eta=0.01
    )),
    ("FedYogi", lambda: ProgressFedYogi(
        total_rounds=5,
        initial_parameters=initial_params,
        eta=0.01
    )),
    ("FedAvgM", lambda: ProgressFedAvgM(
        total_rounds=5,
        initial_parameters=initial_params,
        server_momentum=0.9
    )),
]

for name, create_fn in strategies_to_test:
    try:
        strategy = create_fn()
        assert hasattr(strategy, 'total_rounds'), f"{name} missing total_rounds"
        assert hasattr(strategy, 'final_parameters'), f"{name} missing final_parameters"
        assert hasattr(strategy, 'aggregate_fit'), f"{name} missing aggregate_fit"
        print(f"✅ {name}: instantiated and has required attributes")
    except Exception as e:
        print(f"❌ {name}: {e}")
        sys.exit(1)

# Test 4: Verify MedMNIST simulation function signature
print("\n" + "=" * 60)
print("Test 4: Verify simulation function signatures")
print("=" * 60)
try:
    from app import run_medmnist_simulation, run_synthetic_simulation
    import inspect
    
    # Check run_medmnist_simulation params
    med_sig = inspect.signature(run_medmnist_simulation)
    med_params = list(med_sig.parameters.keys())
    required_params = ['strategy_name', 'proximal_mu', 'server_lr', 'server_momentum']
    for p in required_params:
        assert p in med_params, f"run_medmnist_simulation missing {p}"
    print(f"✅ run_medmnist_simulation has all strategy parameters")
    
    # Check run_synthetic_simulation params
    syn_sig = inspect.signature(run_synthetic_simulation)
    syn_params = list(syn_sig.parameters.keys())
    for p in required_params:
        assert p in syn_params, f"run_synthetic_simulation missing {p}"
    print(f"✅ run_synthetic_simulation has all strategy parameters")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 5: Quick strategy selection logic test
print("\n" + "=" * 60)
print("Test 5: Strategy selection logic")
print("=" * 60)

def get_strategy_for_name(name, n_rounds=5):
    """Simulate strategy selection logic from app.py"""
    strategy_params = dict(
        total_rounds=n_rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        initial_parameters=initial_params,
    )
    
    if name == "FedProx":
        return ProgressFedProx(proximal_mu=0.5, **strategy_params)
    elif name == "FedAdam":
        return ProgressFedAdam(eta=0.01, **strategy_params)
    elif name == "FedYogi":
        return ProgressFedYogi(eta=0.01, **strategy_params)
    elif name == "FedAdagrad":
        return ProgressFedAdagrad(eta=0.01, **strategy_params)
    elif name == "FedAvgM":
        return ProgressFedAvgM(server_momentum=0.9, **strategy_params)
    else:  # FedAvg
        return ProgressFedAvg(**strategy_params)

for name in ["FedAvg", "FedProx", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]:
    try:
        strategy = get_strategy_for_name(name)
        print(f"✅ {name}: strategy selection works")
    except Exception as e:
        print(f"❌ {name}: {e}")
        sys.exit(1)

print("\n" + "=" * 60)
print("🎉 All tests passed!")
print("=" * 60)
