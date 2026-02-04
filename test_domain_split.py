#!/usr/bin/env python3
"""Test ColorMNIST domain-based partitioning."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medmnist_utils import load_medmnist, partition_colormnist_by_domain
import numpy as np

# Load ColorMNIST with 4 domains
print("Loading ColorMNIST with 4 domains...")
X_train, y_train, domains, X_val, y_val, X_test, y_test = load_medmnist(
    'colormnist', colormnist_n_train=20000, colormnist_n_domains=4
)
print(f"Total samples: {len(X_train)}, Domains: {np.unique(domains)}")
print(f"Samples per domain: {[(d, np.sum(domains==d)) for d in np.unique(domains)]}")

print("\n--- Test 1: 3 clients, 4 domains (clients < domains) ---")
client_data = partition_colormnist_by_domain(X_train, y_train, domains, n_clients=3)
for i, (X_c, y_c) in enumerate(client_data):
    print(f"Client {i}: {len(X_c)} samples")
total = sum(len(X_c) for X_c, _ in client_data)
print(f"Total across clients: {total}")

print("\n--- Test 2: 6 clients, 4 domains (clients > domains) ---")
client_data = partition_colormnist_by_domain(X_train, y_train, domains, n_clients=6)
for i, (X_c, y_c) in enumerate(client_data):
    print(f"Client {i}: {len(X_c)} samples")
total = sum(len(X_c) for X_c, _ in client_data)
print(f"Total across clients: {total}")

print("\n--- Test 3: 4 clients, 4 domains (clients == domains) ---")
client_data = partition_colormnist_by_domain(X_train, y_train, domains, n_clients=4)
for i, (X_c, y_c) in enumerate(client_data):
    print(f"Client {i}: {len(X_c)} samples")
total = sum(len(X_c) for X_c, _ in client_data)
print(f"Total across clients: {total}")

print("\nSUCCESS!")
