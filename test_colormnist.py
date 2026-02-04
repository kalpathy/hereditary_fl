#!/usr/bin/env python3
"""Test ColorMNIST loading and partitioning."""

from medmnist_utils import (
    load_medmnist, 
    partition_colormnist_by_domain, 
    get_colormnist_domain_info,
    partition_medmnist_iid
)
import numpy as np

medmnist_dataset = "colormnist"
color_correlation = 0.9
colormnist_samples = 30000
colormnist_n_domains = 4
n_clients = 3
use_domain_split = True

print(f"Loading {medmnist_dataset}...")
print(f"  correlation={color_correlation}")
print(f"  samples={colormnist_samples}")
print(f"  n_domains={colormnist_n_domains}")

try:
    X_train, y_train, domains_train, X_val, y_val, X_test, y_test = load_medmnist(
        medmnist_dataset,
        colormnist_correlation=color_correlation,
        colormnist_n_train=colormnist_samples,
        colormnist_n_domains=colormnist_n_domains
    )
    print(f"Loaded: X_train={X_train.shape}, domains={domains_train.shape}")
    
    if use_domain_split:
        client_data = partition_colormnist_by_domain(X_train, y_train, domains_train, n_clients)
        domain_info = get_colormnist_domain_info(domains_train, n_clients)
        print(f"Domain info: {domain_info}")
    else:
        client_data = partition_medmnist_iid(X_train, y_train, n_clients)
    
    print(f"Client data: {len(client_data)} clients")
    for i, (X_i, y_i) in enumerate(client_data):
        print(f"  Client {i}: {len(X_i)} samples")
    
    print("SUCCESS!")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
