"""
Streamlit Dashboard for Federated Learning Simulation with MedMNIST support.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import os
import threading
import queue
from typing import Dict, List, Tuple, Optional
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

# FL imports
import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Metrics, Parameters, Scalar, FitRes, EvaluateRes
from flwr.server.client_proxy import ClientProxy

# Progress tracking file
PROGRESS_FILE = Path(__file__).parent / ".fl_progress.json"

def update_progress(current_round: int, total_rounds: int, stage: str = "training"):
    """Update progress file for UI to read."""
    progress = {
        "current_round": current_round,
        "total_rounds": total_rounds,
        "stage": stage,
        "timestamp": time.time()
    }
    PROGRESS_FILE.write_text(json.dumps(progress))

def read_progress():
    """Read current progress from file."""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except:
            return None
    return None

def clear_progress():
    """Clear progress file."""
    if PROGRESS_FILE.exists():
        try:
            PROGRESS_FILE.unlink()
        except:
            pass


class ProgressFedAvg(FedAvg):
    """FedAvg with progress tracking."""
    def __init__(self, total_rounds: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_rounds = total_rounds
        self.current_round = 0
        self.final_parameters = None  # Store final aggregated parameters
    
    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        update_progress(server_round, self.total_rounds, f"Round {server_round}/{self.total_rounds}: Aggregating")
        aggregated = super().aggregate_fit(server_round, results, failures)
        # Store the aggregated parameters after each round
        if aggregated is not None and aggregated[0] is not None:
            self.final_parameters = aggregated[0]
        return aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        update_progress(server_round, self.total_rounds, f"Round {server_round}/{self.total_rounds}: Evaluating")
        return super().aggregate_evaluate(server_round, results, failures)


class ProgressFedProx(FedProx):
    """FedProx with progress tracking."""
    def __init__(self, total_rounds: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_rounds = total_rounds
        self.current_round = 0
        self.final_parameters = None  # Store final aggregated parameters
    
    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        update_progress(server_round, self.total_rounds, f"Round {server_round}/{self.total_rounds}: Aggregating")
        aggregated = super().aggregate_fit(server_round, results, failures)
        # Store the aggregated parameters after each round
        if aggregated is not None and aggregated[0] is not None:
            self.final_parameters = aggregated[0]
        return aggregated
    
    def aggregate_evaluate(self, server_round, results, failures):
        update_progress(server_round, self.total_rounds, "evaluating")
        return super().aggregate_evaluate(server_round, results, failures)

# Local imports - synthetic data
from data_utils import generate_simulated_data, partition_data, create_data_loaders
from model import SimpleNet, get_parameters, set_parameters

# Local imports - MedMNIST
from medmnist_utils import (
    MEDMNIST_DATASETS, load_medmnist, 
    partition_medmnist_iid, partition_medmnist_non_iid,
    partition_colormnist_by_domain, get_colormnist_domain_info,
    create_medmnist_dataloaders, get_dataset_info, get_client_distribution
)
from cnn_model import (
    get_cnn_model, get_parameters as get_cnn_parameters,
    set_parameters as set_cnn_parameters, train_one_epoch, evaluate,
    evaluate_with_predictions
)
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Page config
st.set_page_config(
    page_title="FL Simulation Dashboard",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Federated Learning Dashboard")
st.markdown("Configure and monitor federated learning simulations with Flower")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = None
if "running" not in st.session_state:
    st.session_state.running = False
if "partitions_info" not in st.session_state:
    st.session_state.partitions_info = None
if "medmnist_loaded" not in st.session_state:
    st.session_state.medmnist_loaded = None
if "detailed_results" not in st.session_state:
    st.session_state.detailed_results = None


# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Dataset selection
st.sidebar.subheader("📊 Dataset")
dataset_type = st.sidebar.selectbox(
    "Dataset Type",
    ["Synthetic (Tabular)", "MedMNIST (Medical Images)", "ColorMNIST (Colored Digits)"]
)

if dataset_type == "Synthetic (Tabular)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Synthetic Dataset Settings")
    n_samples = st.sidebar.slider("Total Samples", 500, 5000, 1500, step=100)
    n_features = st.sidebar.slider("Number of Features", 5, 50, 20)
    n_classes = st.sidebar.slider("Number of Classes", 2, 10, 2)
    
    # Placeholders for MedMNIST variables
    medmnist_dataset = None
    model_type = "simple"
    n_channels = 1
    color_correlation = 0.9
    colormnist_samples = 30000
    colormnist_n_domains = 4
    use_domain_split = False

elif dataset_type == "ColorMNIST (Colored Digits)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 ColorMNIST Settings")
    st.sidebar.info("""
    **ColorMNIST**
    - Colored MNIST digits (28x28 RGB)
    - 10 classes (digits 0-9)
    - **Multiple domains** with varying color-label correlations
    - Great for studying spurious correlations & domain shift!
    """)
    
    # ColorMNIST specific settings
    color_correlation = st.sidebar.slider(
        "Max Color-Label Correlation", 
        0.5, 1.0, 0.9, 0.1,
        help="Maximum correlation (Domain 0). Other domains decrease from this value."
    )
    colormnist_samples = st.sidebar.slider("Training Samples", 10000, 50000, 30000, step=5000)
    colormnist_n_domains = st.sidebar.slider(
        "Number of Domains", 2, 6, 4,
        help="Each domain has different color-label correlation. Domains are distributed to clients."
    )
    use_domain_split = st.sidebar.checkbox(
        "Split by Domain", 
        value=True,
        help="If checked, each client gets data from specific domains (natural non-IID). If unchecked, uses standard IID/non-IID split."
    )
    
    medmnist_dataset = "colormnist"
    n_classes = 10
    n_channels = 3
    n_features = None
    n_samples = None
    
    # Model selection
    st.sidebar.subheader("🧠 CNN Model")
    model_type = st.sidebar.selectbox(
        "Model Architecture",
        ["simple", "cnn", "resnet"],
        format_func=lambda x: {
            "simple": "SimpleCNN (Fast, ~50K params)",
            "cnn": "MedMNIST-CNN (Medium, ~300K params)", 
            "resnet": "ResNet-style (Powerful, ~500K params)"
        }[x]
    )
    
else:  # MedMNIST
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 MedMNIST Dataset")
    
    # Dataset selection with descriptions - exclude colormnist
    medmnist_options = {
        name: f"{info['name']} - {info['description']}"
        for name, info in MEDMNIST_DATASETS.items()
        if name != "colormnist"
    }
    medmnist_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=list(medmnist_options.keys()),
        format_func=lambda x: medmnist_options[x]
    )
    
    # Show dataset info
    dataset_info = MEDMNIST_DATASETS[medmnist_dataset]
    st.sidebar.info(f"""
    **{dataset_info['name']}**
    - Channels: {dataset_info['n_channels']} ({'RGB' if dataset_info['n_channels'] == 3 else 'Grayscale'})
    - Classes: {dataset_info['n_classes']}
    - Task: {dataset_info['task']}
    """)
    
    n_classes = dataset_info['n_classes']
    n_channels = dataset_info['n_channels']
    n_features = None
    n_samples = None
    color_correlation = 0.9
    colormnist_samples = 30000
    colormnist_n_domains = 4
    use_domain_split = False
    
    # Model selection
    st.sidebar.subheader("🧠 CNN Model")
    model_type = st.sidebar.selectbox(
        "Model Architecture",
        ["simple", "cnn", "resnet"],
        format_func=lambda x: {
            "simple": "SimpleCNN (Fast, ~50K params)",
            "cnn": "MedMNIST-CNN (Medium, ~300K params)", 
            "resnet": "ResNet-style (Powerful, ~500K params)"
        }[x]
    )

st.sidebar.markdown("---")
st.sidebar.subheader("👥 Client Settings")
n_clients = st.sidebar.slider("Number of Clients", 2, 10, 3)

# For ColorMNIST with domain split, IID option is not applicable
if dataset_type == "ColorMNIST (Colored Digits)" and use_domain_split:
    st.sidebar.info("📍 Using domain-based split (natural non-IID from varying correlations)")
    iid = False  # Domain split is inherently non-IID
    alpha = 0.5  # Not used with domain split
else:
    iid = st.sidebar.checkbox("IID Data Distribution", value=True)
    
    if not iid and dataset_type in ["MedMNIST (Medical Images)", "ColorMNIST (Colored Digits)"]:
        alpha = st.sidebar.slider(
            "Dirichlet α (lower = more non-IID)", 
            0.1, 2.0, 0.5, step=0.1,
            help="Controls data heterogeneity. Lower values create more non-IID distributions."
        )
    else:
        alpha = 0.5

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Training Settings")
n_rounds = st.sidebar.slider("Number of Rounds", 1, 20, 5)
local_epochs = st.sidebar.slider("Local Epochs per Round", 1, 5, 1)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 32, step=8)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    value=0.001 if dataset_type == "MedMNIST (Medical Images)" else 0.01
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Strategy")
strategy_name = st.sidebar.selectbox("Algorithm", ["FedAvg", "FedProx"])
if strategy_name == "FedProx":
    proximal_mu = st.sidebar.slider("Proximal μ (mu)", 0.01, 1.0, 0.5, step=0.01)
else:
    proximal_mu = 0.0


# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Data Distribution Preview")
    
    if st.button("🔄 Load/Generate Data Preview"):
        with st.spinner("Loading data..."):
            if dataset_type == "Synthetic (Tabular)":
                X, y = generate_simulated_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_classes=n_classes,
                    random_seed=42
                )
                partitions = partition_data(X, y, n_clients=n_clients, iid=iid, random_seed=42)
                
                st.session_state.partitions_info = []
                for i, (X_i, y_i) in enumerate(partitions):
                    class_dist = np.bincount(y_i, minlength=n_classes)
                    st.session_state.partitions_info.append({
                        "Client": f"Client {i}",
                        "Samples": len(X_i),
                        **{f"Class {j}": int(class_dist[j]) for j in range(n_classes)}
                    })
                st.session_state.medmnist_loaded = None
                
            else:  # MedMNIST or ColorMNIST
                # Pass ColorMNIST-specific params if applicable
                if medmnist_dataset == "colormnist":
                    X_train, y_train, domains_train, X_val, y_val, X_test, y_test = load_medmnist(
                        medmnist_dataset,
                        colormnist_correlation=color_correlation,
                        colormnist_n_train=colormnist_samples,
                        colormnist_n_domains=colormnist_n_domains
                    )
                    
                    # Partition by domain or standard method
                    if use_domain_split:
                        client_data = partition_colormnist_by_domain(X_train, y_train, domains_train, n_clients)
                        domain_info = get_colormnist_domain_info(domains_train, n_clients)
                        st.info(f"🌈 Using domain-based split: {domain_info['n_domains']} domains distributed to {n_clients} clients")
                    elif iid:
                        client_data = partition_medmnist_iid(X_train, y_train, n_clients)
                    else:
                        client_data = partition_medmnist_non_iid(X_train, y_train, n_clients, alpha=alpha)
                else:
                    X_train, y_train, domains_train, X_val, y_val, X_test, y_test = load_medmnist(medmnist_dataset)
                    
                    # Standard partitioning for MedMNIST
                    if iid:
                        client_data = partition_medmnist_iid(X_train, y_train, n_clients)
                    else:
                        client_data = partition_medmnist_non_iid(X_train, y_train, n_clients, alpha=alpha)
                
                # Store loaded data
                st.session_state.medmnist_loaded = {
                    "dataset": medmnist_dataset,
                    "client_data": client_data,
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "n_channels": n_channels,
                    "n_classes": n_classes,
                    "iid_setting": "domain-split" if (medmnist_dataset == "colormnist" and use_domain_split) else ("iid" if iid else f"non-iid-{alpha}")
                }
                
                # Create partition info
                st.session_state.partitions_info = []
                for i, (X_i, y_i) in enumerate(client_data):
                    class_dist = np.bincount(y_i, minlength=n_classes)
                    st.session_state.partitions_info.append({
                        "Client": f"Client {i}",
                        "Samples": len(X_i),
                        **{f"Class {j}": int(class_dist[j]) for j in range(n_classes)}
                    })
                
                st.success(f"Loaded {medmnist_dataset}: {len(X_train)} train, {len(X_test)} test samples")
    
    if st.session_state.partitions_info:
        df = pd.DataFrame(st.session_state.partitions_info)
        st.dataframe(df, use_container_width=True)
        
        # Visualize distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        class_cols = [col for col in df.columns if col.startswith("Class")]
        df_plot = df.set_index("Client")[class_cols]
        df_plot.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_xlabel("Client")
        ax.set_ylabel("Number of Samples")
        ax.set_title(f"Data Distribution ({'IID' if iid else f'Non-IID (α={alpha})'})")
        ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with col2:
    st.subheader("📊 Configuration Summary")
    
    if dataset_type == "Synthetic (Tabular)":
        config_data = {
            "Parameter": [
                "Dataset Type", "Total Samples", "Features", "Classes", "Clients",
                "Data Distribution", "Rounds", "Local Epochs",
                "Batch Size", "Learning Rate", "Strategy", "Model"
            ],
            "Value": [
                "Synthetic", str(n_samples), str(n_features), str(n_classes), str(n_clients),
                "IID" if iid else "Non-IID", str(n_rounds), str(local_epochs),
                str(batch_size), str(learning_rate),
                f"{strategy_name}" + (f" (μ={proximal_mu})" if strategy_name == "FedProx" else ""),
                "SimpleNet (MLP)"
            ]
        }
    else:
        config_data = {
            "Parameter": [
                "Dataset Type", "MedMNIST Dataset", "Channels", "Classes", "Clients",
                "Data Distribution", "Rounds", "Local Epochs",
                "Batch Size", "Learning Rate", "Strategy", "Model"
            ],
            "Value": [
                "MedMNIST", medmnist_dataset, str(n_channels), str(n_classes), str(n_clients),
                "IID" if iid else f"Non-IID (α={alpha})", str(n_rounds), str(local_epochs),
                str(batch_size), str(learning_rate),
                f"{strategy_name}" + (f" (μ={proximal_mu})" if strategy_name == "FedProx" else ""),
                {"simple": "SimpleCNN", "cnn": "MedMNIST-CNN", "resnet": "ResNet"}[model_type]
            ]
        }
    
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    # Show sample images for MedMNIST
    if dataset_type == "MedMNIST (Medical Images)" and st.session_state.medmnist_loaded:
        st.subheader("🖼️ Sample Images")
        client_data = st.session_state.medmnist_loaded["client_data"]
        
        # Show samples from first client
        X_sample, y_sample = client_data[0]
        fig, axes = plt.subplots(1, 5, figsize=(12, 3))
        for i, ax in enumerate(axes):
            if i < len(X_sample):
                img = X_sample[i]
                if img.ndim == 2:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
                ax.set_title(f"Class {y_sample[i]}", fontsize=10)
                ax.axis('off')
        plt.suptitle("Sample images from Client 0", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()

# Run simulation section
st.subheader("🚀 Run Simulation")


def create_medmnist_client_fn(client_data, n_channels, n_classes, model_type, 
                               batch_size, local_epochs, learning_rate, 
                               use_fedprox=False, mu=0.0):
    """Create a client function for MedMNIST FL simulation."""
    
    def client_fn(cid: str):
        client_id = int(cid)
        X, y = client_data[client_id]
        
        # Create data loaders
        train_loader, test_loader = create_medmnist_dataloaders(
            X, y, n_channels, batch_size=batch_size
        )
        
        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_cnn_model(model_type, n_channels, n_classes).to(device)
        
        class MedMNISTClient(fl.client.NumPyClient):
            def __init__(self):
                self.train_loader = train_loader
                self.test_loader = test_loader
                self.model = model
                self.device = device
                self.local_epochs = local_epochs
                self.lr = learning_rate
                self.mu = mu
                self.use_fedprox = use_fedprox
            
            def get_parameters(self, config):
                return get_cnn_parameters(self.model)
            
            def fit(self, parameters, config):
                set_cnn_parameters(self.model, parameters)
                
                # Store global parameters for FedProx
                if self.use_fedprox:
                    global_params = [p.clone() for p in self.model.parameters()]
                
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                criterion = torch.nn.CrossEntropyLoss()
                
                self.model.train()
                for epoch in range(self.local_epochs):
                    epoch_loss = 0.0
                    for X_batch, y_batch in self.train_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        
                        # Add proximal term for FedProx
                        if self.use_fedprox and self.mu > 0:
                            proximal_term = 0.0
                            for p, g in zip(self.model.parameters(), global_params):
                                proximal_term += ((p - g.to(self.device)) ** 2).sum()
                            loss += (self.mu / 2) * proximal_term
                        
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                
                return get_cnn_parameters(self.model), len(self.train_loader.dataset), {}
            
            def evaluate(self, parameters, config):
                set_cnn_parameters(self.model, parameters)
                loss, accuracy = evaluate(self.model, self.test_loader, self.device)
                return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}
        
        return MedMNISTClient()
    
    return client_fn


def run_medmnist_simulation(client_data, n_channels, n_classes, model_type,
                            n_clients, n_rounds, batch_size, local_epochs,
                            learning_rate, strategy_name, proximal_mu):
    """Run FL simulation with MedMNIST data."""
    
    # Clear any previous progress
    clear_progress()
    update_progress(0, n_rounds, "initializing")
    
    # Create initial model and parameters
    initial_model = get_cnn_model(model_type, n_channels, n_classes)
    initial_params = fl.common.ndarrays_to_parameters(get_cnn_parameters(initial_model))
    
    # Metrics aggregation
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    # Create strategy with progress tracking
    use_fedprox = strategy_name == "FedProx"
    
    if use_fedprox:
        strategy = ProgressFedProx(
            total_rounds=n_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=n_clients,
            min_evaluate_clients=n_clients,
            min_available_clients=n_clients,
            initial_parameters=initial_params,
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=proximal_mu,
        )
    else:
        strategy = ProgressFedAvg(
            total_rounds=n_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=n_clients,
            min_evaluate_clients=n_clients,
            min_available_clients=n_clients,
            initial_parameters=initial_params,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
    # Create client function
    client_fn = create_medmnist_client_fn(
        client_data, n_channels, n_classes, model_type,
        batch_size, local_epochs, learning_rate,
        use_fedprox=use_fedprox, mu=proximal_mu
    )
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    # Compute detailed results for visualization
    update_progress(n_rounds, n_rounds, "Computing detailed metrics...")
    
    detailed_results = None
    try:
        # Get final parameters from strategy - use our tracked final_parameters
        final_params = None
        if hasattr(strategy, 'final_parameters') and strategy.final_parameters is not None:
            final_params = fl.common.parameters_to_ndarrays(strategy.final_parameters)
            print(f"Loaded final parameters: {len(final_params)} arrays")
        else:
            print("Warning: No final parameters found in strategy")
        
        detailed_results = compute_detailed_results(
            final_params, client_data, n_channels, n_classes, model_type,
            batch_size, n_clients, local_epochs=local_epochs, learning_rate=learning_rate
        )
    except Exception as e:
        print(f"Warning: Could not compute detailed results: {e}")
        import traceback
        traceback.print_exc()
    
    return history, detailed_results


def compute_detailed_results(final_params, client_data, n_channels, n_classes, 
                             model_type, batch_size, n_clients, local_epochs=3, learning_rate=0.001):
    """Compute confusion matrix and per-client performance after FL training.
    Also trains local-only models for comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {
        "global_model": {},
        "client_models": [],
        "n_classes": n_classes
    }
    
    # Create global model and load final parameters
    global_model = get_cnn_model(model_type, n_channels, n_classes)
    if final_params is not None:
        try:
            set_cnn_parameters(global_model, final_params)
        except Exception as e:
            print(f"Warning: Could not set parameters: {e}")
    global_model.to(device)
    global_model.eval()
    
    # Evaluate global model on each client's data AND train local models
    all_global_preds = []
    all_global_labels = []
    
    for cid in range(n_clients):
        X_client, y_client = client_data[cid]
        
        # Create test loader for this client - ensure writable arrays
        from torch.utils.data import DataLoader, TensorDataset
        X_copy = np.array(X_client, copy=True)
        y_copy = np.array(y_client, copy=True).flatten()
        
        # Preprocess images: ensure 4D, normalize, and transpose to (N, C, H, W)
        if X_copy.ndim == 3:
            X_copy = X_copy[..., np.newaxis]
        X_copy = X_copy.astype(np.float32) / 255.0
        X_copy = np.transpose(X_copy, (0, 3, 1, 2))  # NHWC -> NCHW
        
        client_dataset = TensorDataset(
            torch.FloatTensor(X_copy),
            torch.LongTensor(y_copy)
        )
        
        # Split into train/test for local model evaluation (80/20)
        n_samples = len(client_dataset)
        n_train = int(0.8 * n_samples)
        n_test = n_samples - n_train
        train_dataset, test_dataset = torch.utils.data.random_split(
            client_dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        full_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluate global model on client's FULL data
        loss, acc, preds, labels = evaluate_with_predictions(global_model, full_loader, device)
        
        all_global_preds.extend(preds)
        all_global_labels.extend(labels)
        
        # Train a local-only model on this client's training data
        local_model = get_cnn_model(model_type, n_channels, n_classes)
        local_model.to(device)
        local_model.train()
        
        optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(local_epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = local_model(X_batch)
                loss_val = criterion(outputs, y_batch)
                loss_val.backward()
                optimizer.step()
        
        # Evaluate local model on client's test data
        local_model.eval()
        local_loss, local_acc, local_preds, local_labels = evaluate_with_predictions(
            local_model, test_loader, device
        )
        
        # Also evaluate global model on test set for fair comparison
        global_test_loss, global_test_acc, _, _ = evaluate_with_predictions(
            global_model, test_loader, device
        )
        
        # Count unique classes in this client's data
        unique_classes = len(np.unique(y_copy))
        
        results["client_models"].append({
            "client_id": cid,
            "n_samples": len(X_client),
            "n_train": n_train,
            "n_test": n_test,
            "unique_classes": unique_classes,
            "global_model_accuracy": acc * 100,  # On full data
            "global_model_loss": loss,
            "global_model_test_accuracy": global_test_acc * 100,  # On test split
            "local_model_accuracy": local_acc * 100,  # Local model on test split
            "local_model_loss": local_loss,
            "predictions": preds,
            "labels": labels
        })
    
    # Compute overall confusion matrix for global model
    results["global_model"]["confusion_matrix"] = confusion_matrix(
        all_global_labels, all_global_preds, labels=list(range(n_classes))
    )
    results["global_model"]["all_preds"] = all_global_preds
    results["global_model"]["all_labels"] = all_global_labels
    results["global_model"]["overall_accuracy"] = (
        np.array(all_global_preds) == np.array(all_global_labels)
    ).mean() * 100
    
    # Add class distribution info
    unique, counts = np.unique(all_global_labels, return_counts=True)
    results["class_distribution"] = dict(zip(unique.tolist(), counts.tolist()))
    
    return results


def run_synthetic_simulation(n_clients, n_rounds, n_samples, n_features, n_classes,
                             iid, strategy_name, proximal_mu, local_epochs, 
                             batch_size, learning_rate):
    """Run FL simulation with synthetic data."""
    from client import create_client_fn
    from fedprox_client import create_fedprox_client_fn
    
    # Clear any previous progress
    clear_progress()
    update_progress(0, n_rounds, "initializing")
    
    # Generate data
    X, y = generate_simulated_data(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_seed=42
    )
    partitions = partition_data(X, y, n_clients=n_clients, iid=iid, random_seed=42)
    
    # Create initial parameters
    initial_model = SimpleNet(n_features=n_features, n_classes=n_classes)
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(initial_model))
    
    # Metrics aggregation
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    # Create strategy with progress tracking
    if strategy_name == "FedProx":
        strategy = ProgressFedProx(
            total_rounds=n_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=n_clients,
            min_evaluate_clients=n_clients,
            min_available_clients=n_clients,
            initial_parameters=initial_params,
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=proximal_mu,
        )
        client_fn = create_fedprox_client_fn(
            partitions, n_features=n_features, n_classes=n_classes, mu=proximal_mu
        )
    else:
        strategy = ProgressFedAvg(
            total_rounds=n_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=n_clients,
            min_evaluate_clients=n_clients,
            min_available_clients=n_clients,
            initial_parameters=initial_params,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        client_fn = create_client_fn(partitions, n_features=n_features, n_classes=n_classes)
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    return history


# Check if data is loaded for MedMNIST
can_run = True
if dataset_type == "MedMNIST (Medical Images)" and st.session_state.medmnist_loaded is None:
    st.warning("⚠️ Please load the MedMNIST dataset first by clicking 'Load/Generate Data Preview'")
    can_run = False

# Store current IID setting to detect changes
current_iid_setting = "iid" if iid else f"non-iid-{alpha}"
if dataset_type == "MedMNIST (Medical Images)" and st.session_state.medmnist_loaded is not None:
    stored_setting = st.session_state.medmnist_loaded.get("iid_setting", None)
    if stored_setting != current_iid_setting:
        st.info("ℹ️ IID setting changed. Click 'Load/Generate Data Preview' to repartition data, or the simulation will repartition automatically.")

if st.button("▶️ Start Simulation", type="primary", disabled=st.session_state.running or not can_run):
    st.session_state.running = True
    
    # Clear previous progress
    clear_progress()
    update_progress(0, n_rounds, "Initializing...")
    
    # Create placeholders for live updates
    progress_bar = st.progress(0, text="🚀 Initializing simulation...")
    status_text = st.empty()
    
    # Copy data BEFORE starting thread (to avoid session_state access in thread)
    medmnist_data = None
    if dataset_type == "MedMNIST (Medical Images)" and st.session_state.medmnist_loaded:
        medmnist_data = {
            "X_train": st.session_state.medmnist_loaded["X_train"],
            "y_train": st.session_state.medmnist_loaded["y_train"],
            "client_data": st.session_state.medmnist_loaded["client_data"],
            "n_channels": st.session_state.medmnist_loaded["n_channels"],
            "n_classes": st.session_state.medmnist_loaded["n_classes"],
            "iid_setting": st.session_state.medmnist_loaded.get("iid_setting"),
        }
    
    # Use dict to store results (mutable, works in nested function)
    sim_state = {"result": None, "error": None, "loaded": None}
    
    # Capture all parameters for the thread
    sim_params = {
        "dataset_type": dataset_type,
        "n_clients": n_clients,
        "n_rounds": n_rounds,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "iid": iid,
        "alpha": alpha,
        "strategy_name": strategy_name,
        "proximal_mu": proximal_mu,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_type": model_type if dataset_type in ["MedMNIST (Medical Images)", "ColorMNIST (Colored Digits)"] else None,
        "use_domain_split": use_domain_split if dataset_type == "ColorMNIST (Colored Digits)" else False,
    }
    
    def run_simulation_thread(params, medmnist_data, sim_state):
        """Run simulation in background thread."""
        try:
            if params["dataset_type"] == "Synthetic (Tabular)":
                sim_state["result"] = run_synthetic_simulation(
                    n_clients=params["n_clients"],
                    n_rounds=params["n_rounds"],
                    n_samples=params["n_samples"],
                    n_features=params["n_features"],
                    n_classes=params["n_classes"],
                    iid=params["iid"],
                    strategy_name=params["strategy_name"],
                    proximal_mu=params["proximal_mu"],
                    local_epochs=params["local_epochs"],
                    batch_size=params["batch_size"],
                    learning_rate=params["learning_rate"]
                )
            else:
                loaded = medmnist_data.copy()
                
                # Determine current partitioning setting
                if params.get("use_domain_split") and loaded.get("dataset") == "colormnist":
                    current_partition = "domain-split"
                else:
                    current_partition = "iid" if params["iid"] else f"non-iid-{params['alpha']}"
                
                # Check if we need to repartition
                if loaded.get("iid_setting") != current_partition:
                    update_progress(0, params["n_rounds"], "Repartitioning data...")
                    X_train = loaded["X_train"]
                    y_train = loaded["y_train"]
                    
                    # Domain-based split is already handled at load time
                    # Only standard IID/non-IID repartitioning here
                    if params["iid"]:
                        client_data = partition_medmnist_iid(X_train, y_train, params["n_clients"])
                    else:
                        client_data = partition_medmnist_non_iid(X_train, y_train, params["n_clients"], alpha=params["alpha"])
                    loaded["client_data"] = client_data
                    loaded["iid_setting"] = current_partition
                else:
                    client_data = loaded["client_data"]
                
                sim_state["loaded"] = loaded
                sim_state["result"] = run_medmnist_simulation(
                    client_data=client_data,
                    n_channels=loaded["n_channels"],
                    n_classes=loaded["n_classes"],
                    model_type=params["model_type"],
                    n_clients=params["n_clients"],
                    n_rounds=params["n_rounds"],
                    batch_size=params["batch_size"],
                    local_epochs=params["local_epochs"],
                    learning_rate=params["learning_rate"],
                    strategy_name=params["strategy_name"],
                    proximal_mu=params["proximal_mu"]
                )
        except Exception as e:
            import traceback
            sim_state["error"] = traceback.format_exc()
    
    # Start simulation in background thread
    sim_thread = threading.Thread(
        target=run_simulation_thread, 
        args=(sim_params, medmnist_data, sim_state)
    )
    sim_thread.start()
    
    # Poll for progress updates while simulation runs
    last_round = 0
    while sim_thread.is_alive():
        progress = read_progress()
        if progress:
            current = progress.get("current_round", 0)
            total = progress.get("total_rounds", n_rounds)
            stage = progress.get("stage", "Training...")
            
            pct = int((current / total) * 100) if total > 0 else 0
            progress_bar.progress(pct, text=f"🔄 {stage}")
            
            if current > last_round:
                status_text.info(f"📊 {stage}")
                last_round = current
        
        time.sleep(0.3)  # Poll every 300ms
    
    # Wait for thread to finish
    sim_thread.join()
    
    # Handle results
    if sim_state["error"]:
        st.error(f"❌ Error during simulation:")
        st.code(sim_state["error"])
    elif sim_state["result"]:
        # MedMNIST returns (history, detailed_results), Synthetic returns just history
        if isinstance(sim_state["result"], tuple):
            history, detailed_results = sim_state["result"]
            st.session_state.history = history
            st.session_state.detailed_results = detailed_results
        else:
            st.session_state.history = sim_state["result"]
            st.session_state.detailed_results = None
        
        progress_bar.progress(100, text=f"✅ Completed all {n_rounds} rounds!")
        status_text.success(f"✅ Federated learning completed successfully!")
        if dataset_type == "MedMNIST (Medical Images)" and sim_state["loaded"]:
            st.session_state.medmnist_loaded.update(sim_state["loaded"])
    
    clear_progress()
    st.session_state.running = False
    st.rerun()

# Display results
if st.session_state.history is not None:
    st.divider()
    st.subheader("📊 Training Results")
    
    history = st.session_state.history
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Loss plot
        if history.losses_distributed:
            fig, ax = plt.subplots(figsize=(8, 5))
            rounds = [r for r, _ in history.losses_distributed]
            losses = [loss for _, loss in history.losses_distributed]
            
            ax.plot(rounds, losses, 'b-o', linewidth=2, markersize=8)
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title("Training Loss per Round", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(rounds)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with col2:
        # Accuracy plot
        if history.metrics_distributed and "accuracy" in history.metrics_distributed:
            fig, ax = plt.subplots(figsize=(8, 5))
            acc_data = history.metrics_distributed["accuracy"]
            rounds = [r for r, _ in acc_data]
            accuracies = [acc * 100 for _, acc in acc_data]
            
            ax.plot(rounds, accuracies, 'g-o', linewidth=2, markersize=8)
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_title("Distributed Accuracy per Round", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 105])
            ax.set_xticks(rounds)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    results_data = []
    for i, (r, loss) in enumerate(history.losses_distributed):
        acc = None
        if history.metrics_distributed and "accuracy" in history.metrics_distributed:
            acc = history.metrics_distributed["accuracy"][i][1] * 100
        results_data.append({
            "Round": r,
            "Loss": f"{loss:.6f}",
            "Accuracy (%)": f"{acc:.2f}" if acc else "N/A"
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    if history.losses_distributed:
        final_loss = history.losses_distributed[-1][1]
        initial_loss = history.losses_distributed[0][1]
        loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
        
        col1.metric("Final Loss", f"{final_loss:.6f}")
        col2.metric("Loss Reduction", f"{loss_reduction:.1f}%")
    else:
        col1.metric("Final Loss", "N/A")
        col2.metric("Loss Reduction", "N/A")
    
    if history.metrics_distributed and "accuracy" in history.metrics_distributed and history.metrics_distributed["accuracy"]:
        final_acc = history.metrics_distributed["accuracy"][-1][1] * 100
        col3.metric("Final Accuracy", f"{final_acc:.2f}%")
    else:
        col3.metric("Final Accuracy", "N/A")

# Display detailed results (confusion matrix, per-client comparison)
if st.session_state.detailed_results is not None:
    st.divider()
    st.subheader("🔬 Detailed Analysis")
    
    detailed = st.session_state.detailed_results
    
    tab1, tab2 = st.tabs(["📊 Confusion Matrix", "🏥 Per-Site Performance"])
    
    with tab1:
        st.markdown("### Global Model Confusion Matrix")
        st.markdown("Shows how well the federated (global) model classifies each class across all clients' data.")
        
        cm = detailed["global_model"]["confusion_matrix"]
        n_classes = detailed["n_classes"]
        
        # Create confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=[f'Class {i}' for i in range(n_classes)],
                    yticklabels=[f'Class {i}' for i in range(n_classes)])
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix - Global Federated Model', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Per-class accuracy
        st.markdown("### Per-Class Performance")
        class_accuracy = []
        for i in range(n_classes):
            total_class = cm[i].sum()
            correct_class = cm[i, i]
            acc = (correct_class / total_class * 100) if total_class > 0 else 0
            class_accuracy.append({
                "Class": i,
                "Samples": total_class,
                "Correct": correct_class,
                "Accuracy (%)": f"{acc:.1f}"
            })
        
        class_df = pd.DataFrame(class_accuracy)
        st.dataframe(class_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### 🏥 Local vs Global Model Comparison")
        st.markdown("""
        **Why this matters:** In federated learning, we want the global model to outperform 
        local-only models. Local models only see data from one site, while the global model 
        learns from all sites combined.
        """)
        
        # Create comparison chart
        client_results = detailed["client_models"]
        
        # Check if local model results exist
        has_local = "local_model_accuracy" in client_results[0]
        
        if has_local:
            # Grouped bar chart: Local vs Global
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(client_results))
            width = 0.35
            
            local_acc = [r['local_model_accuracy'] for r in client_results]
            global_acc = [r['global_model_test_accuracy'] for r in client_results]
            
            bars1 = ax.bar(x - width/2, local_acc, width, label='Local Model', color='#ff7f0e', edgecolor='darkorange', linewidth=1.5)
            bars2 = ax.bar(x + width/2, global_acc, width, label='Global Model', color='#1f77b4', edgecolor='navy', linewidth=1.5)
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel("Site (Client)", fontsize=12)
            ax.set_ylabel("Test Accuracy (%)", fontsize=12)
            ax.set_title("Local-Only vs Federated Global Model Performance", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Site {r['client_id']}" for r in client_results])
            ax.set_ylim([0, max(max(local_acc), max(global_acc)) * 1.15])
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Summary metrics
            st.markdown("### Performance Summary")
            avg_local = np.mean(local_acc)
            avg_global = np.mean(global_acc)
            improvement = avg_global - avg_local
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Local Accuracy", f"{avg_local:.2f}%")
            col2.metric("Avg Global Accuracy", f"{avg_global:.2f}%")
            col3.metric("FL Improvement", f"{improvement:+.2f}%", 
                       delta_color="normal" if improvement > 0 else "inverse")
            
            if improvement > 0:
                st.success(f"✅ **Federated learning helps!** The global model outperforms local models by {improvement:.1f}% on average.")
            else:
                st.warning(f"⚠️ Local models performed better. Try more FL rounds or adjust hyperparameters.")
            
            # Detailed table
            st.markdown("### Detailed Site Comparison")
            site_table = []
            for r in client_results:
                diff = r['global_model_test_accuracy'] - r['local_model_accuracy']
                site_table.append({
                    "Site": f"Site {r['client_id']}",
                    "Samples": r['n_samples'],
                    "Classes": r.get('unique_classes', 'N/A'),
                    "Local Acc (%)": f"{r['local_model_accuracy']:.2f}",
                    "Global Acc (%)": f"{r['global_model_test_accuracy']:.2f}",
                    "Difference": f"{diff:+.2f}%"
                })
            
            site_df = pd.DataFrame(site_table)
            st.dataframe(site_df, use_container_width=True, hide_index=True)
        else:
            # Fallback to old view
            fig, ax = plt.subplots(figsize=(10, 5))
            client_ids = [f"Site {r['client_id']}" for r in client_results]
            accuracies = [r['global_model_accuracy'] for r in client_results]
            n_samples = [r['n_samples'] for r in client_results]
            colors = plt.cm.Blues(np.array(n_samples) / max(n_samples) * 0.7 + 0.3)
            bars = ax.bar(client_ids, accuracies, color=colors, edgecolor='navy', linewidth=1.5)
            for bar, n in zip(bars, n_samples):
                height = bar.get_height()
                ax.annotate(f'n={n}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
            ax.set_xlabel("Site (Client)", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_title("Global Model Performance by Site", fontsize=14)
            ax.set_ylim([0, 105])
            ax.axhline(y=detailed["global_model"]["overall_accuracy"], 
                       color='red', linestyle='--', linewidth=2, label='Overall Avg')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Class distribution info
        if "class_distribution" in detailed:
            st.markdown("### Class Distribution in Dataset")
            dist = detailed["class_distribution"]
            fig, ax = plt.subplots(figsize=(10, 4))
            classes = list(dist.keys())
            counts = list(dist.values())
            ax.bar([f"Class {c}" for c in classes], counts, color='steelblue', edgecolor='navy')
            ax.set_xlabel("Class", fontsize=12)
            ax.set_ylabel("Number of Samples", fontsize=12)
            ax.set_title("Class Distribution Across All Clients", fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            total_samples = sum(counts)
            st.caption(f"Total samples: {total_samples} | Classes: {len(classes)}")
        
        # Performance gap indicator
        acc_values = [r['global_model_accuracy'] for r in client_results]
        gap = max(acc_values) - min(acc_values)
        if gap > 15:
            st.warning(f"⚠️ Large performance gap ({gap:.1f}%) between sites. Consider using non-IID mitigation strategies like FedProx.")
        elif gap > 5:
            st.info(f"ℹ️ Moderate performance variation ({gap:.1f}%) across sites.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Powered by <a href='https://flower.dev'>Flower</a> 🌸 | 
    <a href='https://medmnist.com'>MedMNIST</a> 🏥 |
    Built with <a href='https://streamlit.io'>Streamlit</a></small>
</div>
""", unsafe_allow_html=True)
