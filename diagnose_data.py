import torch
from torch.utils.data import Subset
from server_simulation import RealEEGDataset

def run_diagnosis():
    print("="*40)
    print("      EEG DATA DISTRIBUTION REPORT      ")
    print("="*40)

    # Load dataset
    try:
        dataset = RealEEGDataset(
            tensor_dir="./eeg_tensors", 
            summary_path="./chb01-summary.txt", 
            edf_name="chb01_03.edf"
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    total_windows = len(dataset)
    if total_windows == 0:
        print("No windows found in ./eeg_tensors! Did you run data_pipeline.py?")
        return

    # Extract all labels sequentially
    labels = []
    for i in range(total_windows):
        _, y = dataset[i]
        labels.append(y.item())
    
    total_seizures = sum(labels)
    print(f"Total Extracted Windows: {total_windows}")
    print(f"Total Seizure Windows:   {total_seizures}")
    print(f"Total Normal Windows:    {total_windows - total_seizures}")
    print("-"*40)

    # 1. Check Sequential Train/Val Split (80/20)
    train_size = int(0.8 * total_windows)
    train_labels = labels[:train_size]
    val_labels = labels[train_size:]
    
    print("--- 80/20 Sequential Split Mapping ---")
    print(f"Train Window Range:  0 to {train_size - 1}")
    print(f"Train Seizures:      {sum(train_labels)}")
    print(f"Val Window Range:    {train_size} to {total_windows - 1}")
    print(f"Val Seizures:        {sum(val_labels)}")
    print("-"*40)

    # 2. Check Naive Client Sharding (2 Clients)
    num_clients = 2
    shard_size = total_windows // num_clients
    
    print("--- Naive Federated Sharding Mapping ---")
    for client_id in range(num_clients):
        start_idx = client_id * shard_size
        end_idx = start_idx + shard_size if client_id < num_clients - 1 else total_windows
        client_labels = labels[start_idx:end_idx]
        print(f"Client {client_id} Range:    {start_idx} to {end_idx - 1}")
        print(f"Client {client_id} Seizures: {sum(client_labels)}")
    print("="*40)

if __name__ == "__main__":
    run_diagnosis()
