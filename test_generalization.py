import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import EdgeEEGNet
from train_teacher import FullRealEEGDataset
import glob
import os

def run_generalization_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the finalized model
    model = EdgeEEGNet().to(device)
    model.load_state_dict(torch.load("./deployed_edge_model.pt", map_location=device))
    model.eval()
    
    # 2. Ingest the NEW patient data specifically
    # This will pull only the new tensors you just downloaded
    dataset = FullRealEEGDataset(tensor_dir="./eeg_tensors", summary_path="./chb02-summary.txt")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 3. Evaluate
    true_pos, false_pos, false_neg = 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            _, preds = torch.max(logits, 1)
            
            true_pos += ((preds == 1) & (batch_y == 1)).sum().item()
            false_pos += ((preds == 1) & (batch_y == 0)).sum().item()
            false_neg += ((preds == 0) & (batch_y == 1)).sum().item()
            
    precision = true_pos / (true_pos + false_pos + 1e-9)
    recall = true_pos / (true_pos + false_neg + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    print(f"--- Generalization Results (Patient 2) ---")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")

if __name__ == "__main__":
    run_generalization_test()