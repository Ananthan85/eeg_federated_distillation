import torch
from torch.utils.data import DataLoader, Subset
from models import EdgeEEGNet
from train_teacher import FullRealEEGDataset

def run_sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Audit] Running Sanity Check on original Patient 1 data...")
    
    # 1. Load your student model weights
    model = EdgeEEGNet().to(device)
    model.load_state_dict(torch.load("./deployed_edge_model.pt", map_location=device))
    model.eval()
    
    # 2. Use the original summary for Patient 1
    # This dataset class handles the complex mapping of summary text to tensors
    full_dataset = FullRealEEGDataset(tensor_dir="./eeg_tensors", summary_path="./chb01-summary.txt")
    
    # 3. Filter for a known seizure file (e.g., chb01_03)
    # This ensures we are testing on data the model has "seen" during training
    indices = [i for i, path in enumerate(full_dataset.filepaths) if "chb01_03" in path]
    
    if not indices:
        print("Error: chb01_03 files not found. Ensure ingestion is complete.")
        return

    loader = DataLoader(Subset(full_dataset, indices), batch_size=32, shuffle=False)
    
    # 4. Evaluation
    true_pos, false_pos, false_neg, true_neg = 0, 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            # --- SCALING FIX ---
            batch_x = (batch_x - batch_x.min()) / (batch_x.max() - batch_x.min() + 1e-9)
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            
            # --- THE "GATE" FIX ---
            # 1. Get probabilities via softmax
            probs = torch.softmax(logits, dim=1)
            
            # 2. Thresholding: Only classify as seizure (1) if confidence > 95%
            # This suppresses the False Positives caused by hypersensitivity
            preds = (probs[:, 1] > 0.95).long() 
            # ----------------------
            
            true_pos += ((preds == 1) & (batch_y == 1)).sum().item()
            false_pos += ((preds == 1) & (batch_y == 0)).sum().item()
            false_neg += ((preds == 0) & (batch_y == 1)).sum().item()
            true_neg += ((preds == 0) & (batch_y == 0)).sum().item()
    # Calculate performance metrics
    precision = true_pos / (true_pos + false_pos + 1e-9)
    recall = true_pos / (true_pos + false_neg + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    acc = (true_pos + true_neg) / len(indices)
    
    print(f"\n--- SANITY CHECK RESULTS (chb01_03) ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print(f"TP: {true_pos} | FP: {false_pos} | FN: {false_neg}")

if __name__ == "__main__":
    run_sanity_check()
