import torch
from torch.utils.data import DataLoader, Subset
from models import EdgeEEGNet
from train_teacher import FullRealEEGDataset

def run_true_clinical_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Audit] Initializing True Clinical Audit on Patient 21 (1.5 years later)...")
    print("[Audit] Applying Z-Score Normalization to correct for Covariate Shift...")
    
    # 1. Load Model
    model = EdgeEEGNet().to(device)
    model.load_state_dict(torch.load("./deployed_edge_model.pt", map_location=device))
    model.eval()
    
    # 2. Load Dataset using the Summary File
    full_dataset = FullRealEEGDataset(tensor_dir="./eeg_tensors", summary_path="./chb21-summary.txt")
    
    # 3. Filter for the specific file
    file_id = "chb21_19"
    indices = []
    for i in range(len(full_dataset)):
        if file_id in full_dataset.filepaths[i]:
             indices.append(i)

    if not indices:
        print(f"Error: Could not find {file_id} in dataset. Did you run data_pipeline.py?")
        return

    loader = DataLoader(Subset(full_dataset, indices), batch_size=32, shuffle=False)
    
    # 4. Evaluate
    true_pos, false_pos, false_neg, true_neg = 0, 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            
            # --- THE CSE FIX: Dynamic Z-Score Normalization ---
            # Normalize across the time dimension (dim=-1) for each channel
            mean = batch_x.mean(dim=-1, keepdim=True)
            std = batch_x.std(dim=-1, keepdim=True)
            batch_x = (batch_x - mean) / (std + 1e-9)
            # --------------------------------------------------
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            _, preds = torch.max(logits, 1)
            
            true_pos += ((preds == 1) & (batch_y == 1)).sum().item()
            false_pos += ((preds == 1) & (batch_y == 0)).sum().item()
            false_neg += ((preds == 0) & (batch_y == 1)).sum().item()
            true_neg += ((preds == 0) & (batch_y == 0)).sum().item()
            
    precision = true_pos / (true_pos + false_pos + 1e-9)
    recall = true_pos / (true_pos + false_neg + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    acc = (true_pos + true_neg) / len(indices)
    
    print("\n" + "="*50)
    print(f"   LONG-TERM ROBUSTNESS METRICS (PATIENT 21)")
    print("="*50)
    print(f" Accuracy:  {acc*100:.2f}%")
    print(f" F1 Score:  {f1:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" Windows Evaluated: {len(indices)}")
    print("="*50)

if __name__ == "__main__":
    run_true_clinical_audit()
