import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import os
import glob
import re

from models import TeacherCGRU

# --- 1. UNIVERSAL DATASET LOADER ---
def parse_all_chb_summaries(summary_path: str) -> dict:
    """Parses the summary text file and returns a dictionary mapping filename to seizure times."""
    seizures_by_file = {}
    with open(summary_path, 'r') as f:
        content = f.read()
    
    blocks = content.split("File Name:")
    for block in blocks[1:]: # Skip the first split which is header info
        file_match = re.search(r'(chb\d+_\d+\.edf)', block)
        if not file_match:
            continue
        
        file_name = file_match.group(1).replace('.edf', '')
        starts = re.findall(r'Seizure(?:\s+\d+)?\s+Start Time:\s+(\d+)\s+seconds', block)
        ends = re.findall(r'Seizure(?:\s+\d+)?\s+End Time:\s+(\d+)\s+seconds', block)
        
        seizures = [(int(s), int(e)) for s, e in zip(starts, ends)]
        seizures_by_file[file_name] = seizures
        
    return seizures_by_file

class FullRealEEGDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dir: str, summary_path: str, window_size: int = 5):
        self.filepaths = glob.glob(os.path.join(tensor_dir, "*.pt"))
        # Sort files to ensure some logical order, though shuffle=True handles training
        self.filepaths.sort() 
        
        self.window_size = window_size
        self.seizures_by_file = parse_all_chb_summaries(summary_path)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        file_path = self.filepaths[idx]
        x = torch.load(file_path, weights_only=True)
        
        # Extract the base filename and window number from the path (e.g., "chb01_03_win_5.pt")
        base_name = os.path.basename(file_path)
        file_prefix = re.search(r'(chb\d+_\d+)_win', base_name).group(1)
        win_idx = int(re.search(r'_win_(\d+)\.pt', base_name).group(1))
        
        start_sec = win_idx * self.window_size
        end_sec = start_sec + self.window_size
        y = 0 
        
        # Check if this specific window in this specific file overlaps with a seizure
        if file_prefix in self.seizures_by_file:
            for (sz_start, sz_end) in self.seizures_by_file[file_prefix]:
                if (start_sec < sz_end) and (end_sec > sz_start):
                    y = 1
                    break
                    
        return x, torch.tensor(y, dtype=torch.long)

# --- 2. TRAINING ENGINE ---
def train_central_teacher():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*50)
    print(f"[Oracle Setup] Booting Scale-Up Teacher Training on {device}...")
    
    dataset = FullRealEEGDataset(tensor_dir="./eeg_tensors", summary_path="./chb01-summary.txt")
    total_windows = len(dataset)
    
    if total_windows == 0:
        print("[Error] No tensors found. Wait for data_pipeline.py to finish processing!")
        return

    # Extract Labels for Stratification
    print(f"-> Scanning {total_windows} tensors for anomalies...")
    labels = np.array([dataset[i][1].item() for i in range(total_windows)])
    seizure_idx = np.where(labels == 1)[0]
    normal_idx = np.where(labels == 0)[0]
    
    np.random.seed(42)
    np.random.shuffle(seizure_idx)
    np.random.shuffle(normal_idx)
    
    train_seiz_split = int(0.8 * len(seizure_idx))
    train_norm_split = int(0.8 * len(normal_idx))
    
    train_idx = np.concatenate([seizure_idx[:train_seiz_split], normal_idx[:train_norm_split]])
    val_idx = np.concatenate([seizure_idx[train_seiz_split:], normal_idx[train_norm_split:]])
    
    weight_normal = 1.0
    weight_seizure = len(normal_idx) / len(seizure_idx) if len(seizure_idx) > 0 else 1.0
    
    sample_weights = np.zeros(len(train_idx))
    for i, idx in enumerate(train_idx):
        sample_weights[i] = weight_seizure if labels[idx] == 1 else weight_normal
        
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, sampler=sampler) # Increased batch size for speed
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False)

    print(f"--- Global Dataset Stratification ---")
    print(f"Train Set: {len(train_idx)} Windows ({train_seiz_split} Unique Seizures)")
    print(f"Val Set:   {len(val_idx)} Windows ({len(seizure_idx) - train_seiz_split} Unique Seizures)")
    print("="*50 + "\n")

    teacher = TeacherCGRU().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 30
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        teacher.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = teacher(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        teacher.eval()
        true_pos, false_pos, false_neg = 0, 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = teacher(batch_x)
                _, preds = torch.max(logits, 1)
                
                true_pos += ((preds == 1) & (batch_y == 1)).sum().item()
                false_pos += ((preds == 1) & (batch_y == 0)).sum().item()
                false_neg += ((preds == 0) & (batch_y == 1)).sum().item()
        
        precision = true_pos / (true_pos + false_pos + 1e-9)
        recall = true_pos / (true_pos + false_neg + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {epoch_loss/len(train_loader):.4f} | Val F1: {f1:.4f} (TP:{true_pos} FP:{false_pos} FN:{false_neg})")
        
        if f1 >= best_val_f1 and f1 > 0:
            best_val_f1 = f1
            torch.save(teacher.state_dict(), "./best_teacher_centralized.pth")
            print("   -> [Global Checkpoint Saved!]")
            
    print("\n[Oracle Setup] Complete. Ready for Distillation.")

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_central_teacher()
