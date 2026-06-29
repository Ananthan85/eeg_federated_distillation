import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import glob
import os
import time
import re

from models import EdgeEEGNet, TeacherCGRU

def parse_chb_summary(summary_path: str, target_edf: str) -> list[tuple[int, int]]:
    seizures = []
    with open(summary_path, 'r') as f:
        content = f.read()
    
    blocks = content.split("File Name:")
    for block in blocks:
        if target_edf in block:
            starts = re.findall(r'Seizure(?:\s+\d+)?\s+Start Time:\s+(\d+)\s+seconds', block)
            ends = re.findall(r'Seizure(?:\s+\d+)?\s+End Time:\s+(\d+)\s+seconds', block)
            for s, e in zip(starts, ends):
                seizures.append((int(s), int(e)))
            break
    return seizures

class RealEEGDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dir: str, summary_path: str, edf_name: str, window_size: int = 5):
        self.filepaths = glob.glob(os.path.join(tensor_dir, "*.pt"))
        self.filepaths.sort(key=lambda x: int(x.split('_win_')[-1].split('.pt')[0]))
        
        self.window_size = window_size
        self.seizure_times = parse_chb_summary(summary_path, edf_name)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        x = torch.load(self.filepaths[idx], weights_only=True)
        start_sec = idx * self.window_size
        end_sec = start_sec + self.window_size
        y = 0 
        for (sz_start, sz_end) in self.seizure_times:
            if (start_sec < sz_end) and (end_sec > sz_start):
                y = 1
                break
        return x, torch.tensor(y, dtype=torch.long)

def generate_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    student = EdgeEEGNet()
    
    print("\n" + "="*30)
    print("--- EXTRACTING CLINICAL GROUND TRUTH ---")
    dataset = RealEEGDataset(tensor_dir="./eeg_tensors", summary_path="./chb01-summary.txt", edf_name="chb01_03.edf")
    
    # Calculate dataset imbalance
    seizure_count = sum([1 for i in range(len(dataset)) if dataset[i][1].item() == 1])
    print(f"Total Windows: {len(dataset)} | Normal: {len(dataset)-seizure_count} | Seizure: {seizure_count}")
    print("="*30 + "\n")

    train_size = int(0.8 * len(dataset))
    train_ds = Subset(dataset, range(0, train_size))
    val_ds = Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = student.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    epochs = 10
    best_acc = 0.0
    best_f1 = 0.0
    lowest_loss = float('inf')
    
    print("Beginning Baseline Training on Real Clinical Labels...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < lowest_loss:
            lowest_loss = avg_loss

        model.eval()
        correct, total = 0, 0
        true_positives, false_positives, false_negatives = 0, 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # F1 Metrics
                true_positives += ((predicted == 1) & (batch_y == 1)).sum().item()
                false_positives += ((predicted == 1) & (batch_y == 0)).sum().item()
                false_negatives += ((predicted == 0) & (batch_y == 1)).sum().item()

        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            
        # Calculate F1
        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        if f1 > best_f1:
            best_f1 = f1

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | F1-Score: {f1:.4f}")

    print("\n" + "="*30)
    print("--- FINAL REPORT METRICS ---")
    print(f"Least Loss Value: {lowest_loss:.4f}")
    print(f"Maximum Baseline Accuracy: {best_acc:.2f}%")
    print(f"Maximum Baseline F1-Score: {best_f1:.4f} (Crucial for imbalanced data)")
    print("="*30 + "\n")

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generate_report()
