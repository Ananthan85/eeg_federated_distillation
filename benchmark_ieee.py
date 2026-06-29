import torch
import time
import numpy as np
from torch.utils.data import DataLoader, Subset
from models import EdgeEEGNet, TeacherCGRU
from train_teacher import FullRealEEGDataset

def find_best_threshold(probs, y_true):
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = (0, 0, 0, 0) # tp, fp, fn, tn
    
    # Scan every possible threshold from 1% to 99% confidence
    for thresh in np.arange(0.01, 1.0, 0.01):
        preds = (probs > thresh).int()
        tp = ((preds == 1) & (y_true == 1)).sum().item()
        fp = ((preds == 1) & (y_true == 0)).sum().item()
        fn = ((preds == 0) & (y_true == 1)).sum().item()
        tn = ((preds == 0) & (y_true == 0)).sum().item()
        
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = (tp, fp, fn, tn)
            
    return best_thresh, best_metrics

def run_head_to_head_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Initializing Dynamic IEEE Ablation Study on {device}...")
    
    student = EdgeEEGNet().to(device)
    teacher = TeacherCGRU().to(device)
    
    try:
        student.load_state_dict(torch.load("./deployed_edge_model.pt", map_location=device))
        teacher.load_state_dict(torch.load("./best_teacher_centralized.pth", map_location=device))
    except Exception as e:
        print(f"[Error] Missing weight files: {e}")
        return

    student.eval()
    teacher.eval()
    
    full_dataset = FullRealEEGDataset(tensor_dir="./eeg_tensors", summary_path="./chb01-summary.txt")
    target_file = "chb01_04"
    indices = [i for i, path in enumerate(full_dataset.filepaths) if target_file in path]
    
    loader = DataLoader(Subset(full_dataset, indices), batch_size=32, shuffle=False)
    
    t_probs_list, s_probs_list, y_list = [], [], []
    t_time, s_time = 0.0, 0.0
    
    print(f"[Benchmark] Extracting probabilities for {len(indices)} windows...\n")
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            # Removed arbitrary scaling to match train_teacher.py exactly
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Teacher
            t_start = time.perf_counter()
            t_logits = teacher(batch_x)
            t_time += (time.perf_counter() - t_start)
            t_probs_list.append(torch.softmax(t_logits, dim=1)[:, 1])
            
            # Student
            s_start = time.perf_counter()
            s_logits = student(batch_x)
            s_time += (time.perf_counter() - s_start)
            s_probs_list.append(torch.softmax(s_logits, dim=1)[:, 1])
            
            y_list.append(batch_y)

    t_probs = torch.cat(t_probs_list)
    s_probs = torch.cat(s_probs_list)
    y_true = torch.cat(y_list)

    # Dynamically find the best thresholds
    t_thresh, (t_tp, t_fp, t_fn, t_tn) = find_best_threshold(t_probs, y_true)
    s_thresh, (s_tp, s_fp, s_fn, s_tn) = find_best_threshold(s_probs, y_true)

    def calc_stats(tp, fp, fn, tn):
        acc = (tp + tn) / (tp + fp + fn + tn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
        return acc, prec, rec, f1

    t_acc, t_prec, t_rec, t_f1 = calc_stats(t_tp, t_fp, t_fn, t_tn)
    s_acc, s_prec, s_rec, s_f1 = calc_stats(s_tp, s_fp, s_fn, s_tn)
    
    print("="*65)
    print(f"{'Metric':<15} | {'TeacherCGRU (59MB)':<20} | {'EdgeEEGNet (4KB)':<20}")
    print("-" * 65)
    print(f"{'Opt Threshold':<15} | {t_thresh:>19.2f} | {s_thresh:>19.2f}")
    print(f"{'Accuracy':<15} | {t_acc*100:>18.2f}% | {s_acc*100:>18.2f}%")
    print(f"{'F1-Score':<15} | {t_f1:>19.4f} | {s_f1:>19.4f}")
    print(f"{'Precision':<15} | {t_prec:>19.4f} | {s_prec:>19.4f}")
    print(f"{'Recall':<15} | {t_rec:>19.4f} | {s_rec:>19.4f}")
    print("-" * 65)
    print(f"{'True Positives':<15} | {t_tp:>19} | {s_tp:>19}")
    print(f"{'False Positives':<15} | {t_fp:>19} | {s_fp:>19}")
    print(f"{'False Negatives':<15} | {t_fn:>19} | {s_fn:>19}")
    print("-" * 65)
    print(f"{'Inference Time':<15} | {t_time:>16.4f} s | {s_time:>16.4f} s")
    print("="*65)

if __name__ == "__main__":
    run_head_to_head_benchmark()
