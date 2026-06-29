import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import EdgeEEGNet, TeacherCGRU
from losses import xkd_attention_loss

def enforce_reproducibility(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def run_distillation_epoch(student, teacher, dataloader, optimizer, device, alpha, beta, gamma, temperature) -> float:
    student.train()
    teacher.eval()
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        student_logits, student_feat = student(batch_x, return_features=True)
        with torch.no_grad():
            teacher_logits, teacher_feat = teacher(batch_x, return_features=True)
            
        loss = xkd_attention_loss(student_logits, teacher_logits, student_feat, teacher_feat, batch_y, alpha, beta, gamma, temperature)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    return epoch_loss / len(dataloader.dataset)

if __name__ == "__main__":
    enforce_reproducibility(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_mock, Y_mock = torch.randn(32, 18, 1280), torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(X_mock, Y_mock), batch_size=16, shuffle=True)
    student = EdgeEEGNet().to(device)
    teacher = TeacherCGRU().to(device)
    opt = optim.AdamW(student.parameters(), lr=1e-3)
    loss_metric = run_distillation_epoch(student, teacher, loader, opt, device, 0.3, 0.4, 0.3, 4.0)
    print(f"Epoch structural validation passed successfully. Loss: {loss_metric:.4f}")
