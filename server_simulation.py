import os
import torch
import numpy as np
import flwr as fl
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import OrderedDict

from models import EdgeEEGNet, TeacherCGRU
from losses import xkd_attention_loss
from train_teacher import FullRealEEGDataset 

# --- 1. FLOWER CLIENT LOGIC ---
class FlowerEEGClient(fl.client.NumPyClient):
    def __init__(self, student, teacher, train_loader, val_loader, device):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def get_parameters(self, config):
        # Extracting the actual WEIGHTS to send to the server
        return [val.cpu().numpy() for _, val in self.student.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.student.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.student.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.student.train()
        self.teacher.eval() 
        
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=1e-3)
        
        for epoch in range(1):
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                with torch.no_grad():
                    t_logits, t_feat = self.teacher(batch_x, return_features=True)
                    
                s_logits, s_feat = self.student(batch_x, return_features=True)
                
                loss = xkd_attention_loss(s_logits, t_logits, s_feat, t_feat, batch_y)
                
                # Calculates the Gradients
                loss.backward()

                # Updates the Weights (No LDP noise injection here anymore)
                optimizer.step()
                
        # Returns the final Weights payload (4KB)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.student.eval()
        
        loss_fn = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        true_pos, false_pos, false_neg = 0, 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                s_logits = self.student(batch_x)
                
                loss = loss_fn(s_logits, batch_y)
                total_loss += loss.item()
                
                _, preds = torch.max(s_logits, 1)
                
                true_pos += ((preds == 1) & (batch_y == 1)).sum().item()
                false_pos += ((preds == 1) & (batch_y == 0)).sum().item()
                false_neg += ((preds == 0) & (batch_y == 1)).sum().item()

        return float(total_loss / len(self.val_loader)), len(self.val_loader.dataset), {
            "tp": true_pos, 
            "fp": false_pos, 
            "fn": false_neg
        }

# --- 2. FEDERATED SHARDING & ORCHESTRATION ---
def client_fn(context: fl.common.Context) -> fl.client.Client:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_dataset = FullRealEEGDataset(tensor_dir="./eeg_tensors", summary_path="./chb01-summary.txt")
    
    labels = np.array([full_dataset[i][1].item() for i in range(len(full_dataset))])
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
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=32, sampler=sampler)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=32, shuffle=False)
    
    student_model = EdgeEEGNet().to(device)
    teacher_model = TeacherCGRU().to(device)
    
    teacher_weights_path = "./best_teacher_centralized.pth"
    if os.path.exists(teacher_weights_path):
        teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location=device))
        
    return FlowerEEGClient(student_model, teacher_model, train_loader, val_loader, device).to_client()

def aggregate_fit_metrics(metrics):
    return {}

def aggregate_metrics(metrics):
    global_tp = sum([m["tp"] for _, m in metrics])
    global_fp = sum([m["fp"] for _, m in metrics])
    global_fn = sum([m["fn"] for _, m in metrics])
    
    precision = global_tp / (global_tp + global_fp + 1e-9)
    recall = global_tp / (global_tp + global_fn + 1e-9)
    global_f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    return {"global_precision": precision, "global_recall": recall, "true_global_f1": global_f1}

# --- 3. CUSTOM SERVER STRATEGY ---
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            student = EdgeEEGNet()
            params_dict = zip(student.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            student.load_state_dict(state_dict, strict=True)
            
            torch.save(student.state_dict(), "./deployed_edge_model.pt")
            print(f"--> [Server] Global Edge Model Saved for Round {server_round}!")
            
        return aggregated_parameters, aggregated_metrics

if __name__ == "__main__":
    print("[Orchestrator] Launching Verified IEEE Simulation Framework (LDP Removed)...")
    
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics
    )
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.5 if torch.cuda.is_available() else 0}
    )
