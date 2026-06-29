import flwr as fl
import torch
import numpy as np
import collections
from typing import Dict, List, Tuple
from losses import xkd_attention_loss

class FlowerEEGClient(fl.client.NumPyClient):
    def __init__(self, student_model, teacher_model, train_loader, val_loader, device):
        self.model = student_model
        self.teacher = teacher_model 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.teacher.eval() # Freeze Teacher on the edge

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        state_dict_updates = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict_updates, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        for _ in range(int(config.get("local_epochs", 1))):
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                # Execute Tri-Objective X-KD
                student_logits, student_feat = self.model(batch_x, return_features=True)
                with torch.no_grad():
                    teacher_logits, teacher_feat = self.teacher(batch_x, return_features=True)
                    
                loss = xkd_attention_loss(
                    student_logits, teacher_logits, student_feat, teacher_feat, batch_y,
                    alpha=0.3, beta=0.4, gamma=0.3, temperature=4.0
                )
                
                loss.backward()
                optimizer.step()
                
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss += criterion(outputs, batch_y).item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        accuracy = float(correct) / total
        return loss, len(self.val_loader.dataset), {"accuracy": accuracy}
