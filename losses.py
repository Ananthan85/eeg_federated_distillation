import torch
import torch.nn.functional as F

def xkd_attention_loss(student_logits, teacher_logits, student_features, teacher_features, targets, alpha=0.3, beta=0.4, gamma=0.3, temperature=4.0):
    
    # 1. Hard Target Loss (Cross Entropy)
    l_ce = F.cross_entropy(student_logits, targets)
    
    # 2. Soft Target Parity (KL Divergence)
    s_soft = F.log_softmax(student_logits / temperature, dim=1)
    t_soft = F.softmax(teacher_logits / temperature, dim=1)
    l_kl = F.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)
    
    # 3. Formal Attention Transfer (Energy Mapping)
    # Generate spatial-temporal attention maps via sum of squared activations
    student_attention = torch.sum(torch.pow(student_features, 2), dim=1) # Shape: [Batch, 1280]
    teacher_attention = torch.sum(torch.pow(teacher_features, 2), dim=1) # Shape: [Batch, 320]
    
    # Strict temporal alignment via Adaptive Average Pooling (no broadcasting tricks)
    if student_attention.shape != teacher_attention.shape:
        target_length = teacher_attention.size(1) 
        # Add channel dim for pooling, then remove it
        student_attention = F.adaptive_avg_pool1d(student_attention.unsqueeze(1), target_length).squeeze(1)
    
    # L2 Normalization to ensure feature scales do not overwhelm the gradient
    student_attention = F.normalize(student_attention, p=2, dim=1)
    teacher_attention = F.normalize(teacher_attention, p=2, dim=1)
    
    # Point-to-point Mean Squared Error
    l_at = F.mse_loss(student_attention, teacher_attention)
    
    return (alpha * l_ce) + (beta * l_kl) + (gamma * l_at)
