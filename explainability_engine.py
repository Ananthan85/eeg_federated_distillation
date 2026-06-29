import matplotlib
matplotlib.use('Agg')  # Force headless rendering
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from models import EdgeEEGNet

def generate_ieee_visuals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Explainability] Booting Engine...")

    # 1. Load the Deployed Student Model
    model = EdgeEEGNet().to(device)
    model.load_state_dict(torch.load("./deployed_edge_model.pt", map_location=device))
    model.eval()

    # 2. Find a True Positive Seizure Tensor
    tensor_files = glob.glob("./eeg_tensors/*.pt")
    sample_tensor = None
    
    # We will just grab the first file for demonstration, 
    # but ideally, you point this to a known seizure tensor from your validation set.
    sample_tensor = torch.load(tensor_files[0], weights_only=True).unsqueeze(0).to(device) # Shape: [1, Channels, Time]
    
    time_steps = sample_tensor.size(2)
    channels = sample_tensor.size(1)

    # ==========================================
    # ALGORITHM A: 1D GRAD-CAM
    # ==========================================
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hook into the final convolutional layer of your EdgeEEGNet
    # (Adjust 'conv_layers[-1]' based on your exact models.py naming)
    target_layer = model.conv3 if hasattr(model, 'conv3') else list(model.modules())[-3] 
    
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    logits = model(sample_tensor)
    pred_class = torch.argmax(logits, dim=1).item()
    base_confidence = F.softmax(logits, dim=1)[0, pred_class].item()

    # Backward pass for the predicted class
    model.zero_grad()
    logits[0, pred_class].backward()

    # Calculate Grad-CAM
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2])
    for i in range(activations[0].size(1)):
        activations[0][:, i, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations[0], dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap) # Normalize to 0-1
    
    # Interpolate heatmap back to the 1280 time steps
    # Force dimensions to (Batch=1, Channel=1, Time) for interpolation
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=time_steps, mode='linear', align_corners=False).squeeze()
    heatmap_np = heatmap.cpu().detach().numpy()

    h1.remove()
    h2.remove()

    # ==========================================
    # ALGORITHM B: OCCLUSION SENSITIVITY
    # ==========================================
    window_size = int(time_steps * 0.05) # Occlude 5% of the signal at a time
    stride = int(window_size / 2)
    occlusion_map = np.zeros(time_steps)
    
    for start in range(0, time_steps - window_size, stride):
        end = start + window_size
        
        # Create an occluded copy of the tensor (zero out the time window across all channels)
        occluded_tensor = sample_tensor.clone()
        occluded_tensor[:, :, start:end] = 0.0
        
        with torch.no_grad():
            occ_logits = model(occluded_tensor)
            occ_confidence = F.softmax(occ_logits, dim=1)[0, pred_class].item()
            
        # The "importance" of this window is how much the confidence dropped
        drop = base_confidence - occ_confidence
        occlusion_map[start:end] = np.maximum(occlusion_map[start:end], drop)

    occlusion_map = np.clip(occlusion_map, 0, None)
    if np.max(occlusion_map) > 0:
        occlusion_map /= np.max(occlusion_map) # Normalize to 0-1

    # ==========================================
    # PLOTTING THE IEEE FIGURE
    # ==========================================
    # We plot channel 0 as the background reference brainwave
    bg_signal = sample_tensor[0, 0, :].cpu().numpy()
    time_axis = np.linspace(0, 5, time_steps) # 5 seconds

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=300)
    
    # Plot 1: Grad-CAM
    axes[0].plot(time_axis, bg_signal, color='black', alpha=0.5, linewidth=1, label='Raw EEG (Ch 0)')
    axes[0].set_title(f"1D Grad-CAM (Internal Activation) | Base Confidence: {base_confidence*100:.1f}%")
    ax0_twin = axes[0].twinx()
    ax0_twin.fill_between(time_axis, 0, heatmap_np, color='red', alpha=0.3, label='Attention Heatmap')
    ax0_twin.set_ylim(0, 1.1)
    ax0_twin.axis('off')

    # Plot 2: Occlusion
    axes[1].plot(time_axis, bg_signal, color='black', alpha=0.5, linewidth=1)
    axes[1].set_title("Occlusion Sensitivity (Data Dependency)")
    axes[1].set_xlabel("Time (Seconds)")
    ax1_twin = axes[1].twinx()
    ax1_twin.fill_between(time_axis, 0, occlusion_map, color='blue', alpha=0.3, label='Confidence Drop')
    ax1_twin.set_ylim(0, 1.1)
    ax1_twin.axis('off')

    plt.tight_layout()
    plt.savefig("./ieee_explainability_figure.png")
    print("-> Successfully generated 'ieee_explainability_figure.png'")

if __name__ == "__main__":
    generate_ieee_visuals()
