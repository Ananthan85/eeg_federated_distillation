# 🧠 EEG Federated Distillation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-ffd21e)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end clinical machine learning framework that combines **Federated Learning** and **Knowledge Distillation** to compress a robust EEG seizure detection model into a highly efficient edge-deployable architecture. 

This repository facilitates the ingestion of raw clinical EEG recordings (.edf), the training of a highly accurate centralized Teacher model (`TeacherCGRU`, 59MB), and the federated distillation of its knowledge into an ultra-lightweight Student model (`EdgeEEGNet`, 4KB).

---

## ✨ Key Features

* **Extreme Model Compression:** Compress a robust 59MB `TeacherCGRU` down to a 4KB `EdgeEEGNet` suitable for deployment on edge devices (wearables, BCIs).
* **Federated Knowledge Distillation (Flower):** Allows edge clients to train the student model collaboratively using a frozen teacher without centralizing raw, sensitive EEG patient data.
* **Tri-Objective Attention Loss:** A custom distillation loss (`xkd_attention_loss`) optimizing Hard Targets (Cross-Entropy), Soft Target Parity (KL Divergence, Temp=4.0), and Spatial-Temporal Attention Transfer (Energy Mapping).
* **Clinical Explainability Engine:** Includes IEEE-compliant visualizations utilizing **1D Grad-CAM** and **Occlusion Sensitivity** to interpret network activations on raw EEG waveforms.
* **Covariate Shift Correction:** Handles extreme subject-to-subject variability and temporal drift (e.g., tests on patients 1.5 years later) using dynamic Z-Score Normalization and softmax confidence gating (95% threshold).
* **Clinical Pipeline:** Automated extraction of 18 standard 10-20 EEG channels, sliced into 5-second time windows, mapping directly to clinical summary reports for ground-truth labeling.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `data_pipeline.py` | Ingests `.edf` files, extracts 18 target channels, windows data into 5-second segments, and saves them as PyTorch tensors. |
| `diagnose_data.py` | Audits the dataset distribution, class imbalances (seizure vs normal), and federated sharding mappings. |
| `train_teacher.py` | Trains the centralized `TeacherCGRU` oracle using a `WeightedRandomSampler` to combat dataset imbalance. |
| `losses.py` | Contains the `xkd_attention_loss` function handling the Tri-Objective Distillation. |
| `server_simulation.py` | The Flower (`flwr`) Server orchestrating the federated distillation process using a custom `SaveModelStrategy`. |
| `federated_client.py` | The Flower Client logic where the EdgeEEGNet student learns locally from the frozen Teacher model. |
| `benchmark_ieee.py` | Head-to-head ablation study comparing the Teacher and Student model across dynamic thresholds, F1-scores, and inference times. |
| `explainability_engine.py`| Generates `ieee_explainability_figure.png` using 1D Grad-CAM and Occlusion Sensitivity. |
| `sanity_check.py` | Evaluates the deployed edge model using a strict 95% softmax confidence gate to suppress false positives. |
| `personalized_audit.py` | Evaluates model robustness on distinct datasets using dynamic Z-Score Normalization to correct for covariate shifts. |
| `model_stats.py` | Utility to calculate and display model parameter counts, memory footprints, and compression ratios. |

---

## 🚀 Pipeline Workflow

### 1. Data Ingestion & Preprocessing
Extract clinical EDF data into model-ready PyTorch tensors.
```bash
python data_pipeline.py
python diagnose_data.py
