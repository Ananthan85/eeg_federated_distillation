import mne
import numpy as np
import torch
import os
import glob

def process_all_edfs(output_dir="./eeg_tensors", window_size_sec=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find all EDF files in the current directory
    edf_files = glob.glob("*.edf")
    if not edf_files:
        print("[Error] No .edf files found in the directory.")
        return

    print("="*50)
    print(f"[Ingestion Engine] Found {len(edf_files)} EDF files. Beginning batch processing...")
    print("="*50)

    total_windows_extracted = 0

    for edf_path in edf_files:
        base_name = os.path.basename(edf_path).replace('.edf', '')
        print(f"-> Processing {base_name}...")
        
        try:
            # Suppress MNE's verbose output for clean terminal logging
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # The standard 18 channels your architecture is locked onto
            target_channels = [
                'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2',
                'FZ-CZ', 'CZ-PZ'
            ]
            
            # Handle channel naming inconsistencies across different EDF files
            available_channels = raw.ch_names
            actual_channels = []
            for tc in target_channels:
                # Some EDFs use T8-P8 instead of T8-P8-0
                if tc in available_channels:
                    actual_channels.append(tc)
                elif tc == 'T8-P8-0' and 'T8-P8' in available_channels:
                    actual_channels.append('T8-P8')
            
            raw.pick_channels(actual_channels)
            
            sfreq = raw.info['sfreq']
            data, times = raw[:]
            
            window_samples = int(window_size_sec * sfreq)
            num_windows = data.shape[1] // window_samples
            
            file_windows = 0
            for i in range(num_windows):
                start_idx = i * window_samples
                end_idx = start_idx + window_samples
                window_data = data[:, start_idx:end_idx]
                
                tensor_data = torch.tensor(window_data, dtype=torch.float32)
                
                # Save with the specific prefix so the DataLoader can map it to the summary file
                file_name = f"{base_name}_win_{i}.pt"
                torch.save(tensor_data, os.path.join(output_dir, file_name))
                file_windows += 1
                
            total_windows_extracted += file_windows
            print(f"   Saved {file_windows} tensors.")
            
        except Exception as e:
            print(f"   [Error] Failed to process {base_name}: {e}")

    print("="*50)
    print(f"[Ingestion Engine] Batch Complete! Total windows extracted: {total_windows_extracted}")
    print("="*50)

if __name__ == "__main__":
    process_all_edfs()
