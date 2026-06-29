from train_teacher import FullRealEEGDataset

# Initialize with a dummy path to just inspect the object
dataset = FullRealEEGDataset(tensor_dir="./eeg_tensors", summary_path="./chb01-summary.txt")

# This prints every attribute available in your class
print("Attributes found in your dataset class:")
print(dir(dataset))