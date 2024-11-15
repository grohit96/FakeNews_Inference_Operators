import numpy as np

# Paths to the dev and train set files
dev_set_path = "/scratch/rganesh5/PaperGraph_release_connected/dev_set_0.npy"  # Replace with actual paths
train_set_path = "/scratch/rganesh5/PaperGraph_release_connected/train_set_0.npy"  # Replace with actual paths

# Load the files
dev_set = np.load(dev_set_path, allow_pickle=True)
train_set = np.load(train_set_path, allow_pickle=True)

# Print the types and first few entries
print("Dev Set Format:")
print(type(dev_set), dev_set[:5])

print("\nTrain Set Format:")
print(type(train_set), train_set[:5])
