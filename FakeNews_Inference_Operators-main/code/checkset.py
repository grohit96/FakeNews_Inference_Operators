import numpy as np

dev_set_path = "/scratch/rganesh5/PaperGraph_release_connected/dev_set_0.npy"  # Replace with actual paths
train_set_path = "/scratch/rganesh5/PaperGraph_release_connected/train_set_0.npy"  # Replace with actual paths

dev_set = np.load(dev_set_path, allow_pickle=True)
train_set = np.load(train_set_path, allow_pickle=True)

print("Dev Set Format:")
print(type(dev_set), dev_set[:5])

print("\nTrain Set Format:")
print(type(train_set), train_set[:5])
