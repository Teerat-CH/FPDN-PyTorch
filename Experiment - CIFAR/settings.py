import torch
import numpy as np
import os

hidden_size = 128
output_size = 1
learning_rate = 0.05
epochs = 25
precision = torch.float32

seed = int(os.environ.get("RANDOM_STATE", 42))

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

data = unpickle("/Users/teeratc/Desktop/Research/Floating Point + Deep Network/FPDN-PyTorch/data/cifar-10-batches-py/data_batch_1")

# Extract features and labels
X = data[b'data']  # shape: (10000, 3072), uint8
y = np.array(data[b'labels'])  # shape: (10000,), int

# Normalize features to [0, 1]
X = X.astype(np.float32) / 255.0

# Optionally, you can standardize or further preprocess here

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=precision)
y = torch.tensor(y, dtype=torch.long)  # For classification

# If you want to use y as regression (output_size=1), convert to float and unsqueeze
# y = torch.tensor(y, dtype=precision).unsqueeze(1)