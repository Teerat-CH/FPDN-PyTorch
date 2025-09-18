import sys
import os
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from layers.accuLinear import AccuLinearLayer
from KahanMSELoss import KahanMSELoss
from settings import hidden_size, output_size, learning_rate, epochs, precision, X, y, seed

torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

data = unpickle("/Users/teeratc/Desktop/Research/Floating Point + Deep Network/FPDN-PyTorch/data/cifar-10-batches-py/data_batch_1")

# Extract features and labels from the CIFAR-10 batch
X = data[b'data']  # shape: (10000, 3072), uint8
y = np.array(data[b'labels'])  # shape: (10000,), int

# Normalize features to [0, 1]
X = X.astype(np.float32) / 255.0

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=precision)
y_tensor = torch.tensor(y, dtype=torch.long)  # For classification

# Reshape X_tensor to (N, 3, 32, 32) for Conv2d
X_tensor = X_tensor.view(-1, 3, 32, 32)

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),  # (N, 32, 15, 15)
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.ReLU(),
    nn.Flatten(),
    AccuLinearLayer(32 * 15 * 15, 512, True),  # 32*15*15 = 7200
    nn.ReLU(),
    AccuLinearLayer(512, 10, True)
)

model.to(precision)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_logs = []

for epoch in range(epochs):
    y_pred = model(X_tensor)
    loss = loss_fn(y_pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
    loss_logs.append(loss.item())

script_dir = os.path.dirname(os.path.abspath(__file__))
results_filename = os.path.join(script_dir, "results/kahan_false.json")
os.makedirs(os.path.dirname(results_filename), exist_ok=True)

if os.path.exists(results_filename):
    with open(results_filename, "r") as f:
        data = json.load(f)
else:
    data = {}

data[str(seed)] = loss_logs

with open(results_filename, "w") as f:
    json.dump(data, f, indent=4)

# save_dir = os.path.join(script_dir, "savemodels/baseline_kahan_false")
# os.makedirs(save_dir, exist_ok=True)
# model_path = os.path.join(save_dir, f"{seed}.pth")

# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'epochs': epochs,
#     'loss_logs': loss_logs,
#     'seed': seed
# }, model_path)