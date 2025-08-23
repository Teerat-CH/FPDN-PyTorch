import sys
import os
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from settings import hidden_size, output_size, learning_rate, epochs, precision, X, y, seed

torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

X_tensor = torch.tensor(X.values, dtype=precision)
y_tensor = torch.tensor(y.values, dtype=precision).unsqueeze(1)

input_size = X_tensor.shape[1]

model = nn.Sequential(
    nn.Linear(input_size, hidden_size, True),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size, True),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size, True)
)

model.to(precision)

loss_fn = nn.MSELoss()
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
filename = os.path.join(script_dir, "results/baseline_pytorch.json")

if os.path.exists(filename):
    with open(filename, "r") as f:
        data = json.load(f)
else:
    data = {}

data[seed] = loss_logs

with open(filename, "w") as f:
    json.dump(data, f, indent=4)