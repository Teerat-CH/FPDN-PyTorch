import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from layers.accuLinear import AccuLinearLayer
from KahanMSELoss import KahanMSELoss

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

precision = torch.float64

data = pd.read_csv("data/GPA.csv", delimiter=',').head(8000)
features = [feature for feature in data.columns if feature != 'gpa']
X = data[features]
y = data.gpa

X = (X - X.min()) / (X.max() - X.min())

X_tensor = torch.tensor(X.values, dtype=precision)
y_tensor = torch.tensor(y.values, dtype=precision).unsqueeze(1)

input_size = X_tensor.shape[1]
hidden_size = 128
output_size = 1
learning_rate = 0.01
epochs = 25

model = nn.Sequential(
    AccuLinearLayer(input_size, hidden_size, False),
    nn.ReLU(),
    AccuLinearLayer(hidden_size, output_size, False)
)

model.to(precision)

loss_fn = KahanMSELoss(kahan=False)
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
