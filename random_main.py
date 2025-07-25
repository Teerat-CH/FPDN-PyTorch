from layers.randomLayer import RandomLayer
import torch
import torch.nn as nn
import numpy as np

input_size = 10
hidden_size = 32
output_size = 1
learning_rate = 0.01
epochs = 50

model = nn.Sequential(
    RandomLayer(input_size, hidden_size),
    nn.ReLU(),
    RandomLayer(hidden_size, output_size)
)

print(model)

batch_size = 64
X_train = torch.randn(batch_size, input_size)
y_train = torch.randn(batch_size, output_size)

MSE = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model(X_train)
    loss = MSE(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

grad_check = model[0].weight.grad
if grad_check is not None:
    print("gradients for weights are computed successfully")
    print("Gradient norm:", grad_check.norm().item())
else:
    print("something is wrong.")