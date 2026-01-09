# FPDN-PyTorch

We investigate how errors from floating-point arithmetic affect the training of deep neural networks. We provide custom layers and loss functions implementation that maintain higher numerical accuracy compared to standard implementations through compensated summation. The experiment examines how the number of perceptrons affects training loss in a more numerically accurate model compared to a baseline, and whether different orders of matrix addition introduce significant numerical errors that impact model learning.

#### Note
This repository includes multiple experimental iterations conducted through trial and error. A more organized and refined demo derived from these experiments is available here: https://colab.research.google.com/drive/1oVqbR6DorVDn3-HfVM9TXdsJ-VmM0i_4?usp=sharing

**Advisors**: Prof. Eliot Moss, Prof. Philip Thomas

## Project Structure

```
├── AccuMSELoss.py                   # Compensated MSE Loss implementation to replace torch.nn.MSELoss()
├── compensated_ops.py               # A file containing compensated matmul (compensated matrix multiplication) and compensated sum (summing a list/array of float with compensated summation)
├── logs.py                          # Logging utilities
├── plot.py                          # Plotting utilities
├── functions/                       # Custom autograd functions, defining forward pass and backward pass for an acculinear layer
├── layers/                          # Custom neural network layers
│   └── accuLinear.py                # Accurate Linear Layer with compensated ops
├── data/                            # Dataset files
├── Experiment - CIFAR/              # CIFAR dataset experiments
├── Experiment - Layer size/         # Layer size experiments (How number of perceptrons affect loss and accumulated error)
└── Experiment - Order of Addition/  # Addition order experiments (How the order of addition affects loss and accumulated error, which can be especially important in settings such as federated learning where results are aggregated from multiple sources.)
```

## Key Components

### Custom Layers
- **AccuLinearLayer**: A linear layer using compensated arithmetic for improved precision

### Custom Loss Functions
- **CompensatedMSELoss**: MSE loss with optional compensated summation for reduced floating-point errors

## Usage

### Basic Training Example

```python
import torch
import torch.nn as nn
from layers.accuLinear import AccuLinearLayer
from AccuMSELoss import CompensatedMSELoss

# Define model with accurate layers
model = nn.Sequential(
    AccuLinearLayer(input_size, hidden_size, True),
    nn.ReLU(),
    AccuLinearLayer(hidden_size, output_size, True)
)

# Use compensated loss function
loss_fn = CompensatedMSELoss(kahan=True)

# Standard training loop
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model(X_tensor)
    loss = loss_fn(y_pred, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Experiments

The project includes several experiments:
- **Layer Size**: Investigating the effect of layer dimensions on numerical precision
- **CIFAR**: Experiments on the CIFAR dataset (because convolution layer can generate arbitrary number of features)
- **Order of Addition**: Studying how summation order affects floating-point accuracy

## Requirements

- Python 3.11+
- PyTorch
- NumPy
- Pandas
