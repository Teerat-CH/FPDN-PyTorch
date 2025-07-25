import torch
import torch.nn as nn
import numpy as np
from functions.randomFunction import RandomFunction

class RandomLayer(nn.Module):
    """
    A custom linear layer that uses our CustomLinearFunction.
    """
    def __init__(self, input_features, output_features):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        # Define trainable parameters (weight and bias)
        # nn.Parameter ensures they are registered as model parameters
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        self.bias = nn.Parameter(torch.empty(output_features))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5) # Same as nn.Linear
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        In the forward pass, we apply our custom function.
        """
        # Use the .apply() method to call the custom autograd function
        return RandomFunction.apply(x, self.weight, self.bias)

    def extra_repr(self):
        return f'input_features={self.input_features}, output_features={self.output_features}'
    