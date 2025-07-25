import torch
import torch.nn as nn
import numpy as np

class RandomFunction(torch.autograd.Function):
    """
    Defines the forward and backward passes for a custom linear operation.
    """
    @staticmethod
    def forward(ctx, input_tensor, weight, bias):
        print('--- Executing custom forward with RANDOM output ---')
        """
        Forward pass: output = RANDOM TENSOR
        """
        # Save tensors for the backward pass
        ctx.save_for_backward(input_tensor, weight)
        
        # Determine output shape
        batch_size = input_tensor.shape[0]
        output_features = weight.shape[0]
        output_shape = (batch_size, output_features)

        # Get device and dtype for creating the new tensor
        device = input_tensor.device
        dtype = input_tensor.dtype

        # Generate a random output tensor instead of calculating it
        output = torch.randn(output_shape, device=device, dtype=dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print('--- Executing custom backward with RANDOM gradients ---')
        """
        Backward pass: compute gradients for input, weight, and bias.
        This version generates RANDOM tensors to prove PyTorch trusts our logic.
        """
        # Retrieve saved tensors to get their shapes
        input_tensor, weight = ctx.saved_tensors
        
        # Get device and dtype for creating new tensors
        device = input_tensor.device
        dtype = input_tensor.dtype

        # Gradients for forward's inputs (input_tensor, weight, bias)
        grad_input = None
        grad_weight = None
        grad_bias = None

        # Instead of correct calculations, we generate random gradients.
        # PyTorch will accept these without question.
        if ctx.needs_input_grad[0]:
            # Create a random numpy array and convert it to a torch tensor
            rand_np_input = np.random.randn(*input_tensor.shape)
            grad_input = torch.from_numpy(rand_np_input).to(device, dtype=dtype)

        if ctx.needs_input_grad[1]:
            # Create a random numpy array for the weight gradient
            rand_np_weight = np.random.randn(*weight.shape)
            grad_weight = torch.from_numpy(rand_np_weight).to(device, dtype=dtype)

        if ctx.needs_input_grad[2]:
            # Create a random numpy array for the bias gradient
            bias_shape = weight.shape[0]
            rand_np_bias = np.random.randn(bias_shape)
            grad_bias = torch.from_numpy(rand_np_bias).to(device, dtype=dtype)
            
        return grad_input, grad_weight, grad_bias