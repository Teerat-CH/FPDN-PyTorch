import torch
import torch.nn as nn
import numpy as np
from kahan import kahan_dot

class AccuLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, kahan):
        # Save tensors for backward pass and the kahan flag
        ctx.save_for_backward(input, weight)
        ctx.kahan = kahan
        
        if kahan:
            # Convert tensors to numpy for kahan_dot
            input_np = input.detach().numpy()
            weight_np = weight.T.detach().numpy() # Transpose weight for matmul
            bias_np = bias.detach().numpy()
            output = kahan_dot(input_np, weight_np) + bias_np
            return torch.from_numpy(output).to(input.device, dtype=input.dtype)
        else:
            # Use standard torch matrix multiplication
            output = torch.addmm(bias, input, weight.T)
            return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        kahan = ctx.kahan

        grad_input = grad_weight = grad_bias = None

        if kahan:
            # Convert tensors to numpy for kahan_dot
            grad_output_np = grad_output.detach().numpy()
            input_np = input_tensor.detach().numpy()
            weight_np = weight.detach().numpy()

            grad_input_np = kahan_dot(grad_output_np, weight_np)
            grad_weight_np = kahan_dot(grad_output_np.T, input_np)
            grad_bias_np = grad_output_np.sum(axis=0)

            # Convert results back to tensors
            grad_input = torch.from_numpy(grad_input_np).to(grad_output.device, dtype=grad_output.dtype)
            grad_weight = torch.from_numpy(grad_weight_np).to(grad_output.device, dtype=grad_output.dtype)
            grad_bias = torch.from_numpy(grad_bias_np).to(grad_output.device, dtype=grad_output.dtype)
        else:
            # Use standard torch operations for gradients
            grad_input = grad_output @ weight
            grad_weight = grad_output.T @ input_tensor
            grad_bias = grad_output.sum(dim=0)

        # The number of returned gradients must match the number of inputs to forward
        # (excluding ctx), so we return None for the 'kahan' input.
        return grad_input, grad_weight, grad_bias, None
