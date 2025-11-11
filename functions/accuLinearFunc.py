import torch
import torch.nn as nn
import numpy as np
from compensated_ops import compensated_matmul

class AccuLinearFunction(torch.autograd.Function):

    call_count = 0

    @staticmethod
    def forward(ctx, input, weight, bias, compensated):
        AccuLinearFunction.call_count += 1
        print(f"Forward called {AccuLinearFunction.call_count} time(s)")

        # --- Debug code to save tensors ---
        # This will save the input and weight from the first forward pass
        if not hasattr(AccuLinearFunction, 'saved_tensors_for_debug') or AccuLinearFunction.call_count == 49:
            print("DEBUG: Saving input and weight tensors to files...")
            torch.save(input, f'debug_input_{AccuLinearFunction.call_count}.pt')
            torch.save(weight, f'debug_weight_{AccuLinearFunction.call_count}.pt')
            AccuLinearFunction.saved_tensors_for_debug = True
        # --- End of debug code ---

        # Save tensors for backward pass and the compensated flag
        ctx.save_for_backward(input, weight)
        ctx.compensated = compensated
        
        if compensated:
            # Convert tensors to numpy for compensated_matmul
            input_np = input.detach().numpy()
            weight_np = weight.T.detach().numpy() # Transpose weight for matmul
            bias_np = bias.detach().numpy()
            output = compensated_matmul(input_np, weight_np) + bias_np
            return torch.from_numpy(output).to(input.device, dtype=input.dtype)
        else:
            # Use standard torch matrix multiplication
            output = torch.addmm(bias, input, weight.T)
            return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        compensated = ctx.compensated

        grad_input = grad_weight = grad_bias = None

        if compensated:
            # Convert tensors to numpy for compensated_matmul
            grad_output_np = grad_output.detach().numpy()
            input_np = input_tensor.detach().numpy()
            weight_np = weight.detach().numpy()

            grad_input_np = compensated_matmul(grad_output_np, weight_np)
            grad_weight_np = compensated_matmul(grad_output_np.T, input_np)
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
        # (excluding ctx), so we return None for the 'compensated' input.
        return grad_input, grad_weight, grad_bias, None
