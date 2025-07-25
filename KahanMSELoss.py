import torch
import torch.nn as nn

class KahanMSELoss(nn.Module):
    def __init__(self, reduction='mean', kahan=True):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), "Reduction must be 'mean', 'sum', or 'none'"
        self.reduction = reduction
        self.kahan = kahan

    def forward(self, input, target):

        if self.kahan:

            squared_errors = (input - target) ** 2

            if self.reduction == 'none':
                return squared_errors

            loss = torch.zeros(1, dtype=input.dtype, device=input.device)
            c = torch.zeros(1, dtype=input.dtype, device=input.device)

            for err in squared_errors.view(-1):
                y = err - c
                t = loss + y
                c = (t - loss) - y
                loss = t

            if self.reduction == 'mean':
                loss = loss / squared_errors.numel()

            return loss

        else:
            loss = nn.functional.mse_loss(input, target, reduction=self.reduction)
            return loss