import torch
import torch.nn as nn
from compensated_ops import compensated_sum_torch

class CompensatedMSELoss(nn.Module):
    def __init__(self, reduction='mean', kahan=True):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), "Reduction must be 'mean', 'sum', or 'none'"
        self.reduction = reduction
        self.kahan = kahan

    def forward(self, input, target):
        squared = (input - target).pow(2)

        if self.reduction == 'none':
            return squared

        if not self.kahan:
            return nn.F.mse_loss(input, target, reduction=self.reduction)

        total = compensated_sum_torch(squared)

        if self.reduction == 'mean':
            total = total / squared.numel()

        return total