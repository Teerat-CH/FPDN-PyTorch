from matplotlib import pyplot as plt
from baseline import loss_logs as baseline_loss
# from improved import loss_logs as kahan_loss_true
from improve_test import loss_logs as kahan_loss_false
import torch

differences = [a - b for a, b in zip(kahan_loss_false, baseline_loss)]

plt.figure(figsize=(10, 5))
plt.plot(differences, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss Difference (baseline - kahan)')
plt.title('Loss Differences Between Baseline and Kahan Training Runs')
plt.grid(True)
plt.tight_layout()
plt.show()