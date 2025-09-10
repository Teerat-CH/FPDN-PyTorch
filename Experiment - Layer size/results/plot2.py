import json
import numpy as np
import matplotlib.pyplot as plt

def to_array(data_dict):
    runs = list(data_dict.values())
    return np.array(runs)

# Load data for 2048 units
with open("Experiment - Layer size/results/baseline_kahan_false_2048.json", "r") as f:
    expA = json.load(f)
with open("Experiment - Layer size/results/improved_kahan_true_2048.json", "r") as f:
    expB = json.load(f)

arrA = to_array(expA)
arrB = to_array(expB)

diffs = arrA - arrB  # shape: (num_runs, num_epochs)

# Find the run with the highest total absolute difference
run_scores = np.sum(np.abs(diffs), axis=1)
max_idx = np.argmax(run_scores)

best_diff = diffs[max_idx]
epochs = np.arange(1, len(best_diff) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs, best_diff, label=f"2048 units (Run {max_idx}, max diff)", color="green")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Loss Difference Across Epochs (Single Run with Max Difference, 2048 units)")
plt.xlabel("Epoch")
plt.ylabel("Loss Difference")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()