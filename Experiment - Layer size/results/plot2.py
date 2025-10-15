import json
import numpy as np
import matplotlib.pyplot as plt

def to_array(data_dict):
    runs = list(data_dict.values())
    return np.array(runs)

def compute_diff(fileA, fileB):
    with open(fileA, "r") as f:
        expA = json.load(f)
    with open(fileB, "r") as f:
        expB = json.load(f)
    arrA = to_array(expA)
    arrB = to_array(expB)
    diffs = np.abs(arrA - arrB)
    # mean_diff = np.mean(diffs, axis=0)
    std_diff = np.std(diffs, axis=0)

    mean_diff = np.percentile(diffs, 95, axis=0)
    return mean_diff, std_diff

mean_1024, std_1024 = compute_diff(
    "Experiment - Layer size/results/improved_kahan_true_1024.json",
    "Experiment - Layer size/results/baseline_kahan_false_1024.json"
)

mean_2048, std_2048 = compute_diff(
    "Experiment - Layer size/results/improved_kahan_true_2048.json",
    "Experiment - Layer size/results/baseline_kahan_false_2048.json"
)

mean_3072, std_3072 = compute_diff(
    "Experiment - Layer size/results/improved_kahan_true_3072.json",
    "Experiment - Layer size/results/baseline_kahan_false_3072.json"
)

mean_4096, std_4096 = compute_diff(
    "Experiment - Layer size/results/improved_kahan_true_4096.json",
    "Experiment - Layer size/results/baseline_kahan_false_4096.json"
)

epochs_1024 = np.arange(1, len(mean_1024) + 1)
epochs_2048 = np.arange(1, len(mean_2048) + 1)
epochs_3072 = np.arange(1, len(mean_3072) + 1)
epochs_4096 = np.arange(1, len(mean_4096) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs_1024, mean_1024, label="1024 units (Improved - Unity)", color="purple")
# plt.fill_between(epochs_128, mean_128 - std_128, mean_128 + std_128, color="purple", alpha=0.2)
plt.plot(epochs_2048, mean_2048, label="2048 units (Improved - Unity)", color="green")
# plt.fill_between(epochs_128, mean_128 - std_128, mean_128 + std_128, color="purple", alpha=0.2)
plt.plot(epochs_3072, mean_3072, label="3072 units (Improved - Unity)", color="orange")
# plt.fill_between(epochs_3072, mean_3072 - std_3072, mean_3072 + std_3072, color="orange", alpha=0.2)
plt.plot(epochs_4096, mean_4096, label="4096 units (Improved - Unity)", color="brown")
# plt.fill_between(epochs_4096, mean_4096 - std_4096, mean_3072 + std_3072, color="orange", alpha=0.2)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Mean Loss Difference Across Epochs (Improved - Unity, 128 units)")
plt.xlabel("Epoch")
plt.ylabel("Loss Difference")
plt.yscale("log")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()