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
    # mean_diff = np.percentile(diffs, 99, axis=0)
    mean_diff = np.mean(diffs, axis=0)
    std_diff = np.std(diffs, axis=0)
    return mean_diff, std_diff

# Compute both comparisons
mean_128, std_128 = compute_diff(
    "Experiment - Layer size/results/baseline_kahan_false_128.json",
    "Experiment - Layer size/results/improved_kahan_true_128.json"
)

mean_512, std_512 = compute_diff(
    "Experiment - Layer size/results/baseline_kahan_false_512.json",
    "Experiment - Layer size/results/improved_kahan_true_512.json"
)

mean_1024, std_1024 = compute_diff(
    "Experiment - Layer size/results/baseline_kahan_false_1024.json",
    "Experiment - Layer size/results/improved_kahan_true_1024.json"
)

mean_2048, std_2048 = compute_diff(
    "Experiment - Layer size/results/baseline_kahan_false_2048.json",
    "Experiment - Layer size/results/improved_kahan_true_2048.json"
)

mean_3000, std_3000 = compute_diff(
    "Experiment - Layer size/results/baseline_kahan_false_3000.json",
    "Experiment - Layer size/results/improved_kahan_true_3000.json"
)

# Plot
epochs_128 = np.arange(1, len(mean_128) + 1)
epochs_512 = np.arange(1, len(mean_512) + 1)
epochs_1024 = np.arange(1, len(mean_1024) + 1)
epochs_2048 = np.arange(1, len(mean_2048) + 1)
epochs_3000 = np.arange(1, len(mean_3000) + 1)

plt.figure(figsize=(8,5))

plt.plot(epochs_128, mean_128, label="128 units (Baseline - Improved)", color="purple")
# plt.fill_between(epochs_128, mean_128 - std_128, mean_128 + std_128,
#                  color="purple", alpha=0.2)

plt.plot(epochs_512, mean_512, label="512 units (Baseline - Improved)", color="orange")
# plt.fill_between(epochs_512, mean_512 - std_512, mean_512 + std_512,
#                  color="orange", alpha=0.2)

plt.plot(epochs_1024, mean_1024, label="1024 units (Baseline - Improved)", color="blue")
# plt.fill_between(epochs_1024, mean_1024 - std_1024, mean_1024 + std_1024,
#                  color="blue", alpha=0.2)

plt.plot(epochs_2048, mean_2048, label="2048 units (Baseline - Improved)", color="green")
# plt.fill_between(epochs_2048, mean_2048 - std_2048, mean_2048 + std_2048,
#                  color="orange", alpha=0.2)

plt.plot(epochs_3000, mean_3000, label="3000 units (Baseline - Improved)", color="red")
# plt.fill_between(epochs_3000, mean_3000 - std_3000, mean_3000 + std_3000,
#                  color="red", alpha=0.2)

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.yscale("log")
plt.title("Mean Loss Difference Across Epochs (Baseline - Improved)")
plt.xlabel("Epoch")
plt.ylabel("Loss Difference")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()