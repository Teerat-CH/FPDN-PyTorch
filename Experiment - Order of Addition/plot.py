import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch

def two_sum(x, y):
    s = x + y
    w = s - x
    v = s - w
    a = y - w
    b = v - x
    e = a - b
    return s, e

def matrices_mean(matrices):
    result = np.zeros(matrices[0].shape, dtype=matrices[0].dtype)
    error = np.zeros_like(result)

    for matrix in matrices:
        s1, e1 = two_sum(result, matrix)
        result, e2 = two_sum(s1, error)
        error = e1 + e2

    mean_matrix = result / len(matrices)
    return mean_matrix
print("Starting analysis of compensated matmul results...")
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "json_outputs_compensated_matmul_1")

    json_files = glob.glob(os.path.join(results_dir, "output_*.json"))
    
    if not json_files:
        print("Error: No 'output_*.json' files found in the 'results' directory.")

    files = [f for f in json_files]

    print(f"Found {len(files)} result files for analysis.")
    
    matrices = []

    for file_path in files:
        with open(file_path, 'r') as f:
            current_data = np.array(json.load(f)['result'])
        
        matrices.append(current_data)
        print(f"Loaded matrix from {file_path} with shape {current_data.shape}")

    print("Calculating baseline matrix using compensated mean...")
    baseline = matrices_mean(matrices)

    differences = []

    for matrix in matrices:
        diff = np.abs((matrix - baseline))
        diff = np.divide(diff, np.abs(baseline))
        differences.append(diff)
        print(f"Calculated difference matrix with shape {diff.shape}")

    # mean_diff = matrices_mean(differences)

    # print("Calculating 90th percentile of differences...")
    # p90 = np.percentile(differences, 90, axis=0)

    # scalar_differences = p90.flatten().tolist()

    # print("Plotting the distribution of differences...")
    # plt.figure(figsize=(10, 6))
    # plt.hist(scalar_differences, bins=30, edgecolor='black', alpha=0.7)
    # plt.title('Distribution of Differences from Baseline - Compensated MatMul')
    # plt.xlabel('calculate mean -> absolute difference -> mean of differences -> flatten')
    # plt.ylabel('Frequency')
    # plt.grid(True, linestyle='--', alpha=0.6)
    
    # mean_diff = np.mean(scalar_differences)
    # plt.axvline(mean_diff, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_diff:.2e}')
    # plt.legend()
    # plt.show()

    print("Plotting the entire distribution of all flattened differences...")
    
    flattened_differences = np.concatenate([d.flatten() for d in differences])
    
    # Calculate the 95th percentile to draw a line on the plot
    p95_threshold = np.percentile(flattened_differences, 95)
    print(f"95th percentile threshold: {p95_threshold:.2e}")

    plt.figure(figsize=(10, 6))
    # Plot the entire 'flattened_differences' array
    plt.hist(flattened_differences, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    # Update the title to reflect the full distribution
    plt.title('Distribution of All Flattened Differences')
    plt.xlabel('Relative Difference')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Keep the line to show where the 95th percentile is on the full distribution
    plt.axvline(p95_threshold, color='darkred', linestyle='dashed', linewidth=2, label=f'95th Percentile: {p95_threshold:.2e}')
    plt.legend()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")