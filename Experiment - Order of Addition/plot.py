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

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "json_outputs_compensated_matmul")

    json_files = glob.glob(os.path.join(results_dir, "output_*.json"))
    
    if not json_files:
        print("Error: No 'output_*.json' files found in the 'results' directory.")

    files = [f for f in json_files]
    
    matrices = []

    for file_path in files:
        with open(file_path, 'r') as f:
            current_data = np.array(json.load(f)['result'])
        
        matrices.append(current_data)

    baseline = matrices_mean(matrices)

    differences = []

    for matrix in matrices:
        diff = np.abs((matrix - baseline))
        differences.append(diff)

    mean_diff = matrices_mean(differences)

    scalar_differences = mean_diff.flatten().tolist()

    plt.figure(figsize=(10, 6))
    plt.hist(scalar_differences, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Differences from Baseline - Compensated MatMul')
    plt.xlabel('calculate mean -> absolute difference -> mean of differences -> flatten')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    mean_diff = np.mean(scalar_differences)
    plt.axvline(mean_diff, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_diff:.2e}')
    plt.legend()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")