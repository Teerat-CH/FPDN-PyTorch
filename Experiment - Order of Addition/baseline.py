import torch
import numpy as np
import os 
import json

def matmul(A, B, random_seed=42):
    np.random.seed(random_seed)

    num_row_A, num_col_A = A.shape
    num_row_B, num_col_B = B.shape

    result = np.zeros((num_row_A, num_col_B), dtype=A.dtype)

    index_order = np.random.permutation(num_col_A)

    for i in index_order:
        A_ith_col = A[:, i:i+1]
        B_ith_row = B[i:i+1]
        prod = A_ith_col * B_ith_row
        result += prod
    return result

input_tensor = torch.load('/Users/teeratc/Documents/FPDN-PyTorch/Experiment - Order of Addition/debug_input_49.pt')
weight_tensor = torch.load('/Users/teeratc/Documents/FPDN-PyTorch/Experiment - Order of Addition/debug_weight_49.pt')

input_np = input_tensor.detach().numpy()
weight_np = weight_tensor.T.detach().numpy()

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "json_outputs_numpy_49")
os.makedirs(output_dir, exist_ok=True)

for run in range(0, 50, 1):
    output_filename = os.path.join(output_dir, f"output_{run}.json")

    if os.path.exists(output_filename):
        print(f"Skipping run {run}, file already exists.")
        continue

    print(f"Processing run {run}...")
    result_list = matmul(input_np, weight_np, random_seed=run).tolist()
    
    data_to_save = {
        "run_id": run,
        "result": result_list
    }
    
    with open(output_filename, "w") as f:
        json.dump(data_to_save, f, indent=4)
    
print("\nAll runs completed and saved.")