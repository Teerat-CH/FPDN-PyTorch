import numpy as np
import torch

def two_sum(x, y):
    s = x + y
    w = s - x
    v = s - w
    a = y - w
    b = v - x
    e = a - b
    return s, e

def compensated_sum_torch(arr):
    flat = arr.view(-1)
    device = flat.device
    dtype = flat.dtype

    s = torch.tensor(0., dtype=dtype, device=device)
    c = torch.tensor(0., dtype=dtype, device=device)

    for x in flat:
        s1, e1 = two_sum(s, x) # should be s, e1 = two_sum(s, x)
        s, e2 = two_sum(s1, c) # see if this should be removed
        c = e1 + e2 # c = c + e1

    s, rem = two_sum(s, c) # this should be s + c
    s = s + rem # not useful?
    return s

def compensated_matmul(A, B):
    num_row_A, num_col_A = A.shape
    num_row_B, num_col_B = B.shape

    result = np.zeros((num_row_A, num_col_B), dtype=A.dtype)
    error = np.zeros_like(result)

    for i in range(num_col_A):
        A_ith_col = A[:, i:i+1]
        B_ith_row = B[i:i+1]
        prod = A_ith_col * B_ith_row
        s1, e1 = two_sum(result, prod)
        result, e2 = two_sum(s1, error)
        error = e1 + e2
    return result

if __name__ == "__main__":
    # Example usage
    X = [[1.0, 2.0, 3.0], [2, 1, 1]]
    X = np.array(X, dtype=np.float32)
    print(X.shape)
    W = [[1, 2], [3, 4], [1, 2]]
    W = np.array(W, dtype=np.float32)


    result = compensated_matmul(X, W)
    print(result)