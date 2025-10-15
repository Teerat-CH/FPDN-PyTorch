import numpy as np

def two_sum(x, y):
    s = x + y
    w = s - x
    v = s - w
    a = y - w
    b = v - x
    e = a - b
    return s, e

def compensated_sum(arr):
    s = 0.0
    c = 0.0
    for x in arr:
        s1, e1 = two_sum(s, x)    # add element with error correction
        s, e2 = two_sum(s1, c)    # add compensation
        c = e1 + e2               # update compensation
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

def kahan_dot(X, W):
    batch_size, input_dim = X.shape
    output_dim = W.shape[1]

    result = np.zeros((batch_size, output_dim), dtype=X.dtype)
    c = np.zeros_like(result)

    for i in range(input_dim):
        front = X[:, i:i+1]
        back = W[i:i+1]
        prod = front * back
        temp_sum, c_new = two_sum(prod, c) # add product with compensation
        new_result, result_err = two_sum(result, temp_sum) # add temp_sum to result
        result = new_result
        c = c_new + result_err
    return result + c

A = np.array([[ 1e16, 1.0, -1e16], [ 1e16, 1.0, -1e16]])
B = np.array([[1,0], [1,0], [1,0]])
print(np.matmul(A, B))
print(compensated_matmul(A, B))
print(kahan_dot(A, B))