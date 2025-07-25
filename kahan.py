import numpy as np

def two_sum(x, y):
    s = x + y
    w = s - x
    v = s - w
    a = y - w
    b = v - x
    e = a - b
    return s, e

def kahan_dot(X, W):
    batch_size, input_dim = X.shape
    output_dim = W.shape[1]

    result = np.zeros((batch_size, output_dim), dtype=X.dtype)
    error  = np.zeros_like(result)

    for i in range(input_dim):
        prod = X[:, i:i+1] * W[i:i+1]
        temp_sum, temp_err = two_sum(result, prod + error)
        result = temp_sum
        error  = temp_err

    return result