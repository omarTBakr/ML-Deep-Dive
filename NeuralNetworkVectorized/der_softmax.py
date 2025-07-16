import numpy as np


def softmax(arr):
    den = np.sum(np.exp(arr))
    return np.array([np.exp(num) / den for num in arr])


def softmax_grad_iterative(arr):
    """
    arr: pre calculated softmax

    """
    rows, cols = len(arr), len(arr)

    ret = np.zeros(shape=(rows, cols))

    for i in range(rows):
        for j in range(cols):
            if i == j:
                ret[i][j] = arr[i] * (1 - arr[i])
            else:
                ret[i][j] = -arr[i] * arr[j]

    return ret

def softmax_grad_vectorized(arr):
 
    ret = -np.outer(arr , arr  )
    
    # then we need to modify the diagonal 
    
    diagonal = arr*(1-arr)
    np.fill_diagonal(ret , diagonal)
    
    return ret 

if __name__ == "__main__":
    arr = [5, 7, 8]
    arr = softmax(arr)

    print(softmax_grad_iterative(arr))
    print(softmax_grad_vectorized(arr))
    
