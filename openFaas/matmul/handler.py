import numpy as np
from time import time

def matmul(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time()
    C = np.matmul(A, B)
    latency = time() - start
    return latency

def handle(req):
    """handle a request to the function
    Args:
        req (str): request body
    """

    n = int(req)
    result = matmul(n)
    print(result)
    return result
