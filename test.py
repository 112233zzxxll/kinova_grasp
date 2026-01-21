import numpy as np

A = [[0,0,0], [1,1,1]]
B = [[2,2,2], [3,3,3]]
A = np.array(A)
B = np.array(B)

result = np.sum((A[0] - B[0]) ** 2)
print(result)