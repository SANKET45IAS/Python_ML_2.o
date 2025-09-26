# ===========================================================
# 6. BROADCASTING & VECTORIZATION
# ===========================================================
import numpy as np


matrix = np.array([[1,2,3],[4,5,6]])
print("\nMatrix:\n", matrix)

# Broadcasting scalar
print("Add 10:\n", matrix + 10)

# Broadcasting row vector
row = np.array([10,20,30])
print("Matrix + Row:\n", matrix + row)

# Broadcasting column vector
col = np.array([[100],[200]])
print("Matrix + Column:\n", matrix + col)

# Vectorization (no loops needed)
arr = np.arange(1,6)
print("Original:", arr)
print("Squared:", arr**2)     # vectorized operation
