import numpy as np

# ===========================================================
# 1. NUMPY BASICS
# ===========================================================
# Create arrays from Python lists
arr = np.array([1, 2, 3, 4, 5])
print("Basic 1D array:", arr)

# 2D array (matrix)
matrix = np.array([[1,2,3],[4,5,6]])
print("\n2D Array:\n", matrix)

# Useful initializers
print("Zeros:\n", np.zeros((2,3)))
print("Ones:\n", np.ones((2,3)))
print("Range with arange:", np.arange(0, 10, 2))   # step
print("Linspace (even spacing):", np.linspace(0, 1, 5))
