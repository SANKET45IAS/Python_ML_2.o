# ===========================================================
# 2. ARRAY PROPERTIES & OPERATIONS
# ===========================================================
import numpy as np


arr = np.array([[1,2,3],[4,5,6]])

print("\nArray Properties")
print("Shape:", arr.shape)      # rows, cols
print("Dimensions:", arr.ndim)  # number of dimensions
print("Size (#elements):", arr.size)
print("Data type:", arr.dtype)

# Operations
print("\nBasic Operations")
print("Add 10:", arr + 10)    # element-wise add
print("Multiply by 2:", arr * 2)
print("Square:", arr ** 2)
print("Sum of all elements:", arr.sum())
print("Row-wise sum:", arr.sum(axis=1))  # across rows
print("Column-wise sum:", arr.sum(axis=0))  # across columns
