# ===========================================================
# 3. INDEXING & SLICING ARRAYS
# ===========================================================
import numpy as np


arr = np.array([10, 20, 30, 40, 50])
print("\nIndexing & Slicing")
print("First element:", arr[0])
print("Slice 1:4:", arr[1:4])

matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("\nMatrix:\n", matrix)
print("Element [1,2]:", matrix[1,2])   # row=1, col=2
print("Row 1:", matrix[1])             # full row
print("Column 2:", matrix[:,2])        # full column
print("Submatrix:\n", matrix[0:2,1:3]) # slicing submatrix
