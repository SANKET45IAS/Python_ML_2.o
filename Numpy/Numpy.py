# pip install numpy

"""
==============================
 NumPy Complete Example Script
==============================

Author: Sanket's Learning Notes
Goal: Cover all the most useful parts of NumPy in one place
"""

# Step 1: Import NumPy
import numpy as np

# -----------------------------------------------------------
# 1. ARRAY CREATION
# -----------------------------------------------------------

# From Python list
arr1 = np.array([1, 2, 3, 4, 5])  # Creates a 1D array
print("1D Array:", arr1, "| dtype:", arr1.dtype)

# 2D Array (Matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", arr2)
print("Shape:", arr2.shape)    # Rows and Columns
print("Dimensions:", arr2.ndim) # Number of dimensions
print("Data type:", arr2.dtype) # Default dtype is int64/float64

# -----------------------------------------------------------
# 2. ARRAY CREATION HELPERS (CLEAN + CORRECT)
# -----------------------------------------------------------

print("\nArray Creation Helpers")

# 1. arange -> values in a range with step
arr3 = np.arange(0, 10, 2)  # Start=0, Stop=10, Step=2
print("Arange:", arr3)

# 2. reshape -> reshape a sequence of numbers into matrix
arrRange = np.arange(0, 12).reshape(3, 4)  # reshape to 3x4
print("Arange + Reshape (3x4):\n", arrRange)

# 3. linspace -> evenly spaced values
arr4 = np.linspace(0, 2, 5) # 5 evenly spaced points between 0 and 2
print("Linspace:", arr4)

# 4. zeros -> matrix of all 0s
print("Zeros (2x3):\n", np.zeros((2,3)))

# 5. ones -> matrix of all 1s
print("Ones (2x3):\n", np.ones((2,3)))

# 6. identity / eye -> identity matrix
print("Identity matrix (3x3):\n", np.eye(3))

# 7. full -> fill with custom value
arr_full = np.full((2, 3), 7)
print("Full (2x3) with 7:\n", arr_full)

# 8. diag -> diagonal matrix
arr_diag = np.diag([10, 20, 30])
print("Diagonal Matrix:\n", arr_diag)

# 9. random integers + random floats
print("Random Integers (0–9, 2x2):\n", np.random.randint(0, 10, (2, 2)))
print("Random Uniform Floats (0–1, 2x2):\n", np.random.random((2, 2)))


# -----------------------------------------------------------
# 3. INDEXING & SLICING
# -----------------------------------------------------------

arr5 = np.array([10, 20, 30, 40, 50])
print("\nIndexing & Slicing")
print("First element:", arr5[0])
print("Slice (1:4):", arr5[1:4]) # 20, 30, 40

matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("\nMatrix:\n", matrix)
print("Element [1,2]:", matrix[1,2])   # Row=1, Col=2 -> 6
print("Row 1:", matrix[1])             # Entire 2nd row
print("Column 2:", matrix[:,2])        # Entire 3rd column

# -----------------------------------------------------------
# 4. VECTORIZED OPERATIONS (NO LOOPS)
# -----------------------------------------------------------

arr6 = np.array([1, 2, 3, 4, 5])
print("\nVectorized Operations")
print("Multiply by 2:", arr6 * 2)
print("Add 10:", arr6 + 10)
print("Square:", arr6 ** 2)

arr7 = np.array([10, 20, 30, 40, 50])
print("Element-wise Sum:", arr6 + arr7)
print("Element-wise Product:", arr6 * arr7)
print("Element-wise Division:", arr7 / arr6)

# -----------------------------------------------------------
# 5. AGGREGATIONS / STATISTICS
# -----------------------------------------------------------

print("\nAggregations")
print("Sum:", arr6.sum())
print("Mean:", arr6.mean())
print("Max:", arr6.max())
print("Min:", arr6.min())
print("Standard Deviation:", arr6.std())

# -----------------------------------------------------------
# 6. BROADCASTING (DIFFERENT SHAPES)
# -----------------------------------------------------------

matrix = np.array([[1,2,3],[4,5,6]])
print("\nBroadcasting Examples")
print("Matrix:\n", matrix)

print("Add scalar 10:\n", matrix + 10) # Scalar broadcast

row = np.array([10,20,30]) # Row vector
print("Add Row Vector:\n", matrix + row)

col = np.array([[100],[200]]) # Column vector
print("Add Column Vector:\n", matrix + col)

# -----------------------------------------------------------
# 7. LINEAR ALGEBRA
# -----------------------------------------------------------

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print("\nLinear Algebra")
print("Matrix A:\n", A)
print("Matrix B:\n", B)

# 1. Matrix multiplication
print("Matrix Multiplication (A·B):\n", np.dot(A, B))

# 2. Transpose
print("Transpose of A:\n", A.T)

# 3. Determinant
print("Determinant of A:", np.linalg.det(A))

# 4. Inverse
print("Inverse of A:\n", np.linalg.inv(A))

# 5. Rank of a matrix
print("Rank of A:", np.linalg.matrix_rank(A))

# 6. Trace (sum of diagonal elements)
print("Trace of A:", np.trace(A))

# 7. Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A:", eigenvalues)
print("Eigenvectors of A:\n", eigenvectors)

# 8. Solve system of linear equations (Ax = b)
b = np.array([5, 6])
x = np.linalg.solve(A, b)
print("Solution of Ax = b where b = [5,6]:", x)

# 9. Norm (magnitude of vector/matrix)
print("Frobenius Norm of A:", np.linalg.norm(A))

# -----------------------------------------------------------
# 8. RANDOM NUMBERS
# -----------------------------------------------------------

print("\nRandom Numbers")
print("Random Integers (2x3):\n", np.random.randint(1, 10, size=(2,3)))
print("Random Floats (2x3):\n", np.random.rand(2,3))
print("Normal Distribution (2x3):\n", np.random.randn(2,3))

# -----------------------------------------------------------
# 9. RESHAPING & FLATTENING
# -----------------------------------------------------------

arr8 = np.arange(12)  # 0 to 11
print("\nReshaping & Flattening")
print("Original:", arr8)
reshaped = arr8.reshape(3,4)
print("Reshaped (3x4):\n", reshaped)

print("Flattened:", reshaped.flatten()) # Back to 1D

# -----------------------------------------------------------
# 🔟 INTERNAL INFO: SHAPE, STRIDES, MEMORY
# -----------------------------------------------------------

print("\nInternal Info")
print("Shape:", reshaped.shape)
print("Strides:", reshaped.strides) # Bytes to move between elements
print("Data pointer (memory address):", reshaped.__array_interface__['data'])
print("Total bytes:", reshaped.nbytes) # Total memory used
print("Item size (bytes per element):", reshaped.itemsize)
print("Number of elements:", reshaped.size) # Total number of elements
print("Number of dimensions:", reshaped.ndim) # Number of dimensions
print("Data type:", reshaped.dtype) # Data type of elements]
print("Is C-contiguous:", reshaped.flags['C_CONTIGUOUS']) # Memory layout'
print("Is F-contiguous:", reshaped.flags['F_CONTIGUOUS']) # Fortran layout
print("Is writeable:", reshaped.flags['WRITEABLE']) # Can be modified
print("Is aligned:", reshaped.flags['ALIGNED']) # Memory alignment

# -----------------------------------------------------------
# EXTRA: MUTABILITY IN NUMPY ARRAYS
# -----------------------------------------------------------

print("\n==============================")
print("   MUTABILITY EXAMPLES")
print("==============================")

# Example 1: Change a single element
arr_mut = np.array([1, 2, 3, 4, 5])
print("Original:", arr_mut)
arr_mut[1] = 99
print("After changing index 1 to 99:", arr_mut)

# Example 2: Change a slice (multiple elements at once)
arr_mut[2:4] = [77, 88]
print("After changing slice [2:4]:", arr_mut)

# Example 3: Broadcasting assignment (set all values at once)
arr_mut[:] = 100
print("After broadcasting assignment (all = 100):", arr_mut)

# Example 4: Mutability in 2D array (replace a whole row)
matrix2 = np.array([[1,2,3],[4,5,6]])
print("\nOriginal 2D Matrix:\n", matrix2)
matrix2[0] = [10,20,30]  # Replace entire first row
print("After replacing first row:\n", matrix2)

# Example 5: Type enforcement (int array truncates float values)
arr_int = np.array([1, 2, 3], dtype=int)
print("\nOriginal int array:", arr_int)
arr_int[0] = 3.14   # Will truncate float → becomes 3
print("After assigning 3.14 to index 0:", arr_int)

