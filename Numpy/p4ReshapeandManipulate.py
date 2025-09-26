# ===========================================================
# 4. RESHAPING & MANIPULATION
# ===========================================================
import numpy as np


arr = np.arange(12)
print("\nOriginal:", arr)

reshaped = arr.reshape(3,4)   # reshape to 3x4
print("Reshaped (3x4):\n", reshaped)

flattened = reshaped.flatten() # convert back to 1D
print("Flattened:", flattened)

transposed = reshaped.T       # transpose
print("Transposed:\n", transposed)

# Concatenation
a = np.array([1,2,3])
b = np.array([4,5,6])
print("Concatenated:", np.concatenate([a,b]))

# Stack vertically & horizontally
print("Vertical Stack:\n", np.vstack([a,b]))
print("Horizontal Stack:\n", np.hstack([a,b]))
