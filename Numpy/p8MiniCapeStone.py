# ===========================================================
# 8. MINI CAPSTONE PROJECT (SMALL TASK)
# ===========================================================
# Example: Normalize dataset (scaling values between 0 and 1)

import numpy as np


data = np.array([[2,4,6],
                 [1,3,5],
                 [7,8,9]])

print("\nOriginal Data:\n", data)

# Find min & max column-wise
min_vals = data.min(axis=0)
max_vals = data.max(axis=0)

# Normalize (x - min) / (max - min)
normalized = (data - min_vals) / (max_vals - min_vals)
print("Normalized Data:\n", normalized)
