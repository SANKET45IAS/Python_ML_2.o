# ===========================================================
# 7. HANDLING MISSING VALUES
# ===========================================================
# NumPy uses NaN (Not a Number) for missing values

import numpy as np


arr = np.array([1,2,np.nan,4,5])
print("\nArray with NaN:", arr)

# Check for NaN
print("Is NaN:", np.isnan(arr))

# Replace NaN with mean
mean_val = np.nanmean(arr)   # mean ignoring NaN
arr[np.isnan(arr)] = mean_val
print("Replaced NaN with mean:", arr)

