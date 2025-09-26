# ===========================================================
# 5. ARRAY MODIFICATION
# ===========================================================
import numpy as np


arr = np.array([1,2,3,4,5])
print("\nOriginal:", arr)

arr[0] = 100          # modify single element
print("Modified first element:", arr)

arr[1:3] = [200,300]  # modify slice
print("Modified slice:", arr)

# Add element (creates new array)
arr2 = np.append(arr, [400,500])
print("Appended array:", arr2)

# Delete element
arr3 = np.delete(arr2, [0,2]) # delete index 0 and 2
print("Deleted array:", arr3)

# Insert element
arr4 = np.insert(arr3, 1, 999) # insert 999 at index 1
print("Inserted array:", arr4)
