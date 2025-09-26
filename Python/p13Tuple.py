# ---------------------------------------------------------
# ✅ Python Tuple Detailed Example + Where Tuples Are Used
# ---------------------------------------------------------

# 🔹 1. Creating Tuples
empty_tuple = ()                           # Empty tuple
single_tuple = (10,)                       # Single element tuple (comma is must)
mixed_tuple = (1, "Hello", 3.14, True)     # Tuple with multiple datatypes
nested_tuple = (1, (2, 3), (4, 5))         # Tuple inside tuple (nested)

print("Empty Tuple:", empty_tuple)
print("Single Element Tuple:", single_tuple)
print("Mixed Tuple:", mixed_tuple)
print("Nested Tuple:", nested_tuple)

# 🔹 2. Indexing and Slicing
print("\nIndexing Examples:")
print("mixed_tuple[0] =", mixed_tuple[0])   # First element
print("mixed_tuple[-1] =", mixed_tuple[-1]) # Last element

print("\nSlicing Examples:")
print("mixed_tuple[1:3] =", mixed_tuple[1:3])      # Slice
print("nested_tuple[1][0] =", nested_tuple[1][0])  # Access inner tuple element

# 🔹 3. Immutability (Tuples cannot be modified)
# mixed_tuple[0] = 100   # ❌ Error → tuple is immutable

# 🔹 4. Tuple Methods
num_tuple = (1, 2, 3, 2, 4, 2)
print("\nTuple Methods:")
print("Count of 2:", num_tuple.count(2))    # counts how many times '2' occurs
print("Index of 3:", num_tuple.index(3))    # first index of '3'

# 🔹 5. Tuple Operations
tuple_a = (1, 2, 3)
tuple_b = (4, 5)

print("\nTuple Operations:")
print("Concatenation:", tuple_a + tuple_b)   # (1,2,3,4,5)
print("Repetition:", tuple_a * 3)            # (1,2,3,1,2,3,1,2,3)
print("Is 2 in tuple_a?", 2 in tuple_a)      # True
print("Is 10 not in tuple_a?", 10 not in tuple_a)  # True

# 🔹 6. Packing and Unpacking
packed = 10, 20, 30   # Packing
a, b, c = packed      # Unpacking
print("\nPacking and Unpacking:")
print("Packed tuple:", packed)
print("Unpacked values:", a, b, c)

# 🔹 7. Tuple as Dictionary Keys (possible since immutable)
location = {(10.0, 20.0): "Home", (30.5, 40.5): "Office"}
print("\nTuple as Dictionary Key:")
print("Location Dictionary:", location)
print("Value at (10.0,20.0):", location[(10.0,20.0)])

# 🔹 8. Tuple vs List (Immutability difference)
my_list = [1, 2, 3]
my_tuple = (1, 2, 3)

my_list[0] = 100      # ✅ Allowed (lists are mutable)
# my_tuple[0] = 100   # ❌ Not allowed (tuples are immutable)

print("\nTuple vs List:")
print("Modified List:", my_list)
print("Original Tuple (unchanged):", my_tuple)

# ---------------------------------------------------------
# ✅ WHY TUPLES ARE USED (Practical Applications)
# ---------------------------------------------------------

# 1. When data should not change (read-only)
days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
print("\nDays of Week (Fixed):", days)

# 2. As dictionary keys (lists ❌ cannot be keys)
locations = {(0, 0): "Origin", (10, 20): "Point A"}
print("Dictionary with Tuple Keys:", locations)

# 3. Returning multiple values from function
def min_max(numbers):
    return (min(numbers), max(numbers))  # returns tuple

result = min_max([5, 10, 2, 8])
print("\nFunction returning tuple:", result)
low, high = result
print("Min:", low, "Max:", high)

# 4. Tuples are memory efficient & faster
import sys
list_example = [1, 2, 3, 4]
tuple_example = (1, 2, 3, 4)
print("\nMemory Size:")
print("List size:", sys.getsizeof(list_example))
print("Tuple size:", sys.getsizeof(tuple_example))

# 5. Ensures data integrity (cannot be modified accidentally)
def process(data):
    # data[0] = 99  # ❌ would fail if tuple
    print("Processing (safe, unmodified):", data)

safe_data = (1, 2, 3)
process(safe_data)  # tuple ensures safety


# 🔑 Key Notes (explained in comments above):

# Tuples are immutable → once created, cannot change.

# Indexing & slicing → works like lists, but result is a tuple.

# Tuple methods → only .count() and .index().

# Operations → concatenation (+), repetition (*), membership (in, not in).

# Packing/Unpacking → grouping values into a tuple & extracting them.

# Dictionary keys → Tuples can be keys (lists cannot).

# Difference with list → list is mutable, tuple is not.