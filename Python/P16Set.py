# ---------------------------------------------------------
# ✅ Python Set Detailed Example + Practical Uses
# ---------------------------------------------------------

# 🔹 1. Creating Sets
# Sets are unordered, mutable, and store **unique elements only**
empty_set = set()                         # Empty set (must use set(), {} is dict)
numbers_set = {1, 2, 3, 4, 5,}             # Set with integers
mixed_set = {1, "Hello", 3.14, True}      # Set with different data types
duplicate_set = {1, 2, 2, 3, 3, 3}        # Duplicates removed automatically
nested_set = frozenset([4, 5, 6])         # Immutable set (frozenset)

print("Empty Set:", empty_set)
print("Numbers Set:", numbers_set)
print("Mixed Set:", mixed_set)
print("Set with Duplicates Removed:", duplicate_set)
print("Immutable frozenset:", nested_set)

# 🔹 2. Adding Elements
numbers_set.add(6)         # Add single element
numbers_set.update([7, 8]) # Add multiple elements from iterable
print("\nAfter Adding Elements:", numbers_set)

# 🔹 3. Removing Elements
numbers_set.remove(3)       # Remove element (KeyError if not found)
numbers_set.discard(10)     # Remove element (no error if not found)
removed_element = numbers_set.pop()  # Removes and returns a random element
print("After Removal:", numbers_set)
print("Randomly Removed Element:", removed_element)

# 🔹 4. Set Operations
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

print("\nSet Operations:")
print("Union (A | B):", A | B)           # All elements in A or B
print("Intersection (A & B):", A & B)    # Elements common in A and B
print("Difference (A - B):", A - B)      # Elements in A but not in B
print("Symmetric Difference (A ^ B):", A ^ B)  # Elements in A or B but not both

# 🔹 5. Iterating Over a Set
print("\nIterating Over Set:")
for item in A:
    print(item, end=" ")
print()

# 🔹 6. Membership Test
print("\nMembership Test:")
print("Is 2 in A?", 2 in A)
print("Is 10 not in A?", 10 not in A)

# 🔹 7. Set Comprehension
squares_set = {x**2 for x in range(1, 6)}  # Creates a set of squares
print("\nSet Comprehension (Squares):", squares_set)

# 🔹 8. Practical Uses of Sets
# 1. Remove duplicates from a list
nums = [1, 2, 2, 3, 4, 4, 5]
unique_nums = set(nums)
print("\nUnique numbers (from list):", unique_nums)

# 2. Fast membership checking
valid_ids = {101, 102, 103, 104}
user_input = 102
if user_input in valid_ids:
    print(f"User ID {user_input} is valid")
else:
    print(f"User ID {user_input} is invalid")

# 3. Mathematical operations
students_in_maths = {"Sanket", "Aayush", "Raj"}
students_in_science = {"Aayush", "Raj", "Rohit"}
all_students = students_in_maths | students_in_science         # Union
common_students = students_in_maths & students_in_science      # Intersection
print("All students:", all_students)
print("Common students:", common_students)




# 🔑 Key Notes (in comments)

# Sets are mutable and unordered → cannot access by index.

# Stores only unique elements → duplicates are removed automatically.

# Set methods → add(), update(), remove(), discard(), pop().

# Set operations → union |, intersection &, difference -, symmetric difference ^.

# Membership test → in, not in (fast due to hashing).

# Set comprehension → concise way to create sets.

# Practical uses → remove duplicates, fast lookup, mathematical operations.

# Immutable sets (frozenset) → cannot be modified, can be used as dictionary keys.