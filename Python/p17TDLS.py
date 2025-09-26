# ---------------------------------------------------------
# ✅ Difference Between List, Tuple, Set, and Dictionary
# ---------------------------------------------------------

# 🔹 1. Creation
print("\n--- Creation ---")

# List: Ordered, mutable, allows duplicates
my_list = [1, 2, 3, 3, 4]
print("List:", my_list)

# Tuple: Ordered, immutable, allows duplicates
my_tuple = (1, 2, 3, 3, 4)
print("Tuple:", my_tuple)

# Set: Unordered, mutable, stores unique elements only
my_set = {1, 2, 3, 3, 4}
print("Set:", my_set)

# Dictionary: Unordered, mutable, stores key-value pairs, keys must be unique
my_dict = {"a": 1, "b": 2, "c": 3}
print("Dictionary:", my_dict)

# 🔹 2. Mutability
print("\n--- Mutability ---")

# List can be changed
my_list[0] = 10
my_list.append(5)
print("Modified List:", my_list)

# Tuple cannot be changed (immutable)
# my_tuple[0] = 10  # ❌ TypeError
print("Tuple (immutable, cannot modify):", my_tuple)

# Set can be changed (add/remove elements)
my_set.add(5)
my_set.remove(1)
print("Modified Set:", my_set)

# Dictionary can be changed (add/modify key-value)
my_dict["d"] = 4
my_dict["a"] = 10
print("Modified Dictionary:", my_dict)

# 🔹 3. Order
print("\n--- Order ---")
print("List order:", my_list)        # Preserves insertion order
print("Tuple order:", my_tuple)      # Preserves insertion order
print("Set order:", my_set)          # Unordered
print("Dictionary order:", my_dict)  # Ordered since Python 3.7 (insertion order preserved)

# 🔹 4. Duplicates
print("\n--- Duplicates ---")
print("List allows duplicates:", my_list)  # Yes
print("Tuple allows duplicates:", my_tuple)  # Yes
print("Set removes duplicates:", my_set)   # No duplicates
print("Dictionary keys must be unique:", my_dict)  # Keys unique, values can duplicate

# 🔹 5. Indexing / Access
print("\n--- Indexing / Access ---")
print("List first element:", my_list[0])
print("Tuple first element:", my_tuple[0])
# print(my_set[0])  # ❌ Error → sets are unordered
print("Dictionary value for key 'a':", my_dict["a"])

# 🔹 6. Methods / Operations
print("\n--- Methods / Operations ---")

# List
my_list.append(6)
print("List after append:", my_list)

# Tuple
print("Tuple methods:", my_tuple.count(3), my_tuple.index(3))

# Set
my_set.add(6)
print("Set after add:", my_set)
print("Set union with {2, 7}:", my_set | {2, 7})  # Union
print("Set intersection with {2, 7}:", my_set & {2, 7})  # Intersection

# Dictionary
my_dict.update({"e": 5})
print("Dictionary after update:", my_dict)
print("Dictionary keys:", list(my_dict.keys()))
print("Dictionary values:", list(my_dict.values()))

# 🔹 7. Use Cases
print("\n--- Practical Uses ---")

# List → ordered collection of items, allows duplicates
shopping_list = ["milk", "bread", "milk"]
print("Shopping List:", shopping_list)

# Tuple → fixed data, immutable, e.g., coordinates
coordinate = (10.0, 20.0)
print("Coordinate Tuple:", coordinate)

# Set → unique elements, fast membership checking
unique_ids = {101, 102, 103}
print("Is 102 valid ID?", 102 in unique_ids)

# Dictionary → key-value mapping, structured data
student = {"name": "Sanket", "age": 20, "course": "CSE"}
print("Student Info:", student)
print("Student name:", student["name"])




# 🔑 Key Differences Summary
# Feature	List	Tuple	Set	Dictionary
# Mutable	✅ Yes	❌ No	✅ Yes	✅ Yes
# Ordered	✅ Yes	✅ Yes	❌ No	✅ Yes (Python 3.7+)
# Allows Duplicates	✅ Yes	✅ Yes	❌ No	Keys ❌, Values ✅
# Indexing	✅ Yes	✅ Yes	❌ No	Keys (via key) ✅
# Syntax	[ ]	( )	{ }	{key: value}
# Use Case Example	Sequence, mutable	Fixed data, immutable	Unique items, fast lookup	Key-value mapping