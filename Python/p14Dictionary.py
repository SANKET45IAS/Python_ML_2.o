# ---------------------------------------------------------
# ✅ Python Dictionary Detailed Example + Practical Uses
# ---------------------------------------------------------

# 🔹 1. Creating Dictionaries
# Dictionaries are mutable key-value pairs, keys must be immutable
empty_dict = {}   # Empty dictionary
person = {"name": "Sanket", "age": 20, "course": "CSE"}  # key-value pairs
mixed_dict = {1: "One", "two": 2, (3, 4): "Tuple Key"}  # keys can be string, number, tuple

print("Empty Dictionary:", empty_dict)
print("Person Dictionary:", person)
print("Mixed Dictionary:", mixed_dict)

# 🔹 2. Accessing Values
print("\nAccessing Values:")
print("Name:", person["name"])            # Access using key
print("Age:", person.get("age"))           # get() method avoids KeyError
print("Non-existent key with get():", person.get("grade", "N/A"))  # default if key not found

# 🔹 3. Adding / Modifying Items
person["grade"] = "A+"       # Add new key-value pair
person["age"] = 21            # Modify existing value
print("\nAfter Adding/Modifying:", person)

# 🔹 4. Removing Items
removed_value = person.pop("grade")       # Remove key and return value
print("Removed Grade:", removed_value)
person.popitem()                           # Remove last inserted item
# del person["age"]                        # Remove specific key
print("After Removal:", person)

# 🔹 5. Dictionary Methods
print("\nDictionary Methods:")
sample_dict = {"a": 1, "b": 2, "c": 3}
print("Keys:", list(sample_dict.keys()))    # List of keys
print("Values:", list(sample_dict.values()))# List of values
print("Items:", list(sample_dict.items()))  # List of (key,value) tuples

# update() method to merge dictionaries
sample_dict.update({"d": 4, "b": 20})      # updates 'b' and adds 'd'
print("After update:", sample_dict)

# 🔹 6. Iterating Dictionaries
print("\nIterating Dictionary:")
for key in sample_dict:
    print(f"Key: {key}, Value: {sample_dict[key]}")

# Using items() to get key and value together
for key, value in sample_dict.items():
    print(f"{key} -> {value}")

# 🔹 7. Nested Dictionaries
students = {
    "Sanket": {"age": 20, "course": "CSE"},
    "Aayush": {"age": 19, "course": "IT"},
    "Raj": {"age": 21, "course": "ECE"}
}
print("\nNested Dictionary:")
print(students)
print("Sanket's Course:", students["Sanket"]["course"])

# 🔹 8. Dictionary Comprehension
squares = {x: x**2 for x in range(1, 6)}
print("\nDictionary Comprehension (Squares):", squares)

# 🔹 9. Practical Uses of Dictionary
# 1. Storing structured data (like JSON)
employee = {"id": 101, "name": "Sanket", "salary": 50000}
print("\nEmployee Info:", employee)

# 2. Counting occurrences (word frequency)
words = ["apple", "banana", "apple", "orange", "banana", "apple"]
word_count = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1
print("Word Frequency:", word_count)

# 3. Fast lookup (O(1) average)
numbers = {1: "One", 2: "Two", 3: "Three"}
key_to_find = 2
if key_to_find in numbers:
    print(f"Found key {key_to_find}:", numbers[key_to_find])



# 🔑 Key Notes (Explained in Comments)

# Mutable → can add, modify, delete items.

# Keys must be immutable (string, number, tuple).

# Access values → dict[key] or dict.get(key) (safer).

# Methods → keys(), values(), items(), update(), pop(), popitem().

# Nested dictionaries → store structured data like JSON.

# Dictionary comprehension → create dictionaries concisely.

# Practical uses → structured data, counting frequency, fast lookup.