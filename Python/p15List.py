# ---------------------------------------------------------
# ✅ Python List Detailed Example + Practical Uses
# ---------------------------------------------------------

# 🔹 1. Creating Lists
empty_list = []                        # Empty list
numbers = [1, 2, 3, 4, 5]              # List of integers
mixed_list = [1, "Hello", 3.14, True]  # List with different datatypes
nested_list = [1, [2, 3], [4, 5]]      # List inside list (nested)

print("Empty List:", empty_list)
print("Numbers List:", numbers)
print("Mixed List:", mixed_list)
print("Nested List:", nested_list)

# 🔹 2. Accessing Elements (Indexing and Slicing)
print("\nIndexing Examples:")
print("numbers[0] =", numbers[0])       # First element
print("numbers[-1] =", numbers[-1])     # Last element

print("\nSlicing Examples:")
print("numbers[1:4] =", numbers[1:4])       # Slice from index 1 to 3
print("nested_list[1][0] =", nested_list[1][0])  # Access element of inner list

# 🔹 3. Modifying Lists (Mutable)
numbers[0] = 10            # Modify element
numbers.append(6)           # Add element at the end
numbers.insert(1, 15)       # Insert 15 at index 1
numbers.extend([7, 8, 9])   # Add multiple elements
print("\nModified List:", numbers)

# 🔹 4. Removing Elements
numbers.remove(15)          # Remove first occurrence of 15
last_item = numbers.pop()   # Remove last item and return it
print("After Removal:", numbers)
print("Popped Item:", last_item)

# 🔹 5. List Methods
print("\nList Methods:")
fruits = ["apple", "banana", "orange", "banana"]
print("Original Fruits:", fruits)
print("Count of 'banana':", fruits.count("banana"))  # counts occurrences
fruits.sort()                  # sorts alphabetically
print("Sorted Fruits:", fruits)
fruits.reverse()               # reverses list
print("Reversed Fruits:", fruits)
fruits.clear()                 # removes all elements
print("Cleared Fruits:", fruits)

# 🔹 6. Iterating Lists
numbers = [1, 2, 3, 4, 5]
print("\nIterating Numbers List:")
for num in numbers:
    print(num, end=" ")
print()

# 🔹 7. Nested Lists and List Comprehension
matrix = [[1, 2], [3, 4], [5, 6]]
print("\nNested List (Matrix):", matrix)
flattened = [num for row in matrix for num in row]  # flattening nested list
print("Flattened List using Comprehension:", flattened)

# 🔹 8. Practical Uses of Lists
# 1. Storing sequence of elements
names = ["Sanket", "Aayush", "Raj"]
print("\nNames List:", names)

# 2. Dynamic data storage
numbers = []
for i in range(5):
    numbers.append(i * 10)
print("Dynamic List:", numbers)

# 3. Counting occurrences
words = ["apple", "banana", "apple", "orange", "banana", "apple"]
word_count = []
for word in words:
    word_count.append(words.count(word))  # can also use dictionary for better efficiency
print("Word Occurrences (List Version):", word_count)

# 4. Easy iteration and manipulation
squares = [x**2 for x in range(1, 6)]
print("Squares using List Comprehension:", squares)





# 🔑 Key Notes (in comments)

# Lists are mutable → you can add, modify, remove elements.

# Supports indexing & slicing → access parts of list.

# Methods → append, insert, extend, remove, pop, count, sort, reverse, clear.

# Nested lists → can store lists inside lists.

# List comprehension → concise way to create/manipulate lists.

# Practical uses → sequence storage, dynamic data, iteration, computation.