# ---------------------------------------------------------
# ✅ Python Error Handling Example (Detailed)
# ---------------------------------------------------------

# 🔹 1. Basic Try-Except
print("\n--- Basic Try-Except ---")
try:
    num1 = int(input("Enter a number: "))
    num2 = int(input("Enter another number: "))
    result = num1 / num2   # May raise ZeroDivisionError
    print("Result:", result)
except ZeroDivisionError:   # Catch division by zero
    print("Error: Division by zero is not allowed!")
except ValueError:          # Catch invalid integer input
    print("Error: Please enter valid integers!")

# 🔹 2. Catching Multiple Exceptions
print("\n--- Multiple Exceptions ---")
try:
    a = int(input("Enter first number: "))
    b = int(input("Enter second number: "))
    c = a / b
except (ZeroDivisionError, ValueError) as e:  # Catch multiple exceptions
    print("Error occurred:", e)
else:
    print("Division successful! Result:", c)  # Runs only if no exception
finally:
    print("Execution completed (finally block runs always)")

# 🔹 3. Raising Exceptions
print("\n--- Raising Exceptions ---")
def check_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative!")  # Manually raise exception
    elif age < 18:
        print("Minor")
    else:
        print("Adult")

try:
    age = int(input("Enter your age: "))
    check_age(age)
except ValueError as e:
    print("Error:", e)

# 🔹 4. Handling File Errors
print("\n--- File Handling Errors ---")
try:
    f = open("non_existent_file.txt", "r")  # May raise FileNotFoundError
    content = f.read()
except FileNotFoundError as e:
    print("File Error:", e)
finally:
    print("File handling attempt completed")

# 🔹 5. Custom Exception Example
print("\n--- Custom Exception ---")
class NegativeNumberError(Exception):
    """Custom exception for negative numbers"""
    pass

def square_root(x):
    if x < 0:
        raise NegativeNumberError("Cannot calculate square root of negative number")
    return x ** 0.5

try:
    num = -25
    print("Square root:", square_root(num))
except NegativeNumberError as e:
    print("Custom Error:", e)

# 🔹 6. Practical Uses of Error Handling
# 1. Prevent program from crashing due to invalid user input
# 2. Handle file reading/writing safely
# 3. Validate data and raise custom exceptions
# 4. Ensure some code always runs (finally block)
