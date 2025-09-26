# ----------------------------------------
# 1. Basic Function
# ----------------------------------------
def greet():
    """This function prints a greeting message."""  # <-- docstring
    print("Hello, welcome to Python functions!")

# Calling the function
greet()


# ----------------------------------------
# 2. Function with Parameters
# ----------------------------------------
def add(a, b):
    """Takes two numbers and prints their sum."""
    print("Sum =", a + b)

add(10, 5)


# ----------------------------------------
# 3. Function with Default Arguments
# ----------------------------------------
def power(base, exponent=2):  # exponent defaults to 2
    """Returns base raised to the power of exponent."""
    return base ** exponent

print("Power(5):", power(5))        # exponent=2 (default)
print("Power(2, 3):", power(2, 3))  # exponent=3 (overridden)


# ----------------------------------------
# 4. Function with Return Values
# ----------------------------------------
def multiply(x, y):
    """Returns the product of two numbers."""
    return x * y

result = multiply(4, 6)
print("Multiply result:", result)


# ----------------------------------------
# 5. Function with Multiple Return Values (tuple)
# ----------------------------------------
def arithmetic_operations(a, b):
    """Performs basic arithmetic and returns multiple values."""
    return a + b, a - b, a * b, a / b

add_res, sub_res, mul_res, div_res = arithmetic_operations(10, 5)
print("Addition:", add_res)
print("Subtraction:", sub_res)
print("Multiplication:", mul_res)
print("Division:", div_res)


# ----------------------------------------
# 6. Variable-length Arguments (*args, **kwargs)
# ----------------------------------------
def sum_all(*args):  # accepts multiple values
    """Returns the sum of all numbers passed."""
    return sum(args)

print("Sum all (1,2,3,4):", sum_all(1, 2, 3, 4))


def introduce(**kwargs):  # accepts key-value pairs
    """Prints introduction using keyword arguments."""
    for key, value in kwargs.items():
        print(f"{key} -> {value}")

introduce(Name="Sanket", Age=20, Language="Python")


# ----------------------------------------
# 7. Keyword-only arguments
# ----------------------------------------
def student_info(name, *, age, grade):
    """Requires age and grade as keyword-only arguments."""
    print(f"Student {name}, Age: {age}, Grade: {grade}")

student_info("Sanket", age=20, grade="A")  
# student_info("Sanket", 20, "A") ❌ Error (must use keyword)


# ----------------------------------------
# 8. Nested Functions
# ----------------------------------------
def outer_function(text):
    """Outer function with an inner function."""
    
    def inner_function():  # defined inside
        print("Inner function says:", text.upper())
    
    inner_function()

outer_function("hello sanket")


# ----------------------------------------
# 9. Lambda Functions (anonymous function)
# ----------------------------------------
square = lambda x: x ** 2
add_two = lambda a, b: a + b

print("Square(6):", square(6))
print("Add(3,7):", add_two(3, 7))


# ----------------------------------------
# 10. Docstring Example
# ----------------------------------------
def factorial(n):
    """
    Returns factorial of a number.
    
    Parameters:
        n (int): number whose factorial is to be found
    
    Returns:
        int: factorial of n
    """
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    return fact

print("Factorial of 5:", factorial(5))
print("Docstring of factorial function:")
print(factorial.__doc__)




























